//! The public inference API.

use stdbev_format::{context_text, encode, option_texts};
use stdbev_math::softmax_inplace;
use stdbev_types::{
    ChoiceAnswer, DecisionAnswer, DecisionRequest, ModelConfig, NoulAnswer, OptionEncoder,
    OptionFormat, Question, ScoreAnswer, TensorId,
};

use crate::error::RuntimeError;
use crate::model::{Encoder, Shape, build_keys_values, option_layers, pool_option, score_option};
use crate::parse::{Header, parse_directory, parse_header};
use crate::scratch::Scratch;
use crate::view::{Tensor, resolve};

/// A loaded model, borrowing the artifact bytes.
pub struct Runtime<'a> {
    header: Header,
    config: ModelConfig,
    tensors: Vec<(TensorId, Tensor<'a>)>,
    model_id: String,
}

impl<'a> Runtime<'a> {
    /// Parses and validates an `STDBEV01` artifact.
    ///
    /// # Errors
    /// Returns [`RuntimeError::Malformed`] for any structural problem and
    /// [`RuntimeError::MissingTensor`] if a required tensor is absent.
    pub fn from_bytes(bytes: &'a [u8]) -> Result<Self, RuntimeError> {
        let header = parse_header(bytes)?;
        let entries = parse_directory(bytes, header.tensor_count)?;
        let mut tensors = Vec::with_capacity(entries.len());
        for e in &entries {
            tensors.push((e.id, resolve(e, bytes)?));
        }
        let option_encoder = match header.option_encoder_tag {
            0 => OptionEncoder::Pooled,
            1 => OptionEncoder::Encoded,
            _ => return Err(RuntimeError::Malformed("unknown option_encoder tag")),
        };
        // Serving an artifact the option encoding it was not trained on produces
        // confident nonsense with no error, so the encoding travels in the header.
        let option_format = match header.option_format_tag {
            0 => OptionFormat::Verbose,
            1 => OptionFormat::Compact,
            _ => return Err(RuntimeError::Malformed("unknown option_format tag")),
        };
        let config = ModelConfig {
            vocabulary_size: header.vocabulary_size,
            model_width: header.model_width,
            transformer_layers: header.transformer_layers,
            attention_heads: header.attention_heads,
            feed_forward_width: header.feed_forward_width,
            option_attention_rank: header.option_attention_rank,
            context_length: header.context_length,
            option_length: header.option_length,
            option_encoder,
            option_format,
        };
        if config.attention_heads == 0 || !config.model_width.is_multiple_of(config.attention_heads)
        {
            return Err(RuntimeError::ArchitectureMismatch {
                field: "attention_heads",
            });
        }
        let model_id = bytes.get(32..64).map(hex).unwrap_or_default();
        Ok(Self {
            header,
            config,
            tensors,
            model_id,
        })
    }

    /// The architecture this artifact describes.
    #[must_use]
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// Hex SHA-256 identifying this artifact.
    #[must_use]
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Fitted calibration temperature.
    #[must_use]
    pub fn temperature(&self) -> f32 {
        self.header.temperature
    }

    fn shape(&self) -> Shape {
        Shape {
            width: self.config.model_width as usize,
            ff_width: self.config.feed_forward_width as usize,
            layers: self.config.transformer_layers,
            heads: self.config.attention_heads as usize,
            head_width: self.config.head_width(),
            rank: self.config.option_attention_rank as usize,
        }
    }

    /// Scores a context against a dynamic option set, returning calibrated
    /// probabilities that sum to 1.
    ///
    /// This is the single scorer every typed question delegates to, which is what
    /// makes Choice, Noul and Score provably the same machine.
    ///
    /// # Errors
    /// Returns [`RuntimeError`] if a tensor is missing or the option set is empty.
    pub fn score_options(
        &self,
        context: &str,
        options: &[String],
        scratch: &mut Scratch,
    ) -> Result<Vec<f32>, RuntimeError> {
        if options.is_empty() {
            return Err(RuntimeError::Invalid("no options".into()));
        }
        let encoder = Encoder {
            tensors: &self.tensors,
            shape: self.shape(),
        };
        let ctx_tokens = encode(context, self.config.context_length as usize);
        let ctx_len = ctx_tokens.ids.len();

        encoder.encode_sequence(
            &mut scratch.context,
            &mut scratch.wide,
            &mut scratch.attention,
            &ctx_tokens,
            self.config.transformer_layers,
        )?;
        let context_hidden = scratch.context.clone();
        let context_mask = ctx_tokens.mask.clone();
        // Keys and values are a function of the context alone, so they are built once
        // here rather than once per option.
        build_keys_values(&encoder, &context_hidden, scratch, ctx_len)?;
        // Keys and values are a function of the context alone, so they are built once
        // here rather than inside the option loop.
        build_keys_values(&encoder, &context_hidden, scratch, ctx_len)?;

        let opt_layers = option_layers(self.config.option_encoder, encoder.shape.layers);
        let width = encoder.shape.width;
        let mut logits = Vec::with_capacity(options.len());
        for text in options {
            let tokens = encode(text, self.config.option_length as usize);
            encoder.encode_option(
                &mut scratch.option,
                &mut scratch.wide,
                &mut scratch.attention,
                &tokens,
                opt_layers,
            )?;
            let mut pooled = vec![0.0f32; width];
            pool_option(&mut pooled, &scratch.option, &tokens.mask, width);
            logits.push(score_option(
                &encoder,
                &pooled,
                &context_mask,
                scratch,
                ctx_len,
            )?);
        }
        let mut probs = logits;
        let t = self.header.temperature;
        for l in &mut probs {
            *l /= t;
        }
        softmax_inplace(&mut probs);
        Ok(probs)
    }

    /// Answers a typed question.
    ///
    /// # Errors
    /// Returns [`RuntimeError::Invalid`] if the request fails validation.
    pub fn decide(
        &self,
        request: &DecisionRequest,
        scratch: &mut Scratch,
    ) -> Result<DecisionAnswer, RuntimeError> {
        request.validate()?;
        let context = context_text(request);
        let options = option_texts(&request.question, self.config.option_format);
        let probabilities = self.score_options(&context, &options, scratch)?;
        Ok(self.interpret(&request.question, probabilities))
    }

    /// Wraps a raw distribution in the shape the asked primitive expects.
    fn interpret(&self, question: &Question, probabilities: Vec<f32>) -> DecisionAnswer {
        let (index, confidence) = stdbev_math::argmax(&probabilities);
        let model_id = self.model_id.clone();
        match question {
            Question::Choice(q) => DecisionAnswer::Choice(ChoiceAnswer {
                selected_index: index,
                selected_label: q
                    .criteria
                    .get(index)
                    .map(|c| c.label.clone())
                    .unwrap_or_default(),
                confidence,
                probabilities,
                model_id,
            }),
            Question::Noul(_) => {
                let p_true = probabilities.get(1).copied().unwrap_or(0.0);
                let p_false = probabilities.first().copied().unwrap_or(0.0);
                DecisionAnswer::Noul(NoulAnswer {
                    probability: p_true,
                    probabilities: [p_false, p_true],
                    model_id,
                })
            }
            Question::Score(q) => {
                let score = probabilities
                    .iter()
                    .enumerate()
                    .map(|(i, p)| p * level_as_f32(i))
                    .sum();
                DecisionAnswer::Score(ScoreAnswer {
                    score,
                    selected_index: index,
                    selected_label: q
                        .levels
                        .get(index)
                        .map(|l| l.label.clone())
                        .unwrap_or_default(),
                    confidence,
                    probabilities,
                    model_id,
                })
            }
        }
    }
}

/// Converts a level ordinal to `f32`.
///
/// Bounded by the 10-level Score maximum, so exactly representable.
#[allow(clippy::cast_precision_loss)]
fn level_as_f32(i: usize) -> f32 {
    i as f32
}

/// Lowercase hex encoding of a byte slice.
fn hex(bytes: &[u8]) -> String {
    use core::fmt::Write as _;
    bytes
        .iter()
        .fold(String::with_capacity(bytes.len() * 2), |mut acc, b| {
            let _ = write!(acc, "{b:02x}");
            acc
        })
}
