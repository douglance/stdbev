//! The v1 architecture in Candle.
//!
//! This must stay numerically identical to `stdbev-runtime`, because the weights
//! trained here are executed there. Two choices are load-bearing and easy to get
//! wrong:
//!
//! * **GELU must be the tanh approximation** (`gelu`, not `gelu_erf`). The two differ
//!   by ~1e-3, which is a hundred times the parity budget.
//! * **`sqrt(rank)` divides twice** in option attention -- once on the scores and
//!   again on the final logit -- matching the reference implementation.

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Module, VarBuilder, VarMap};
use stdbev_types::{ModelConfig, OptionEncoder};

/// Dropout probability during training; zero at inference.
///
/// Zero by default, matching the reference implementation, which uses none. The spec
/// asks for 0.10, but the spec also misattributes its architecture -- and the apparent
/// overfitting that first motivated adding dropout here turned out to be a learning
/// rate 6.7x too low fighting weight decay 100x too high. Regularizing a model that is
/// underfitting makes it worse. Kept wired up so it can be turned on with evidence.
const DROPOUT: f32 = 0.0;

/// One pre-LayerNorm transformer block.
pub struct Block {
    ln1: candle_nn::LayerNorm,
    ln2: candle_nn::LayerNorm,
    wq: candle_nn::Linear,
    wk: candle_nn::Linear,
    wv: candle_nn::Linear,
    wo: candle_nn::Linear,
    ff1: candle_nn::Linear,
    ff2: candle_nn::Linear,
    dropout: candle_nn::Dropout,
    heads: usize,
    head_width: usize,
}

impl Block {
    fn load(vb: &VarBuilder<'_>, config: &ModelConfig) -> Result<Self> {
        let d = config.model_width as usize;
        let ff = config.feed_forward_width as usize;
        Ok(Self {
            ln1: candle_nn::layer_norm(d, 1e-5, vb.pp("ln1"))?,
            ln2: candle_nn::layer_norm(d, 1e-5, vb.pp("ln2"))?,
            wq: candle_nn::linear(d, d, vb.pp("wq"))?,
            wk: candle_nn::linear(d, d, vb.pp("wk"))?,
            wv: candle_nn::linear(d, d, vb.pp("wv"))?,
            wo: candle_nn::linear(d, d, vb.pp("wo"))?,
            ff1: candle_nn::linear(d, ff, vb.pp("ff1"))?,
            ff2: candle_nn::linear(ff, d, vb.pp("ff2"))?,
            dropout: candle_nn::Dropout::new(DROPOUT),
            heads: config.attention_heads as usize,
            head_width: config.head_width(),
        })
    }

    /// `hidden` is `[seq, width]`; `mask` is `[seq]` with 1 for real tokens.
    fn forward(&self, hidden: &Tensor, mask: &Tensor, train: bool) -> Result<Tensor> {
        let (seq, width) = hidden.dims2()?;
        let normed = self.ln1.forward(hidden)?;
        let reshape = |t: Tensor| -> Result<Tensor> {
            t.reshape((seq, self.heads, self.head_width))?
                .transpose(0, 1)
        };
        let q = reshape(self.wq.forward(&normed)?)?;
        let k = reshape(self.wk.forward(&normed)?)?;
        let v = reshape(self.wv.forward(&normed)?)?;

        #[allow(clippy::cast_precision_loss)]
        let scale = 1.0 / (self.head_width as f64).sqrt();
        let scores = (q.matmul(&k.transpose(1, 2)?)? * scale)?;
        // Masked positions go to a large negative value before the softmax, so they
        // contribute exactly zero probability rather than a small one.
        let bias = ((mask.to_dtype(DType::F32)? - 1.0)? * 1e9)?;
        let scores = scores.broadcast_add(&bias.reshape((1, 1, seq))?)?;
        let weights = candle_nn::ops::softmax_last_dim(&scores)?;
        let attended = weights
            .matmul(&v)?
            .transpose(0, 1)?
            .reshape((seq, width))?
            .contiguous()?;
        let projected = self.dropout.forward(&self.wo.forward(&attended)?, train)?;
        let hidden = (hidden + projected)?;

        let normed = self.ln2.forward(&hidden)?;
        let wide = self.ff1.forward(&normed)?.gelu()?;
        let out = self.dropout.forward(&self.ff2.forward(&wide)?, train)?;
        hidden + out
    }
}

/// The full scorer.
pub struct Scorer {
    token: candle_nn::Embedding,
    position: candle_nn::Embedding,
    blocks: Vec<Block>,
    context_ln: candle_nn::LayerNorm,
    option_ln: candle_nn::LayerNorm,
    opt_q: candle_nn::Linear,
    opt_k: candle_nn::Linear,
    opt_v: candle_nn::Linear,
    rank: usize,
    option_encoder: OptionEncoder,
}

impl Scorer {
    /// Builds the network and its parameter map.
    ///
    /// # Errors
    /// Propagates Candle shape and device errors.
    pub fn new(config: &ModelConfig, device: &Device) -> Result<(Self, VarMap)> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
        let d = config.model_width as usize;
        let rank = config.option_attention_rank as usize;
        let blocks = (0..config.transformer_layers)
            .map(|i| Block::load(&vb.pp(format!("block{i}")), config))
            .collect::<Result<Vec<_>>>()?;
        let scorer = Self {
            token: candle_nn::embedding(config.vocabulary_size as usize, d, vb.pp("token"))?,
            position: candle_nn::embedding(config.context_length as usize, d, vb.pp("position"))?,
            blocks,
            context_ln: candle_nn::layer_norm(d, 1e-5, vb.pp("opt_context_ln"))?,
            option_ln: candle_nn::layer_norm(d, 1e-5, vb.pp("opt_option_ln"))?,
            opt_q: candle_nn::linear_no_bias(d, rank, vb.pp("opt_q"))?,
            opt_k: candle_nn::linear_no_bias(d, rank, vb.pp("opt_k"))?,
            opt_v: candle_nn::linear_no_bias(d, rank, vb.pp("opt_v"))?,
            rank,
            option_encoder: config.option_encoder,
        };
        Ok((scorer, varmap))
    }

    /// Token + position embedding, for the context.
    fn embed(&self, ids: &Tensor) -> Result<Tensor> {
        let seq = ids.dims1()?;
        let seq = u32::try_from(seq).unwrap_or(u32::MAX);
        let positions = Tensor::arange(0u32, seq, ids.device())?;
        self.token.forward(ids)? + self.position.forward(&positions)?
    }

    /// Token embedding only, for options.
    ///
    /// The reference implementation adds position to the context and **not** to
    /// options, and the reason shows up in the arithmetic: mean-pooling `token +
    /// position` leaves a `mean(position[0..len])` term that depends only on the
    /// option's length. Two options of similar length therefore share a large constant
    /// component that carries no information about which option they are. Measured on
    /// the test corpus, removing it drops the mean cosine between pooled Noul options
    /// from 0.38 to 0.04.
    fn embed_option(&self, ids: &Tensor) -> Result<Tensor> {
        self.token.forward(ids)
    }

    /// Applies every transformer block in order.
    fn run_blocks(&self, hidden: &Tensor, mask: &Tensor, train: bool) -> Result<Tensor> {
        let mut out = hidden.clone();
        for block in &self.blocks {
            out = block.forward(&out, mask, train)?;
        }
        Ok(out)
    }

    /// Runs the encoder stack over the context.
    ///
    /// # Errors
    /// Propagates Candle errors.
    pub fn encode_context(&self, ids: &Tensor, mask: &Tensor, train: bool) -> Result<Tensor> {
        self.run_blocks(&self.embed(ids)?, mask, train)
    }

    /// Pools an option's tokens to one vector.
    ///
    /// `Pooled` is embedding plus masked mean with no transformer -- `jevlike`'s actual
    /// architecture, and what Phase 0 measured as 6.6x cheaper and independent of
    /// option count. `Encoded` runs the shared encoder stack first, which is
    /// `cua-s1-forms`. Which one wins is an evidence question, so both exist.
    ///
    /// # Errors
    /// Propagates Candle errors.
    pub fn pool_option(&self, ids: &Tensor, mask: &Tensor, train: bool) -> Result<Tensor> {
        // Position is dropped for the pooled path and kept for the encoded one, and the
        // asymmetry is principled rather than incidental: mean pooling discards order,
        // so a positional term contributes nothing but a length-dependent constant. A
        // transformer reads order, so it needs one.
        let embedded = match self.option_encoder {
            OptionEncoder::Pooled => self.embed_option(ids)?,
            OptionEncoder::Encoded => self.run_blocks(&self.embed(ids)?, mask, train)?,
        };
        let mask = mask.to_dtype(DType::F32)?.unsqueeze(1)?;
        let summed = embedded.broadcast_mul(&mask)?.sum(0)?;
        let count = mask.sum_all()?.clamp(1.0, f32::MAX)?;
        summed.broadcast_div(&count)
    }

    /// One logit per option.
    ///
    /// # Errors
    /// Propagates Candle errors.
    pub fn logits(
        &self,
        context: &Tensor,
        context_mask: &Tensor,
        options: &[Tensor],
    ) -> Result<Tensor> {
        let normed = self.context_ln.forward(context)?;
        let keys = self.opt_k.forward(&normed)?;
        let values = self.opt_v.forward(&normed)?;
        #[allow(clippy::cast_precision_loss)]
        let scale = 1.0 / (self.rank as f64).sqrt();
        let bias = ((context_mask.to_dtype(DType::F32)? - 1.0)? * 1e9)?.unsqueeze(0)?;

        let mut out = Vec::with_capacity(options.len());
        for option in options {
            // Candle's Linear needs at least rank 2, so the pooled option vector is
            // carried as [1, width] and squeezed back at the end.
            let normed = self.option_ln.forward(&option.unsqueeze(0)?)?;
            let query = self.opt_q.forward(&normed)?;
            let scores = ((query.matmul(&keys.t()?)? * scale)? + &bias)?;
            let weights = candle_nn::ops::softmax_last_dim(&scores)?;
            let attended = weights.matmul(&values)?;
            // Second sqrt(rank) division; dropping it changes the model's temperature.
            out.push(((query * attended)?.sum_all()? * scale)?);
        }
        Tensor::stack(&out, 0)
    }
}
