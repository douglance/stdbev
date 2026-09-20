//! The v1 architecture, as a value.
//!
//! Every field here is written into the artifact header and checked on load, so a
//! runtime can never silently execute a model it was not built for.

use serde::{Deserialize, Serialize};

/// How option text is reduced to the single vector that queries the context.
///
/// Phase 0 measured this as the dominant cost lever: option encoding is ~86% of a
/// decision's arithmetic, and it is the only knob that materially moves latency.
/// Both settings have published precedent, so this is an evidence question, not a
/// taste question -- Phase 3 trains both and ships the cheaper one that passes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OptionEncoder {
    /// Embedding + position, then masked mean pool. No transformer over options.
    ///
    /// This is `jevlike`'s actual architecture (~98% on synthetic menus). Latency
    /// becomes independent of option count: measured 35 ms at K=2 and at K=16 alike.
    Pooled,
    /// Options run through the full shared encoder stack.
    ///
    /// This is `cua-s1-forms` (99.95% synthetic / 100% on 196 real form decisions).
    /// Measured 59 ms at K=2 rising to 229 ms at K=16.
    Encoded,
}

/// How a typed question's options are rendered to bytes.
///
/// This is part of the model's identity, not a presentation choice: an artifact trained
/// on one encoding and served the other produces confident nonsense with no error. It
/// therefore travels in the artifact header and is checked on load.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OptionFormat {
    /// `LABEL\n{label}\nDESCRIPTION\n{description}`, as the spec's section 8 defines it.
    ///
    /// Measured across the test corpus, any two options share **0.636** of their bytes
    /// because the scaffolding is common to all of them. A pooled option encoder cannot
    /// discount that; an encoded one can.
    Verbose,
    /// The label alone. Noul becomes bare polarity markers.
    ///
    /// Descriptions are deliberately dropped, and the reason is a measured property of
    /// mean pooling rather than a preference. Keeping them as `{label}: {description}`
    /// leaves pooled options at **0.845** cosine for Choice and **0.858** for Score --
    /// barely better than the 0.896/0.907 of the verbose encoding, because the
    /// descriptions carry most of the bytes and share most of their vocabulary. Dropping
    /// them reaches **0.350** and **0.057**.
    ///
    /// A pooled byte encoder cannot use free-text descriptions: pooling averages them
    /// into the same direction. If descriptions must inform the decision, that requires
    /// [`OptionEncoder::Encoded`], which can read their structure.
    ///
    /// This also matches jev's own API, where `criteria` maps a label to an *optional*
    /// description and the documented default is `null`.
    Compact,
}

/// Frozen description of a STDBEV network.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelConfig {
    pub vocabulary_size: u16,
    pub model_width: u16,
    pub transformer_layers: u16,
    pub attention_heads: u16,
    pub feed_forward_width: u16,
    pub option_attention_rank: u16,
    pub context_length: u16,
    pub option_length: u16,
    pub option_encoder: OptionEncoder,
    pub option_format: OptionFormat,
}

impl ModelConfig {
    /// The v1 configuration.
    pub const V1: Self = Self {
        vocabulary_size: 257,
        model_width: 128,
        transformer_layers: 2,
        attention_heads: 4,
        feed_forward_width: 512,
        option_attention_rank: 128,
        context_length: 224,
        option_length: 96,
        option_encoder: OptionEncoder::Pooled,
        option_format: OptionFormat::Compact,
    };

    /// Width of a single attention head.
    #[must_use]
    pub const fn head_width(&self) -> usize {
        if self.attention_heads == 0 {
            return 0;
        }
        (self.model_width / self.attention_heads) as usize
    }

    /// Number of trainable parameters implied by this configuration.
    ///
    /// Computed from the shape, never hardcoded, so `model inspect` cannot drift
    /// from what the artifact actually contains.
    #[must_use]
    pub fn parameter_count(&self) -> usize {
        let d = self.model_width as usize;
        let ff = self.feed_forward_width as usize;
        let layers = self.transformer_layers as usize;
        let embeddings = self.vocabulary_size as usize * d + self.context_length as usize * d;
        let block = 4 * (d * d + d) + (d * ff + ff) + (ff * d + d) + 2 * (2 * d);
        let option_attention = 3 * (d * d + d) + 2 * (2 * d);
        embeddings + layers * block + option_attention + 2 * d
    }
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self::V1
    }
}
