//! Request-local working memory.
//!
//! Allocated once at maximum size before inference begins, so the hot path never
//! allocates. Options are encoded and scored one at a time, so working memory does
//! not grow with the option count -- only the final logits vector does.

use stdbev_types::ModelConfig;

/// Reusable buffers for one decision.
pub struct Scratch {
    pub(crate) context: Vec<f32>,
    pub(crate) context_alt: Vec<f32>,
    pub(crate) option: Vec<f32>,
    pub(crate) option_alt: Vec<f32>,
    pub(crate) wide: Vec<f32>,
    pub(crate) attention: Vec<f32>,
    pub(crate) keys: Vec<f32>,
    pub(crate) values: Vec<f32>,
    pub(crate) pooled: Vec<f32>,
    pub(crate) query: Vec<f32>,
    allocated: usize,
}

impl Scratch {
    /// Allocates every buffer at its maximum size for `config`.
    #[must_use]
    pub fn new(config: &ModelConfig) -> Self {
        let d = config.model_width as usize;
        let ctx = config.context_length as usize;
        let opt = config.option_length as usize;
        let ff = config.feed_forward_width as usize;
        let rank = config.option_attention_rank as usize;
        let s = Self {
            context: vec![0.0; ctx * d],
            context_alt: vec![0.0; ctx * d],
            option: vec![0.0; opt * d],
            option_alt: vec![0.0; opt * d],
            wide: vec![0.0; ff],
            attention: vec![0.0; ctx],
            keys: vec![0.0; ctx * rank],
            values: vec![0.0; ctx * rank],
            pooled: vec![0.0; d],
            query: vec![0.0; rank],
            allocated: 0,
        };
        let allocated = (s.context.len()
            + s.context_alt.len()
            + s.option.len()
            + s.option_alt.len()
            + s.wide.len()
            + s.attention.len()
            + s.keys.len()
            + s.values.len()
            + s.pooled.len()
            + s.query.len())
            * size_of::<f32>();
        Self { allocated, ..s }
    }

    /// Bytes held by the preallocated buffers.
    ///
    /// The system test asserts this stays under the 4 MiB budget.
    #[must_use]
    pub fn allocated_bytes(&self) -> usize {
        self.allocated
    }
}
