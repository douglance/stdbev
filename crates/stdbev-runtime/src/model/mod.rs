//! The model driver.
//!
//! [`Encoder`] bundles the tensor table and the architecture so the individual stages
//! do not each have to thread eight parameters through.

mod block;
mod embed;
mod score;

use stdbev_types::{OptionEncoder, TensorId};

use crate::view::Tensor;

pub(crate) use score::{build_keys_values, score_option};

/// Architecture dimensions, resolved once per request.
#[derive(Clone, Copy)]
pub(crate) struct Shape {
    pub width: usize,
    pub ff_width: usize,
    pub layers: u16,
    pub heads: usize,
    pub head_width: usize,
    pub rank: usize,
}

/// Borrowed tensors plus the shape they describe.
pub(crate) struct Encoder<'a> {
    pub tensors: &'a [(TensorId, Tensor<'a>)],
    pub shape: Shape,
}

/// How many encoder layers the option path runs.
///
/// Zero is `jevlike`'s architecture: options are reduced by masked mean pooling over
/// raw embeddings. Phase 0 measured that as 6.6x cheaper, and it makes latency
/// independent of option count.
pub(crate) const fn option_layers(encoder: OptionEncoder, layers: u16) -> u16 {
    match encoder {
        OptionEncoder::Pooled => 0,
        OptionEncoder::Encoded => layers,
    }
}

/// Reduces an option's encoded tokens to the single vector that queries the context.
pub(crate) fn pool_option(out: &mut [f32], tokens: &[f32], mask: &[bool], width: usize) {
    stdbev_math::masked_mean_pool(out, tokens, mask, width);
}
