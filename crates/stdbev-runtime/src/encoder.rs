//! The shared byte encoder: embeddings, transformer blocks, masked attention.

use stdbev_math::{gelu_inplace, layer_norm, masked_softmax_inplace, matvec_i8};
use stdbev_types::{TensorId, slot};

use crate::error::RuntimeError;
use crate::view::{Tensor, find};

/// Weights for one linear layer: quantized matrix, per-channel scale, optional bias.
pub(crate) struct Linear<'a> {
    pub weights: &'a [i8],
    pub scale: &'a [f32],
    pub bias: Option<&'a [f32]>,
}

pub(crate) fn linear<'a>(
    tensors: &'a [(TensorId, Tensor<'a>)],
    layer: u16,
    w: u16,
    s: u16,
    b: Option<u16>,
) -> Result<Linear<'a>, RuntimeError> {
    let id = |slot_id: u16| TensorId::layer(layer, slot_id).0;
    let bias = match b {
        Some(bid) => Some(find(tensors, id(bid))?.as_f32()?),
        None => None,
    };
    Ok(Linear {
        weights: find(tensors, id(w))?.as_i8()?,
        scale: find(tensors, id(s))?.as_f32()?,
        bias,
    })
}

/// Applies `out = linear(input)` for a single token vector.
pub(crate) fn apply(out: &mut [f32], input: &[f32], l: &Linear<'_>, cols: usize) {
    matvec_i8(out, input, l.weights, l.scale, l.bias, cols);
}

/// Scaled dot-product attention over one head, writing into `out`.
///
/// Padding positions are masked to `-inf` before the softmax rather than zeroed
/// after it, so the surviving positions still form a proper distribution.
pub(crate) struct AttentionArgs<'a> {
    pub q: &'a [f32],
    pub k: &'a [f32],
    pub v: &'a [f32],
    pub mask: &'a [bool],
    pub head: usize,
    pub head_width: usize,
    pub width: usize,
    pub seq: usize,
}

pub(crate) fn attend(out: &mut [f32], scores: &mut [f32], a: &AttentionArgs<'_>) {
    let base = a.head * a.head_width;
    // `head_width` is a few dozen: exactly representable.
    #[allow(clippy::cast_precision_loss)]
    let hw = a.head_width as f32;
    let inv = 1.0 / libm::sqrtf(hw);
    for (t, score) in scores.iter_mut().enumerate().take(a.seq) {
        let mut acc = 0.0f32;
        for i in 0..a.head_width {
            let qi = a.q.get(base + i).copied().unwrap_or(0.0);
            let ki = a.k.get(t * a.width + base + i).copied().unwrap_or(0.0);
            acc += qi * ki;
        }
        *score = acc * inv;
    }
    let live = scores.get_mut(..a.seq).unwrap_or(&mut []);
    masked_softmax_inplace(live, a.mask);
    for i in 0..a.head_width {
        let mut acc = 0.0f32;
        for t in 0..a.seq {
            let p = live.get(t).copied().unwrap_or(0.0);
            acc += p * a.v.get(t * a.width + base + i).copied().unwrap_or(0.0);
        }
        if let Some(slot_ref) = out.get_mut(base + i) {
            *slot_ref = acc;
        }
    }
}

/// Feed-forward sublayer: `W2(GELU(W1(x)))`.
pub(crate) fn feed_forward(
    out: &mut [f32],
    wide: &mut [f32],
    input: &[f32],
    ff1: &Linear<'_>,
    ff2: &Linear<'_>,
    width: usize,
    ff_width: usize,
) {
    apply(wide, input, ff1, width);
    gelu_inplace(wide.get_mut(..ff_width).unwrap_or(&mut []));
    apply(out, wide.get(..ff_width).unwrap_or(&[]), ff2, ff_width);
}

/// LayerNorm using tensors addressed by layer and slot.
pub(crate) fn norm_layer<'a>(
    out: &mut [f32],
    input: &[f32],
    tensors: &'a [(TensorId, Tensor<'a>)],
    layer: u16,
    gamma_slot: u16,
    beta_slot: u16,
) -> Result<(), RuntimeError> {
    let g = find(tensors, TensorId::layer(layer, gamma_slot).0)?.as_f32()?;
    let b = find(tensors, TensorId::layer(layer, beta_slot).0)?.as_f32()?;
    layer_norm(out, input, g, b, 1e-5);
    Ok(())
}

/// Slot ids for the two LayerNorms in a block, re-exported for the model driver.
pub(crate) const LN1: (u16, u16) = (slot::LN1_GAMMA, slot::LN1_BETA);
pub(crate) const LN2: (u16, u16) = (slot::LN2_GAMMA, slot::LN2_BETA);
