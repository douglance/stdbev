//! Attention-support kernels: masked softmax and masked mean pooling.

use crate::activation::softmax_inplace;

/// Softmax over `scores`, with masked-out positions forced to zero probability.
///
/// Masking is applied by setting scores to `-inf` before the softmax rather than
/// zeroing probabilities after it, so the surviving positions still sum to exactly 1.
pub fn masked_softmax_inplace(scores: &mut [f32], mask: &[bool]) {
    for (i, s) in scores.iter_mut().enumerate() {
        if !mask.get(i).copied().unwrap_or(false) {
            *s = f32::NEG_INFINITY;
        }
    }
    softmax_inplace(scores);
}

/// Mean of the rows of `tokens` (row-major, `width` columns) where `mask` is true.
///
/// This is how an option's token sequence collapses to the single vector that queries
/// the context. With no valid tokens the result is all zeros.
pub fn masked_mean_pool(out: &mut [f32], tokens: &[f32], mask: &[bool], width: usize) {
    for v in out.iter_mut() {
        *v = 0.0;
    }
    let mut count = 0usize;
    for (t, valid) in mask.iter().enumerate() {
        if !valid {
            continue;
        }
        let Some(row) = tokens.get(t * width..(t + 1) * width) else {
            continue;
        };
        for (slot, v) in out.iter_mut().zip(row) {
            *slot += v;
        }
        count += 1;
    }
    if count > 0 {
        // `count` is bounded by the option window, well inside f32's exact range.
        #[allow(clippy::cast_precision_loss)]
        let inv = 1.0 / count as f32;
        for v in out.iter_mut() {
            *v *= inv;
        }
    }
}
