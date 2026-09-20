//! Layer normalization.

/// `out = gamma * (x - mean) / sqrt(var + eps) + beta`, computed in index order.
///
/// Variance uses the biased (population) estimator, matching Candle's `LayerNorm`
/// so the trainer and the runtime agree.
pub fn layer_norm(out: &mut [f32], input: &[f32], gamma: &[f32], beta: &[f32], eps: f32) {
    let n = input.len();
    if n == 0 {
        return;
    }
    // `n` is the model width, a few hundred at most: exactly representable.
    #[allow(clippy::cast_precision_loss)]
    let inv_n = 1.0 / n as f32;
    let mean = input.iter().sum::<f32>() * inv_n;
    let var = input.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() * inv_n;
    let inv_std = 1.0 / libm::sqrtf(var + eps);
    for (i, slot) in out.iter_mut().enumerate() {
        let g = gamma.get(i).copied().unwrap_or(1.0);
        let b = beta.get(i).copied().unwrap_or(0.0);
        let normed = (input.get(i).copied().unwrap_or(0.0) - mean) * inv_std;
        *slot = normed * g + b;
    }
}
