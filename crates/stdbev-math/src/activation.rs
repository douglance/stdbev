//! Activations and softmax.
//!
//! Both routines call [`libm`] rather than `std`, so results are bit-identical on
//! native and wasm. See the crate docs for why that matters.

/// Converts a length to `f32`.
///
/// Lengths here are bounded by the model config -- a few thousand at most -- which is
/// far inside the range f32 represents exactly, so nothing is lost.
#[allow(clippy::cast_precision_loss)]
#[inline]
fn len_as_f32(n: usize) -> f32 {
    n as f32
}

/// Tanh-approximation GELU, pinned explicitly.
///
/// The exact-erf and tanh-approximation forms of GELU differ by around 1e-3, far above
/// the 1e-5 native/WASM parity budget. Trainer and runtime must therefore agree on
/// *which* GELU, not merely on "GELU".
#[must_use]
pub fn gelu(x: f32) -> f32 {
    const SQRT_2_OVER_PI: f32 = 0.797_884_6;
    const COEFF: f32 = 0.044_715;
    // Plain mul/add, never `mul_add`: FMA rounds once where `a * b + c` rounds twice,
    // and that difference is not guaranteed identical across targets.
    let inner = SQRT_2_OVER_PI * (x + COEFF * x * x * x);
    0.5 * x * (1.0 + libm::tanhf(inner))
}

/// Applies [`gelu`] to every element in place.
pub fn gelu_inplace(values: &mut [f32]) {
    for v in values.iter_mut() {
        *v = gelu(*v);
    }
}

/// Numerically stable softmax in place.
///
/// Subtracting the max before `exp` keeps large logits from overflowing to `inf`,
/// which would otherwise yield NaN probabilities exactly when the model is most
/// confident. An all-`-inf` input -- every position masked -- gives a uniform
/// distribution rather than NaN.
pub fn softmax_inplace(values: &mut [f32]) {
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        let uniform = if values.is_empty() {
            0.0
        } else {
            1.0 / len_as_f32(values.len())
        };
        for v in values.iter_mut() {
            *v = uniform;
        }
        return;
    }
    let mut sum = 0.0f32;
    for v in values.iter_mut() {
        *v = libm::expf(*v - max);
        sum += *v;
    }
    if sum > 0.0 {
        for v in values.iter_mut() {
            *v /= sum;
        }
    }
}
