//! Framework-free numerical kernels for STDBEV inference.
//!
//! Two invariants govern this crate, and both exist to make native/WASM parity
//! exact rather than approximate:
//!
//! 1. Every transcendental goes through [`libm`], never `std`. Platform libm
//!    differs between macOS and the wasm build, and those differences propagate
//!    through `exp` in softmax and `tanh` in GELU.
//! 2. No floating-point reassociation. Sums accumulate in index order, so the
//!    same inputs produce bit-identical outputs on every target.

#![no_std]

mod activation;
mod attention;
mod matmul;
mod norm;

pub use activation::{gelu, gelu_inplace, softmax_inplace};
pub use attention::{masked_mean_pool, masked_softmax_inplace};
pub use matmul::{dot, matvec_f32, matvec_i8};
pub use norm::layer_norm;

/// Index of the largest value, and that value.
///
/// Ties resolve to the lowest index, which makes the choice deterministic across
/// targets -- a tie broken by iteration order would be a parity failure waiting to
/// happen. Returns `(0, 0.0)` for an empty or all-non-finite slice.
#[must_use]
pub fn argmax(values: &[f32]) -> (usize, f32) {
    let mut best = (0usize, f32::NEG_INFINITY);
    for (i, v) in values.iter().enumerate() {
        if *v > best.1 {
            best = (i, *v);
        }
    }
    if best.1.is_finite() {
        best
    } else {
        (best.0, 0.0)
    }
}
