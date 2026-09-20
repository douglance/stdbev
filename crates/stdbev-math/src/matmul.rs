//! Matrix-vector kernels.
//!
//! [`matvec_i8`] is the hot loop of the entire runtime: roughly 86% of a decision's
//! arithmetic passes through it, so its throughput sets the latency budget.

/// Number of independent accumulators in the INT8 inner loop.
///
/// A single accumulator serializes the whole reduction: float addition is not
/// associative, so LLVM may not reassociate it into SIMD lanes, and the loop runs at
/// scalar speed. Splitting into `LANES` independent chains breaks the dependency and
/// lets both NEON and wasm `simd128` vectorize, while the *fixed* pairwise reduction
/// in [`reduce_lanes`] keeps the summation order identical on every target -- which
/// is what makes native/WASM parity exact instead of approximate.
const LANES: usize = 8;

/// Dot product of two equal-length slices, accumulated in index order.
#[must_use]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// `out[o] = bias[o] + dot(input, weights[o])` for a row-major FP32 matrix.
///
/// Used by the FP32 reference model the quantized runtime is checked against.
pub fn matvec_f32(
    out: &mut [f32],
    input: &[f32],
    weights: &[f32],
    bias: Option<&[f32]>,
    cols: usize,
) {
    for (o, slot) in out.iter_mut().enumerate() {
        let row = weights.get(o * cols..(o + 1) * cols).unwrap_or_default();
        let acc = dot(input, row);
        *slot = bias.and_then(|b| b.get(o)).map_or(acc, |b| acc + b);
    }
}

/// Reduces `LANES` partial sums in a fixed pairwise order.
///
/// Hardcoded rather than left to `iter().sum()` so the result cannot depend on how a
/// given backend chooses to unroll it.
#[inline]
fn reduce_lanes(acc: [f32; LANES]) -> f32 {
    let a = (acc[0] + acc[1]) + (acc[2] + acc[3]);
    let b = (acc[4] + acc[5]) + (acc[6] + acc[7]);
    a + b
}

/// Adds one LANES-wide slab of products into the lane accumulators.
///
/// Fixed-size arrays on both sides mean no bounds checks and a fully unrolled body.
#[inline]
fn accumulate(acc: &mut [f32; LANES], xc: &[f32; LANES], wc: [i8; LANES]) {
    for ((a, x), w) in acc.iter_mut().zip(xc).zip(wc) {
        *a += x * f32::from(w);
    }
}

/// `out[o] = bias[o] + scale[o] * Σ input[i] * weight[o][i]` over INT8 weights.
///
/// Weights stay INT8 in memory and widen to `f32` inside the accumulation;
/// activations remain FP32. The per-output-channel `scale` is applied once to the
/// finished accumulator rather than per element, keeping the inner loop a plain
/// multiply-add.
pub fn matvec_i8(
    out: &mut [f32],
    input: &[f32],
    weights: &[i8],
    scale: &[f32],
    bias: Option<&[f32]>,
    cols: usize,
) {
    for (o, slot) in out.iter_mut().enumerate() {
        let row = weights.get(o * cols..(o + 1) * cols).unwrap_or_default();
        let mut acc = [0.0f32; LANES];
        // `as_chunks` gives fixed-size arrays rather than slices, so the inner loop
        // has a compile-time-known length, needs no bounds check, and lowers to whole
        // SIMD registers. Phase 0 measured this path at 26.5 GFLOP/s native against
        // 2.3 for the naive one-accumulator version.
        let (row_chunks, row_tail) = row.as_chunks::<LANES>();
        let (input_chunks, input_tail) = input.as_chunks::<LANES>();
        for (wc, xc) in row_chunks.iter().zip(input_chunks) {
            accumulate(&mut acc, xc, *wc);
        }
        let mut total = reduce_lanes(acc);
        for (x, w) in input_tail.iter().zip(row_tail) {
            total += x * f32::from(*w);
        }
        let scaled = total * scale.get(o).copied().unwrap_or(1.0);
        *slot = bias.and_then(|b| b.get(o)).map_or(scaled, |b| scaled + b);
    }
}
