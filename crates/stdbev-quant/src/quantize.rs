//! Symmetric per-output-channel INT8 quantization.

/// A quantized matrix: one `i8` per weight, one `f32` scale per output row.
pub struct Quantized {
    pub values: Vec<i8>,
    pub scales: Vec<f32>,
}

/// Quantizes a row-major matrix, one scale per output row.
///
/// Per-row rather than per-tensor because rows of a learned matrix routinely differ
/// in magnitude by an order of magnitude; a single global scale would quantize the
/// small rows to near-zero.
///
/// An all-zero row gets scale 1 rather than 0, so reconstruction stays finite instead
/// of producing NaN.
#[must_use]
pub fn quantize_rows(weights: &[f32], cols: usize) -> Quantized {
    if cols == 0 {
        return Quantized {
            values: Vec::new(),
            scales: Vec::new(),
        };
    }
    let rows = weights.len() / cols;
    let mut values = Vec::with_capacity(weights.len());
    let mut scales = Vec::with_capacity(rows);
    for r in 0..rows {
        let row = weights.get(r * cols..(r + 1) * cols).unwrap_or(&[]);
        let max = row.iter().fold(0.0f32, |m, w| m.max(w.abs()));
        let scale = if max == 0.0 { 1.0 } else { max / 127.0 };
        scales.push(scale);
        for w in row {
            let q = (w / scale).round().clamp(-127.0, 127.0);
            // The clamp above already guarantees -127..=127, so the cast is exact.
            // Truncation toward zero cannot occur: `round` made it integral first.
            #[allow(clippy::cast_possible_truncation)]
            values.push(q as i8);
        }
    }
    Quantized { values, scales }
}

/// Reconstructs the approximate FP32 matrix, for error measurement.
#[must_use]
pub fn dequantize(q: &Quantized, cols: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(q.values.len());
    for (i, v) in q.values.iter().enumerate() {
        let scale = q.scales.get(i / cols.max(1)).copied().unwrap_or(1.0);
        out.push(f32::from(*v) * scale);
    }
    out
}
