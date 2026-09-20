//! Borrowed view over an artifact.
//!
//! ## Why f32 tensors are copied but INT8 tensors are not
//!
//! `include_bytes!` guarantees only 1-byte alignment, and reading `&[f32]` out of an
//! arbitrarily-aligned slice is unsound -- which cannot be papered over here, because
//! the workspace forbids `unsafe`. The resolution follows the sizes: f32 tensors are
//! LayerNorm parameters, biases and quantization scales, together around 30 KB against
//! 504 KB of INT8 weights. Copying the small ones costs almost nothing; borrowing the
//! large ones is what matters, and `i8` has the same 1-byte alignment as `u8`, so it
//! needs no copy at all.

use stdbev_types::{Dtype, TensorId};

use crate::error::RuntimeError;
use crate::parse::TensorEntry;

/// One tensor, resolved against the artifact bytes.
pub enum Tensor<'a> {
    /// Borrowed in place -- no alignment requirement.
    I8(&'a [i8]),
    /// Copied at load time to guarantee alignment.
    F32(Vec<f32>),
}

impl Tensor<'_> {
    /// Quantized weights, or an error if this tensor is f32.
    pub fn as_i8(&self) -> Result<&[i8], RuntimeError> {
        match self {
            Self::I8(v) => Ok(v),
            Self::F32(_) => Err(RuntimeError::Malformed("expected i8 tensor")),
        }
    }

    /// Float values, or an error if this tensor is quantized.
    pub fn as_f32(&self) -> Result<&[f32], RuntimeError> {
        match self {
            Self::F32(v) => Ok(v),
            Self::I8(_) => Err(RuntimeError::Malformed("expected f32 tensor")),
        }
    }
}

/// Resolves a directory entry into a usable tensor.
pub(crate) fn resolve<'a>(entry: &TensorEntry, blob: &'a [u8]) -> Result<Tensor<'a>, RuntimeError> {
    let bytes = entry.bytes(blob)?;
    match entry.dtype {
        // `i8` and `u8` share size and alignment, so this cast never fails.
        Dtype::I8 => Ok(Tensor::I8(bytemuck::cast_slice(bytes))),
        Dtype::F32 => {
            let (words, tail) = bytes.as_chunks::<4>();
            if !tail.is_empty() {
                return Err(RuntimeError::Malformed(
                    "f32 tensor length not a multiple of 4",
                ));
            }
            Ok(Tensor::F32(
                words.iter().copied().map(f32::from_le_bytes).collect(),
            ))
        }
    }
}

/// Looks up a tensor by id.
pub(crate) fn find<'a>(
    tensors: &'a [(TensorId, Tensor<'a>)],
    id: u16,
) -> Result<&'a Tensor<'a>, RuntimeError> {
    tensors
        .iter()
        .find(|(t, _)| t.0 == id)
        .map(|(_, v)| v)
        .ok_or(RuntimeError::MissingTensor(id))
}
