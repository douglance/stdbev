//! Converts trained FP32 tensors into the `STDBEV01` runtime representation.

mod genesis;
mod quantize;
mod write;

pub use genesis::genesis_artifact;
pub use quantize::{Quantized, dequantize, quantize_rows};
pub use write::{ArtifactBuilder, TensorSource};
