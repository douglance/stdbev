//! Quantized STDBEV inference.
//!
//! This is the exact code compiled into the SpacetimeDB module, so anything that runs
//! here runs there. It depends on no ML framework and contains no `unsafe`.

mod encoder;
mod error;
mod model;
mod parse;
mod runtime;
mod scratch;
mod view;

pub use error::RuntimeError;
pub use parse::{Header, TensorEntry, parse_directory, parse_header};
pub use runtime::Runtime;
pub use scratch::Scratch;
