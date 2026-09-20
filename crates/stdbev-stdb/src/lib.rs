//! STDBEV as a SpacetimeDB module.
//!
//! The model is immutable data compiled into the WASM binary. That is not a
//! convenience: SpacetimeDB documents relying on globals or statics persisting
//! between reducer calls as *undefined behavior*, and the host may run any call on a
//! fresh module instance. `include_bytes!` sidesteps the question entirely, because
//! there is no mutable state to lose.

mod model;
mod reducers;

pub use model::MODEL_BYTES;
