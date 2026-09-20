//! Canonical text formatting and byte tokenization.
//!
//! Everything the model sees passes through here, so this crate defines the contract
//! between the trainer and the runtime: identical bytes in, identical bytes out.

mod canonical;
mod tokenize;

pub use canonical::{context_text, option_texts};
pub use tokenize::{PAD, Tokens, encode, token_id};
