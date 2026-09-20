//! Domain types, validation and errors for STDBEV.
//!
//! This crate is the vocabulary the rest of the workspace shares. It deliberately
//! depends on no tensor library and no database, so the trainer, the runtime and the
//! SpacetimeDB module can all agree on what a decision *is* without agreeing on how
//! it is computed.

mod answer;
mod artifact;
mod config;
mod error;
mod question;
mod validate;

pub use answer::{ChoiceAnswer, DecisionAnswer, NoulAnswer, ScoreAnswer};
pub use artifact::{
    DIRECTORY_ENTRY_BYTES, Dtype, FORMAT_VERSION, HEADER_BYTES, MAGIC, TensorId, global, slot,
};
pub use config::{ModelConfig, OptionEncoder, OptionFormat};
pub use error::{StdbevError, ValidationError};
pub use question::{
    ChoiceCriterion, ChoiceQuestion, DecisionRequest, NoulQuestion, Question, ScoreLevel,
    ScoreQuestion,
};
pub use validate::{check_distribution, limits};
