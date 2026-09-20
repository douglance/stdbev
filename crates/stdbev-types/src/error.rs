//! Error types.

use thiserror::Error;

/// A rejected input, named precisely enough that a caller can fix it without guessing.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum ValidationError {
    #[error("{field} is empty")]
    Empty { field: &'static str },

    #[error("{field} is {actual} bytes, limit is {limit}")]
    TooLong {
        field: &'static str,
        actual: usize,
        limit: usize,
    },

    #[error("{field} has {actual} entries, expected {min}..={max}")]
    Count {
        field: &'static str,
        actual: usize,
        min: usize,
        max: usize,
    },

    #[error("duplicate {field} label {label:?}")]
    DuplicateLabel { field: &'static str, label: String },

    #[error("probabilities must be finite, non-negative and sum to 1.0; got sum {sum}")]
    NotADistribution { sum: f32 },
}

/// Anything that can go wrong outside validation.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum StdbevError {
    #[error(transparent)]
    Validation(#[from] ValidationError),

    #[error("artifact: {0}")]
    Artifact(String),

    #[error("inference: {0}")]
    Inference(String),
}
