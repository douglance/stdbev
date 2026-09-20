//! Runtime errors.

use core::fmt;

/// Why an artifact could not be loaded or a decision could not be produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuntimeError {
    /// The artifact bytes are not a well-formed `STDBEV01` file.
    Malformed(&'static str),
    /// The artifact is well-formed but describes a different architecture.
    ArchitectureMismatch { field: &'static str },
    /// A tensor the runtime needs is absent.
    MissingTensor(u16),
    /// The request failed validation.
    Invalid(String),
}

impl fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Malformed(why) => write!(f, "malformed artifact: {why}"),
            Self::ArchitectureMismatch { field } => {
                write!(f, "artifact architecture mismatch: {field}")
            }
            Self::MissingTensor(id) => write!(f, "artifact missing tensor 0x{id:04x}"),
            Self::Invalid(why) => write!(f, "invalid request: {why}"),
        }
    }
}

impl core::error::Error for RuntimeError {}

impl From<stdbev_types::ValidationError> for RuntimeError {
    fn from(e: stdbev_types::ValidationError) -> Self {
        Self::Invalid(e.to_string())
    }
}
