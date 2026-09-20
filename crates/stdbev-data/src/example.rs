//! The training example format.

use serde::{Deserialize, Serialize};

/// One training example: a context, a dynamic option set, and a target distribution.
///
/// `target_probabilities` is a full distribution rather than a label index so the same
/// format carries both hard synthetic labels `[0, 1, 0]` and soft teacher
/// distributions `[0.51, 0.36, 0.13]`. Distillation needs no second format.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrainingExample {
    pub id: String,
    /// Examples sharing a group never cross a split boundary.
    pub group: String,
    /// The phrasing pattern this was generated from. Held out wholesale, so the test
    /// set uses wordings the model has never seen.
    #[serde(default)]
    pub template: String,
    pub context: String,
    pub options: Vec<String>,
    pub target_probabilities: Vec<f32>,
}

/// Why an example was rejected.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DataError {
    Empty(&'static str),
    OptionCount(usize),
    LengthMismatch { options: usize, targets: usize },
    NotADistribution(String),
}

impl core::fmt::Display for DataError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Empty(field) => write!(f, "{field} is empty"),
            Self::OptionCount(n) => write!(f, "expected 2..=16 options, got {n}"),
            Self::LengthMismatch { options, targets } => {
                write!(f, "{options} options but {targets} target probabilities")
            }
            Self::NotADistribution(why) => write!(f, "target_probabilities: {why}"),
        }
    }
}

impl core::error::Error for DataError {}

/// Checks that `values` is a probability distribution.
fn check_distribution(values: &[f32]) -> Result<(), DataError> {
    let mut sum = 0.0f32;
    for p in values {
        if !p.is_finite() {
            return Err(DataError::NotADistribution("not finite".into()));
        }
        if *p < 0.0 {
            return Err(DataError::NotADistribution(format!("negative: {p}")));
        }
        sum += p;
    }
    if (sum - 1.0).abs() > 1e-4 {
        return Err(DataError::NotADistribution(format!("sums to {sum}, not 1")));
    }
    Ok(())
}

impl TrainingExample {
    /// Checks every structural invariant.
    ///
    /// # Errors
    /// Returns the first violation found.
    pub fn validate(&self) -> Result<(), DataError> {
        if self.id.is_empty() {
            return Err(DataError::Empty("id"));
        }
        if self.group.is_empty() {
            return Err(DataError::Empty("group"));
        }
        if self.context.is_empty() {
            return Err(DataError::Empty("context"));
        }
        let n = self.options.len();
        if !(2..=16).contains(&n) {
            return Err(DataError::OptionCount(n));
        }
        if n != self.target_probabilities.len() {
            return Err(DataError::LengthMismatch {
                options: n,
                targets: self.target_probabilities.len(),
            });
        }
        check_distribution(&self.target_probabilities)
    }

    /// Index of the highest-probability option. Ties resolve to the lowest index.
    #[must_use]
    pub fn label(&self) -> usize {
        stdbev_math::argmax(&self.target_probabilities).0
    }
}
