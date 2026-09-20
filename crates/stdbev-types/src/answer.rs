//! Decision results.

use serde::{Deserialize, Serialize};

/// The answer to a [`crate::Question`], shaped to match the primitive that was asked.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "type")]
pub enum DecisionAnswer {
    Choice(ChoiceAnswer),
    Noul(NoulAnswer),
    Score(ScoreAnswer),
}

impl DecisionAnswer {
    /// The full calibrated distribution over the live option set.
    #[must_use]
    pub fn probabilities(&self) -> &[f32] {
        match self {
            Self::Choice(a) => &a.probabilities,
            Self::Noul(a) => &a.probabilities,
            Self::Score(a) => &a.probabilities,
        }
    }

    /// Identity of the model that produced this answer.
    #[must_use]
    pub fn model_id(&self) -> &str {
        match self {
            Self::Choice(a) => &a.model_id,
            Self::Noul(a) => &a.model_id,
            Self::Score(a) => &a.model_id,
        }
    }

    /// Index of the selected option.
    ///
    /// For Noul this is 1 when P(TRUE) wins, matching the FALSE/TRUE option order.
    #[must_use]
    pub fn selected_index(&self) -> usize {
        match self {
            Self::Choice(a) => a.selected_index,
            Self::Noul(a) => usize::from(a.probability >= 0.5),
            Self::Score(a) => a.selected_index,
        }
    }
}

/// Result of a Choice question.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChoiceAnswer {
    pub selected_index: usize,
    pub selected_label: String,
    pub confidence: f32,
    pub probabilities: Vec<f32>,
    pub model_id: String,
}

/// Result of a Noul question.
///
/// There is deliberately **no** `confidence` field. `jevon` defines a noul as
/// "answered as a probability of yes when used alone, with no separate confidence" --
/// a noul of 0.5 means yes and no are equally likely, which is information, not
/// uncertainty about the estimate. Adding a second number invites callers to threshold
/// on the wrong one.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NoulAnswer {
    /// P(TRUE). Always equal to `probabilities[1]`.
    pub probability: f32,
    /// `[P(FALSE), P(TRUE)]`.
    pub probabilities: [f32; 2],
    pub model_id: String,
}

/// Result of a Score question.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScoreAnswer {
    /// Expected level index, `Σ probability[i] * i`, in `0..=levels-1`.
    ///
    /// Reported alongside the full distribution rather than instead of it: the mean
    /// of a bimodal distribution names a level nobody voted for.
    pub score: f32,
    pub selected_index: usize,
    pub selected_label: String,
    pub confidence: f32,
    pub probabilities: Vec<f32>,
    pub model_id: String,
}
