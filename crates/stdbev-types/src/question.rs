//! The three public question primitives.
//!
//! Choice, Noul and Score are deterministic transformations around one
//! probability-producing core: every question becomes a context plus a dynamic set of
//! options, and the answer is a softmax over those options.

use serde::{Deserialize, Serialize};

/// Application state plus a typed question.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DecisionRequest {
    pub state: String,
    pub question: Question,
}

/// One of the three public primitives.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "type")]
pub enum Question {
    Choice(ChoiceQuestion),
    Noul(NoulQuestion),
    Score(ScoreQuestion),
}

impl Question {
    /// Instructions shared by every primitive.
    #[must_use]
    pub fn instructions(&self) -> &str {
        match self {
            Self::Choice(q) => &q.instructions,
            Self::Noul(q) => &q.instructions,
            Self::Score(q) => &q.instructions,
        }
    }

    /// How many options this question will present to the scorer.
    ///
    /// Noul is always 2: it is a Choice between FALSE and TRUE.
    #[must_use]
    pub fn option_count(&self) -> usize {
        match self {
            Self::Choice(q) => q.criteria.len(),
            Self::Noul(_) => 2,
            Self::Score(q) => q.levels.len(),
        }
    }
}

/// Pick one of a dynamic set of named options.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChoiceQuestion {
    pub instructions: String,
    pub criteria: Vec<ChoiceCriterion>,
}

/// One candidate in a [`ChoiceQuestion`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChoiceCriterion {
    pub label: String,
    pub description: String,
}

/// Estimate the probability that a condition holds.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NoulQuestion {
    pub instructions: String,
    pub false_description: String,
    pub true_description: String,
}

/// Score against an ordered ladder of levels.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScoreQuestion {
    pub instructions: String,
    pub levels: Vec<ScoreLevel>,
}

/// One rung of a [`ScoreQuestion`] ladder. Order is meaningful.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScoreLevel {
    pub label: String,
    pub description: String,
}
