//! Input validation. Runs before canonicalization, so malformed input never reaches
//! the tokenizer or the model.

use crate::error::ValidationError;
use crate::question::{DecisionRequest, Question};

/// Hard bounds on every field a caller controls.
pub mod limits {
    pub const STATE_BYTES: usize = 16 * 1024;
    pub const INSTRUCTIONS_BYTES: usize = 4 * 1024;
    pub const DESCRIPTION_BYTES: usize = 2 * 1024;
    pub const LABEL_BYTES: usize = 128;
    pub const CHOICE_MIN: usize = 2;
    pub const CHOICE_MAX: usize = 16;
    pub const SCORE_MIN: usize = 2;
    pub const SCORE_MAX: usize = 10;
    /// Tolerance on a probability distribution summing to 1.
    pub const SUM_TOLERANCE: f32 = 1e-5;
}

fn text(value: &str, field: &'static str, limit: usize) -> Result<(), ValidationError> {
    if value.len() > limit {
        return Err(ValidationError::TooLong {
            field,
            actual: value.len(),
            limit,
        });
    }
    Ok(())
}

fn label(value: &str, field: &'static str) -> Result<(), ValidationError> {
    if value.is_empty() {
        return Err(ValidationError::Empty { field });
    }
    text(value, field, limits::LABEL_BYTES)
}

fn unique_labels<'a>(
    labels: impl Iterator<Item = &'a str>,
    field: &'static str,
) -> Result<(), ValidationError> {
    let mut seen: Vec<&str> = Vec::new();
    for l in labels {
        if seen.contains(&l) {
            return Err(ValidationError::DuplicateLabel {
                field,
                label: l.to_owned(),
            });
        }
        seen.push(l);
    }
    Ok(())
}

fn count(
    actual: usize,
    field: &'static str,
    min: usize,
    max: usize,
) -> Result<(), ValidationError> {
    if actual < min || actual > max {
        return Err(ValidationError::Count {
            field,
            actual,
            min,
            max,
        });
    }
    Ok(())
}

impl DecisionRequest {
    /// Checks every bound in [`limits`].
    ///
    /// # Errors
    /// Returns the first violated bound.
    pub fn validate(&self) -> Result<(), ValidationError> {
        text(&self.state, "state", limits::STATE_BYTES)?;
        text(
            self.question.instructions(),
            "instructions",
            limits::INSTRUCTIONS_BYTES,
        )?;
        self.question.validate()
    }
}

/// Validates a label/description pair shared by Choice criteria and Score levels.
fn option_entry(
    entry_label: &str,
    description: &str,
    label_field: &'static str,
    description_field: &'static str,
) -> Result<(), ValidationError> {
    label(entry_label, label_field)?;
    text(description, description_field, limits::DESCRIPTION_BYTES)
}

impl Question {
    /// Checks the option set for this primitive.
    ///
    /// # Errors
    /// Returns the first violated bound.
    pub fn validate(&self) -> Result<(), ValidationError> {
        match self {
            Self::Choice(q) => Self::validate_choice(q),
            Self::Noul(q) => Self::validate_noul(q),
            Self::Score(q) => Self::validate_score(q),
        }
    }

    fn validate_choice(q: &crate::question::ChoiceQuestion) -> Result<(), ValidationError> {
        count(
            q.criteria.len(),
            "criteria",
            limits::CHOICE_MIN,
            limits::CHOICE_MAX,
        )?;
        for c in &q.criteria {
            option_entry(
                &c.label,
                &c.description,
                "choice label",
                "choice description",
            )?;
        }
        unique_labels(q.criteria.iter().map(|c| c.label.as_str()), "choice label")
    }

    fn validate_noul(q: &crate::question::NoulQuestion) -> Result<(), ValidationError> {
        text(
            &q.false_description,
            "false_description",
            limits::DESCRIPTION_BYTES,
        )?;
        text(
            &q.true_description,
            "true_description",
            limits::DESCRIPTION_BYTES,
        )
    }

    fn validate_score(q: &crate::question::ScoreQuestion) -> Result<(), ValidationError> {
        count(
            q.levels.len(),
            "levels",
            limits::SCORE_MIN,
            limits::SCORE_MAX,
        )?;
        for l in &q.levels {
            option_entry(&l.label, &l.description, "score label", "score description")?;
        }
        unique_labels(q.levels.iter().map(|l| l.label.as_str()), "score label")
    }
}

/// Checks that `probabilities` is a valid distribution.
///
/// # Errors
/// Returns [`ValidationError::NotADistribution`] if any value is non-finite or
/// negative, or the sum is outside [`limits::SUM_TOLERANCE`] of 1.
pub fn check_distribution(probabilities: &[f32]) -> Result<(), ValidationError> {
    let mut sum = 0.0f32;
    for p in probabilities {
        if !p.is_finite() || *p < 0.0 || *p > 1.0 {
            return Err(ValidationError::NotADistribution { sum: *p });
        }
        sum += p;
    }
    if (sum - 1.0).abs() > limits::SUM_TOLERANCE {
        return Err(ValidationError::NotADistribution { sum });
    }
    Ok(())
}
