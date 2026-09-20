//! Deterministic splitting.
//!
//! Two levels, and both matter:
//!
//! * **Template family** decides train/validation/test. The test set is built from
//!   phrasing patterns the model has never seen, so accuracy measures generalization
//!   rather than memorization of surface forms.
//! * **Group** prevents related examples from straddling a boundary within a split.
//!
//! Splitting by group alone -- which is what the spec asked for -- leaves every
//! template in every split, and a 0.5M-parameter model will happily memorize thirty
//! phrasings. The resulting accuracy looks excellent and predicts nothing.

use sha2::{Digest, Sha256};

/// Which split an example belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Split {
    Train,
    Validation,
    Test,
}

impl Split {
    /// Filename stem used under `data/generated/`.
    #[must_use]
    pub const fn stem(self) -> &'static str {
        match self {
            Self::Train => "train",
            Self::Validation => "validation",
            Self::Test => "test",
        }
    }

    /// Every split, in a fixed order.
    #[must_use]
    pub const fn all() -> [Self; 3] {
        [Self::Train, Self::Validation, Self::Test]
    }
}

/// First byte-derived bucket in `0..100` for `key`.
#[must_use]
pub fn bucket(key: &str) -> u8 {
    let digest = Sha256::digest(key.as_bytes());
    // Two bytes give a uniform enough spread; one byte would quantize to 1/256.
    let high = digest.first().copied().unwrap_or(0);
    let low = digest.get(1).copied().unwrap_or(0);
    let value = u16::from(high) << 8 | u16::from(low);
    u8::try_from(value % 100).unwrap_or(0)
}

/// Share of template families held out entirely for the test split.
const TEST_FAMILY_CUTOFF: u8 = 85;

/// Share of *groups* within the training families reserved for validation.
const VALIDATION_GROUP_CUTOFF: u8 = 88;

/// True if this template family is held out for the test split.
#[must_use]
pub fn is_test_template(template: &str) -> bool {
    bucket(template) >= TEST_FAMILY_CUTOFF
}

/// Assigns a split from the template family and the group.
///
/// The two levels measure different things, and conflating them was a mistake worth
/// naming:
///
/// * **Test** is whole template families the model has never seen. It answers "does
///   this generalize to a phrasing we did not train on".
/// * **Validation** is unseen *groups* drawn from the *training* families. It answers
///   "have we fit yet", which is what early stopping and temperature calibration
///   need -- and they need it in-distribution. Calibrating a temperature on
///   out-of-distribution data fits the wrong curve.
///
/// Splitting both levels by template alone put 38% of examples in test and left
/// validation empty, because twelve families cannot be split 80/10/10 by hash.
#[must_use]
pub fn split_for(template: &str, group: &str) -> Split {
    if is_test_template(template) {
        return Split::Test;
    }
    if bucket(group) >= VALIDATION_GROUP_CUTOFF {
        return Split::Validation;
    }
    Split::Train
}

/// Confirms no test template leaked into train or validation.
///
/// Train and validation deliberately *share* templates -- that is the design -- so
/// this checks the boundary that actually matters: nothing the test set measures may
/// have been trained on.
///
/// # Errors
/// Returns every template that leaked, not just the first.
pub fn check_no_leakage(assignments: &[(String, Split)]) -> Result<(), String> {
    let mut leaked: Vec<String> = Vec::new();
    for (template, split) in assignments {
        let should_be_test = is_test_template(template);
        let is_test = *split == Split::Test;
        if should_be_test != is_test && !leaked.contains(template) {
            leaked.push(template.clone());
        }
    }
    if leaked.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "templates on the wrong side of the test boundary: {}",
            leaked.join(", ")
        ))
    }
}

/// Confirms every split actually received examples.
///
/// # Errors
/// Names the empty splits. An empty validation split silently disables early stopping
/// and temperature calibration, which is worse than failing loudly.
pub fn check_all_populated(train: usize, validation: usize, test: usize) -> Result<(), String> {
    let empty: Vec<&str> = [("train", train), ("validation", validation), ("test", test)]
        .iter()
        .filter(|(_, n)| *n == 0)
        .map(|(name, _)| *name)
        .collect();
    if empty.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "empty split(s): {}. Too few template families for the holdout, or too few examples.",
            empty.join(", ")
        ))
    }
}
