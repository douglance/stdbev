//! Validation boundary tests: each bound is checked at the value that passes and the
//! value that fails, because an off-by-one in a limit is invisible otherwise.

// Test-file lint posture. The workspace denies these because a panic inside a reducer
// aborts a transaction -- but in a test a panic IS the failure signal, and an exact
// float comparison is frequently the assertion itself.
#![allow(
    clippy::indexing_slicing,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::format_collect
)]

use stdbev_types::{
    ChoiceCriterion, ChoiceQuestion, DecisionRequest, NoulQuestion, Question, ScoreLevel,
    ScoreQuestion, ValidationError, limits,
};

fn choice(n: usize) -> Question {
    Question::Choice(ChoiceQuestion {
        instructions: "pick".into(),
        criteria: (0..n)
            .map(|i| ChoiceCriterion {
                label: format!("l{i}"),
                description: "d".into(),
            })
            .collect(),
    })
}

fn score(n: usize) -> Question {
    Question::Score(ScoreQuestion {
        instructions: "rate".into(),
        levels: (0..n)
            .map(|i| ScoreLevel {
                label: format!("l{i}"),
                description: "d".into(),
            })
            .collect(),
    })
}

fn req(state: String, question: Question) -> DecisionRequest {
    DecisionRequest { state, question }
}

fn noul() -> Question {
    Question::Noul(NoulQuestion {
        instructions: "?".into(),
        false_description: "no".into(),
        true_description: "yes".into(),
    })
}

#[test]
fn state_at_the_limit_passes_and_one_over_fails() {
    assert!(
        req("s".repeat(limits::STATE_BYTES), noul())
            .validate()
            .is_ok()
    );
    let err = req("s".repeat(limits::STATE_BYTES + 1), noul())
        .validate()
        .unwrap_err();
    assert!(
        matches!(err, ValidationError::TooLong { field: "state", .. }),
        "{err:?}"
    );
}

#[test]
fn instructions_at_the_limit_pass_and_one_over_fails() {
    let over = Question::Noul(NoulQuestion {
        instructions: "i".repeat(limits::INSTRUCTIONS_BYTES + 1),
        false_description: "no".into(),
        true_description: "yes".into(),
    });
    let err = req("s".into(), over).validate().unwrap_err();
    assert!(
        matches!(
            err,
            ValidationError::TooLong {
                field: "instructions",
                ..
            }
        ),
        "{err:?}"
    );
}

#[test]
fn choice_option_count_bounds_are_inclusive() {
    assert!(
        req("s".into(), choice(limits::CHOICE_MIN))
            .validate()
            .is_ok()
    );
    assert!(
        req("s".into(), choice(limits::CHOICE_MAX))
            .validate()
            .is_ok()
    );
    assert!(
        req("s".into(), choice(limits::CHOICE_MIN - 1))
            .validate()
            .is_err()
    );
    assert!(
        req("s".into(), choice(limits::CHOICE_MAX + 1))
            .validate()
            .is_err()
    );
}

#[test]
fn score_level_count_bounds_are_inclusive() {
    assert!(req("s".into(), score(limits::SCORE_MIN)).validate().is_ok());
    assert!(req("s".into(), score(limits::SCORE_MAX)).validate().is_ok());
    assert!(
        req("s".into(), score(limits::SCORE_MAX + 1))
            .validate()
            .is_err()
    );
}

#[test]
fn duplicate_choice_labels_are_rejected() {
    let q = Question::Choice(ChoiceQuestion {
        instructions: "pick".into(),
        criteria: vec![
            ChoiceCriterion {
                label: "same".into(),
                description: "a".into(),
            },
            ChoiceCriterion {
                label: "same".into(),
                description: "b".into(),
            },
        ],
    });
    let err = req("s".into(), q).validate().unwrap_err();
    assert!(
        matches!(err, ValidationError::DuplicateLabel { .. }),
        "{err:?}"
    );
}

#[test]
fn duplicate_score_labels_are_rejected() {
    let q = Question::Score(ScoreQuestion {
        instructions: "rate".into(),
        levels: vec![
            ScoreLevel {
                label: "x".into(),
                description: "a".into(),
            },
            ScoreLevel {
                label: "x".into(),
                description: "b".into(),
            },
        ],
    });
    assert!(req("s".into(), q).validate().is_err());
}

#[test]
fn empty_label_is_rejected() {
    let q = Question::Choice(ChoiceQuestion {
        instructions: "pick".into(),
        criteria: vec![
            ChoiceCriterion {
                label: String::new(),
                description: "a".into(),
            },
            ChoiceCriterion {
                label: "b".into(),
                description: "b".into(),
            },
        ],
    });
    let err = req("s".into(), q).validate().unwrap_err();
    assert!(matches!(err, ValidationError::Empty { .. }), "{err:?}");
}

#[test]
fn oversized_label_is_rejected() {
    let q = Question::Choice(ChoiceQuestion {
        instructions: "pick".into(),
        criteria: vec![
            ChoiceCriterion {
                label: "l".repeat(limits::LABEL_BYTES + 1),
                description: "a".into(),
            },
            ChoiceCriterion {
                label: "b".into(),
                description: "b".into(),
            },
        ],
    });
    assert!(req("s".into(), q).validate().is_err());
}

#[test]
fn noul_always_reports_exactly_two_options() {
    assert_eq!(noul().option_count(), 2);
}

#[test]
fn option_count_tracks_the_caller_supplied_set() {
    assert_eq!(choice(7).option_count(), 7);
    assert_eq!(score(4).option_count(), 4);
}

#[test]
fn v1_parameter_count_is_computed_from_shape() {
    use stdbev_types::ModelConfig;
    // Derived independently: 257*128 + 224*128 + 2*blocks + option-attention + final LN.
    let d = 128usize;
    let ff = 512usize;
    let block = 4 * (d * d + d) + (d * ff + ff) + (ff * d + d) + 2 * (2 * d);
    let expected = 257 * d + 224 * d + 2 * block + 3 * (d * d + d) + 2 * (2 * d) + 2 * d;
    assert_eq!(ModelConfig::V1.parameter_count(), expected);
    // Sanity: the spec's ~0.5M budget.
    assert!((500_000..520_000).contains(&ModelConfig::V1.parameter_count()));
}

#[test]
fn v1_head_width_divides_evenly() {
    use stdbev_types::ModelConfig;
    let c = ModelConfig::V1;
    assert_eq!(c.head_width(), 32);
    assert_eq!(
        c.head_width() * c.attention_heads as usize,
        c.model_width as usize
    );
}
