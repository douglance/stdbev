//! Shared decision fixtures.
//!
//! Defined once so the native path and the SpacetimeDB path are provably answering
//! the *same* question. If each side kept its own copy, the parity test would be
//! comparing two implementations of the fixture as much as two implementations of the
//! model.

use stdbev_types::{
    ChoiceCriterion, ChoiceQuestion, DecisionRequest, NoulQuestion, Question, ScoreLevel,
    ScoreQuestion,
};

/// A fixture plus the id its result is stored under.
pub struct Fixture {
    pub id: &'static str,
    pub request: DecisionRequest,
}

fn pairs(items: &[(&str, &str)]) -> Vec<(String, String)> {
    items
        .iter()
        .map(|(a, b)| ((*a).to_owned(), (*b).to_owned()))
        .collect()
}

/// Labels and descriptions, in the shape the reducers take them.
#[must_use]
pub fn choice_options() -> (Vec<String>, Vec<String>) {
    pairs(&[
        ("billing", "Payments, charges, or invoices"),
        ("technical", "Software or hardware problem"),
        ("other", "Everything else"),
    ])
    .into_iter()
    .unzip()
}

/// Score ladder, in the shape the reducers take it.
#[must_use]
pub fn score_levels() -> (Vec<String>, Vec<String>) {
    pairs(&[
        ("low", "Minor degradation"),
        ("medium", "Material degradation"),
        ("high", "Major outage"),
        ("critical", "System-wide emergency"),
    ])
    .into_iter()
    .unzip()
}

pub const CHOICE_STATE: &str = "Customer says the app crashes after login.";
pub const CHOICE_INSTRUCTIONS: &str = "What type of request is this?";
pub const NOUL_STATE: &str = "status=failed retries=4";
pub const NOUL_INSTRUCTIONS: &str = "Should this be retried?";
pub const NOUL_FALSE: &str = "Do not retry";
pub const NOUL_TRUE: &str = "Retry the operation";
pub const SCORE_STATE: &str = "error_rate=0.21 affected_users=8400";
pub const SCORE_INSTRUCTIONS: &str = "How severe is this incident?";

/// Every fixture the parity test runs, in a fixed order.
#[must_use]
pub fn all() -> Vec<Fixture> {
    let (choice_labels, choice_descriptions) = choice_options();
    let (score_labels, score_descriptions) = score_levels();
    vec![
        Fixture {
            id: "c1",
            request: DecisionRequest {
                state: CHOICE_STATE.to_owned(),
                question: Question::Choice(ChoiceQuestion {
                    instructions: CHOICE_INSTRUCTIONS.to_owned(),
                    criteria: choice_labels
                        .into_iter()
                        .zip(choice_descriptions)
                        .map(|(label, description)| ChoiceCriterion { label, description })
                        .collect(),
                }),
            },
        },
        Fixture {
            id: "n1",
            request: DecisionRequest {
                state: NOUL_STATE.to_owned(),
                question: Question::Noul(NoulQuestion {
                    instructions: NOUL_INSTRUCTIONS.to_owned(),
                    false_description: NOUL_FALSE.to_owned(),
                    true_description: NOUL_TRUE.to_owned(),
                }),
            },
        },
        Fixture {
            id: "s1",
            request: DecisionRequest {
                state: SCORE_STATE.to_owned(),
                question: Question::Score(ScoreQuestion {
                    instructions: SCORE_INSTRUCTIONS.to_owned(),
                    levels: score_labels
                        .into_iter()
                        .zip(score_descriptions)
                        .map(|(label, description)| ScoreLevel { label, description })
                        .collect(),
                }),
            },
        },
    ]
}
