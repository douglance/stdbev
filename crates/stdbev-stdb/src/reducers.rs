//! The three public inference reducers.

use spacetimedb::{Identity, ReducerContext, Table, Timestamp};
use stdbev_runtime::{Runtime, Scratch};
use stdbev_types::{
    ChoiceCriterion, ChoiceQuestion, DecisionAnswer, DecisionRequest, NoulQuestion, Question,
    ScoreLevel, ScoreQuestion,
};

use crate::model::MODEL_BYTES;

/// Which primitive produced a row.
#[derive(spacetimedb::SpacetimeType, Clone, Copy, PartialEq, Eq, Debug)]
pub enum DecisionKind {
    Choice,
    Noul,
    Score,
}

/// One inference result.
///
/// Stores the output, never the submitted context: a decision log should not quietly
/// become a copy of whatever application state callers passed in.
#[spacetimedb::table(accessor = decision_result, public)]
pub struct DecisionResult {
    #[primary_key]
    pub request_id: String,
    pub caller: Identity,
    pub kind: DecisionKind,
    pub selected_index: u32,
    pub selected_label: String,
    /// Choice: confidence. Noul: P(TRUE). Score: expected level.
    pub scalar: f32,
    pub confidence: f32,
    pub probabilities: Vec<f32>,
    pub model_id: String,
    pub created_at: Timestamp,
}

fn pairs(
    labels: &[String],
    descriptions: &[String],
    min: usize,
    max: usize,
) -> Result<(), String> {
    if labels.len() != descriptions.len() {
        return Err(format!(
            "labels ({}) and descriptions ({}) must have equal length",
            labels.len(),
            descriptions.len()
        ));
    }
    if labels.len() < min || labels.len() > max {
        return Err(format!("expected {min}..={max} options, got {}", labels.len()));
    }
    Ok(())
}

/// Runs inference and records the result.
fn run(ctx: &ReducerContext, request_id: String, request: &DecisionRequest) -> Result<(), String> {
    let runtime = Runtime::from_bytes(MODEL_BYTES).map_err(|e| e.to_string())?;
    let mut scratch = Scratch::new(runtime.config());
    let answer = runtime.decide(request, &mut scratch).map_err(|e| e.to_string())?;

    let (kind, scalar, confidence, label) = match &answer {
        DecisionAnswer::Choice(a) => {
            (DecisionKind::Choice, a.confidence, a.confidence, a.selected_label.clone())
        }
        DecisionAnswer::Noul(a) => {
            let idx = usize::from(a.probability >= 0.5);
            let conf = a.probabilities[idx];
            (DecisionKind::Noul, a.probability, conf, ["FALSE", "TRUE"][idx].to_owned())
        }
        DecisionAnswer::Score(a) => {
            (DecisionKind::Score, a.score, a.confidence, a.selected_label.clone())
        }
    };

    // Upsert rather than insert: a repeated request_id is a caller mistake, and an
    // aborted transaction is a worse answer than an idempotent overwrite.
    ctx.db.decision_result().request_id().delete(&request_id);
    ctx.db.decision_result().insert(DecisionResult {
        request_id,
        caller: ctx.sender(),
        kind,
        selected_index: answer.selected_index() as u32,
        selected_label: label,
        scalar,
        confidence,
        probabilities: answer.probabilities().to_vec(),
        model_id: answer.model_id().to_owned(),
        created_at: ctx.timestamp,
    });
    Ok(())
}

/// Choose among 2..=16 named options.
#[spacetimedb::reducer]
pub fn decide_choice(
    ctx: &ReducerContext,
    request_id: String,
    state: String,
    instructions: String,
    labels: Vec<String>,
    descriptions: Vec<String>,
) -> Result<(), String> {
    pairs(&labels, &descriptions, 2, 16)?;
    let criteria = labels
        .into_iter()
        .zip(descriptions)
        .map(|(label, description)| ChoiceCriterion { label, description })
        .collect();
    let request = DecisionRequest {
        state,
        question: Question::Choice(ChoiceQuestion { instructions, criteria }),
    };
    run(ctx, request_id, &request)
}

/// Estimate the probability that a condition holds.
#[spacetimedb::reducer]
pub fn decide_noul(
    ctx: &ReducerContext,
    request_id: String,
    state: String,
    instructions: String,
    false_description: String,
    true_description: String,
) -> Result<(), String> {
    let request = DecisionRequest {
        state,
        question: Question::Noul(NoulQuestion {
            instructions,
            false_description,
            true_description,
        }),
    };
    run(ctx, request_id, &request)
}

/// Score against an ordered ladder of 2..=10 levels.
#[spacetimedb::reducer]
pub fn decide_score(
    ctx: &ReducerContext,
    request_id: String,
    state: String,
    instructions: String,
    labels: Vec<String>,
    descriptions: Vec<String>,
) -> Result<(), String> {
    pairs(&labels, &descriptions, 2, 10)?;
    let levels = labels
        .into_iter()
        .zip(descriptions)
        .map(|(label, description)| ScoreLevel { label, description })
        .collect();
    let request = DecisionRequest {
        state,
        question: Question::Score(ScoreQuestion { instructions, levels }),
    };
    run(ctx, request_id, &request)
}
