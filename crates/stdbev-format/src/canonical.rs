//! Canonical textual encoding of a typed question.
//!
//! Every question becomes a context string plus a list of option strings. The exact
//! byte layout here is part of the model contract: changing a separator changes what
//! the model sees and silently invalidates every trained checkpoint. Treat these
//! formats as frozen.

use stdbev_types::{DecisionRequest, OptionFormat, Question};

/// Builds the context the options will query.
///
/// State leads, because [`crate::encode`] truncates from the end.
#[must_use]
pub fn context_text(request: &DecisionRequest) -> String {
    format!(
        "STATE\n{}\nQUESTION\n{}",
        request.state,
        request.question.instructions()
    )
}

/// Builds one string per option, in the order the caller supplied them.
///
/// Noul expands to exactly two options, FALSE then TRUE, which is why
/// `NoulAnswer::probability` is `probabilities[1]`.
#[must_use]
pub fn option_texts(question: &Question, format: OptionFormat) -> Vec<String> {
    match format {
        OptionFormat::Verbose => verbose(question),
        OptionFormat::Compact => compact(question),
    }
}

/// The compact encoding: the label alone.
///
/// Descriptions are dropped because mean pooling cannot preserve them -- see
/// [`stdbev_types::OptionFormat::Compact`] for the measurements. They still reach the
/// model through the question's instructions, which are part of the context.
fn compact(question: &Question) -> Vec<String> {
    match question {
        Question::Choice(q) => q.criteria.iter().map(|c| c.label.clone()).collect(),
        // Bare polarity markers. jev's own Noul takes only a question and returns a
        // probability; it has no option descriptions. Encoding a condition and its
        // negation as two options makes them differ by one word, which mean pooling
        // cannot see. The question already reaches the model through the context.
        Question::Noul(_) => vec!["no".to_owned(), "yes".to_owned()],
        Question::Score(q) => q.levels.iter().map(|l| l.label.clone()).collect(),
    }
}

/// The spec's section 8 encoding, kept so existing artifacts stay servable.
fn verbose(question: &Question) -> Vec<String> {
    match question {
        Question::Choice(q) => q
            .criteria
            .iter()
            .map(|c| format!("LABEL\n{}\nDESCRIPTION\n{}", c.label, c.description))
            .collect(),
        Question::Noul(q) => vec![
            format!("FALSE\n{}", q.false_description),
            format!("TRUE\n{}", q.true_description),
        ],
        Question::Score(q) => q
            .levels
            .iter()
            .enumerate()
            .map(|(i, l)| {
                format!(
                    "LEVEL\n{}\nLABEL\n{}\nDESCRIPTION\n{}",
                    i, l.label, l.description
                )
            })
            .collect(),
    }
}
