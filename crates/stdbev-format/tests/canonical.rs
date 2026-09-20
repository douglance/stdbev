//! Canonical-encoding contract tests.
//!
//! These assert exact bytes on purpose. A change here invalidates every trained
//! checkpoint, so it should require deliberately editing a test, not just editing code.

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

use stdbev_format::{PAD, context_text, encode, option_texts, token_id};
use stdbev_types::{
    ChoiceCriterion, ChoiceQuestion, DecisionRequest, NoulQuestion, OptionFormat, Question,
    ScoreLevel, ScoreQuestion,
};

fn request(state: &str, question: Question) -> DecisionRequest {
    DecisionRequest {
        state: state.to_owned(),
        question,
    }
}

#[test]
fn context_leads_with_state_then_instructions() {
    let r = request(
        "status=failed",
        Question::Noul(NoulQuestion {
            instructions: "Retry?".into(),
            false_description: "stop".into(),
            true_description: "retry".into(),
        }),
    );
    assert_eq!(context_text(&r), "STATE\nstatus=failed\nQUESTION\nRetry?");
}

#[test]
fn noul_expands_to_false_then_true() {
    let q = Question::Noul(NoulQuestion {
        instructions: "Retry?".into(),
        false_description: "stop".into(),
        true_description: "go".into(),
    });
    assert_eq!(
        option_texts(&q, OptionFormat::Verbose),
        vec!["FALSE\nstop", "TRUE\ngo"]
    );
    // Compact drops the near-identical descriptions for bare polarity markers: a
    // condition and its negation share 0.636 of their bytes, which mean pooling over
    // byte embeddings cannot separate.
    assert_eq!(option_texts(&q, OptionFormat::Compact), vec!["no", "yes"]);
}

#[test]
fn score_levels_carry_their_ordinal() {
    let q = Question::Score(ScoreQuestion {
        instructions: "Severity?".into(),
        levels: vec![
            ScoreLevel {
                label: "low".into(),
                description: "minor".into(),
            },
            ScoreLevel {
                label: "high".into(),
                description: "major".into(),
            },
        ],
    });
    let opts = option_texts(&q, OptionFormat::Verbose);
    assert_eq!(opts[0], "LEVEL\n0\nLABEL\nlow\nDESCRIPTION\nminor");
    assert_eq!(opts[1], "LEVEL\n1\nLABEL\nhigh\nDESCRIPTION\nmajor");
    let compact = option_texts(&q, OptionFormat::Compact);
    assert_eq!(compact, vec!["low", "high"]);
}

#[test]
fn choice_preserves_caller_order() {
    let q = Question::Choice(ChoiceQuestion {
        instructions: "Which?".into(),
        criteria: vec![
            ChoiceCriterion {
                label: "b".into(),
                description: "second".into(),
            },
            ChoiceCriterion {
                label: "a".into(),
                description: "first".into(),
            },
        ],
    });
    let opts = option_texts(&q, OptionFormat::Verbose);
    assert!(opts[0].starts_with("LABEL\nb"));
    assert!(opts[1].starts_with("LABEL\na"));
    let compact = option_texts(&q, OptionFormat::Compact);
    assert_eq!(compact, vec!["b", "a"]);
}

#[test]
fn token_ids_never_collide_with_pad() {
    for b in 0u8..=255 {
        assert_ne!(token_id(b), PAD);
    }
    assert_eq!(token_id(0), 1);
    assert_eq!(token_id(255), 256);
}

#[test]
fn encode_pads_short_input_to_the_window() {
    let t = encode("hi", 8);
    assert_eq!(
        t.ids,
        vec![token_id(b'h'), token_id(b'i'), 0, 0, 0, 0, 0, 0]
    );
    assert_eq!(
        t.mask,
        vec![true, true, false, false, false, false, false, false]
    );
    assert_eq!(t.valid_len(), 2);
}

#[test]
fn encode_truncates_long_input_to_exactly_the_window() {
    let t = encode(&"x".repeat(1000), 224);
    assert_eq!(t.ids.len(), 224);
    assert_eq!(t.valid_len(), 224);
    assert!(t.mask.iter().all(|m| *m));
}

#[test]
fn encode_is_byte_oriented_not_character_oriented() {
    // 'é' is two UTF-8 bytes. A 1-byte window keeps the first of them: harmless,
    // because the model consumes bytes rather than characters.
    let t = encode("é", 1);
    assert_eq!(t.ids.len(), 1);
    assert_eq!(t.valid_len(), 1);
    assert_eq!(t.ids[0], token_id("é".as_bytes()[0]));
}

#[test]
fn empty_input_is_all_padding() {
    let t = encode("", 4);
    assert_eq!(t.valid_len(), 0);
    assert!(t.mask.iter().all(|m| !*m));
}

#[test]
fn compact_drops_descriptions_entirely() {
    // jev's own API takes `criteria: {label: null}`, so an absent description must not
    // leave a dangling ": " for the model to read.
    let q = Question::Choice(ChoiceQuestion {
        instructions: "Which?".into(),
        criteria: vec![
            ChoiceCriterion {
                label: "billing".into(),
                description: String::new(),
            },
            ChoiceCriterion {
                label: "technical".into(),
                description: String::new(),
            },
        ],
    });
    assert_eq!(
        option_texts(&q, OptionFormat::Compact),
        vec!["billing", "technical"]
    );
}

#[test]
fn compact_options_share_far_fewer_bytes_than_verbose_ones() {
    // This is the property that matters for a pooled option encoder: it can only
    // separate options that differ in their bytes.
    fn overlap(a: &str, b: &str) -> f64 {
        let (mut shared, mut total) = (0usize, 0usize);
        for byte in 0..=255u8 {
            let ca = a.bytes().filter(|x| *x == byte).count();
            let cb = b.bytes().filter(|x| *x == byte).count();
            shared += ca.min(cb);
            total += ca.max(cb);
        }
        shared as f64 / total.max(1) as f64
    }
    let q = Question::Noul(NoulQuestion {
        instructions: "Suspended?".into(),
        false_description: "The account is not suspended".into(),
        true_description: "The account is suspended".into(),
    });
    let verbose = option_texts(&q, OptionFormat::Verbose);
    let compact = option_texts(&q, OptionFormat::Compact);
    let v = overlap(&verbose[0], &verbose[1]);
    let c = overlap(&compact[0], &compact[1]);
    assert!(
        v > 0.6,
        "verbose Noul options should be near-identical, got {v:.3}"
    );
    assert!(
        c < 0.2,
        "compact Noul options should be distinct, got {c:.3}"
    );
}
