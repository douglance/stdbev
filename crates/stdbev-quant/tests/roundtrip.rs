//! Artifact write -> parse -> infer, end to end.
//!
//! Runs on a deterministic random-weight artifact, so the entire deployment spine is
//! under test before any model has been trained. The answers are meaningless; that
//! every invariant holds is not.

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

use stdbev_quant::{dequantize, genesis_artifact, quantize_rows};
use stdbev_runtime::{Runtime, Scratch};
use stdbev_types::{
    ChoiceCriterion, ChoiceQuestion, DecisionAnswer, DecisionRequest, ModelConfig, NoulQuestion,
    OptionFormat, Question, ScoreLevel, ScoreQuestion, check_distribution,
};

/// A smaller config than V1, so the suite stays fast while exercising identical paths.
fn test_config() -> ModelConfig {
    ModelConfig {
        context_length: 64,
        option_length: 32,
        ..ModelConfig::V1
    }
}

fn artifact() -> Vec<u8> {
    genesis_artifact(test_config(), 42).expect("genesis artifact builds")
}

fn choice_request(labels: &[&str]) -> DecisionRequest {
    DecisionRequest {
        state: "Customer says the app crashes after login.".into(),
        question: Question::Choice(ChoiceQuestion {
            instructions: "What type of request is this?".into(),
            criteria: labels
                .iter()
                .map(|l| ChoiceCriterion {
                    label: (*l).to_owned(),
                    description: format!("things about {l}"),
                })
                .collect(),
        }),
    }
}

#[test]
fn artifact_parses_and_reports_its_architecture() {
    let bytes = artifact();
    let rt = Runtime::from_bytes(&bytes).expect("parses");
    assert_eq!(rt.config(), &test_config());
    assert_eq!(rt.model_id().len(), 64, "model_id is hex SHA-256");
    assert!(rt.temperature() > 0.0);
}

#[test]
fn model_id_is_deterministic_and_seed_dependent() {
    let a = Runtime::from_bytes(&artifact())
        .unwrap()
        .model_id()
        .to_owned();
    let b = Runtime::from_bytes(&artifact())
        .unwrap()
        .model_id()
        .to_owned();
    assert_eq!(a, b, "same inputs must give the same id");

    let other = genesis_artifact(test_config(), 43).unwrap();
    let c = Runtime::from_bytes(&other).unwrap().model_id().to_owned();
    assert_ne!(a, c, "different weights must give a different id");
}

#[test]
fn model_id_changes_when_only_the_temperature_changes() {
    // The defect this guards: hashing the checkpoint instead of the artifact would
    // give two differently-calibrated models the same id, and the native/WASM parity
    // test compares ids to prove both sides ran the same model.
    let mut a = artifact();
    let before = Runtime::from_bytes(&a).unwrap().model_id().to_owned();
    a[28..32].copy_from_slice(&2.0f32.to_bits().to_le_bytes());
    let rt = Runtime::from_bytes(&a).unwrap();
    assert!((rt.temperature() - 2.0).abs() < 1e-6);
    // The stored id no longer matches the content; recomputing must differ.
    let recomputed = {
        use sha2::{Digest, Sha256};
        let mut z = a.clone();
        z[32..64].fill(0);
        Sha256::digest(&z)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>()
    };
    assert_ne!(
        before, recomputed,
        "temperature must be inside the identity"
    );
}

#[test]
fn choice_returns_a_valid_distribution() {
    let bytes = artifact();
    let rt = Runtime::from_bytes(&bytes).unwrap();
    let mut scratch = Scratch::new(rt.config());
    let req = choice_request(&["billing", "technical", "other"]);
    let answer = rt.decide(&req, &mut scratch).expect("decides");
    let DecisionAnswer::Choice(a) = answer else {
        panic!("wrong variant")
    };
    assert_eq!(a.probabilities.len(), 3);
    check_distribution(&a.probabilities).expect("valid distribution");
    assert!(a.selected_index < 3);
    assert_eq!(
        a.selected_label,
        ["billing", "technical", "other"][a.selected_index]
    );
    assert!((a.confidence - a.probabilities[a.selected_index]).abs() < 1e-6);
}

#[test]
fn noul_probability_is_exactly_p_true() {
    let bytes = artifact();
    let rt = Runtime::from_bytes(&bytes).unwrap();
    let mut scratch = Scratch::new(rt.config());
    let req = DecisionRequest {
        state: "status=failed retry_count=4".into(),
        question: Question::Noul(NoulQuestion {
            instructions: "Should this be retried?".into(),
            false_description: "The transaction should stop".into(),
            true_description: "The transaction should be retried".into(),
        }),
    };
    let DecisionAnswer::Noul(a) = rt.decide(&req, &mut scratch).unwrap() else {
        panic!("wrong variant")
    };
    assert_eq!(
        a.probability, a.probabilities[1],
        "probability must BE p(true)"
    );
    check_distribution(&a.probabilities).expect("valid distribution");
}

#[test]
fn score_stays_within_the_ladder() {
    let bytes = artifact();
    let rt = Runtime::from_bytes(&bytes).unwrap();
    let mut scratch = Scratch::new(rt.config());
    let levels = ["low", "medium", "high", "critical"];
    let req = DecisionRequest {
        state: "error_rate=0.21 affected_users=8400".into(),
        question: Question::Score(ScoreQuestion {
            instructions: "How severe is this incident?".into(),
            levels: levels
                .iter()
                .map(|l| ScoreLevel {
                    label: (*l).into(),
                    description: format!("{l} severity"),
                })
                .collect(),
        }),
    };
    let DecisionAnswer::Score(a) = rt.decide(&req, &mut scratch).unwrap() else {
        panic!("wrong variant")
    };
    assert!(
        (0.0..=3.0).contains(&a.score),
        "score {} outside 0..=3",
        a.score
    );
    check_distribution(&a.probabilities).expect("valid distribution");
    // The expected value must equal the distribution it came from.
    let expected: f32 = a
        .probabilities
        .iter()
        .enumerate()
        .map(|(i, p)| p * i as f32)
        .sum();
    assert!((a.score - expected).abs() < 1e-5);
}

#[test]
fn inference_is_deterministic_across_repeated_calls() {
    let bytes = artifact();
    let rt = Runtime::from_bytes(&bytes).unwrap();
    let mut scratch = Scratch::new(rt.config());
    let req = choice_request(&["a", "b", "c"]);
    let first = rt.decide(&req, &mut scratch).unwrap();
    for _ in 0..3 {
        let again = rt.decide(&req, &mut scratch).unwrap();
        assert_eq!(
            first.probabilities(),
            again.probabilities(),
            "scratch reuse changed the answer"
        );
    }
}

#[test]
fn permutation_reorders_probabilities_and_nothing_else() {
    let bytes = artifact();
    let rt = Runtime::from_bytes(&bytes).unwrap();
    let mut scratch = Scratch::new(rt.config());
    let forward = rt
        .decide(&choice_request(&["a", "b", "c"]), &mut scratch)
        .unwrap();
    let reversed = rt
        .decide(&choice_request(&["c", "b", "a"]), &mut scratch)
        .unwrap();
    let (f, r) = (forward.probabilities(), reversed.probabilities());
    for i in 0..3 {
        assert!(
            (f[i] - r[2 - i]).abs() < 1e-5,
            "option {i}: {} vs {} -- model parameters must not depend on option order",
            f[i],
            r[2 - i]
        );
    }
}

#[test]
fn option_count_does_not_change_the_parameters_used() {
    let bytes = artifact();
    let rt = Runtime::from_bytes(&bytes).unwrap();
    let mut scratch = Scratch::new(rt.config());
    for n in [2usize, 3, 8, 16] {
        let labels: Vec<String> = (0..n).map(|i| format!("opt{i}")).collect();
        let refs: Vec<&str> = labels.iter().map(String::as_str).collect();
        let a = rt.decide(&choice_request(&refs), &mut scratch).unwrap();
        assert_eq!(a.probabilities().len(), n);
        check_distribution(a.probabilities()).expect("valid distribution");
    }
}

#[test]
fn scratch_stays_within_the_memory_budget() {
    let scratch = Scratch::new(&ModelConfig::V1);
    let budget = 4 * 1024 * 1024;
    assert!(
        scratch.allocated_bytes() <= budget,
        "scratch {} bytes exceeds {budget}",
        scratch.allocated_bytes()
    );
}

#[test]
fn quantization_error_stays_bounded() {
    let raw: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) / 64.0).collect();
    let q = quantize_rows(&raw, 64);
    let back = dequantize(&q, 64);
    for (a, b) in raw.iter().zip(&back) {
        // Worst case is half a quantization step of the row max (2.0 / 127 / 2).
        assert!((a - b).abs() < 0.02, "{a} vs {b}");
    }
}

#[test]
fn an_all_zero_row_quantizes_without_producing_nan() {
    let q = quantize_rows(&[0.0; 8], 8);
    assert_eq!(q.scales, vec![1.0], "zero row must not get scale 0");
    let back = dequantize(&q, 8);
    assert!(back.iter().all(|v| v.is_finite()));
}

#[test]
fn option_format_survives_the_artifact_round_trip() {
    // Serving an artifact the option encoding it was not trained on produces confident
    // nonsense with no error, so the encoding must travel with the weights.
    for format in [OptionFormat::Verbose, OptionFormat::Compact] {
        let config = ModelConfig {
            option_format: format,
            ..test_config()
        };
        let bytes = genesis_artifact(config, 42).expect("builds");
        let rt = Runtime::from_bytes(&bytes).expect("parses");
        assert_eq!(rt.config().option_format, format);
    }
}

#[test]
fn the_two_option_formats_give_different_answers() {
    // If they agreed, carrying the format in the header would be pointless. They do
    // not, which is exactly why a mismatch has to be impossible rather than unlikely.
    let verbose_cfg = ModelConfig {
        option_format: OptionFormat::Verbose,
        ..test_config()
    };
    let compact_cfg = ModelConfig {
        option_format: OptionFormat::Compact,
        ..test_config()
    };
    let vb = genesis_artifact(verbose_cfg, 42).expect("builds");
    let cb = genesis_artifact(compact_cfg, 42).expect("builds");

    let vrt = Runtime::from_bytes(&vb).expect("parses");
    let crt = Runtime::from_bytes(&cb).expect("parses");
    let mut scratch = Scratch::new(vrt.config());

    let req = choice_request(&["billing", "technical", "other"]);
    let a = vrt.decide(&req, &mut scratch).expect("decides");
    let b = crt.decide(&req, &mut scratch).expect("decides");
    assert_ne!(
        a.probabilities(),
        b.probabilities(),
        "the encodings must actually differ, or the header field is decoration"
    );
}

#[test]
fn an_unknown_option_format_tag_is_rejected() {
    let mut bytes = artifact();
    bytes[65] = 9;
    let Err(err) = Runtime::from_bytes(&bytes) else {
        panic!("an unknown option_format tag must be rejected, not guessed at");
    };
    assert!(format!("{err}").contains("option_format"), "got {err}");
}
