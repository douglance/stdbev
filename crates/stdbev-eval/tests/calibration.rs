//! Calibration and metric tests.
//!
//! Expectations are derived by hand or from a property that must hold regardless of
//! the implementation, never from running the code and pasting the result back.

#![allow(
    clippy::indexing_slicing,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp,
    clippy::cast_precision_loss
)]

use stdbev_eval::calibrate::{Logits, fit, nll_at};
use stdbev_eval::metrics::{Scored, evaluate, expected_calibration_error};

fn logits(values: &[f32], label: usize) -> Logits {
    Logits {
        values: values.to_vec(),
        label,
    }
}

#[test]
fn temperature_of_one_is_optimal_for_already_calibrated_logits() {
    // Two options, logit gap chosen so the true class gets exactly p = 0.75 at T = 1,
    // and it is right 75% of the time. That is calibrated by construction.
    let gap = (0.75f32 / 0.25).ln();
    let mut examples = Vec::new();
    for i in 0..100 {
        examples.push(logits(&[0.0, gap], usize::from(i % 4 != 0)));
    }
    let t = fit(&examples);
    assert!((t - 1.0).abs() < 0.15, "expected T near 1.0, got {t}");
}

#[test]
fn an_overconfident_model_gets_a_temperature_above_one() {
    // Huge logit gaps but the label is right only half the time: the model is far too
    // sure of itself, so calibration must soften it.
    let mut examples = Vec::new();
    for i in 0..100 {
        examples.push(logits(&[0.0, 12.0], usize::from(i % 2 == 0)));
    }
    let t = fit(&examples);
    assert!(t > 1.5, "overconfident model should need T > 1.5, got {t}");
}

#[test]
fn an_underconfident_model_gets_a_temperature_below_one() {
    // Tiny gaps but always correct: the model knows more than it is admitting.
    let mut examples = Vec::new();
    for _ in 0..100 {
        examples.push(logits(&[0.0, 0.3], 1));
    }
    let t = fit(&examples);
    assert!(t < 1.0, "underconfident model should need T < 1.0, got {t}");
}

#[test]
fn calibration_never_changes_which_option_wins() {
    // The defining property of temperature scaling: accuracy is invariant.
    let values = [0.2f32, 1.7, -0.4, 0.9];
    let best = 1usize;
    for t in [0.05f32, 0.5, 1.0, 2.0, 10.0] {
        let mut scaled: Vec<f32> = values.iter().map(|v| v / t).collect();
        stdbev_math::softmax_inplace(&mut scaled);
        let winner = stdbev_math::argmax(&scaled).0;
        assert_eq!(winner, best, "temperature {t} changed the winner");
    }
}

#[test]
fn the_fitted_temperature_beats_its_neighbours() {
    // Mildly overconfident: p(correct) = sigmoid(2) = 0.88 at T = 1, but right 85% of
    // the time. The optimum is just above 1 and safely inside the search bracket.
    let mut examples = Vec::new();
    for i in 0..200 {
        examples.push(logits(&[0.0, 2.0], usize::from(i % 20 >= 3)));
    }
    let t = fit(&examples);
    let best = nll_at(&examples, t);
    assert!(t > 0.1 && t < 9.0, "optimum should be interior, got {t}");
    for delta in [-0.3f32, -0.1, 0.1, 0.3] {
        let other = nll_at(&examples, (t + delta).max(0.05));
        assert!(
            best <= other + 1e-4,
            "T={t} (nll {best}) lost to T={} (nll {other})",
            t + delta
        );
    }
}

#[test]
fn a_hopeless_model_clamps_to_the_search_ceiling() {
    // Confident logits, right only a third of the time: the true optimum is T -> inf.
    // Clamping is correct -- chasing it would produce a temperature that makes every
    // probability 1/K, which is not calibration, it is surrender. The caller should
    // read a bound-hugging temperature as "this model is broken", so the behavior is
    // pinned here rather than left to be rediscovered.
    let mut examples = Vec::new();
    for i in 0..60 {
        examples.push(logits(&[0.0, 5.0, 1.0], i % 3));
    }
    let t = fit(&examples);
    assert!(t > 9.9, "expected the ceiling, got {t}");
}

#[test]
fn a_perfectly_calibrated_set_has_near_zero_ece() {
    // 80 examples at confidence 0.8, correct exactly 80% of the time.
    let scored: Vec<Scored> = (0..100)
        .map(|i| Scored {
            probabilities: vec![0.2, 0.8],
            label: usize::from(i % 5 != 0),
        })
        .collect();
    let ece = expected_calibration_error(&scored);
    assert!(ece < 0.02, "expected near-zero ECE, got {ece}");
}

#[test]
fn a_confidently_wrong_set_has_large_ece() {
    let scored: Vec<Scored> = (0..50)
        .map(|_| Scored {
            probabilities: vec![0.02, 0.98],
            label: 0,
        })
        .collect();
    let ece = expected_calibration_error(&scored);
    assert!(
        ece > 0.9,
        "confidently wrong should give ECE near 1, got {ece}"
    );
}

#[test]
fn confidence_of_exactly_one_is_counted_not_dropped() {
    // The reference implementation's half-open top bin silently discards these. A
    // quantized, confident model produces them in quantity, so dropping them would
    // flatter the calibration number precisely where it matters most.
    let scored: Vec<Scored> = (0..20)
        .map(|_| Scored {
            probabilities: vec![0.0, 1.0],
            label: 0,
        })
        .collect();
    let ece = expected_calibration_error(&scored);
    assert!(
        ece > 0.99,
        "confidence 1.0 and always wrong must give ECE ~1, got {ece}"
    );
}

#[test]
fn metrics_agree_with_hand_counted_values() {
    let scored = vec![
        Scored {
            probabilities: vec![0.7, 0.2, 0.1],
            label: 0,
        }, // correct
        Scored {
            probabilities: vec![0.1, 0.8, 0.1],
            label: 1,
        }, // correct
        Scored {
            probabilities: vec![0.6, 0.3, 0.1],
            label: 2,
        }, // wrong, rank 2
    ];
    let report = evaluate(&scored);
    assert_eq!(report.example_count, 3);
    assert!((report.top_1_accuracy - 2.0 / 3.0).abs() < 1e-6);
    // All three labels sit within the top 3 of a 3-option problem.
    assert!((report.top_3_accuracy - 1.0).abs() < 1e-6);
    assert!(
        (report.mean_confidence - 0.7).abs() < 1e-6,
        "got {}",
        report.mean_confidence
    );
}
