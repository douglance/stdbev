//! Numerical kernel tests.
//!
//! Expectations are hand-computed or derived independently of the implementation --
//! a test whose expected and actual sides come from the same code only proves the
//! code agrees with itself.

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

use proptest::prelude::*;
use stdbev_math::{
    dot, gelu, layer_norm, masked_mean_pool, masked_softmax_inplace, matvec_i8, softmax_inplace,
};

const EPS: f32 = 1e-5;

#[test]
fn softmax_matches_hand_computed_values() {
    // Uniform input must give a uniform distribution.
    let mut v = vec![2.0, 2.0, 2.0, 2.0];
    softmax_inplace(&mut v);
    for p in &v {
        assert!((p - 0.25).abs() < EPS, "got {p}");
    }
    // e^0 : e^1 = 1 : e, so p0 = 1/(1+e) = 0.26894142
    let mut v = vec![0.0, 1.0];
    softmax_inplace(&mut v);
    assert!((v[0] - 0.268_941_42).abs() < EPS, "got {}", v[0]);
    assert!((v[1] - 0.731_058_6).abs() < EPS, "got {}", v[1]);
}

#[test]
fn softmax_is_stable_against_huge_logits() {
    // Naive exp() would overflow to inf here and yield NaN after division.
    let mut v = vec![1e30, 1e30 + 1.0, 0.0];
    softmax_inplace(&mut v);
    assert!(v.iter().all(|p| p.is_finite()), "got {v:?}");
    assert!((v.iter().sum::<f32>() - 1.0).abs() < EPS);
}

#[test]
fn softmax_of_all_masked_input_is_uniform_not_nan() {
    let mut v = vec![f32::NEG_INFINITY; 3];
    softmax_inplace(&mut v);
    assert!(v.iter().all(|p| p.is_finite()), "got {v:?}");
    assert!((v.iter().sum::<f32>() - 1.0).abs() < EPS);
}

#[test]
fn masked_softmax_gives_zero_probability_to_masked_positions() {
    let mut s = vec![1.0, 5.0, 1.0];
    masked_softmax_inplace(&mut s, &[true, false, true]);
    assert_eq!(s[1], 0.0, "masked position kept mass");
    assert!((s[0] - 0.5).abs() < EPS);
    assert!((s.iter().sum::<f32>() - 1.0).abs() < EPS);
}

#[test]
fn layer_norm_produces_zero_mean_unit_variance() {
    let input = [1.0f32, 2.0, 3.0, 4.0];
    let mut out = [0.0f32; 4];
    layer_norm(&mut out, &input, &[1.0; 4], &[0.0; 4], 1e-5);
    let mean = out.iter().sum::<f32>() / 4.0;
    let var = out.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / 4.0;
    assert!(mean.abs() < 1e-4, "mean {mean}");
    assert!((var - 1.0).abs() < 1e-3, "var {var}");
}

#[test]
fn layer_norm_applies_gamma_and_beta() {
    let input = [1.0f32, 2.0, 3.0, 4.0];
    let mut plain = [0.0f32; 4];
    let mut scaled = [0.0f32; 4];
    layer_norm(&mut plain, &input, &[1.0; 4], &[0.0; 4], 1e-5);
    layer_norm(&mut scaled, &input, &[2.0; 4], &[3.0; 4], 1e-5);
    for (p, s) in plain.iter().zip(&scaled) {
        assert!((p * 2.0 + 3.0 - s).abs() < EPS);
    }
}

#[test]
fn gelu_matches_known_points() {
    assert!(gelu(0.0).abs() < EPS);
    // GELU is asymptotically identity for large positive input and 0 for large negative.
    assert!((gelu(10.0) - 10.0).abs() < 1e-3, "got {}", gelu(10.0));
    assert!(gelu(-10.0).abs() < 1e-3, "got {}", gelu(-10.0));
    // Published tanh-approximation value at x = 1.
    assert!((gelu(1.0) - 0.841_191_5).abs() < 1e-4, "got {}", gelu(1.0));
}

#[test]
fn mean_pool_ignores_masked_rows() {
    // Row 1 is masked, so the mean must be of rows 0 and 2 only: (1+5)/2, (2+6)/2.
    let tokens = [1.0f32, 2.0, 99.0, 99.0, 5.0, 6.0];
    let mut out = [0.0f32; 2];
    masked_mean_pool(&mut out, &tokens, &[true, false, true], 2);
    assert!((out[0] - 3.0).abs() < EPS, "got {out:?}");
    assert!((out[1] - 4.0).abs() < EPS, "got {out:?}");
}

#[test]
fn mean_pool_of_nothing_is_zero_not_nan() {
    let mut out = [9.0f32; 2];
    masked_mean_pool(&mut out, &[1.0, 2.0], &[false], 2);
    assert_eq!(out, [0.0, 0.0]);
}

#[test]
fn matvec_i8_matches_an_independently_computed_result() {
    // 2 outputs x 3 inputs. Expected values computed by hand, not by the kernel.
    // row0 = [1,2,3] scale 0.5 -> (1*1 + 1*2 + 1*3) * 0.5 = 3.0, + bias 1.0 = 4.0
    // row1 = [-1,0,4] scale 2.0 -> (-1 + 0 + 4) * 2.0 = 6.0, + bias -1.0 = 5.0
    let mut out = [0.0f32; 2];
    matvec_i8(
        &mut out,
        &[1.0, 1.0, 1.0],
        &[1, 2, 3, -1, 0, 4],
        &[0.5, 2.0],
        Some(&[1.0, -1.0]),
        3,
    );
    assert!((out[0] - 4.0).abs() < EPS, "got {out:?}");
    assert!((out[1] - 5.0).abs() < EPS, "got {out:?}");
}

#[test]
fn matvec_i8_handles_rows_longer_than_the_lane_count() {
    // 20 columns exercises both the 8-wide chunks and the 4-element tail.
    let cols = 20;
    let weights = vec![2i8; cols];
    let input: Vec<f32> = (0..cols).map(|i| i as f32).collect();
    let mut out = [0.0f32];
    matvec_i8(&mut out, &input, &weights, &[1.0], None, cols);
    let expected: f32 = input.iter().map(|x| x * 2.0).sum();
    assert!(
        (out[0] - expected).abs() < 1e-3,
        "got {} want {expected}",
        out[0]
    );
}

proptest! {
    #[test]
    fn softmax_always_produces_a_valid_distribution(
        logits in proptest::collection::vec(-50.0f32..50.0, 1..32)
    ) {
        let mut v = logits;
        softmax_inplace(&mut v);
        prop_assert!(v.iter().all(|p| p.is_finite()));
        prop_assert!(v.iter().all(|p| (0.0..=1.0).contains(p)));
        prop_assert!((v.iter().sum::<f32>() - 1.0).abs() < EPS);
    }

    #[test]
    fn softmax_is_shift_invariant(
        logits in proptest::collection::vec(-20.0f32..20.0, 2..16),
        shift in -10.0f32..10.0,
    ) {
        let mut a = logits.clone();
        let mut b: Vec<f32> = logits.iter().map(|x| x + shift).collect();
        softmax_inplace(&mut a);
        softmax_inplace(&mut b);
        for (x, y) in a.iter().zip(&b) {
            prop_assert!((x - y).abs() < 1e-4, "{x} vs {y}");
        }
    }

    #[test]
    fn dot_is_commutative(
        a in proptest::collection::vec(-10.0f32..10.0, 1..24),
    ) {
        let b: Vec<f32> = a.iter().rev().copied().collect();
        prop_assert!((dot(&a, &b) - dot(&b, &a)).abs() < 1e-3);
    }
}
