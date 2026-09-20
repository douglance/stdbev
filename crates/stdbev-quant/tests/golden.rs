//! Golden numerical fixtures.
//!
//! Mutation testing proved the structural tests blind to two real defects: dropping
//! the second `sqrt(rank)` scaling in option attention, and omitting position
//! embeddings. Both leave every invariant intact -- the distribution still sums to 1,
//! permutation equivariance still holds, scores stay in range -- so only an assertion
//! on the actual numbers catches them.
//!
//! Pinned against `artifacts/genesis.stdbq`, the deterministic random-weight artifact,
//! not the deployable model. The genesis artifact is reproducible from a seed, so this
//! test measures the numerical path rather than whichever model happens to be shipped.
//!
//! A diff here means the numerical path changed. Confirm that was intended, then
//! regenerate with `cargo xtask golden`.

// Test-file lint posture: see the note in roundtrip.rs.
#![allow(
    clippy::indexing_slicing,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation
)]

use std::path::PathBuf;

use stdbev_runtime::{Runtime, Scratch};

/// Largest permitted departure from a committed golden probability.
///
/// Tight enough that a changed formula fails, loose enough to survive the last bit of
/// a JSON round-trip through f64.
const TOLERANCE: f32 = 1e-6;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root")
        .to_path_buf()
}

#[test]
fn committed_probabilities_still_reproduce() {
    let root = repo_root();
    let golden: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(root.join("tests/golden/probabilities.json"))
            .expect("golden file present; run `cargo xtask golden`"),
    )
    .expect("golden file is valid JSON");

    let bytes = std::fs::read(root.join("artifacts/genesis.stdbq")).expect("committed artifact");
    let runtime = Runtime::from_bytes(&bytes).expect("artifact parses");
    let mut scratch = Scratch::new(runtime.config());

    assert_eq!(
        golden["model_id"].as_str().unwrap(),
        runtime.model_id(),
        "golden file was generated from a different artifact"
    );

    for fixture in stdbev_fixtures::all() {
        let answer = runtime
            .decide(&fixture.request, &mut scratch)
            .expect("decides");
        let expected = &golden["fixtures"][fixture.id];
        let want: Vec<f32> = expected["probabilities"]
            .as_array()
            .expect("probabilities array")
            .iter()
            .map(|v| v.as_f64().expect("number") as f32)
            .collect();
        let got = answer.probabilities();

        assert_eq!(
            got.len(),
            want.len(),
            "{}: option count changed",
            fixture.id
        );
        assert_eq!(
            answer.selected_index(),
            expected["selected_index"].as_u64().expect("index") as usize,
            "{}: selection changed",
            fixture.id
        );
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(
                (g - w).abs() <= TOLERANCE,
                "{}: probability[{i}] drifted {w} -> {g}. If intended, \
                 re-run `cargo xtask golden`.",
                fixture.id
            );
        }
    }
}
