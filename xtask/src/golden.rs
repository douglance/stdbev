//! Regenerates the golden numerical fixtures.
//!
//! These exist because mutation testing proved the rest of the suite blind to two real
//! defects: dropping the second `sqrt(rank)` scaling in option attention, and omitting
//! position embeddings entirely. Both keep every structural invariant intact -- the
//! distribution still sums to 1, permutation equivariance still holds, scores stay in
//! range -- so only an assertion on the actual numbers can catch them.

use std::fs;
use std::path::PathBuf;

use stdbev_runtime::{Runtime, Scratch};

/// Location of the committed golden file.
///
/// # Errors
/// Returns a message if the workspace root cannot be resolved.
pub fn golden_path() -> Result<PathBuf, String> {
    Ok(std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .ok_or("no workspace root")?
        .join("tests")
        .join("golden")
        .join("probabilities.json"))
}

/// Computes the golden values for every fixture against the committed artifact.
///
/// # Errors
/// Propagates artifact and inference failures.
pub fn compute() -> Result<serde_json::Value, String> {
    let bytes = fs::read(crate::genesis::genesis_path()?).map_err(|e| e.to_string())?;
    let runtime = Runtime::from_bytes(&bytes).map_err(|e| e.to_string())?;
    let mut scratch = Scratch::new(runtime.config());
    let mut entries = serde_json::Map::new();
    for fixture in stdbev_fixtures::all() {
        let answer = runtime
            .decide(&fixture.request, &mut scratch)
            .map_err(|e| format!("{}: {e}", fixture.id))?;
        entries.insert(
            fixture.id.to_owned(),
            serde_json::json!({
                "probabilities": answer.probabilities(),
                "selected_index": answer.selected_index(),
            }),
        );
    }
    Ok(serde_json::json!({
        "model_id": runtime.model_id(),
        "note": "Regenerate with `cargo xtask golden`. A diff here means the numerical \
                 path changed -- confirm that was intended before accepting it.",
        "fixtures": entries,
    }))
}

/// Writes the golden file.
///
/// # Errors
/// Propagates computation and IO failures.
pub fn run() -> Result<(), String> {
    let value = compute()?;
    let path = golden_path()?;
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    }
    let text = serde_json::to_string_pretty(&value).map_err(|e| e.to_string())?;
    fs::write(&path, format!("{text}\n")).map_err(|e| e.to_string())?;
    println!("wrote {}", path.display());
    Ok(())
}

/// Largest permitted departure from a committed golden probability.
///
/// Not an exact comparison: an f32 written to JSON and read back as f64 differs in
/// its last digit, so exact equality would fail for a reason that has nothing to do
/// with the model. A gate that fails for the wrong reason gets ignored.
const TOLERANCE: f64 = 1e-6;

/// Compares one fixture's probabilities within [`TOLERANCE`].
fn fixture_matches(want: &serde_json::Value, got: &serde_json::Value) -> bool {
    if want.get("selected_index") != got.get("selected_index") {
        return false;
    }
    let (Some(a), Some(b)) = (
        want.get("probabilities").and_then(|v| v.as_array()),
        got.get("probabilities").and_then(|v| v.as_array()),
    ) else {
        return false;
    };
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(x, y)| match (x.as_f64(), y.as_f64()) {
                (Some(x), Some(y)) => (x - y).abs() <= TOLERANCE,
                _ => false,
            })
}

/// Fails if the committed golden file disagrees with what the artifact now produces.
///
/// # Errors
/// Returns a diff-style message naming the first fixture that changed.
pub fn check() -> Result<(), String> {
    let path = golden_path()?;
    let committed: serde_json::Value = serde_json::from_str(
        &fs::read_to_string(&path)
            .map_err(|e| format!("{}: {e} (run `cargo xtask golden`)", path.display()))?,
    )
    .map_err(|e| e.to_string())?;
    let current = compute()?;
    if committed.get("model_id") != current.get("model_id") {
        return Err("golden file was generated from a different artifact".into());
    }
    let null = serde_json::Value::Null;
    let want = committed.get("fixtures").unwrap_or(&null);
    let got = current.get("fixtures").unwrap_or(&null);
    let mut changed = Vec::new();
    for fixture in stdbev_fixtures::all() {
        let (w, g) = (&want[fixture.id], &got[fixture.id]);
        if !fixture_matches(w, g) {
            changed.push(format!("  {}: {w} -> {g}", fixture.id));
        }
    }
    if changed.is_empty() {
        println!("golden: numerical output unchanged");
        return Ok(());
    }
    Err(format!(
        "golden values changed:\n{}\nIf this was intended, re-run `cargo xtask golden`.",
        changed.join("\n")
    ))
}
