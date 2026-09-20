//! The native <-> WASM parity gate.
//!
//! This is the single most important check in the repository. The model runs in two
//! places -- a native binary and a WASM module inside SpacetimeDB -- and if those ever
//! disagree, every downstream result is untrustworthy. It runs against the committed
//! genesis artifact, so it is meaningful long before anything is trained.
//!
//! The tolerance is 1e-5, but the expectation is *exact* agreement: every
//! transcendental goes through `libm` on both targets and float sums accumulate in a
//! fixed order, so there is no legitimate source of divergence. A non-zero delta here
//! is a bug to find, not a tolerance to widen.

// f64 -> f32 narrowing is intentional: the wire format carries JSON numbers and the
// model works in f32. The parity comparison is against f32 values on both sides.
#![allow(clippy::cast_possible_truncation)]

use std::process::Command;

use stdbev_fixtures::Fixture;
use stdbev_runtime::{Runtime, Scratch};

/// Maximum permitted per-probability difference.
const TOLERANCE: f32 = 1e-5;

struct Remote {
    probabilities: Vec<f32>,
    model_id: String,
    selected_index: u32,
    selected_label: String,
}

/// Compares every fixture across both runtimes.
///
/// # Errors
/// Returns a report naming every fixture that diverged.
pub fn run(server: &str, database: &str) -> Result<(), String> {
    let artifact = crate::genesis::artifact_path()?;
    let bytes = std::fs::read(&artifact).map_err(|e| format!("{}: {e}", artifact.display()))?;
    let runtime = Runtime::from_bytes(&bytes).map_err(|e| e.to_string())?;
    let mut scratch = Scratch::new(runtime.config());

    let mut failures = Vec::new();
    let mut worst = 0.0f32;
    for fixture in stdbev_fixtures::all() {
        let native = runtime
            .decide(&fixture.request, &mut scratch)
            .map_err(|e| format!("{}: native inference failed: {e}", fixture.id))?;
        let remote = fetch(server, database, fixture.id)?;

        match compare(&fixture, &native, &remote) {
            Ok(delta) => {
                worst = worst.max(delta);
                println!("  {} ok   max delta {delta:.3e}", fixture.id);
            }
            Err(why) => failures.push(format!("  {}: {why}", fixture.id)),
        }
    }

    if failures.is_empty() {
        println!(
            "parity: {} fixtures agree, worst delta {worst:.3e}",
            stdbev_fixtures::all().len()
        );
        return Ok(());
    }
    Err(format!(
        "native/WASM parity FAILED:\n{}",
        failures.join("\n")
    ))
}

fn compare(
    fixture: &Fixture,
    native: &stdbev_types::DecisionAnswer,
    remote: &Remote,
) -> Result<f32, String> {
    if native.model_id() != remote.model_id {
        return Err(format!(
            "model_id differs: native {} vs wasm {} -- the two sides ran different models",
            native.model_id(),
            remote.model_id
        ));
    }
    let n = native.probabilities();
    if n.len() != remote.probabilities.len() {
        return Err(format!(
            "{} probabilities vs {}",
            n.len(),
            remote.probabilities.len()
        ));
    }
    if u32::try_from(native.selected_index()).unwrap_or(u32::MAX) != remote.selected_index {
        return Err(format!(
            "selected_index differs: native {} vs wasm {}",
            native.selected_index(),
            remote.selected_index
        ));
    }
    let _ = (&fixture.id, &remote.selected_label);
    let mut worst = 0.0f32;
    for (i, (a, b)) in n.iter().zip(&remote.probabilities).enumerate() {
        let d = (a - b).abs();
        worst = worst.max(d);
        if d > TOLERANCE {
            return Err(format!(
                "probability[{i}] native {a} vs wasm {b} (delta {d:.3e})"
            ));
        }
    }
    Ok(worst)
}

/// Reads one stored result back out of SpacetimeDB.
fn fetch(server: &str, database: &str, id: &str) -> Result<Remote, String> {
    let query = format!(
        "SELECT probabilities, model_id, selected_index, selected_label \
         FROM decision_result WHERE request_id = '{id}'"
    );
    let out = Command::new("spacetime")
        .args(["sql", "-s", server, database, "--format", "json", &query])
        .output()
        .map_err(|e| format!("spacetime sql: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "spacetime sql failed: {}",
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let json: serde_json::Value = serde_json::from_str(text.trim())
        .map_err(|e| format!("parsing sql output: {e}\n{text}"))?;
    let row = json
        .get(0)
        .and_then(|r| r.get("rows"))
        .and_then(|r| r.get(0))
        .ok_or_else(|| format!("no row for request_id {id}; was the reducer called?"))?;
    Ok(Remote {
        probabilities: row
            .get(0)
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_f64().map(|f| f as f32))
                    .collect()
            })
            .ok_or("missing probabilities")?,
        model_id: row
            .get(1)
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_owned(),
        selected_index: row
            .get(2)
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(u64::MAX) as u32,
        selected_label: row
            .get(3)
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_owned(),
    })
}
