//! Writes the committed random-weight artifact.
//!
//! This exists to break a chicken-and-egg: the SpacetimeDB module embeds the artifact
//! with `include_bytes!`, so it cannot compile until one exists -- but CI builds the
//! module from a clean checkout, long before anything is trained. Committing a
//! deterministic genesis artifact makes a fresh clone buildable and gives the
//! native/WASM parity test something to run against from day one.

use std::fs;
use std::path::Path;

use stdbev_quant::genesis_artifact;
use stdbev_runtime::Runtime;
use stdbev_types::ModelConfig;

/// Seed for the committed artifact. Changing it changes `model_id` everywhere.
const SEED: u64 = 42;

/// The deterministic random-weight artifact.
///
/// Kept separate from the deployable model so CI can regenerate and verify it without
/// overwriting trained weights -- which it did, silently, until the two were split.
///
/// # Errors
/// Returns a message if the workspace root cannot be resolved.
pub fn genesis_path() -> Result<std::path::PathBuf, String> {
    Ok(artifacts_dir()?.join("genesis.stdbq"))
}

/// The deployable model embedded in the SpacetimeDB module.
///
/// # Errors
/// Returns a message if the workspace root cannot be resolved.
pub fn artifact_path() -> Result<std::path::PathBuf, String> {
    Ok(artifacts_dir()?.join("model.stdbq"))
}

fn artifacts_dir() -> Result<std::path::PathBuf, String> {
    Ok(Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .ok_or("no workspace root")?
        .join("artifacts"))
}

/// `part` as a percentage of `whole`.
///
/// Both are artifact sizes under a megabyte, far inside f64's exact-integer range.
#[allow(clippy::cast_precision_loss)]
fn percent(part: usize, whole: usize) -> f64 {
    part as f64 / whole as f64 * 100.0
}

/// Generates `artifacts/model.stdbq` and its metadata sidecar.
///
/// # Errors
/// Returns a message if generation, parsing or writing fails.
pub fn run() -> Result<(), String> {
    let config = ModelConfig::V1;
    let bytes = genesis_artifact(config, SEED)?;
    let rt =
        Runtime::from_bytes(&bytes).map_err(|e| format!("genesis artifact is invalid: {e}"))?;

    let limit = 1024 * 1024;
    if bytes.len() > limit {
        return Err(format!(
            "artifact is {} bytes, limit is {limit}",
            bytes.len()
        ));
    }

    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .ok_or("no workspace root")?
        .join("artifacts");
    fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    fs::write(dir.join("genesis.stdbq"), &bytes).map_err(|e| e.to_string())?;

    let meta = serde_json::json!({
        "format": "STDBEV01",
        "provenance": "genesis: deterministic random weights, NOT trained",
        "seed": SEED,
        "model_id": rt.model_id(),
        "architecture": config,
        "artifact_bytes": bytes.len(),
        "parameter_count": config.parameter_count(),
        "temperature": rt.temperature(),
    });
    fs::write(
        dir.join("genesis.meta.json"),
        serde_json::to_string_pretty(&meta).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;

    println!(
        "wrote artifacts/genesis.stdbq  {} bytes ({:.1}% of 1 MiB)  model_id {}",
        bytes.len(),
        percent(bytes.len(), limit),
        rt.model_id()
    );
    Ok(())
}
