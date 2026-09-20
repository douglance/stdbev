//! WASM module size gate.

use std::path::PathBuf;

/// Hard ceiling on the published module.
const LIMIT: u64 = 8 * 1024 * 1024;

fn module_path() -> Result<PathBuf, String> {
    Ok(std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .ok_or("no workspace root")?
        .join("crates/stdbev-stdb/target/wasm32-unknown-unknown/release/stdbev_stdb.wasm"))
}

/// Fails if the built module exceeds the budget.
///
/// # Errors
/// Returns a message if the module is missing or too large.
pub fn check_size() -> Result<(), String> {
    let path = module_path()?;
    let bytes = std::fs::metadata(&path)
        .map_err(|e| format!("{}: {e} (run `cargo xtask stdb-build`)", path.display()))?
        .len();
    if bytes > LIMIT {
        return Err(format!("wasm module is {bytes} bytes, limit is {LIMIT}"));
    }
    #[allow(clippy::cast_precision_loss)]
    let pct = bytes as f64 / LIMIT as f64 * 100.0;
    println!("wasm-size: {bytes} bytes ({pct:.1}% of 8 MiB)");
    Ok(())
}
