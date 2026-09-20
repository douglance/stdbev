//! CI entry points.
//!
//! Split into a hermetic gate and a full one. `ci` needs nothing but Rust, so a
//! contributor without the SpacetimeDB CLI can still get a green local run; `ci-full`
//! adds everything that needs a live database. A gate that cannot be run locally is a
//! gate that gets ignored.

use std::process::Command;

/// Runs one command, streaming its output, and fails on a non-zero exit.
fn step(label: &str, program: &str, args: &[&str]) -> Result<(), String> {
    println!("\n--- {label} ---");
    let status = Command::new(program)
        .args(args)
        .status()
        .map_err(|e| format!("{label}: {program}: {e}"))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("{label} failed"))
    }
}

/// Everything that runs without a database.
///
/// # Errors
/// Returns at the first failing step.
pub fn hermetic() -> Result<(), String> {
    step("fmt", "cargo", &["fmt", "--all", "--", "--check"])?;
    step(
        "clippy",
        "cargo",
        &[
            "clippy",
            "--workspace",
            "--all-targets",
            "--all-features",
            "--",
            "-D",
            "warnings",
        ],
    )?;
    step("test", "cargo", &["test", "--workspace", "--all-features"])?;
    crate::policy::run()?;
    crate::genesis::run()?;
    crate::golden::check()?;
    println!("\nci: hermetic gate passed");
    Ok(())
}

/// The hermetic gate plus everything that needs the SpacetimeDB CLI.
///
/// # Errors
/// Returns at the first failing step.
pub fn full(server: &str, database: &str) -> Result<(), String> {
    hermetic()?;
    step(
        "stdb-build",
        "cargo",
        &[
            "build",
            "--release",
            "--target",
            "wasm32-unknown-unknown",
            "--manifest-path",
            "crates/stdbev-stdb/Cargo.toml",
        ],
    )?;
    crate::wasm::check_size()?;
    crate::parity::run(server, database)?;
    println!("\nci: full gate passed");
    Ok(())
}
