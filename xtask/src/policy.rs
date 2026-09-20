//! Source-size policy.
//!
//! Enforced as a CI failure rather than an advisory warning, because a warning that
//! nobody must act on is just noise.

use std::fs;
use std::path::{Path, PathBuf};

const SOURCE_LIMIT: usize = 300;
const TEST_LIMIT: usize = 500;
const EXCLUDED: &[&str] = &["target", "module_bindings"];

/// Fails if any owned Rust file exceeds its line budget.
///
/// # Errors
/// Returns a list of every violation, not just the first.
pub fn run() -> Result<(), String> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .ok_or("no workspace root")?
        .to_path_buf();
    let mut files = Vec::new();
    collect(&root, &mut files)?;

    let mut violations = Vec::new();
    let mut counted = 0usize;
    for path in files {
        let text = fs::read_to_string(&path).map_err(|e| e.to_string())?;
        let lines = text.lines().count();
        let is_test = path.components().any(|c| c.as_os_str() == "tests")
            || path.file_name().is_some_and(|n| n == "tests.rs");
        let limit = if is_test { TEST_LIMIT } else { SOURCE_LIMIT };
        counted += 1;
        if lines > limit {
            let rel = path.strip_prefix(&root).unwrap_or(&path).display();
            violations.push(format!("  {rel}: {lines} lines (limit {limit})"));
        }
    }
    if violations.is_empty() {
        println!("source-policy: {counted} files, all within budget");
        return Ok(());
    }
    Err(format!(
        "{} file(s) over budget:\n{}",
        violations.len(),
        violations.join("\n")
    ))
}

fn collect(dir: &Path, out: &mut Vec<PathBuf>) -> Result<(), String> {
    for entry in fs::read_dir(dir).map_err(|e| e.to_string())? {
        let path = entry.map_err(|e| e.to_string())?.path();
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if EXCLUDED.contains(&name) || name.starts_with('.') {
            continue;
        }
        if path.is_dir() {
            collect(&path, out)?;
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
    Ok(())
}
