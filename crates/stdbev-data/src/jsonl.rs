//! JSONL reading and writing.

use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

use crate::example::TrainingExample;

/// Reads and validates every line.
///
/// # Errors
/// Returns the failing line number with the reason, because "invalid data" in a
/// 100,000-line file is not actionable.
pub fn read(path: &Path) -> Result<Vec<TrainingExample>, String> {
    let file = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut out = Vec::new();
    for (i, line) in BufReader::new(file).lines().enumerate() {
        let line = line.map_err(|e| format!("{}:{}: {e}", path.display(), i + 1))?;
        if line.trim().is_empty() {
            continue;
        }
        let example: TrainingExample = serde_json::from_str(&line)
            .map_err(|e| format!("{}:{}: {e}", path.display(), i + 1))?;
        example
            .validate()
            .map_err(|e| format!("{}:{}: {e}", path.display(), i + 1))?;
        out.push(example);
    }
    Ok(out)
}

/// Writes one example per line.
///
/// # Errors
/// Propagates IO and serialization failures.
pub fn write(path: &Path, examples: &[TrainingExample]) -> Result<(), String> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    }
    let file = File::create(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut writer = BufWriter::new(file);
    for example in examples {
        let line = serde_json::to_string(example).map_err(|e| e.to_string())?;
        writeln!(writer, "{line}").map_err(|e| e.to_string())?;
    }
    writer.flush().map_err(|e| e.to_string())?;
    Ok(())
}
