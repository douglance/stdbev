//! Deterministic synthetic dataset generation.

mod choice;
mod noul;
pub mod rng;
mod score;
mod vocab;

use std::collections::BTreeMap;
use std::path::Path;

use crate::example::TrainingExample;
use crate::jsonl;
use crate::split::{Split, check_all_populated, check_no_leakage, split_for};
use rng::Rng;

/// Counts per split, for the generation report.
#[derive(Debug, Default, Clone, serde::Serialize)]
pub struct SplitCounts {
    pub train: usize,
    pub validation: usize,
    pub test: usize,
}

/// What generation produced.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Report {
    pub requested: usize,
    pub generated: usize,
    pub counts: SplitCounts,
    pub templates_per_split: BTreeMap<String, Vec<String>>,
}

/// Fails if a validation template never appears in training.
///
/// Validation drives early stopping and temperature calibration, and both assume
/// in-distribution data. A template that lands entirely in validation quietly breaks
/// that assumption, so it is caught here rather than discovered as a bad calibration
/// months later.
fn check_validation_in_distribution(
    templates: &BTreeMap<String, Vec<String>>,
) -> Result<(), String> {
    let empty = Vec::new();
    let train = templates.get("train").unwrap_or(&empty);
    let orphans: Vec<&String> = templates
        .get("validation")
        .unwrap_or(&empty)
        .iter()
        .filter(|t| !train.contains(t))
        .collect();
    if orphans.is_empty() {
        return Ok(());
    }
    Err(format!(
        "template(s) {} appear in validation but never in training; \
         the family needs more distinct groups to split on",
        orphans
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    ))
}

/// Generates `count` examples, split by template family, and writes them to `out`.
///
/// # Errors
/// Returns a message if a generated example is invalid, a template leaked across
/// splits, or writing failed.
pub fn generate(
    count: usize,
    seed: u64,
    out: &Path,
    format: stdbev_types::OptionFormat,
) -> Result<Report, String> {
    let mut rng = Rng::new(seed);
    let mut by_split: BTreeMap<&'static str, Vec<TrainingExample>> = BTreeMap::new();
    let mut assignments: Vec<(String, Split)> = Vec::new();

    for i in 0..count {
        // Even thirds, so no primitive is starved.
        let example = match i % 3 {
            0 => choice::generate(&mut rng, i, format),
            1 => noul::generate(&mut rng, i, format),
            _ => score::generate(&mut rng, i, format),
        };
        example
            .validate()
            .map_err(|e| format!("generated invalid example {}: {e}", example.id))?;
        let split = split_for(&example.template, &example.group);
        assignments.push((example.template.clone(), split));
        by_split.entry(split.stem()).or_default().push(example);
    }

    check_no_leakage(&assignments)?;

    let mut templates_per_split: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut counts = SplitCounts::default();
    for split in Split::all() {
        let examples = by_split.remove(split.stem()).unwrap_or_default();
        let mut templates: Vec<String> = examples.iter().map(|e| e.template.clone()).collect();
        templates.sort();
        templates.dedup();
        templates_per_split.insert(split.stem().to_owned(), templates);
        match split {
            Split::Train => counts.train = examples.len(),
            Split::Validation => counts.validation = examples.len(),
            Split::Test => counts.test = examples.len(),
        }
        jsonl::write(&out.join(format!("{}.jsonl", split.stem())), &examples)?;
    }

    check_all_populated(counts.train, counts.validation, counts.test)?;
    check_validation_in_distribution(&templates_per_split)?;

    Ok(Report {
        requested: count,
        generated: counts.train + counts.validation + counts.test,
        counts,
        templates_per_split,
    })
}
