//! Synthetic dataset generation.

use std::path::Path;

/// Seed for the committed dataset. Changing it changes every split.
const SEED: u64 = 42;

/// Generates `count` examples into `data/generated/`.
///
/// # Errors
/// Propagates generation and IO failures.
pub fn run(count: usize) -> Result<(), String> {
    let out = std::env::var("STDBEV_DATA_OUT").map_or_else(
        |_| {
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .map(|r| r.join("data").join("generated"))
                .unwrap_or_default()
        },
        std::path::PathBuf::from,
    );
    // The corpus must be rendered in the same option format the model will be served,
    // or training and serving disagree silently.
    // STDBEV_OPTION_FORMAT overrides the default so both encodings can be generated
    // for comparison without a rebuild.
    let format = match std::env::var("STDBEV_OPTION_FORMAT").as_deref() {
        Ok("verbose") => stdbev_types::OptionFormat::Verbose,
        Ok("compact") => stdbev_types::OptionFormat::Compact,
        _ => stdbev_types::ModelConfig::V1.option_format,
    };
    println!("option format: {format:?}");
    let report = stdbev_data::generate::generate(count, SEED, &out, format)?;
    println!(
        "{}",
        serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?
    );
    Ok(())
}
