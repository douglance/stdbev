//! Checkpoint -> artifact, with the export gate.

use std::path::{Path, PathBuf};

use candle_core::Device;
use stdbev_data::TrainingExample;
use stdbev_runtime::{Runtime, Scratch};
use stdbev_train::export::{Checkpoint, to_artifact};
use stdbev_train::model::Scorer;
use stdbev_types::ModelConfig;

fn root() -> Result<PathBuf, String> {
    Ok(Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .ok_or("no workspace root")?
        .to_path_buf())
}

/// How far a single probability may differ between Candle and the INT8 runtime.
const MEAN_PROBABILITY_LIMIT: f32 = 0.02;
/// Minimum share of examples where both pick the same option.
const AGREEMENT_LIMIT: f32 = 0.99;

/// Runs one example through the INT8 runtime.
fn runtime_probabilities(
    runtime: &Runtime<'_>,
    scratch: &mut Scratch,
    example: &TrainingExample,
) -> Result<Vec<f32>, String> {
    runtime
        .score_options(&example.context, &example.options, scratch)
        .map_err(|e| e.to_string())
}

/// Runs one example through Candle.
fn candle_probabilities(
    scorer: &Scorer,
    example: &TrainingExample,
    config: &ModelConfig,
    device: &Device,
    temperature: f32,
) -> Result<Vec<f32>, String> {
    let batched =
        stdbev_train::batch::prepare(example, config, device, None).map_err(|e| e.to_string())?;
    let context = scorer
        .encode_context(&batched.context_ids, &batched.context_mask, false)
        .map_err(|e| e.to_string())?;
    let mut pooled = Vec::with_capacity(batched.option_ids.len());
    for (ids, mask) in batched.option_ids.iter().zip(&batched.option_masks) {
        pooled.push(
            scorer
                .pool_option(ids, mask, false)
                .map_err(|e| e.to_string())?,
        );
    }
    let logits = scorer
        .logits(&context, &batched.context_mask, &pooled)
        .map_err(|e| e.to_string())?;
    let mut values = logits.to_vec1::<f32>().map_err(|e| e.to_string())?;
    for v in &mut values {
        *v /= temperature;
    }
    stdbev_math::softmax_inplace(&mut values);
    Ok(values)
}

/// Exports the trained checkpoint, refusing to write if quantization changed answers.
///
/// The gate runs *before* the artifact is written, so a model that fails cannot be
/// deployed by accident. This is also where a wrong name-to-tensor-id mapping in the
/// exporter surfaces: it would produce a valid artifact that disagrees with Candle.
///
/// # Errors
/// Returns a message if the comparison fails any threshold.
pub fn run(temperature: f32, sample: usize) -> Result<(), String> {
    let root = root()?;
    // Must match the trained checkpoint, or the weights land in the wrong shapes.
    let config = crate::train::config_from_env();
    let device = Device::Cpu;
    let checkpoint_path = root.join("runs/default/model.safetensors");

    let checkpoint = Checkpoint::load(&checkpoint_path)?;
    let bytes = to_artifact(&checkpoint, config, temperature)?;
    println!("artifact: {} bytes", bytes.len());

    let (scorer, mut varmap) = Scorer::new(&config, &device).map_err(|e| e.to_string())?;
    varmap
        .load(&checkpoint_path)
        .map_err(|e| format!("reloading checkpoint into Candle: {e}"))?;

    let data = std::env::var("STDBEV_DATA")
        .map_or_else(|_| root.join("data/generated"), std::path::PathBuf::from);
    let validation = stdbev_data::jsonl::read(&data.join("validation.jsonl"))?;
    let examples: Vec<&TrainingExample> = validation.iter().take(sample).collect();
    let (agreement, mean_delta) =
        compare(&scorer, &bytes, &examples, &config, &device, temperature)?;
    println!(
        "candle FP32 vs INT8 runtime on {} examples: top-1 agreement {agreement:.4}, \
         mean |dp| {mean_delta:.5}",
        examples.len()
    );

    if agreement < AGREEMENT_LIMIT {
        return Err(format!(
            "top-1 agreement {agreement:.4} below {AGREEMENT_LIMIT}; refusing to write. \
             Either quantization is lossy here or the exporter's tensor mapping is wrong."
        ));
    }
    if mean_delta > MEAN_PROBABILITY_LIMIT {
        return Err(format!(
            "mean probability error {mean_delta:.5} above {MEAN_PROBABILITY_LIMIT}; \
             refusing to write"
        ));
    }

    std::fs::write(root.join("artifacts/model.stdbq"), &bytes).map_err(|e| e.to_string())?;
    let runtime = Runtime::from_bytes(&bytes).map_err(|e| e.to_string())?;
    println!(
        "wrote artifacts/model.stdbq  model_id {}",
        runtime.model_id()
    );
    Ok(())
}

/// Scores every example both ways and returns (top-1 agreement, mean |delta p|).
fn compare(
    scorer: &Scorer,
    bytes: &[u8],
    examples: &[&TrainingExample],
    config: &ModelConfig,
    device: &Device,
    temperature: f32,
) -> Result<(f32, f32), String> {
    let runtime = Runtime::from_bytes(bytes).map_err(|e| e.to_string())?;
    let mut scratch = Scratch::new(runtime.config());
    let mut agree = 0usize;
    let mut total_delta = 0.0f32;
    let mut compared = 0usize;

    for example in examples {
        let fp32 = candle_probabilities(scorer, example, config, device, temperature)?;
        let int8 = runtime_probabilities(&runtime, &mut scratch, example)?;
        if fp32.len() != int8.len() {
            return Err(format!("{}: option count differs", example.id));
        }
        if stdbev_math::argmax(&fp32).0 == stdbev_math::argmax(&int8).0 {
            agree += 1;
        }
        for (a, b) in fp32.iter().zip(&int8) {
            total_delta += (a - b).abs();
            compared += 1;
        }
    }
    #[allow(clippy::cast_precision_loss)]
    let out = (
        agree as f32 / examples.len().max(1) as f32,
        total_delta / compared.max(1) as f32,
    );
    Ok(out)
}
