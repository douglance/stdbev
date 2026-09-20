//! Training entry point.

use std::path::{Path, PathBuf};

use candle_core::Device;
use stdbev_train::evaluate::evaluate;
use stdbev_train::model::Scorer;
use stdbev_train::train::{Settings, run as train_run};
use stdbev_types::ModelConfig;

fn root() -> Result<PathBuf, String> {
    Ok(Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .ok_or("no workspace root")?
        .to_path_buf())
}

/// Reads the architecture from the environment so configurations can be compared
/// without a rebuild.
///
/// The defaults are `ModelConfig::V1`. `STDBEV_SHAPE=jevlike` selects the reference
/// implementation's actual shape -- width 64, rank 64, no transformer layers -- which
/// is **12x smaller** than V1 (41k parameters against 508k). At 61k training examples
/// that is 0.68 parameters per example rather than 8.3, and the first runs here showed
/// exactly the symptom that ratio predicts: accuracy climbing while calibrated
/// likelihood degraded.
pub fn config_from_env() -> ModelConfig {
    let option_encoder = match std::env::var("STDBEV_OPTION_ENCODER").as_deref() {
        Ok("encoded") => stdbev_types::OptionEncoder::Encoded,
        _ => stdbev_types::OptionEncoder::Pooled,
    };
    let base = match std::env::var("STDBEV_SHAPE").as_deref() {
        Ok("jevlike") => ModelConfig {
            model_width: 64,
            option_attention_rank: 64,
            transformer_layers: 0,
            attention_heads: 4,
            feed_forward_width: 256,
            ..ModelConfig::V1
        },
        Ok("small") => ModelConfig {
            transformer_layers: 1,
            ..ModelConfig::V1
        },
        _ => ModelConfig::V1,
    };
    ModelConfig {
        option_encoder,
        ..base
    }
}

/// The training and validation splits.
type Splits = (
    Vec<stdbev_data::TrainingExample>,
    Vec<stdbev_data::TrainingExample>,
);

/// Validation examples used for per-epoch early stopping.
const VALIDATION_SAMPLE: usize = 2500;

/// Trains a model and writes the checkpoint under `runs/default/`.
///
/// # Errors
/// Propagates data, Candle and IO failures.
pub fn run(max_epochs: usize, limit: Option<usize>) -> Result<(), String> {
    let root = root()?;
    let (train, validation) = load_splits(&root, limit)?;
    println!("train {} / validation {}", train.len(), validation.len());
    let _ = std::io::Write::flush(&mut std::io::stdout());

    let config = config_from_env();
    println!(
        "config: width {} rank {} layers {} option_encoder {:?} -> {} params",
        config.model_width,
        config.option_attention_rank,
        config.transformer_layers,
        config.option_encoder,
        config.parameter_count()
    );
    let device = Device::Cpu;
    let (scorer, varmap) = Scorer::new(&config, &device).map_err(|e| e.to_string())?;

    let before = evaluate(&scorer, &validation, &config, &device).map_err(|e| e.to_string())?;
    println!(
        "untrained: val loss {:.4}  cal_nll {:.4}  val_acc {:.3}",
        before.loss, before.calibrated_nll, before.accuracy
    );
    let _ = std::io::Write::flush(&mut std::io::stdout());

    let run_dir = root.join("runs/default");
    std::fs::create_dir_all(&run_dir).map_err(|e| e.to_string())?;
    let checkpoint = run_dir.join("model.safetensors");

    let settings = Settings {
        max_epochs,
        ..Settings::default()
    };
    let session = stdbev_train::train::Session {
        scorer: &scorer,
        varmap: &varmap,
        train: &train,
        validation: &validation,
        config: &config,
        device: &device,
        checkpoint: &checkpoint,
    };
    let history = train_run(&session, settings).map_err(|e| e.to_string())?;
    report(&run_dir, &history, before.accuracy)
}

/// Loads the training and validation splits.
fn load_splits(root: &Path, limit: Option<usize>) -> Result<Splits, String> {
    // STDBEV_DATA lets an experiment point at an alternative corpus without moving
    // the committed one.
    let data = std::env::var("STDBEV_DATA")
        .map_or_else(|_| root.join("data/generated"), std::path::PathBuf::from);
    let mut train = stdbev_data::jsonl::read(&data.join("train.jsonl"))?;
    let mut validation = stdbev_data::jsonl::read(&data.join("validation.jsonl"))?;
    if let Some(n) = limit {
        train.truncate(n);
    }
    // Early stopping and per-epoch calibration do not need the whole validation split,
    // and at 10k examples it would dominate epoch time. The full split is still used
    // for the final calibration in `xtask eval`.
    validation.truncate(VALIDATION_SAMPLE);
    Ok((train, validation))
}

/// Writes the training history and prints the selected epoch.
fn report(
    run_dir: &Path,
    history: &[stdbev_train::train::EpochReport],
    untrained_accuracy: f32,
) -> Result<(), String> {
    let best = history
        .iter()
        .rfind(|h| h.improved)
        .ok_or("no epoch improved on the initial model")?;
    println!(
        "\nbest epoch {}: cal_nll {:.4}  val_acc {:.3}  (untrained {untrained_accuracy:.3})",
        best.epoch, best.calibrated_nll, best.validation_accuracy
    );
    std::fs::write(
        run_dir.join("training.json"),
        serde_json::to_string_pretty(history).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    println!(
        "checkpoint: {}",
        run_dir.join("model.safetensors").display()
    );
    Ok(())
}
