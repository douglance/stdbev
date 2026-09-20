//! Test-set evaluation with calibration and the shuffled-context control.

use std::path::{Path, PathBuf};

use candle_core::Device;
use stdbev_data::TrainingExample;
use stdbev_eval::{Logits, Scored};
use stdbev_train::model::Scorer;
use stdbev_types::ModelConfig;

fn root() -> Result<PathBuf, String> {
    Ok(Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .ok_or("no workspace root")?
        .to_path_buf())
}

/// Logits paired with the true label, for a set of examples.
type Scores = Vec<(Vec<f32>, usize)>;

/// Raw logits for one example.
fn logits_for(
    scorer: &Scorer,
    example: &TrainingExample,
    config: &ModelConfig,
    device: &Device,
    shuffle_context_with: Option<&TrainingExample>,
) -> Result<Vec<f32>, String> {
    let batched =
        stdbev_train::batch::prepare(example, config, device, None).map_err(|e| e.to_string())?;
    // The control swaps in *another example's* context. If accuracy survives that, the
    // model is exploiting option priors rather than reading the state.
    let context_source = shuffle_context_with.unwrap_or(example);
    let context_batch = stdbev_train::batch::prepare(context_source, config, device, None)
        .map_err(|e| e.to_string())?;
    let context = scorer
        .encode_context(
            &context_batch.context_ids,
            &context_batch.context_mask,
            false,
        )
        .map_err(|e| e.to_string())?;
    let mut pooled = Vec::with_capacity(batched.option_ids.len());
    for (ids, mask) in batched.option_ids.iter().zip(&batched.option_masks) {
        pooled.push(
            scorer
                .pool_option(ids, mask, false)
                .map_err(|e| e.to_string())?,
        );
    }
    scorer
        .logits(&context, &context_batch.context_mask, &pooled)
        .map_err(|e| e.to_string())?
        .to_vec1::<f32>()
        .map_err(|e| e.to_string())
}

fn scored_at(raw: &Scores, temperature: f32) -> Vec<Scored> {
    raw.iter()
        .map(|(values, label)| {
            let mut p: Vec<f32> = values.iter().map(|v| v / temperature).collect();
            stdbev_math::softmax_inplace(&mut p);
            Scored {
                probabilities: p,
                label: *label,
            }
        })
        .collect()
}

/// Fits a temperature on validation, then reports test metrics and the control.
///
/// # Errors
/// Propagates data, Candle and IO failures.
pub fn run(limit: Option<usize>) -> Result<(), String> {
    let root = root()?;
    // Must match the checkpoint's architecture, or the weights are loaded into the
    // wrong shapes and every number is meaningless.
    let config = crate::train::config_from_env();
    println!("config: {} params", config.parameter_count());
    let device = Device::Cpu;
    let (scorer, mut varmap) = Scorer::new(&config, &device).map_err(|e| e.to_string())?;
    varmap
        .load(root.join("runs/default/model.safetensors"))
        .map_err(|e| format!("loading checkpoint: {e}"))?;

    let data = std::env::var("STDBEV_DATA")
        .map_or_else(|_| root.join("data/generated"), std::path::PathBuf::from);
    let mut validation = stdbev_data::jsonl::read(&data.join("validation.jsonl"))?;
    let mut test = stdbev_data::jsonl::read(&data.join("test.jsonl"))?;
    if let Some(n) = limit {
        validation.truncate(n);
        test.truncate(n);
    }

    let temperature = calibrate(&scorer, &validation, &config, &device)?;
    println!("fitted temperature: {temperature:.4}");

    let (raw, shuffled) = score_test(&scorer, &test, &config, &device)?;
    let report = stdbev_eval::evaluate(&scored_at(&raw, temperature));
    let control = stdbev_eval::evaluate(&scored_at(&shuffled, temperature));

    let output = serde_json::json!({
        "temperature": temperature,
        "test": report,
        // Per primitive, because an aggregate hides the case that is actually broken.
        // Noul at chance inside a healthy-looking average is invisible otherwise.
        "by_primitive": by_primitive(&test, &raw, temperature),
        // Splits the test set by whether the correct label appears verbatim in the
        // state. This architecture matches lexically, so the two subsets measure
        // different things and an aggregate over them means very little.
        "by_lexical_cue": by_lexical_cue(&test, &raw, temperature),
        "shuffled_context_control": {
            "top_1_accuracy": control.top_1_accuracy,
            // The evidence that matters: a model reading context collapses toward
            // chance when given someone else's context. One holding its accuracy is
            // exploiting option priors.
            "degradation": report.top_1_accuracy - control.top_1_accuracy,
        },
    });
    let rendered = serde_json::to_string_pretty(&output).map_err(|e| e.to_string())?;
    println!("{rendered}");
    std::fs::write(root.join("runs/default/evaluation.json"), rendered)
        .map_err(|e| e.to_string())?;
    Ok(())
}

/// Accuracy split by whether the correct label is literally present in the state.
fn by_lexical_cue(test: &[TrainingExample], raw: &Scores, temperature: f32) -> serde_json::Value {
    let mut present: Scores = Vec::new();
    let mut absent: Scores = Vec::new();
    for (example, scored) in test.iter().zip(raw) {
        let label = example
            .options
            .get(example.label())
            .map(|o| o.to_lowercase())
            .unwrap_or_default();
        let state = example.context.to_lowercase();
        if !label.is_empty() && state.contains(&label) {
            present.push(scored.clone());
        } else {
            absent.push(scored.clone());
        }
    }
    let score = |set: &Scores| {
        if set.is_empty() {
            return serde_json::Value::Null;
        }
        let r = stdbev_eval::evaluate(&scored_at(set, temperature));
        serde_json::json!({ "n": r.example_count, "top_1_accuracy": r.top_1_accuracy })
    };
    serde_json::json!({
        "label_in_state": score(&present),
        "label_absent": score(&absent),
    })
}

/// Accuracy broken down by question primitive.
fn by_primitive(test: &[TrainingExample], raw: &Scores, temperature: f32) -> serde_json::Value {
    let mut out = serde_json::Map::new();
    for kind in ["choice", "noul", "score"] {
        let subset: Scores = test
            .iter()
            .zip(raw)
            .filter(|(e, _)| e.id.starts_with(kind))
            .map(|(_, s)| s.clone())
            .collect();
        if subset.is_empty() {
            continue;
        }
        let report = stdbev_eval::evaluate(&scored_at(&subset, temperature));
        out.insert(
            kind.to_owned(),
            serde_json::json!({
                "n": report.example_count,
                "top_1_accuracy": report.top_1_accuracy,
                "expected_calibration_error": report.expected_calibration_error,
            }),
        );
    }
    serde_json::Value::Object(out)
}

/// Fits the temperature on validation, which is in-distribution by construction.
fn calibrate(
    scorer: &Scorer,
    validation: &[TrainingExample],
    config: &ModelConfig,
    device: &Device,
) -> Result<f32, String> {
    let mut calibration = Vec::with_capacity(validation.len());
    for example in validation {
        let values = logits_for(scorer, example, config, device, None)?;
        calibration.push(Logits {
            values,
            label: example.label(),
        });
    }
    let temperature = stdbev_eval::fit(&calibration);
    if stdbev_eval::hit_bound(temperature) {
        println!(
            "WARNING: fitted temperature {temperature:.3} is at a search bound; the model \
             is badly miscalibrated and rescaling will not fix it"
        );
    }
    Ok(temperature)
}

/// Scores the test split normally and with shuffled contexts.
fn score_test(
    scorer: &Scorer,
    test: &[TrainingExample],
    config: &ModelConfig,
    device: &Device,
) -> Result<(Scores, Scores), String> {
    let mut raw: Scores = Vec::with_capacity(test.len());
    let mut shuffled: Scores = Vec::with_capacity(test.len());
    for (i, example) in test.iter().enumerate() {
        raw.push((
            logits_for(scorer, example, config, device, None)?,
            example.label(),
        ));
        // Deterministic derangement: every example is paired with a different one, so
        // unlike a batch roll the control never degenerates into a no-op.
        let other = test
            .get((i + 1 + i % 7) % test.len().max(1))
            .unwrap_or(example);
        shuffled.push((
            logits_for(scorer, example, config, device, Some(other))?,
            example.label(),
        ));
    }
    Ok((raw, shuffled))
}
