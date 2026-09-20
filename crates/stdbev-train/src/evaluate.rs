//! Scoring a split without touching gradients.

use candle_core::{Device, Result};
use stdbev_data::TrainingExample;
use stdbev_types::ModelConfig;

use crate::batch::prepare;
use crate::model::Scorer;
use crate::train::{forward, loss};

/// What one evaluation pass measured.
pub struct Evaluation {
    pub loss: f32,
    pub accuracy: f32,
    /// Negative log likelihood after fitting a temperature on this same split.
    ///
    /// This is the number early stopping uses. Raw loss punishes a model for being
    /// overconfident, but the deployed artifact carries a fitted temperature that
    /// repairs exactly that -- so selecting on raw loss discards checkpoints that
    /// would have been fine once calibrated. Selecting on accuracy alone has the
    /// opposite problem: it plateaus while the model is still improving.
    pub calibrated_nll: f32,
}

/// Mean loss, accuracy and calibrated NLL over a split, without touching gradients.
///
/// # Errors
/// Propagates Candle errors.
pub fn evaluate(
    scorer: &Scorer,
    examples: &[TrainingExample],
    config: &ModelConfig,
    device: &Device,
) -> Result<Evaluation> {
    let mut total = 0.0f32;
    let mut correct = 0usize;
    let mut collected: Vec<stdbev_eval::Logits> = Vec::with_capacity(examples.len());
    for example in examples {
        let batched = prepare(example, config, device, None)?;
        let logits = forward(scorer, &batched, false)?;
        total += loss(&logits, &batched.targets)?.to_scalar::<f32>()?;
        let values = logits.to_vec1::<f32>()?;
        let predicted = stdbev_math::argmax(&values).0;
        if predicted == example.label() {
            correct += 1;
        }
        collected.push(stdbev_eval::Logits {
            values,
            label: example.label(),
        });
    }
    let n = examples.len().max(1);
    let temperature = stdbev_eval::fit(&collected);
    #[allow(clippy::cast_precision_loss)]
    let out = Evaluation {
        loss: total / n as f32,
        accuracy: correct as f32 / n as f32,
        calibrated_nll: stdbev_eval::nll_at(&collected, temperature),
    };
    Ok(out)
}
