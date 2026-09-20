//! The training loop.

use candle_core::{Device, Result, Tensor};
use candle_nn::{Optimizer, VarMap};
use serde::Serialize;
use stdbev_data::TrainingExample;
use stdbev_types::ModelConfig;

use crate::Shuffler;
use crate::batch::prepare;
use crate::evaluate::{Evaluation, evaluate};
use crate::model::Scorer;

/// Hyperparameters.
#[derive(Debug, Clone, Copy)]
pub struct Settings {
    pub seed: u64,
    pub learning_rate: f64,
    pub weight_decay: f64,
    pub batch_size: usize,
    pub max_epochs: usize,
    pub patience: usize,
}

/// Largest permitted gradient norm.
///
/// The reference implementation clips at 1.0 and this one did not, which lets a single
/// bad batch move every parameter a long way -- exactly the kind of instability that
/// looks like overfitting from the outside.
const GRAD_CLIP: f64 = 1.0;

impl Default for Settings {
    fn default() -> Self {
        // These match `jevlike`, the implementation these numbers are actually known
        // to work for. The first pass used 3e-4 / 0.01 -- a learning rate 6.7x too low
        // and weight decay 100x too high -- which produced a model that barely moved
        // off its random initialization and looked, misleadingly, like overfitting.
        Self {
            seed: 42,
            learning_rate: 2e-3,
            weight_decay: 1e-4,
            batch_size: 64,
            max_epochs: 30,
            patience: 5,
        }
    }
}

/// What one epoch achieved.
#[derive(Debug, Clone, Serialize)]
pub struct EpochReport {
    pub epoch: usize,
    pub train_loss: f32,
    pub validation_loss: f32,
    pub validation_accuracy: f32,
    pub calibrated_nll: f32,
    pub improved: bool,
}

/// Cross-entropy against a soft target distribution.
///
/// `-Σ target[i] * log_softmax(logit)[i]`, which reduces to ordinary cross-entropy
/// when the target is one-hot and carries the teacher's full distribution when it is
/// not. One loss serves both synthetic hard labels and distilled soft ones.
pub(crate) fn loss(logits: &Tensor, targets: &Tensor) -> Result<Tensor> {
    let log_probs = candle_nn::ops::log_softmax(logits, 0)?;
    (targets * log_probs)?.sum_all()?.neg()
}

pub(crate) fn forward(
    scorer: &Scorer,
    batched: &crate::batch::Batched,
    train: bool,
) -> Result<Tensor> {
    let context = scorer.encode_context(&batched.context_ids, &batched.context_mask, train)?;
    let mut pooled = Vec::with_capacity(batched.option_ids.len());
    for (ids, mask) in batched.option_ids.iter().zip(&batched.option_masks) {
        pooled.push(scorer.pool_option(ids, mask, train)?);
    }
    scorer.logits(&context, &batched.context_mask, &pooled)
}

/// Scales every gradient down so their combined L2 norm is at most [`GRAD_CLIP`].
///
/// Applied to the whole gradient set at once, not per tensor: clipping each tensor
/// separately changes the update *direction*, which is a different operation.
fn clip_gradients(
    grads: &mut candle_core::backprop::GradStore,
    vars: &[candle_core::Var],
) -> Result<()> {
    let mut total = 0.0f64;
    for var in vars {
        if let Some(g) = grads.get(var.as_tensor()) {
            total += f64::from(g.sqr()?.sum_all()?.to_scalar::<f32>()?);
        }
    }
    let norm = total.sqrt();
    if norm <= GRAD_CLIP || !norm.is_finite() {
        return Ok(());
    }
    let scale = GRAD_CLIP / norm;
    for var in vars {
        if let Some(g) = grads.get(var.as_tensor()) {
            let scaled = (g * scale)?;
            grads.insert(var.as_tensor(), scaled);
        }
    }
    Ok(())
}

/// Runs one pass over the training set, returning mean loss.
///
/// Gradients accumulate across `batch_size` examples before a step, because option
/// count varies per example and padding every batch to the maximum would waste most
/// of the compute on padding.
#[allow(clippy::too_many_arguments)]
fn one_epoch(
    scorer: &Scorer,
    optimizer: &mut candle_nn::AdamW,
    vars: &[candle_core::Var],
    train: &[TrainingExample],
    order: &[usize],
    config: &ModelConfig,
    device: &Device,
    settings: Settings,
    shuffler: &mut Shuffler,
) -> Result<f32> {
    let mut epoch_loss = 0.0f32;
    let mut batch: Option<Tensor> = None;
    let mut in_batch = 0usize;

    for (seen, index) in order.iter().enumerate() {
        let Some(example) = train.get(*index) else {
            continue;
        };
        let batched = prepare(example, config, device, Some(shuffler))?;
        let logits = forward(scorer, &batched, true)?;
        let example_loss = loss(&logits, &batched.targets)?;
        epoch_loss += example_loss.to_scalar::<f32>()?;
        batch = Some(match batch {
            Some(acc) => (acc + example_loss)?,
            None => example_loss,
        });
        in_batch += 1;

        let last = seen + 1 == order.len();
        if in_batch < settings.batch_size && !last {
            continue;
        }
        if let Some(total) = batch.take() {
            #[allow(clippy::cast_precision_loss)]
            let mean = (total / in_batch as f64)?;
            let mut grads = mean.backward()?;
            clip_gradients(&mut grads, vars)?;
            optimizer.step(&grads)?;
        }
        in_batch = 0;
    }
    #[allow(clippy::cast_precision_loss)]
    let mean = epoch_loss / train.len().max(1) as f32;
    Ok(mean)
}

/// Everything `run` needs that is not a hyperparameter.
///
/// Bundled into a struct because eight positional parameters of mostly-reference type
/// is an easy place to transpose two arguments and get a silently wrong run.
pub struct Session<'a> {
    pub scorer: &'a Scorer,
    pub varmap: &'a VarMap,
    pub train: &'a [TrainingExample],
    pub validation: &'a [TrainingExample],
    pub config: &'a ModelConfig,
    pub device: &'a Device,
    pub checkpoint: &'a std::path::Path,
}

/// Records one epoch and prints it.
fn record(
    history: &mut Vec<EpochReport>,
    epoch: usize,
    train_loss: f32,
    evaluation: &Evaluation,
    improved: bool,
) {
    history.push(EpochReport {
        epoch,
        train_loss,
        validation_loss: evaluation.loss,
        validation_accuracy: evaluation.accuracy,
        calibrated_nll: evaluation.calibrated_nll,
        improved,
    });
    println!(
        "epoch {epoch:2}  train {train_loss:.4}  val {:.4}  cal_nll {:.4}  \
         val_acc {:.3}{}",
        evaluation.loss,
        evaluation.calibrated_nll,
        evaluation.accuracy,
        if improved { "  *" } else { "" }
    );
    // Explicit flush: stdout is block-buffered when piped, so without this a long run
    // looks hung for its entire duration.
    let _ = std::io::Write::flush(&mut std::io::stdout());
}

/// Builds the optimizer over every trainable parameter.
fn optimizer_for(session: &Session<'_>, settings: Settings) -> Result<candle_nn::AdamW> {
    let params = candle_nn::ParamsAdamW {
        lr: settings.learning_rate,
        weight_decay: settings.weight_decay,
        ..Default::default()
    };
    candle_nn::AdamW::new(session.varmap.all_vars(), params)
}

/// Trains to convergence, keeping the best validation checkpoint.
///
/// Selection is on validation **loss**, not accuracy: accuracy is a step function that
/// plateaus while the model is still sharpening, and a model picked on accuracy is
/// routinely worse calibrated than one picked on loss.
///
/// # Errors
/// Propagates Candle errors.
pub fn run(session: &Session<'_>, settings: Settings) -> Result<Vec<EpochReport>> {
    let vars = session.varmap.all_vars();
    let mut optimizer = optimizer_for(session, settings)?;
    let mut shuffler = Shuffler::new(settings.seed);
    let mut order: Vec<usize> = (0..session.train.len()).collect();

    let mut history = Vec::new();
    let mut best = f32::INFINITY;
    let mut since_improved = 0usize;

    for epoch in 0..settings.max_epochs {
        shuffler.shuffle(&mut order);
        let train_loss = one_epoch(
            session.scorer,
            &mut optimizer,
            &vars,
            session.train,
            &order,
            session.config,
            session.device,
            settings,
            &mut shuffler,
        )?;
        let evaluation = evaluate(
            session.scorer,
            session.validation,
            session.config,
            session.device,
        )?;
        let improved = evaluation.calibrated_nll < best;
        if improved {
            best = evaluation.calibrated_nll;
            since_improved = 0;
            session.varmap.save(session.checkpoint)?;
        } else {
            since_improved += 1;
        }
        record(&mut history, epoch, train_loss, &evaluation, improved);
        if since_improved >= settings.patience {
            println!(
                "early stop: {} epochs without improvement",
                settings.patience
            );
            break;
        }
    }
    Ok(history)
}
