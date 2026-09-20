//! Turning examples into tensors.
//!
//! Tokenization goes through `stdbev-format`, the same code the runtime uses. A second
//! implementation here would be a second thing to keep in sync, and the failure would
//! be silent: the model would train on bytes the runtime never produces.

use candle_core::{Device, Result, Tensor};
use stdbev_data::TrainingExample;
use stdbev_format::encode;
use stdbev_types::ModelConfig;

/// One example, tokenized and ready for the model.
pub struct Batched {
    pub context_ids: Tensor,
    pub context_mask: Tensor,
    pub option_ids: Vec<Tensor>,
    pub option_masks: Vec<Tensor>,
    pub targets: Tensor,
}

fn to_tensors(text: &str, window: usize, device: &Device) -> Result<(Tensor, Tensor)> {
    let tokens = encode(text, window);
    let ids: Vec<u32> = tokens.ids.iter().map(|id| u32::from(*id)).collect();
    let mask: Vec<f32> = tokens
        .mask
        .iter()
        .map(|m| f32::from(u8::from(*m)))
        .collect();
    Ok((
        Tensor::from_vec(ids, window, device)?,
        Tensor::from_vec(mask, window, device)?,
    ))
}

/// Prepares one example.
///
/// When `shuffle` is set, option order is permuted and the target distribution is
/// permuted identically. That makes permutation equivariance a *trained* property as
/// well as a tested one -- without it the model can learn "the answer tends to be
/// option 1", which passes every structural test and fails in production.
///
/// # Errors
/// Propagates Candle errors.
pub fn prepare(
    example: &TrainingExample,
    config: &ModelConfig,
    device: &Device,
    shuffle: Option<&mut crate::Shuffler>,
) -> Result<Batched> {
    let mut order: Vec<usize> = (0..example.options.len()).collect();
    if let Some(rng) = shuffle {
        rng.shuffle(&mut order);
    }

    let (context_ids, context_mask) =
        to_tensors(&example.context, config.context_length as usize, device)?;

    let mut option_ids = Vec::with_capacity(order.len());
    let mut option_masks = Vec::with_capacity(order.len());
    let mut targets = Vec::with_capacity(order.len());
    for index in &order {
        let text = example.options.get(*index).map_or("", String::as_str);
        let (ids, mask) = to_tensors(text, config.option_length as usize, device)?;
        option_ids.push(ids);
        option_masks.push(mask);
        targets.push(
            example
                .target_probabilities
                .get(*index)
                .copied()
                .unwrap_or(0.0),
        );
    }
    let count = targets.len();
    Ok(Batched {
        context_ids,
        context_mask,
        option_ids,
        option_masks,
        targets: Tensor::from_vec(targets, count, device)?,
    })
}
