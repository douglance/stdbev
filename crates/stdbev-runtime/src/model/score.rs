//! Dynamic option-attention scoring.
//!
//! Each option is the *query* and the context supplies keys and values -- inverted
//! from ordinary self-attention. This is what makes the model independent of option
//! count and option identity: no parameter anywhere depends on how many options there
//! are or what they are called.

use stdbev_math::{dot, layer_norm, masked_softmax_inplace, matvec_i8};
use stdbev_types::global;

use crate::error::RuntimeError;
use crate::model::Encoder;
use crate::scratch::Scratch;
use crate::view::find;

/// Quantized projection without bias.
struct Proj<'a> {
    weights: &'a [i8],
    scale: &'a [f32],
}

fn proj<'a>(e: &Encoder<'a>, w: u16, s: u16) -> Result<Proj<'a>, RuntimeError> {
    Ok(Proj {
        weights: find(e.tensors, w)?.as_i8()?,
        scale: find(e.tensors, s)?.as_f32()?,
    })
}

/// Converts a dimension to `f32` for the attention scale.
///
/// Bounded by `option_attention_rank`, so exactly representable.
#[allow(clippy::cast_precision_loss)]
fn rank_as_f32(rank: usize) -> f32 {
    rank as f32
}

/// Scores one pooled option against the encoded context.
///
/// The `sqrt(rank)` scaling is applied **twice** -- once to the attention scores and
/// again to the final logit -- matching the reference implementation. Dropping either
/// one changes the effective temperature of the whole model.
/// Projects the pooled option into the query space.
fn build_query(
    encoder: &Encoder<'_>,
    pooled: &[f32],
    scratch: &mut Scratch,
    width: usize,
) -> Result<(), RuntimeError> {
    let param = |id: u16| -> Result<&[f32], RuntimeError> { find(encoder.tensors, id)?.as_f32() };
    let query_proj = proj(encoder, global::OPT_WQ, global::OPT_WQ_SCALE)?;
    let mut normed = vec![0.0f32; width];
    layer_norm(
        &mut normed,
        pooled,
        param(global::OPT_OPTION_LN_GAMMA)?,
        param(global::OPT_OPTION_LN_BETA)?,
        1e-5,
    );
    matvec_i8(
        &mut scratch.query,
        &normed,
        query_proj.weights,
        query_proj.scale,
        None,
        width,
    );
    Ok(())
}

/// Projects every context token into key and value space.
///
/// Depends only on the context, so it runs **once per decision**, not once per option.
/// Computing it inside the option loop measured at ~8.5 ms of redundant work per
/// option -- 130 ms wasted on a 16-option Choice.
pub(crate) fn build_keys_values(
    encoder: &Encoder<'_>,
    context: &[f32],
    scratch: &mut Scratch,
    ctx_len: usize,
) -> Result<(), RuntimeError> {
    let (width, rank) = (encoder.shape.width, encoder.shape.rank);
    let param = |id: u16| -> Result<&[f32], RuntimeError> { find(encoder.tensors, id)?.as_f32() };
    let key_proj = proj(encoder, global::OPT_WK, global::OPT_WK_SCALE)?;
    let value_proj = proj(encoder, global::OPT_WV, global::OPT_WV_SCALE)?;
    let gamma = param(global::OPT_CONTEXT_LN_GAMMA)?;
    let beta = param(global::OPT_CONTEXT_LN_BETA)?;
    let mut normed = vec![0.0f32; width];
    for t in 0..ctx_len {
        let src = context.get(t * width..(t + 1) * width).unwrap_or(&[]);
        layer_norm(&mut normed, src, gamma, beta, 1e-5);
        let key = scratch
            .keys
            .get_mut(t * rank..(t + 1) * rank)
            .unwrap_or(&mut []);
        matvec_i8(key, &normed, key_proj.weights, key_proj.scale, None, width);
        let value = scratch
            .values
            .get_mut(t * rank..(t + 1) * rank)
            .unwrap_or(&mut []);
        matvec_i8(
            value,
            &normed,
            value_proj.weights,
            value_proj.scale,
            None,
            width,
        );
    }
    Ok(())
}

/// Attends over the context and reduces to one logit.
fn attend_to_logit(
    scratch: &mut Scratch,
    context_mask: &[bool],
    ctx_len: usize,
    rank: usize,
) -> f32 {
    let inv = 1.0 / libm::sqrtf(rank_as_f32(rank));
    for t in 0..ctx_len {
        let key = scratch.keys.get(t * rank..(t + 1) * rank).unwrap_or(&[]);
        let s = dot(&scratch.query, key) * inv;
        if let Some(cell) = scratch.attention.get_mut(t) {
            *cell = s;
        }
    }
    let live = scratch.attention.get_mut(..ctx_len).unwrap_or(&mut []);
    masked_softmax_inplace(live, context_mask);

    let mut attended = vec![0.0f32; rank];
    for t in 0..ctx_len {
        let p = scratch.attention.get(t).copied().unwrap_or(0.0);
        let value = scratch.values.get(t * rank..(t + 1) * rank).unwrap_or(&[]);
        for (a, x) in attended.iter_mut().zip(value) {
            *a += p * x;
        }
    }
    dot(&scratch.query, &attended) * inv
}

/// Scores one pooled option against the encoded context.
///
/// The `sqrt(rank)` scaling is applied **twice** -- once to the attention scores and
/// again to the final logit -- matching the reference implementation. Dropping either
/// changes the effective temperature of the whole model.
pub(crate) fn score_option(
    encoder: &Encoder<'_>,
    pooled: &[f32],
    context_mask: &[bool],
    scratch: &mut Scratch,
    ctx_len: usize,
) -> Result<f32, RuntimeError> {
    build_query(encoder, pooled, scratch, encoder.shape.width)?;
    Ok(attend_to_logit(
        scratch,
        context_mask,
        ctx_len,
        encoder.shape.rank,
    ))
}
