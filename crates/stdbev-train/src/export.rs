//! Candle checkpoint -> `STDBEV01` artifact.
//!
//! The mapping from Candle's parameter names to tensor ids lives here and nowhere
//! else. A wrong mapping does not crash: it produces a valid artifact that computes
//! confident nonsense, which is why `cargo xtask export` compares Candle's own logits
//! against the exported runtime's rather than trusting this table.

use std::collections::HashMap;
use std::path::Path;

use candle_core::{Device, Tensor};
use stdbev_quant::{ArtifactBuilder, TensorSource, quantize_rows};
use stdbev_types::{ModelConfig, TensorId, global, slot};

/// A matrix read from the checkpoint: values, rows, columns.
type Matrix = (Vec<f32>, u16, u16);

/// Loaded parameters, keyed by Candle's dotted name.
pub struct Checkpoint(HashMap<String, Tensor>);

/// A missing tensor is almost always a rename, so list what *is* there.
fn missing(name: &str, have: &HashMap<String, Tensor>) -> String {
    let mut names: Vec<&str> = have.keys().map(String::as_str).collect();
    names.sort_unstable();
    format!(
        "checkpoint has no tensor {name:?}; it contains: {}",
        names.join(", ")
    )
}

impl Checkpoint {
    /// Reads a safetensors checkpoint.
    ///
    /// # Errors
    /// Propagates IO and Candle errors.
    pub fn load(path: &Path) -> Result<Self, String> {
        let tensors = candle_core::safetensors::load(path, &Device::Cpu)
            .map_err(|e| format!("{}: {e}", path.display()))?;
        Ok(Self(tensors))
    }

    fn rows(&self, name: &str) -> Result<Matrix, String> {
        let tensor = self.0.get(name).ok_or_else(|| missing(name, &self.0))?;
        let (rows, cols) = tensor
            .dims2()
            .map_err(|e| format!("{name}: expected a matrix: {e}"))?;
        let values = tensor
            .flatten_all()
            .and_then(|t| t.to_vec1::<f32>())
            .map_err(|e| format!("{name}: {e}"))?;
        let rows = u16::try_from(rows).map_err(|_| format!("{name}: too many rows"))?;
        let cols = u16::try_from(cols).map_err(|_| format!("{name}: too many columns"))?;
        Ok((values, rows, cols))
    }

    fn vector(&self, name: &str) -> Result<Vec<f32>, String> {
        let tensor = self.0.get(name).ok_or_else(|| missing(name, &self.0))?;
        tensor
            .flatten_all()
            .and_then(|t| t.to_vec1::<f32>())
            .map_err(|e| format!("{name}: {e}"))
    }
}

/// Quantizes one matrix and writes it with its per-row scales.
fn matrix(
    b: &mut ArtifactBuilder,
    c: &Checkpoint,
    name: &str,
    weight: TensorId,
    scale: TensorId,
) -> Result<(), String> {
    let (values, rows, cols) = c.rows(name)?;
    let q = quantize_rows(&values, cols as usize);
    b.push(
        weight,
        TensorSource::I8 {
            values: q.values,
            dims: vec![rows, cols],
        },
    );
    b.push(
        scale,
        TensorSource::F32 {
            values: q.scales,
            dims: vec![rows],
        },
    );
    Ok(())
}

/// Writes one f32 vector verbatim.
fn vector(b: &mut ArtifactBuilder, c: &Checkpoint, name: &str, id: TensorId) -> Result<(), String> {
    let values = c.vector(name)?;
    let len = u16::try_from(values.len()).map_err(|_| format!("{name}: too long"))?;
    b.push(
        id,
        TensorSource::F32 {
            values,
            dims: vec![len],
        },
    );
    Ok(())
}

/// The four attention projections and their biases.
fn attention(b: &mut ArtifactBuilder, c: &Checkpoint, layer: u16) -> Result<(), String> {
    let id = |s: u16| TensorId::layer(layer, s);
    let p = format!("block{layer}");
    for (name, w, s, bias) in [
        ("wq", slot::WQ, slot::WQ_SCALE, slot::WQ_BIAS),
        ("wk", slot::WK, slot::WK_SCALE, slot::WK_BIAS),
        ("wv", slot::WV, slot::WV_SCALE, slot::WV_BIAS),
        ("wo", slot::WO, slot::WO_SCALE, slot::WO_BIAS),
    ] {
        matrix(b, c, &format!("{p}.{name}.weight"), id(w), id(s))?;
        vector(b, c, &format!("{p}.{name}.bias"), id(bias))?;
    }
    Ok(())
}

/// The feed-forward sublayer.
fn feed_forward(b: &mut ArtifactBuilder, c: &Checkpoint, layer: u16) -> Result<(), String> {
    let id = |s: u16| TensorId::layer(layer, s);
    let p = format!("block{layer}");
    matrix(
        b,
        c,
        &format!("{p}.ff1.weight"),
        id(slot::FF1),
        id(slot::FF1_SCALE),
    )?;
    vector(b, c, &format!("{p}.ff1.bias"), id(slot::FF1_BIAS))?;
    matrix(
        b,
        c,
        &format!("{p}.ff2.weight"),
        id(slot::FF2),
        id(slot::FF2_SCALE),
    )?;
    vector(b, c, &format!("{p}.ff2.bias"), id(slot::FF2_BIAS))
}

/// One transformer block.
fn layer(b: &mut ArtifactBuilder, c: &Checkpoint, n: u16) -> Result<(), String> {
    let id = |s: u16| TensorId::layer(n, s);
    let p = format!("block{n}");
    vector(b, c, &format!("{p}.ln1.weight"), id(slot::LN1_GAMMA))?;
    vector(b, c, &format!("{p}.ln1.bias"), id(slot::LN1_BETA))?;
    attention(b, c, n)?;
    vector(b, c, &format!("{p}.ln2.weight"), id(slot::LN2_GAMMA))?;
    vector(b, c, &format!("{p}.ln2.bias"), id(slot::LN2_BETA))?;
    feed_forward(b, c, n)
}

/// The option-attention head.
fn option_attention(b: &mut ArtifactBuilder, c: &Checkpoint) -> Result<(), String> {
    let g = TensorId::global;
    vector(
        b,
        c,
        "opt_context_ln.weight",
        g(global::OPT_CONTEXT_LN_GAMMA),
    )?;
    vector(b, c, "opt_context_ln.bias", g(global::OPT_CONTEXT_LN_BETA))?;
    vector(b, c, "opt_option_ln.weight", g(global::OPT_OPTION_LN_GAMMA))?;
    vector(b, c, "opt_option_ln.bias", g(global::OPT_OPTION_LN_BETA))?;
    matrix(
        b,
        c,
        "opt_q.weight",
        g(global::OPT_WQ),
        g(global::OPT_WQ_SCALE),
    )?;
    matrix(
        b,
        c,
        "opt_k.weight",
        g(global::OPT_WK),
        g(global::OPT_WK_SCALE),
    )?;
    matrix(
        b,
        c,
        "opt_v.weight",
        g(global::OPT_WV),
        g(global::OPT_WV_SCALE),
    )
}

/// Converts a trained checkpoint into a deployable artifact.
///
/// # Errors
/// Returns a message naming the tensor that could not be mapped.
pub fn to_artifact(
    checkpoint: &Checkpoint,
    config: ModelConfig,
    temperature: f32,
) -> Result<Vec<u8>, String> {
    let mut b = ArtifactBuilder::new(config, temperature);
    let g = TensorId::global;
    matrix(
        &mut b,
        checkpoint,
        "token.weight",
        g(global::TOKEN_EMBEDDING),
        g(global::TOKEN_EMBEDDING_SCALE),
    )?;
    matrix(
        &mut b,
        checkpoint,
        "position.weight",
        g(global::POSITION_EMBEDDING),
        g(global::POSITION_EMBEDDING_SCALE),
    )?;
    for n in 0..config.transformer_layers {
        layer(&mut b, checkpoint, n)?;
    }
    // The final LayerNorm is part of the format but unused by the forward pass, so it
    // is written as an identity rather than omitted -- the parser requires it.
    let width = config.model_width;
    b.push(
        g(global::FINAL_LN_GAMMA),
        TensorSource::F32 {
            values: vec![1.0; width as usize],
            dims: vec![width],
        },
    );
    b.push(
        g(global::FINAL_LN_BETA),
        TensorSource::F32 {
            values: vec![0.0; width as usize],
            dims: vec![width],
        },
    );
    option_attention(&mut b, checkpoint)?;
    b.build()
}
