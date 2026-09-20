//! Deterministic random-weight artifact generation.
//!
//! This exists so the entire deployment spine -- parser, runtime, WASM module,
//! native/WASM parity -- can be tested before a single example has been trained.
//! The weights are meaningless; the plumbing they exercise is not.

use stdbev_types::{ModelConfig, TensorId, global, slot};

use crate::quantize::quantize_rows;
use crate::write::{ArtifactBuilder, TensorSource};

/// A SplitMix64 generator, written out rather than pulled from a crate.
///
/// The genesis artifact must be byte-identical on every machine and every future
/// version of this workspace, because it is committed and its `model_id` is asserted
/// in tests. A third-party RNG explicitly does not promise value stability across
/// releases; twelve lines here do.
struct SplitMix(u64);

impl SplitMix {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[-1, 1)`.
    fn next_signed(&mut self) -> f32 {
        // Top 24 bits give an exactly-representable f32 mantissa.
        // Shifted to 24 bits, which f32 represents exactly.
        #[allow(clippy::cast_precision_loss)]
        let bits = (self.next_u64() >> 40) as f32;
        bits / 8_388_608.0 - 1.0
    }
}

fn normal(rng: &mut SplitMix, n: usize, scale: f32) -> Vec<f32> {
    // Sum of three uniforms is a good enough bell shape for plumbing weights, and
    // needs no distributions dependency.
    (0..n)
        .map(|_| {
            let s: f32 = (0..3).map(|_| rng.next_signed()).sum();
            s * scale
        })
        .collect()
}

fn push_matrix(
    b: &mut ArtifactBuilder,
    rng: &mut SplitMix,
    w_id: TensorId,
    s_id: TensorId,
    rows: u16,
    cols: u16,
) {
    let raw = normal(rng, rows as usize * cols as usize, 0.08);
    let q = quantize_rows(&raw, cols as usize);
    b.push(
        w_id,
        TensorSource::I8 {
            values: q.values,
            dims: vec![rows, cols],
        },
    );
    b.push(
        s_id,
        TensorSource::F32 {
            values: q.scales,
            dims: vec![rows],
        },
    );
}

fn push_vec(b: &mut ArtifactBuilder, id: TensorId, values: Vec<f32>) {
    let n = u16::try_from(values.len()).unwrap_or(u16::MAX);
    b.push(
        id,
        TensorSource::F32 {
            values,
            dims: vec![n],
        },
    );
}

/// Emits every tensor belonging to one transformer block.
fn push_layer(b: &mut ArtifactBuilder, rng: &mut SplitMix, layer: u16, d: u16, ff: u16) {
    let id = |s: u16| TensorId::layer(layer, s);
    push_vec(b, id(slot::LN1_GAMMA), vec![1.0; d as usize]);
    push_vec(b, id(slot::LN1_BETA), vec![0.0; d as usize]);
    for (w, s, bias) in [
        (slot::WQ, slot::WQ_SCALE, slot::WQ_BIAS),
        (slot::WK, slot::WK_SCALE, slot::WK_BIAS),
        (slot::WV, slot::WV_SCALE, slot::WV_BIAS),
        (slot::WO, slot::WO_SCALE, slot::WO_BIAS),
    ] {
        push_matrix(b, rng, id(w), id(s), d, d);
        push_vec(b, id(bias), vec![0.0; d as usize]);
    }
    push_vec(b, id(slot::LN2_GAMMA), vec![1.0; d as usize]);
    push_vec(b, id(slot::LN2_BETA), vec![0.0; d as usize]);
    push_matrix(b, rng, id(slot::FF1), id(slot::FF1_SCALE), ff, d);
    push_vec(b, id(slot::FF1_BIAS), vec![0.0; ff as usize]);
    push_matrix(b, rng, id(slot::FF2), id(slot::FF2_SCALE), d, ff);
    push_vec(b, id(slot::FF2_BIAS), vec![0.0; d as usize]);
}

/// Emits the token and position embedding tables.
fn push_embeddings(b: &mut ArtifactBuilder, rng: &mut SplitMix, config: ModelConfig) {
    let d = config.model_width;
    push_matrix(
        b,
        rng,
        TensorId::global(global::TOKEN_EMBEDDING),
        TensorId::global(global::TOKEN_EMBEDDING_SCALE),
        config.vocabulary_size,
        d,
    );
    push_matrix(
        b,
        rng,
        TensorId::global(global::POSITION_EMBEDDING),
        TensorId::global(global::POSITION_EMBEDDING_SCALE),
        config.context_length,
        d,
    );
}

/// Emits the final LayerNorm and every option-attention tensor.
fn push_option_attention(b: &mut ArtifactBuilder, rng: &mut SplitMix, config: ModelConfig) {
    let (d, rank) = (config.model_width, config.option_attention_rank);
    push_vec(
        b,
        TensorId::global(global::FINAL_LN_GAMMA),
        vec![1.0; d as usize],
    );
    push_vec(
        b,
        TensorId::global(global::FINAL_LN_BETA),
        vec![0.0; d as usize],
    );
    for (gamma, beta) in [
        (global::OPT_CONTEXT_LN_GAMMA, global::OPT_CONTEXT_LN_BETA),
        (global::OPT_OPTION_LN_GAMMA, global::OPT_OPTION_LN_BETA),
    ] {
        push_vec(b, TensorId::global(gamma), vec![1.0; d as usize]);
        push_vec(b, TensorId::global(beta), vec![0.0; d as usize]);
    }
    for (w, s) in [
        (global::OPT_WQ, global::OPT_WQ_SCALE),
        (global::OPT_WK, global::OPT_WK_SCALE),
        (global::OPT_WV, global::OPT_WV_SCALE),
    ] {
        push_matrix(b, rng, TensorId::global(w), TensorId::global(s), rank, d);
    }
}

/// Builds a complete, valid artifact with deterministic weights from `seed`.
///
/// # Errors
/// Propagates any structural error from the builder.
pub fn genesis_artifact(config: ModelConfig, seed: u64) -> Result<Vec<u8>, String> {
    let mut rng = SplitMix(seed);
    let mut b = ArtifactBuilder::new(config, 1.0);
    push_embeddings(&mut b, &mut rng, config);
    for layer in 0..config.transformer_layers {
        push_layer(
            &mut b,
            &mut rng,
            layer,
            config.model_width,
            config.feed_forward_width,
        );
    }
    push_option_attention(&mut b, &mut rng, config);
    b.build()
}
