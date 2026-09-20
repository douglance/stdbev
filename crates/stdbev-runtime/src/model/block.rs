//! One pre-LayerNorm transformer block.

use stdbev_format::Tokens;
use stdbev_types::slot;

use crate::encoder::{AttentionArgs, LN1, LN2, apply, attend, feed_forward, linear, norm_layer};
use crate::error::RuntimeError;
use crate::model::Encoder;

/// Per-block working buffers, allocated once per encode rather than per token.
struct Buffers {
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    normed: Vec<f32>,
    attended: Vec<f32>,
    projected: Vec<f32>,
}

impl Buffers {
    fn new(seq: usize, width: usize) -> Self {
        Self {
            q: vec![0.0; seq * width],
            k: vec![0.0; seq * width],
            v: vec![0.0; seq * width],
            normed: vec![0.0; width],
            attended: vec![0.0; width],
            projected: vec![0.0; width],
        }
    }
}

/// Borrowed inputs for one token's attention.
struct Heads<'a> {
    keys: &'a [f32],
    values: &'a [f32],
    mask: &'a [bool],
    query: &'a [f32],
}

/// Dimensions for one token's attention.
#[derive(Clone, Copy)]
struct HeadShape {
    count: usize,
    head_width: usize,
    width: usize,
    seq: usize,
}

/// Runs every attention head for one token, accumulating into `attended`.
fn run_heads(attended: &mut [f32], scores: &mut [f32], h: &Heads<'_>, shape: HeadShape) {
    for head in 0..shape.count {
        let args = AttentionArgs {
            q: h.query,
            k: h.keys,
            v: h.values,
            mask: h.mask,
            head,
            head_width: shape.head_width,
            width: shape.width,
            seq: shape.seq,
        };
        attend(attended, scores, &args);
    }
}

/// Adds `delta` into `hidden` at token `t`.
fn residual(hidden: &mut [f32], t: usize, width: usize, delta: &[f32]) {
    let row = hidden
        .get_mut(t * width..(t + 1) * width)
        .unwrap_or(&mut []);
    for (cell, d) in row.iter_mut().zip(delta) {
        *cell += d;
    }
}

impl Encoder<'_> {
    /// Projects every token to Q, K and V after the first LayerNorm.
    fn project_qkv(
        &self,
        b: &mut Buffers,
        hidden: &[f32],
        layer: u16,
        seq: usize,
    ) -> Result<(), RuntimeError> {
        let w = self.shape.width;
        let wq = linear(
            self.tensors,
            layer,
            slot::WQ,
            slot::WQ_SCALE,
            Some(slot::WQ_BIAS),
        )?;
        let wk = linear(
            self.tensors,
            layer,
            slot::WK,
            slot::WK_SCALE,
            Some(slot::WK_BIAS),
        )?;
        let wv = linear(
            self.tensors,
            layer,
            slot::WV,
            slot::WV_SCALE,
            Some(slot::WV_BIAS),
        )?;
        for t in 0..seq {
            let src = hidden.get(t * w..(t + 1) * w).unwrap_or(&[]);
            norm_layer(&mut b.normed, src, self.tensors, layer, LN1.0, LN1.1)?;
            apply(
                b.q.get_mut(t * w..(t + 1) * w).unwrap_or(&mut []),
                &b.normed,
                &wq,
                w,
            );
            apply(
                b.k.get_mut(t * w..(t + 1) * w).unwrap_or(&mut []),
                &b.normed,
                &wk,
                w,
            );
            apply(
                b.v.get_mut(t * w..(t + 1) * w).unwrap_or(&mut []),
                &b.normed,
                &wv,
                w,
            );
        }
        Ok(())
    }

    /// Multi-head attention plus output projection and residual.
    fn attention_sublayer(
        &self,
        hidden: &mut [f32],
        b: &mut Buffers,
        scores: &mut [f32],
        tokens: &Tokens,
        layer: u16,
    ) -> Result<(), RuntimeError> {
        let (w, hw, seq) = (self.shape.width, self.shape.head_width, tokens.ids.len());
        let wo = linear(
            self.tensors,
            layer,
            slot::WO,
            slot::WO_SCALE,
            Some(slot::WO_BIAS),
        )?;
        for t in 0..seq {
            let query = b.q.get(t * w..(t + 1) * w).unwrap_or(&[]);
            // Distinct struct fields, so these borrows coexist and the query needs no
            // copy -- that copy cost ~450 allocations per decision.
            run_heads(
                &mut b.attended,
                scores,
                &Heads {
                    keys: &b.k,
                    values: &b.v,
                    mask: &tokens.mask,
                    query,
                },
                HeadShape {
                    count: self.shape.heads,
                    head_width: hw,
                    width: w,
                    seq,
                },
            );
            apply(&mut b.projected, &b.attended, &wo, w);
            residual(hidden, t, w, &b.projected);
        }
        Ok(())
    }

    /// Feed-forward sublayer with its own pre-norm and residual.
    fn ffn_sublayer(
        &self,
        hidden: &mut [f32],
        b: &mut Buffers,
        wide: &mut [f32],
        layer: u16,
        seq: usize,
    ) -> Result<(), RuntimeError> {
        let w = self.shape.width;
        let ff1 = linear(
            self.tensors,
            layer,
            slot::FF1,
            slot::FF1_SCALE,
            Some(slot::FF1_BIAS),
        )?;
        let ff2 = linear(
            self.tensors,
            layer,
            slot::FF2,
            slot::FF2_SCALE,
            Some(slot::FF2_BIAS),
        )?;
        let mut out = vec![0.0f32; w];
        for t in 0..seq {
            let src = hidden.get(t * w..(t + 1) * w).unwrap_or(&[]);
            norm_layer(&mut b.normed, src, self.tensors, layer, LN2.0, LN2.1)?;
            feed_forward(
                &mut out,
                wide,
                &b.normed,
                &ff1,
                &ff2,
                w,
                self.shape.ff_width,
            );
            residual(hidden, t, w, &out);
        }
        Ok(())
    }

    /// Runs the full encoder stack over the context.
    pub(crate) fn encode_sequence(
        &self,
        hidden: &mut [f32],
        wide: &mut [f32],
        scores: &mut [f32],
        tokens: &Tokens,
        layers: u16,
    ) -> Result<(), RuntimeError> {
        self.embed(hidden, tokens)?;
        self.run_layers(hidden, wide, scores, tokens, layers)
    }

    /// Runs the encoder stack over an option.
    ///
    /// Position is dropped when there are no layers and kept when there are, and the
    /// asymmetry is principled: mean pooling discards order, so a positional term adds
    /// only a length-dependent constant shared by options of similar length. A
    /// transformer reads order, so it needs one.
    pub(crate) fn encode_option(
        &self,
        hidden: &mut [f32],
        wide: &mut [f32],
        scores: &mut [f32],
        tokens: &Tokens,
        layers: u16,
    ) -> Result<(), RuntimeError> {
        if layers == 0 {
            self.embed_option(hidden, tokens)?;
        } else {
            self.embed(hidden, tokens)?;
        }
        self.run_layers(hidden, wide, scores, tokens, layers)
    }

    fn run_layers(
        &self,
        hidden: &mut [f32],
        wide: &mut [f32],
        scores: &mut [f32],
        tokens: &Tokens,
        layers: u16,
    ) -> Result<(), RuntimeError> {
        let seq = tokens.ids.len();
        let mut buffers = Buffers::new(seq, self.shape.width);
        for layer in 0..layers {
            self.project_qkv(&mut buffers, hidden, layer, seq)?;
            self.attention_sublayer(hidden, &mut buffers, scores, tokens, layer)?;
            self.ffn_sublayer(hidden, &mut buffers, wide, layer, seq)?;
        }
        Ok(())
    }
}
