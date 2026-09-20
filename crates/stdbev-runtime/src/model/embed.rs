//! Token and position embedding lookup.

use stdbev_format::Tokens;
use stdbev_types::global;

use crate::error::RuntimeError;
use crate::model::Encoder;
use crate::view::find;

/// Dequantized embedding tables.
struct Tables<'a> {
    token: &'a [i8],
    token_scale: &'a [f32],
    position: &'a [i8],
    position_scale: &'a [f32],
}

/// Writes one token's embedding row, with or without the position term.
///
/// Options deliberately omit position. Mean-pooling `token + position` leaves a term
/// that depends only on the option's length, so options of similar length share a large
/// constant component carrying no information about which option they are. Measured on
/// the test corpus, dropping it takes the mean cosine between pooled Noul options from
/// 0.38 to 0.04. The reference implementation does the same.
fn embed_one(
    out: &mut [f32],
    t: &Tables<'_>,
    token_id: usize,
    pos: usize,
    width: usize,
    with_position: bool,
) {
    let ts = t.token_scale.get(token_id).copied().unwrap_or(0.0);
    let ps = if with_position {
        t.position_scale.get(pos).copied().unwrap_or(0.0)
    } else {
        0.0
    };
    for (i, cell) in out.iter_mut().enumerate().take(width) {
        let tv = t.token.get(token_id * width + i).copied().unwrap_or(0);
        let pv = t.position.get(pos * width + i).copied().unwrap_or(0);
        *cell = f32::from(tv) * ts + f32::from(pv) * ps;
    }
}

impl Encoder<'_> {
    /// Fills `out` with the embedded token sequence.
    pub(crate) fn embed(&self, out: &mut [f32], tokens: &Tokens) -> Result<(), RuntimeError> {
        self.embed_with(out, tokens, true)
    }

    /// Fills `out` with token embeddings only, for the option path.
    pub(crate) fn embed_option(
        &self,
        out: &mut [f32],
        tokens: &Tokens,
    ) -> Result<(), RuntimeError> {
        self.embed_with(out, tokens, false)
    }

    fn embed_with(
        &self,
        out: &mut [f32],
        tokens: &Tokens,
        with_position: bool,
    ) -> Result<(), RuntimeError> {
        let width = self.shape.width;
        let tables = Tables {
            token: find(self.tensors, global::TOKEN_EMBEDDING)?.as_i8()?,
            token_scale: find(self.tensors, global::TOKEN_EMBEDDING_SCALE)?.as_f32()?,
            position: find(self.tensors, global::POSITION_EMBEDDING)?.as_i8()?,
            position_scale: find(self.tensors, global::POSITION_EMBEDDING_SCALE)?.as_f32()?,
        };
        for (t, id) in tokens.ids.iter().enumerate() {
            let row = out.get_mut(t * width..(t + 1) * width).unwrap_or(&mut []);
            embed_one(row, &tables, *id as usize, t, width, with_position);
        }
        Ok(())
    }
}
