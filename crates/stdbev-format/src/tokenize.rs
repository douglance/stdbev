//! Raw-byte tokenization.
//!
//! There is no learned tokenizer and no vocabulary file to ship or version. The model
//! reads UTF-8 bytes directly, which also means it degrades gracefully on text in any
//! language rather than failing on an out-of-vocabulary token.

/// Padding token. Never produced by [`token_id`], so it is unambiguous.
pub const PAD: u16 = 0;

/// Maps a byte to its token id.
///
/// Ids are `1..=256`; id `0` is reserved for [`PAD`].
#[must_use]
pub const fn token_id(byte: u8) -> u16 {
    byte as u16 + 1
}

/// A padded token sequence with a validity mask.
///
/// `ids` and `mask` are always the same length. `mask[i]` is false exactly where
/// `ids[i] == PAD`, and masked positions contribute nothing to pooling or attention.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tokens {
    pub ids: Vec<u16>,
    pub mask: Vec<bool>,
}

impl Tokens {
    /// Number of non-padding tokens.
    #[must_use]
    pub fn valid_len(&self) -> usize {
        self.mask.iter().filter(|m| **m).count()
    }
}

/// Encodes `text` into exactly `window` tokens, truncating or padding as needed.
///
/// Truncation keeps the **first** `window` bytes. That is a real limitation worth
/// naming: a caller who puts the decisive fact at the end of a long state string will
/// lose it. The canonical formats in [`crate::canonical`] therefore lead with the
/// state and the instructions rather than trailing them.
///
/// Truncation is by byte, not by character, so a multi-byte character may be cut in
/// half. This is harmless here because the model consumes bytes, not characters.
#[must_use]
pub fn encode(text: &str, window: usize) -> Tokens {
    let mut ids = Vec::with_capacity(window);
    let mut mask = Vec::with_capacity(window);
    for byte in text.as_bytes().iter().take(window) {
        ids.push(token_id(*byte));
        mask.push(true);
    }
    while ids.len() < window {
        ids.push(PAD);
        mask.push(false);
    }
    Tokens { ids, mask }
}
