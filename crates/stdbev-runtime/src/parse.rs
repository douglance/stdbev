//! `STDBEV01` parsing.
//!
//! The parser is the only place untrusted bytes are interpreted, so it does no
//! unchecked indexing anywhere: every read goes through a bounds-checked helper and
//! every arithmetic step is overflow-checked. A malformed artifact must return an
//! error, never panic and never read out of bounds.

use stdbev_types::{DIRECTORY_ENTRY_BYTES, Dtype, FORMAT_VERSION, HEADER_BYTES, MAGIC, TensorId};

use crate::error::RuntimeError;

/// A tensor located inside the artifact byte slice.
#[derive(Debug, Clone, Copy)]
pub struct TensorEntry {
    pub id: TensorId,
    pub dtype: Dtype,
    pub dims: [u16; 4],
    pub rank: u8,
    data: (usize, usize),
}

impl TensorEntry {
    /// Total element count implied by the declared shape.
    #[must_use]
    pub fn element_count(&self) -> usize {
        self.dims
            .iter()
            .take(self.rank as usize)
            .map(|d| *d as usize)
            .product()
    }

    /// The tensor's raw bytes within `blob`.
    pub(crate) fn bytes<'a>(&self, blob: &'a [u8]) -> Result<&'a [u8], RuntimeError> {
        blob.get(self.data.0..self.data.0 + self.data.1)
            .ok_or(RuntimeError::Malformed("tensor data out of range"))
    }
}

fn u16_at(bytes: &[u8], offset: usize) -> Result<u16, RuntimeError> {
    let slice = bytes
        .get(offset..offset + 2)
        .ok_or(RuntimeError::Malformed("truncated: u16"))?;
    let arr: [u8; 2] = slice
        .try_into()
        .map_err(|_| RuntimeError::Malformed("u16"))?;
    Ok(u16::from_le_bytes(arr))
}

fn u32_at(bytes: &[u8], offset: usize) -> Result<u32, RuntimeError> {
    let slice = bytes
        .get(offset..offset + 4)
        .ok_or(RuntimeError::Malformed("truncated: u32"))?;
    let arr: [u8; 4] = slice
        .try_into()
        .map_err(|_| RuntimeError::Malformed("u32"))?;
    Ok(u32::from_le_bytes(arr))
}

/// Parsed header fields.
#[derive(Debug, Clone, Copy)]
pub struct Header {
    pub model_width: u16,
    pub transformer_layers: u16,
    pub attention_heads: u16,
    pub feed_forward_width: u16,
    pub option_attention_rank: u16,
    pub context_length: u16,
    pub option_length: u16,
    pub vocabulary_size: u16,
    pub tensor_count: u16,
    pub temperature: f32,
    pub option_encoder_tag: u8,
    pub option_format_tag: u8,
}

/// Parses and validates the fixed header.
///
/// # Errors
/// Returns [`RuntimeError::Malformed`] on a bad magic, unsupported version, or a
/// temperature that is not finite and positive -- a zero or NaN temperature would
/// turn every probability into NaN downstream.
pub fn parse_header(blob: &[u8]) -> Result<Header, RuntimeError> {
    let magic = blob
        .get(0..8)
        .ok_or(RuntimeError::Malformed("truncated: header"))?;
    if magic != MAGIC {
        return Err(RuntimeError::Malformed("bad magic"));
    }
    if u16_at(blob, 8)? != FORMAT_VERSION {
        return Err(RuntimeError::Malformed("unsupported format version"));
    }
    if blob.len() < HEADER_BYTES {
        return Err(RuntimeError::Malformed("truncated: header"));
    }
    let temperature = f32::from_bits(u32_at(blob, 28)?);
    if !temperature.is_finite() || temperature <= 0.0 {
        return Err(RuntimeError::Malformed(
            "temperature must be finite and positive",
        ));
    }
    Ok(Header {
        model_width: u16_at(blob, 10)?,
        transformer_layers: u16_at(blob, 12)?,
        attention_heads: u16_at(blob, 14)?,
        feed_forward_width: u16_at(blob, 16)?,
        option_attention_rank: u16_at(blob, 18)?,
        context_length: u16_at(blob, 20)?,
        option_length: u16_at(blob, 22)?,
        vocabulary_size: u16_at(blob, 24)?,
        tensor_count: u16_at(blob, 26)?,
        temperature,
        option_encoder_tag: blob.get(64).copied().unwrap_or(0),
        option_format_tag: blob.get(65).copied().unwrap_or(0),
    })
}

/// Parses the tensor directory that follows the header.
///
/// # Errors
/// Returns [`RuntimeError::Malformed`] on an unknown dtype, a duplicate id, a shape
/// that disagrees with the stored byte length, or any offset that escapes the blob.
pub fn parse_directory(blob: &[u8], count: u16) -> Result<Vec<TensorEntry>, RuntimeError> {
    let mut entries: Vec<TensorEntry> = Vec::with_capacity(count as usize);
    for i in 0..count as usize {
        let base = HEADER_BYTES
            .checked_add(i.checked_mul(DIRECTORY_ENTRY_BYTES).ok_or(OVERFLOW)?)
            .ok_or(OVERFLOW)?;
        let id = TensorId(u16_at(blob, base)?);
        let dtype = blob
            .get(base + 2)
            .copied()
            .and_then(Dtype::from_tag)
            .ok_or(RuntimeError::Malformed("unknown dtype"))?;
        let rank = blob.get(base + 3).copied().unwrap_or(0);
        if rank == 0 || rank > 4 {
            return Err(RuntimeError::Malformed("rank must be 1..=4"));
        }
        let mut dims = [0u16; 4];
        for (d, slot) in dims.iter_mut().enumerate() {
            *slot = u16_at(blob, base + 4 + d * 2)?;
        }
        let offset = u32_at(blob, base + 12)? as usize;
        let length = u32_at(blob, base + 16)? as usize;
        let end = offset.checked_add(length).ok_or(OVERFLOW)?;
        if end > blob.len() {
            return Err(RuntimeError::Malformed("tensor data out of range"));
        }
        let entry = TensorEntry {
            id,
            dtype,
            dims,
            rank,
            data: (offset, length),
        };
        let expected = entry
            .element_count()
            .checked_mul(dtype.width())
            .ok_or(OVERFLOW)?;
        if expected != length {
            return Err(RuntimeError::Malformed(
                "tensor length disagrees with shape",
            ));
        }
        if entries.iter().any(|e| e.id == id) {
            return Err(RuntimeError::Malformed("duplicate tensor id"));
        }
        entries.push(entry);
    }
    Ok(entries)
}

const OVERFLOW: RuntimeError = RuntimeError::Malformed("offset arithmetic overflow");
