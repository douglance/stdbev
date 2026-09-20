//! The `STDBEV01` on-disk format, defined once and shared by the exporter and the
//! runtime so they cannot drift.
//!
//! All multi-byte integers and floats are little-endian.
//!
//! ## Header, 128 bytes
//!
//! ```text
//! offset size field
//! 0      8    ASCII "STDBEV01"
//! 8      2    format_version = 1
//! 10     2    model_width
//! 12     2    transformer_layers
//! 14     2    attention_heads
//! 16     2    feed_forward_width
//! 18     2    option_attention_rank
//! 20     2    context_length
//! 22     2    option_length
//! 24     2    vocabulary_size
//! 26     2    tensor_count
//! 28     4    temperature (f32)
//! 32     32   model_id (SHA-256)
//! 64     1    option_encoder (0 = pooled, 1 = encoded)
//! 65     1    option_format  (0 = verbose, 1 = compact)
//! 66     62   reserved, must be zero
//! ```
//!
//! The spec called for a 64-byte header. This is 128 because `option_encoder` had to
//! be carried (Phase 0 made it the dominant cost lever, so the runtime must know
//! which network it is executing), and because a 128-byte header leaves room for
//! future config fields without a format break -- and lands tensor data on a
//! cache-line boundary.

/// Magic bytes at offset 0.
pub const MAGIC: &[u8; 8] = b"STDBEV01";
/// Current format version.
pub const FORMAT_VERSION: u16 = 1;
/// Size of the fixed header.
pub const HEADER_BYTES: usize = 128;
/// Size of one tensor directory entry.
pub const DIRECTORY_ENTRY_BYTES: usize = 32;

/// Element type of a stored tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Dtype {
    /// Quantized weight, paired with a per-output-channel f32 scale.
    I8 = 1,
    /// Stored directly: biases, LayerNorm parameters, scales.
    F32 = 2,
}

impl Dtype {
    /// Parses a dtype tag.
    #[must_use]
    pub const fn from_tag(tag: u8) -> Option<Self> {
        match tag {
            1 => Some(Self::I8),
            2 => Some(Self::F32),
            _ => None,
        }
    }

    /// Bytes per element.
    #[must_use]
    pub const fn width(self) -> usize {
        match self {
            Self::I8 => 1,
            Self::F32 => 4,
        }
    }
}

/// Identifies one tensor inside an artifact.
///
/// Per-layer tensors encode the layer index in the id, so the set of ids is derived
/// from the config rather than enumerated by hand.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct TensorId(pub u16);

/// Slots within a transformer block.
pub mod slot {
    /// Base id for per-layer tensors; layer `n` occupies `LAYER_BASE + n * LAYER_STRIDE ..`.
    pub const LAYER_BASE: u16 = 0x1000;
    /// Id space reserved per layer.
    pub const LAYER_STRIDE: u16 = 0x0100;

    pub const LN1_GAMMA: u16 = 0;
    pub const LN1_BETA: u16 = 1;
    pub const WQ: u16 = 2;
    pub const WQ_SCALE: u16 = 3;
    pub const WQ_BIAS: u16 = 4;
    pub const WK: u16 = 5;
    pub const WK_SCALE: u16 = 6;
    pub const WK_BIAS: u16 = 7;
    pub const WV: u16 = 8;
    pub const WV_SCALE: u16 = 9;
    pub const WV_BIAS: u16 = 10;
    pub const WO: u16 = 11;
    pub const WO_SCALE: u16 = 12;
    pub const WO_BIAS: u16 = 13;
    pub const LN2_GAMMA: u16 = 14;
    pub const LN2_BETA: u16 = 15;
    pub const FF1: u16 = 16;
    pub const FF1_SCALE: u16 = 17;
    pub const FF1_BIAS: u16 = 18;
    pub const FF2: u16 = 19;
    pub const FF2_SCALE: u16 = 20;
    pub const FF2_BIAS: u16 = 21;
}

/// Ids for tensors that exist once per model.
pub mod global {
    pub const TOKEN_EMBEDDING: u16 = 0x0001;
    pub const TOKEN_EMBEDDING_SCALE: u16 = 0x0002;
    pub const POSITION_EMBEDDING: u16 = 0x0003;
    pub const POSITION_EMBEDDING_SCALE: u16 = 0x0004;
    pub const FINAL_LN_GAMMA: u16 = 0x0005;
    pub const FINAL_LN_BETA: u16 = 0x0006;

    pub const OPT_CONTEXT_LN_GAMMA: u16 = 0x0020;
    pub const OPT_CONTEXT_LN_BETA: u16 = 0x0021;
    pub const OPT_OPTION_LN_GAMMA: u16 = 0x0022;
    pub const OPT_OPTION_LN_BETA: u16 = 0x0023;
    pub const OPT_WQ: u16 = 0x0024;
    pub const OPT_WQ_SCALE: u16 = 0x0025;
    pub const OPT_WK: u16 = 0x0026;
    pub const OPT_WK_SCALE: u16 = 0x0027;
    pub const OPT_WV: u16 = 0x0028;
    pub const OPT_WV_SCALE: u16 = 0x0029;
}

impl TensorId {
    /// Id of `slot` within transformer layer `layer`.
    #[must_use]
    pub const fn layer(layer: u16, slot: u16) -> Self {
        Self(slot::LAYER_BASE + layer * slot::LAYER_STRIDE + slot)
    }

    /// Id of a once-per-model tensor.
    #[must_use]
    pub const fn global(id: u16) -> Self {
        Self(id)
    }
}
