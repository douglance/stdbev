//! `STDBEV01` artifact writer.

use sha2::{Digest, Sha256};
use stdbev_types::{
    DIRECTORY_ENTRY_BYTES, Dtype, FORMAT_VERSION, HEADER_BYTES, MAGIC, ModelConfig, OptionEncoder,
    OptionFormat, TensorId,
};

/// A tensor to be written.
pub enum TensorSource {
    /// Already-quantized weights.
    I8 { values: Vec<i8>, dims: Vec<u16> },
    /// Stored verbatim: biases, LayerNorm parameters, scales.
    F32 { values: Vec<f32>, dims: Vec<u16> },
}

impl TensorSource {
    fn dtype(&self) -> Dtype {
        match self {
            Self::I8 { .. } => Dtype::I8,
            Self::F32 { .. } => Dtype::F32,
        }
    }

    fn dims(&self) -> &[u16] {
        match self {
            Self::I8 { dims, .. } | Self::F32 { dims, .. } => dims,
        }
    }

    fn bytes(&self) -> Vec<u8> {
        match self {
            // Reinterpretation, not conversion: i8 and u8 have identical layout, and
            // the reader casts straight back.
            #[allow(clippy::cast_sign_loss)]
            Self::I8 { values, .. } => values.iter().map(|v| *v as u8).collect(),
            Self::F32 { values, .. } => values.iter().flat_map(|v| v.to_le_bytes()).collect(),
        }
    }
}

/// Accumulates tensors and emits a complete artifact.
pub struct ArtifactBuilder {
    config: ModelConfig,
    temperature: f32,
    tensors: Vec<(TensorId, TensorSource)>,
}

/// Fails if any tensor id appears twice.
fn reject_duplicates(sorted: &[&(TensorId, TensorSource)]) -> Result<(), String> {
    for pair in sorted.windows(2) {
        let [a, b] = pair else { continue };
        if a.0 == b.0 {
            return Err(format!("duplicate tensor id 0x{:04x}", a.0.0));
        }
    }
    Ok(())
}

/// Fails if the declared shape disagrees with the serialized length.
fn check_shape(id: TensorId, source: &TensorSource, actual: usize) -> Result<(), String> {
    let elements: usize = source.dims().iter().map(|d| *d as usize).product();
    let expected = elements * source.dtype().width();
    if expected == actual {
        return Ok(());
    }
    Err(format!("tensor 0x{:04x} shape disagrees with data", id.0))
}

/// Builds one 32-byte directory entry.
fn entry(
    id: TensorId,
    source: &TensorSource,
    offset: usize,
    length: usize,
) -> Result<Vec<u8>, String> {
    let dims = source.dims();
    let mut e = Vec::with_capacity(DIRECTORY_ENTRY_BYTES);
    e.extend_from_slice(&id.0.to_le_bytes());
    e.push(source.dtype() as u8);
    e.push(u8::try_from(dims.len()).map_err(|_| "rank > 255")?);
    for d in 0..4 {
        e.extend_from_slice(&dims.get(d).copied().unwrap_or(0).to_le_bytes());
    }
    e.extend_from_slice(
        &u32::try_from(offset)
            .map_err(|_| "offset > 4GiB")?
            .to_le_bytes(),
    );
    e.extend_from_slice(
        &u32::try_from(length)
            .map_err(|_| "len > 4GiB")?
            .to_le_bytes(),
    );
    // scale_offset / scale_length are unused: scales are ordinary F32 tensors with
    // their own ids, which keeps one lookup path instead of two.
    e.extend_from_slice(&0u32.to_le_bytes());
    e.extend_from_slice(&0u32.to_le_bytes());
    e.extend_from_slice(&0u32.to_le_bytes());
    Ok(e)
}

impl ArtifactBuilder {
    /// Starts an artifact for `config` with a fitted `temperature`.
    #[must_use]
    pub fn new(config: ModelConfig, temperature: f32) -> Self {
        Self {
            config,
            temperature,
            tensors: Vec::new(),
        }
    }

    /// Adds one tensor. Later duplicates of an id are rejected at build time.
    pub fn push(&mut self, id: TensorId, source: TensorSource) -> &mut Self {
        self.tensors.push((id, source));
        self
    }

    /// Serializes the artifact.
    ///
    /// `model_id` is the SHA-256 of the finished artifact with the id field itself
    /// zeroed. Hashing the *output* rather than the input checkpoint is deliberate:
    /// calibration temperature is fitted after training, so two artifacts from one
    /// checkpoint can give different answers. Hashing the checkpoint would give them
    /// the same id and quietly defeat the native/WASM parity check, which compares ids
    /// to prove both sides ran the same model.
    ///
    /// # Errors
    /// Returns an error if a tensor id is duplicated or a shape disagrees with its data.
    pub fn build(&self) -> Result<Vec<u8>, String> {
        let mut sorted: Vec<&(TensorId, TensorSource)> = self.tensors.iter().collect();
        sorted.sort_by_key(|(id, _)| id.0);
        reject_duplicates(&sorted)?;

        let directory_bytes = sorted.len() * DIRECTORY_ENTRY_BYTES;
        let mut data_offset = HEADER_BYTES + directory_bytes;
        let mut directory = Vec::with_capacity(directory_bytes);
        let mut data = Vec::new();

        for (id, source) in &sorted {
            let payload = source.bytes();
            check_shape(*id, source, payload.len())?;
            directory.extend_from_slice(&entry(*id, source, data_offset, payload.len())?);
            data_offset += payload.len();
            data.extend_from_slice(&payload);
        }

        let count = u16::try_from(sorted.len()).map_err(|_| "too many tensors")?;
        let mut out = self.header(count);
        out.extend_from_slice(&directory);
        out.extend_from_slice(&data);

        let digest = Sha256::digest(&out);
        out.get_mut(32..64)
            .ok_or("header truncated")?
            .copy_from_slice(&digest);
        Ok(out)
    }

    fn header(&self, tensor_count: u16) -> Vec<u8> {
        let c = &self.config;
        let mut h = Vec::with_capacity(HEADER_BYTES);
        h.extend_from_slice(MAGIC);
        for v in [
            FORMAT_VERSION,
            c.model_width,
            c.transformer_layers,
            c.attention_heads,
            c.feed_forward_width,
            c.option_attention_rank,
            c.context_length,
            c.option_length,
            c.vocabulary_size,
            tensor_count,
        ] {
            h.extend_from_slice(&v.to_le_bytes());
        }
        h.extend_from_slice(&self.temperature.to_bits().to_le_bytes());
        h.extend_from_slice(&[0u8; 32]); // model_id, filled after hashing
        h.push(match c.option_encoder {
            OptionEncoder::Pooled => 0,
            OptionEncoder::Encoded => 1,
        });
        h.push(match c.option_format {
            OptionFormat::Verbose => 0,
            OptionFormat::Compact => 1,
        });
        h.resize(HEADER_BYTES, 0);
        h
    }
}
