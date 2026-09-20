//! The embedded model artifact.

/// Forces 64-byte alignment on the embedded bytes.
///
/// `include_bytes!` promises only 1-byte alignment. The runtime borrows INT8 weights
/// in place, which needs no alignment to be *sound*, but aligning to a cache line
/// keeps quantized rows from straddling two lines in the hot loop.
#[repr(C, align(64))]
struct Aligned<T: ?Sized>(T);

static ALIGNED_MODEL: &Aligned<[u8; include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../artifacts/model.stdbq"
)).len()]> = &Aligned(*include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../artifacts/model.stdbq"
)));

/// The artifact this module was built with.
pub const fn model_bytes() -> &'static [u8] {
    &ALIGNED_MODEL.0
}

/// Convenience alias used by the reducers.
pub static MODEL_BYTES: &[u8] = model_bytes();
