//! Candle training for STDBEV.
//!
//! The trained checkpoint is the *only* thing this crate produces; the runtime that
//! executes it shares no code with Candle. That separation is deliberate -- it means
//! the trainer can be replaced without touching anything that ships -- and it is why
//! the golden parity test comparing Candle FP32 against the native FP32 and INT8
//! implementations is not optional.

pub mod batch;
pub mod evaluate;
pub mod export;
pub mod model;
pub mod train;

/// Deterministic shuffling, written out so a seed reproduces a run exactly.
pub struct Shuffler(u64);

impl Shuffler {
    /// Seeds the shuffler.
    #[must_use]
    pub const fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Fisher-Yates.
    pub fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            let j = usize::try_from(self.next_u64() % (i as u64 + 1)).unwrap_or(0);
            items.swap(i, j);
        }
    }
}
