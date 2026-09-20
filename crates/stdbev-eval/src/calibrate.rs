//! Temperature calibration.
//!
//! One scalar `T > 0` rescales the logits before the softmax: `softmax(logits / T)`.
//! `T > 1` softens an overconfident model, `T < 1` sharpens an underconfident one. It
//! cannot change which option wins, so accuracy is untouched and only the
//! probabilities move -- which is the whole point, because a caller thresholding on
//! 0.9 needs 0.9 to mean something.

/// Search bounds. A temperature outside this range indicates a broken model rather
/// than an unusual one, so the search refuses to chase it.
const MIN_TEMPERATURE: f32 = 0.05;
const MAX_TEMPERATURE: f32 = 10.0;
/// Stop when the bracket is narrower than this.
const PRECISION: f32 = 1e-5;
/// Golden-section ratio.
const INV_PHI: f32 = 0.618_034;

/// Logits and the true label for one example.
pub struct Logits {
    pub values: Vec<f32>,
    pub label: usize,
}

/// Mean negative log likelihood at a given temperature.
#[must_use]
pub fn nll_at(examples: &[Logits], temperature: f32) -> f32 {
    if examples.is_empty() || temperature <= 0.0 {
        return f32::INFINITY;
    }
    let mut total = 0.0f32;
    for example in examples {
        let mut scaled: Vec<f32> = example.values.iter().map(|v| v / temperature).collect();
        stdbev_math::softmax_inplace(&mut scaled);
        let p = scaled.get(example.label).copied().unwrap_or(0.0);
        total -= libm::logf(p.max(1e-9));
    }
    #[allow(clippy::cast_precision_loss)]
    let out = total / examples.len() as f32;
    out
}

/// True if `temperature` is hugging a search bound.
///
/// A fitted temperature at the ceiling means the optimum is outside the bracket, which
/// in practice means the model is confidently wrong and no rescaling will fix it.
/// Callers should treat this as a failed model rather than a calibrated one.
#[must_use]
pub fn hit_bound(temperature: f32) -> bool {
    temperature <= MIN_TEMPERATURE * 1.01 || temperature >= MAX_TEMPERATURE * 0.99
}

/// Finds the temperature minimizing validation NLL, by golden-section search.
///
/// NLL as a function of temperature is unimodal, so a derivative-free bracket search
/// is both sufficient and deterministic -- no learning rate, no seed, no chance of a
/// different answer on a rerun.
#[must_use]
pub fn fit(examples: &[Logits]) -> f32 {
    if examples.is_empty() {
        return 1.0;
    }
    let (mut low, mut high) = (MIN_TEMPERATURE, MAX_TEMPERATURE);
    let mut c = high - (high - low) * INV_PHI;
    let mut d = low + (high - low) * INV_PHI;
    let mut fc = nll_at(examples, c);
    let mut fd = nll_at(examples, d);

    while (high - low) > PRECISION {
        if fc < fd {
            high = d;
            d = c;
            fd = fc;
            c = high - (high - low) * INV_PHI;
            fc = nll_at(examples, c);
        } else {
            low = c;
            c = d;
            fc = fd;
            d = low + (high - low) * INV_PHI;
            fd = nll_at(examples, d);
        }
    }
    f32::midpoint(low, high)
}
