//! Evaluation metrics.

use serde::Serialize;

/// One scored example: the predicted distribution and the true label.
pub struct Scored {
    pub probabilities: Vec<f32>,
    pub label: usize,
}

impl Scored {
    fn predicted(&self) -> usize {
        stdbev_math::argmax(&self.probabilities).0
    }

    fn confidence(&self) -> f32 {
        self.probabilities.iter().copied().fold(0.0, f32::max)
    }

    fn rank_of_label(&self) -> usize {
        let target = self.probabilities.get(self.label).copied().unwrap_or(0.0);
        self.probabilities.iter().filter(|p| **p > target).count()
    }
}

/// Number of bins used for expected calibration error.
///
/// Pinned here because "ECE <= 0.05" is unverifiable without it: the same model scores
/// differently at 10 bins and at 20. The top edge is **inclusive**, unlike the
/// reference implementation, whose half-open `[0.9, 1.0)` top bin silently drops every
/// prediction with confidence exactly 1.0 -- which a confident quantized model produces
/// in quantity.
pub const ECE_BINS: usize = 15;

/// The full report.
#[derive(Debug, Clone, Serialize)]
pub struct Report {
    pub example_count: usize,
    pub top_1_accuracy: f32,
    pub top_3_accuracy: f32,
    pub negative_log_likelihood: f32,
    pub brier_score: f32,
    pub expected_calibration_error: f32,
    pub mean_confidence: f32,
}

/// Computes every metric over a scored set.
#[must_use]
pub fn evaluate(scored: &[Scored]) -> Report {
    let n = scored.len().max(1);
    #[allow(clippy::cast_precision_loss)]
    let count = n as f32;

    let top1 = scored.iter().filter(|s| s.predicted() == s.label).count();
    let top3 = scored.iter().filter(|s| s.rank_of_label() < 3).count();

    let mut nll = 0.0f32;
    let mut brier = 0.0f32;
    let mut confidence_sum = 0.0f32;
    for s in scored {
        let p = s.probabilities.get(s.label).copied().unwrap_or(0.0);
        // Floor the probability: a single confident miss would otherwise send NLL to
        // infinity and destroy the average.
        nll -= libm::logf(p.max(1e-9));
        for (i, q) in s.probabilities.iter().enumerate() {
            let target = f32::from(u8::from(i == s.label));
            brier += (q - target) * (q - target);
        }
        confidence_sum += s.confidence();
    }

    #[allow(clippy::cast_precision_loss)]
    let report = Report {
        example_count: scored.len(),
        top_1_accuracy: top1 as f32 / count,
        top_3_accuracy: top3 as f32 / count,
        negative_log_likelihood: nll / count,
        brier_score: brier / count,
        expected_calibration_error: expected_calibration_error(scored),
        mean_confidence: confidence_sum / count,
    };
    report
}

/// Expected calibration error over [`ECE_BINS`] equal-width confidence bins.
#[must_use]
pub fn expected_calibration_error(scored: &[Scored]) -> f32 {
    if scored.is_empty() {
        return 0.0;
    }
    #[allow(clippy::cast_precision_loss)]
    let total = scored.len() as f32;
    let mut ece = 0.0f32;
    for bin in 0..ECE_BINS {
        #[allow(clippy::cast_precision_loss)]
        let low = bin as f32 / ECE_BINS as f32;
        #[allow(clippy::cast_precision_loss)]
        let high = (bin + 1) as f32 / ECE_BINS as f32;
        let last = bin + 1 == ECE_BINS;
        let members: Vec<&Scored> = scored
            .iter()
            .filter(|s| {
                let c = s.confidence();
                c >= low && (c < high || (last && c <= high))
            })
            .collect();
        if members.is_empty() {
            continue;
        }
        #[allow(clippy::cast_precision_loss)]
        let size = members.len() as f32;
        let accuracy = members.iter().filter(|s| s.predicted() == s.label).count();
        #[allow(clippy::cast_precision_loss)]
        let gap =
            (accuracy as f32 / size) - members.iter().map(|s| s.confidence()).sum::<f32>() / size;
        ece += size / total * gap.abs();
    }
    ece
}
