//! Score examples: ordered severity from numeric state.
//!
//! The ladder is derived from a number, so the target is exact. Adjacent levels get a
//! small share of the mass near a boundary, which teaches the model that severity is
//! ordered rather than categorical -- a hard 1.0 at a boundary would teach it that
//! 0.99 and 0.01 of the way into a band are equally "high".

use crate::example::TrainingExample;
use crate::generate::rng::Rng;
use stdbev_types::OptionFormat;

const LEVELS: &[(&str, &str)] = &[
    ("low", "Minor degradation, few users affected"),
    (
        "medium",
        "Material degradation, a noticeable share of users affected",
    ),
    ("high", "Major outage, most users affected"),
    ("critical", "System-wide emergency, everything is down"),
];

/// Which rung of a `levels`-rung ladder a rate falls on.
///
/// The thresholds rescale with the ladder, which is both the right semantics and the
/// fix for a severe artifact. Rating against the levels you are *given* means the ladder
/// spans the range: offered only `[low, high]`, the boundary belongs at the midpoint.
///
/// The previous version used fixed thresholds and clamped, `band(rate).min(levels - 1)`,
/// so every rate above the last fixed threshold collapsed onto the final option. That
/// made the last option correct **95%** of the time at two levels and **70%** overall --
/// a shortcut scoring 0.70 on nearly half the corpus with no need to read the state, and
/// exactly what an untrained model was exploiting to beat every fixed baseline.
fn band(rate: f32, levels: usize) -> usize {
    if levels == 0 {
        return 0;
    }
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    let idx = (rate * levels as f32) as usize;
    idx.min(levels - 1)
}

/// Moves `blur` of the mass onto the adjacent rung, or keeps it if there is none.
///
/// At the outer edge of the ladder there is nowhere to spread to, so the rung keeps the
/// full mass rather than losing it -- otherwise the distribution would not sum to 1.
fn spread(out: &mut [f32], idx: usize, levels: usize, blur: f32, downward: bool) {
    let neighbour = if downward {
        idx.checked_sub(1)
    } else if idx + 1 < levels {
        Some(idx + 1)
    } else {
        None
    };
    // With no neighbour the mass returns to the rung it came from, so the distribution
    // still sums to 1.
    let target = neighbour.unwrap_or(idx);
    if let Some(slot) = out.get_mut(target) {
        *slot += blur;
    }
}

/// Spreads a little mass onto the neighbouring rung near a boundary.
fn targets(rate: f32, levels: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; levels];
    let idx = band(rate, levels);
    // Distance to the nearest rung boundary, as a fraction of a rung's width.
    #[allow(clippy::cast_precision_loss)]
    let width = 1.0 / levels as f32;
    #[allow(clippy::cast_precision_loss)]
    let lower = idx as f32 * width;
    let nearest = (rate - lower).min(lower + width - rate).max(0.0);
    let blur = (width * 0.15 - nearest.min(width * 0.15)) / (width * 0.15) * 0.3;
    if let Some(slot) = out.get_mut(idx) {
        *slot = 1.0 - blur;
    }
    if blur > 0.0 {
        spread(&mut out, idx, levels, blur, rate < lower + width * 0.5);
    }
    let sum: f32 = out.iter().sum();
    if sum > 0.0 {
        for v in &mut out {
            *v /= sum;
        }
    }
    out
}

/// Builds one severity example.
pub fn generate(rng: &mut Rng, index: usize, format: OptionFormat) -> TrainingExample {
    #[allow(clippy::cast_precision_loss)]
    let rate = rng.below(1000) as f32 / 1000.0;
    let users = rng.range(10, 50_000);
    let levels = rng.range(2, LEVELS.len());

    let phrasings = [
        "score-fields",
        "score-prose",
        "score-table",
        "score-alert",
        "score-report",
    ];
    let template = phrasings
        .get(rng.below(phrasings.len()))
        .copied()
        .unwrap_or("score-fields");
    let pct = rate * 100.0;
    let state = match template {
        "score-prose" => {
            format!("About {users} users are affected and roughly {pct:.0}% of requests fail.")
        }
        "score-table" => format!("| metric | value |\n| errors | {rate:.3} |\n| users | {users} |"),
        "score-alert" => {
            format!("ALERT error_rate={rate:.3} threshold_breached users_impacted={users}")
        }
        "score-report" => format!(
            "Incident report: request failure rate measured at {pct:.1} percent across {users} affected accounts."
        ),
        _ => format!("error_rate={rate:.3} affected_users={users}"),
    };

    TrainingExample {
        id: format!("score-{index:06}"),
        group: format!("{template}-{}", band(rate, levels)),
        template: template.to_owned(),
        context: format!("STATE\n{state}\nQUESTION\nHow severe is this incident?"),
        options: LEVELS
            .iter()
            .take(levels)
            .enumerate()
            .map(|(i, (label, description))| match format {
                OptionFormat::Verbose => {
                    format!("LEVEL\n{i}\nLABEL\n{label}\nDESCRIPTION\n{description}")
                }
                OptionFormat::Compact => (*label).to_owned(),
            })
            .collect(),
        target_probabilities: targets(rate, levels),
    }
}
