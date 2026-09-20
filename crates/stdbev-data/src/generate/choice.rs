//! Choice examples: route a support ticket.

use crate::example::TrainingExample;
use crate::generate::rng::Rng;
use crate::generate::vocab;
use stdbev_types::OptionFormat;

/// The three routing categories, with the descriptions the model sees.
const CATEGORIES: &[(&str, &str)] = &[
    ("billing", "Payments, invoices, charges, or refunds"),
    (
        "technical",
        "Software errors, crashes, or things not working",
    ),
    ("other", "Anything that is neither billing nor technical"),
];

/// Extra categories that are sometimes present, so option count varies.
const DISTRACTORS: &[(&str, &str)] = &[
    ("sales", "Pre-purchase questions about plans and pricing"),
    (
        "legal",
        "Contracts, compliance, and data protection requests",
    ),
    (
        "partnership",
        "Business development and integration proposals",
    ),
];

fn noise(rng: &mut Rng, count: usize) -> String {
    let mut parts = Vec::new();
    for _ in 0..count {
        let field = rng.pick(vocab::NOISE_FIELDS).copied().unwrap_or("region");
        let value = rng.pick(vocab::NOISE_VALUES).copied().unwrap_or("eu-west");
        parts.push(format!("{field}={value}"));
    }
    parts.join(" ")
}

/// Renders one option in the requested encoding.
fn render(label: &str, description: &str, format: OptionFormat) -> String {
    match format {
        OptionFormat::Verbose => format!("LABEL\n{label}\nDESCRIPTION\n{description}"),
        OptionFormat::Compact => label.to_owned(),
    }
}

/// Builds one routing example.
///
/// The correct category is chosen first, then a body is drawn from that category's
/// pool -- so the label is ground truth by construction, not by an annotator.
pub fn generate(rng: &mut Rng, index: usize, format: OptionFormat) -> TrainingExample {
    let (template, pattern) = rng
        .pick(vocab::TICKET_PHRASINGS)
        .copied()
        .unwrap_or(("ticket-plain", "{name} reports: {body}"));
    let truth = rng.below(3);
    let bodies = match truth {
        0 => vocab::BILLING_BODIES,
        1 => vocab::TECHNICAL_BODIES,
        _ => vocab::OTHER_BODIES,
    };
    let body = rng.pick(bodies).copied().unwrap_or("something is wrong");
    let name = rng.pick(vocab::NAMES).copied().unwrap_or("Alice");

    let rendered = pattern.replace("{name}", name).replace("{body}", body);
    let noise_count = rng.range(0, 3);
    let extra = noise(rng, noise_count);
    let state = if extra.is_empty() {
        rendered
    } else if rng.chance(1, 2) {
        format!("{rendered}\n{extra}")
    } else {
        format!("{extra}\n{rendered}")
    };

    // Always include the three real categories; sometimes add distractors so option
    // count varies between 3 and 6 and the model cannot assume a fixed menu.
    let mut options: Vec<(&str, &str)> = CATEGORIES.to_vec();
    let extras = rng.range(0, DISTRACTORS.len());
    for d in DISTRACTORS.iter().take(extras) {
        options.push(*d);
    }
    rng.shuffle(&mut options);

    let correct_label = CATEGORIES.get(truth).map_or("other", |c| c.0);
    let correct_index = options
        .iter()
        .position(|(label, _)| *label == correct_label)
        .unwrap_or(0);

    let mut targets = vec![0.0f32; options.len()];
    if let Some(slot) = targets.get_mut(correct_index) {
        *slot = 1.0;
    }

    TrainingExample {
        id: format!("choice-{index:06}"),
        group: format!("{template}-{body}"),
        template: template.to_owned(),
        context: format!("STATE\n{state}\nQUESTION\nWhat type of request is this?"),
        options: options.iter().map(|(l, d)| render(l, d, format)).collect(),
        target_probabilities: targets,
    }
}
