//! Noul examples: deterministic predicates over structured state.
//!
//! Every answer is computable from the state by a rule, so the label is ground truth.
//! Decoy fields carry values that would flip the answer if the model attended to the
//! wrong key -- which is the failure this family is designed to punish.

use crate::example::TrainingExample;
use crate::generate::rng::Rng;
use crate::generate::vocab;
use stdbev_types::OptionFormat;

/// One predicate family: how to render the question, and how to decide it.
struct Predicate {
    template: &'static str,
    question: &'static str,
    key: &'static str,
    decoy: &'static str,
    yes: &'static str,
    no: &'static str,
    false_text: &'static str,
    true_text: &'static str,
}

const PREDICATES: &[Predicate] = &[
    Predicate {
        template: "noul-suspended",
        question: "Is the account suspended?",
        key: "account_status",
        decoy: "billing_status",
        yes: "suspended",
        no: "active",
        false_text: "The account is not suspended",
        true_text: "The account is suspended",
    },
    Predicate {
        template: "noul-transfer",
        question: "Did the transfer complete?",
        key: "transfer_state",
        decoy: "sync_state",
        yes: "complete",
        no: "pending",
        false_text: "The transfer did not complete",
        true_text: "The transfer completed",
    },
    Predicate {
        template: "noul-threshold",
        question: "Is the resource above its threshold?",
        key: "usage_state",
        decoy: "quota_state",
        yes: "over",
        no: "under",
        false_text: "The resource is within its threshold",
        true_text: "The resource is above its threshold",
    },
    Predicate {
        template: "noul-verified",
        question: "Has the email been verified?",
        key: "email_state",
        decoy: "phone_state",
        yes: "verified",
        no: "unverified",
        false_text: "The email has not been verified",
        true_text: "The email has been verified",
    },
    Predicate {
        template: "noul-trial",
        question: "Is the trial still running?",
        key: "trial_state",
        decoy: "contract_state",
        yes: "running",
        no: "expired",
        false_text: "The trial has ended",
        true_text: "The trial is still running",
    },
    Predicate {
        template: "noul-backup",
        question: "Did last night's backup succeed?",
        key: "backup_result",
        decoy: "restore_result",
        yes: "succeeded",
        no: "failed",
        false_text: "The backup did not succeed",
        true_text: "The backup succeeded",
    },
    Predicate {
        template: "noul-owner",
        question: "Is the object owned by Alice?",
        key: "owner",
        decoy: "last_editor",
        yes: "Alice",
        no: "Bob",
        false_text: "Alice does not own the object",
        true_text: "Alice owns the object",
    },
];

/// Minimal well-formed example, used only if the predicate table were ever emptied.
fn fallback(index: usize) -> TrainingExample {
    TrainingExample {
        id: format!("noul-{index:06}"),
        group: "noul-fallback".into(),
        template: "noul-fallback".into(),
        context: "STATE\nstate=unknown\nQUESTION\nIs this true?".into(),
        options: vec!["FALSE\nno".into(), "TRUE\nyes".into()],
        target_probabilities: vec![1.0, 0.0],
    }
}

/// Builds one predicate example.
pub fn generate(rng: &mut Rng, index: usize, format: OptionFormat) -> TrainingExample {
    let Some(p) = PREDICATES.get(rng.below(PREDICATES.len())) else {
        // PREDICATES is a non-empty const so this is unreachable, but returning a
        // well-formed example keeps the function total without an indexing panic.
        return fallback(index);
    };
    let truth = rng.chance(1, 2);
    let value = if truth { p.yes } else { p.no };
    // The decoy carries the OPPOSITE value, so reading the wrong key inverts the answer.
    let decoy_value = if truth { p.no } else { p.yes };

    let mut fields = vec![
        format!("{}={}", p.key, value),
        format!("{}={}", p.decoy, decoy_value),
    ];
    for _ in 0..rng.range(1, 3) {
        let field = rng.pick(vocab::NOISE_FIELDS).copied().unwrap_or("region");
        let noise_value = rng.pick(vocab::NOISE_VALUES).copied().unwrap_or("eu-west");
        fields.push(format!("{field}={noise_value}"));
    }
    rng.shuffle(&mut fields);

    // The group must cluster examples whose *text* is effectively the same, so that
    // near-duplicates cannot straddle a split. Grouping on (template, value) alone
    // gave only two groups per predicate, and when both hashed to validation the
    // template disappeared from training entirely -- which silently turned validation
    // into an out-of-distribution set for that predicate. Including the noise
    // signature keeps true duplicates together while giving enough groups to split.
    let signature = fields.join(",");
    TrainingExample {
        id: format!("noul-{index:06}"),
        group: format!("{}-{value}-{signature}", p.template),
        template: p.template.to_owned(),
        context: format!("STATE\n{}\nQUESTION\n{}", fields.join(" "), p.question),
        options: match format {
            OptionFormat::Verbose => vec![
                format!("FALSE\n{}", p.false_text),
                format!("TRUE\n{}", p.true_text),
            ],
            // Bare polarity markers. The verbose pair is a condition and its negation,
            // which pool to vectors 0.91 cosine apart -- indistinguishable. These pool
            // to 0.04. The question already reaches the model through the context.
            OptionFormat::Compact => vec!["no".to_owned(), "yes".to_owned()],
        },
        target_probabilities: if truth {
            vec![0.0, 1.0]
        } else {
            vec![1.0, 0.0]
        },
    }
}
