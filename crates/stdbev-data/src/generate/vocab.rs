//! Surface vocabulary the generator draws from.
//!
//! Variety here is what stops the model latching onto a single phrasing. Entity names,
//! field order, irrelevant fields and decoy values all vary independently.

/// Names used for people and accounts.
pub const NAMES: &[&str] = &[
    "Alice", "Bob", "Carla", "Dmitri", "Priya", "Sven", "Wen", "Ravi", "Noor", "Tomas",
];

/// Irrelevant fields mixed into state to punish keyword matching.
pub const NOISE_FIELDS: &[&str] = &[
    "region",
    "tier",
    "locale",
    "shard",
    "client_version",
    "session_age",
    "device",
];

/// Values for the noise fields above.
pub const NOISE_VALUES: &[&str] = &[
    "eu-west", "gold", "en-GB", "7", "3.14.0", "412s", "ios", "us-east", "free", "de-DE",
];

/// Support-ticket surface forms, one per template family.
pub const TICKET_PHRASINGS: &[(&str, &str)] = &[
    ("ticket-plain", "{name} reports: {body}"),
    ("ticket-quoted", "Customer wrote: \"{body}\""),
    ("ticket-log", "support_note={body} reporter={name}"),
    ("ticket-email", "From: {name}\nSubject: help\n\n{body}"),
    ("ticket-chat", "[{name}] {body}"),
    ("ticket-terse", "{body}"),
    (
        "ticket-transcript",
        "Agent: how can I help?\n{name}: {body}",
    ),
    ("ticket-form", "name: {name}\nissue: {body}"),
    (
        "ticket-escalation",
        "ESCALATED from tier 1. Original complaint from {name}: {body}",
    ),
    ("ticket-summary", "Summary of call with {name} -- {body}"),
    (
        "ticket-thirdperson",
        "The customer ({name}) says that {body}.",
    ),
    (
        "ticket-tagged",
        "<ticket><reporter>{name}</reporter><body>{body}</body></ticket>",
    ),
];

/// Bodies that unambiguously indicate a billing problem.
pub const BILLING_BODIES: &[&str] = &[
    "I was charged twice for the same month",
    "my invoice shows an amount I never agreed to",
    "the refund from last week still has not arrived",
    "my card was declined but the money left my account",
    "I want to cancel my subscription and get a partial refund",
];

/// Bodies that unambiguously indicate a technical problem.
pub const TECHNICAL_BODIES: &[&str] = &[
    "the app crashes every time I open the settings screen",
    "uploads fail with a timeout after about thirty seconds",
    "the page is completely blank after I log in",
    "sync has been stuck on 'pending' for two days",
    "I get an error 500 whenever I save a draft",
];

/// Bodies that belong to neither category.
pub const OTHER_BODIES: &[&str] = &[
    "do you have an office in Lisbon",
    "can I get a copy of your security whitepaper",
    "who should I speak to about a partnership",
    "is there a student discount",
    "please remove me from your mailing list",
];
