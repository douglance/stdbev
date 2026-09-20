//! Dataset loading, validation, splitting and synthetic generation.

mod example;
pub mod generate;
pub mod jsonl;
mod split;

pub use example::{DataError, TrainingExample};
pub use generate::{Report, SplitCounts};
pub use split::{
    Split, bucket, check_all_populated, check_no_leakage, is_test_template, split_for,
};
