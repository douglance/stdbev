//! Metrics, calibration and controls.

pub mod calibrate;
pub mod metrics;

pub use calibrate::{Logits, fit, hit_bound, nll_at};
pub use metrics::{ECE_BINS, Report, Scored, evaluate, expected_calibration_error};
