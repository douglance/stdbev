//! Repository automation.

mod ci;
mod data;
mod evaluate;
mod export;
mod genesis;
mod golden;
mod parity;
mod policy;
mod train;
mod wasm;

use std::process::ExitCode;

/// Second positional argument, parsed, or a default.
fn arg<T: std::str::FromStr>(n: usize, default: T) -> T {
    std::env::args()
        .nth(n)
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Optional positional argument.
fn opt_arg<T: std::str::FromStr>(n: usize) -> Option<T> {
    std::env::args().nth(n).and_then(|v| v.parse().ok())
}

/// Positional string argument, or a default.
fn str_arg(n: usize, default: &str) -> String {
    std::env::args()
        .nth(n)
        .unwrap_or_else(|| default.to_owned())
}

/// Dispatches one task.
fn dispatch(task: &str) -> Result<(), String> {
    match task {
        "ci" => ci::hermetic(),
        "ci-full" => ci::full(&str_arg(2, "local"), &str_arg(3, "stdbev-dev")),
        "data" => data::run(arg(2, 30_000)),
        "train" => train::run(arg(2, 20), opt_arg(3)),
        "export" => export::run(arg(2, 1.0), arg(3, 400)),
        "eval" => evaluate::run(opt_arg(2)),
        "genesis" => genesis::run(),
        "golden" => golden::run(),
        "golden-check" => golden::check(),
        "source-policy" => policy::run(),
        "wasm-size" => wasm::check_size(),
        "parity" => parity::run(&str_arg(2, "local"), &str_arg(3, "stdbev-dev")),
        other => Err(format!("unknown task {other:?}")),
    }
}

fn main() -> ExitCode {
    let task = std::env::args().nth(1).unwrap_or_default();
    if task.is_empty() || task == "help" {
        eprintln!(
            "usage: cargo xtask <ci|ci-full|data|train|eval|export|genesis|golden|\
             golden-check|source-policy|wasm-size|parity>"
        );
        return ExitCode::FAILURE;
    }
    match dispatch(&task) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("xtask: {e}");
            ExitCode::FAILURE
        }
    }
}
