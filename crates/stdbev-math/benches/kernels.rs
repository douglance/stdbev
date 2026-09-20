//! Phase 0 measurement: how fast is the INT8 hot loop, really?
//!
//! The plan replaces the spec's invented "<=50 ms p95" with a number measured here.
//! `matvec_i8` carries ~86% of a decision's arithmetic, so its throughput sets the
//! whole latency budget.

// Test-file lint posture. The workspace denies these because a panic inside a reducer
// aborts a transaction -- but in a test a panic IS the failure signal, and an exact
// float comparison is frequently the assertion itself.
#![allow(
    clippy::indexing_slicing,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::format_collect,
    clippy::excessive_nesting
)]

use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;

const WIDTH: usize = 128;
const FF: usize = 512;

fn bench_matvec_i8(c: &mut Criterion) {
    let mut group = c.benchmark_group("matvec_i8");
    for (rows, cols, label) in [
        (WIDTH, WIDTH, "128x128"),
        (FF, WIDTH, "512x128"),
        (WIDTH, FF, "128x512"),
    ] {
        let weights = vec![7i8; rows * cols];
        let scale = vec![0.01f32; rows];
        let bias = vec![0.1f32; rows];
        let input = vec![0.5f32; cols];
        let mut out = vec![0f32; rows];
        // FLOPs = 2 * rows * cols (one multiply + one add per weight).
        group.throughput(criterion::Throughput::Elements((2 * rows * cols) as u64));
        group.bench_function(label, |b| {
            b.iter(|| {
                stdbev_math::matvec_i8(
                    black_box(&mut out),
                    black_box(&input),
                    black_box(&weights),
                    black_box(&scale),
                    Some(black_box(&bias)),
                    cols,
                );
            });
        });
    }
    group.finish();
}

fn bench_softmax(c: &mut Criterion) {
    let mut values = vec![0.3f32; 224];
    c.bench_function("softmax_224", |b| {
        b.iter(|| {
            let mut v = values.clone();
            stdbev_math::softmax_inplace(black_box(&mut v));
            values[0] = v[0];
        });
    });
}

fn bench_gelu(c: &mut Criterion) {
    let mut values = vec![0.3f32; FF];
    c.bench_function("gelu_512", |b| {
        b.iter(|| stdbev_math::gelu_inplace(black_box(&mut values)));
    });
}

criterion_group!(benches, bench_matvec_i8, bench_softmax, bench_gelu);
criterion_main!(benches);
