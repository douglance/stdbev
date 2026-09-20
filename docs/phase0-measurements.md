# Phase 0 — measured, not assumed

Every latency number in the acceptance table comes from here. The spec's original
`<= 50 ms p95` was invented; these replace it.

**Rig:** Apple M4 Pro (arm64), Rust 1.98.1, SpacetimeDB 2.10.1 (wasmtime host),
`wasm32-unknown-unknown`, `opt-level=3 lto=fat codegen-units=1`.
Date: 2026-09-20.

## Kernel throughput

`matvec_i8` carries ~86% of a decision's arithmetic, so its throughput sets the
entire latency budget.

| Target | GFLOP/s | vs native |
|---|---|---|
| Native (M4 Pro, NEON) | **26.5** | 1.00× |
| wasmtime + `simd128` | **7.00** | 0.26× |
| wasmtime, no SIMD | 4.39 | 0.17× |
| V8 / node 26 + `simd128` | 6.8 | — |
| V8 / node 26, no SIMD | 8.5 | — |

Measured in-reducer by slope: `bench(iters)` at 200 and 2200 iterations, best of 5,
differenced to cancel the ~41 ms `spacetime call` round-trip overhead.

### Three findings that changed the code

1. **A single accumulator runs at scalar speed.** The first kernel hit 2.3 GFLOP/s
   native. Float addition is not associative, so LLVM may not reassociate a
   one-accumulator reduction into SIMD lanes. Splitting into 8 independent lanes with
   a *fixed* pairwise reduction took it to 26.5 GFLOP/s — **11.5×** — while keeping
   summation order identical across targets, which is what makes parity exact.
2. **Per-element bounds checks cost ~4×.** Indexing `input.get(base + lane)` inside
   the inner loop kept it scalar. Pairing `chunks_exact` on *both* slices gives the
   inner loop a compile-time-known length and removes the checks entirely.
3. **`+simd128` is worth 1.6× in wasmtime — and is not on by default.** Without the
   flag, zero SIMD opcodes are emitted (verified with `wasm-opt --print`); with it,
   157. Note V8 shows the *opposite* sign; engine behavior does not transfer, which
   is why this was measured in the real host rather than node.

## Measured decision latency, in the reducer

Round-trip via `spacetime call`, best of 7, minus the 40.7 ms CLI process/connect
overhead measured separately.

| Request | first build | after two fixes |
|---|---|---|
| Noul (K=2) | 132 ms | **106 ms** |
| Score (K=10) | 172 ms | **118 ms** |
| Choice (K=16) | 232 ms | **114 ms** |

**Latency is now flat in option count**, which is the architectural claim
`OptionEncoder::Pooled` exists to make. It was *not* flat when first measured, and
finding out why produced the two fixes below.

### Two fixes the measurement forced

1. **Keys and values were recomputed per option.** They depend only on the context,
   so a 16-option Choice projected all 224 context tokens into key/value space 16
   times. Cost was linear in K at ~8.5 ms per option -- 130 ms of pure waste at K=16.
   Hoisting it out of the loop halved worst-case latency and is what made the curve
   flat. No test caught this, and none could: the answers were identical. Only the
   latency measurement could see it.
2. **~900 allocations per decision in the token loop.** Two `to_vec()` calls copied
   a 128-float row per token per layer, one of them needlessly. Split-borrowing the
   distinct struct fields removed both, worth ~20% on Noul.

Remaining headroom: ~110 ms against a kernel-only floor of ~33 ms. The gap is
per-token overhead in the context encode, which is the next place to look if the gate
needs to come down.

## Earlier projection, for comparison @ 7.00 GFLOP/s

| Request | `option_encoder_layers = 2` | `= 0` |
|---|---|---|
| Noul (2 options) | 59 ms | **35 ms** |
| Choice (4) | 83 ms | **35 ms** |
| Score (10) | 156 ms | **35 ms** |
| Choice (16, worst case) | 229 ms | **35 ms** |

**`option_encoder_layers = 0` makes latency independent of option count**, because
pooled options cost almost nothing next to the context encode. The 2-layer path
misses the original 50 ms target by 4.6×; the 0-layer path meets it with headroom.

This is jevlike's actual architecture (embedding + position + one attention head,
no transformer over options), which reports ~98% on synthetic menus. Phase 3 trains
both and ships the cheaper one that clears the eval gate.

## Module size

| | |
|---|---|
| Module with 1 MiB embedded blob | **1,150,878 B (1.10 MiB)** |
| Module overhead excluding blob | ~100 KB |
| Projected real module (536 KB artifact) | ~650 KB |
| Spec gate | 8 MiB |

Published successfully to a local server, so a ~1 MiB `include_bytes!` is fine in
practice. **No maximum module size is documented anywhere**; this establishes only
that 1.1 MiB works, not where the ceiling is.

## Execution budget

Not hit, and not close. SpacetimeDB's reducer budget is
`FunctionBudget::DEFAULT_BUDGET = 120_000_000_000_000` eV, annotated in host source
as *"roughly 1 minute of wasm runtime"*. A 35 ms inference is ~0.06% of it. There is
no wall-clock timeout; the 10 ms epoch tick logs and resumes rather than killing.

Undocumented — read from host source. Do not design against its exact value.

## Gate

```
p95 inference <= 180 ms     (measured worst case 118 ms x 1.5 regression guard)
```

Set from measurement, not aspiration. The spec's invented `<= 50 ms` was never
achievable with this architecture on this host: even a perfect implementation with
zero overhead sits at ~33 ms of pure arithmetic, and real per-token overhead triples
that. Quoting 50 ms would have meant either a permanently red gate or a quietly
ignored one.
Recorded with host CPU, SpacetimeDB version, option count and simd128 state, per the
spec's own requirement that regressions stay interpretable.
