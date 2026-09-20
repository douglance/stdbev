# STDBEV

A tiny typed decision model that runs **inside** a SpacetimeDB database.

When a program needs a judgment call — *is this ticket billing or a bug? how severe is
this? should we retry?* — it normally asks a language model over the network. That costs
money, takes a second, and can fail. STDBEV answers in the same transaction that asked,
from a 48 KB model compiled into the database's WASM module.

It stays small by never generating anything. It only picks from a list you hand it, and
returns a calibrated probability for each option.

| Ask | Means | Example |
|---|---|---|
| **Choice** | pick one from a menu | billing / technical / other |
| **Noul** | how likely is this true? | "retry?" → 0.83 |
| **Score** | rate on a ladder | low/medium/high/critical → 2.4 |

All three are the same machine underneath. Noul is a Choice between two options named
`no` and `yes`; Score is a Choice over ladder rungs whose expectation is then taken.

## Status, honestly

**The infrastructure works.** Train → quantize to INT8 → embed in WASM → run in
SpacetimeDB, with answers that are *bit-identical* to running natively.

**The model works inside its training distribution and is unsafe outside it.**

```
test top-1                0.948
expected calibration err  0.0485
shuffled-context control  0.337   (= chance; it genuinely reads state)
artifact                  47,940 bytes
native ↔ WASM agreement   exact — delta 0.000e0
```

But on unfamiliar input it is **confidently wrong**:

| state | answer | confidence |
|---|---|---|
| `[billing] zzz` | **technical** | **1.000** ❌ |
| `[technical] qqq www eee` | **billing** | **0.9996** ❌ |

The calibration figure is an in-distribution measurement. It does not hold for arbitrary
input, and a caller thresholding at 0.9 will get wrong answers at 1.0. Read
[`docs/what-this-architecture-can-learn.md`](docs/what-this-architecture-can-learn.md)
before using this for anything.

## What this architecture can and cannot learn

The most useful result here. Identical code, identical architecture, only the task
differs:

| task | example | untrained | trained |
|---|---|---|---|
| **lexical** | `"[billing] I was charged twice"` → `[billing, technical, other]` | 0.382 | **0.955** |
| **semantic** | `"I was charged twice"` → `[billing, technical, other]` | 0.334 | **0.424** |

Capacity is not the constraint — 508k and 43k parameter models both stall near 0.42 on
the semantic task, a 12× size difference worth 0.03.

A pooled byte encoder needs options to be **distinguishable** (→ short) *and* to **share
vocabulary with the state** (→ long). Mean pooling cannot deliver both. So this design
routes on cues already present in the state; it does not supply world knowledge.

## Why "bit-identical" and not "close enough"

The model runs natively and as WASM inside SpacetimeDB. If those disagree, nothing
downstream is trustworthy. Two decisions make them agree exactly:

- Every transcendental (`exp`, `tanh`, `sqrt`) goes through the `libm` crate on both
  targets. Platform libm differs between macOS and wasm.
- Float sums accumulate in a fixed order — eight independent accumulators reduced in a
  hardcoded pairwise order, fast enough to vectorize and deterministic enough to
  reproduce.

Verified across three environments and two CPU architectures, including SpacetimeDB
Maincloud. A non-zero parity delta is a bug to find, not a tolerance to widen.

## Quick start

```bash
cargo xtask ci                 # fmt, clippy, 74 tests, source policy, golden fixtures
cargo xtask data 100000        # synthetic corpus, split by template family
cargo xtask train 5 20000      # Candle; early stops on CALIBRATED validation NLL
cargo xtask eval               # fit temperature, score held-out families
cargo xtask export 0.67        # quantize + refuse to write if answers changed
```

End to end through SpacetimeDB:

```bash
spacetime start &
cargo build --release --target wasm32-unknown-unknown \
  --manifest-path crates/stdbev-stdb/Cargo.toml
spacetime publish -s local -y --delete-data=always stdbev \
  --bin-path crates/stdbev-stdb/target/wasm32-unknown-unknown/release/stdbev_stdb.wasm
cargo xtask parity local stdbev          # native vs WASM, must be exact
```

## Layout

| crate | owns |
|---|---|
| `stdbev-types` | domain types, validation, the `STDBEV01` format definition |
| `stdbev-format` | byte tokenization and canonical question encoding |
| `stdbev-math` | framework-free kernels (`no_std`, so platform float ops cannot sneak in) |
| `stdbev-runtime` | artifact parsing and quantized inference |
| `stdbev-quant` | quantization and artifact writing |
| `stdbev-data` | synthetic generation, splits, JSONL |
| `stdbev-train` | Candle model, training loop, export |
| `stdbev-eval` | metrics, temperature calibration |
| `stdbev-stdb` | the SpacetimeDB module (own workspace: WASM-only) |
| `xtask` | automation and CI gates |

`crates/stdbev-stdb` is deliberately **not** a workspace member: it links SpacetimeDB
host imports that exist only inside the WASM runtime, so it cannot build for the host at
all. Including it would break `cargo build --workspace` for everyone.

## Documentation

- [What this architecture can and cannot learn](docs/what-this-architecture-can-learn.md)
  — the controlled experiment. Read first.
- [Baselines](docs/baselines.md) — what an accuracy number has to beat, and a generator
  bug that made an untrained model beat every fixed guess.
- [Architecture reference](docs/architecture-reference.md) — what `jevlike` actually
  does, where this diverged, and two bugs found by reading its source.
- [Phase 0 measurements](docs/phase0-measurements.md) — every performance number,
  measured on the real target rather than estimated.

## Prior art

- [`jevlike`](https://github.com/vinnylarouge/jevlike) — the open reference for the
  one-pass dynamic-option scorer. This implementation's attention head matches it to
  machine epsilon (5.5e-17), verified numerically.
- [`cua-s1-forms`](https://huggingface.co/cua-ai/cua-s1-forms) — an independent
  research checkpoint using the same head with a transformer encoder.
- [`jev`](https://docs.typesafe.ai) — TypeSafe's production model, whose Choice / Score /
  Noul vocabulary this adopts. Its architecture is not public; neither of the above is a
  reproduction of it.

## License

MIT
