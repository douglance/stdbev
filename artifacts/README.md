# Artifacts

| file | what it is |
|---|---|
| `model.stdbq` | The deployable model. Embedded into the SpacetimeDB WASM module at compile time. |
| `genesis.stdbq` | Deterministic random weights, reproducible from a seed with `cargo xtask genesis`. |
| `genesis.meta.json` | Architecture and provenance for the genesis artifact. |

## Why two

The genesis artifact exists so the whole deployment spine -- parser, runtime, WASM
module, native/WASM parity -- can be tested without a trained model, and so the golden
numerical fixtures pin the *numerical path* rather than whichever model happens to be
shipped. It is reproducible byte for byte from seed 42.

They were originally one file, and `cargo xtask ci` silently overwrote trained weights
with random ones on every run. Hence the split.

## What `model.stdbq` currently contains

A **support-ticket routing** model: 43,648 parameters, trained on a synthetic corpus
where 80% of tickets carry a category tag (`[billing]`, `category: technical`).

| metric | value |
|---|---|
| test top-1 | 0.948 |
| expected calibration error | 0.0485 |
| shuffled-context control | 0.337 (chance) |
| artifact size | 47,940 bytes |

**Read `docs/what-this-architecture-can-learn.md` before using it.** It performs well
inside its training distribution and is *confidently wrong* outside it -- returning
incorrect answers at confidence 1.000 on unfamiliar inputs. The calibration figure above
is an in-distribution measurement and does not hold for arbitrary input.
