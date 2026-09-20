# Baselines — what an accuracy number has to beat

Computed from the actual test split (28,506 examples across 6 held-out template
families), not assumed.

| Primitive | n | uniform guess | best fixed *index* | best fixed *label* |
|---|---|---|---|---|
| Choice | 5,653 | 0.239 | 0.245 | 0.341 (`other`) |
| Noul | 9,534 | 0.500 | 0.501 | 0.501 (`yes`) |
| Score | 13,319 | 0.361 | 0.452 | 0.452 (`medium`) |
| **All** | **28,506** | **0.383** | **0.427** | **0.446** |

Two fixed-guess baselines, because they differ: always picking the same *position* in the
menu scores 0.427, while always naming the same *label* -- which a model with a fixed
option preference does -- scores 0.446.

**A model must clear 0.427 to have learned anything at all**, because that is what
always picking the same option index achieves.

## The number that actually matters

An **untrained** network — random weights, never shown an example — scored
**0.551** on validation. That is well above both chance baselines.

This is not a bug, and it is worth understanding before reading any headline number:
a randomly-initialized scorer is *not* neutral across options. It assigns different
logits to different option *text*, and the option texts are not symmetric. Noul's
FALSE/TRUE descriptions differ systematically, and the Score ladder's bands are
unevenly sized (the thresholds 0.05/0.25/0.60 make the top band 40% of the range).
A random model that happens to favour one of those wins for free.

Consequences for reading results:

1. **Accuracy alone is nearly uninformative here.** A trained model reporting 0.60 has
   barely moved off an untrained network.
2. **The untrained baseline is seed-dependent and swings widely.** Measured across runs
   in this project: 0.189, 0.340, 0.399, 0.428, 0.577, 0.581. Quoting one from memory is
   worthless; the only meaningful figure is the untrained-to-trained delta *within the
   same run*, which is why every run prints its own untrained score first.
3. **The shuffled-context control is the real test.** Swap in another example's context
   and re-score: a model that is genuinely reading the state collapses toward 0.427,
   while one exploiting option priors holds its accuracy. The gap, not the accuracy, is
   the evidence.

This is why the spec's bare "top-1 accuracy >= 0.95" gate is not a quality bar on its
own. On a same-generator split it is nearly automatic; on this template-holdout split
it may be unreachable — and in neither case does it establish that the model reads
context.


## The ceiling: this task is trivially learnable

The baselines above say what a model must *beat*. This says what it could *reach*.

A 1-nearest-neighbour classifier over 4-gram Jaccard similarity of the state text,
trained on 4,000 Choice examples and tested on the **held-out template families**:

| | Choice accuracy |
|---|---|
| chance (uniform) | 0.239 |
| trigram-overlap between state and option | 0.477 |
| **1-NN on state 4-grams** | **1.000** |
| the 508k-parameter model | ~0.60 |

**1-NN gets every single one right**, and still does with the template scaffolding
stripped out. So:

1. The data is not the problem.
2. The template holdout is not the problem -- the body pool is shared across templates,
   so only the phrasing is unseen, and 4-gram similarity sails through it.
3. The task is not hard.

The model is being beaten by a lookup table, which means the deficit is in the model,
not in what it was asked to do. Since it has two transformer layers over the context and
can plainly represent the state, the bottleneck is what happens *after* that: matching
the encoded state to the right option. That points squarely at the option
representation, where mean pooling over bytes has to separate strings that share 0.636
of their bytes.

Worth stating the general lesson, because it was nearly missed: before tuning a model
that underperforms, measure what a trivial baseline achieves on the same split. A model
losing to 1-NN is not a model that needs more epochs.


## A shortcut the generator handed the model

Found only because an *untrained* network scored 0.581 -- above every fixed-guess
baseline, which random weights should not manage.

The Score generator picked a rung with fixed thresholds and then clamped:

```rust
const BOUNDS: [f32; 3] = [0.05, 0.25, 0.60];
let idx = band(rate).min(levels - 1);      // every high rate collapses onto the last rung
```

With a rate uniform on `[0, 1)`, that made the final option correct:

| levels offered | P(answer is the last option) |
|---|---|
| 2 | **0.951** |
| 3 | 0.751 |
| 4 | 0.401 |
| overall | **0.702** |

"Always pick the last option" scored **0.70** on Score, which is 47% of the corpus,
with no need to read the state at all. Every "accuracy improved" reading was partly
measuring how well the model had found that shortcut.

The fix rescales the thresholds with the ladder, which is also the better semantics --
rating against the levels you are *given* means the ladder spans the range, so offered
only `[low, high]` the boundary belongs at the midpoint:

```rust
fn band(rate: f32, levels: usize) -> usize {
    ((rate * levels as f32) as usize).min(levels - 1)
}
```

| | before | after |
|---|---|---|
| K=2 label distribution | `{0: 0.049, 1: 0.951}` | `{0: 0.483, 1: 0.517}` |
| K=3 | `{0: 0.046, 1: 0.203, 2: 0.751}` | `{0: 0.329, 1: 0.340, 2: 0.331}` |
| K=4 | `{0: 0.048, ..., 3: 0.401}` | `{0: 0.243, ..., 3: 0.251}` |
| best fixed-guess, whole corpus | 0.446 | **0.408** |

The general lesson, and the cheapest check in this document: **print the label
distribution of generated data before training on it.** A degenerate label distribution
is invisible in every loss curve, and it makes a model look like it is learning when it
is memorizing an artifact of the generator.
