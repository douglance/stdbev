# What this architecture can and cannot learn

The single most useful result in this project, and the one that took longest to see.

## The controlled experiment

Identical code, identical architecture, identical optimizer, identical loss. Only the
*task* differs.

| task | example | untrained | trained |
|---|---|---|---|
| **lexical** | `"the correct answer is delta"` → `['bravo','alpha','delta']` | 0.354 | **0.797** |
| **semantic** | `"I was charged twice"` → `['billing','technical','other']` | 0.334 | **0.424** |

The lexical task is learned in a single epoch. The semantic one barely moves off its
random initialization and then declines.

Capacity is not the constraint:

| model | parameters | peak on the semantic task |
|---|---|---|
| spec V1 | 507,776 | 0.453 |
| jevlike shape | 43,648 | 0.424 |

A **12x** difference in size moves the result by 0.03.

## Why: mean pooling forces a choice it cannot win

A pooled option encoder needs two things at once, and they pull in opposite directions.

| option text | separability (cosine between pooled options) | shared words with the state |
|---|---|---|
| bare label, `billing` | **0.350** ✅ | **0.00** ❌ |
| keyword list, `billing charged invoice refund …` | 0.852 ❌ | **0.87** ✅ |

Short options stay distinguishable but share no vocabulary with the state, so there is
nothing for attention to match on. Long options bridge to the state lexically but pool
into nearly the same vector, so there is nothing to tell them apart. Mean pooling
averages; it cannot keep one discriminative word salient inside a longer string.

The architecture therefore works when the label is **short and present in the state**.
That is precisely jevlike's documented example:

```json
{"context": "The customer needs a refund.", "options": ["refund", "sales", "technical support"]}
```

The answer is in the context. Its reported ~98% on "synthetic menus" measures
byte-level lexical matching, not semantic routing -- with zero transformer layers, the
context is individual *bytes* with no composition, and the head matches options against
those bytes.

## What this means for the spec

The spec's premise -- a ~0.5M-parameter byte model making typed decisions over arbitrary
application state -- holds only for decisions whose answer is lexically recoverable from
the state. `"charged twice"` -> `billing` is not such a decision. Neither is most
real-world routing.

Three ways forward, none free:

1. **Keep it lexical.** Design option labels to share vocabulary with the state. Works
   today, demonstrated at 0.80, keeps every budget. Narrows what the model is for.
2. **Pretrained encoder.** jevlike ships this (`--encoder hf`, frozen Qwen2.5-0.5B) and
   it is how it gets semantics at all. Breaks the 1 MiB artifact budget by three orders
   of magnitude.
3. **Distil from jev.** Soft targets from a model that already has the semantics. The
   training format takes a distribution rather than a label index specifically so this
   needs no second pipeline. Costs metered API calls and inherits the teacher's ceiling.

## The methodological lesson

Five defects were found and fixed before this one -- hyperparameters, position
embeddings on options, option encoding, a degenerate Score label distribution, model
size. Every one was real. **None of them was the blocker.**

The experiment that settled it took ten minutes to write: generate a task the
architecture *must* be able to learn, and check that it does. Had it run first, it
would have separated "the code is broken" from "the task is beyond this model" before
any of the tuning, and most of the tuning would have been unnecessary.

Run the sanity task first.
