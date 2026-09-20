# What the reference actually does

## First: jev's architecture is not public

Worth stating plainly, because it bounds every claim below. Jev is a commercial model
from TypeSafe, post-trained with a proprietary method the docs call **RLCD**. Its
architecture, parameter count and training data are unpublished. There is no source to
match.

What *is* public:

| | what it is | status |
|---|---|---|
| **jev** | TypeSafe's production model | proprietary, architecture unpublished |
| **jevlike** | third-party "jev-*like*" reimplementation | full source, 41k params |
| **cua-s1-forms** | independent research checkpoint | open weights, 706k params |

cua-s1-forms says so itself: *"Not calibrated with TypeSafe's RLCD method -- this is an
independent research checkpoint, not a reproduction of Jev."*

So "matching jev" is not available. The achievable goal is to match one of the two open
references *coherently*, which is exactly where this implementation went wrong.

### jev's API does tell us one thing

Its canonical Choice form passes **bare labels with descriptions optional**:

```bash
jev ask "I was charged twice" --questions '{"category":{"type":"choice",
  "instructions":"What is this about?",
  "criteria":{"billing":null,"technical":null,"other":null}}}'
# -> {"choice":"billing","probabilities":{"billing":1.0,"technical":0.0,"other":0.0}}
```

`criteria` maps label to description, and `null` is the documented default. This spec
mandates `LABEL\n{label}\nDESCRIPTION\n{description}` for every option -- scaffolding
that jev itself treats as optional, and that measurably hurts a pooled encoder.



Read from `jevlike`'s source, not from the spec's description of it. The spec is wrong
about this in a way that matters.

## jevlike's `TinyScorer` has no transformer at all

```python
def forward(self, batch, shuffle_context=False):
    context_ids = batch["context_ids"]
    positions = torch.arange(context_ids.shape[1], device=context_ids.device)
    context = self.embedding(context_ids) + self.position(positions)
    option_tokens = self.embedding(batch["option_ids"])
    weights = batch["option_token_mask"].unsqueeze(-1)
    options = (option_tokens * weights).sum(2) / weights.sum(2).clamp_min(1)
    return self.head(context, batch["context_mask"], options, batch["option_mask"], ...)
```

That is the whole model: an embedding table, a position table, and **one**
`AttentionHead`. Not two transformer layers over the context and pooling over options
-- *zero* layers on **both** sides. The context the options query is raw embeddings.

The spec attributes "2-layer Transformer, width 128, four attention heads" to jevlike.
Those numbers belong to **cua-s1-forms**, which adds an encoder on top of jevlike's
head. Anyone reading the spec would build the wrong thing and believe it was validated.

## Where this implementation stands

| | jevlike | cua-s1-forms | STDBEV |
|---|---|---|---|
| transformer layers (context) | **0** | 2 | 2 |
| transformer layers (options) | **0** | 2 | 0 (`Pooled`) or 2 (`Encoded`) |
| width | 64 | 128 | 128 |
| attention head rank | 64 | — | 128 |
| context window | 192 B | 224 B | 224 B |
| option window | **32 B** | 96 B | 96 B |
| option text | bare label: `"refund"` | field text | `LABEL\n…\nDESCRIPTION\n…` |
| loss | cross-entropy, hard label | cross-entropy | soft-target cross-entropy |

**The scoring head matches exactly** -- verified numerically, not by reading. Feeding
identical weights and inputs through jevlike's batched `einsum` formulation and through
this implementation's per-option loop agrees to **5.5e-17**, machine epsilon:

```
jevlike : [-0.21340272 -0.00128462  0.08739451]
stdbev  : [-0.21340272 -0.00128462  0.08739451]
```

Applying `sqrt(rank)` only once, a natural misreading of the source, shifts every logit
by a factor of 2.24 -- large enough to change answers, small enough to look plausible.

What matches:
LayerNorm both sides, no-bias Q/K/V projections, query from the *option*, key and value
from the *context*, `sqrt(rank)` divided **twice** -- once on the attention scores and
again on the final dot product. Verified line by line against the source.

The soft-target loss is a strict generalization: it reduces to plain cross-entropy on a
one-hot target, and carries a teacher's distribution when there is one.

## Hyperparameters — where this went wrong

| | jevlike | first pass here | |
|---|---|---|---|
| learning rate | **2e-3** | 3e-4 | 6.7x too low |
| weight decay | **1e-4** | 0.01 | 100x too high |
| gradient clipping | **1.0** | none | missing |
| dropout | **none** | 0.10 (added) | wrong direction |
| epochs | 8 | 14 | |

The first run moved only +0.025 above its random initialization in an epoch, and showed
training loss falling while validation loss rose. That pattern reads as overfitting, and
the response was to add dropout and cut the dataset -- **both wrong**. A learning rate
nearly seven times too low, fighting weight decay a hundred times too strong, produces a
model that has barely moved; regularizing it further makes it worse.

The lesson is narrow and worth keeping: when a from-scratch model underperforms, check
the optimizer settings against a known-good reference *before* concluding anything about
architecture or data. Those settings are cheap to verify and were sitting in the
reference's `train.py` the whole time.

## Noul is the clearest case

jev's Noul takes **only a question**:

```
--noul, -n <string>   The question to judge. Answered as a probability of yes when
                      used alone, with no separate confidence.
```

```json
{"type": "noul", "noul": 0.27}
```

There are no option descriptions. This spec invents two -- `false_description` and
`true_description` -- and §8 encodes them as:

```
FALSE\nThe account is not suspended
TRUE\nThe account is suspended
```

Those two strings are a condition and its negation. They differ by one word in thirty,
and measured across the test set they share **0.636** of their bytes. Mean pooling has
no mechanism to separate them, so every Noul example -- a third of the corpus -- asks
the model to discriminate on a signal the architecture cannot see.

Replacing them with bare polarity markers drops the overlap to **0.125**, a 5x
improvement, and loses nothing: the question already lives in the context as
`QUESTION\n{instructions}`.

## Bug: position embeddings were added to options

jevlike adds position to the context and **not** to options:

```python
context = self.embedding(context_ids) + self.position(positions)   # position
option_tokens = self.embedding(batch["option_ids"])                # no position
```

This implementation added it to both. The arithmetic says why that is wrong: pooling
`token + position` over `len` tokens yields

```
mean(token[0..len]) + mean(position[0..len])
```

The second term depends **only on the option's length**. Two options of similar length
therefore share a large constant component that carries no information about which
option they are -- it is pure additive noise that pulls every pooled option toward the
same direction.

Short options suffer most, which is the opposite of what one would guess: with a
seven-byte label, the shared positional term is a large fraction of the whole vector.

## The mechanism, measured

Byte overlap is suggestive. What the model actually sees is the *pooled vector*, so
that is what to measure. Pooling each option's byte embeddings and taking the cosine
between every pair of options in the same question:

| primitive | verbose + position (before) | verbose, no position | compact + position | compact, no position (reference) |
|---|---|---|---|---|
| Choice | **0.896** | 0.868 | 0.613 | **0.350** |
| Noul | **0.914** | 0.909 | 0.377 | **0.041** |
| Score | **0.907** | 0.895 | 0.403 | **0.057** |

The two fixes compose, and neither alone is sufficient: dropping position from a verbose
encoding barely moves Choice (0.896 -> 0.868), because the scaffolding still dominates.
Together they take Noul from indistinguishable to essentially orthogonal.

1.0 means indistinguishable. Under the spec's encoding every option in a question pools
to a vector roughly 0.90 similar to every other, and the option-attention head is then
asked to pick between them. It is not a capacity problem and no amount of training
fixes it: the information is destroyed in the pooling step, before any learned parameter
sees it.

This holds at initialization and is a property of the *encoding*, not of the weights,
which is why it was invisible in every training curve.

## Resolved

| | status |
|---|---|
| hyperparameters (lr, weight decay, clipping, dropout) | aligned with the reference |
| position embeddings on options | dropped for pooling, kept for the encoded path |
| option encoding | compact (label only); the format travels in the artifact header |
| Score label distribution | rescaled thresholds; near-uniform at every ladder length |

An untrained network now scores **0.329**, below the 0.383 uniform baseline, which is
where random weights belong. Before these fixes it scored 0.581 -- above every fixed
guess -- because the generator and the encoding were between them handing it free
accuracy. That number is the single best summary of whether the setup is honest.

## Still open

**Descriptions need the encoded option path.** The compact encoding drops them, because
mean pooling cannot preserve free text: keeping them as `{label}: {description}` leaves
pooled options at 0.845 cosine, against 0.350 for labels alone. Descriptions still reach
the model through the instructions, which are context and do pass through the
transformer. If they must discriminate *between options*, that requires
`OptionEncoder::Encoded` -- measured at 6.6x the compute in Phase 0, and not yet
evaluated for benefit.

**Model size.** 508k parameters against jevlike's 41k, on a corpus of 65k examples.
That is 7.8 parameters per example rather than 0.63. The `STDBEV_SHAPE=jevlike` config
exists to test whether the smaller shape generalizes better; the A/B was cut short when
the encoding bugs turned out to dominate, and is worth rerunning now that they are
fixed.
