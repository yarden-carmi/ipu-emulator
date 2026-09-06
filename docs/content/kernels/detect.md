# Detector head

Two kernels, both **wide-vector FP32 mode only** (`wide_vector_debug=True`):

| Kernel | Computes |
|---|---|
| `channel_peak` | per-cell confidence `max_c logits[c,n]`, plus `relu(conf − τ)` |
| `score_threshold` | `relu(s − τ)` over a score map, plus a soft survivor count |

These are the read-out steps of SuperPoint's keypoint detector that the ISA can
express. The rest — `simple_nms`'s comparisons, the top-k bisection — cannot be,
and are listed in [the SuperPoint layer map](superpoint.md#what-stays-on-the-host)
rather than left as an apparent coverage hole.

Neither is a `torch.nn` layer, so neither has a layer adapter.

## A threshold is a resident vector, not a CR scalar

Both kernels share this. CR scalars are **integer-only** in wide-vector mode —
`MULT.*`'s CR operand supplies its *low byte* — so a fractional threshold cannot
ride in a CR. It is written into a 128-element XMEM row instead, kept resident
in an R_CYCLIC slot for the whole run, and subtracted with `ACC.SUB`.

The older kernels negated the threshold and added it
(`AGG max value_cr(-1)` then `ACC.ADD_AAQ`). `ACC.SUB` does it directly, the
same replacement `softmax_rows` makes in its max-subtract.

## `channel_peak`

```
confidence[n] = max over c of logits[c, n]
keep[n]       = relu(confidence[n] - tau)
```

Cells in lanes, channels as planes. Every cell is an independent maximum and the
datapath is 128 lanes wide, so one pass down the planes reduces 128 cells at
once with `ACC.MAX` — the running maximum stays a full vector and never
collapses, so there is no `AGG` and no fan-out.

Plane 0 is issued **before** the loop so `ACC.MAX.FIRST` can seed `R_ACC`
without a `-inf` vector; the channel count is a run-time bound, so the first
iteration cannot carry a different accumulate mode. A `BGE` skips the loop
entirely when there is only one plane — without it, the loop would fold in
whatever plane 1's address points at.

`ACTIVATE.QUANTIZE` does not modify `R_ACC`, so the confidence is stored and
then subtracted from **in place**: the maximum is computed once and used twice.

### It is argmax-equivalent to the softmax path, not a softmax substitute

`argmax(softmax(x)) == argmax(x)`, so which cell wins is unchanged by taking the
maximum over raw logits instead of over probabilities. The **value** differs: it
is a logit, not a probability.

Use this where the ranking matters. Where the value matters, use a softmax
kernel — a `(65, H·W)` detector map is a reduction down columns, so
`softmax_columns` covers it with no new kernel. The verdict carries that caveat
on every `channel_peak` query.

## `score_threshold`

```
selected[i] = relu(s[i] - tau)                  exact: the set {s > tau}
count       = SUM_i sigmoid(T * (s[i] - tau))   approximate |{s > tau}|
```

The gate is exact. The count is what the **host** bisects on to hit a target
`k`: it re-launches with a new `τ`, reads the count, and narrows.

**This does not produce ranked indices.** Exact ranked top-k needs a per-element
compare and lane-index extraction, and the ISA has neither. That is the one
behavioural approximation in SuperPoint's detector path — the cap keeps *about*
`k` keypoints rather than exactly `k`.

The gate is element-wise, so the input's rank carries no meaning, only its
element count. The shape is preserved in the verdict and `selected` comes back
in it.

### Two resident threshold vectors, not one

The count needs `T·(s − τ)`, and scaling `R_ACC` by `T` would cost an XMEM round
trip. A *second* resident vector holds `T·τ` instead, and the count path
computes `T·s` — a vector-vector multiply against a resident `T` vector in `R0`
— then subtracts it. Both subtractions are `ACC.SUB`.

### The padding lanes are suppressed, not ignored

Elements past the real count fill the last row, and `sigmoid(0 − τ)` is **not**
zero — padding would inflate the count. Those lanes hold `τ − 800/T`, so
`T·(pad − τ)` is exactly `−800` and the sigmoid underflows to a true zero.
`relu` of the same value is zero too, so the padding costs nothing in either
output.

This is why the temperature has a floor: below `1e-6` the suppressor stops being
representable, and the sigmoid is flat across any realistic score range anyway,
so the count would be about half the element count whatever `τ` is — useless for
a bisection.

### Why the count needs its own pass

`AGG.SUM` writes **one** R_ACC slot, and the next row's `ACC.ADD.FIRST`
overwrites all 128 — including that slot. The sigmoid plane is therefore staged,
reduced with `ACC.ADD` down the rows into 128 partial sums, and collapsed once
at the end.

```
Pass 1   per row: the gate and the staged sigmoid plane    8 words/row
Pass 2   ACC.ADD down the staged plane -> 128 partial sums 3 words/row
Drain    stage the partials, reload, AGG.SUM -> one scalar
```

## Picking one

```python
from ipu_apps.kernel_registry import resolve

resolve("channel_peak",    shape=(65, 4800), threshold=0.005)
resolve("score_threshold", shape=(480, 640), threshold=0.005)
```

`temperature` is deliberately **not** required by `score_threshold`: it has a
meaningful default (64.0) and only affects the count, whereas omitting
`threshold` would leave the kernel with no gate at all.

### What is refused, and why

| Query | Refused because |
|---|---|
| a rank outside 2–3 (`channel_peak`) | the reduction axis has to be the leading one |
| a non-finite `threshold` or `temperature` | a NaN threshold makes every output zero, which looks like a working kernel on a quiet image (raised as a malformed query) |
| `temperature < 1e-6` | see [above](#the-padding-lanes-are-suppressed-not-ignored) |
| over the XMEM budget | `channel_peak` suggests a channel split, `score_threshold` an element chunk — the gate is element-wise, so a chunk needs no context and the counts add |

## `cell_nms` is a composition, not a kernel

The hand-written `cell_nms.asm` computed, over one cell's 64 sub-grid channels:
`peak = max_c p_c` and `(a, b) = Σ_c softmax(T·p)_c · (c//8, c%8)`. Every piece
of it already exists:

| step | kernel |
|---|---|
| `peak` | `channel_peak` |
| `softmax(T·p)` | `softmax_columns` (or `softmax_columns_packed` below 65 cells — route it, don't pick) |
| `a` and `b` | `conv1x1` — a 1×1 convolution **is** a per-channel weighted sum, so the two coordinate dots are two output channels with `W[0,c] = c//8` and `W[1,c] = c%8` |

Porting it as a fourth kernel would carry a second copy of the softmax. Two
further differences from the original are improvements: the composition
processes **many cells per launch** (cells in lanes) rather than one, and it
returns real XMEM planes rather than the `aaq0..aaq3` scalar registers the
current ISA no longer has.

One conversion matters. `cell_nms_ref` is a **base-2** softmax, `2^(T·p)`, while
the softmax kernels compute the natural one via the base-2 reformulation
`2^(log2(e)·x)`. Feeding them `T·p` runs at an effective temperature of `T/ln2`,
about 1.44× too sharp; scale the logits by `ln 2` and the two are exactly equal.

`test/test_cell_nms_composition.py` runs all three kernels and checks the result
against `cell_nms_ref`, so this table cannot go stale.
