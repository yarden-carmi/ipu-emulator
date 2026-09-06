# Normalization

One kernel, **wide-vector FP32 mode only** (`wide_vector_debug=True`) with no
narrow (INT8/FP8) variant:

| Kernel | Computes | Reduction axis |
|---|---|---|
| `l2_normalize_channels` | `x / ‖x‖₂` | the **leading** axis |

```
out[c, n] = x[c, n] / sqrt( SUM_c x[c, n]^2 )
```

SuperPoint's dense descriptor normalization is this with 256 channels
(`convDb`'s output) and `H*W` columns, so a `(256, H, W)` tensor with `dim=0`
routes straight here.

## Why the leading axis

Every column is an **independent** normalization and the datapath is 128 lanes
wide, so one pass down the rows reduces 128 columns at once with `ACC.ADD`: the
running sum of squares stays a full 128-element vector and never collapses to a
scalar. There is no `AGG`, no fan-out, and no per-row bookkeeping vector.

That is the same structural saving `softmax_columns` has over `softmax_rows`,
and it is why a row-wise L2 norm would be a genuinely different kernel rather
than a transpose of this one. A `dim` that reduces along rows is **refused**,
naming what it would need.

## `1/‖x‖` is an activation, not a reduction post-function

The older byte-addressed kernel wrote `AGG sum inv_sqrt`. That post-function no
longer exists; `rsqrt` is one of `ACTIVATE.QUANTIZE`'s twelve activations, so
the reciprocal square root is taken on the way out of `R_ACC`:

```
ACTIVATE.QUANTIZE rsqrt cr15 ; STR_POST_AAQ_REG ...    ; the scale row
```

The emulator's `rsqrt` is **guarded**: `x ≤ 0` yields `0`, not `inf`. So an
all-zero column normalizes to zeros, matching `l2_normalize_ref`'s
`‖x‖ == 0 → zeros`. That is not a theoretical edge — a `convDb` output can be
identically zero wherever the input image is flat.

## Seeding the sum

The reduction loop's bound is a run-time CR, so its first iteration cannot be
peeled to carry an `ACC.ADD.FIRST`. One word before the loop multiplies
R_CYCLIC by `CR0` (= `0.0`) into `ACC.ADD.FIRST` instead, which clears `R_ACC`
without touching XMEM and without a resident zero row.

## Memory layout

All sizes in 128-element rows; one row is 512 bytes in wide-vector debug mode.

```
input   ROWS x TPR rows,  row (c, t) at c*TPR + t
rvec    1 row: 1/‖x‖ for the 128 columns of the current tile
output  ROWS x TPR rows,  the same offsets against a different base
```

`TPR = ceil(COLS / 128)`. Input and output share one offset, so a single
walking register addresses both. Columns past `COLS` are zero; they form their
own all-zero columns, which the guard sends to zero, and are trimmed on
read-back.

Two passes per column tile: the sum of squares (2 words per matrix row) and the
scale (3 words per matrix row).

## Picking it

```python
from ipu_apps.kernel_registry import resolve

resolve("l2_normalize", shape=(256, 60, 80), dim=0)
resolve("l2_normalize", shape=(1, 256, 60, 80), dim=1)   # the torch layout
```

A rank-4 `(N, C, H, W)` input has its batch axis split off first, so `dim=1` --
the channel axis, and the one a descriptor normalization uses -- reduces to the
leading axis of a rank-3 problem. Without that, `flatten_to_matrix` would refuse
it as an interior axis. Both the batch split and the flatten are reported as
notes on the verdict; neither is silent.

There is **no layer adapter**: torch exposes L2 normalization as
`nn.functional.normalize`, a function rather than a layer class, so there is no
class name to dispatch on.

### What is refused, and why

| Query | Refused because |
|---|---|
| a `dim` that reduces along rows | needs an `AGG` reduction and a per-row fan-out; no kernel implements that yet |
| a batch > 1 | one image per launch |
| `dim` naming a rank-4 input's batch axis | scaling across images is not an L2 normalization of a feature map (raised as a malformed query, not a refusal) |
| an interior `dim` | flattening around it would require transposing the other axes |
| over the XMEM budget | reported with the per-region row arithmetic |

The over-budget advice points at the **column** axis, never the reduction axis.
Splitting the reduction would give each band its own partial sum of squares,
which is not a normalization of anything.
