# Pooling

Two kernels, both over a `(C, H, W)` activation, both **wide-vector FP32 mode
only** (`wide_vector_debug=True`) with no narrow (INT8/FP8) variant:

| Kernel | Computes | Output extent |
|---|---|---|
| `maxpool2d_halve` | 2×2 window, stride 2, no padding | `(C, H//2, W//2)` |
| `maxpool2d_window` | K×K window (K odd), stride 1, padded by `K//2` | `(C, H, W)` |

Both loop over channels internally, so one launch pools the whole tensor.

Their domains are **disjoint** — one is stride 2, the other stride 1 — so
`cost` never decides between them, and each one's refusal names the other.

## What they compute

```
maxpool2d_halve    out[c,y,x] = max( in[c, 2y,   2x], in[c, 2y,   2x+1],
                                     in[c, 2y+1, 2x], in[c, 2y+1, 2x+1] )

maxpool2d_window   out[c,y,x] = max over dy,dx in [0,K)
                                    of pad(in)[c, y+dy-P, x+dx-P],  P = K//2
```

`maxpool2d_halve` is `nn.MaxPool2d(2, 2)` — the layer SuperPoint applies after
`conv1b`, `conv2b` and `conv3b`. An odd `H` or `W` drops the trailing row or
column, matching `ceil_mode=False`; the verdict says so as a caveat, because a
silently dropped row is exactly the kind of thing that should not be silent.

`maxpool2d_window` is the pooling step of SuperPoint's `simple_nms`
(`K = 2·nms_radius + 1`, so `K = 9` at the default radius 4). **It is the pool,
not the whole NMS** — see [what stays on the host](superpoint.md#what-stays-on-the-host).

## Where the taps come from

XMEM addressing is row-granular: a load reaches a whole 128-element row and
cannot shift by one element, so the shifted *loads* a byte-addressed pooling
kernel would use are not expressible. `MULT.RC.*` reads R_CYCLIC at an
arbitrary **element** index and may cross slot boundaries, so the horizontal
shift moves into the register instead — `dx` is a `+1` step on the read index.
`MULT.RC.VE rc, cr1` is the identity move (R_CYCLIC × 1.0), so one tap is one
`MULT` plus one `ACC.MAX`.

The two kernels then diverge on where the *rows* live.

`maxpool2d_halve` needs two spatial rows at once and holds both, in R_CYCLIC
slots 0/1 and 2/3 — the second slot of each pair supplies element 128, which is
where the `dx=1` tap of the last lane lands.

`maxpool2d_window` cannot: R_CYCLIC has four 128-element slots, and a K×K
window needs K rows, so nothing above `K = 4` fits. It streams one row at a
time through slot 0 instead, with `K` as a run-time CR bound rather than an
unroll — one assembled binary serves every window size.

### `ACC.STRIDE` is what makes stride 2 host-free

Four taps give the **stride-1** 2×2 maximum at every column; a stride-2 pool
wants only the even ones. `ACC.STRIDE 64, on, off` reads MULT_RES as two rows
of 64, takes every second element of each, and writes the 64 survivors
*contiguously* at `R_ACC[(offset % 4) · 32]` — exactly the even columns, packed.

It writes MULT_RES into R_ACC **overwriting**, though, so it cannot itself take
a maximum: the maximum has to be finished first. Hence a round trip per half
output row — stage the stride-1 result to a scratch row, reload it, decimate on
the way back in.

Both halves are staged *before* either is decimated. Half B's `ACC.MAX.FIRST`
overwrites all 128 R_ACC elements and would destroy half A's decimated result.
Decimating afterwards works because `ACC.STRIDE` leaves the R_ACC indices it
does not write untouched: half A lands at base 0, half B at base 64.

This is the step the original hand-written kernels documented as permanently
host-side (*"stride-2 decimation is a host gather"*). It is not.

## Memory layout

All sizes in 128-element rows; one row is 512 bytes in wide-vector debug mode.

```
maxpool2d_halve     input    C planes x H spatial rows x IN_ROW_STRIDE rows
                    scratch  2 rows (one per half of the output row)
                    output   C planes x (H//2) rows x OUT_TILES_PER_ROW rows

maxpool2d_window    input    C planes x (H + 2P) spatial rows x TPR rows
                    seed     1 row of -FLT_MAX (resident in R_CYCLIC slot 3)
                    output   C planes x H rows x TPR rows
```

### Tiling

`maxpool2d_window` uses **halo tiling**, the same scheme as `conv3x3_relu`
generalised to any window: a spatial row is cut into tiles of
`TC = 128 - (K-1)` output columns, and element `e` of tile `t` is input column
`t·TC + e - P`. Output lane `j` reads element `j + dx`, and the largest element
any valid lane reads is `(TC-1) + (K-1) = 127` — the last element of the slot,
with nothing to spare. A wide window therefore costs lanes: `K = 9` leaves 120
usable columns of every 128.

`maxpool2d_halve` spends **no** lanes on a halo. One output XMEM row is 128
output columns and so spans 256 input columns — two full-width input tiles —
and the `+1` shift is satisfied by loading the *next* tile rather than by
reserving lanes. `IN_ROW_STRIDE` is sized from what the kernel reads
(`2 · OUT_TILES_PER_ROW`) rather than from `ceil(W/128)`, because a partly-filled
last output tile still needs its input tiles to exist, plus one **guard tile**
past them.

### `-FLT_MAX`, and where it is load-bearing

Every lane a kernel may read that holds no image column is filled with
`-FLT_MAX`, the identity of a maximum: the border rows and halo elements of a
padded plane, the columns past `W` in a partly-filled tile, and the guard tiles.

For `maxpool2d_window` this **is** the border: a centred window at the image
edge genuinely reads outside the image, and no other fill value is correct.

For `maxpool2d_halve` it is not — output lane `j < W//2` reads input columns
`2j` and `2j+1`, both below `W`, so no kept output ever touches a filled lane.
It is filled anyway, so the discarded lanes stay finite and debuggable rather
than holding whatever was there before.

`ACC.MAX.FIRST` seeds R_ACC from the first tap, so neither kernel needs a
`-inf` seed *vector* for the accumulator — except `maxpool2d_window`, whose tap
loop is uniform (`K` is a run-time bound, so no tap can be peeled). It keeps one
resident `-FLT_MAX` row in R_CYCLIC slot 3 and spends one `ACC.MAX.FIRST` word
per output tile on it.

## Picking one

Don't pick by hand — ask the registry:

```python
from ipu_apps.kernel_registry import lookup_layer, resolve

lookup_layer(nn.MaxPool2d(2, 2), input_shape=(64, 60, 80))

resolve("maxpool2d", shape=(1, 60, 80), kernel_size=9, stride=1, padding=4)
```

`kernel_size`, `stride` and `padding` are all **required**. A pooling spec that
silently assumed stride 2 would answer for an operation no kernel computes.

### What is refused, and why

| Query | Refused because |
|---|---|
| stride 2 with `K ≠ 2`, or stride 2 with padding | `ACC.STRIDE`'s decimation phase is an immediate: it always keeps the even columns, so a pad would shift which columns survive |
| stride 1 with even `K` | an even window has no centre, so the output cannot stay `H × W` |
| stride 1 with `padding ≠ K//2` | input and output share one tile grid, so a different output extent is not expressible |
| `K ≥ 129` | the halo would consume the whole 128-element row, leaving no usable output columns |
| any other stride | neither kernel implements it |
| a batch > 1 | one image per launch |
| over the XMEM budget | reported with the per-region row arithmetic and the largest row band that *would* fit |

The `MaxPool2d` adapter additionally refuses `ceil_mode=True` (a different
output extent), `dilation ≠ 1` (a different window) and `return_indices=True`
(the ISA has no lane-index extraction, so the argmax positions do not exist).
`AvgPool2d`, `MaxPool1d`, `MaxPool3d` and `AdaptiveMaxPool2d` are refused by
name: they share `MaxPool2d`'s attributes and compute something else.

## Budget

The over-budget refusal costs a band per **output** row, not per input row —
banding the input bands the output too, so charging the full output region
against a shortened input would make almost every real layer look untileable.

A useful calibration: a single-plane 480×640 score map at `K = 9` needs 2928
input rows and 2880 output rows against a 16384-row budget, so SuperPoint's NMS
pool runs in **one launch, unbanded** — unlike every convolution in the network.
The difference is the channel count, not the resolution.
