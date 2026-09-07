# Pooling

Five registered kernels, all over a `(C, H, W)` activation, all **wide-vector FP32 mode
only** (`wide_vector_debug=True`) with no narrow (INT8/FP8) variant:

| Kernel | Computes | Output extent |
|---|---|---|
| `maxpool2d_stride2` | 2×2 stride 2; every output XMEM row has two input XMEM rows | `(C, H//2, W//2)` |
| `maxpool2d_stride2_tail` | Same operation; final output XMEM row has one input XMEM row | `(C, H//2, W//2)` |
| `maxpool2d_window` | K×K window (K odd), stride 1, padded by `K//2` | `(C, H, W)` |
| `maxpool2d_nms9` | **unrolled** 9×9, stride 1, padding 4 | `(C, H, W)` |
| `maxpool2d_nms7` | **unrolled** 7×7, stride 1, padding 3 | `(C, H, W)` |

All five loop over channels internally, so one launch pools the whole tensor.

The two stride-2 variants split that operation by XMEM geometry and have
disjoint domains. Together their domain is disjoint from the stride-1 kernels,
so `cost` never decides between them.

The two `nms` kernels **overlap** `maxpool2d_window` on purpose: they compute
exactly what it computes at K=9 and K=7, about 30% faster, and win on cost.
`maxpool2d_window` still claims those windows (its `supports` states its true
domain) and appears as the alternative in the verdict.

## What they compute

```
maxpool2d_stride2    out[c,y,x] = max( in[c, 2y,   2x], in[c, 2y,   2x+1],
                                     in[c, 2y+1, 2x], in[c, 2y+1, 2x+1] )

maxpool2d_window   out[c,y,x] = max over dy,dx in [0,K)
                                    of pad(in)[c, y+dy-P, x+dx-P],  P = K//2
```

`maxpool2d_stride2` is `nn.MaxPool2d(2, 2)` — the layer SuperPoint applies after
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

The stride-2 variants need two spatial rows at once and hold their current XMEM
rows in R_CYCLIC slots 0 and 2. The `dx=1` tap's final temporary position enters
the following slot, but `ACC.STRIDE` always discards that position, so the
following input XMEM row is not loaded.

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

## Unrolling K, and what it buys

`maxpool2d_window` takes `K` as a run-time CR, so one 19-word binary serves
every window size. That generality has a fixed price, because a uniform loop
body cannot peel or pipeline:

```
per output tile:   K² + 3K + 5 words
```

The `3K` is three words per `dy` row that are not taps — the row load, a **dead
separation word** (R_CYCLIC's *contents* come from the start-of-word snapshot,
so a load needs a word before its first consumer), and the `dy` branch. The `5`
is the tile prologue, the accumulator seed, the store and the loop control.

Fixing `K` and emitting every tap removes all three:

| | general | unrolled | why |
|---|---|---|---|
| row load latency | 1 dead word per row | none | the load is issued inside the *previous* row's taps and lands nine words early |
| accumulator seed | resident `-FLT_MAX` row + 1 word/tile | none | tap (0,0) simply **is** `ACC.MAX.FIRST` |
| one-past prefetch | needs a guard row | none | the last two rows just do not issue a load |

```
per output tile:   K² + 5 words
```

Measured at 480×640 against `torch.nn.functional.max_pool2d`, both bit-exact:

| K | general | unrolled | speedup |
|---|---|---:|---:|
| 9 | 326,887 | **249,127** | **1.31×** |
| 7 | 217,447 | **156,967** | **1.39×** |

### Three rotating R_CYCLIC slots, not two

Row `dy` is read from slot `dy % 3` while row `dy + 2` loads into slot
`(dy + 2) % 3` — the slot holding row `dy - 1`, whose taps are finished. **Two
slots cannot do this**: the one not being read holds the row needed next. Rows
0 and 1 are preloaded before the tap stream.

### Where it stops

The tile body *is* `K²` words, so the program grows quadratically:

| K | tap words | total | fits the 128-word IMEM bank? |
|---|---|---|---|
| 7 | 49 | **64** | yes |
| 9 | 81 | **96** | yes |
| 11 | 121 | ~136 | **no** |

K=9 — SuperPoint's default — fits with 32 words to spare; K=11 does not fit at
all. That, and not effort, is why there is no `nms11`.

## Memory layout

All sizes in 128-element rows; one row is 512 bytes in wide-vector debug mode.

```
maxpool2d_stride2     input    C planes x H spatial rows x IN_ROW_STRIDE rows
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

The stride-2 variants use compact input storage: exactly `ceil(W/128)` XMEM rows
per matrix row. One output XMEM row normally consumes two input XMEM rows. When
the final output XMEM row consumes only one, the registry selects
`maxpool2d_stride2_tail`; it runs the complete pairs first and then a separate
one-input-row tail section. Neither variant needs an all-padding or guard XMEM
row.

### `-FLT_MAX`, and where it is load-bearing

Every position a kernel may read that holds no image column is filled with
`-FLT_MAX`, the identity of a maximum: the border rows and halo elements of a
padded plane, and the positions past `W` in a partly-filled final XMEM row.

For `maxpool2d_window` this **is** the border: a centred window at the image
edge genuinely reads outside the image, and no other fill value is correct.

For the stride-2 variants it is not — output element `j < W//2` reads input
elements `2j` and `2j+1`, both below `W`, so no kept output touches a filled
position.

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
