# Reshape

One kernel, **wide-vector FP32 mode only** (`wide_vector_debug=True`):

| Kernel | Computes | Upscale factors |
|---|---|---|
| `depth_to_space` | pixel shuffle, one output channel | 1, 2, 4, 8 |

```
out[r*h + a, r*w + b] = in[r*a + b, h, w]        for a, b in [0, r)
```

This is `nn.PixelShuffle(r)`. SuperPoint's detector head is the `r = 8` case:
the 64 sub-grid channels left after the softmax drops the dustbin become the
full-resolution `(H*8, W*8)` heatmap.

Data movement rather than arithmetic — but not therefore free.

## Why this needs `ACC.RESHAPE`

One output row interleaves `r` input planes at stride `r`:
`out_row[r*w + b] = plane_b[w]`. There is no scatter-store, no vector shuffle,
and no inverse of `ACC.STRIDE` (which decimates; it does not expand).

`ACC.RESHAPE` is the only instruction that writes MULT_RES elements to
**arbitrary** R_ACC indices: eight per instruction, addressed by two `LRDn`
register pairs read as eight source and eight destination byte indices.

```
ACC.RESHAPE LRD12, LRD14, 0 ;;    ; 8 elements -> their output lanes
```

The original hand-written kernel did only the plane-granular part of this and
documented the interleave as permanently host-side. It is not.

### The source indices are always `[0..7]`

Output tile `T` covers output columns `128T..128T+127`, which come from input
columns `E·T .. E·T+E-1` of each plane, where `E = 128/r`. Those `E` columns all
live in **one** input XMEM row — `E` divides 128, so the offset within the row
is a multiple of `E` and at most `128-E`.

Rather than encoding that offset in the `ACC.RESHAPE` source array — which would
need a fresh pair of CR constants per output tile — it rides in the `rc_idx` of
the `MULT.RC.VE` that stages the row. `MULT_RES[i]` is then already
`plane[S + i]`, and the source array is the constant `[0..7]` stepped by `+8`.
The same place every other kernel here puts a horizontal shift.

### The destination indices

Element `j` of plane `b` lands at output column `r·j + b`, so the destination
array for plane `b`, instruction `k` is `r·(8k + i) + b`:

```
seeded   [0, r, 2r, ..., 7r]     from CR14 (low four) and CR7 (high four)
per k    += 8r                   ADDB CR10
per b    -= 127                  undoing the r*E = 128 total drift, +1 for b
```

After `r` planes the array is re-seeded from the CRs rather than stepped back,
which is why no `-r` constant is needed.

**R_ACC is never cleared, and does not need to be.** Across `b` in `[0, r)` and
`j` in `[0, E)` the destinations `r·j + b` cover `0..127` exactly once, so every
lane of the output row is written before it is stored.

### Why the factor stops at 8

Two limits bound the supported factors, and `r = 16` fails the second:

- `128/r` elements per plane per output tile must be a whole number of
  8-element `ACC.RESHAPE` instructions, so `r` must divide 16; **and**
- the per-instruction destination step is `8r`, applied with `ADDB`, whose
  source byte is reinterpreted as **signed**. `8r` must therefore be at most
  127, so `r ≤ 15`.

At `r = 16` the step 128 reads as `-128` and walks the index array down to zero
instead of up. The refusal says which of the two limits a rejected factor hit.

## Memory layout

All sizes in 128-element rows; one row is 512 bytes in wide-vector debug mode.

```
input   C = r*r planes, each H spatial rows x IN_TILES_PER_ROW rows
output  1 plane, (H*r) spatial rows x (IN_TILES_PER_ROW * r) rows
```

The loop nest is `h → a → wt → sub → b → k`, chosen so the output row counter
is monotonic: output row `r·h + a` and output tile `wt·r + sub` both advance in
step with it, so one register addresses every store.

Roughly `2r + 16` words per 128 output elements: 69 at `r = 8`.

**Idle output lanes are multiplied by `r`.** Each input tile fans out to exactly
`r` output tiles whether or not it is full, so an input width of 80 (48 idle
lanes of 128) becomes an output row of 1024 lanes holding 640 real columns.
Padding the input width to a multiple of 128 costs nothing extra, and the
caveat on the verdict quantifies it.

## Picking it

```python
from ipu_apps.kernel_registry import lookup_layer, resolve

lookup_layer(nn.PixelShuffle(8), input_shape=(64, 60, 80))
resolve("depth_to_space", shape=(64, 60, 80), upscale_factor=8)
```

`PixelUnshuffle` is refused by name: it sits beside `PixelShuffle` in `torch.nn`
and exposes a factor in the same shape, but it moves space into depth — the
exact inverse.

### What is refused, and why

| Query | Refused because |
|---|---|
| `C` not a multiple of `r²` | a factor-`r` shuffle consumes `r²` channels per output channel |
| `C/r² ≠ 1` | this kernel emits one output channel; a multi-channel shuffle needs an outer loop offsetting the plane index by `c'·r²` |
| `r ∉ {1, 2, 4, 8}` | see [above](#why-the-factor-stops-at-8) |
| a batch > 1 | one image per launch |
| over the XMEM budget | reported with the per-region row arithmetic and a row band that would fit; the shuffle is independent across input rows, so a band needs no halo |

## Budget

A useful calibration: SuperPoint's `60×80×64 → 480×640` detector shuffle needs
3840 input rows and 3840 output rows against a 16384-row budget, so it runs in
**one launch, unbanded** — unlike every convolution in the network.
