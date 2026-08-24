# Convolution

Two kernels, both over a `(Cin, H, W)` activation, both **wide-vector FP32
mode only** (`wide_vector_debug=True`) with no narrow (INT8/FP8) variant:

| Kernel | Computes | Fused activation |
|---|---|---|
| `conv1x1_fp32` | pointwise (1×1), stride 1, no padding | none |
| `conv3x3_relu_fp32` | 3×3, stride 1, zero-padded by 1 | **ReLU** |

Both loop over output channels internally, so one launch produces the whole
`(Cout, H, W)` result.

## What they compute

```
conv1x1_fp32        out[o,y,x] =      bias[o] + SUM_ci W[o,ci] * in[ci,y,x]

conv3x3_relu_fp32   out[o,y,x] = relu(bias[o] + SUM_ci SUM_kr SUM_kc
                                      W[o,ci,kr,kc] * in[ci, y+kr, x+kc])
```

A 1×1 convolution has no spatial window, so every output element is a pure
channel-space dot product and all 128 lanes of a tile reduce independently —
one accumulation loop over input channels, no taps, no border, no mask.

The 3×3 kernel adds the nine taps and the zero border. Its ReLU is **fused and
not optional**; see below.

## The activation is part of the query

`ACTIVATE.QUANTIZE` takes its activation function as an **immediate**, not a
CR, so it is fixed when the `.asm` is assembled. A kernel implements exactly
one activation and cannot be asked for another at run time.

That makes `activation` a query parameter rather than a detail:

```python
resolve("conv2d", ..., activation="relu")   # -> conv3x3_relu_fp32
resolve("conv2d", ..., activation="none")   # -> conv1x1_fp32 (1x1 only)
```

Omitting it means `"none"`. A caller asking for a plain 3×3 convolution is
**refused**, not handed the ReLU kernel — that would be a confidently wrong
answer rather than a near miss. There is currently no 3×3 kernel without ReLU.

The `Conv2d` layer adapter emits `activation="none"`, because a bare
`nn.Conv2d` applies none. A fused conv+ReLU is two layers in torch and the
adapter sees one at a time, so it cannot assume a ReLU follows.

## Picking one

Don't pick by hand — ask the registry:

```python
from ipu_apps.kernel_registry import lookup_layer, resolve

lookup_layer(nn.Conv2d(256, 65, kernel_size=1), input_shape=(256, 8, 80))

resolve("conv2d",
        shape=(256, 8, 80), weight_shape=(65, 256, 1, 1),
        stride=1, padding=0, dilation=1, groups=1)
```

The verdict carries the app class, its constructor arguments, and the caveats
that apply at that shape (idle lanes, XMEM headroom).

### What is refused, and why

The registry refuses anything outside these kernels' domain with the offending
value named. `stride`, `padding`, `dilation` and `groups` are **required**
query parameters rather than assumed defaults: a conv spec that silently
ignored `stride=2` would answer confidently for an operation no kernel here
computes.

| Refused | Reason |
|---|---|
| a `kh`×`kw` window other than 1×1 or 3×3 | no kernel |
| `stride` ≠ 1 | no kernel subsamples |
| `padding` mismatched to the window | 1×1 needs `padding=0` (it reads one input element per output element, so padding only grows the output with a border it never writes); 3×3 needs `padding=1`, the value that keeps the output `H × W` |
| `dilation` ≠ 1 | no kernel dilates |
| `groups` ≠ 1 | grouped and depthwise convolution is a different dataflow |
| a batch extent > 1 | one image per launch |
| an activation no kernel fuses | see above |
| regions over the XMEM budget | reported with the row arithmetic and the largest row band that would fit |

`ConvTranspose2d`, `Conv1d` and `Conv3d` are refused explicitly rather than
routed here: they expose the same attribute names as `Conv2d`, so a permissive
adapter would return confidently wrong numbers.

## Memory layout

XMEM operands are **row numbers**, not byte addresses. One row is 128 elements
= 512 bytes in wide-vector debug mode, and XMEM holds 16384 such rows. Both
kernels share the same four regions, differing only in two numbers: how many
of a row's 128 lanes are usable output columns (`TC`), and how many zero rows
border a plane (`PAD`).

| | `TC` | `PAD` | weights/channel | group cap |
|---|---|---|---|---|
| `conv1x1_fp32` | 128 | 0 | 1 | 128 |
| `conv3x3_relu_fp32` | 126 | 1 | 9 | 14 |

With `TPR = ceil(W / TC)` rows per spatial row and `NGROUPS = ceil(Cin / cap)`:

| Region | CR | Size (rows) | Layout |
|---|---|---|---|
| input | `CR2` | `(Cin + 1) * (H + 2*PAD) * TPR` | channel-major planes; padded spatial row `r` at `+r*TPR`, one row per column tile |
| weight | `CR4` | `Cout * NGROUPS` | row `o*NGROUPS + g` holds one group's weights, `weights/channel` consecutive elements each |
| bias | `CR5` | `Cout` | one row per output channel, `bias[o]` in element 0 |
| output | `CR3` | `Cout * H * TPR` | same tiling, no border |

The `+1` on the input region is a **guard plane** — the pointwise kernel's
pipelined channel loop prefetches one channel past the end, and the 3×3 kernel
reserves it so the same is safe there.

The harness trims the unusable lanes in `teardown`, so the output file is a
dense `(Cout, H, W)` FP32 array in the same element order as the input.

### Halo tiling (3×3)

A load reaches a whole 128-element row and cannot shift by one element, so a
3×3 kernel cannot get its horizontal taps from shifted *loads* the way a
byte-addressed one did. Instead each tile stores its own horizontal halo:

```
element  0        = input column (t*126 - 1)      <- left halo
elements 1..126   = input columns t*126 .. t*126+125
element  127      = input column (t*126 + 126)    <- right halo
```

Output lane `j` (0..125) is column `t*126 + j`, and tap `kc` reads element
`j + kc + 1` — so every tap of every valid lane is satisfied from inside the
one row, with no neighbouring-tile dependency. Columns outside the image are
stored as zero, which *is* the convolution's zero padding. Lanes 126 and 127
read past the slot and are discarded.

**The vertical border then costs nothing.** Each plane carries an all-zero row
band above and below (`H + 2` spatial rows), so output row `y` reads padded
rows `y`, `y+1`, `y+2` unconditionally. There is no top/bottom special case
anywhere in the kernel — the zero rows *are* the padding.

### CR map

`CR0` and `CR1` are read-only hardware constants and both are exploited
directly: `CR0` is the zero source, and `CR1` serves as both the `1.0` scalar
for the bias broadcast and every `+1` increment.

| CR | Value |
|---|---|
| `CR2`–`CR5` | input / output / weight / bias base row |
| `CR6` | 1×1: `H * TPR`, rows per channel plane. 3×3: `H * TPR`, the walk from padded row `(y+2, t)` of one channel to `(y, t)` of the next — which is `(H+2)*TPR - 2*TPR`, the same number |
| `CR7` | `TPR` — rows per spatial row, and the tile-loop bound |
| `CR8` | `H` |
| `CR9` | `Cin` |
| `CR10` | `Cout` |
| `CR11` | `NGROUPS` — also the per-output-channel weight-row advance |
| `CR12` | `128` — the `Ra` element index selecting `R1[0]`; also the 1×1 group cap and the 3×3 `R_CYCLIC` slot-1 index |
| `CR13` | 3×3 only: `14`, the channel-group cap |
| `CR14` | 3×3 only: `126`, the tap walk's slot-to-slot step |
| `CR15` | dstructure (`valid_elements = 128`) |

The group weight stride is one row, which is `CR1` — it needs no CR of its own.
`CR6` is precomputed by the harness because the LR slot has no multiply. The
3×3 kernel uses all 16 CRs.

## Two techniques worth knowing

### Bias and accumulator reset in one word

This ISA has no `RESET_ACC`, so the reset folds into an `ACC.ADD.FIRST`. Rather
than peel the first channel's MAC out of both loops to carry it, the bias
supplies it:

```
MULT.EE lr_bias, CR1, 0, lr_zero, CR15 ; ACC.ADD.FIRST ;;
```

`MULT.EE` broadcasts one `Ra` element × a CR scalar, so this word reads
`R1[0]` (= `bias[o]`), multiplies by `1.0`, and overwrites `R_ACC` with the
result. Every MAC in every group is then a uniform `ACC.ADD` — no peel, no
duplicated group body — and because the word touches no input data it cannot
manufacture a `0 * inf` NaN the way a multiply-by-zero reset would.

This also retires the bias-as-an-all-ones-input-channel trick the older
byte-addressed kernel used, so `Cin' == Cin` with no synthetic plane.

### The software-pipelined channel loop

`MULT.RC.*` reads `R_CYCLIC` from the **start-of-word snapshot**, while
`LDR_CYCLIC_MULT_REG`'s offset is read **live**. A load co-issued with the
multiply that consumes it would therefore be one channel too late. The body
instead advances the address and loads channel *c+1* in the same word whose
multiply consumes channel *c*:

```
ADD lr_addr, lr_addr, CR6 ; ADD lr_widx, lr_widx, CR1 ; ADD lr_c, lr_c, CR1 ;
LDR_CYCLIC_MULT_REG lr_addr, CR2, lr_zero ;
MULT.RC.VE lr_zero, lr_widx, 0, lr_zero, CR15 ;
ACC.ADD ;;
```

The final iteration prefetches channel index `Cin`, whose data is never
consumed but whose row must still be in bounds — that is what the guard plane
is for.

`lr_widx` is seeded to `-1` rather than `0` because `MULT.RC.VE`'s `src`
operand is resolved live inside the handler, so the same-word increment lands
before the read. The `-1` is never itself read; the LR add wraps
`0xFFFFFFFF → 0`.

### The walking tap index (3×3)

`MULT.RC.*` reads `R_CYCLIC` at an arbitrary **element** index and may cross
slot boundaries. That is what lets the horizontal shift live in the register
rather than in the load. Three vertically-neighbouring rows occupy slots 0, 1
and 2, and one walking index steps through all nine taps in weight order:

```
tap  1     2  3    4    5   6    7    8  9
rc   0     1  2  128  129 130  256  257 258
step -258 +1 +1  +126  +1  +1  +126  +1  +1
```

Three step constants: `CR1` (=1), `CR14` (=126) and an LR holding `-258`.
`rc_idx` is read live, so the same-word add lands before the read; tap 1's
step wraps the index from 258 back to 0 in place, so the per-channel reset
costs no word of its own. A second index walks `+1` in lockstep through `R0`,
which is why the weights are simply `W[o, ci].ravel()` — `(kr, kc)` row-major,
nine per channel.

### The pipelined channel loop (3×3): 9 words, 9 MACs

The three row loads and the loop branch all hide inside the nine tap words, so
the MULT slot issues every cycle. Slots are read `0,0,0 / 1,1,1 / 2,2,2`, which
frees each one three taps before it is next needed:

```
tap 1   read slot 0 | load slot 2 <- THIS channel's row y+2
tap 4   read slot 1 | load slot 0 <- NEXT channel's row y
tap 7   read slot 2 | load slot 1 <- NEXT channel's row y+1
tap 9   read slot 2 | branch
```

Every load lands six words before its first read — necessary because
`MULT.RC.*` reads `R_CYCLIC` from the start-of-word **snapshot**, so a load
co-issued with its consuming multiply would be one word too late. (The ZDconv
kernel this is modelled on co-issues them; that ISA made a load visible in its
own cycle. This one does not, so the schedule is re-derived rather than
copied.) The last channel prefetches into the **guard plane**, which is what
the `+1` on the input region is for.

`lr_addr` walks one delta per load, cycling `+TPR, +H*TPR, +TPR`. `BLT` sits in
tap 9 and reads `lr_done` pre-increment, so the group bound is stored as
`gend-1`.

This took the 3×3 kernel from 14 words per input channel to **9** — a measured
1.51× on every channel-rich layer.

### Exact channel groups

One `LDR_MULT_REG` row holds 128 FP32 weights. A 1×1 kernel needs one weight
per channel, so a group covers up to **128** channels; the 3×3 kernel needs
nine, so a group covers **14** (126 of 128 elements used). That 14 is a
taps-per-row limit, not a width one — it is unchanged by FP32, and it is why
the pointwise kernel is not similarly capped.

Either way the group size is computed exactly as `min(cap, Cin - done)`, so a
129-channel pointwise input runs 128 then 1 with no padding. `R_ACC` is never
reset between groups; the accumulation simply continues, which is what makes an
arbitrary `Cin` work.

## Slot ordering

Within a VLIW word the slots execute
`LR → LOAD → MULT → ACC → AAQ → STORE → COND`. Two consequences the kernel
depends on:

- The store's offset register is incremented in the word **after** the store.
  Incrementing it in the store's own word would write one row too far, because
  LR runs first.
- `BLT` reads the start-of-word snapshot, so a counter incremented in the
  previous word is already advanced when the branch sees it.

## Not yet covered

- **A 3×3 convolution without ReLU.** It needs its own `.asm` (the activation
  is an assembly-time immediate), not a flag.
- **Strided, dilated, grouped and depthwise convolutions**, and window sizes
  other than 1×1 and 3×3. All are refused by name rather than approximated.
- **Row packing for narrow widths.** `_mult_mask_and_shift` supports a
  `partition` field that splits the 128 lanes into groups and, on a `mask_shift`
  of ±1, zeroes the lane at each group's start/end — exactly a row border, free,
  as an operand of the multiply. With `partition = 128/W` several spatial rows
  could share one XMEM row with no halo at all, taking lane utilisation to 100%.
  It applies only where `W ∈ {8, 16, 32, 64}` (the `Partition` enum's group
  sizes), so it would be a large win for small feature maps and no help to
  SuperPoint, whose widths are 640/320/160/80.
