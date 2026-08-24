# Convolution

One kernel so far: `conv1x1_fp32`, a pointwise (1×1) FP32 convolution over a
`(Cin, H, W)` activation. It is **wide-vector FP32 mode only**
(`wide_vector_debug=True`) and has no narrow (INT8/FP8) variant.

## What it computes

```
out[o, y, x] = bias[o] + SUM_ci W[o, ci] * in[ci, y, x]
```

A 1×1 convolution has no spatial window, so every output element is a pure
channel-space dot product and all 128 lanes of a tile reduce independently.
That collapses the whole kernel to a single accumulation loop over input
channels — no taps, no border, no mask.

The kernel loops over output channels internally, so one launch produces the
whole `(Cout, H, W)` result.

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

The registry refuses anything outside this kernel's domain with the offending
value named. `stride`, `padding`, `dilation` and `groups` are **required**
query parameters rather than assumed defaults: a conv spec that silently
ignored `stride=2` would answer confidently for an operation no kernel here
computes.

| Refused | Reason |
|---|---|
| a `kh`×`kw` window other than 1×1 | needs the shifted-tap kernel, not migrated yet |
| `stride` ≠ 1 | no kernel subsamples |
| `padding` ≠ 0 | a 1×1 kernel reads exactly one input element per output element, so padding only grows the output with a border this kernel never writes |
| `dilation` ≠ 1 | meaningless for a 1×1 window |
| `groups` ≠ 1 | grouped and depthwise convolution is a different dataflow |
| a batch extent > 1 | one image per launch |
| regions over the XMEM budget | reported with the row arithmetic and the largest row band that would fit |

`ConvTranspose2d`, `Conv1d` and `Conv3d` are refused explicitly rather than
routed here: they expose the same attribute names as `Conv2d`, so a permissive
adapter would return confidently wrong numbers.

## Memory layout

XMEM operands are **row numbers**, not byte addresses. One row is 128 elements
= 512 bytes in wide-vector debug mode, and XMEM holds 16384 such rows. Every
region below is therefore sized in rows, with `NCT = ceil(W / 128)` column
tiles per spatial row and `NGROUPS = ceil(Cin / 128)` channel groups:

| Region | CR | Size (rows) | Layout |
|---|---|---|---|
| input | `CR2` | `(Cin + 1) * H * NCT` | channel-major planes; spatial row `y` at `+y*NCT`, one row per column tile |
| weight | `CR4` | `Cout * NGROUPS` | row `o*NGROUPS + g` holds `W[o, g*128 : (g+1)*128]` |
| bias | `CR5` | `Cout` | one row per output channel, `bias[o]` in element 0 |
| output | `CR3` | `Cout * H * NCT` | same plane layout as the input |

Because addressing is row-granular, a spatial row shorter than `NCT*128` pads
with idle lanes. There is no tighter packing available — and correspondingly
none of the byte-level tight packing, guard row and last-tile spill handling
that a byte-addressed kernel would need. The harness slices the padding back
off in `teardown`, so the output file is a dense `(Cout, H, W)` FP32 array in
the same element order as the input.

The `+1` on the input region is one **guard plane**; see the channel loop below.

### CR map

`CR0` and `CR1` are read-only hardware constants and both are exploited
directly: `CR0` is the zero source, and `CR1` serves as both the `1.0` scalar
for the bias broadcast and every `+1` increment.

| CR | Value |
|---|---|
| `CR2`–`CR5` | input / output / weight / bias base row |
| `CR6` | `H * NCT` — rows per channel plane, input and output alike |
| `CR7` | `NCT` — rows per spatial row, and the column-tile loop bound |
| `CR8` | `H` |
| `CR9` | `Cin` |
| `CR10` | `Cout` |
| `CR11` | `NGROUPS` — also the per-output-channel weight-row advance |
| `CR12` | `128` — the channel-group cap, and the `Ra` element index selecting `R1[0]` |
| `CR15` | dstructure (`valid_elements = 128`) |

The group weight stride is one row, which is `CR1` — it needs no CR of its own.
`CR6 = CR8 * CR7` is precomputed by the harness because the LR slot has no
multiply.

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

### Exact channel groups

One `LDR_MULT_REG` row holds 128 FP32 weights and a 1×1 kernel needs one weight
per channel, so a group covers up to 128 channels — not the 14 the older 3×3
kernel was limited to, which was a taps-per-row limit rather than a width one.
The group size is computed exactly as `min(128, Cin - done)`, so a 129-channel
input runs 128 then 1, with no padding. `R_ACC` is never reset between groups;
the accumulation simply continues, which is what makes an arbitrary `Cin` work.

## Slot ordering

Within a VLIW word the slots execute
`LR → LOAD → MULT → ACC → AAQ → STORE → COND`. Two consequences the kernel
depends on:

- The store's offset register is incremented in the word **after** the store.
  Incrementing it in the store's own word would write one row too far, because
  LR runs first.
- `BLT` reads the start-of-word snapshot, so a counter incremented in the
  previous word is already advanced when the branch sees it.

## Not yet migrated

The 3×3 shifted-tap kernel (`conv_fp32_full`, with ReLU) reuses this package
and CR map. Its additional work is the nine-tap walking pointer through
`R_CYCLIC`, the one-pixel border, and a `relu` activation — which needs a
separate `.asm`, since `ACTIVATE.QUANTIZE`'s function is an immediate rather
than a CR and so cannot be selected at run time.
