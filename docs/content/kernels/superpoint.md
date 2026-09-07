# SuperPoint layer map

Which kernel runs each operation of SuperPoint, and — just as importantly —
which operations do **not** have one and why. This page exists so that question
is answered once rather than re-derived from the kernel docs each time.

All kernels here run in **wide-vector FP32 debug mode**
(`wide_vector_debug=True`, `WideVectorArithmetic.FP32`,
`wide_vector_quantize_output=False`) with `state.dtype = DType.INT8`.

## The forward network

| # | Operation | Kernel |
|---|---|---|
| 1 | `conv1a`–`conv4b` (8× 3×3 + ReLU) | [`conv3x3_relu`](conv2d.md) |
| 2 | ReLU | fused into the store (`ACTIVATE.QUANTIZE relu`) |
| 3 | `MaxPool2d(2, 2)` ×3 | [`maxpool2d_stride2`](pooling.md) |
| 4 | `convPa` (3×3 + ReLU) | [`conv3x3_relu`](conv2d.md) |
| 5 | `convPb` (1×1 → 65) | [`conv1x1`](conv2d.md) |
| 6 | detector softmax over 65 channels | [`softmax_columns`](softmax.md) |
| 7 | drop the dustbin channel | a slice — host data movement |
| 8 | depth-to-space (pixel shuffle, r = 8) | [`depth_to_space`](reshape.md) |
| 9 | `simple_nms` pooling (9×9, stride 1) | [`maxpool2d_nms9`](pooling.md) |
| 10 | score threshold | [`score_threshold`](detect.md) |
| 11 | border removal | a data-independent mask — host |
| 12 | top-k cap | [`score_threshold`](detect.md)'s soft count + a host bisection |
| 13 | `(h,w) → (x,y)` flip | a reshape — host |
| 14 | `convDa` (3×3 + ReLU) | [`conv3x3_relu`](conv2d.md) |
| 15 | `convDb` (1×1 → 256) | [`conv1x1`](conv2d.md) |
| 16 | dense descriptor L2 normalize | [`l2_normalize_channels`](normalize.md) |
| 17 | dense integer-pixel descriptor interpolation | [`sample_descriptors_separable`](sample-descriptors.md) for stock; `sample_descriptors` for shared/direct |
| 18 | per-keypoint L2 normalize | [`l2_normalize_channels`](normalize.md) |

Operation 6 needs no kernel of its own: a softmax over the channel axis of a
`(65, H·W)` matrix is a reduction **down columns**, which is exactly
`softmax_columns`'s `dim=0` case.

An alternative detector read-out, replacing 6–9 where only the ranking matters,
is [`channel_peak`](detect.md) — `argmax(softmax(x)) == argmax(x)`, so the peak
over raw logits picks the same cell. `cell_nms` is a composition of
`channel_peak`, `softmax_columns` and `conv1x1`; see
[the detector-head page](detect.md#cell_nms-is-a-composition-not-a-kernel).

## What stays on the host

These are not ports that were skipped. They are operations the ISA cannot
express, and saying so plainly is more useful than an empty row.

| Operation | Why |
|---|---|
| `simple_nms`'s `==` / `\|` / `where` | no vector compare and no boolean vector. The **pool** is `maxpool2d_window`; the local-max test and the two suppression rounds around it are host work |
| the top-k τ bisection | the device emits `Σ sigmoid(T(s−τ))`; comparing it to `k` and bisecting is a host loop that re-launches the kernel |
| sparse / arbitrary-coordinate `grid_sample` | data-dependent gathers remain host work; [`sample_descriptors`](sample-descriptors.md) computes every integer pixel using precomputed spatial filters |
| dustbin slice, border removal, `(h,w)→(x,y)` | data movement, no compute |

Two of these are **behavioural** rather than exact, and it is worth being
precise about which:

- **the top-k cap** keeps *about* `k` keypoints, not exactly `k` — the threshold
  is calibrated, not sorted. Everything up to and including the threshold gate
  is exact; only the cap approximates.
- descriptor interpolation's `shared` mode uses cell-centered coordinates;
  `stock` mode reproduces the checked-in sampler's interpolation coordinates.
  Neither includes the final descriptor L2 normalization. The convolutions
  and the L2 norm differ from a stock PyTorch run only by FP32 accumulation order.

## Two limits that turned out not to hold

The hand-written kernels in `kernels/superpoint_superglue/` documented two steps
as permanently host-side. The current ISA expresses both:

- **stride-2 decimation.** `ACC.STRIDE` decimates `MULT_RES` into `R_ACC` by
  horizontal and/or vertical stride 2, so `MaxPool2d(2, 2)` is host-free. See
  [Pooling](pooling.md#accstride-is-what-makes-stride-2-host-free).
- **the pixel-shuffle interleave.** `ACC.RESHAPE` writes eight `MULT_RES`
  elements to arbitrary `R_ACC` indices, and `ADDB`/`ADDBI` step the index
  arrays, so depth-to-space is host-free. See
  [Reshape](reshape.md#why-this-needs-accreshape).

## What fits XMEM, and what does not

Wide-vector XMEM now holds 1,048,576 rows of 512 B (512 MiB). At full
480×640 resolution, the dense descriptor kernel fits in one launch: its
stock separable layout is approximately 305.7 MiB, including 300 MiB of output.
The original direct stock implementation remains available and uses 326.1 MiB.

The earlier 8 MiB capacity assumptions no longer apply. Use each kernel's
memory-layout guard to check its complete input, padding, weights, scratch,
and output requirements. Larger maps that exceed the current capacity still
require tiling by their caller.

## Not ported

- **SuperGlue** — `layernorm`, `attention_scores`, `sinkhorn_iter[_mt]`,
  `sinkhorn_col[_mt]`, `argmax_match[_mt]`. A different network.
- **An end-to-end pipeline harness.** The kernels are the deliverable; callers
  compose them. `test/test_cell_nms_composition.py` is the one exception, and
  exists only to keep a documented composition honest.
