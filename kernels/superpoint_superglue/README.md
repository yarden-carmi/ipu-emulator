# SuperPoint + SuperGlue non-convolution kernels

Hand-written IPU assembly kernels — **one `.asm` file per layer** — for the
non-convolution operations used by SuperPoint (keypoint detector/descriptor)
and SuperGlue (graph matching). Convolution / fully-connected matmul is out of
scope (see `ipu-apps/.../fully_connected` for matmul).

Each `.asm` file's header comment is the authoritative **register / memory
contract** (which CRs hold which addresses/constants, input/output layout).

## Kernels

| File | Layer | Op |
|------|-------|----|
| `softmax.asm` | detector-head channel softmax / attention softmax | stable base-2 softmax over ≤128 lanes |
| `l2_normalize.asm` | descriptor normalization | `x / ‖x‖₂` |
| `layernorm.asm` | SuperGlue MLP/attention norm | `γ·(x−μ)/σ + β` |
| `maxpool.asm` | keypoint max-pool / NMS core | element-wise max over K gathered taps |
| `cell_nms.asm` | keypoint detection (NMS alternative) | channel-space per-cell peak + soft-argmax sub-pixel coords; **no depth-to-space, no gather** |
| `attention_scores.asm` | attention QKᵀ | scaled dot product `(q·k)/√d` |
| `softmax.asm` (reused) | attention softmax | (same kernel) |
| `sinkhorn_iter.asm` | optimal-matching layer | log-domain row half-step (logsumexp→max) |
| `sinkhorn_col.asm` | optimal-matching layer | log-domain column half-step (transpose-free, via element-wise max across rows) |
| `sinkhorn_iter_mt.asm` | optimal-matching layer | multi-tile row half-step (C>128 columns, T=⌈C/128⌉ col-tiles) |
| `sinkhorn_col_mt.asm` | optimal-matching layer | multi-tile column half-step (C>128 columns, transpose-free) |
| `argmax_match.asm` | match read-out | temperature hard-argmax → one-hot assignment row |
| `argmax_match_mt.asm` | match read-out | multi-tile one-hot for C>128 wide rows (e.g. 512/513) |
| `topk.asm` | keypoint selection | confidence threshold `relu(x−thr)` + top-1 value |
| `topk_mt.asm` | keypoint selection (C>128) | global top-1 + count-above-threshold (`Σ sigmoid`) → host-calibrated top-k |
| `pixel_shuffle.asm` | heatmap reshape | depth-to-space plane relocation (strided copy) |

## Numeric mode (required)

All kernels target the emulator's **wide-vector FP32 debug mode** — 128 lanes of
real `float32`:

```python
from ipu_emu.ipu_state import IpuState, WideVectorArithmetic
from ipu_emu.ipu_math import DType
state = IpuState(wide_vector_debug=True,
                 wide_vector_arithmetic=WideVectorArithmetic.FP32)
state.regfile.set_cr(15, DType.INT8)   # dtype sentinel required by wide-mode setup
```

Conventions used throughout:

- **Float constants passed in CR registers are float32 bit patterns.** In wide
  mode `AGG ... value_cr cr_idx` reads the CR as `float32` bits and multiplies,
  so e.g. negate-the-max ⇒ `CR = 0xBF800000` (`struct.pack("<f", -1.0)`).
- **Exponential is base-2 only** (`ACTIVATE exp2` = 2ˣ). For natural-base
  softmax/attention the caller pre-scales logits by `log2(e) = 1.4426950409`.
- 128-lane registers; vectors longer than 128 are tiled by the caller.
- Memory layout: each "plane" is 128×f32 = 512 bytes; addresses are byte
  addresses; cyclic loads/`MULT` offsets must be 4-byte aligned.

## Assembling

No Bazel in this environment; assemble with the `ipu-as` module directly
(equivalent to `bazel run //src/tools/ipu-as-py:ipu-as -- assemble ...`):

```bash
python -m ipu_as.cli assemble \
  --input kernels/superpoint_superglue/softmax.asm \
  --output /tmp/softmax.bin --format bin
```

All kernels have been assembled and numerically spot-checked against reference
implementations in wide FP32 mode (softmax sums to 1; L2 gives unit norm;
layernorm matches `(x−μ)/σ`; max-pool/sinkhorn/argmax/topk/attention all match).

## ISA limitations (why some kernels are partial)

The IPU ISA has **no vector-lane → scalar move, no element-wise vector compare,
no gather/scatter load-store, and no logarithm**. Consequences:

- **`topk.asm`** produces thresholded scores and the top-1 value, not a ranked
  list of the *k* largest **indices** (no lane-index extraction). Host selects
  the final *k*.
- **`argmax_match.asm`** emits a one-hot **vector** (assignment row), not an
  integer index — which is exactly what the assignment matrix needs; mutual-NN
  = AND of row one-hot and column one-hot (run on `Pᵀ`).
- **Sinkhorn (`sinkhorn_iter.asm` + `sinkhorn_col.asm`)** runs in the **log
  domain** but, lacking a `log`, replaces `logsumexp` with `max` (the
  tropical/hard limit): `Z_ij ← Z_ij − max_j Z_ij` (row) and
  `Z_ij ← Z_ij − max_i Z_ij` (column). The column step is done **without a
  transpose**: the per-column max is the element-wise max of the row vectors,
  built by sweeping contiguous rows with `ACC.MAX`, then subtracted per row. A
  full iteration is therefore entirely on-device. (Only the `logsumexp→max`
  approximation remains; an exact entropic Sinkhorn would need `log`.)
- **`pixel_shuffle.asm`** relocates whole channel planes (the contiguous part of
  depth-to-space); the fine within-plane interleave is a per-element scatter the
  host completes (or the source is pre-laid-out so a plane copy suffices).
- **`attention_scores.asm`** computes one `(q·k)/√d` score per call and
  broadcasts it; scoring against M keys is a host loop advancing the addresses
  (no scatter to pack M scalars into one vector).
