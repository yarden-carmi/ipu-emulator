# SuperPoint + SuperGlue — all operations, with host/device marking

Every operation in both pipelines, marked by where it runs:

- 🟢 **on-device kernel** — computed by an `.asm` kernel
- 🔵 **host data-movement** — no compute / no decisions (the cache unit: reshape,
  residual add, concat, band streaming, IMEM bank swaps)
- 🔴 **host compute** — a gather / sort / integer-index / compare the IPU ISA has
  no instruction for

Split ops show both parts (device math + the host gather/index around it).

## SuperPoint

| # | operation | where |
|---|-----------|-------|
| 1 | conv1a 3×3 + ReLU | 🟢 |
| 2 | conv1b 3×3 + ReLU | 🟢 |
| 3 | maxpool 2×2/s2 | 🟢 max · 🔴 stride-2 decimation (gather) |
| 4 | conv2a 3×3 + ReLU | 🟢 |
| 5 | conv2b 3×3 + ReLU | 🟢 |
| 6 | maxpool 2×2/s2 | 🟢 max · 🔴 stride-2 gather |
| 7 | conv3a 3×3 + ReLU | 🟢 |
| 8 | conv3b 3×3 + ReLU | 🟢 |
| 9 | maxpool 2×2/s2 | 🟢 max · 🔴 stride-2 gather |
| 10 | conv4a 3×3 + ReLU | 🟢 |
| 11 | conv4b 3×3 + ReLU | 🟢 |
| 12 | convPa 3×3 + ReLU | 🟢 |
| 13 | convPb 1×1 (→65) | 🟢 |
| 14 | softmax(65) | 🟢 |
| 15 | drop dustbin (65→64) | 🔵 slice |
| 16 | depth-to-space | 🟢 plane move · 🔴 within-plane interleave *(avoided if cell_nms used)* |
| 17 | simple_nms (9×9) | 🟢 maxpool · 🔴 ==/suppress masking |
| 18 | score threshold | 🟢 `relu(s−τ)` · 🔴 nonzero extraction (gather) |
| 19 | remove_borders | 🔴 mask / gather |
| 20 | top-k cap (→max_keypoints) | 🟢 soft count · 🔴 τ binary-search + select |
| 21 | (h,w)→(x,y) flip | 🔵 reshape |
| 22 | convDa 3×3 + ReLU | 🟢 |
| 23 | convDb 1×1 (→256) | 🟢 |
| 24 | dense L2 normalize | 🟢 |
| 25 | sample_descriptors (grid_sample) | 🟢 4-corner blend · 🔴 per-kpt corner gather |
| 26 | per-keypoint L2 normalize | 🟢 |

## SuperGlue

| # | operation | where |
|---|-----------|-------|
| 27 | normalize_keypoints | 🔵 affine on coords |
| 28 | encoder MLP [3,32,64,128,256] (Conv1d+BN+ReLU) | 🟢 |
| 29 | desc += enc (residual) | 🔵 add *(device-capable)* |
| 30 | Q projection (D→D) | 🟢 |
| 31 | K projection (D→D) | 🟢 |
| 32 | V projection (D→D) | 🟢 |
| 33 | head reshape (4 interleaved heads) | 🔵 view |
| 34 | QKᵀ (per head) | 🟢 |
| 35 | scale 1/√d | 🟢 |
| 36 | attention softmax (over keys) | 🟢 |
| 37 | attention·V (per head) | 🟢 |
| 38 | concat heads | 🔵 view |
| 39 | merge projection (D→D) | 🟢 |
| 40 | residual x += message | 🔵 add *(device-capable)* |
| 41 | merge MLP [2D,2D,D] (Conv1d+BN+ReLU) | 🟢 |
| 42 | residual x += mlp | 🔵 add *(device-capable)* |
| 43 | final projection (D→D) | 🟢 |
| 44 | score matrix mdesc0ᵀ·mdesc1/√d | 🟢 |
| 45 | dustbin augmentation (bin_score row/col) | 🔵 concat |
| 46 | Sinkhorn row normalization (×T) | 🟢 |
| 47 | Sinkhorn column normalization (×T) | 🟢 |
| 48 | row/col argmax | 🟢 one-hot · 🔴 integer index extraction |
| 49 | mutual-NN check | 🔴 integer compare + gather |
| 50 | match threshold + assignment | 🟢 threshold · 🔴 index gather |

## Summary

- **Fully 🟢 on-device** (all convs, matmuls, softmax, L2, attention, Sinkhorn):
  1,2,4,5,7,8,10–14,22,23,24,26,28,30–37,39,41,43,44,46,47.
- **🔵 Host data-movement only** (no decisions; residual adds are device-capable
  but done host-side in the chained harness): 15,21,27,29,33,38,40,42,45 — plus
  band streaming and IMEM bank swaps (not per-op).
- **🔴 Genuine host compute** (gather / sort / integer-index / compare the ISA
  lacks): stride-2 maxpool decimation (3,6,9), NMS masking (17),
  keypoint/border extraction (18,19), top-k τ search (20), descriptor corner
  gather (25), and the SuperGlue match readout — integer argmax index +
  mutual-NN (48,49,50).

Every 🔴 host op is a **gather / index / sort** primitive — exactly what the IPU
ISA has no instruction for. All arithmetic is 🟢 on-device.

See [`PIPELINE.md`](PIPELINE.md) for the kernel + parity + validation detail.
