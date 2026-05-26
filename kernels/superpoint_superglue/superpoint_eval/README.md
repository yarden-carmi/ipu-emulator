# SuperPoint op-by-op evaluation (asm vs real pretrained network)

Validates the `.asm` kernels against the **real SuperPoint** network
(`third_party/superpoint`, Magic Leap pretrained `superpoint_v1.pth`), one
operation at a time.

## Flow
1. `extract_weights.py` (needs `torch`, run once) → caches per-layer
   weights/biases to `weights/*.npy` (gitignored; regenerate from the submodule).
2. `modified_ops.py` — NumPy SuperPoint ops written to match each kernel's exact
   math (e.g. `conv2d_relu` = conv_fp32's 3×3 + zero-pad + bias + ReLU).
3. `asm_harness.py` — runs the `.asm` kernel in the wide-FP32 emulator with real
   weights. Bias is folded with **no kernel change** as an extra all-ones input
   channel whose center-tap weight = bias[oc].
4. `compare.py` — per op: `.asm` vs `modified_ops` (same math, expect ≤1e-3) and
   `.asm` vs torch SuperPoint (the true op). Reports error + cycles + ALU util.

## Status (asm vs real SuperPoint)

Op-by-op, the `.asm` conv kernels reproduce the real pretrained network's math.
Full detector path chained in the emulator (16×16 image) vs the SuperPoint
reference — every layer matches:

| layer | shape | max_abs_diff |
|-------|-------|-------------:|
| conv1a | 64×16×16 | 1.5e-5 |
| conv1b | 64×16×16 | 3.6e-5 |
| conv2a/2b | 64×8×8 | 1.6e-5 / 2.9e-5 |
| conv3a/3b | 128×4×4 | 2.5e-5 / 2.2e-5 |
| conv4a/4b | 128×2×2 | 2.2e-5 / 1.5e-5 |
| convPa | 256×2×2 | 3.8e-6 |
| **convPb (`semi`)** | 65×2×2 | **2.7e-5** |

**Compare as one:** the detector logits `semi` from the chained `.asm` forward
match SuperPoint to **2.7e-5** (FP32 accumulation-order rounding only). The
detector decision — our `superpoint_detect` max-over-64-channels argmax — equals
SuperPoint's softmax argmax **100%** of the time (softmax is monotonic, so
`argmax(softmax)=argmax(logits)`), so the selected keypoint per cell is identical;
the fixed-threshold top-k then operates on matching logits.

Kernels used: `conv_fp32` (Cin≤13), `conv_fp32_tiled` (channel-group tiling, any
Cin; ReLU), `conv_fp32_tiled_norelu` (convPb/convDb), `superpoint_detect`
(max+threshold). 2×2/s2 maxpool is the one host step (stride-2 = a gather the ISA
leaves to host). Only `.asm` + Python; simulator untouched.

## Run
```
source /tmp/ipuenv/bin/activate            # torch + numpy + ipu packages
python superpoint_eval/extract_weights.py  # once
python superpoint_eval/compare.py
```
