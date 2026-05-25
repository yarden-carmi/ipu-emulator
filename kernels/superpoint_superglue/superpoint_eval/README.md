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

## Status (asm vs real torch SuperPoint)
- **conv1a** (Cin=1→64, 3×3, `conv_fp32`): matches to **3e-5**; ~48% MULT util.
- **conv1b** (Cin=64→64, 3×3, `conv_fp32_tiled`, 5 channel-groups): matches to
  **1.9e-6**; ~66% MULT util.
- `conv_fp32_tiled.asm` adds channel-group tiling (G≤14 channels/group, R_ACC
  accumulates across groups) so any Cin (up to 256) runs; 35 words.
- Remaining (same pattern): conv2a..conv4b, detector head (convPa/Pb + detect),
  descriptor head (convDa/Db + L2-norm).

## Run
```
source /tmp/ipuenv/bin/activate            # torch + numpy + ipu packages
python superpoint_eval/extract_weights.py  # once
python superpoint_eval/compare.py
```
