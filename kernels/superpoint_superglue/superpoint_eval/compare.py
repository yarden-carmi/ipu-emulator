#!/usr/bin/env python3
"""Compare an .asm kernel against real SuperPoint, one operation at a time.

Phase 1: conv1a (Cin=1) with real pretrained weights:
  .asm (conv_fp32) vs modified_ops (NumPy, same math) vs torch SuperPoint conv1a.

Run: python kernels/superpoint_superglue/superpoint_eval/compare.py
"""
import os, sys, numpy as np
HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
import modified_ops, asm_harness

W = os.path.join(HERE, "weights")
def wload(layer):
    return (np.load(os.path.join(W, f"{layer}.weight.npy")),
            np.load(os.path.join(W, f"{layer}.bias.npy")))

def torch_conv(layer, x):
    import torch, torch.nn.functional as F
    w, b = wload(layer)
    t = F.conv2d(torch.from_numpy(x)[None], torch.from_numpy(w), torch.from_numpy(b), padding=1)
    return F.relu(t)[0].numpy()

def err(a, b):
    d = np.abs(a - b)
    return float(d.max()), float(d.mean())

def eval_conv1a(H=24, Wd=120, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((1, H, Wd)).astype(np.float32)   # 1-channel input
    w, b = wload("conv1a")                                    # (64,1,3,3),(64,)
    asm_out, cyc, mu, au = asm_harness.conv_fp32_layer(x, w, b, wcrop=Wd)
    ref = modified_ops.conv2d_relu(x, w, b)[:, :, :Wd]
    e_am = err(asm_out, ref)
    print("=== conv1a (real weights, Cin=1->Cout=64) ===")
    print(f"  .asm vs modified_ops (same math): max {e_am[0]:.2e}  mean {e_am[1]:.2e}  "
          f"{'MATCH' if e_am[0] < 1e-3 else 'FAIL'}")
    try:
        tref = torch_conv("conv1a", x)[:, :, :Wd]
        e_at = err(asm_out, tref)
        print(f"  .asm vs torch SuperPoint conv1a:  max {e_at[0]:.2e}  mean {e_at[1]:.2e}  "
              f"{'MATCH' if e_at[0] < 1e-3 else 'DIFF'}")
    except Exception as ex:
        print(f"  (torch comparison skipped: {ex})")
    print(f"  cycles={cyc:,d}  MULT%={mu:.1f}  ACC%={au:.1f}  (64 channel launches, {H}x{Wd})")

if __name__ == "__main__":
    eval_conv1a()
