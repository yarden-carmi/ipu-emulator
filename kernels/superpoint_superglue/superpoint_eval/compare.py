#!/usr/bin/env python3
"""Compare the .asm conv kernels against real SuperPoint, one layer at a time.

For each layer: .asm (conv_fp32 if Cin<=13 else conv_fp32_tiled) with the real
pretrained weights, vs modified_ops (NumPy, same math) and vs torch SuperPoint.

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

def torch_conv(layer, x, pad):
    import torch, torch.nn.functional as F
    w, b = wload(layer)
    t = F.conv2d(torch.from_numpy(x)[None], torch.from_numpy(w), torch.from_numpy(b), padding=pad)
    return F.relu(t)[0].numpy()

def err(a, b):
    d = np.abs(a - b); return float(d.max()), float(d.mean())

def eval_conv(layer, H, Wd, Cout_lim=None, seed=0):
    w, b = wload(layer)                       # (Cout,Cin,3,3),(Cout,)
    Cout, Cin = w.shape[0], w.shape[1]
    Cl = Cout_lim or Cout
    w, b = w[:Cl], b[:Cl]
    rng = np.random.default_rng(seed)
    x = (rng.standard_normal((Cin, H, Wd)).astype(np.float32) * 0.3)
    if Cin <= 13:
        asm_out, cyc, mu, au = asm_harness.conv_fp32_layer(x, w, b, wcrop=Wd)
        kern = "conv_fp32"
    else:
        asm_out, cyc, mu, au = asm_harness.conv_fp32_tiled_layer(x, w, b, wcrop=Wd)
        kern = "conv_fp32_tiled"
    ref = modified_ops.conv2d_relu(x, w, b)[:, :, :Wd]
    e_am = err(asm_out, ref)
    print(f"=== {layer} (Cin={Cin}->Cout={Cout}, eval {Cl} ch, {H}x{Wd}, {kern}) ===")
    print(f"  .asm vs modified_ops (same math): max {e_am[0]:.2e} mean {e_am[1]:.2e}  "
          f"{'MATCH' if e_am[0] < 1e-3 else 'FAIL'}")
    try:
        tref = torch_conv(layer, x, pad=1)[:Cl, :, :Wd]
        e_at = err(asm_out, tref)
        print(f"  .asm vs torch SuperPoint:         max {e_at[0]:.2e} mean {e_at[1]:.2e}  "
              f"{'MATCH' if e_at[0] < 2e-3 else 'DIFF'}")
    except Exception as ex:
        print(f"  (torch skipped: {ex})")
    print(f"  cycles={cyc:,d}  MULT%={mu:.1f}  ACC%={au:.1f}")

if __name__ == "__main__":
    eval_conv("conv1a", H=24, Wd=120)               # Cin=1  -> conv_fp32
    eval_conv("conv1b", H=12, Wd=60, Cout_lim=16)   # Cin=64 -> conv_fp32_tiled
