#!/usr/bin/env python3
"""Evaluate each conv kernel at REAL SuperPoint resolution, one op at a time.

The conv kernels handle <=126 cols/row, so wide layers (W up to 640) are width-
tiled on the host: overlapping strips of width <=126 (chunk + 1px halo each
side), take the interior outputs. No kernel change. H (rows) is the kernel's own
loop. A few output channels are run for correctness; full-layer cycles = per-
channel cost x Cout.

Run: python kernels/superpoint_superglue/superpoint_eval/real_res.py [layer ...]
"""
import os, sys, numpy as np
HERE = os.path.dirname(__file__); sys.path.insert(0, HERE)
import asm_harness, modified_ops

W = os.path.join(HERE, "weights")
def wload(L): return (np.load(os.path.join(W, f"{L}.weight.npy")),
                      np.load(os.path.join(W, f"{L}.bias.npy")))

# real SuperPoint spatial size at each layer's input (640x480 image)
RES = {"conv1a": (480, 640), "conv1b": (480, 640),
       "conv2a": (240, 320), "conv2b": (240, 320),
       "conv3a": (120, 160), "conv3b": (120, 160),
       "conv4a": (60, 80),  "conv4b": (60, 80),
       "convPa": (60, 80),  "convPb": (60, 80)}

def conv_real(x, w, b, relu=True, chunk=124):
    """Conv at full (H,W), tiled in BOTH H (row bands, since the activation can
    exceed the 2MB XMEM) and W (<=126 col strips). Each tile feeds the kernel a
    (bh+2 x cw+2) sub-tile with 1px halo; interior output is the real result."""
    Cin, H, Wd = x.shape; Cout = w.shape[0]
    tiled = not (Cin <= 13 and w.shape[2] == 3)
    Ctot = ((Cin + 1) + 12) // 13 * 13 if tiled else (Cin + 1)   # padded channel planes
    band = max(1, min(H, (1_800_000 // (Ctot * 512)) - 2))        # rows that fit XMEM
    out = np.zeros((Cout, H, Wd), np.float32)
    tot = mult_a = acc_a = 0
    for r0 in range(0, H, band):
        bh = min(band, H - r0)
        for c0 in range(0, Wd, chunk):
            cw = min(chunk, Wd - c0)
            strip = np.zeros((Cin, bh + 2, cw + 2), np.float32)
            for i in range(bh + 2):
                rr = r0 - 1 + i
                if 0 <= rr < H:
                    for j in range(cw + 2):
                        cc = c0 - 1 + j
                        if 0 <= cc < Wd: strip[:, i, j] = x[:, rr, cc]
            if tiled:
                o, c, mu, au = asm_harness.conv_fp32_tiled_layer(strip, w, b, wcrop=cw + 2, relu=relu)
            else:
                o, c, mu, au = asm_harness.conv_fp32_layer(strip, w, b, wcrop=cw + 2)
            out[:, r0:r0 + bh, c0:c0 + cw] = o[:, 1:bh + 1, 1:cw + 1]   # interior
            tot += c; mult_a += mu * c / 100; acc_a += au * c / 100
    return out, tot, 100 * mult_a / tot, 100 * acc_a / tot

def eval_layer(layer, Cout_lim=2, seed=0):
    H, Wd = RES[layer]
    w, b = wload(layer); Cout, Cin = w.shape[0], w.shape[1]
    if w.shape[2] == 1:                              # 1x1 -> center-only 3x3 (both paths)
        w33 = np.zeros((Cout, Cin, 3, 3), np.float32); w33[:, :, 1, 1] = w[:, :, 0, 0]; w = w33
    w, b = w[:Cout_lim], b[:Cout_lim]
    rng = np.random.default_rng(seed)
    x = (rng.standard_normal((Cin, H, Wd)).astype(np.float32) * 0.3)
    relu = (layer != "convPb")
    out, cyc, mu, au = conv_real(x, w, b, relu=relu)
    ref = modified_ops.conv2d_relu(x, w, b, relu=relu)
    d = np.abs(out - ref)
    full = cyc * (Cout / Cout_lim)
    print(f"{layer}: {H}x{Wd}, Cin={Cin}->Cout={Cout} (eval {Cout_lim}ch)  "
          f"max={d.max():.2e} {'MATCH' if d.max()<2e-3 else 'FAIL'}  "
          f"cyc/{Cout_lim}ch={cyc:,d} MULT%={mu:.0f}  full-layer~{int(full):,d}cyc", flush=True)

if __name__ == "__main__":
    layers = sys.argv[1:] or ["conv1a"]
    for L in layers: eval_layer(L)
