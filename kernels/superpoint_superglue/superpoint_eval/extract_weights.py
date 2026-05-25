#!/usr/bin/env python3
"""Extract SuperPoint pretrained conv weights/biases to .npy (one-time, needs torch).

Reads third_party/superpoint/superpoint_v1.pth and writes
superpoint_eval/weights/<layer>.{weight,bias}.npy for every conv layer.
After this runs once, the harness/compare scripts are NumPy-only.
"""
import os, torch, numpy as np
HERE = os.path.dirname(__file__)
PTH = os.path.join(HERE, "..", "..", "..", "third_party", "superpoint", "superpoint_v1.pth")
OUT = os.path.join(HERE, "weights")
os.makedirs(OUT, exist_ok=True)

sd = torch.load(PTH, map_location="cpu")
layers = ["conv1a","conv1b","conv2a","conv2b","conv3a","conv3b","conv4a","conv4b",
          "convPa","convPb","convDa","convDb"]
print(f"{'layer':8} {'weight shape':>18} {'bias':>6}")
for L in layers:
    w = sd[f"{L}.weight"].cpu().numpy().astype(np.float32)   # (Cout, Cin, kh, kw)
    b = sd[f"{L}.bias"].cpu().numpy().astype(np.float32)      # (Cout,)
    np.save(os.path.join(OUT, f"{L}.weight.npy"), w)
    np.save(os.path.join(OUT, f"{L}.bias.npy"), b)
    print(f"{L:8} {str(w.shape):>18} {str(b.shape):>6}")
print("done ->", OUT)
