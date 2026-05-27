#!/usr/bin/env python3
"""Hardware-faithful SuperPoint + SuperGlue in PyTorch.

Patches the official Magic Leap models so the math matches EXACTLY what our .asm
kernels compute, so you can run/test the real demo with the hardware behaviour:

  * attention softmax  -> base-2 (exp2) with a log2(e) pre-scale   [softmax.asm]
      (2^(x*log2 e) = e^x, so this is numerically identical to stock softmax)
  * optimal transport  -> tropical max-plus Sinkhorn               [sinkhorn_iter.asm
      + sinkhorn_col.asm]:  Z <- Z - max_j Z ; Z <- Z - max_i Z, T iters, on the
      dustbin-augmented coupling, NO log-marginal (log_mu/log_nu) terms. This is
      the one behavioural difference: the assignment is more permissive than the
      stock log-domain optimal transport.

Everything else (convolutions, the GNN matmuls, grid_sample descriptor sampling,
L2 normalisation, max-pool NMS) is FP32-exact on hardware, so it is left as-is.

Usage:
    from superglue_hw import build_matching, apply_hw_math
    m = build_matching({'superpoint': {...}, 'superglue': {'weights': 'indoor'}})
    apply_hw_math()           # switch the math to the hardware kernels' behaviour
    pred = m({'image0': im0, 'image1': im1})

Or run directly to compare stock vs hardware on an image pair and draw matches:
    python superglue_hw.py [img0.png img1.png]      (defaults to the freiburg pair)
"""
import os, sys, glob, math
import torch

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
SGDIR = os.path.join(ROOT, "third_party", "superglue")
sys.path.insert(0, SGDIR)
import models.superglue as sgmod
from models.matching import Matching

LOG2E = 1.4426950408889634
_ORIG = {}


# ---- hardware kernel math ----------------------------------------------------
def hw_attention(query, key, value):
    """attention() with base-2 softmax (softmax.asm). Identical to stock up to FP."""
    dim = query.shape[1]
    scores = torch.einsum("bdhn,bdhm->bhnm", query, key) / dim ** .5
    # softmax via exp2: 2^(log2e*(s-m)) / sum
    s = scores * LOG2E
    s = s - s.max(dim=-1, keepdim=True).values
    e = torch.pow(2.0, s)
    prob = e / e.sum(dim=-1, keepdim=True)
    return torch.einsum("bhnm,bdhm->bdhn", prob, value), prob


def hw_log_optimal_transport(scores, alpha, iters):
    """Tropical (max-plus) Sinkhorn on the dustbin-augmented coupling -- exactly
    sinkhorn_iter.asm (row) + sinkhorn_col.asm (col), T iterations. No marginals."""
    b, m, n = scores.shape
    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    a = alpha.expand(b, 1, 1)
    Z = torch.cat([torch.cat([scores, bins0], -1),
                   torch.cat([bins1, a], -1)], 1)        # (b, m+1, n+1)
    for _ in range(iters):
        Z = Z - Z.max(dim=2, keepdim=True).values        # row half-step  (sinkhorn_iter)
        Z = Z - Z.max(dim=1, keepdim=True).values        # col half-step  (sinkhorn_col)
    return Z


def apply_hw_math(exp2_softmax=True, tropical_ot=True):
    """Switch the SuperGlue module to the hardware kernels' math (in place)."""
    if tropical_ot:
        _ORIG.setdefault("ot", sgmod.log_optimal_transport)
        sgmod.log_optimal_transport = hw_log_optimal_transport
    if exp2_softmax:
        _ORIG.setdefault("attn", sgmod.attention)
        sgmod.attention = hw_attention


def restore_stock_math():
    if "ot" in _ORIG:
        sgmod.log_optimal_transport = _ORIG.pop("ot")
    if "attn" in _ORIG:
        sgmod.attention = _ORIG.pop("attn")


def build_matching(config):
    return Matching(config).eval()


# ---- runnable demo -----------------------------------------------------------
def _load_gray(path):
    from PIL import Image
    import numpy as np
    im = __import__("numpy").asarray(Image.open(path).convert("L"), dtype="float32") / 255.0
    return torch.from_numpy(im)[None, None], im.shape[::-1]   # tensor, (W,H)


def _draw_matches(path, im0p, im1p, kp0, kp1, matches, out):
    from PIL import Image, ImageDraw
    a = Image.open(im0p).convert("RGB"); b = Image.open(im1p).convert("RGB")
    W0, H0 = a.size; W1, H1 = b.size; H = max(H0, H1)
    canvas = Image.new("RGB", (W0 + W1, H), (0, 0, 0))
    canvas.paste(a, (0, 0)); canvas.paste(b, (W0, 0))
    d = ImageDraw.Draw(canvas)
    n = 0
    for i, j in enumerate(matches):
        if j < 0:
            continue
        x0, y0 = float(kp0[i][0]), float(kp0[i][1])
        x1, y1 = float(kp1[j][0]) + W0, float(kp1[j][1])
        d.line([(x0, y0), (x1, y1)], fill=(0, 255, 0), width=1)
        d.ellipse([x0 - 2, y0 - 2, x0 + 2, y0 + 2], outline=(255, 80, 0))
        d.ellipse([x1 - 2, y1 - 2, x1 + 2, y1 + 2], outline=(255, 80, 0))
        n += 1
    canvas.save(out)
    return n


if __name__ == "__main__":
    import numpy as np
    args = sys.argv[1:]
    if len(args) >= 2:
        p0, p1 = args[0], args[1]
    else:
        fr = sorted(glob.glob(os.path.join(SGDIR, "assets", "freiburg_sequence", "*.png")))
        p0, p1 = fr[0], fr[4]
    print(f"image pair:\n  {p0}\n  {p1}")
    im0, _ = _load_gray(p0); im1, _ = _load_gray(p1)
    cfg = {"superpoint": {"max_keypoints": 256, "keypoint_threshold": 0.005, "nms_radius": 4},
           "superglue": {"weights": "indoor"}}

    def run():
        m = build_matching(cfg)
        with torch.no_grad():
            pred = m({"image0": im0, "image1": im1})
        return {k: (v[0].cpu().numpy() if torch.is_tensor(v) else v[0]) for k, v in pred.items()}

    restore_stock_math()
    stock = run()
    apply_hw_math()
    hw = run()
    restore_stock_math()

    kp0, kp1 = stock["keypoints0"], stock["keypoints1"]
    m_s, m_h = stock["matches0"], hw["matches0"]
    both = (m_s >= 0) & (m_h >= 0)
    print(f"keypoints: {len(kp0)} / {len(kp1)}")
    print(f"  stock (log-Sinkhorn) matches:   {(m_s>=0).sum()}")
    print(f"  hardware (max-Sinkhorn) matches:{(m_h>=0).sum()}")
    print(f"  matches0 agreement (incl -1):    {(m_s==m_h).mean()*100:.1f}%")
    print(f"  same target when both match:     "
          f"{(m_s[both]==m_h[both]).mean()*100:.1f}%  ({both.sum()} pairs)")
    ns = _draw_matches(None, p0, p1, kp0, kp1, m_s, "/tmp/matches_stock.png")
    nh = _draw_matches(None, p0, p1, kp0, kp1, m_h, "/tmp/matches_hw.png")
    print(f"wrote /tmp/matches_stock.png ({ns} lines), /tmp/matches_hw.png ({nh} lines)")
