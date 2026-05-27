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
import models.superpoint as spmod
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


def hw_top_k_keypoints(keypoints, scores, k):
    """SuperPoint top-k keypoint cap as the hardware does it: no sort (the ISA
    has none) -- topk_mt.asm emits the soft survivor count Sum sigmoid(T(s-tau))
    and the HOST binary-searches tau so the count ~ k, then keeps {s > tau}
    (topk.asm: relu(s-tau)). The selected set equals torch.topk's set up to the
    calibration precision + ties (tau converges to the k-th largest score)."""
    if k >= len(keypoints):
        return keypoints, scores
    T = 64.0
    lo, hi = float(scores.min()), float(scores.max())
    for _ in range(24):                                   # host bisection on tau
        mid = 0.5 * (lo + hi)
        cnt = torch.sigmoid(T * (scores - mid)).sum().item()   # device soft count
        lo, hi = (mid, hi) if cnt > k else (lo, mid)
    tau = 0.5 * (lo + hi)
    mask = scores > tau                                   # topk.asm threshold gate
    return keypoints[mask], scores[mask]


def apply_hw_math(exp2_softmax=True, tropical_ot=True, hw_topk=True):
    """Switch SuperPoint+SuperGlue to the hardware kernels' math (in place)."""
    if tropical_ot:
        _ORIG.setdefault("ot", sgmod.log_optimal_transport)
        sgmod.log_optimal_transport = hw_log_optimal_transport
    if exp2_softmax:
        _ORIG.setdefault("attn", sgmod.attention)
        sgmod.attention = hw_attention
    if hw_topk:
        _ORIG.setdefault("topk", spmod.top_k_keypoints)
        spmod.top_k_keypoints = hw_top_k_keypoints


def restore_stock_math():
    if "ot" in _ORIG:
        sgmod.log_optimal_transport = _ORIG.pop("ot")
    if "attn" in _ORIG:
        sgmod.attention = _ORIG.pop("attn")
    if "topk" in _ORIG:
        spmod.top_k_keypoints = _ORIG.pop("topk")


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

    m = build_matching(cfg)
    m = build_matching(cfg)
    def run(data):
        with torch.no_grad():
            return m(data)                                   # raw pred (tensors/lists)
    def npy(pred):
        return {k: pred[k][0].detach().cpu().numpy() for k in pred}   # [0] = drop batch / list

    # full pipelines (hardware top-k changes the keypoint COUNT, so draw each on
    # its own keypoints; counts reported separately)
    restore_stock_math(); raw_s = run({"image0": im0, "image1": im1}); stock = npy(raw_s)
    apply_hw_math();       hw = npy(run({"image0": im0, "image1": im1})); restore_stock_math()
    print("FULL pipeline (SuperPoint topk + SuperGlue OT):")
    print(f"  keypoints  stock {len(stock['keypoints0'])}/{len(stock['keypoints1'])}"
          f"   hardware {len(hw['keypoints0'])}/{len(hw['keypoints1'])}   (hw topk = calibrated threshold)")
    print(f"  matches    stock {(stock['matches0']>=0).sum()}   hardware {(hw['matches0']>=0).sum()}")
    ns = _draw_matches(None, p0, p1, stock["keypoints0"], stock["keypoints1"], stock["matches0"], "/tmp/matches_stock.png")
    nh = _draw_matches(None, p0, p1, hw["keypoints0"], hw["keypoints1"], hw["matches0"], "/tmp/matches_hw.png")
    print(f"  wrote /tmp/matches_stock.png ({ns}), /tmp/matches_hw.png ({nh})")

    # isolated OT comparison on IDENTICAL keypoints (feed stock features to both)
    fixed = {"image0": im0, "image1": im1}
    for i in "01":
        for a in ("keypoints", "scores", "descriptors"):
            v = raw_s[a + i]
            fixed[a + i] = (v[0] if isinstance(v, (list, tuple)) else v[0])[None]
    restore_stock_math(); s = npy(run(fixed))
    apply_hw_math();       h = npy(run(fixed)); restore_stock_math()
    ms, mh = s["matches0"], h["matches0"]; both = (ms >= 0) & (mh >= 0)
    print("Isolated OT (same keypoints fed to both):")
    print(f"  matches    stock {(ms>=0).sum()}   hardware {(mh>=0).sum()}")
    print(f"  same target when both match: {(ms[both]==mh[both]).mean()*100:.1f}%  ({both.sum()} pairs)")
