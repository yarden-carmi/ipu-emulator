#!/usr/bin/env python3
"""Verify the hardware-faithful drop-in models: math-equivalent ops must match
the stock Magic Leap models to FP rounding; the two behavioural diffs (top-k cap,
optimal transport) are quantified. Run:
    python hw_models/check_hw_models.py
"""
import os, sys, glob, importlib.util
import numpy as np
import torch

HW = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HW, "..", "..", ".."))
SG = os.path.join(ROOT, "third_party", "superglue")
sys.path.insert(0, SG)
import models.superpoint as sp_stock          # stock
import models.superglue as sg_stock

def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
sp_hw = _load("sp_hw", os.path.join(HW, "superpoint.py"))
sg_hw = _load("sg_hw", os.path.join(HW, "superglue.py"))


def load_gray(p):
    from PIL import Image
    im = np.asarray(Image.open(p).convert("L"), np.float32) / 255.0
    return torch.from_numpy(im)[None, None]


if __name__ == "__main__":
    g = torch.Generator().manual_seed(0)
    print("== math-equivalent primitives (hw vs stock torch) ==")
    x = torch.randn(2, 65, 5, 5, generator=g)
    d = (sp_hw.exp2_softmax(x, 1) - torch.softmax(x, 1)).abs().max()
    print(f"  exp2 softmax  vs F.softmax:   {d:.2e}")
    v = torch.randn(2, 256, 7, generator=g)
    d = (sp_hw.l2_normalize(v, 1) - torch.nn.functional.normalize(v, p=2, dim=1)).abs().max()
    print(f"  rsqrt L2norm  vs F.normalize: {d:.2e}")
    # bilinear: explicit 4-corner vs F.grid_sample (via stock sample_descriptors)
    desc = torch.randn(1, 256, 30, 40, generator=g); kp = (torch.rand(1, 20, 2, generator=g) * torch.tensor([320., 240.]))
    a = sp_hw.sample_descriptors(kp.clone(), desc.clone(), 8)
    b = sp_stock.sample_descriptors(kp.clone(), desc.clone(), 8)
    print(f"  4-corner bilinear vs grid_sample: {(a-b).abs().max():.2e}")

    print("== full SuperPoint (hw vs stock), freiburg frame ==")
    fr = sorted(glob.glob(os.path.join(SG, "assets", "freiburg_sequence", "*.png")))
    im = load_gray(fr[0])
    cfg = {"max_keypoints": -1, "keypoint_threshold": 0.005, "nms_radius": 4}
    with torch.no_grad():
        ph = sp_hw.SuperPoint(cfg).eval()({"image": im})
        ps = sp_stock.SuperPoint(cfg).eval()({"image": im})
    kh, ks = ph["keypoints"][0].numpy(), ps["keypoints"][0].numpy()
    print(f"  keypoints: hw {len(kh)}  stock {len(ks)}  (max_keypoints=-1, no topk cap)")
    # match keypoints by coordinate, compare descriptors
    sset = {tuple(map(int, p)): i for i, p in enumerate(ks)}
    dh, ds = ph["descriptors"][0].numpy(), ps["descriptors"][0].numpy()
    diffs = [np.abs(dh[:, i] - ds[:, sset[tuple(map(int, p))]]).max()
             for i, p in enumerate(kh) if tuple(map(int, p)) in sset]
    common = len(diffs)
    print(f"  common keypoints: {common}/{len(kh)}   descriptor max-diff: {max(diffs):.2e}")

    print("== full SuperGlue (hw vs stock) on identical keypoints ==")
    im1 = load_gray(fr[4])
    with torch.no_grad():
        f0 = sp_stock.SuperPoint({"max_keypoints": 256}).eval()({"image": im})
        f1 = sp_stock.SuperPoint({"max_keypoints": 256}).eval()({"image": im1})
    data = {"image0": im, "image1": im1,
            "keypoints0": f0["keypoints"][0][None], "keypoints1": f1["keypoints"][0][None],
            "scores0": f0["scores"][0][None], "scores1": f1["scores"][0][None],
            "descriptors0": f0["descriptors"][0][None], "descriptors1": f1["descriptors"][0][None]}
    with torch.no_grad():
        mh = sg_hw.SuperGlue({"weights": "indoor"}).eval()(data)["matches0"][0].numpy()
        ms = sg_stock.SuperGlue({"weights": "indoor"}).eval()(data)["matches0"][0].numpy()
    both = (mh >= 0) & (ms >= 0)
    print(f"  matches: hw {(mh>=0).sum()}  stock {(ms>=0).sum()}  "
          f"same target (both): {(mh[both]==ms[both]).mean()*100:.1f}%  ({both.sum()})")
