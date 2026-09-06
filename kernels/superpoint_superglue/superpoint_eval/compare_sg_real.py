#!/usr/bin/env python3
"""SuperGlue end-to-end on a REAL image pair (SuperGlue's own freiburg office
demo frames), addressing the 'synthetic descriptors' caveat of compare_sg.py.

Pipeline: official SuperPoint frontend extracts real keypoints/descriptors from
two frames -> SuperGlue stock (log-Sinkhorn) vs our tropical max-Sinkhorn
(sinkhorn_iter.asm + sinkhorn_col.asm). Keypoints capped so the (N+1)x(N+1)
coupling fits one 128-lane tile, so the REAL coupling is also run through the
actual .asm Sinkhorn kernels and checked against the numpy tropical reference.

Run: python superpoint_superglue/superpoint_eval/compare_sg_real.py
"""
import os, sys, glob
import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "third_party", "superglue"))
sys.path.insert(0, HERE)
import models.superglue as sgmod
from models.superglue import SuperGlue
from models.superpoint import SuperPoint
from compare_sg import tropical_sinkhorn, asm_tropical, couplings

SEQ = os.path.join(ROOT, "third_party", "superglue", "assets", "freiburg_sequence")
MAXK = 120          # keep N+1 <= 128 so the real coupling runs on the asm kernels


def load_gray(path):
    im = np.asarray(Image.open(path).convert("L"), np.float32) / 255.0
    return torch.from_numpy(im)[None, None]


def run_sg(data, sinkhorn_fn, sg):
    orig = sgmod.log_sinkhorn_iterations
    sgmod.log_sinkhorn_iterations = sinkhorn_fn
    try:
        with torch.no_grad():
            out = sg(data)
            k0 = sgmod.normalize_keypoints(data["keypoints0"], data["image0"].shape)
            k1 = sgmod.normalize_keypoints(data["keypoints1"], data["image1"].shape)
            de0 = data["descriptors0"] + sg.kenc(k0, data["scores0"])
            de1 = data["descriptors1"] + sg.kenc(k1, data["scores1"])
            de0, de1 = sg.gnn(de0, de1)
            md0, md1 = sg.final_proj(de0), sg.final_proj(de1)
            scores = torch.einsum("bdn,bdm->bnm", md0, md1) / sg.config["descriptor_dim"] ** .5
    finally:
        sgmod.log_sinkhorn_iterations = orig
    return out, scores, float(sg.bin_score)


if __name__ == "__main__":
    frames = sorted(glob.glob(os.path.join(SEQ, "*.png")))
    img0, img1 = load_gray(frames[0]), load_gray(frames[4])   # ~real camera motion
    print(f"REAL pair: {os.path.basename(frames[0])} <-> {os.path.basename(frames[4])}  "
          f"{img0.shape[-2]}x{img0.shape[-1]}")

    sp = SuperPoint({"max_keypoints": MAXK, "keypoint_threshold": 0.005, "nms_radius": 4}).eval()
    with torch.no_grad():
        f0 = sp({"image": img0}); f1 = sp({"image": img1})
    data = {
        "keypoints0": f0["keypoints"][0][None], "keypoints1": f1["keypoints"][0][None],
        "descriptors0": f0["descriptors"][0][None], "descriptors1": f1["descriptors"][0][None],
        "scores0": f0["scores"][0][None], "scores1": f1["scores"][0][None],
        "image0": img0, "image1": img1,
    }
    N0 = data["keypoints0"].shape[1]; N1 = data["keypoints1"].shape[1]
    print(f"  SuperPoint keypoints: {N0} / {N1}")

    sg = SuperGlue({"weights": "indoor"}).eval()    # indoor = closer to office scene
    out_log, scores, bs = run_sg(data, sgmod.log_sinkhorn_iterations, sg)
    out_trop, _, _ = run_sg(data, tropical_sinkhorn, sg)

    m_log = out_log["matches0"][0].numpy(); m_trop = out_trop["matches0"][0].numpy()
    both = (m_log >= 0) & (m_trop >= 0)
    print(f"  stock log-Sinkhorn valid matches: {(m_log>=0).sum()}/{N0}")
    print(f"  our  max-Sinkhorn  valid matches: {(m_trop>=0).sum()}/{N0}")
    print(f"  matches0 agreement (incl no-match): {(m_log==m_trop).mean()*100:.1f}%")
    print(f"  pairs valid in BOTH, same target:   "
          f"{(m_log[both]==m_trop[both]).mean()*100:.1f}%  ({both.sum()} pairs)")

    # real coupling through the actual asm Sinkhorn kernels
    if N1 + 1 <= 128:
        Z0 = couplings(scores, bs)[0].numpy().astype(np.float32)
        ITERS = 5
        ref = tropical_sinkhorn(torch.from_numpy(Z0[None]), None, None, ITERS)[0].numpy()
        got = asm_tropical(Z0, ITERS)
        d = np.abs(got - ref).max()
        print(f"  sinkhorn_iter.asm+sinkhorn_col.asm vs numpy tropical (REAL "
              f"{Z0.shape[0]}x{Z0.shape[1]}, {ITERS} it): max {d:.2e} "
              f"{'MATCH' if d<1e-3 else 'FAIL'}")
