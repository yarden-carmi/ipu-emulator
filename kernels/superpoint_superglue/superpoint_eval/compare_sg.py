#!/usr/bin/env python3
"""SuperGlue end-to-end validation against the official pretrained model.

Closes the one real gap: our pipeline differs from official SuperGlue only in the
optimal-transport step -- the kernels use the tropical (max-plus) Sinkhorn
(sinkhorn_iter.asm row + sinkhorn_col.asm col: Z -= max_j ; Z -= max_i) instead
of the log-domain logsumexp Sinkhorn. (base-2 softmax with a log2(e) pre-scale is
exactly natural softmax, so attention is bit-exact.)

This script:
  1. loads pretrained SuperGlue (third_party/superglue) and runs the FULL stock
     forward on a structured keypoint pair;
  2. runs the SAME forward with log_sinkhorn_iterations swapped for the tropical
     (max) version our .asm kernels implement, and compares the assignment matrix
     argmax + the final matched pairs (the actual SuperGlue output);
  3. confirms the real sinkhorn_iter.asm + sinkhorn_col.asm reproduce the numpy
     tropical iteration on the true (dustbin-augmented) coupling matrix.

Run: python superpoint_superglue/superpoint_eval/compare_sg.py
"""
import os, sys, struct
import numpy as np
import torch

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "third_party", "superglue"))
sys.path.insert(0, HERE)
import models.superglue as sgmod
from models.superglue import SuperGlue


# ---- tropical (max-plus) Sinkhorn -- exactly what our two kernels compute ----
def tropical_sinkhorn(Z, log_mu, log_nu, iters):
    for _ in range(iters):
        Z = Z - Z.max(dim=2, keepdim=True).values     # row: Z_ij -= max_j  (sinkhorn_iter.asm)
        Z = Z - Z.max(dim=1, keepdim=True).values     # col: Z_ij -= max_i  (sinkhorn_col.asm)
    return Z


def make_inputs(N=64, M=48, D=256, W=640, H=480, seed=0):
    g = torch.Generator().manual_seed(seed)
    rand = lambda *s: torch.rand(*s, generator=g)
    randn = lambda *s: torch.randn(*s, generator=g)
    kpts0 = rand(1, N, 2) * torch.tensor([[W, H]])
    d0 = torch.nn.functional.normalize(randn(1, D, N), dim=1)
    perm = torch.randperm(N, generator=g)[:M]
    d1 = torch.nn.functional.normalize(randn(1, D, N), dim=1)
    d1[:, :, :M] = torch.nn.functional.normalize(d0[:, :, perm] + 0.15 * randn(1, D, M), dim=1)
    kpts1 = rand(1, N, 2) * torch.tensor([[W, H]])
    kpts1[:, :M, :] = kpts0[:, perm, :] + 4.0 * randn(1, M, 2)
    data = {"keypoints0": kpts0, "keypoints1": kpts1,
            "descriptors0": d0, "descriptors1": d1,
            "scores0": rand(1, N), "scores1": rand(1, N),
            "image0": torch.zeros(1, 1, H, W), "image1": torch.zeros(1, 1, H, W)}
    return data


def run_model(data, sinkhorn_fn):
    sg = SuperGlue({"weights": "outdoor"}).eval()
    orig = sgmod.log_sinkhorn_iterations
    sgmod.log_sinkhorn_iterations = sinkhorn_fn
    try:
        with torch.no_grad():
            out = sg(data)
            # recompute the score matrix (pre-OT) for the asm check
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


# ---- run the real Sinkhorn .asm kernels on the dustbin-augmented matrix ------
def asm_tropical(Z0, iters):
    from ipu_emu.emulator import load_program, run_until_complete
    from ipu_emu.execute import decode_instruction_word
    from ipu_emu.ipu_state import IpuState, WideVectorArithmetic
    from ipu_emu.ipu_math import DType
    from ipu_as.lark_tree import assemble
    from ipu_as.label import reset_labels
    KD = os.path.join(HERE, "..")
    ROW = open(os.path.join(KD, "sinkhorn_iter.asm")).read()
    COL = open(os.path.join(KD, "sinkhorn_col.asm")).read()
    n = Z0.shape[0]
    assert n <= 128
    s = IpuState(wide_vector_debug=True, wide_vector_arithmetic=WideVectorArithmetic.FP32)
    s.regfile.set_cr(15, DType.INT8)
    MAT = 0x10000
    def write_mat(Z):
        for r in range(n):
            row = list(Z[r]) + [0.0] * (128 - n)
            s.xmem.write_address(MAT + r * 512, struct.pack("<128f", *row))
    def read_mat():
        out = np.zeros((n, n), np.float32)
        for r in range(n):
            d = s.xmem.read_address(MAT + r * 512, 512)
            for c in range(n):
                out[r, c] = struct.unpack_from("<f", d, c * 4)[0]
        return out
    for a, v in [(0x4000, 1.0), (0x5000, -1.0), (0x8000, -3.4e38)]:
        s.xmem.write_address(a, struct.pack("<128f", *([v] * 128)))
    write_mat(Z0.astype(np.float32))
    row_crs = [(0, 0), (1, 1), (2, MAT), (4, 0x4000), (6, n), (7, 512), (8, n), (9, 0xBF800000)]
    col_crs = [(0, 0), (1, 1), (2, MAT), (4, 0x4000), (5, 0x5000), (6, n), (7, 512),
               (8, n), (9, 0x6000), (10, 0x7000), (11, 0x8000)]
    def launch(asm, crs):
        reset_labels()
        prog = [decode_instruction_word(w) for w in assemble(asm)]
        for cr, v in crs:
            s.regfile.set_cr(cr, v)
        s.program_counter = 0
        load_program(s, list(prog))
        run_until_complete(s, max_cycles=5_000_000)
    for _ in range(iters):
        launch(ROW, row_crs)
        launch(COL, col_crs)
    return read_mat()


def couplings(scores, bin_score):
    b, m, n = scores.shape
    a = torch.full((b, m, 1), bin_score)
    Z = torch.cat([torch.cat([scores, a], -1),
                   torch.cat([torch.full((b, 1, n), bin_score),
                              torch.full((b, 1, 1), bin_score)], -1)], 1)
    return Z


if __name__ == "__main__":
    print("SuperGlue end-to-end vs pretrained 'outdoor'  (N=64 kpts/image)")
    print("our pipeline = official EXCEPT optimal transport: tropical max-Sinkhorn")
    print("(sinkhorn_iter.asm row + sinkhorn_col.asm col) vs official log-domain OT.\n")
    agrees, sames, nlog, ntrop = [], [], [], []
    last = None
    for seed in range(5):
        data = make_inputs(seed=seed)
        out_log, scores, bs = run_model(data, sgmod.log_sinkhorn_iterations)
        out_trop, _, _ = run_model(data, tropical_sinkhorn)
        m_log = out_log["matches0"][0].numpy(); m_trop = out_trop["matches0"][0].numpy()
        both = (m_log >= 0) & (m_trop >= 0)
        agrees.append((m_log == m_trop).mean())
        sames.append((m_log[both] == m_trop[both]).mean() if both.any() else np.nan)
        nlog.append(int((m_log >= 0).sum())); ntrop.append(int((m_trop >= 0).sum()))
        last = (scores, bs)
    print(f"  stock log-Sinkhorn valid matches:  mean {np.mean(nlog):.1f}/64")
    print(f"  our  max-Sinkhorn  valid matches:  mean {np.mean(ntrop):.1f}/64")
    print(f"  matches0 agreement (incl. no-match): {np.mean(agrees)*100:.1f}%  "
          f"(range {min(agrees)*100:.0f}-{max(agrees)*100:.0f}%)")
    print(f"  pairs valid in BOTH, same target:    {np.nanmean(sames)*100:.1f}%")

    # ---- asm kernels reproduce the numpy tropical iteration on real couplings
    scores, bs = last
    Z0 = couplings(scores, bs)[0].numpy().astype(np.float32)
    ITERS = 5
    ref = Z0.copy()[None]
    ref = tropical_sinkhorn(torch.from_numpy(ref), None, None, ITERS)[0].numpy()
    got = asm_tropical(Z0, ITERS)
    d = np.abs(got - ref).max()
    print(f"  sinkhorn_iter.asm+sinkhorn_col.asm vs numpy tropical ({ITERS} iters, "
          f"{Z0.shape[0]}x{Z0.shape[1]}): max {d:.2e} {'MATCH' if d<1e-3 else 'FAIL'}")
