#!/usr/bin/env python3
"""Reference + comparison harness for the channel-space cell-NMS kernel.

Compares two SuperPoint keypoint-detection strategies on the SAME detector
output (per-cell 8x8 = 64 channel scores), measuring ACCURACY and PERFORMANCE:

  A) cell-NMS (cell_nms.asm)   -- channel-space, depth-to-space FREE.
       per cell: peak = max_c p_c ; (a,b) = soft-argmax sub-pixel coordinate
       = sum_c softmax(T*p)_c * (c//8, c%8). One candidate per cell.

  B) dense NMS (standard)      -- depth-to-space -> H x W heatmap -> 2-D NMS.
       reshape the 64 channels into an 8x8 block per cell, then keep pixels
       that are the local max in a (2r+1)^2 window.

The kernel is run in the IPU emulator (wide-vector FP32); the dense path and the
soft-argmax formula are pure-Python references.

Run:  python kernels/superpoint_superglue/cell_nms_reference.py
"""
from __future__ import annotations
import math, struct, time, random

# ----------------------------- references -----------------------------------
def softmax(v):
    m = max(v); e = [math.exp(x - m) for x in v]; Z = sum(e); return [x / Z for x in e]

def cell_softargmax(p, T=64.0):
    """Pure-Python twin of cell_nms.asm: (peak, a, b)."""
    m = max(p)
    e = [2.0 ** (T * (pi - m)) for pi in p]; Z = sum(e); s = [ei / Z for ei in e]
    a = sum(s[c] * (c // 8) for c in range(64))
    b = sum(s[c] * (c % 8) for c in range(64))
    return m, a, b

def depth_to_space(cells, Hc, Wc, r=8):
    """cells[h][w] is a length-(r*r) vector -> heatmap[(Hc*r) x (Wc*r)]."""
    H, W = Hc * r, Wc * r
    hm = [[0.0] * W for _ in range(H)]
    for h in range(Hc):
        for w in range(Wc):
            v = cells[h][w]
            for c in range(r * r):
                a, b = c // r, c % r
                hm[h * r + a][w * r + b] = v[c]
    return hm

def dense_nms(hm, radius, thr):
    """Standard 2-D NMS: keep pixels that are the strict local max and > thr."""
    H, W = len(hm), len(hm[0]); pts = []
    for y in range(H):
        for x in range(W):
            v = hm[y][x]
            if v <= thr:
                continue
            is_max = True
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < H and 0 <= xx < W and hm[yy][xx] > v:
                        is_max = False; break
                if not is_max:
                    break
            if is_max:
                pts.append((y, x, v))
    return pts

def match_sets(A, B, tol=2.0):
    """Greedy 1-1 match of keypoints by pixel distance <= tol. Returns #matched."""
    used = [False] * len(B); m = 0
    for (ay, ax, _) in A:
        best, bj = tol + 1, -1
        for j, (by, bx, _) in enumerate(B):
            if used[j]:
                continue
            d = math.hypot(ay - by, ax - bx)
            if d < best:
                best, bj = d, j
        if bj >= 0 and best <= tol:
            used[bj] = True; m += 1
    return m

# --------------------------- IPU emulator runner ----------------------------
def make_kernel_runner():
    from ipu_emu.emulator import load_program, run_until_complete
    from ipu_emu.execute import decode_instruction_word
    from ipu_emu.ipu_state import IpuState, WideVectorArithmetic
    from ipu_emu.ipu_math import DType
    from ipu_as.lark_tree import assemble
    from ipu_as.label import reset_labels
    import os
    here = os.path.dirname(__file__)
    reset_labels()
    prog = [decode_instruction_word(w)
            for w in assemble(open(os.path.join(here, "cell_nms.asm")).read())]
    bits = lambda x: struct.unpack("<I", struct.pack("<f", x))[0]
    faaq = lambda s, i: struct.unpack("<f", struct.pack("<I", s.regfile.get_aaq(i) & 0xFFFFFFFF))[0]

    def run(p):
        s = IpuState(wide_vector_debug=True, wide_vector_arithmetic=WideVectorArithmetic.FP32)
        s.regfile.set_cr(15, DType.INT8); s.regfile.set_cr(0, 0)
        def wt(a, vals):
            b = [0.0] * 128
            for i, v in enumerate(vals): b[i] = v
            s.xmem.write_address(a, struct.pack("<128f", *b))
        wt(0x1000, p); wt(0x5000, [1.0] * 128)
        wt(0x6000, [c // 8 for c in range(64)]); wt(0x7000, [c % 8 for c in range(64)])
        for cr, v in [(2,0x1000),(3,0x2000),(4,0x5000),(5,0x3000),(6,0x6000),(7,0x7000),
                      (8,bits(-1.0)),(9,64),(10,bits(1.0/64)),(11,64)]:
            s.regfile.set_cr(cr, v)
        load_program(s, list(prog)); run_until_complete(s, max_cycles=100000)
        return faaq(s, 0), faaq(s, 2), faaq(s, 3), s.stats.total_cycles
    return run

# --------------------------------- main -------------------------------------
def main():
    random.seed(11)
    Hc, Wc, r = 16, 16, 8            # 128x128 image, 256 cells
    NKP = 30                         # planted ground-truth keypoints
    thr = 0.30                       # confidence threshold

    # --- synthetic detector output: peaky logits with planted peaks ---------
    gt = set()
    while len(gt) < NKP:
        gt.add((random.randrange(Hc), random.randrange(Wc), random.randrange(64)))
    gt = list(gt)
    cells = [[None] * Wc for _ in range(Hc)]
    gtmap = {(h, w): c for (h, w, c) in gt}
    for h in range(Hc):
        for w in range(Wc):
            logits = [random.gauss(0, 0.6) for _ in range(64)]
            if (h, w) in gtmap:
                logits[gtmap[(h, w)]] += random.uniform(5, 8)   # sharp planted peak
            cells[h][w] = softmax(logits)

    run = make_kernel_runner()

    # --- A) cell-NMS via kernel ---------------------------------------------
    t0 = time.time(); A = []; tot_cyc = 0; err = 0.0
    for h in range(Hc):
        for w in range(Wc):
            peak, a, b, cyc = run(cells[h][w]); tot_cyc += cyc
            rpeak, ra, rb = cell_softargmax(cells[h][w])
            err = max(err, abs(peak-rpeak), abs(a-ra), abs(b-rb))
            if peak > thr:
                A.append((h * r + a, w * r + b, peak))
    t_cell = time.time() - t0
    cyc_per_cell = tot_cyc // (Hc * Wc)

    # --- B) dense depth-to-space + 2-D NMS ----------------------------------
    hm = depth_to_space(cells, Hc, Wc, r)
    B = dense_nms(hm, radius=4, thr=thr)

    # --- ground-truth pixel coords ------------------------------------------
    G = [(h * r + c // r, w * r + c % r, cells[h][w][c]) for (h, w, c) in gt
         if cells[h][w][c] > thr]

    mA = match_sets(A, G); mB = match_sets(B, G)
    mAB = match_sets(A, B)

    print("="*70)
    print(f"Synthetic detector: {Hc}x{Wc} cells ({Hc*r}x{Wc*r} px), {len(G)} GT keypoints>{thr}")
    print("="*70)
    print("\n--- KERNEL CORRECTNESS ---")
    print(f"cell_nms kernel vs soft-argmax formula: max abs err = {err:.2e}  "
          f"({'EXACT' if err < 1e-3 else 'MISMATCH'})")
    print("\n--- ACCURACY (recall of planted keypoints, tol=2px) ---")
    print(f"A) cell-NMS (kernel):  {len(A):3d} detections, matched {mA}/{len(G)} GT = {100*mA/len(G):.1f}%")
    print(f"B) dense  NMS (r=4):   {len(B):3d} detections, matched {mB}/{len(G)} GT = {100*mB/len(G):.1f}%")
    print(f"A vs B agreement (tol=2px): {mAB}/{min(len(A),len(B))} mutual")
    print("\n--- PERFORMANCE (IPU emulator cycles) ---")
    print(f"A) cell-NMS:   {cyc_per_cell} cyc/cell  x {Hc*Wc} cells = {cyc_per_cell*Hc*Wc:,d} cyc")
    print(f"               (one pass: detect + localize, NO depth-to-space, NO gather)")
    print(f"B) dense path: depth-to-space (BLOCKED on-device / host) + maxpool NMS over")
    print(f"               {Hc*r*Wc*r//128} tiles x ~64 cyc + suppression test (BLOCKED: no vector compare)")
    dense_maxpool = (Hc*r*Wc*r//128)*64
    print(f"               maxpool-only lower bound = {dense_maxpool:,d} cyc (excludes reshape+compare)")
    print(f"\nspeedup (cell vs dense maxpool-only lower bound): "
          f"{dense_maxpool/(cyc_per_cell*Hc*Wc):.2f}x, and cell-NMS removes 2 blocked ops")

if __name__ == "__main__":
    main()
