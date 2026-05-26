#!/usr/bin/env python3
"""grid_sample (descriptor bilinear interpolation) recreated as precomputed
weights + convolution (weighted-shift accumulate) for a FIXED resolution.

SuperPoint samples the coarse (Cd x Hc x Wc) descriptor map at keypoints with
F.grid_sample(bilinear, align_corners=True). At a FIXED image size the mapping
full-res pixel (X,Y) -> coarse coord is data-INDEPENDENT, so every output
pixel's 4 neighbour coarse-indices and bilinear weights are CONSTANTS, computed
once. The interpolation then becomes:

    out[c,Y,X] = sum_{dy,dx in {0,1}} Wd[dy,dx][Y,X] * coarse[c, i0(Y)+dy, j0(X)+dx]

i.e. 4 element-wise multiplies of (a fixedly-resampled coarse map) by precomputed
weight fields, accumulated -- expressible on the IPU with MULT.EE + ACC and
precomputed weights (no data-dependent gather; the resample indices are fixed).

This module: (1) validates the precomputed form == torch grid_sample; (2) runs
the 4-corner weighted accumulate on the IPU emulator for a small case.

Run: python kernels/superpoint_superglue/superpoint_eval/grid_sample_conv.py
"""
import os, sys, struct, numpy as np
HERE = os.path.dirname(__file__)

def precompute(Hc, Wc, H, W):
    """Per output pixel (Y,X): floor coarse indices i0,j0 and the 4 bilinear
    weight fields. align_corners=True mapping (matches SuperPoint demo)."""
    # grid_sample align_corners=True: norm in [-1,1] -> coarse idx in [0,size-1]
    # SuperPoint: norm_x = X/(W/2)-1 -> coarse_x = (norm_x+1)/2*(Wc-1) = X*(Wc-1)/W ... via /2
    cx = (np.arange(W) / (W / 2.0) - 1.0 + 1.0) / 2.0 * (Wc - 1)   # (W,)
    cy = (np.arange(H) / (H / 2.0) - 1.0 + 1.0) / 2.0 * (Hc - 1)   # (H,)
    j0 = np.clip(np.floor(cx).astype(int), 0, Wc - 2); fx = cx - j0
    i0 = np.clip(np.floor(cy).astype(int), 0, Hc - 2); fy = cy - i0
    FY, FX = np.meshgrid(fy, fx, indexing="ij")     # (H,W)
    Wd = {(0, 0): (1 - FY) * (1 - FX), (0, 1): (1 - FY) * FX,
          (1, 0): FY * (1 - FX),       (1, 1): FY * FX}
    return i0, j0, Wd

def bilinear_ref(C, Hc, Wc, H, W):
    """Precomputed bilinear resample (the 4-corner weighted sum)."""
    i0, j0, Wd = precompute(Hc, Wc, H, W)
    Cd = C.shape[0]
    out = np.zeros((Cd, H, W), np.float32)
    II, JJ = np.meshgrid(i0, j0, indexing="ij")     # (H,W) coarse row/col floor
    for (dy, dx), w in Wd.items():
        out += C[:, np.clip(II + dy, 0, Hc - 1), np.clip(JJ + dx, 0, Wc - 1)] * w[None]
    return out

def interp_matrices(Hc, Wc, H, W):
    """Precomputed 1D bilinear interpolation matrices Mh (HxHc), Mw (WxWc) for the
    fixed-resolution align_corners mapping. grid_sample is separable on a regular
    grid, so out = Mh @ coarse @ Mw.T (per channel). Each row of Mh/Mw has exactly
    two nonzeros (the bilinear pair) -- DATA-INDEPENDENT, computed once."""
    cx = (np.arange(W) / (W / 2.0) - 1.0 + 1.0) / 2.0 * (Wc - 1)
    cy = (np.arange(H) / (H / 2.0) - 1.0 + 1.0) / 2.0 * (Hc - 1)
    jj = np.clip(np.floor(cx).astype(int), 0, Wc - 2); fx = cx - jj
    ii = np.clip(np.floor(cy).astype(int), 0, Hc - 2); fy = cy - ii
    Mw = np.zeros((W, Wc), np.float32)
    Mh = np.zeros((H, Hc), np.float32)
    for X in range(W): Mw[X, jj[X]] += 1 - fx[X]; Mw[X, jj[X] + 1] += fx[X]
    for Y in range(H): Mh[Y, ii[Y]] += 1 - fy[Y]; Mh[Y, ii[Y] + 1] += fy[Y]
    return Mh, Mw

def matmul_ref(C, Hc, Wc, H, W):
    """grid_sample as two matmuls by the precomputed constant matrices."""
    Mh, Mw = interp_matrices(Hc, Wc, H, W)
    return np.stack([Mh @ C[c] @ Mw.T for c in range(C.shape[0])])

def torch_grid_sample(C, H, W):
    import torch, torch.nn.functional as F
    gx = (np.arange(W) / (W / 2.0) - 1.0).astype(np.float32)   # (W,)
    gy = (np.arange(H) / (H / 2.0) - 1.0).astype(np.float32)   # (H,)
    GX = np.broadcast_to(gx[None, :], (H, W))
    GY = np.broadcast_to(gy[:, None], (H, W))
    grid = np.stack([GX, GY], axis=-1).astype(np.float32)      # (H,W,2)
    grid = torch.from_numpy(grid)[None]                        # (1,H,W,2)
    t = F.grid_sample(torch.from_numpy(C)[None], grid, mode="bilinear", align_corners=True)
    return t[0].numpy()

# -------- IPU: GATHER-FREE separable matmul (precomputed interp matrices) ------
def ipu_bilinear_matmul(C, Hc, Wc, H, W):
    """Run the FULLY ON-DEVICE gather-free grid_sample: out = Mh @ coarse @ Mw.T,
    two matmuls by the precomputed constant interpolation matrices (no host gather
    -- the column/row resample stays lane-aligned via MULT.VE.CYCLIC scalar-from-R0
    times a precomputed basis row in R_CYCLIC, accumulated). W<=128, 1 channel.

    Stage 1 (column resample): Tcol[ic] = sum_jc coarse[ic,jc] * Mw[:,jc]  (W lanes).
    Stage 2 (row resample):    out[Y]   = sum_ic Mh[Y,ic]    * Tcol[ic]    (W lanes).
    Both are the matmul MAC pattern: scalar (R0[idx]) x precomputed basis (R_CYCLIC),
    producing lane-aligned output rows -- no gather, no scatter, host-free at run."""
    sys.path.insert(0, HERE)
    from ipu_emu.emulator import load_program, run_until_complete
    from ipu_emu.execute import decode_instruction_word
    from ipu_emu.ipu_state import IpuState, WideVectorArithmetic
    from ipu_emu.ipu_math import DType
    from ipu_as.lark_tree import assemble
    from ipu_as.label import reset_labels
    assert W <= 128 and C.shape[0] == 1
    Mh, Mw = interp_matrices(Hc, Wc, H, W)
    asm = """\
SET lr10 cr0 ;;
SET lr0 cr0 ;; SET lr1 cr0 ;; SET lr2 cr0 ;;
s1_row:
LDR_MULT_REG r0 lr1 cr3 ;;
RESET_ACC ;;
SET lr3 cr0 ;; SET lr4 cr0 ;; SET lr5 cr0 ;;
s1_col:
LDR_CYCLIC_MULT_REG lr4 cr4 lr10 ; MULT.VE.CYCLIC lr10 0 lr10 lr5 ; ACC ;;
ADD lr4 lr4 cr2 ; ADD lr5 lr5 cr1 ; ADD lr3 lr3 cr1 ;;
BLT lr3 cr7 s1_col ;;
STR_ACC_REG lr2 cr5 ;;
ADD lr1 lr1 cr2 ; ADD lr2 lr2 cr2 ; ADD lr0 lr0 cr1 ;;
BLT lr0 cr6 s1_row ;;
SET lr0 cr0 ;; SET lr1 cr0 ;; SET lr2 cr0 ;;
s2_row:
LDR_MULT_REG r0 lr1 cr8 ;;
RESET_ACC ;;
SET lr3 cr0 ;; SET lr4 cr0 ;; SET lr5 cr0 ;;
s2_col:
LDR_CYCLIC_MULT_REG lr4 cr5 lr10 ; MULT.VE.CYCLIC lr10 0 lr10 lr5 ; ACC ;;
ADD lr4 lr4 cr2 ; ADD lr5 lr5 cr1 ; ADD lr3 lr3 cr1 ;;
BLT lr3 cr6 s2_col ;;
STR_ACC_REG lr2 cr9 ;;
ADD lr1 lr1 cr2 ; ADD lr2 lr2 cr2 ; ADD lr0 lr0 cr1 ;;
BLT lr0 cr10 s2_row ;;
BKPT ;;
"""
    reset_labels(); prog = [decode_instruction_word(w) for w in assemble(asm)]
    s = IpuState(wide_vector_debug=True, wide_vector_arithmetic=WideVectorArithmetic.FP32)
    s.regfile.set_cr(15, DType.INT8)
    RS = ((max(H, Hc, Wc) * 512 + 0xFFFF) // 0x10000) * 0x10000   # per-region stride
    COARSE, BW, TCOL, MH, OUT = 0, RS, 2 * RS, 3 * RS, 4 * RS
    def wr(base, row, n):
        v = [0.0] * 128
        for x in range(n): v[x] = float(row[x])
        s.xmem.write_address(base, struct.pack("<128f", *v))
    for ic in range(Hc): wr(COARSE + ic * 512, C[0, ic], Wc)   # coarse rows (Wc lanes)
    for jc in range(Wc): wr(BW + jc * 512, Mw[:, jc], W)        # Mw column-basis (W lanes)
    for Y in range(H):  wr(MH + Y * 512, Mh[Y], Hc)            # Mh rows (Hc lanes)
    for cr, v in [(0, 0), (1, 1), (2, 512), (3, COARSE), (4, BW), (5, TCOL),
                  (6, Hc), (7, Wc), (8, MH), (9, OUT), (10, H)]:
        s.regfile.set_cr(cr, v)
    load_program(s, list(prog)); run_until_complete(s, max_cycles=5_000_000)
    ipu_bilinear_matmul.cycles = s.stats.total_cycles
    out = np.zeros((H, W), np.float32)
    for y in range(H):
        d = s.xmem.read_address(OUT + y * 512, 512)
        for x in range(W): out[y, x] = struct.unpack_from("<f", d, x * 4)[0]
    return out

# ---------------- IPU: 4-corner weighted accumulate (MULT.EE + ACC) ----------
def ipu_bilinear(C, Hc, Wc, H, W):
    """Run the precomputed 4-corner accumulate on the IPU for W<=128 (one tile).
    out = sum_corner (resampled coarse corner) elementwise* (weight field)."""
    sys.path.insert(0, HERE)
    from ipu_emu.emulator import load_program, run_until_complete
    from ipu_emu.execute import decode_instruction_word
    from ipu_emu.ipu_state import IpuState, WideVectorArithmetic
    from ipu_emu.ipu_math import DType
    from ipu_as.lark_tree import assemble
    from ipu_as.label import reset_labels
    assert W <= 128 and C.shape[0] == 1
    i0, j0, Wd = precompute(Hc, Wc, H, W)
    II, JJ = np.meshgrid(i0, j0, indexing="ij")
    # The 4 fixedly-resampled coarse "corner" fields (precomputed index pattern):
    corners = {k: C[0, np.clip(II + k[0], 0, Hc - 1), np.clip(JJ + k[1], 0, Wc - 1)]
               for k in Wd}                          # each (H,W)
    # IPU kernel: for each row, R_ACC = sum_corner (corner_row .* weight_row).
    # lr0=const 0 (cyclic idx), lr1=row byte offset (+512/row), lr2=row counter.
    # Two XMEM loads can't share a compound (1 XMEM slot), and MULT.EE reads R0
    # from the wide-mode snapshot (1-cycle load-use delay), so each corner spans
    # two compounds: LDR_MULT_REG (value->R0) then LDR_CYCLIC(weight->R_CYCLIC)+MULT.EE+ACC.
    asm = """\
SET lr0 cr0 ;;
SET lr1 cr0 ;;
SET lr2 cr0 ;;
row_loop:
LDR_MULT_REG r0 lr1 cr2 ;;
LDR_CYCLIC_MULT_REG lr1 cr3 lr0 ; MULT.EE r0 lr0 0 lr0 ; ACC.FIRST ;;
LDR_MULT_REG r0 lr1 cr4 ;;
LDR_CYCLIC_MULT_REG lr1 cr5 lr0 ; MULT.EE r0 lr0 0 lr0 ; ACC ;;
LDR_MULT_REG r0 lr1 cr6 ;;
LDR_CYCLIC_MULT_REG lr1 cr7 lr0 ; MULT.EE r0 lr0 0 lr0 ; ACC ;;
LDR_MULT_REG r0 lr1 cr8 ;;
LDR_CYCLIC_MULT_REG lr1 cr9 lr0 ; MULT.EE r0 lr0 0 lr0 ; ACC ;;
STR_ACC_REG lr1 cr10 ;;
ADD lr1 lr1 cr11 ;;
ADD lr2 lr2 cr1 ;;
BLT lr2 cr12 row_loop ;;
BKPT ;;
"""
    reset_labels(); prog = [decode_instruction_word(w) for w in assemble(asm)]
    s = IpuState(wide_vector_debug=True, wide_vector_arithmetic=WideVectorArithmetic.FP32)
    for cr, v in [(15, DType.INT8), (0, 0), (1, 1)]: s.regfile.set_cr(cr, v)
    def wr(base, fld):
        for y in range(H):
            row = [0.0] * 128
            for x in range(W): row[x] = float(fld[y, x])
            s.xmem.write_address(base + y * 512, struct.pack("<128f", *row))
    # corner value fields + weight fields (each H rows), then output -- 9 H-row
    # regions stacked at an H-scaled stride so larger H stays in bounds.
    RS = ((H * 512 + 0xFFFF) // 0x10000) * 0x10000
    addrs = {(0, 0): (0 * RS, 1 * RS), (0, 1): (2 * RS, 3 * RS),
             (1, 0): (4 * RS, 5 * RS), (1, 1): (6 * RS, 7 * RS)}
    for k, (va, wa) in addrs.items():
        wr(va, corners[k]); wr(wa, Wd[k])
    OUT = 8 * RS
    for cr, v in [(2, 0x0), (3, 0x10000), (4, 0x20000), (5, 0x30000), (6, 0x40000),
                  (7, 0x50000), (8, 0x60000), (9, 0x70000), (10, OUT), (11, 512), (12, H)]:
        s.regfile.set_cr(cr, v)
    load_program(s, list(prog)); run_until_complete(s, max_cycles=2_000_000)
    ipu_bilinear.cycles = s.stats.total_cycles
    out = np.zeros((H, W), np.float32)
    for y in range(H):
        d = s.xmem.read_address(OUT + y * 512, 512)
        for x in range(W): out[y, x] = struct.unpack_from("<f", d, x * 4)[0]
    return out

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # full SuperPoint descriptor map size / image size
    Hc, Wc, H, W = 60, 80, 480, 640
    C = rng.standard_normal((4, Hc, Wc)).astype(np.float32)
    ref = bilinear_ref(C, Hc, Wc, H, W)
    print(f"precomputed bilinear resample {Hc}x{Wc} -> {H}x{W}, {C.shape[0]} ch")
    try:
        t = torch_grid_sample(C, H, W)
        d = np.abs(ref - t)
        print(f"  precomputed-conv form vs torch grid_sample: max {d.max():.2e} mean {d.mean():.2e} "
              f"{'MATCH' if d.max() < 1e-4 else 'DIFF'}")
        dm = np.abs(matmul_ref(C, Hc, Wc, H, W) - t)
        print(f"  separable matmul form vs torch grid_sample: max {dm.max():.2e} "
              f"{'MATCH' if dm.max() < 1e-4 else 'DIFF'}")
    except Exception as e:
        print("  torch check skipped:", e)
    # IPU runs (small: 1ch, W<=128)
    Hc2, Wc2, H2, W2 = 16, 16, 64, 100
    C2 = rng.standard_normal((1, Hc2, Wc2)).astype(np.float32)
    r2 = bilinear_ref(C2, Hc2, Wc2, H2, W2)[0]
    o2 = ipu_bilinear(C2, Hc2, Wc2, H2, W2)
    d2 = np.abs(o2 - r2)
    print(f"  IPU 4-corner accumulate ({Hc2}x{Wc2}->{H2}x{W2}, 1ch): max {d2.max():.2e} "
          f"{'MATCH' if d2.max() < 1e-3 else 'FAIL'}  {ipu_bilinear.cycles} cyc (host builds gathered corners)")
    o3 = ipu_bilinear_matmul(C2, Hc2, Wc2, H2, W2)
    d3 = np.abs(o3 - r2)
    print(f"  IPU gather-free matmul  ({Hc2}x{Wc2}->{H2}x{W2}, 1ch): max {d3.max():.2e} "
          f"{'MATCH' if d3.max() < 1e-3 else 'FAIL'}  {ipu_bilinear_matmul.cycles} cyc (host-free at run)")
    print(f"  cycle ratio (matmul / 4-corner): {ipu_bilinear_matmul.cycles / ipu_bilinear.cycles:.1f}x")

    # ---- real descriptor size: coarse 60x80 -> 480x640, 256 channels ----
    # cycles depend only on loop trip counts (Hc,Wc,H,W_tile), not weight values,
    # so run one (channel, 128-col tile) at the real Hc/Wc/H and scale by
    # CH=256 channels x TILES=ceil(640/128)=5 column tiles.
    CH, TILES = 256, 5
    Cr = rng.standard_normal((1, 60, 80)).astype(np.float32)
    ipu_bilinear_matmul(Cr, 60, 80, 480, 128)                 # H=480 fits XMEM
    mm = ipu_bilinear_matmul.cycles
    ipu_bilinear(Cr, 60, 80, 240, 128)                        # H=240 (9 fields fit)
    fc = ipu_bilinear.cycles * 2                              # linear in H -> H=480
    print(f"\n  real descriptor size 60x80 -> 480x640, {CH} channels ({TILES} col-tiles):")
    print(f"    4-corner : {fc:>9,} cyc/(ch,tile)  ->  {fc*CH*TILES:>14,} cyc total  (host pre-gathers)")
    print(f"    matmul   : {mm:>9,} cyc/(ch,tile)  ->  {mm*CH*TILES:>14,} cyc total  (host-free)")
    print(f"    ratio matmul/4-corner: {mm/fc:.1f}x")
