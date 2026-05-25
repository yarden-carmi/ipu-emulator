"""Run conv_fp32.asm in the IPU emulator with real SuperPoint weights.

Bias is folded with NO kernel change: an extra all-ones input channel whose
center-tap weight = bias[oc] (so the conv accumulates +bias). Cin' = Cin+1, so
this path needs (Cin+1)*9 <= 128, i.e. Cin <= 13 (conv1a Cin=1 fits). Larger
layers use the channel-tiled kernel (Phase 2).
"""
import os, struct, numpy as np
from ipu_emu.emulator import load_program, run_until_complete
from ipu_emu.execute import decode_instruction_word
from ipu_emu.ipu_state import IpuState, WideVectorArithmetic
from ipu_emu.ipu_math import DType
from ipu_as.lark_tree import assemble
from ipu_as.label import reset_labels
from ipu_emu.stats import RunStats

HERE = os.path.dirname(__file__)
KD = os.path.join(HERE, "..")

def _load(name):
    reset_labels()
    return [decode_instruction_word(w) for w in assemble(open(os.path.join(KD, name)).read())]

def _f32(s, addr, vals):
    b = [0.0] * 128
    for i, v in enumerate(vals[:128]): b[i] = float(v)
    s.xmem.write_address(addr, struct.pack("<128f", *b))

def conv_fp32_layer(x, w, b, wcrop=None):
    """x:(Cin,H,W), w:(Cout,Cin,3,3), b:(Cout,) -> (Cout,H,wcrop) via conv_fp32.asm.
    Returns (output, total_cycles, mult_util, acc_util)."""
    PROG = _load("conv_fp32.asm")
    Cin, H, W = x.shape
    Cout = w.shape[0]
    wcrop = wcrop or min(W, 126)
    CinP = Cin + 1                      # +1 ones channel for bias
    assert CinP * 9 <= 128, f"Cin too large for R0 ({CinP*9}>128); use tiled kernel"
    Hp = H + 2
    CS = Hp * 512                        # bytes per padded input plane
    IN = 0x0; OUTB = 0x100000; WB = 0x1F0000
    out = np.zeros((Cout, H, wcrop), np.float32)
    tot = mult_a = acc_a = 0
    s = IpuState(wide_vector_debug=True, wide_vector_arithmetic=WideVectorArithmetic.FP32)
    s.regfile.set_cr(15, DType.INT8); s.regfile.set_cr(0, 0); s.regfile.set_cr(1, 1)
    # padded input planes (orig (r,c) -> padded (r+1,c+1)); channel Cin = all-ones
    for ci in range(Cin):
        for r in range(Hp):
            row = [0.0] * 128
            if 1 <= r <= H:
                for c in range(wcrop): row[c + 1] = float(x[ci, r - 1, c])
            _f32(s, IN + ci * CS + r * 512, row)
    for r in range(Hp):                  # ones channel (valid region = 1.0)
        row = [0.0] * 128
        if 1 <= r <= H:
            for c in range(wcrop): row[c + 1] = 1.0
        _f32(s, IN + Cin * CS + r * 512, row)
    for cr, v in [(2, IN), (4, CS), (5, 512), (6, H), (7, CinP), (8, 504), (10, 128)]:
        s.regfile.set_cr(cr, v)
    for oc in range(Cout):
        wflat = [float(w[oc, ci, kr, kc]) for ci in range(Cin) for kr in range(3) for kc in range(3)]
        wflat += [0, 0, 0, 0, float(b[oc]), 0, 0, 0, 0]   # bias channel: center tap = bias
        _f32(s, WB, wflat)
        s.regfile.set_cr(3, OUTB); s.regfile.set_cr(9, WB)
        s.program_counter = 0; s.stats = RunStats()   # clean per-launch stats
        load_program(s, list(PROG)); run_until_complete(s, max_cycles=20_000_000)
        st = s.stats
        tot += st.total_cycles; mult_a += st.mult_active_cycles; acc_a += st.acc_active_cycles
        for y in range(H):
            d = s.xmem.read_address(OUTB + y * 512, 512)
            for c in range(wcrop):
                out[oc, y, c] = struct.unpack_from("<f", d, c * 4)[0]
    return out, tot, 100 * mult_a / tot, 100 * acc_a / tot


def conv_fp32_tiled_layer(x, w, b, wcrop=None, G=13, relu=True):
    """Channel-tiled conv (any Cin) via conv_fp32_tiled[_norelu].asm. Bias folded
    as an all-ones channel (center-tap weight = bias); channels padded to
    G*ngroups. Accepts 3x3 or 1x1 weights (1x1 expanded to center-only 3x3)."""
    PROG = _load("conv_fp32_tiled.asm" if relu else "conv_fp32_tiled_norelu.asm")
    if w.shape[2] == 1:                              # 1x1 conv -> center-only 3x3
        w33 = np.zeros((w.shape[0], w.shape[1], 3, 3), np.float32)
        w33[:, :, 1, 1] = w[:, :, 0, 0]; w = w33
    Cin, H, W = x.shape
    Cout = w.shape[0]
    wcrop = wcrop or min(W, 126)
    CinP = Cin + 1                                   # +1 ones channel (bias)
    ngroups = (CinP + G - 1) // G
    Ctot = ngroups * G                               # padded channel count
    assert G * 9 <= 128
    Hp = H + 2; CS = Hp * 512
    IN = 0x0; OUTB = 0x100000; WB = 0x1C0000
    gws = G * 9 * 4
    out = np.zeros((Cout, H, wcrop), np.float32)
    tot = mult_a = acc_a = 0
    s = IpuState(wide_vector_debug=True, wide_vector_arithmetic=WideVectorArithmetic.FP32)
    s.regfile.set_cr(15, DType.INT8); s.regfile.set_cr(0, 0); s.regfile.set_cr(1, 1)
    for ci in range(Ctot):                           # input planes
        for r in range(Hp):
            row = [0.0] * 128
            if 1 <= r <= H:
                if ci < Cin:
                    for c in range(wcrop): row[c + 1] = float(x[ci, r - 1, c])
                elif ci == Cin:
                    for c in range(wcrop): row[c + 1] = 1.0     # ones (bias) channel
            _f32(s, IN + ci * CS + r * 512, row)
    for cr, v in [(2, IN), (4, CS), (5, 512), (6, H), (8, 504), (10, 128),
                  (11, G), (12, ngroups), (13, gws)]:
        s.regfile.set_cr(cr, v)
    for oc in range(Cout):
        for g in range(ngroups):                     # weights per group [cig][tap]
            wf = []
            for cig in range(G):
                ci = g * G + cig
                if ci < Cin:
                    wf += [float(w[oc, ci, kr, kc]) for kr in range(3) for kc in range(3)]
                elif ci == Cin:
                    wf += [0, 0, 0, 0, float(b[oc]), 0, 0, 0, 0]   # bias at center
                else:
                    wf += [0.0] * 9                                 # padding channel
            _f32(s, WB + g * gws, wf)
        s.regfile.set_cr(3, OUTB); s.regfile.set_cr(9, WB)
        s.program_counter = 0; s.stats = RunStats()
        load_program(s, list(PROG)); run_until_complete(s, max_cycles=40_000_000)
        st = s.stats; tot += st.total_cycles; mult_a += st.mult_active_cycles; acc_a += st.acc_active_cycles
        for y in range(H):
            d = s.xmem.read_address(OUTB + y * 512, 512)
            for c in range(wcrop):
                out[oc, y, c] = struct.unpack_from("<f", d, c * 4)[0]
    return out, tot, 100 * mult_a / tot, 100 * acc_a / tot
