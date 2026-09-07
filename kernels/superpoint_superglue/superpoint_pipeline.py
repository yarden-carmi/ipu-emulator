#!/usr/bin/env python3
"""End-to-end SuperPoint detector pipeline on the IPU emulator (wide FP32).

Bank-swap pipeline, host-free through selection:
    feature map --conv_fp32 (xNc detector channels)--> channel-major logits
                --superpoint_detect--> per-cell keep = relu(max_c logit_c - tau)

The host only swaps IMEM banks between stages (no computation/decisions). This
harness (1) functionally verifies the chained pipeline vs a NumPy reference on a
small cell grid, and (2) measures cycles + MULT/ACC utilization at the 640x480
cell-grid scale (80x60 = 4800 cells).

Run: python kernels/superpoint_superglue/superpoint_pipeline.py
"""
from __future__ import annotations
import struct, random
from ipu_emu.emulator import load_program, run_until_complete
from ipu_emu.execute import decode_instruction_word
from ipu_emu.ipu_state import IpuState, WideVectorArithmetic
from ipu_emu.ipu_math import DType
from ipu_as.lark_tree import assemble
from ipu_as.label import reset_labels
import os

KD = os.path.dirname(__file__)
NEG = -3.4e38
def _asm(name):
    reset_labels()
    return [decode_instruction_word(w) for w in assemble(open(os.path.join(KD, name)).read())]
def f32(vals):
    b = [0.0] * 128
    for i, v in enumerate(vals[:128]): b[i] = v
    return struct.pack("<128f", *b)
def relu(x): return x if x > 0 else 0.0

CONV = _asm("old/conv_fp32.asm")
DETECT = _asm("superpoint_detect.asm")
print(f"conv_fp32 = {len(CONV)} words, superpoint_detect = {len(DETECT)} words (<=128: {len(CONV)<=128 and len(DETECT)<=128})")

def new_state():
    s = IpuState(wide_vector_debug=True, wide_vector_arithmetic=WideVectorArithmetic.FP32)
    s.regfile.set_cr(15, DType.INT8); s.regfile.set_cr(0, 0); s.regfile.set_cr(1, 1)
    return s

def run_conv_channel(s, IN, OUT_ch, Hp, Wvalid, H, Cin, weights, chan_stride):
    """One detector-conv output channel (3x3) over H rows; writes channel-major plane."""
    WADDR = 0x1F0000
    s.xmem.write_address(WADDR, f32(weights + [0.0] * (128 - len(weights))))
    for cr, v in [(2, IN), (3, OUT_ch), (4, chan_stride), (5, 512),
                  (6, H), (7, Cin), (8, 504), (9, WADDR), (10, 128)]:
        s.regfile.set_cr(cr, v)
    s.program_counter = 0
    load_program(s, list(CONV)); run_until_complete(s, max_cycles=20_000_000)
    return s.stats

# ============================== (1) FUNCTIONAL ==============================
def functional_check():
    random.seed(7)
    # small cell grid: Hc rows x Wc cols of cells, but conv works on a padded plane.
    Hc, Wc = 8, 80           # 8 rows, 80 cols (one 128-tile per row after flatten? use Wc<=126)
    Cin, Nc = 4, 8           # 4 input channels -> 8 detector channels
    tau = 0.3
    Hp = Hc + 2
    feat = [[[random.uniform(-1, 1) for _ in range(Wc)] for _ in range(Hc)] for _ in range(Cin)]
    W = [[[random.uniform(-1, 1) for _ in range(9)] for _ in range(Cin)] for _ in range(Nc)]

    s = new_state()
    IN = 0x0; CONV_OUT = 0x8000
    in_chan_stride = Hp * 512
    # padded input planes
    for ci in range(Cin):
        for r in range(Hp):
            row = [0.0] * 128
            for c in range(Wc):
                if 1 <= r <= Hc: row[c + 1] = feat[ci][r - 1][c]
            s.xmem.write_address(IN + ci * in_chan_stride + r * 512, f32(row))
    # conv: Nc output channels, each a plane of Hc rows (channel-major for detect)
    # detect wants channel-major over cells; here "cells" = Hc rows x 128 lanes.
    # Lay conv output channel oc at CONV_OUT + oc*det_chan_stride, row y at +y*512.
    det_chan_stride = Hc * 512
    conv_cycles = 0
    for oc in range(Nc):
        wflat = [W[oc][ci][t] for ci in range(Cin) for t in range(9)]
        st = run_conv_channel(s, IN, CONV_OUT + oc * det_chan_stride, Hp, Wc, Hc, Cin, wflat, in_chan_stride)
        conv_cycles += st.total_cycles
    # detect: Nc channels, num_tiles = Hc (one 128-lane tile per conv output row)
    NT = Hc
    s.xmem.write_address(0x12000, f32([1.0] * 128))
    s.xmem.write_address(0x13000, f32([NEG] * 128))
    s.xmem.write_address(0x14000, f32([-tau] * 128))
    DET_OUT = 0x18000
    for cr, v in [(2, CONV_OUT), (3, DET_OUT), (4, det_chan_stride), (5, 512),
                  (6, NT), (7, Nc), (8, 0x12000), (9, 0x13000), (10, 0x14000), (11, 128)]:
        s.regfile.set_cr(cr, v)
    s.program_counter = 0
    load_program(s, list(DETECT)); run_until_complete(s, max_cycles=5_000_000)
    det_cycles = s.stats.total_cycles

    # reference: conv (relu) then max over channels then relu(-tau)
    def conv_ref(oc, y, x):
        acc = 0.0
        for ci in range(Cin):
            for kr in range(3):
                for kc in range(3):
                    yy, xx = y + kr - 1, x + kc - 1
                    v = feat[ci][yy][xx] if 0 <= yy < Hc and 0 <= xx < Wc else 0.0
                    acc += W[oc][ci][kr * 3 + kc] * v
        return relu(acc)
    bad = 0; maxerr = 0.0
    for y in range(Hc):
        d = s.xmem.read_address(DET_OUT + y * 512, 512)
        for x in range(Wc):
            ref = relu(max(conv_ref(oc, y, x) for oc in range(Nc)) - tau)
            got = struct.unpack_from("<f", d, x * 4)[0]
            e = abs(got - ref); maxerr = max(maxerr, e)
            if e > 1e-3: bad += 1
    print(f"\n(1) FUNCTIONAL chained conv->detect ({Hc}x{Wc} cells, Cin={Cin}, Nc={Nc}):")
    print(f"    max abs err = {maxerr:.2e}, mismatch = {bad}/{Hc*Wc}  -> {'CORRECT' if bad==0 else 'FAIL'}")
    return bad == 0

# ========================= (2) 640x480 MEASUREMENT =========================
def measure_640x480():
    # cell grid for 640x480 (/8) = 80x60 = 4800 cells. Detector: Cin=14 (max that
    # fits weights in R0), Nc=65 channels. conv per output channel measured once
    # (identical cost) then x65; detect run at full 4800 cells x 65 channels.
    Hc, Wc = 60, 80; Cin = 14; Nc = 65; tau = 0.5
    Hp = Hc + 2; in_chan_stride = Hp * 512
    random.seed(1)
    s = new_state()
    IN = 0x0; CONV_OUT = 0x8000
    for ci in range(Cin):
        for r in range(Hp):
            s.xmem.write_address(IN + ci * in_chan_stride + r * 512,
                                 f32([random.uniform(-1, 1) for _ in range(128)]))
    wflat = [random.uniform(-1, 1) for _ in range(Cin * 9)]
    det_chan_stride = Hc * 512
    st = run_conv_channel(s, IN, CONV_OUT, Hp, Wc, Hc, Cin, wflat, in_chan_stride)
    conv_per_ch = st.total_cycles
    conv_mu = st.mult_active_cycles / st.total_cycles * 100
    conv_total = conv_per_ch * Nc

    # detect at full scale: 4800 cells = ceil(4800/128)=38 tiles, Nc channels
    NT = (Hc * Wc + 127) // 128
    cells = Hc * Wc
    # channel-major logit planes (random); chan c at IN2 + c*cstride, tile t at +t*512
    s2 = new_state()
    IN2 = 0x0; cstride = NT * 512
    for c in range(Nc):
        for t in range(NT):
            s2.xmem.write_address(IN2 + c * cstride + t * 512,
                                  f32([random.uniform(-2, 2) for _ in range(128)]))
    s2.xmem.write_address(0x150000, f32([1.0] * 128))
    s2.xmem.write_address(0x151000, f32([NEG] * 128))
    s2.xmem.write_address(0x152000, f32([-tau] * 128))
    for cr, v in [(2, IN2), (3, 0x158000), (4, cstride), (5, 512), (6, NT), (7, Nc),
                  (8, 0x150000), (9, 0x151000), (10, 0x152000), (11, 128)]:
        s2.regfile.set_cr(cr, v)
    load_program(s2, list(DETECT)); run_until_complete(s2, max_cycles=20_000_000)
    det_cycles = s2.stats.total_cycles
    det_mu = s2.stats.mult_active_cycles / s2.stats.total_cycles * 100
    det_au = s2.stats.acc_active_cycles / s2.stats.total_cycles * 100

    total = conv_total + det_cycles
    print(f"\n(2) 640x480 cell-grid ({Hc}x{Wc}={cells} cells, Cin={Cin}, Nc={Nc}):")
    print(f"    detector conv : {conv_per_ch:,d} cyc/out-channel x {Nc} = {conv_total:,d} cyc  (mult {conv_mu:.0f}%)")
    print(f"    detect+thresh : {det_cycles:,d} cyc  ({NT} tiles x {Nc} ch)  (mult {det_mu:.0f}% / acc {det_au:.0f}%)")
    print(f"    TOTAL host-free conv->detect->select = {total:,d} cycles")
    print(f"    (host only swaps IMEM banks between the {Nc}+1 launches; no computation)")

if __name__ == "__main__":
    ok = functional_check()
    measure_640x480()
    print("\nfunctional:", "PASS" if ok else "FAIL")
