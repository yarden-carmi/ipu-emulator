#!/usr/bin/env python3
"""Measure REAL emulator cycles for the SuperGlue kernel primitives, so the
performance model can use measured cycles instead of the idealized
elements/128 count. Runs each kernel in the wide-FP32 emulator at a couple of
sizes and extracts: matmul cycles per (128-output-tile, contraction-step),
softmax cycles per 128-lane row, and Sinkhorn row/col cycles per row.

Run: python superpoint_eval/measure_kernels.py
"""
import os, sys, struct
HERE = os.path.dirname(__file__)
KDIR = os.path.join(HERE, "..")
sys.path.insert(0, HERE)
from ipu_emu.emulator import load_program, run_until_complete
from ipu_emu.execute import decode_instruction_word
from ipu_emu.ipu_state import IpuState, WideVectorArithmetic
from ipu_emu.ipu_math import DType
from ipu_as.lark_tree import assemble
from ipu_as.label import reset_labels


def run(asm, crs, writes=()):
    reset_labels()
    prog = [decode_instruction_word(w) for w in assemble(asm)]
    s = IpuState(wide_vector_debug=True, wide_vector_arithmetic=WideVectorArithmetic.FP32)
    s.regfile.set_cr(15, DType.INT8)
    for cr, v in crs:
        s.regfile.set_cr(cr, v)
    for addr, vals in writes:
        s.xmem.write_address(addr, struct.pack("<128f", *(list(vals) + [0.0] * (128 - len(vals)))))
    s.program_counter = 0
    load_program(s, list(prog))
    run_until_complete(s, max_cycles=50_000_000)
    return s.stats.total_cycles


# ---- matmul inner loop: per 128-output-tile, K packed MACs (MULT.VE.CYCLIC+ACC)
MM = """\
SET lr0 cr0 ;;
LDR_MULT_REG r0 lr0 cr3 ;;
SET lr1 cr0 ;;
tile_loop:
RESET_ACC ;;
SET lr2 cr0 ;; SET lr3 cr0 ;;
kloop:
LDR_CYCLIC_MULT_REG lr0 cr2 lr0 ; MULT.VE.CYCLIC lr0 0 lr0 lr3 ; ACC ;;
ADD lr3 lr3 cr1 ; ADD lr2 lr2 cr1 ;;
BLT lr2 cr5 kloop ;;
STR_ACC_REG lr0 cr6 ;;
ADD lr1 lr1 cr1 ;;
BLT lr1 cr7 tile_loop ;;
BKPT ;;
"""

def matmul_cyc(K, ntiles):
    return run(MM, [(0, 0), (1, 1), (2, 0x1000), (3, 0x2000), (5, K), (6, 0x40000), (7, ntiles)],
               writes=[(0x1000, [1.0] * 128), (0x2000, [0.5] * 128)])


def measure_matmul():
    # cycles ~ ntiles*(a*K + b). Solve for per-MAC-step a and per-tile overhead b.
    c1 = matmul_cyc(64, 50); c2 = matmul_cyc(128, 50)
    a = (c2 - c1) / (50 * (128 - 64))          # cycles per MAC step (per 128-tile)
    b = c1 / 50 - a * 64                        # per-tile fixed overhead
    return a, b, c1, c2


SOFTMAX = open(os.path.join(KDIR, "softmax.asm")).read()
def measure_softmax(N=128):
    return run(SOFTMAX, [(0, 0), (1, 1), (2, 0x1000), (3, 0x2000), (4, 0x3000),
                         (5, 0x4000), (6, N), (7, 0xBF800000)],
               writes=[(0x1000, [0.1 * i for i in range(N)]), (0x3000, [1.0] * 128)])


SROW = open(os.path.join(KDIR, "sinkhorn_iter.asm")).read()
SCOL = open(os.path.join(KDIR, "sinkhorn_col.asm")).read()
def measure_sink_row(R, cols=128):
    # CR2=mat,CR3=?,CR6=num_rows,CR7=row_stride; read header for exact ids
    crs = [(0, 0), (1, 1), (2, 0x10000), (6, R), (7, 512), (8, cols),
           (3, 0xBF800000), (4, 0x4000), (5, 0x5000)]
    w = [(0x10000 + r * 512, [0.1 * r] * cols) for r in range(R)]
    return run(SROW, crs, writes=w)
def measure_sink_col(R, cols=128):
    crs = [(0, 0), (1, 1), (2, 0x10000), (4, 0x4000), (5, 0x5000), (6, R),
           (7, 512), (8, cols), (9, 0x6000), (10, 0x7000), (11, 0x8000)]
    w = [(0x10000 + r * 512, [0.1 * r] * cols) for r in range(R)]
    w += [(0x4000, [1.0] * 128), (0x5000, [-1.0] * 128), (0x8000, [-3.4e38] * 128)]
    return run(SCOL, crs, writes=w)


# ---- pure elementwise: per 128-tile load+MULT.EE+store (scale/add/relu/exp2 class)
EW = """\
SET lr0 cr0 ;;
SET lr1 cr0 ;;
ew_loop:
LDR_MULT_REG r0 lr0 cr2 ;;
LDR_CYCLIC_MULT_REG lr0 cr3 lr0 ; MULT.EE r0 lr0 0 lr0 ; ACC.FIRST ;;
STR_ACC_REG lr0 cr6 ;;
ADD lr1 lr1 cr1 ;;
BLT lr1 cr7 ew_loop ;;
BKPT ;;
"""
def ew_cyc(ntiles):
    return run(EW, [(0, 0), (1, 1), (2, 0x1000), (3, 0x2000), (6, 0x40000), (7, ntiles)],
               writes=[(0x1000, [1.0] * 128), (0x2000, [0.5] * 128)])


def rates():
    """Measured cycle rates used by the spreadsheet builder."""
    a, b, _, _ = measure_matmul()
    e2 = ew_cyc(50); e1 = ew_cyc(10)
    ew = (e2 - e1) / 40.0                                   # cyc per 128-tile elementwise
    sm = measure_softmax(128)                               # cyc per 128-lane softmax row
    sr = measure_sink_row(64) / 64.0                        # cyc per row, row half-step (<=128 col)
    sc = measure_sink_col(64) / 64.0                        # cyc per row, col half-step (<=128 col)
    return {"mac_step": a, "tile_ovh": b, "ew_tile": ew, "softmax128": sm,
            "sink_row": sr, "sink_col": sc}


if __name__ == "__main__":
    a, b, c1, c2 = measure_matmul()
    print("MATMUL inner loop:")
    print(f"  K=64,50tiles={c1} cyc ; K=128,50tiles={c2} cyc")
    print(f"  -> {a:.4f} cyc / MAC-step(128-tile), {b:.2f} cyc / tile overhead")
    print(f"  -> a 128-output x K-contraction GEMM tile costs ~ {a:.3f}*K + {b:.1f} cyc")
    try:
        print("SOFTMAX 128-lane row:", measure_softmax(128), "cyc")
    except Exception as e:
        print("softmax FAIL:", e)
    try:
        for R in (32, 64):
            print(f"SINKHORN row half-step, {R} rows:", measure_sink_row(R), "cyc",
                  f"(~{measure_sink_row(R)/R:.1f}/row)")
    except Exception as e:
        print("sink_row FAIL:", e)
    try:
        for R in (32, 64):
            print(f"SINKHORN col half-step, {R} rows:", measure_sink_col(R), "cyc")
    except Exception as e:
        print("sink_col FAIL:", e)
