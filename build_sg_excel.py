#!/usr/bin/env python3
"""Build the SuperPoint + SuperGlue performance-model Excel from scratch, with
SuperPoint and SuperGlue as SEPARATE sheets, host operations marked clearly per
row (Host column: D=device, M=host data-movement, H=host compute), and built-in
Python<->XLS self-validation.

Sheets produced (kept alongside the source MobileViT):
  SuperPoint                        bare-op analytical, per image
  SuperGlue (original)              log-domain Sinkhorn, dual-softmax readout
  SG max-Sinkhorn                   Sinkhorn -> sum + scale  (our sinkhorn_iter/col)
  SG base2-softmax                  attn softmax via exp2    (same cost)
  SG transpose-free col             col norm via cross-row max, no transpose
  SG fixed-thresh match             readout = argmax + threshold (1 pass)
  SuperGlue (measured)              SG max-Sinkhorn measured from emulator
  Pipeline Summary                  2*SP + SG variant -> total + FPS

Run: python build_sg_excel.py [src.xlsx] [out.xlsx]
"""
import os, sys, re, math
from copy import copy
import openpyxl
from openpyxl.utils import get_column_letter

# ---------------- parameters -------------------------------------------------
N, D, HEADS, HD, NP1, T = 512, 256, 4, 64, 513, 100
GNN_BLOCKS = 18 * 2
IMG_H, IMG_W = 480, 640
COLS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
HOST_COL = 29                                                  # column AC = "Host"

SRC = sys.argv[1] if len(sys.argv) > 1 else \
    "/root/.claude/uploads/eecfeba7-80e7-4f8a-8032-ec4e983b04a4/cb06a587-IPU_2.xlsx"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/home/user/ipu-emulator/IPU_2_SuperGlue.xlsx"


# ---------------- bare-op recipe -- 7-tuple (group, op, c, d, K, kind, host) -
# host: 'D'=device kernel, 'M'=host movement, 'H'=host compute (gather/sort/idx)
def macd(group, c, d, K):
    return [(group, "multiply", c, d, K, "mac", "D"),
            (group, "accumulate", c, d, K, "mac", "D")]


def rows_for_superpoint(K_kpts=N):
    """Per-image SuperPoint bare-op rows (encoder + detector + descriptor)."""
    rows = [("Input image load", "kpt+desc load", IMG_H, IMG_W, 0, "rsh", "M")]
    enc = [(IMG_H, IMG_W, [(1, 64, 'conv1a'), (64, 64, 'conv1b')]),
           (IMG_H // 2, IMG_W // 2, [(64, 64, 'conv2a'), (64, 64, 'conv2b')]),
           (IMG_H // 4, IMG_W // 4, [(64, 128, 'conv3a'), (128, 128, 'conv3b')]),
           (IMG_H // 8, IMG_W // 8, [(128, 128, 'conv4a'), (128, 128, 'conv4b')])]
    for i, (H, W, convs) in enumerate(enc):
        for cin, cout, name in convs:
            rows += macd(f'{name} 3x3+ReLU', cout, H * W, cin * 9)
        if i < len(enc) - 1:
            outH, outW = H // 2, W // 2
            rows.append((f'maxpool 2x2/s2 -> {outH}x{outW}', 'max(4 taps)',
                         convs[-1][1], outH * outW, 4, 'mac', 'D'))
    H, W = IMG_H // 8, IMG_W // 8
    rows += macd('convPa 3x3+ReLU', 256, H * W, 128 * 9)
    rows += macd('convPb 1x1 (->65)', 65, H * W, 256)
    for op in ['max', 'subtract', 'exp2', 'sum', 'multiply(1/Z)']:
        rows.append(('detector softmax(65)', op, 65, H * W, 0, 'ew', 'D'))
    rows.append(('depth-to-space', 'reshape', 64, H * W, 0, 'rsh', 'M'))
    Hf, Wf = H * 8, W * 8
    rows.append(('simple_nms 9x9 iter 3', 'maxpool 243 taps', Hf, Wf, 243, 'mac', 'D'))
    rows.append(('score threshold', 'relu(s-tau)', Hf, Wf, 0, 'ew', 'D'))
    rows.append(('topk cap (calibrated)', 'soft count + host bisect', Hf, Wf, 30, 'mac', 'H'))
    rows += macd('convDa 3x3+ReLU', 256, H * W, 128 * 9)
    rows += macd('convDb 1x1 (->256)', 256, H * W, 256)
    for op in ['mul (squares)', 'sum-reduce', 'multiply (scale by rsqrt)']:
        rows.append(('descriptor L2 (dense)', op, 256, H * W, 0, 'ew', 'D'))
    rows += macd('grid_sample (4-corner)', 256, K_kpts, 4)
    for op in ['mul (squares)', 'sum-reduce', 'multiply (scale by rsqrt)']:
        rows.append(('per-keypoint L2', op, 256, K_kpts, 0, 'ew', 'D'))
    return rows


def rows_for_sg(variant):
    """SuperGlue bare-op rows (encoder + 1 GNN block expanded + matching)."""
    rows = [("Input (kpts+desc)", "load", D, N, 0, "rsh", "M")]
    for cin, cout in [(3, 32), (32, 64), (64, 128), (128, 256)]:
        rows += macd(f"KptEnc {cin}->{cout}", cout, N, cin)
    rows.append(("Residual desc+=enc", "add", D, N, 0, "ew", "M"))
    for nm in "QKV":
        rows += macd(f"{nm} proj D->D", D, N, D)
    for nm in "QKV":
        rows.append((f"head reshape {nm}", "reshape", HD, N, 0, "rsh", "M"))
    rows += macd("QKt (1 head)", N, N, HD)
    rows.append(("scale 1/sqrt(d)", "multiply", N, N, 0, "ew", "D"))
    for op in ["max", "subtract", "exp2", "sum", "multiply(1/Z)"]:
        rows.append(("softmax (1 head)", op, N, N, 0, "ew", "D"))
    rows += macd("attn @ V (1 head)", N, HD, N)
    rows.append(("concat heads", "reshape", D, N, 0, "rsh", "M"))
    rows += macd("out proj D->D", D, N, D)
    rows.append(("Residual x+=msg", "add", D, N, 0, "ew", "M"))
    rows += macd("merge MLP 2D->2D", 2 * D, N, 2 * D)
    rows.append(("merge MLP relu", "relu", 2 * D, N, 0, "ew", "D"))
    rows += macd("merge MLP 2D->D", D, N, 2 * D)
    rows.append(("Residual x+=mlp", "add", D, N, 0, "ew", "M"))
    rows += macd("final proj D->D", D, N, D)
    rows += macd("score = A^T B", N, N, D)
    rows.append(("scale 1/sqrt(d)", "multiply", N, N, 0, "ew", "D"))
    rows.append(("dustbin augment", "reshape", NP1, NP1, 0, "rsh", "M"))
    if variant == "max_sinkhorn":
        row_ops, col_ops = ["sum", "multiply(1/r)"], ["sum", "multiply(1/c)"]
    else:
        row_ops = col_ops = ["max", "exp2", "sum", "log", "subtract"]
    for op in row_ops:
        rows.append(("Sinkhorn row norm", op, NP1, NP1, 0, "ew", "D"))
    if variant == "transpose_free":
        rows.append(("Sinkhorn col transpose (eliminated)", "reshape", NP1, NP1, 0, "rsh", "M"))
    else:
        rows.append(("Sinkhorn col transpose", "move", NP1, NP1, 0, "ew", "M"))
    for op in col_ops:
        rows.append(("Sinkhorn col norm", op, NP1, NP1, 0, "ew", "D"))
    if variant == "fixed_thresh":
        for op in ["max(argmax)", "compare(threshold)"]:
            rows.append(("match readout", op, NP1, NP1, 0, "ew", "H"))
    else:
        for op in ["max", "exp2", "sum", "subtract", "multiply"]:
            rows.append(("match readout (dual-softmax)", op, NP1, NP1, 0, "ew", "D"))
        rows.append(("argmax index + mutual-NN", "host gather+compare", NP1, 1, 0, "rsh", "H"))
    return rows


# ---------------- spreadsheet plumbing --------------------------------------
wb = openpyxl.load_workbook(SRC)
mob = wb["MobileViT"]
HEADER = [mob.cell(1, c).value for c in range(1, 27)]


def cstyle(dst, src):
    dst.font = copy(src.font); dst.fill = copy(src.fill); dst.border = copy(src.border)
    dst.alignment = copy(src.alignment); dst.number_format = src.number_format


def make_sheet(name):
    if name in wb.sheetnames:
        del wb[name]
    ws = wb.create_sheet(name)
    for c in range(1, 27):
        ws.cell(1, c, HEADER[c - 1]); cstyle(ws.cell(1, c), mob.cell(1, c))
    ws.cell(1, HOST_COL, "Host"); cstyle(ws.cell(1, HOST_COL), mob.cell(1, 1))
    for i in range(1, 27):
        L = get_column_letter(i); ws.column_dimensions[L].width = mob.column_dimensions[L].width
    ws.column_dimensions[get_column_letter(HOST_COL)].width = 6
    ws.freeze_panes = "B2"
    ws["Y2"], ws["Z2"] = 1, 8
    return ws


def emit_row(ws, r, group, op, c, d, g, kind, host):
    S = {"mac": f"=C{r}*D{r}*G{r}/128", "ew": f"=C{r}*D{r}/128", "rsh": "0"}[kind]
    vals = {"A": group, "B": op, "C": c, "D": d, "E": 0, "F": 0, "G": g, "H": 0,
            "I": 0, "J": 0, "K": 0, "L": c, "M": d, "N": 0,
            "O": f"=C{r}*D{r}/1024", "P": 0, "Q": 0, "R": f"=L{r}*M{r}/1024", "S": S}
    if kind == "rsh":
        vals["S"] = "0"; vals["T"] = 0
        for cc in "UVWX": vals[cc] = 0
    else:
        vals["T"] = f"=(S{r}/$Y$2)/1000"
        for cc, src in [("U", "O"), ("V", "P"), ("W", "Q"), ("X", "R")]:
            vals[cc] = f"=(({src}{r}*8*1024)/$T{r})/1000"
    for col, v in vals.items():
        cstyle(ws.cell(r, COLS.index(col) + 1, v), mob.cell(5, COLS.index(col) + 1))
    cell = ws.cell(r, HOST_COL, host)
    cstyle(cell, mob.cell(5, 1))


def write_total(ws, r, label, zformula):
    cstyle(ws.cell(r, 25, label), mob.cell(68, 25))
    cstyle(ws.cell(r, 26, zformula), mob.cell(68, 26))


def build_sp_sheet():
    ws = make_sheet("SuperPoint")
    rows = rows_for_superpoint(K_kpts=N)
    for i, row in enumerate(rows):
        emit_row(ws, 2 + i, *row)
    last = 1 + len(rows)
    # totals
    write_total(ws, last, "SuperPoint per image [us] ->", f"=SUM(T2:T{last})")
    gt = last + 2
    cstyle(ws.cell(gt, 1, "GRAND TOTAL [us] (x2 images)"), mob.cell(5, 1))
    write_total(ws, gt, "2 images ->", f"=2*Z{last}")
    fr = gt + 1
    cstyle(ws.cell(fr, 1, "FPS (SP alone)"), mob.cell(5, 1))
    write_total(ws, fr, "frames / sec ->", f"=1000000/Z{gt}")
    return ws, gt


def build_sg_sheet(variant, name):
    ws = make_sheet(name)
    rows = rows_for_sg(variant)
    idx_groups = {}
    for i, (g, *_) in enumerate(rows):
        idx_groups.setdefault(g, []).append(2 + i)
    for i, row in enumerate(rows):
        emit_row(ws, 2 + i, *row)
    last = 1 + len(rows)
    enc_rows = [r for r in range(2, last + 1) if ws.cell(r, 1).value
                and ("KptEnc" in str(ws.cell(r, 1).value) or "Residual desc" in str(ws.cell(r, 1).value))]
    head_rows = [r for r in range(2, last + 1)
                 if ws.cell(r, 1).value in ("QKt (1 head)", "scale 1/sqrt(d)",
                                             "softmax (1 head)", "attn @ V (1 head)")
                 and r < idx_groups.get("concat heads", [last + 1])[0]]
    block_start = idx_groups["Q proj D->D"][0]
    block_end = idx_groups["Residual x+=mlp"][-1]
    sink_rows = sorted(set(idx_groups.get("Sinkhorn row norm", [])
                            + idx_groups.get("Sinkhorn col norm", [])
                            + idx_groups.get("Sinkhorn col transpose", [])))
    read_rows = sorted(set(idx_groups.get("match readout", [])
                            + idx_groups.get("match readout (dual-softmax)", [])))
    final_rows = idx_groups["final proj D->D"]
    score_rows = idx_groups["score = A^T B"]
    scale_match = [r for r in idx_groups["scale 1/sqrt(d)"] if r not in head_rows][0]

    write_total(ws, enc_rows[-1], "Encoder total [us] ->",
                f"=2*(SUM(T{enc_rows[0]}:T{enc_rows[-1]}))")
    write_total(ws, idx_groups["concat heads"][0], "total 4 heads [us]",
                f"=4*(SUM(T{head_rows[0]}:T{head_rows[-1]}))")
    gnn_z = (f"={GNN_BLOCKS}*(SUM(T{block_start}:T{block_end})"
             f"+3*(SUM(T{head_rows[0]}:T{head_rows[-1]})))")
    write_total(ws, block_end, "GNN total [us] ->", gnn_z)
    write_total(ws, sink_rows[-1], f"Sinkhorn total (x{T}) [us] ->",
                f"={T}*(SUM(T{sink_rows[0]}:T{sink_rows[-1]}))")
    gt = last + 2
    cstyle(ws.cell(gt, 1, "GRAND TOTAL [us]"), mob.cell(5, 1))
    grand = (f"=Z{enc_rows[-1]}+Z{block_end}+2*(SUM(T{final_rows[0]}:T{final_rows[-1]}))"
             f"+SUM(T{score_rows[0]}:T{score_rows[-1]})+T{scale_match}"
             f"+Z{sink_rows[-1]}+SUM(T{read_rows[0]}:T{read_rows[-1]})")
    write_total(ws, gt, "variant: " + variant, grand)
    fr = gt + 1
    cstyle(ws.cell(fr, 1, "FPS (SG alone)"), mob.cell(5, 1))
    write_total(ws, fr, "frames / sec ->", f"=1000000/Z{gt}")
    return ws, gt


def build_measured_sg_sheet():
    """SG max-Sinkhorn with cycles MEASURED via the emulator (running each kernel)."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                    "kernels/superpoint_superglue/superpoint_eval"))
    from measure_kernels import rates
    R = rates()
    def tiles(e): return math.ceil(e / 128)
    def opcyc(kind, c, d, K):
        if kind == "mac":  return tiles(c * d) * (R["mac_step"] * K + R["tile_ovh"])
        if kind == "ew":   return tiles(c * d) * R["ew_tile"]
        if kind == "taps": return tiles(c * d) * K * R["ew_tile"]
        if kind == "sm":   return d * tiles(c) * R["softmax128"]
        if kind == "sr":   return d * tiles(c) * R["sink_row"]
        if kind == "sc":   return d * tiles(c) * R["sink_col"]
        return 0.0
    # SG-only (max-Sinkhorn variant). mult bakes the x36 / x2 / xT in.
    g = GNN_BLOCKS
    ROWS = [
        ("Input (kpts+desc)", "load", D, N, 0, "rsh", "M", 1),
    ]
    for cin, cout in [(3, 32), (32, 64), (64, 128), (128, 256)]:
        ROWS.append((f"KptEnc {cin}->{cout}", "MAC matmul", cout, N, cin, "mac", "D", 2))
    ROWS.append(("Residual desc+=enc", "add", D, N, 0, "ew", "M", 2))
    ROWS += [
        ("Q proj D->D", "MAC", D, N, D, "mac", "D", g),
        ("K proj D->D", "MAC", D, N, D, "mac", "D", g),
        ("V proj D->D", "MAC", D, N, D, "mac", "D", g),
        ("QKt", "MAC", N, N, HD, "mac", "D", g * HEADS),
        ("scale", "multiply", N, N, 0, "ew", "D", g * HEADS),
        ("softmax", "max,exp2,sum,mul", N, N, 0, "sm", "D", g * HEADS),
        ("attn @ V", "MAC", N, HD, N, "mac", "D", g * HEADS),
        ("out proj", "MAC", D, N, D, "mac", "D", g),
        ("Residual x+=msg", "add", D, N, 0, "ew", "M", g),
        ("merge MLP 2D->2D", "MAC", 2 * D, N, 2 * D, "mac", "D", g),
        ("merge MLP relu", "relu", 2 * D, N, 0, "ew", "D", g),
        ("merge MLP 2D->D", "MAC", D, N, 2 * D, "mac", "D", g),
        ("Residual x+=mlp", "add", D, N, 0, "ew", "M", g),
        ("final proj D->D", "MAC", D, N, D, "mac", "D", 2),
        ("score = A^T B", "MAC", N, N, D, "mac", "D", 1),
        ("scale", "multiply", N, N, 0, "ew", "D", 1),
        ("Sinkhorn row norm (max)", "sum+scale", NP1, NP1, 0, "sr", "D", T),
        ("Sinkhorn col norm (max)", "ACC.MAX+scale", NP1, NP1, 0, "sc", "D", T),
        ("match readout", "argmax+thresh", NP1, NP1, 0, "ew", "H", 1),
    ]
    ws = make_sheet("SuperGlue (measured)")
    total = 0.0
    for i, (gr, op, c, d, K, kind, host, mult) in enumerate(ROWS):
        r = 2 + i
        cyc = round(opcyc(kind, c, d, K) * mult)
        total += cyc
        vals = {"A": gr, "B": op, "C": c, "D": d, "E": K, "F": 0, "G": K, "H": mult,
                "I": 0, "J": 0, "K": 0, "L": c, "M": d, "N": 0,
                "O": f"=C{r}*D{r}/1024", "P": 0, "Q": 0, "R": f"=L{r}*M{r}/1024",
                "S": cyc, "T": f"=(S{r}/$Y$2)/1000"}
        if kind == "rsh" or cyc == 0:
            vals["S"] = 0; vals["T"] = 0
            for cc in "UVWX": vals[cc] = 0
        else:
            for cc, src in [("U", "O"), ("V", "P"), ("W", "Q"), ("X", "R")]:
                vals[cc] = f"=(({src}{r}*8*1024)/$T{r})/1000"
        for col, v in vals.items():
            cstyle(ws.cell(r, COLS.index(col) + 1, v), mob.cell(5, COLS.index(col) + 1))
        cstyle(ws.cell(r, HOST_COL, host), mob.cell(5, 1))
        if mult > 1:
            ws.cell(r, 27, f"x{mult}")
    gt = 2 + len(ROWS) + 1
    cstyle(ws.cell(gt, 1, "GRAND TOTAL [us] (measured)"), mob.cell(5, 1))
    write_total(ws, gt, "measured emulator cycles ->", f"=SUM(T2:T{1 + len(ROWS)})")
    fr = gt + 1
    cstyle(ws.cell(fr, 1, "FPS (SG measured)"), mob.cell(5, 1))
    write_total(ws, fr, "frames / sec ->", f"=1000000/Z{gt}")
    return ws, gt, total, R


# ---------------- Python validation oracle -----------------------------------
def py_sp_per_image():
    us = lambda cyc: cyc / 1000.0
    mac_us = lambda c, d, K: 2 * us(c * d * K / 128.0)
    ew_us = lambda c, d: us(c * d / 128.0)
    taps_us = lambda c, d, K: K * ew_us(c, d)
    sp = 0
    enc = [(IMG_H, IMG_W, [(1, 64), (64, 64)]),
           (IMG_H // 2, IMG_W // 2, [(64, 64), (64, 64)]),
           (IMG_H // 4, IMG_W // 4, [(64, 128), (128, 128)]),
           (IMG_H // 8, IMG_W // 8, [(128, 128), (128, 128)])]
    for i, (H, W, cs) in enumerate(enc):
        for cin, cout in cs:
            sp += mac_us(cout, H * W, cin * 9)
        if i < len(enc) - 1:
            sp += taps_us(cs[-1][1], (H // 2) * (W // 2), 4)
    H, W = IMG_H // 8, IMG_W // 8
    sp += mac_us(256, H * W, 128 * 9)
    sp += mac_us(65, H * W, 256)
    sp += 5 * ew_us(65, H * W)
    Hf, Wf = H * 8, W * 8
    sp += taps_us(Hf, Wf, 243)
    sp += ew_us(Hf, Wf)
    sp += taps_us(Hf, Wf, 30)
    sp += mac_us(256, H * W, 128 * 9)
    sp += mac_us(256, H * W, 256)
    sp += 3 * ew_us(256, H * W)
    sp += mac_us(256, N, 4)
    sp += 3 * ew_us(256, N)
    return sp


def py_sg(variant):
    us = lambda cyc: cyc / 1000.0
    mac_us = lambda c, d, K: 2 * us(c * d * K / 128.0)
    ew_us = lambda c, d: us(c * d / 128.0)
    enc = 0
    for cin, cout in [(3, 32), (32, 64), (64, 128), (128, 256)]:
        enc += mac_us(cout, N, cin)
    enc += ew_us(D, N)
    enc *= 2
    head = mac_us(N, N, HD) + ew_us(N, N) + 5 * ew_us(N, N) + mac_us(N, HD, N)
    block = 3 * mac_us(D, N, D) + 4 * head + mac_us(D, N, D) + ew_us(D, N) \
        + mac_us(2 * D, N, 2 * D) + ew_us(2 * D, N) + mac_us(D, N, 2 * D) + ew_us(D, N)
    gnn = GNN_BLOCKS * block
    fin = 2 * mac_us(D, N, D); score = mac_us(N, N, D); scl = ew_us(N, N)
    if variant == "max_sinkhorn":
        sink = T * (2 * ew_us(NP1, NP1) + ew_us(NP1, NP1) + 2 * ew_us(NP1, NP1))
    elif variant == "transpose_free":
        sink = T * (5 * ew_us(NP1, NP1) + 0 + 5 * ew_us(NP1, NP1))
    else:
        sink = T * (5 * ew_us(NP1, NP1) + ew_us(NP1, NP1) + 5 * ew_us(NP1, NP1))
    read = 2 * ew_us(NP1, NP1) if variant == "fixed_thresh" else 5 * ew_us(NP1, NP1)
    return enc + gnn + fin + score + scl + sink + read


# ---------------- formula evaluator ------------------------------------------
CELLRE = re.compile(r'\$?([A-Z]+)\$?([0-9]+)')
SUMRE = re.compile(r'SUM\(\$?([A-Z]+)\$?([0-9]+):\$?([A-Z]+)\$?([0-9]+)\)')


def evaluate(ws, addr):
    cache = {}
    def cell(a):
        if a in cache: return cache[a]
        cache[a] = 0.0
        v = ws[a].value
        if v is None:
            return 0.0
        if isinstance(v, (int, float)):
            cache[a] = float(v); return cache[a]
        s = str(v)
        if not s.startswith('='):
            try: cache[a] = float(s)
            except ValueError: cache[a] = 0.0
            return cache[a]
        e = SUMRE.sub(lambda m: '(' + '+'.join(f'{m.group(1)}{i}' for i in range(int(m.group(2)), int(m.group(4)) + 1)) + ')', s[1:])
        e = CELLRE.sub(lambda m: f'cell("{m.group(1)}{m.group(2)}")', e)
        try: cache[a] = eval(e, {'cell': cell})
        except Exception: cache[a] = float('nan')
        return cache[a]
    return cell(addr)


# ---------------- build sheets -----------------------------------------------
sp_ws, sp_gt = build_sp_sheet()

VARIANTS = [("original", "SuperGlue (original)"),
            ("max_sinkhorn", "SG max-Sinkhorn"),
            ("base2_softmax", "SG base2-softmax"),
            ("transpose_free", "SG transpose-free col"),
            ("fixed_thresh", "SG fixed-thresh match")]
gt_by_sheet = {}
for v, nm in VARIANTS:
    _, gt = build_sg_sheet(v, nm)
    gt_by_sheet[nm] = (v, gt)

meas_ws, meas_gt, meas_total_cyc, meas_rates = build_measured_sg_sheet()

# ---- Pipeline Summary sheet -------------------------------------------------
def build_summary():
    ws = make_sheet("Pipeline Summary")
    ws.cell(1, 1, "variant"); ws.cell(1, 2, "SuperPoint x2 [us]")
    ws.cell(1, 3, "SuperGlue [us]"); ws.cell(1, 4, "Total [us]"); ws.cell(1, 5, "FPS")
    for c in range(1, 6): cstyle(ws.cell(1, c), mob.cell(1, 1))
    for r, (v, nm) in enumerate(VARIANTS, start=2):
        sg_gt = gt_by_sheet[nm][1]
        ws.cell(r, 1, nm)
        ws.cell(r, 2, f"='SuperPoint'!Z{sp_gt}")
        ws.cell(r, 3, f"='{nm}'!Z{sg_gt}")
        ws.cell(r, 4, f"=B{r}+C{r}")
        ws.cell(r, 5, f"=1000000/D{r}")
        for c in range(1, 6): cstyle(ws.cell(r, c), mob.cell(5, 1))
    r = 2 + len(VARIANTS) + 1
    ws.cell(r, 1, "SG max-Sinkhorn (measured)")
    ws.cell(r, 2, f"='SuperPoint'!Z{sp_gt}")
    ws.cell(r, 3, f"='SuperGlue (measured)'!Z{meas_gt}")
    ws.cell(r, 4, f"=B{r}+C{r}")
    ws.cell(r, 5, f"=1000000/D{r}")
    for c in range(1, 6): cstyle(ws.cell(r, c), mob.cell(5, 1))
    return ws


build_summary()
wb.save(OUT)

# ---------------- validation -------------------------------------------------
print(f"\nwrote {OUT}\nsheets: {wb.sheetnames}\n")

sp_py = 2 * py_sp_per_image()
sp_xls = evaluate(wb["SuperPoint"], f"Z{sp_gt}")
sp_ok = abs(sp_py - sp_xls) / max(sp_py, 1) < 1e-9
print(f"{'SuperPoint (x2 images)':30s}  PY {sp_py:11.1f}  XLS {sp_xls:11.1f}  "
      f"FPS {evaluate(wb['SuperPoint'], f'Z{sp_gt+1}'):5.2f}  {'PASS' if sp_ok else 'FAIL'}")
all_ok = sp_ok
for v, nm in VARIANTS:
    py = py_sg(v)
    ws = wb[nm]; gt = gt_by_sheet[nm][1]
    xls = evaluate(ws, f"Z{gt}")
    fps = evaluate(ws, f"Z{gt + 1}")
    ok = abs(py - xls) / max(py, 1) < 1e-9
    all_ok &= ok
    print(f"{nm:30s}  PY {py:11.1f}  XLS {xls:11.1f}  FPS {fps:5.2f}  {'PASS' if ok else 'FAIL'}")
nm = "SuperGlue (measured)"
xls_meas = evaluate(wb[nm], f"Z{meas_gt}")
fps_meas = evaluate(wb[nm], f"Z{meas_gt + 1}")
print(f"{nm:30s}  PY {meas_total_cyc/1000:11.1f}  XLS {xls_meas:11.1f}  FPS {fps_meas:5.2f}  "
      f"{'PASS' if abs(meas_total_cyc/1000-xls_meas)/max(xls_meas,1)<1e-6 else 'FAIL'}")
print(f"\nmeasured rates: {dict((k, round(v, 3)) for k, v in meas_rates.items())}")
print(f"\nfull-pipeline totals (SP x2 + SG variant):")
for v, nm in VARIANTS:
    py = py_sg(v); full = sp_py + py
    print(f"  {nm:30s}  total {full:11.1f} us  FPS {1e6/full:5.2f}")
full_meas = sp_py + meas_total_cyc / 1000
print(f"  {'SuperGlue (measured)':30s}  total {full_meas:11.1f} us  FPS {1e6/full_meas:5.2f}")
print("\n", "ALL PASS" if all_ok else "SOME FAILED")
