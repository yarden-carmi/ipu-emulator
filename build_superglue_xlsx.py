#!/usr/bin/env python3
"""Build SuperGlue performance-model sheets in the MobileViT spreadsheet's style,
broken down to BARE OPERATIONS (one elementary vector op per row -- multiply,
accumulate, max, sum, exp2, sub, recip, move, compare) so there are no hidden
pass-count multipliers. A multiply-accumulate is two bare rows (multiply +
accumulate); a softmax is max+sub+exp2+sum+mul; etc.  Adds an FPS cell.

Same column layout / formulas / formatting as the MobileViT sheet (theme fills,
borders, coloured header, frozen panes, yellow/green section brackets, totals).
N=512 keypoints, D=256, 4 heads (head_dim 64), 18 GNN self/cross layers over an
image pair, Sinkhorn T=100 on the (N+1)x(N+1) assignment matrix.
"""
import openpyxl
from openpyxl.utils import get_column_letter
from copy import copy

SRC = "/root/.claude/uploads/8c029b26-5a4c-4fdb-96ec-f1feb856d256/fa8d8c24-IPU_2.xlsx"
OUT = "/home/user/ipu-emulator/IPU_2_SuperGlue.xlsx"

N, D, HEADS, HD, NP1, T = 512, 256, 4, 64, 513, 100
GNN_BLOCKS = 18 * 2
COLS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

wb = openpyxl.load_workbook(SRC)
mob = wb["MobileViT"]
HEADER = [mob.cell(1, c).value for c in range(1, 27)]


def cstyle(dst, src):
    dst.font = copy(src.font); dst.fill = copy(src.fill); dst.border = copy(src.border)
    dst.alignment = copy(src.alignment); dst.number_format = src.number_format


def make_sheet(name):
    ws = wb.create_sheet(name)
    for c in range(1, 27):
        ws.cell(1, c, HEADER[c - 1]); cstyle(ws.cell(1, c), mob.cell(1, c))
    for i in range(1, 27):
        L = get_column_letter(i)
        ws.column_dimensions[L].width = mob.column_dimensions[L].width
    ws.freeze_panes = "B2"
    ws["Y2"], ws["Z2"] = 1, 8
    return ws


# bare-op row: kind 'mac' -> S=C*D*G/128 (one of mul/add); 'ew' -> S=C*D/128; 'reshape' -> 0
def emit(ws, r, group, op, c, d, g, kind):
    S = {"mac": f"=C{r}*D{r}*G{r}/128", "ew": f"=C{r}*D{r}/128", "reshape": "0"}[kind]
    vals = {"A": group, "B": op, "C": c, "D": d, "E": 0, "F": 0, "G": g, "H": 0,
            "I": 0, "J": 0, "K": 0, "L": c, "M": d, "N": 0,
            "O": f"=C{r}*D{r}/1024", "P": 0, "Q": 0, "R": f"=L{r}*M{r}/1024", "S": S}
    if kind == "reshape":
        vals["S"] = "0"; vals["T"] = 0
        for cc in "UVWX": vals[cc] = 0
    else:
        vals["T"] = f"=(S{r}/$Y$2)/1000"
        for cc, src in [("U", "O"), ("V", "P"), ("W", "Q"), ("X", "R")]:
            vals[cc] = f"=(({src}{r}*8*1024)/$T{r})/1000"
    for col, v in vals.items():
        cstyle(ws.cell(r, COLS.index(col) + 1, v), mob.cell(5, COLS.index(col) + 1))
    return r + 1


def bracket(ws, r0, r1, text, theme_cell):
    ws.merge_cells(start_row=r0, start_column=27, end_row=r1, end_column=27)
    c = ws.cell(r0, 27, text); cstyle(c, theme_cell); c.alignment = copy(theme_cell.alignment)


def total(ws, r, label, zformula):
    cstyle(ws.cell(r, 25, label), mob.cell(68, 25))
    cstyle(ws.cell(r, 26, zformula), mob.cell(68, 26))


YTAG, AATAG = mob.cell(5, 25), mob.cell(27, 27)

# bare-op recipes (return list of (op_name, c, d, g, kind)) ----------------------
def mac(c, d, K):                       # out (c x d) elems, contraction K -> mul + add
    return [("multiply", c, d, K, "mac"), ("accumulate", c, d, K, "mac")]

def ew(name, c, d):
    return [(name, c, d, 0, "ew")]


def build(ws, variant):
    r = 2
    r = emit(ws, r, "Input", "kpt+desc load", D, N, 0, "reshape")

    # ---- Keypoint encoder (MLP 3->32->64->128->256), x2 images ----
    enc0 = r
    for cin, cout in [(3, 32), (32, 64), (64, 128), (128, 256)]:
        for op, c, d, g, k in mac(cout, N, cin):
            r = emit(ws, r, f"KptEnc {cin}->{cout}", op, c, d, g, k)
    for op, c, d in [("add", D, N)]:
        r = emit(ws, r, "Residual desc+=enc", op, c, d, 0, "ew")
    enc_end = r - 1
    bracket(ws, enc0, enc_end, "Keypoint Encoder (x2 images)", YTAG)
    total(ws, enc_end, "Encoder total [us] ->", f"=2*(SUM(T{enc0}:T{enc_end}))")
    z_enc = enc_end

    # ---- One GNN attention block (expanded; heads x4; block x36) ----
    blk0 = r
    for nm in ("Q", "K", "V"):
        for op, c, d, g, k in mac(D, N, D):
            r = emit(ws, r, f"{nm} proj D->D", op, c, d, g, k)
    for nm in "QKV":
        r = emit(ws, r, f"head reshape {nm}", "reshape", HD, N, 0, "reshape")
    head0 = r
    for op, c, d, g, k in mac(N, N, HD):
        r = emit(ws, r, "QKt (1 head)", op, c, d, g, k)
    r = emit(ws, r, "scale 1/sqrt(d)", "multiply", N, N, 0, "ew")
    sm = "exp2" if variant == "base2_softmax" else "exp2"   # softmax uses exp2 either way
    for op in ["max", "subtract", sm, "sum", "multiply(1/Z)"]:
        r = emit(ws, r, "softmax (1 head)", op, N, N, 0, "ew")
    for op, c, d, g, k in mac(N, HD, N):
        r = emit(ws, r, "attn @ V (1 head)", op, c, d, g, k)
    head_end = r - 1
    bracket(ws, head0, head_end, "4X (heads)", AATAG)
    rcc = r; r = emit(ws, r, "concat heads", "reshape", D, N, 0, "reshape")
    total(ws, rcc, "total 4 heads [us]", f"=4*(SUM(T{head0}:T{head_end}))")
    for op, c, d, g, k in mac(D, N, D):
        r = emit(ws, r, "out proj D->D", op, c, d, g, k)
    r = emit(ws, r, "Residual x+=msg", "add", D, N, 0, "ew")
    for op, c, d, g, k in mac(2 * D, N, 2 * D):
        r = emit(ws, r, "merge MLP 2D->2D", op, c, d, g, k)
    r = emit(ws, r, "merge MLP relu", "relu", 2 * D, N, 0, "ew")
    for op, c, d, g, k in mac(D, N, 2 * D):
        r = emit(ws, r, "merge MLP 2D->D", op, c, d, g, k)
    r = emit(ws, r, "Residual x+=mlp", "add", D, N, 0, "ew")
    blk_end = r - 1
    bracket(ws, blk0, blk_end, f"Attentional GNN block (x{GNN_BLOCKS}=18 layers x2 img)", YTAG)
    gnn_z = f"={GNN_BLOCKS}*(SUM(T{blk0}:T{blk_end})+3*(SUM(T{head0}:T{head_end})))"
    total(ws, blk_end, "GNN total [us] ->", gnn_z)
    z_gnn = blk_end

    # ---- Matching ----
    mat0 = r
    rf = r
    for op, c, d, g, k in mac(D, N, D):
        r = emit(ws, r, "final proj D->D", op, c, d, g, k)
    rfe = r - 1
    rsc = r
    for op, c, d, g, k in mac(N, N, D):
        r = emit(ws, r, "score = A^T B", op, c, d, g, k)
    rsce = r - 1
    r = emit(ws, r, "scale 1/sqrt(d)", "multiply", N, N, 0, "ew")
    rsc_mul = r - 1
    r = emit(ws, r, "dustbin augment", "reshape", NP1, NP1, 0, "reshape")

    # Sinkhorn iteration -> x T
    rsk = r
    if variant == "max_sinkhorn":
        row_ops = ["sum", "multiply(1/r)"]; col_ops = ["sum", "multiply(1/c)"]
    else:
        row_ops = ["max", "exp2", "sum", "log", "subtract"]
        col_ops = ["max", "exp2", "sum", "log", "subtract"]
    for op in row_ops:
        r = emit(ws, r, "Sinkhorn row norm", op, NP1, NP1, 0, "ew")
    if variant != "transpose_free":
        r = emit(ws, r, "Sinkhorn col transpose", "move", NP1, NP1, 0, "ew")
    for op in col_ops:
        r = emit(ws, r, "Sinkhorn col norm", op, NP1, NP1, 0, "ew")
    sk_end = r - 1
    bracket(ws, rsk, sk_end, f"Sinkhorn (x{T} iters)", AATAG)
    total(ws, sk_end, f"Sinkhorn total (x{T}) [us] ->", f"={T}*(SUM(T{rsk}:T{sk_end}))")
    z_sink = sk_end

    rd = r
    if variant == "fixed_thresh":
        for op in ["max(argmax)", "compare(threshold)"]:
            r = emit(ws, r, "match readout", op, NP1, NP1, 0, "ew")
    else:
        for op in ["max", "exp2", "sum", "subtract", "multiply"]:
            r = emit(ws, r, "match readout (dual-softmax)", op, NP1, NP1, 0, "ew")
    rde = r - 1
    bracket(ws, mat0, rde, "Matching", YTAG)
    read_z = f"=SUM(T{rd}:T{rde})"

    # ---- Grand total + FPS ----
    gt = r + 1
    cstyle(ws.cell(gt, 1, "GRAND TOTAL [us]"), mob.cell(5, 1))
    grand = (f"=Z{z_enc}+Z{z_gnn}+2*(SUM(T{rf}:T{rfe}))+SUM(T{rsc}:T{rsce})"
             f"+T{rsc_mul}+Z{z_sink}+{read_z[1:]}")
    total(ws, gt, "variant: " + variant, grand)
    fr = gt + 1
    cstyle(ws.cell(fr, 1, "FPS (image pair)"), mob.cell(5, 1))
    total(ws, fr, "frames / sec ->", f"=1000000/Z{gt}")
    return ws


for v, nm in [("original", "SuperGlue (original)"),
              ("max_sinkhorn", "SG max-Sinkhorn"),
              ("base2_softmax", "SG base2-softmax"),
              ("transpose_free", "SG transpose-free col"),
              ("fixed_thresh", "SG fixed-thresh match")]:
    build(make_sheet(nm), v)

wb.save(OUT)
print("wrote", OUT, "\nsheets:", wb.sheetnames)
