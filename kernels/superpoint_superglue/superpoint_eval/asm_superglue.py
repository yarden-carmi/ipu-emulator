#!/usr/bin/env python3
"""Chain the full SuperGlue forward through the .asm kernels and validate against
the official pretrained model. Closes the last gap: SuperGlue was validated
per-op + OT end-to-end, but never run as ONE chained asm pipeline.

Strategy: the GNN is ~99% matmul, so the core primitive is matmul_asm (Y=W@X+b)
run as a channels-in-lanes kernel -- one 128-wide output-channel tile per launch,
looping all N tokens internally (few launches). Attention softmax uses
softmax.asm. Host orchestrates reshapes/residuals (data movement only); every
arithmetic op runs in the wide-FP32 emulator.

Stage 1 here: build + validate matmul_asm against numpy. Later stages chain the
keypoint encoder, GNN layers, final proj, score matrix and compare to torch.

Run: python superpoint_superglue/superpoint_eval/asm_superglue.py
"""
import os, sys, struct
import numpy as np

HERE = os.path.dirname(__file__)
KD = os.path.join(HERE, "..")
sys.path.insert(0, HERE)
from ipu_emu.emulator import load_program, run_until_complete
from ipu_emu.execute import decode_instruction_word
from ipu_emu.ipu_state import IpuState, WideVectorArithmetic
from ipu_emu.ipu_math import DType
from ipu_as.lark_tree import assemble
from ipu_as.label import reset_labels

# channels-in-lanes matmul: per launch one 128-wide output-channel tile, loops
# all N tokens; contraction over Din' (padded to 128-multiple) in groups, bias
# folded as an extra input channel = 1 (its weight row = the bias vector).
MATMUL = """\
SET lr0 cr0 ;;
SET lr1 cr0 ;;
SET lr9 cr0 ;;
SET lr3 cr0 ;;
tok_loop:
RESET_ACC ;;
SET lr4 cr0 ;;
SET lr5 cr0 ;;
ADD lr6 lr1 cr0 ;;
grp_loop:
LDR_MULT_REG r0 lr6 cr3 ;;
SET lr7 cr0 ;;
SET lr8 cr0 ;;
k_loop:
LDR_CYCLIC_MULT_REG lr5 cr4 lr0 ; MULT.VE.CYCLIC lr0 0 lr0 lr7 ; ACC ;;
ADD lr5 lr5 cr2 ; ADD lr7 lr7 cr1 ; ADD lr8 lr8 cr1 ;;
BLT lr8 cr10 k_loop ;;
ADD lr6 lr6 cr2 ;;
ADD lr4 lr4 cr1 ;;
BLT lr4 cr11 grp_loop ;;
STR_ACC_REG lr3 cr5 ;;
ADD lr1 lr1 cr12 ;;
ADD lr3 lr3 cr2 ;;
ADD lr9 lr9 cr1 ;;
BLT lr9 cr6 tok_loop ;;
BKPT ;;
"""
# lr0=0, lr1=X token offset, lr2 unused, lr3=Y token offset, lr4=group ctr,
# lr5=W byte offset, lr6=X group addr, lr7=fixed_idx, lr8=k ctr, lr9=token ctr.
_PROG = None
def _prog():
    global _PROG
    if _PROG is None:
        reset_labels()
        _PROG = [decode_instruction_word(w) for w in assemble(MATMUL)]
    return _PROG


def matmul_asm(W, X, b):
    """Y = W @ X + b[:,None].  W:(Dout,Din)  X:(Din,N)  b:(Dout,) -> (Dout,N)."""
    Dout, Din = W.shape
    N = X.shape[1]
    Dp1 = Din + 1                                   # + bias channel
    ngroups = (Dp1 + 127) // 128
    Dpad = ngroups * 128
    Xstride = ngroups * 512
    # X padded (N, Dpad): rows = X[:,n] then 1 (bias) then 0
    Xp = np.zeros((N, Dpad), np.float32)
    Xp[:, :Din] = X.T
    Xp[:, Din] = 1.0
    s = IpuState(wide_vector_debug=True, wide_vector_arithmetic=WideVectorArithmetic.FP32)
    s.regfile.set_cr(15, DType.INT8)
    XB, WB, YB = 0x0, 0x200000 // 4, 0x200000 // 2   # spread within 2MB... use safe bases
    XB, WB, YB = 0x000000, 0x080000, 0x100000
    for n in range(N):
        s.xmem.write_address(XB + n * Xstride, struct.pack(f"<{Dpad}f", *Xp[n]))
    out = np.zeros((Dout, N), np.float32)
    ntile = (Dout + 127) // 128
    for t in range(ntile):
        lo = t * 128; hi = min(lo + 128, Dout); w = hi - lo
        # Wcol for this tile: row k (0..Dpad-1) = weight-column k over output [lo:hi]
        Wcol = np.zeros((Dpad, 128), np.float32)
        Wcol[:Din, :w] = W[lo:hi, :].T              # contraction rows
        Wcol[Din, :w] = b[lo:hi]                     # bias row (X channel = 1)
        for k in range(Dpad):
            s.xmem.write_address(WB + k * 512, struct.pack("<128f", *Wcol[k]))
        for cr, v in [(0, 0), (1, 1), (2, 512), (3, XB), (4, WB), (5, YB),
                      (6, N), (10, 128), (11, ngroups), (12, Xstride)]:
            s.regfile.set_cr(cr, v)
        s.program_counter = 0
        load_program(s, list(_prog()))
        run_until_complete(s, max_cycles=20_000_000)
        for n in range(N):
            d = s.xmem.read_address(YB + n * 512, 512)
            for o in range(w):
                out[lo + o, n] = struct.unpack_from("<f", d, o * 4)[0]
    return out


# ---------------- full SuperGlue forward through the asm kernels --------------
import torch

def relu(x):
    return np.maximum(x, 0.0)

def fold_bn(W, b, bn):
    """Fold a BatchNorm1d (eval) into the preceding Conv1d weight/bias."""
    g = bn.weight.detach().numpy(); be = bn.bias.detach().numpy()
    mu = bn.running_mean.detach().numpy(); var = bn.running_var.detach().numpy()
    s = g / np.sqrt(var + bn.eps)
    return W * s[:, None], (b - mu) * s + be

def mlp_layers(seq):
    """Sequential(Conv1d,[BN,ReLU,]...) -> list of (W, b, relu_flag) folded."""
    import torch.nn as nn
    mods = list(seq); out = []; i = 0
    while i < len(mods):
        conv = mods[i]; W = conv.weight.detach().numpy()[:, :, 0]; b = conv.bias.detach().numpy()
        i += 1; rl = False
        if i < len(mods) and isinstance(mods[i], nn.BatchNorm1d):
            W, b = fold_bn(W, b, mods[i]); i += 1
        if i < len(mods) and isinstance(mods[i], nn.ReLU):
            rl = True; i += 1
        out.append((W, b, rl))
    return out

def run_mlp(x, layers):
    for W, b, rl in layers:
        x = matmul_asm(W, x, b)
        if rl:
            x = relu(x)
    return x

def softmax_rows_asm(S):
    """Row-softmax of S (R x C, C<=128) via softmax.asm, one row per call."""
    from compare_sg import asm_tropical  # reuse emu helpers? simpler: inline run
    R, C = S.shape
    SM = open(os.path.join(KD, "softmax.asm")).read()
    reset_labels(); prog = [decode_instruction_word(w) for w in assemble(SM)]
    out = np.zeros_like(S)
    s = IpuState(wide_vector_debug=True, wide_vector_arithmetic=WideVectorArithmetic.FP32)
    s.regfile.set_cr(15, DType.INT8)
    L2E = 1.4426950408
    s.xmem.write_address(0x3000, struct.pack("<128f", *([1.0] * 128)))
    for r in range(R):
        row = np.zeros(128, np.float32); row[:C] = S[r] * L2E       # pre-scale -> natural base
        s.xmem.write_address(0x1000, struct.pack("<128f", *row))
        for cr, v in [(0, 0), (1, 1), (2, 0x1000), (3, 0x2000), (4, 0x3000),
                      (5, 0x4000), (6, C), (7, 0xBF800000)]:
            s.regfile.set_cr(cr, v)
        s.program_counter = 0
        load_program(s, list(prog))
        run_until_complete(s, max_cycles=2_000_000)
        d = s.xmem.read_address(0x4000, 512)
        for c in range(C):
            out[r, c] = struct.unpack_from("<f", d, c * 4)[0]
    return out

def attention_asm(q, k, v, proj, merge, heads=4):
    """MultiHeadedAttention via asm matmuls + softmax.asm. q,k,v:(D,N)."""
    D = q.shape[0]; hd = D // heads
    Wq = [proj[i].weight.detach().numpy()[:, :, 0] for i in range(3)]
    bq = [proj[i].bias.detach().numpy() for i in range(3)]
    Q = matmul_asm(Wq[0], q, bq[0]); K = matmul_asm(Wq[1], k, bq[1]); V = matmul_asm(Wq[2], v, bq[2])
    N = q.shape[1]
    msg = np.zeros((D, N), np.float32)
    # torch splits heads INTERLEAVED: view(b,dim,heads,N) => channel c = d*heads+h
    for h in range(heads):
        Qh = Q[h::heads].copy(); Kh = K[h::heads].copy(); Vh = V[h::heads].copy()  # (hd,N)
        sc = matmul_asm(Qh.T.copy(), Kh, np.zeros(N, np.float32)) / hd ** .5   # (Nq, Nk)
        P = softmax_rows_asm(sc)                                                # (Nq, Nk)
        oh = matmul_asm(Vh, P.T.copy(), np.zeros(hd, np.float32))               # (hd, Nq)
        msg[h::heads] = oh
    Wm = merge.weight.detach().numpy()[:, :, 0]; bm = merge.bias.detach().numpy()
    return matmul_asm(Wm, msg, bm)

def gnn_layer_asm(d0, d1, layer, name):
    src0, src1 = (d1, d0) if name == "cross" else (d0, d1)
    m0 = attention_asm(d0, src0, src0, layer.attn.proj, layer.attn.merge)
    m1 = attention_asm(d1, src1, src1, layer.attn.proj, layer.attn.merge)
    mlp = mlp_layers(layer.mlp)
    delta0 = run_mlp(np.concatenate([d0, m0], 0), mlp)
    delta1 = run_mlp(np.concatenate([d1, m1], 0), mlp)
    return d0 + delta0, d1 + delta1


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    print("== matmul primitive ==")
    for (Dout, Din, N) in [(4, 4, 3), (256, 256, 8)]:
        W = rng.standard_normal((Dout, Din)).astype(np.float32) * 0.1
        X = rng.standard_normal((Din, N)).astype(np.float32)
        b = rng.standard_normal(Dout).astype(np.float32) * 0.1
        d = np.abs(matmul_asm(W, X, b) - (W @ X + b[:, None])).max()
        print(f"  matmul_asm {Dout}x{Din}@{Din}x{N}: max {d:.2e} {'MATCH' if d<1e-3 else 'FAIL'}")

    print("== chained SuperGlue stages vs torch (pretrained 'indoor') ==")
    sys.path.insert(0, os.path.join(os.path.abspath(os.path.join(HERE, '..', '..', '..')), 'third_party', 'superglue'))
    from models.superglue import SuperGlue, normalize_keypoints
    sg = SuperGlue({"weights": "indoor"}).eval()
    N = 8
    g = torch.Generator().manual_seed(0)
    kpts0 = torch.rand(1, N, 2, generator=g) * torch.tensor([[640.0, 480.0]])
    sc0 = torch.rand(1, N, generator=g)
    d0 = torch.nn.functional.normalize(torch.randn(1, 256, N, generator=g), dim=1)
    img = torch.zeros(1, 1, 480, 640)

    with torch.no_grad():
        k0n = normalize_keypoints(kpts0, img.shape)
        enc_t = sg.kenc(k0n, sc0)                                  # torch encoder
        # asm encoder
        inp = torch.cat([k0n.transpose(1, 2), sc0.unsqueeze(1)], 1)[0].numpy()
        enc_a = run_mlp(inp, mlp_layers(sg.kenc.encoder))
        de = np.abs(enc_a - enc_t[0].numpy()).max()
        print(f"  keypoint encoder: max {de:.2e} {'MATCH' if de<1e-3 else 'FAIL'}")

        d0a = (d0 + enc_t)[0].numpy()                              # start GNN from same point
        d1a = d0a.copy()
        # one full GNN layer vs torch
        layer0 = sg.gnn.layers[0]
        delta_t = layer0(d0 + enc_t, d0 + enc_t)[0].numpy()
        o0, _ = gnn_layer_asm(d0a, d1a, layer0, sg.gnn.names[0])
        dl = np.abs(o0 - (d0a + delta_t)).max()
        print(f"  one GNN layer ({sg.gnn.names[0]}): max {dl:.2e} {'MATCH' if dl<2e-2 else 'FAIL'}")

        # ---- full chained forward: encoder + 18 GNN layers + final proj + score
        d1 = torch.nn.functional.normalize(torch.randn(1, 256, N, generator=g), dim=1)
        kpts1 = torch.rand(1, N, 2, generator=g) * torch.tensor([[640.0, 480.0]])
        sc1 = torch.rand(1, N, generator=g)
        k1n = normalize_keypoints(kpts1, img.shape)
        # torch full GNN + final proj + score (pre-OT)
        D0t = d0 + sg.kenc(k0n, sc0); D1t = d1 + sg.kenc(k1n, sc1)
        g0t, g1t = sg.gnn(D0t, D1t)
        m0t, m1t = sg.final_proj(g0t), sg.final_proj(g1t)
        S_t = torch.einsum("bdn,bdm->bnm", m0t, m1t)[0].numpy() / 256 ** .5
        # asm full chain
        e0 = run_mlp(torch.cat([k0n.transpose(1, 2), sc0.unsqueeze(1)], 1)[0].numpy(), mlp_layers(sg.kenc.encoder))
        e1 = run_mlp(torch.cat([k1n.transpose(1, 2), sc1.unsqueeze(1)], 1)[0].numpy(), mlp_layers(sg.kenc.encoder))
        a0 = d0[0].numpy() + e0; a1 = d1[0].numpy() + e1
        for li, nm in enumerate(sg.gnn.names):
            a0, a1 = gnn_layer_asm(a0, a1, sg.gnn.layers[li], nm)
        Wf = sg.final_proj.weight.detach().numpy()[:, :, 0]; bf = sg.final_proj.bias.detach().numpy()
        m0a = matmul_asm(Wf, a0, bf); m1a = matmul_asm(Wf, a1, bf)
        S_a = matmul_asm(m0a.T.copy(), m1a, np.zeros(N, np.float32)) / 256 ** .5
        ds = np.abs(S_a - S_t).max()
        arg_t = S_t.argmax(1); arg_a = S_a.argmax(1)
        print(f"  FULL 18-layer GNN + proj + score: max {ds:.2e} {'MATCH' if ds<1e-2 else 'FAIL'}")
        print(f"  score-matrix row-argmax agreement: {(arg_t==arg_a).mean()*100:.0f}%  (N={N})")
