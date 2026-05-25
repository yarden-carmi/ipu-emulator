#!/usr/bin/env python3
"""Pure-Python reference implementations for every SuperPoint/SuperGlue kernel.

Each function is the mathematical twin of the corresponding ``.asm`` kernel and
documents (a) the original operation, (b) what the kernel computes, and (c) the
parity relationship. These references are what the emulator-run comparisons
(cell_nms_reference.py and the in-repo measurements) check against.

No third-party deps; pure Python (lists/`math`). Run ``python reference.py`` for
a self-check of the formulas on small inputs.
"""
from __future__ import annotations
import math
from typing import Sequence

# ============================================================================
# Exact reductions / activations  (kernels: softmax, l2_normalize, layernorm)
# ============================================================================

def softmax_ref(x: Sequence[float], base2: bool = False) -> list[float]:
    """softmax(x)_i = e^{x_i} / sum_j e^{x_j}, stabilized by max-subtraction.

    Kernel: softmax.asm. The kernel uses 2^(.) (exp2), so to match natural-base
    softmax the caller pre-scales logits by log2(e). With base2=True this
    reference mimics the kernel's raw base-2 output. Parity: EXACT.
    """
    m = max(x)
    if base2:
        e = [2.0 ** (xi - m) for xi in x]
    else:
        e = [math.exp(xi - m) for xi in x]
    Z = sum(e)
    return [ei / Z for ei in e]

def l2_normalize_ref(x: Sequence[float]) -> list[float]:
    """y = x / ||x||_2.  Kernel: l2_normalize.asm (uses inv_sqrt). Parity: EXACT.
    Degenerate ||x||==0 -> zeros (guarded reciprocal)."""
    n = math.sqrt(sum(v * v for v in x))
    return [0.0] * len(x) if n == 0 else [v / n for v in x]

def layernorm_ref(x: Sequence[float], gamma: Sequence[float], beta: Sequence[float],
                  eps: float = 0.0) -> list[float]:
    """y_i = gamma_i * (x_i - mu)/sqrt(sigma^2 + eps) + beta_i.

    Kernel: layernorm.asm computes (x-mu)*rsqrt(sum (x-mu)^2) and folds sqrt(N)
    into a caller-supplied gamma' = gamma*sqrt(N); it omits eps. Parity: EXACT
    when eps==0 (else differs by sqrt(sigma^2/(sigma^2+eps)))."""
    N = len(x)
    mu = sum(x) / N
    var = sum((xi - mu) ** 2 for xi in x) / N
    inv = 1.0 / math.sqrt(var + eps) if var + eps > 0 else 0.0
    return [gamma[i] * (x[i] - mu) * inv + beta[i] for i in range(N)]

def attention_score_ref(q: Sequence[float], k: Sequence[float], d: int | None = None) -> float:
    """s = (q . k) / sqrt(d).  Kernel: attention_scores.asm (one (q,k) per call).
    Parity: EXACT per element; full M x N matrix is a host loop over keys."""
    d = d or len(q)
    return sum(qi * ki for qi, ki in zip(q, k)) / math.sqrt(d)

# ============================================================================
# Pooling / NMS  (kernels: maxpool, maxpool_shift, cell_nms)
# ============================================================================

def maxpool_taps_ref(taps: Sequence[Sequence[float]]) -> list[float]:
    """y[i] = max_t taps[t][i].  Kernel: maxpool.asm (host pre-gathers K taps).
    Parity: EXACT for the max; the window gather + NMS suppression are host."""
    K = len(taps); n = len(taps[0])
    return [max(taps[t][i] for t in range(K)) for i in range(n)]

def maxpool3x3_shift_ref(heat: Sequence[Sequence[float]], pad_value: float = -3.0e38
                         ) -> list[list[float]]:
    """3x3 sliding-window max-pool with -inf border padding.

    Kernel: maxpool_shift.asm -- GATHER-FREE: the 9 taps are constant flat
    offsets (dy*W+dx), each a contiguous shifted load. Parity: EXACT, host-free
    (single launch). ``heat`` is HxW; returns HxW of local 3x3 maxima."""
    H = len(heat); W = len(heat[0])
    def g(y, x):
        return heat[y][x] if 0 <= y < H and 0 <= x < W else pad_value
    out = [[pad_value] * W for _ in range(H)]
    for y in range(H):
        for x in range(W):
            out[y][x] = max(g(y + dy, x + dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1))
    return out

def cell_nms_ref(p: Sequence[float], T: float = 64.0):
    """Channel-space keypoint detection over one cell's 64 sub-grid channels.

    Kernel: cell_nms.asm (depth-to-space FREE). Returns (peak, a, b) where
        peak = max_c p_c
        (a,b) = soft-argmax expected coords = sum_c softmax(T*p)_c * (c//8, c%8)
    As T->inf, (a,b) -> the hard argmax sub-position. Parity: EXACT formula
    (kernel matches to ~3e-7); detection-accuracy depends on cell peakiness."""
    m = max(p)
    e = [2.0 ** (T * (pi - m)) for pi in p]; Z = sum(e); s = [ei / Z for ei in e]
    a = sum(s[c] * (c // 8) for c in range(64))
    b = sum(s[c] * (c % 8) for c in range(64))
    return m, a, b

# ============================================================================
# Optimal transport  (kernels: sinkhorn_iter[_mt], sinkhorn_col[_mt])
# ============================================================================

def sinkhorn_row_ref(Z: list[list[float]]) -> list[list[float]]:
    """Log-domain row half-step, MAX surrogate for logsumexp:
        Z_ij <- Z_ij - max_j Z_ij      (each row max -> 0)
    Kernels: sinkhorn_iter.asm / sinkhorn_iter_mt.asm (multi-tile for C>128).
    Parity: tropical approximation (logsumexp -> max); exact for the max op."""
    return [[v - max(row) for v in row] for row in Z]

def sinkhorn_col_ref(Z: list[list[float]]) -> list[list[float]]:
    """Log-domain column half-step (transpose-free):
        Z_ij <- Z_ij - max_i Z_ij      (each col max -> 0)
    Kernels: sinkhorn_col.asm / sinkhorn_col_mt.asm. The per-column max is the
    element-wise max of the row vectors, built by ACC.MAX across contiguous
    rows -- no transpose. Parity: tropical approximation; exact for the max."""
    R = len(Z); C = len(Z[0])
    colmax = [max(Z[i][j] for i in range(R)) for j in range(C)]
    return [[Z[i][j] - colmax[j] for j in range(C)] for i in range(R)]

def sinkhorn_iteration_ref(Z: list[list[float]]) -> list[list[float]]:
    """One full (max-domain) Sinkhorn iteration = row then column half-step.
    Both halves are on-device (transpose-free); host only sequences the two."""
    return sinkhorn_col_ref(sinkhorn_row_ref(Z))

# ============================================================================
# Selection / matching  (kernels: argmax_match[_mt], topk[_mt])
# ============================================================================

def argmax_onehot_ref(x: Sequence[float], T: float = 16.0) -> list[float]:
    """Temperature hardmax one-hot: e_i = 2^{T (x_i - max_j x_j)}.

    Kernels: argmax_match.asm / argmax_match_mt.asm (multi-tile for C>128).
    As T->inf, e -> indicator(argmax). Parity: argmax(e) == argmax(x) exactly
    once T is sharp (verified 512/512 at C=512); integer index / mutual AND are
    host. Mutual-NN match = run on rows of P and rows of P^T, then AND."""
    m = max(x)
    return [2.0 ** (T * (xi - m)) for xi in x]

def mutual_nn_ref(P: list[list[float]]):
    """Mutual nearest-neighbour matches from assignment scores P (host-side
    argmax of the per-row / per-col one-hots). Returns set of (i,j)."""
    M, N = len(P), len(P[0])
    row = [max(range(N), key=lambda j: P[i][j]) for i in range(M)]
    col = [max(range(M), key=lambda i: P[i][j]) for j in range(N)]
    return {(i, row[i]) for i in range(M) if col[row[i]] == i}

def topk_threshold_ref(x: Sequence[float], tau: float) -> list[float]:
    """Confidence-threshold selection: sel_i = relu(x_i - tau).
    Kernel: topk.asm / topk_mt.asm. Parity: EXACT (the survivor set {x>tau})."""
    return [max(0.0, xi - tau) for xi in x]

def topk_count_ref(x: Sequence[float], tau: float, T: float = 64.0) -> float:
    """Soft survivor count = sum_i sigmoid(T (x_i - tau)) ~ |{i: x_i > tau}|.
    Kernel: topk_mt.asm (aaq1). Drives a HOST binary search on tau to hit k.
    Parity: approximate count (sigmoid soft band); selection set is exact."""
    def sig(z):
        return 1.0 / (1.0 + math.exp(-z)) if z >= 0 else math.exp(z) / (1.0 + math.exp(z))
    return sum(sig(T * (xi - tau)) for xi in x)

def topk_calibrate_ref(x: Sequence[float], k: int, T: float = 64.0, iters: int = 14) -> float:
    """Host binary-search on tau so that count(tau) == k (uses topk_count_ref).
    Returns tau; {i: x_i > tau} is the top-k set. Models the host-in-the-loop
    calibration: device computes the count, host compares to k and bisects."""
    lo, hi = min(x), max(x)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if topk_count_ref(x, mid, T) > k:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)

# ============================================================================
# Reshape  (kernel: pixel_shuffle)
# ============================================================================

def depth_to_space_ref(x: list[list[float]], Hc: int, Wc: int, r: int = 8) -> list[list[float]]:
    """y[c', h*r+a, w*r+b] = x_cell[h][w][c'*r^2 + a*r + b].

    Kernel: pixel_shuffle.asm realizes only the plane-granular relocation; the
    within-plane interleave (the a,b scatter) is host. Parity: PARTIAL. (In the
    SuperPoint detector this whole step is replaced by cell_nms, which avoids
    depth-to-space entirely.) ``x`` is Hc*Wc cells each length r*r."""
    H, W = Hc * r, Wc * r
    out = [[0.0] * W for _ in range(H)]
    for h in range(Hc):
        for w in range(Wc):
            v = x[h * Wc + w]
            for c in range(r * r):
                out[h * r + c // r][w * r + c % r] = v[c]
    return out


# ============================================================================
# self-check
# ============================================================================
if __name__ == "__main__":
    import random
    random.seed(0)
    print("softmax sums to 1 :", abs(sum(softmax_ref([1.0, 2.0, 3.0, -1.0])) - 1.0) < 1e-9)
    v = [3.0, 4.0]; print("l2 unit norm     :", abs(math.hypot(*l2_normalize_ref(v)) - 1.0) < 1e-9)
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    ln = layernorm_ref(x, [1.0] * 5, [0.0] * 5)
    print("layernorm ~0 mean :", abs(sum(ln) / 5) < 1e-9)
    p = [0.0] * 64; p[27] = 5.0
    pk, a, b = cell_nms_ref(p); print("cell_nms argmax27:", (round(a), round(b)) == (3, 3))
    Z = [[random.uniform(-3, 3) for _ in range(5)] for _ in range(4)]
    it = sinkhorn_iteration_ref(Z)
    print("sinkhorn col max0:", all(abs(max(it[i][j] for i in range(4))) < 1e-9 for j in range(5)))
    xs = [random.gauss(0, 1) for _ in range(20)]
    oh = argmax_onehot_ref(xs, T=32); print("onehot==argmax   :", oh.index(max(oh)) == xs.index(max(xs)))
    tau = topk_calibrate_ref(xs, 5); sel = [i for i, v in enumerate(xs) if v > tau]
    print("topk ~5 selected :", abs(len(sel) - 5) <= 1)
    hm = [[random.random() for _ in range(6)] for _ in range(6)]
    mp = maxpool3x3_shift_ref(hm); print("maxpool3x3 shape :", len(mp) == 6 and len(mp[0]) == 6)
    print("all reference self-checks done.")
