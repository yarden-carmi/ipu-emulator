# Theoretical operations vs. implementation

This document states, for every kernel in this directory, (1) the **theoretical
operation** as used in SuperPoint / SuperGlue, with math and reference
pseudocode, and (2) **what the IPU kernel actually computes**, as
instruction-level pseudocode mapped to the `.asm`, followed by a precise
**parity** statement.

Notation for inputs/outputs: vectors are length `N ≤ 128`; a "plane" is one
128-lane register of `float32` (512 bytes). All kernels run in wide-vector FP32
mode.

---

## 0. The IPU primitive vocabulary

Every kernel is built from this fixed set of operations. Let `a` be the 128-lane
accumulator `R_ACC`, `m` the multiply result `MULT_RES`, `rc` the cyclic operand
register `R_CYCLIC`, `r0` an input register, and `q ∈ {AAQ0..3}` a scalar.

```
load/store (contiguous 128 lanes):
    r0  <- XMEM[addr]                 LDR_MULT_REG
    rc  <- XMEM[addr]                 LDR_CYCLIC_MULT_REG
    XMEM[addr] <- a                   STR_ACC_REG
    XMEM[addr] <- p   (post_aaq)      STR_POST_AAQ_REG

multiply -> m:
    m_i = r0_i * rc_i                 MULT.EE
    m_i = r0_i * r0_i                 MULT.EE.RR
    m_i = q     * rc_i                MULT.VE.AAQ        (scalar * vector)

accumulate -> a:
    a   = 0                           RESET_ACC
    a   = m                           ACC.FIRST
    a  += m                           ACC
    a_i = m_i + q                     ACC.ADD_AAQ.FIRST  (add scalar to each lane)
    a_i = max(a_i, m_i, q)            ACC.MAX

reduce a[0..n) -> scalar q (one VLIW op):
    q = Σ_{i<n} a_i                   AGG sum  value
    q = (Σ_{i<n} a_i) * c             AGG sum  value_cr   (c = float bits in CR)
    q = 1 / Σ_{i<n} a_i               AGG sum  inv
    q = 1 / sqrt(Σ_{i<n} a_i)         AGG sum  inv_sqrt
    q = max_{i<n} a_i                 AGG max  value
    q = (max_{i<n} a_i) * c           AGG max  value_cr

activate a[0..n) -> p (post_aaq):
    p_i = f(a_i),  f ∈ {identity, relu, relu6, sigmoid, tanh, gelu,
                         softplus, elu, exp2=2^x}   ACTIVATE
```

Two recurring tricks:
- **Subtract a scalar `s` from a vector:** there is no vector−scalar op, so put
  `−s` in `q` (via `AGG ... value_cr` with `CR = float32_bits(−1)`) and use
  `ACC.ADD_AAQ.FIRST`.
- **Move activated values back into `a` for a second reduction:** `ACTIVATE`
  writes `post_aaq`, but `AGG` reduces `R_ACC`, so the kernel round-trips
  `STR_POST_AAQ_REG → XMEM → LDR_MULT_REG → MULT.EE×1 → ACC.FIRST`.

---

## 1. softmax — `softmax.asm`

### Theory
For logits `x ∈ ℝ^N`,
```
softmax(x)_i = exp(x_i) / Σ_j exp(x_j)
```
Numerically stabilized by subtracting `m = max_j x_j` (invariant: shifting all
logits by a constant does not change the result):
```
softmax(x)_i = exp(x_i − m) / Σ_j exp(x_j − m)
```

Reference:
```python
m = x.max()
e = np.exp(x - m)
y = e / e.sum()
```

### Implementation
The only exponential is base-2 (`exp2`). Using `2^t = exp(t·ln2)`, the caller
pre-scales logits by `log2(e)=1.4426950409` so `x' = x·log2(e)` and
`2^{x'_i − m'} = exp(x_i − m)`. Then:
```
a = x'                                   # load, MULT.EE x1, ACC.FIRST
q = max(a) * (-1) = -m'                  # AGG.FIRST max value_cr(-1)
a = a + q  = x' - m'                     # ACC.ADD_AAQ.FIRST
p = 2^a    = exp(x - m)                  # ACTIVATE exp2
a = p                                    # round-trip p -> XMEM -> R_ACC
q = 1 / Σ a = 1 / Σ exp(x - m)           # AGG.FIRST sum inv
m_i = q * exp_i ;  a = m                 # MULT.VE.AAQ ; ACC.FIRST
store a                                  # = softmax(x)
```

### Parity — **FULL** (with input convention)
Mathematically exact softmax in FP32 (validated: output sums to 1 and matches
`np.exp`-based softmax to 1e-4). The only obligation is the caller's
`log2(e)` pre-scale; without it the kernel computes the base-2 softmax
`2^{x_i}/Σ2^{x_j}`, a sharper but valid distribution.

---

## 2. L2 normalization — `l2_normalize.asm`

### Theory
```
y = x / ‖x‖₂ ,   ‖x‖₂ = sqrt(Σ_i x_i²)
```
```python
y = x / np.sqrt((x*x).sum())
```

### Implementation
```
m_i = x_i²            # MULT.EE.RR
a   = m               # ACC.FIRST
q   = 1/sqrt(Σ a)     # AGG.FIRST sum inv_sqrt  => 1/‖x‖
m_i = q * x_i         # MULT.VE.AAQ (rc = x)
a   = m ; store       # = x/‖x‖
```

### Parity — **FULL**
Exact (validated against `x/‖x‖` to 1e-5). `inv_sqrt` is guarded: if `‖x‖²=0`
the kernel outputs zeros (matches the usual eps-free convention's degenerate
case).

---

## 3. Layer normalization — `layernorm.asm`

### Theory
```
μ  = (1/N) Σ_i x_i
σ² = (1/N) Σ_i (x_i − μ)²
y_i = γ_i · (x_i − μ) / sqrt(σ² + ε) + β_i
```
```python
mu  = x.mean(); var = ((x-mu)**2).mean()
y   = gamma * (x - mu) / np.sqrt(var + eps) + beta
```

### Implementation
There is no scalar-add-before-rsqrt (so no `ε`), and `inv_sqrt` divides by
`sqrt(Σ(x−μ)²)` not `sqrt(mean)`. Since
`1/sqrt(Σ(x−μ)²) = 1/(sqrt(N)·σ)`, the missing `sqrt(N)` is folded into a
caller-supplied `γ' = γ·sqrt(N)`:
```
a = x                                    # load, MULT.EE x1, ACC.FIRST
q = (Σ a) * (-1/N) = -μ                   # AGG.FIRST sum value_cr(-1/N)
a = a + q = x - μ                         # ACC.ADD_AAQ.FIRST
a = (x-μ)²                                # store/reload, MULT.EE.RR, ACC.FIRST
q = 1/sqrt(Σ(x-μ)²) = 1/(sqrt(N)·σ)       # AGG.FIRST sum inv_sqrt
m_i = q·(x_i-μ) ; a = m                    # MULT.VE.AAQ -> z = (x-μ)/(sqrt(N)σ)
a = z·γ'        (γ' = γ·sqrt(N))           # store/reload, MULT.EE with γ'
a = a + β       (vector add via x1+ACC)    # = γ(x-μ)/σ + β
store a
```

### Parity — **NEAR-FULL** (two caveats)
- **No `ε`.** Exact LayerNorm omits ε only when `σ² > 0`; near-constant inputs
  diverge from the reference. (Validated equal to `(x−μ)/σ` to 1e-4 with γ=1,β=0.)
- **`sqrt(N)` convention:** the caller must pass `γ' = γ·sqrt(N)`. With that, the
  result is exact.

---

## 4. Max-pool / NMS core — `maxpool.asm`

### Theory
Spatial max-pool over a window `W(p)`:
```
y[p] = max_{q ∈ W(p)} x[q]
```
Non-maximum suppression keeps a detection iff it is the local maximum:
```
keep[p] = (x[p] == max_{q ∈ W(p)} x[q])  and  x[p] > thr
```
```python
# K window taps gathered per output position p:
y[p] = max(tap_0[p], ..., tap_{K-1}[p])
```

### Implementation
The caller pre-gathers the `K` window taps for every output lane into `K`
contiguous planes; the kernel does the element-wise max via the 3-way `ACC.MAX`
with `q` seeded to `−∞`:
```
q = -inf                                  # load -inf plane, reduce AGG max value
a = tap_0                                  # ACC.FIRST
for t in 1..K-1:                           # loop
    m = tap_t                              # LDR + MULT.EE x1
    a_i = max(a_i, tap_t_i, -inf)          # ACC.MAX  => running max
store a                                    # = max over taps
```

### Parity — **COMPUTE-ONLY**
The max reduction is exact (validated). Two pieces are **not** on-device:
- the **window gather** (im2col of `W(p)` into the tap planes) is the caller's job;
- NMS's **suppression test** `x[p]==max` and `>thr` needs an element-wise vector
  compare the ISA lacks (the `topk` and `argmax_match` kernels give the closest
  expressible surrogates).

---

## 5. Attention scores (QKᵀ) — `attention_scores.asm`

### Theory
```
S_{ij} = (q_i · k_j) / sqrt(d)     for queries i, keys j, dim d
```
```python
S = (Q @ K.T) / np.sqrt(d)         # then row-softmax, then S @ V
```

### Implementation (one score per call)
```
m_i = q_i · k_i                            # MULT.EE (rc = k)
a   = m                                    # ACC.FIRST
qs  = (Σ a) · (1/sqrt(d))                  # AGG.FIRST sum value_cr(1/sqrt(d))
m_i = qs · 1 ; a = m ; store               # broadcast scalar score to a plane
```

### Parity — **PER-ELEMENT FULL**
Each `S_{ij}` is exact (validated to 1e-4). The full `M×M` score matrix is a
**host loop** over (query,key) pairs — the ISA has no scatter to pack `M` scalar
scores into one vector. No attention masking is applied.

---

## 6. Sinkhorn iteration — `sinkhorn_iter.asm` (row) + `sinkhorn_col.asm` (column)

### Theory (log-domain optimal transport)
SuperGlue normalizes a dustbin-augmented log-score matrix `Z` toward a doubly
stochastic assignment by alternating row/column log-normalizations:
```
row:  Z_ij ← Z_ij − logsumexp_j(Z_ij)        # each row's exp-sum -> 1
col:  Z_ij ← Z_ij − logsumexp_i(Z_ij)        # each col's exp-sum -> 1
where  logsumexp_j(Z) = log Σ_j exp(Z_ij)
```
```python
for _ in range(iters):
    Z = Z - logsumexp(Z, axis=1, keepdims=True)
    Z = Z - logsumexp(Z, axis=0, keepdims=True)
```

### Implementation (max in place of logsumexp; both half-steps on-device)
No logarithm exists, so this uses the **tropical (max-plus) surrogate**
`logsumexp(Z) → max(Z)`.

**Row half-step — `sinkhorn_iter.asm`** (`Z_ij ← Z_ij − max_j Z_ij`):
```
for each row r:                            # loop over rows
    a = Z[r]                               # load, MULT.EE x1, ACC.FIRST
    q = max(a) · (-1) = -max_j Z_rj         # AGG.FIRST max value_cr(-1)
    a = a + q = Z[r] - max_j Z_rj           # ACC.ADD_AAQ.FIRST
    store a                                 # row-max is now 0
```
Validated: `[-2,0.5,1,-0.3] → [-3,-0.5,0,-1.3]`, row-max = 0.

**Column half-step — `sinkhorn_col.asm`** (`Z_ij ← Z_ij − max_i Z_ij`),
transpose-free. The per-COLUMN max is the ELEMENT-WISE max of the row vectors,
so it is built by sweeping contiguous rows with `ACC.MAX` (no strided column
read), then subtracted from every row:
```
# Pass 1: colmax_j = max_i Z_ij   (element-wise max across rows)
q = -inf                                   # seed for 3-way ACC.MAX
a = Z[row0]                                # ACC.FIRST
for r in 1..R-1:
    m = Z[row_r]                           # LDR + MULT.EE x1
    a_i = max(a_i, Z[row_r]_i, -inf)       # ACC.MAX  => running colmax
store a -> colmax ; negcol = -colmax       # negate via MULT.EE with -1 plane

# Pass 2: subtract the colmax vector from every row
for each row r:
    a = Z[row_r]                           # ACC.FIRST
    a = a + negcol                          # vector add: MULT.EE(-colmax)x1 + ACC
    store a                                 # column-max is now 0
```
Validated on a 3×4 matrix: after the kernel every column's max = 0 and each
entry equals `Z_ij − max_i Z_ij` to 1e-6. A full iteration = row kernel then
column kernel, entirely on-device.

### Parity — **APPROXIMATE (logsumexp→max); now FULL-iteration on-device**
- **logsumexp → max.** Since `logsumexp(Z) = max(Z) + log Σ exp(Z−max) ≥ max(Z)`,
  the max underestimates the true normalizer by `log(effective #entries)`. The
  two coincide only in the sharp/low-temperature limit. This yields the
  **hard/tropical** transport (min-/max-plus semiring), not the exact entropic
  Sinkhorn (which would need `log`).
- **No transpose needed.** Both the row and column half-steps are expressible on
  contiguous row planes (the column max via element-wise `ACC.MAX` across rows),
  so a complete iteration runs on-device without host transposition.

---

## 7. Hard-argmax matching — `argmax_match.asm`

### Theory
From a (soft) assignment row, SuperGlue reads out a hard match with a
mutual-nearest-neighbour + confidence test:
```
j*(i) = argmax_j P_ij
match(i) = j*(i)  iff  i == argmax_i P_{i,j*(i)}   (mutual)  and  P_{i,j*} > thr
```

### Implementation (temperature one-hot)
No lane-index can be extracted, so the kernel emits a **one-hot vector**
(the assignment row) instead of an integer index. With the caller pre-scaling
the row by temperature `T` (input is `T·x`):
```
a = T·x                                    # load, MULT.EE x1, ACC.FIRST
q = max(a)·(-1) = -max_j(T x_j)             # AGG.FIRST max value_cr(-1)
a = a + q = T(x - max x)                    # ACC.ADD_AAQ.FIRST   (=0 at argmax, <0 else)
p = 2^a                                     # ACTIVATE exp2  => ~1 at argmax, ~0 else
store p                                     # one-hot assignment row
```
Validated: `T=16`, `x=[.1,.9,.3,.2] → ≈[0,1,0,0]`.

### Parity — **APPROXIMATE**
- Emits a **one-hot vector**, not an integer index. As `T→∞` it converges to the
  exact indicator at the argmax; ties split mass equally.
- **Mutual-NN** = run on rows and on columns (`Pᵀ`) and AND the two one-hots
  (host orchestrates the two calls and the AND).
- The **confidence threshold** on `P_{i,j*}` is not applied here (combine with
  `topk`'s threshold or gate on the host).

---

## 8. Keypoint selection (top-k) — `topk.asm`

### Theory
SuperPoint keypoint selection:
```
candidates = { i : score_i > thr }          # confidence threshold
keep       = top_k(candidates by score)      # rank, keep k largest
# (followed by NMS spacing)
```
```python
mask = scores > thr
idx  = np.argsort(scores[mask])[-k:]
```

### Implementation
```
q = -thr                                    # max of constant thr-plane, ·(-1)
a = scores + q = scores - thr               # ACC.ADD_AAQ.FIRST
p = relu(a) = max(0, scores - thr)          # ACTIVATE relu  -> sel
store p                                     # thresholded scores
q = max(scores)                             # AGG.FIRST max value  (top-1 value)
broadcast q ; store                         # top-1 value plane
```

### Parity — **PARTIAL**
- **Threshold** `relu(score − thr)` is exact; **top-1** value is exact (both
  validated).
- **No ranked top-k indices.** Producing the `k` largest *indices* needs a sort
  + lane-index extraction the ISA lacks; the host selects the final `k` from the
  thresholded scores.

---

## 9. Depth-to-space / pixel-shuffle — `pixel_shuffle.asm`

### Theory
Pixel-shuffle with upscale `r` (SuperPoint detector head: `r=8`, drop the
dustbin channel, fold 64 channels into an 8×8 spatial block per cell):
```
y[c', h·r + a, w·r + b] = x[c'·r² + a·r + b, h, w]
                          0 ≤ a,b < r ;  c' < C/r²
```

### Implementation
```
for t in 0..K-1:                            # loop over planes
    a = src_plane_t                         # LDR + MULT.EE x1 + ACC.FIRST
    store a at dst_addr + t·dst_stride       # relocate plane
```

### Parity — **MINIMAL**
Performs the **plane-granular relocation** only (contiguous copy with
configurable source/destination strides). The **within-plane interleave** that
defines pixel-shuffle — scattering channel `c'·r²+a·r+b`'s pixels to
`(h·r+a, w·r+b)` — is a per-element scatter with non-contiguous addresses the
ISA cannot express; the host completes the interleave, or pre-lays-out the
source so a plane copy finishes the reshape.

---

# Part II — algorithmic advances, scaling, and the host/accelerator split

The kernels in Part I cover the base operations; the following extend them to
the full SuperPoint+SuperGlue workload and remove several gaps that initially
looked ISA-blocked. Pure-Python twins of every kernel live in `reference.py`
(`cell_nms_reference.py` additionally runs the kernel in the emulator and
compares against the dense path).

## 10. Channel-space NMS — `cell_nms.asm` (replaces depth-to-space + NMS)

### Insight
Depth-to-space only **relabels** channel `c` as sub-pixel `(a,b)=(c//8, c%8)`.
So "the local max in a cell's 8×8 block" is just a **reduction over the channel
(lane) axis** — no reshape, no spatial gather.

### Implementation (one cell = 64 channels in 64 lanes)
```
peak = max_c p_c                                   # AGG max  (top-1 confidence)
e_c  = 2^(T*(p_c - max p))                          # one-hot via ACTIVATE exp2
s    = softmax(T*p) = e / sum(e)                    # AGG sum inv + MULT.VE.AAQ
a    = sum_c s_c * (c//8) ;  b = sum_c s_c * (c%8)  # MULT.EE + AGG sum (coord dot)
```
The sub-pixel location is recovered **as a value** (expected coordinate), so no
lane-index extraction is needed.

### Parity — **EXACT formula, host-free**
Kernel matches the soft-argmax formula to **3e-7**. On peaky cells: **100%
recall**, full agreement with dense depth-to-space + 2-D NMS (validated in
`cell_nms_reference.py`). Cost ≈ **30 cyc/cell**, one launch, no gather/compare/
lane-index. This is the key win: a host-bound chain (reshape→pool→compare→index)
collapses to one self-contained kernel.

## 11. Gather-free max-pool — `maxpool_shift.asm`

### Insight (precompute for fixed image size)
For a fixed `W`, the 3×3 window taps are **constant flat-index offsets**
`dy*W+dx`, so each tap is a **contiguous shifted load** — the im2col gather
disappears:
```
y[p_lane] = max over dy,dx in {0,1,2} of heatmap[ p + dy*rowbytes + dx*elem ]
```
The kernel sweeps all output tiles and the 3×3 window with on-device `BLT`
loops; offsets are generated by the `dy`/`dx` counters from `rowbytes`.

### Parity — **EXACT, host-free**
64×126 3×3 max-pool: **0/8064 mismatches**, **6,155 cycles in one launch**, no
host gather (`-inf`-padded layout precomputed offline; per-frame the host only
writes pixels). Moves `maxpool` into the no-host set for fixed windows.

## 12. Multi-tile pattern (C > 128) — `*_mt.asm`

Rows/vectors wider than 128 are handled as `T=⌈C/128⌉` contiguous 128-lane
tiles. Two folds make reductions exact across tiles:
```
global max  : AGG max re-includes the running AAQ  -> max over all tiles
global sum  : accumulate partial vectors with ACC, one final AGG sum
```
Used by `sinkhorn_iter_mt`, `sinkhorn_col_mt`, `argmax_match_mt`, `topk_mt`.
Measured: full Sinkhorn iteration on **513×513** = 40,022 (row) + 46,768 (col)
= **86,790 cyc**; one-hot over a **512-wide** row = **75 cyc**.

## 13. Threshold-calibrated top-k — `topk_mt.asm`

### Implementation
```
top-1 : aaq0 = max over tiles
count : aaq1 = sum_c sigmoid(T*(x_c - tau))          # soft survivor count
```
The host binary-searches `tau` until `count==k` (~13 passes), then `{x>tau}` is
the top-k set. **The selection set is exact**; only the *count* is a soft
estimate and the *ranked order/indices* stay host.

### Parity — **near-exact set**
Top-512-of-4800: **99.6–99.95%** set overlap (sharper `T` → fewer ties), fixed
~11K-cycle cost. With a **fixed `τ`** (SuperPoint's actual detector) it is a
single 853-cyc pass with **exact** selection.

## 14. Multi-tile one-hot matching — `argmax_match_mt.asm`

512-wide matching rows: fold the global max over 4 tiles, emit the one-hot per
tile. **512×512 mutual-NN matched a NumPy integer-argmax reference exactly
(512/512, identical)** at 75 cyc/row; the integer index / cross-pass AND remain
host.

---

# Host vs. accelerator

Every kernel runs **start→`BKPT` autonomously** once launched (loops use
on-device `BLT`). The host (RISC-V over APB) does program load, CR/data setup,
kernel sequencing, and the **value-dependent control** the datapath can't do
(branching needs `LR/CR`, but reductions land in `AAQ` floats — no `AAQ→LR`
path). 

**No-host start-to-end (launch + read):** `softmax`, `l2_normalize`,
`layernorm`, `cell_nms`, `sinkhorn_iter[_mt]`, `sinkhorn_col[_mt]`,
`argmax_match[_mt]`, `topk[_mt]` (fixed-τ), `maxpool_shift`.

**Need host glue (addressing/control, never arithmetic):** `maxpool` (gather),
`attention_scores` (matrix loop), `pixel_shuffle` (scatter), `topk` exact-k
(calibration loop), mutual-NN (AND + integer index), descriptor `grid_sample`
(data-dependent addresses).

**Precompute principle:** any *geometry-only* addressing (fixed-window taps,
pooling, reshape) is computed once for a fixed image size — and for sliding
windows it collapses to constant-offset contiguous loads (`maxpool_shift`).
Only *value-dependent* addressing (keypoint sampling, sorting, indices) resists
precomputation and stays host.

---

## Summary

| Kernel | Parity | Host glue |
|--------|--------|-----------|
| `softmax` | **Full** (exact) | base-2 → pre-scale by `log2(e)` |
| `l2_normalize` | **Full** | — |
| `layernorm` | **Near-full** | no `ε`; fold `sqrt(N)` into `γ` |
| `attention_scores` | **Full per element** | host loops keys; no mask |
| `maxpool` | **Compute-only** | host window gather + suppression |
| `maxpool_shift` | **Full, host-free** | offline padded layout only |
| `cell_nms` | **Full, host-free** | (replaces depth-to-space) |
| `sinkhorn_iter[_mt]` + `sinkhorn_col[_mt]` | **Approx, full iter on-device** | `logsumexp→max` |
| `argmax_match[_mt]` | **Exact match-set** | integer index + mutual AND |
| `topk[_mt]` | **Exact set (fixed τ); ~99.9% (calibrated)** | sort / calibration loop |
| `pixel_shuffle` | **Minimal** | host within-plane interleave |

**Root-cause ISA gaps:** (1) no lane-index→scalar move, (2) no element-wise
vector compare, (3) no gather/scatter, (4) no logarithm (`exp2` only), (5) no
scalar-add before `rsqrt`. Several apparent blocks were removed by reframing:
the Sinkhorn column step (element-wise `ACC.MAX` across rows, no transpose),
NMS (channel-axis reduction, no depth-to-space), keypoint coordinates and the
matched index (recovered as *values* via coordinate/one-hot dots, no lane
index), and the max-pool gather (constant-offset shifted loads). What genuinely
remains host is **data-dependent addressing, sorting, and value-dependent
branching** — control/memory, not arithmetic.
