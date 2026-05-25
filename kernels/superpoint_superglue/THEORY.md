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
    p_i = f(a_i),  f ∈ {relu, relu6, sigmoid, tanh, gelu, silu,
                         softplus, elu, prelu, identity, exp2=2^x}   ACTIVATE
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

## 6. Sinkhorn iteration — `sinkhorn_iter.asm`

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

### Implementation (max in place of logsumexp; row half-step)
No logarithm exists, so this uses the **tropical (max-plus) surrogate**
`logsumexp_j(Z) → max_j(Z)`:
```
for each row r:                            # loop over rows
    a = Z[r]                               # load, MULT.EE x1, ACC.FIRST
    q = max(a) · (-1) = -max_j Z_rj         # AGG.FIRST max value_cr(-1)
    a = a + q = Z[r] - max_j Z_rj           # ACC.ADD_AAQ.FIRST
    store a                                 # row-max is now 0
```
Validated: `[-2,0.5,1,-0.3] → [-3,-0.5,0,-1.3]`, row-max = 0.

### Parity — **APPROXIMATE + ROW-ONLY**
- **logsumexp → max.** Since `logsumexp(Z) = max(Z) + log Σ exp(Z−max) ≥ max(Z)`,
  the max underestimates the true normalizer by `log(effective #entries)`. The
  two coincide only in the sharp/low-temperature limit. This yields the
  **hard/tropical** transport (min-/max-plus semiring), not the exact entropic
  Sinkhorn.
- **Row half only.** The column half-step `Z_ij −= max_i Z_ij` needs transposed
  (strided) column reads the contiguous-only load cannot gather; the host
  transposes `Z` (or stores `Zᵀ`) and re-calls this kernel.

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

## Summary

| Kernel | Parity | Gap |
|--------|--------|-----|
| `softmax` | **Full** | base-2 → caller pre-scales by `log2(e)` |
| `l2_normalize` | **Full** | — |
| `layernorm` | **Near-full** | no `ε`; caller folds `sqrt(N)` into `γ` |
| `attention_scores` | **Full per element** | full matrix = host loop; no mask |
| `maxpool` / NMS | **Compute-only** | window gather + suppression test offloaded |
| `sinkhorn_iter` | **Approx + row-only** | `logsumexp→max`; column step host-transposed |
| `argmax_match` | **Approx** | one-hot not index; mutual/threshold offloaded |
| `topk` | **Partial** | threshold+max only; no ranked indices |
| `pixel_shuffle` | **Minimal** | plane copy only; no within-plane interleave |

**Root-cause ISA gaps:** (1) no lane-index→scalar move, (2) no element-wise
vector compare, (3) no gather/scatter, (4) no logarithm (base-2 `exp2` only),
(5) no scalar-add before `rsqrt`. The operations with full parity (softmax, L2,
dot-product, element-wise max) are exactly the dense reductions/activations the
`AGG`/`ACTIVATE`/`MULT` datapath was designed for; everything needing
data-dependent addressing or index materialization is only partially
expressible and pairs with a host step.
