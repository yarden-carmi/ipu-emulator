# SuperPoint + SuperGlue IPU kernels — full reference

Complete documentation for the **final** `.asm` kernels: the operation each
implements, its math, how it maps to the IPU ISA, the pure-Python reference
(`reference.py`), and the measured parity. Deep derivations live in
[`THEORY.md`](THEORY.md); runnable references in [`reference.py`](reference.py);
op-by-op and end-to-end validation in
[`superpoint_eval/`](superpoint_eval/README.md).

All kernels run in **wide-vector FP32 debug mode**
(`IpuState(wide_vector_debug=True, wide_vector_arithmetic=FP32)`): 128 lanes of
real float32, byte-masks disabled. Each kernel ends in `BKPT` and fits the
**128-word IMEM bank**.

## ISA primitive vocabulary (recap)

| group | ops |
|-------|-----|
| load | `LDR_MULT_REG`→R0/R1, `LDR_CYCLIC_MULT_REG`→R_CYCLIC |
| multiply→MULT_RES | `MULT.EE` (R∘RC), `MULT.EE.RR` (square), `MULT.VE.CYCLIC/CR/AAQ` (scalar∘RC) |
| accumulate→R_ACC | `RESET_ACC`, `ACC`, `ACC.FIRST`, `ACC.ADD_AAQ`, `ACC.MAX` |
| reduce R_ACC→AAQ | `AGG mode postfn` — mode∈{sum,max}, postfn∈{value, value_cr(×c), inv (1/x), inv_sqrt (1/√x)} |
| activate R_ACC→POST_AAQ | `ACTIVATE fn` — fn∈{identity, relu, relu6, sigmoid, tanh, gelu, softplus, elu, exp2} |
| store | `STR_ACC_REG`, `STR_POST_AAQ_REG` |
| control | `SET/ADD/SUB/INCR_MOD_POW2`, `BEQ/BNE/BLT/...`, `BKPT` |

Recurring tricks: **base-2 only** (`exp2`; pre-scale logits by `log2(e)` for
natural base — exact); **no vector−scalar** (negate via `AGG value_cr(−1)` then
`ACC.ADD_AAQ`); **activated→reduce** needs an XMEM round-trip (`ACTIVATE` writes
POST_AAQ, `AGG` reduces R_ACC); reductions span ≤128 lanes, larger vectors loop
128-tiles combined by `ACC`/`ACC.MAX` + a final `AGG`.

---

## Index

| # | kernel | op | python ref | parity | validation |
|---|--------|----|-----------|--------|-----------|
| 1 | `conv_fp32_full` | 3×3 conv + bias + ReLU | `conv3x3_relu_ref` | EXACT | 1e-5 vs real SuperPoint |
| 2 | `conv1x1` | 1×1 conv + bias | `conv1x1_ref` | EXACT | 1e-5 vs real convPb/convDb |
| 3 | `softmax` | softmax over ≤128 | `softmax_ref` | FULL (base convention) | exact |
| 3b | `softmax_mt` | softmax over N=ntiles·128 | `softmax_ref` | FULL | exact (≤1.5e-8) |
| 4 | `l2_normalize` | x/‖x‖₂ | `l2_normalize_ref` | FULL | exact |
| 5 | `layernorm` | (x−μ)/√σ²·γ+β | `layernorm_ref` | NEAR-FULL (ε) | exact (ε=0) |
| 6 | `maxpool_shift` | 3×3 sliding max | `maxpool3x3_shift_ref` | FULL, host-free | exact |
| 7 | `cell_nms` | per-cell peak + soft-argmax | `cell_nms_ref` | EXACT formula | ~3e-7 |
| 8 | `superpoint_detect` | detector max + threshold | `cell_nms_ref`/below | argmax-equivalent | 100% argmax |
| 9 | `attention_scores` | q·k/√d | `attention_score_ref` | PER-ELEMENT FULL | exact |
| 10 | `sinkhorn_iter[_mt]` | OT row half-step | `sinkhorn_row_ref` | tropical (max≈lse) | exact op |
| 11 | `sinkhorn_col[_mt]` | OT col half-step | `sinkhorn_col_ref` | tropical, transpose-free | exact op |
| 12 | `argmax_match[_mt]` | argmax + mutual-NN | `argmax_onehot_ref`,`mutual_nn_ref` | argmax-exact | 512/512 |
| 13 | `topk[_mt]` | threshold select | `topk_threshold_ref` | EXACT set | exact |
| 14 | `pixel_shuffle` | depth→space | `depth_to_space_ref` | PARTIAL (plane only) | — |
| 15 | grid_sample | bilinear resample | `bilinear_grid_sample_ref` | EXACT | 2.7e-5 vs torch |

---

## 1. `conv_fp32_full.asm` — 3×3 conv + bias + ReLU

**Layers:** SuperPoint encoder conv1a–4b, detector convPa, descriptor convDa
(all 3×3 + ReLU). The per-layer files `conv_layers/conv1a.asm…` are this body
with per-layer headers.

**Math** (zero-pad = 1):
```
out[o,y,x] = ReLU( b[o] + Σ_ci Σ_{dy,dx∈{-1,0,1}} W[o,ci,dy,dx] · in[ci, y+dy, x+dx] )
```

**Kernel:** one full output channel per launch. Spatial neighbours are
constant-offset **shifted contiguous loads** (no gather); arbitrary FP32 weights
come from `R0` via `MULT.VE.CYCLIC fixed_idx=tap`, accumulated with `ACC` over
the 9 taps × Cin; then `ACTIVATE relu`, `STR`. **Channel-group tiling** (G≤14,
G·9≤128) handles any Cin (accumulate across groups, no reset). **In-kernel
column tiling** sweeps full width; **tight input packing** (row stride
`(W+2)·4`) shrinks the resident band. Over-2 MB activations stream in **row
bands** with a 1-row halo (see superpoint_eval README "Real-resolution").

**Reference:** `reference.py: conv3x3_relu_ref(x, w, b, relu=True)`.
**Parity:** EXACT — matches real pretrained SuperPoint to ~1e-5 (FP32 rounding).

## 2. `conv1x1.asm` — 1×1 (pointwise) conv + bias, no ReLU

**Layers:** detector convPb (256→65), descriptor convDb (256→256). **Also the
SuperGlue Q/K/V/out projections and merge-MLP** — a 1×1 conv is exactly a linear
layer over the H·W "tokens".

**Math:** `out[o,y,x] = b[o] + Σ_ci W[o,ci]·in[ci,y,x]`. One MAC per input
channel (no spatial window, no row-wrap halo); channel-group tiled, bias as an
all-ones channel.

**Reference:** `reference.py: conv1x1_ref(x, w, b)`.
**Parity:** EXACT — ~1e-5 vs real convPb/convDb.

## 3. `softmax.asm` — softmax over N ≤ 128

**Layers:** SuperPoint detector channel softmax; SuperGlue attention row softmax.
**Math:** `softmax(x)_i = 2^{x_i−m} / Σ_j 2^{x_j−m}`, `m=max x` (stable). For
natural base, caller pre-scales by `log2(e)` (then bit-exact).
**Kernel:** load→R_ACC; `AGG max → aaq`; subtract max (`ACC.ADD_AAQ` of −m);
`ACTIVATE exp2 → POST_AAQ`; round-trip to R_ACC; `AGG sum inv → 1/Σ`;
`MULT.VE.AAQ` scale; store.
**Reference:** `softmax_ref(x, base2=False)`. **Parity:** FULL.

## 4. `l2_normalize.asm` — unit-length normalization

**Layer:** SuperPoint/SuperGlue descriptor normalization. **Math:** `y = x/‖x‖₂`.
**Kernel:** `MULT.EE.RR`→squares; `AGG sum inv_sqrt → 1/‖x‖`; `MULT.VE.AAQ`
scale. **Reference:** `l2_normalize_ref(x)`. **Parity:** FULL (‖x‖=0→zeros).

## 5. `layernorm.asm` — y=(x−μ)/√(σ²+ε)·γ+β

**Math:** μ=mean, σ²=var. **Kernel:** μ via `AGG sum value_cr(1/N)`; subtract;
square + `AGG sum value_cr(1/N)`=σ²; `inv_sqrt`; scale; affine γ,β. Folds √N into
a caller-supplied γ′; omits ε. **Reference:** `layernorm_ref(x, γ, β, eps=0)`.
**Parity:** NEAR-FULL (exact at ε=0).

## 6. `maxpool_shift.asm` — 3×3 sliding-window max

**Math:** `out[y,x] = max_{dy,dx∈{-1,0,1}} pad(heat)[y+dy,x+dx]`, −∞ border.
**Kernel:** GATHER-FREE — the 9 taps are constant flat offsets `dy·W+dx`, each a
shifted contiguous load combined with `ACC.MAX`. Single launch, host-free.
**Reference:** `maxpool3x3_shift_ref(heat)`. **Parity:** FULL.
(Stride-2 decimation, when needed, is a host gather — left to the cache unit.)

## 7. `cell_nms.asm` — per-cell peak + soft-argmax (depth-to-space-free)

**Layer:** SuperPoint detection without the pixel-shuffle reshape. Over a cell's
64 sub-grid channels: `peak = max_c p_c`; `(a,b) = Σ_c softmax(T·p)_c·(c//8, c%8)`
→ as T→∞ the hard argmax sub-position. **Reference:** `cell_nms_ref(p, T=64)`.
**Parity:** EXACT formula (~3e-7); equals SuperPoint NMS as T sharpens.

## 8. `superpoint_detect.asm` — detector decision over all cells

**Layer:** detector head readout. Cells in lanes; confidence = `max` over the 64
channel planes (dustbin excluded), fixed-threshold gate. Equivalent to
SuperPoint's softmax-argmax because `argmax(softmax)=argmax(logits)`.
**Reference:** `cell_nms_ref` (peak) + `topk_threshold_ref` (gate). **Parity:**
detector argmax **100%** identical to SuperPoint; logits match to 2.7e-5.

## 9. `attention_scores.asm` — scaled dot product

**Layer:** SuperGlue attention QKᵀ. **Math:** `s = (q·k)/√d` per (query,key).
**Kernel:** `MULT.EE q∘k → ACC.FIRST`; `AGG sum value_cr(1/√d) → score`.
**Reference:** `attention_score_ref(q,k,d)`. **Parity:** PER-ELEMENT FULL; the
full M×N matrix is a host loop over keys (or the conv1x1 matmul path).

## 10–11. `sinkhorn_iter[_mt]` (row) + `sinkhorn_col[_mt]` (col) — optimal transport

**Layer:** SuperGlue matching, the doubly-stochastic normalization of the
dustbin-augmented score matrix Z.
**Theory (official):** log-domain `Z_ij ← Z_ij − logsumexp_j(Z)` (row) and
`− logsumexp_i(Z)` (col), T iterations.
**Kernel (ours):** **tropical surrogate** `logsumexp → max`:
```
row:  Z_ij ← Z_ij − max_j Z_ij        (sinkhorn_iter:  AGG max value_cr(-1); ACC.ADD_AAQ)
col:  Z_ij ← Z_ij − max_i Z_ij        (sinkhorn_col:   colmax = ACC.MAX across rows; subtract)
```
The column max is the **element-wise max of the row vectors** — built with
`ACC.MAX` over contiguous rows, so **no transpose / no column gather**. `_mt`
variants tile columns >128. Full iteration runs on-device; host only sequences
the two half-steps.
**Reference:** `sinkhorn_row_ref`, `sinkhorn_col_ref`, `sinkhorn_iteration_ref`.
**Parity:** the `.asm` reproduce the tropical iteration **exactly** (0.00e+00 on
the real 121×121 coupling); vs official log-Sinkhorn the **mutual matches are
100% identical on real images** (max-OT is more permissive — it omits the
log-marginal terms). See `superpoint_eval/compare_sg*.py`.

## 12. `argmax_match[_mt].asm` — argmax + mutual nearest-neighbour

**Layer:** SuperGlue assignment readout. `e_i = 2^{T(x_i−max)}` → one-hot as
T→∞ (`AGG max`, subtract, `ACTIVATE exp2`). Integer index + mutual AND (run on
rows of P and Pᵀ) are host. `_mt` tiles >128. **Reference:** `argmax_onehot_ref`,
`mutual_nn_ref`. **Parity:** `argmax(e)=argmax(x)` exactly once T sharp (512/512).

## 13. `topk[_mt].asm` — threshold selection

**Math:** `sel_i = ReLU(x_i − τ)` (exact survivor set {x>τ}); `topk_mt` also emits
a soft count `Σ sigmoid(T(x−τ))` to drive a host binary-search on τ for a target
k. **Reference:** `topk_threshold_ref`, `topk_count_ref`, `topk_calibrate_ref`.
**Parity:** EXACT selection; count is a soft approximation for calibration.

## 14. `pixel_shuffle.asm` — depth-to-space

**Math:** `y[c', h·r+a, w·r+b] = x_cell[h,w][c'·r² + a·r + b]`. Kernel does the
plane-granular relocation; the within-plane interleave is host. **Reference:**
`depth_to_space_ref`. **Parity:** PARTIAL — in the detector this is replaced by
`cell_nms` (which avoids depth-to-space entirely).

## 15. grid_sample — bilinear descriptor sampling as convolution

**Layer:** SuperPoint descriptor sampling at keypoints (fixed resolution). At a
fixed size the 4 neighbour coarse-indices and bilinear weights are
data-independent:
```
out[c,Y,X] = Σ_{dy,dx∈{0,1}} Wd[dy,dx][Y,X] · C[c, i0[Y]+dy, j0[X]+dx]
```
**Two kernels (`grid_sample_conv.py`):** (a) **4-corner accumulate** —
`MULT.EE` of precomputed corner value-fields × weight-fields + `ACC` (cheapest;
host pre-gathers the corners); (b) **gather-free separable matmul**
`out = M_H · C · M_Wᵀ` by precomputed 1-D interpolation matrices — fully
on-device, no gather/scatter (~18× more cycles, the cost of removing the host
gather). **Reference:** `bilinear_grid_sample_ref(C,H,W)`. **Parity:** EXACT vs
`torch.nn.functional.grid_sample` (2.7e-5).

---

## Running the references

```bash
source /tmp/ipuenv/bin/activate
python kernels/superpoint_superglue/reference.py          # pure-python self-checks
python kernels/superpoint_superglue/superpoint_eval/compare.py        # SuperPoint op-by-op vs real net
python kernels/superpoint_superglue/superpoint_eval/compare_sg_real.py # SuperGlue vs pretrained (real images)
python kernels/superpoint_superglue/superpoint_eval/grid_sample_conv.py
```
