# SuperPoint + SuperGlue on the IPU — complete pipeline reference

End-to-end map of both networks on the IPU emulator: every operation, the kernel
that runs it, the hardware form, the parity vs the stock PyTorch model, and the
validation evidence. Companion docs:

- [`KERNELS.md`](KERNELS.md) — per-kernel reference (math + ISA + python ref + parity)
- [`THEORY.md`](THEORY.md) — extended derivations for the non-conv ops
- [`reference.py`](reference.py) — pure-Python twin of every kernel
- [`hw_models/`](hw_models/README.md) — drop-in hardware-faithful PyTorch models
- [`superpoint_eval/`](superpoint_eval/README.md) — validation harnesses (vs pretrained)

All kernels run in **wide-vector FP32** mode and fit the **128-word IMEM bank**.

---

## Full operation list

### SuperPoint

| # | operation | kernel | hardware form | parity |
|---|-----------|--------|---------------|--------|
| 1 | conv1a–conv4b (8× 3×3) | `conv_fp32_full` | direct shifted-tap MAC, channel-group + width tiling, row-band streaming | direct-MAC ~1e-5 vs real net |
| 2 | ReLU | (fused) | `ACTIVATE relu` | exact |
| 3 | 2×2/s2 maxpool (×3) | `maxpool_shift` | running max over 4 shifted taps + stride-2 gather | identical (0) |
| 4 | convPa (3×3)+ReLU | `conv_fp32_full` | direct shifted-tap MAC | direct-MAC |
| 5 | convPb (1×1→65) | `conv1x1` | direct MAC | direct-MAC |
| 6 | detector softmax(65) | `softmax.asm` | `2^(log₂e·x)/Σ` | math-id 7e-8 |
| 7 | drop dustbin | — | slice | exact |
| 8 | depth-to-space | `pixel_shuffle` | plane relocation | identical |
| 9 | simple_nms (9×9) | `maxpool_shift` | local-max via shifted loads, iterated | identical (0) |
| 10 | score threshold | `topk.asm` | `relu(s−τ)` | exact |
| 11 | border removal | host | data-independent mask | host (movement) |
| 12 | **top-k cap** | `topk_mt.asm` | **calibrated τ: soft count + host bisect, no sort** | **behavioral: ≈k** |
| 13 | (h,w)→(x,y) flip | host | reshape | host (movement) |
| 14 | convDa (3×3)+ReLU | `conv_fp32_full` | direct shifted-tap MAC | direct-MAC |
| 15 | convDb (1×1→256) | `conv1x1` | direct MAC | direct-MAC |
| 16 | dense L2 normalize | `l2_normalize` | `x·rsqrt(Σx²)` | math-id 6e-8 |
| 17 | sample_descriptors | grid_sample-as-conv | explicit 4-corner bilinear | math-id 6e-8 |
| 18 | per-kpt L2 normalize | `l2_normalize` | `x·rsqrt(Σx²)` | math-id |

### SuperGlue

| # | operation | kernel | hardware form | parity |
|---|-----------|--------|---------------|--------|
| 19 | keypoint normalize | host | affine | host (movement) |
| 20 | encoder MLP [3,32,64,128,256] | `conv1x1`/matmul | matmul (BN folded) + ReLU | exact 8.9e-8 |
| 21 | desc += enc | host | add | host (movement) |
| 22 | Q/K/V proj (D→D) ×18×2 | `conv1x1`/matmul | matmul | exact |
| 23 | head reshape (interleaved) | host | view | host (movement) |
| 24 | QKᵀ (per head) | matmul | activation×activation matmul | exact |
| 25 | scale 1/√d | — | `MULT.VE.CR` | exact |
| 26 | attention softmax | `softmax.asm` | `2^x` | math-id exact |
| 27 | attention·V (per head) | matmul | activation×activation matmul | exact |
| 28 | concat heads | host | view | host (movement) |
| 29 | merge proj (D→D) | `conv1x1`/matmul | matmul | exact |
| 30 | residual x+=msg | host | add | host (movement) |
| 31 | merge MLP [2D,2D,D]+ReLU | `conv1x1`/matmul | matmul + ReLU | exact |
| 32 | residual x+=mlp | host | add | host (movement) |
| 33 | final proj (D→D) | `conv1x1`/matmul | matmul | exact |
| 34 | score = Sᵀ·S/√d | matmul | activation×activation matmul | exact |
| 35 | dustbin augment | host | concat bin_score | host (movement) |
| 36 | **Sinkhorn row (×T)** | `sinkhorn_iter[_mt]` | `Z−=max_j Z` (tropical) | **behavioral** |
| 37 | **Sinkhorn col (×T)** | `sinkhorn_col[_mt]` | `Z−=max_i Z` (transpose-free) | **behavioral** |
| 38 | argmax (row/col) | `argmax_match[_mt]` | `2^(T(x−max))` one-hot | argmax-exact 512/512 |
| 39 | integer index + mutual-NN | host | lane-index, integer compare, gather | **host compute** |
| 40 | match threshold | `topk.asm` | `relu(score−τ)` | exact |

---

## Classification

- **Exact** (bit-identical to stock): ReLU, dustbin slice, depth-to-space, maxpool
  (0.00e+00), NMS, thresholds, argmax. (ops 2,3,7–10,13,25,38,40)
- **Direct-MAC / math-identical** (same operation, FP32-accumulation-order rounding
  ~1e-7–1e-5; bit-exact only under the emulator): all convs and matmuls
  (1,4,5,14,15,20,22,24,27,29,31,33,34), softmax→`2^x` (6,26), L2→`rsqrt`
  (16,18), grid_sample→4-corner (17). The chained-asm pipelines confirm the
  composition: SuperPoint 2.7e-5, SuperGlue GNN 1.5e-5.
- **Host data movement** (no compute, no decisions — the cache unit): keypoint
  normalize, residual adds, reshapes/views, dustbin concat, band streaming, IMEM
  bank swaps. (11,13,19,21,23,28,30,32,35)
- **Genuine host compute** (ISA can't express): integer argmax index + mutual-NN
  AND (39); and the top-k τ binary search (12).
- **Two behavioral approximations** (everything else equivalent):
  - **top-k cap** (12): calibrated threshold, no sort → keeps ≈k keypoints.
  - **optimal transport** (36–37): tropical max-plus Sinkhorn vs log-domain
    logsumexp → more permissive matching.

---

## Validation evidence

| what | result | harness |
|------|--------|---------|
| each kernel assembles, ≤128 words | 33/33 OK | — |
| pure-Python references self-check | all pass | `reference.py` |
| SuperPoint convs vs **real pretrained** net | conv1a 5.7e-6 … convPb 9.5e-6 | `superpoint_eval/real_res.py` |
| **SuperPoint full detector path chained in asm** | semi 2.7e-5, **keypoint argmax 100%** | `asm_superpoint.py` |
| **SuperGlue full GNN chained in asm** | 18-layer + proj + score **1.5e-5**, argmax **100%** | `asm_superglue.py` |
| matmul kernel vs numpy | ≤2e-6 | `asm_superglue.py` |
| Sinkhorn `.asm` vs numpy tropical (real coupling) | **0.00e+00** | `compare_sg*.py` |
| SuperGlue OT vs pretrained (real freiburg images) | mutual matches **100% same target** | `compare_sg_real.py` |
| drop-in hw models vs stock (math-equiv ops) | softmax 7e-8, L2 6e-8, bilinear 6e-8, maxpool 0 | `hw_models/check_hw_models.py` |
| drop-in SuperPoint vs stock (no topk cap) | keypoints **429/429 identical**, desc 3.5e-7 | `hw_models/check_hw_models.py` |

---

## Performance model (640×480, N=512, T=100 Sinkhorn)

From `IPU_2_SuperGlue.xlsx` (MobileViT-style, bare-op breakdown + measured-from-
emulator sheet). SuperGlue is ~98% matmul; the matching-stage modifications
(max-Sinkhorn, base-2 softmax, fixed-thresh) are ~2% of cost. Analytical
bare-op total ≈ 270k µs (~3.7 FPS @1 GHz); measured-from-kernels ≈ 419k µs
(~2.4 FPS, rolled matmul at 3 cyc/MAC-step). See the workbook for per-op detail.

---

## "No host" status

The compute is on-device; the host/cache-unit does **data movement** (band
streaming for >2 MB activations, IMEM bank swaps, reshapes/residual adds) and a
few **gather/index ops the ISA lacks**: stride-2 maxpool decimation, the
top-k τ binary search, and the SuperGlue integer argmax + mutual-NN. There is no
single fused kernel (IMEM = 128 words/bank) — both nets are multi-launch
pipelines. SuperPoint and SuperGlue are each validated **chained end-to-end** in
asm against the pretrained models.

---

## File map

```
conv_fp32_full.asm  conv1x1.asm                    final conv kernels
softmax l2_normalize layernorm attention_scores     reductions / attention
maxpool_shift cell_nms superpoint_detect pixel_shuffle   pooling / detection
sinkhorn_iter[_mt] sinkhorn_col[_mt]                 optimal transport
argmax_match[_mt] topk[_mt]                          matching / selection
conv_layers/                                         12 generated per-layer convs
old/                                                 superseded kernels
reference.py THEORY.md KERNELS.md PIPELINE.md        documentation + refs
superpoint_eval/   asm_harness compare real_res      SuperPoint vs pretrained
  asm_superpoint.py                                  SuperPoint chained in asm
  asm_superglue.py                                   SuperGlue chained in asm
  compare_sg.py compare_sg_real.py                   SuperGlue OT vs pretrained
  grid_sample_conv.py  measure_kernels.py            grid_sample / cycle measurement
  superglue_hw.py                                    hw-math monkeypatch + match viz
hw_models/superpoint.py superglue.py                 drop-in hardware-faithful models
third_party/superpoint third_party/superglue         pretrained submodules
```
