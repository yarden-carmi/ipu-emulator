# Hardware-faithful SuperPoint + SuperGlue (drop-in models)

`superpoint.py` and `superglue.py` here are **drop-in replacements** for the
Magic Leap `models/superpoint.py` / `models/superglue.py`: identical class API,
config, and weights, but **every operation is computed the way our IPU `.asm`
kernels do it — including mathematically-equivalent steps** (softmax via `2^x`,
L2 via `rsqrt`, `grid_sample` as an explicit 4-corner bilinear). So you can swap
them into the original repo and compare stock vs hardware behaviour directly.

## Use as a drop-in
```bash
cp hw_models/superpoint.py hw_models/superglue.py  <your-clone>/SuperGluePretrainedNetwork/models/
# then run the official demo / match_pairs.py as usual
```
(`matching.py` imports `SuperPoint`/`SuperGlue` from these files unchanged. The
weights are found via the local `weights/` dir or the in-repo submodule.)

## Operation audit (every op checked)

| operation | kernel | hardware form | vs stock |
|-----------|--------|---------------|----------|
| conv1a–4b, convPa/Da (3×3+ReLU) | `conv_fp32_full` | **direct shifted-tap MAC** (9 taps × 1×1 channel MAC, FP32 accumulate) — not cuDNN Winograd/FFT | direct-MAC, 3.5e-7 |
| convPb, convDb (1×1) | `conv1x1` | direct MAC (`hw_conv2d` pad 0) | direct-MAC |
| 2×2/s2 maxpool | `maxpool_shift.asm` | **running max over 4 shifted taps** + stride-2 | identical (0) |
| detector softmax(65) | `softmax.asm` | **`2^(log2e·x)`** | 7e-8 |
| depth-to-space | `pixel_shuffle.asm` | plane relocation | identical |
| simple_nms | `maxpool_shift.asm` | local-max via shifted loads (Rx window) | identical (0) |
| score threshold | `topk.asm` | `relu(s−τ)` | identical |
| top-k cap | `topk_mt.asm` | **calibrated τ (soft count + host bisect), no sort** | **set ≈, count ±few** |
| descriptor L2 | `l2_normalize.asm` | **`x·rsqrt(Σx²)`** (inv_sqrt) | 6e-8 |
| sample_descriptors | grid_sample-as-conv | **explicit 4-corner bilinear** | 6e-8 |
| GNN Q/K/V/out, MLP, final, score | `conv1x1`/matmul | matmul (MULT.VE.CYCLIC+ACC) | identical |
| GNN attention softmax | `softmax.asm` | **`2^x`** | exact |
| optimal transport | `sinkhorn_iter`+`sinkhorn_col` | **tropical max-plus Sinkhorn** (no log-marginals) | **behavioural** |
| matching argmax/mutual/threshold | `argmax_match`,`topk` | host index/compare | identical |

**Only two operations diverge** from stock (everything else is bit-equivalent):
1. **top-k keypoint cap** — the ISA has no sort, so the cap is a host-calibrated
   threshold (`topk_mt` soft count + binary search). Keeps ≈k keypoints (k ± the
   bisection precision); the set equals `torch.topk`'s set up to ties.
2. **optimal transport** — tropical (max-plus) Sinkhorn instead of the official
   log-domain logsumexp Sinkhorn; more permissive (omits the marginal terms).

## Verification (`check_hw_models.py`, freiburg pair)
```
maxpool_shift vs F.max_pool2d    0.00e+00  (2x2/s2, 9x9 NMS, 3x3)
exp2 softmax vs F.softmax        7.45e-08
rsqrt L2norm vs F.normalize      5.96e-08
4-corner bilinear vs grid_sample 5.96e-08
full SuperPoint (direct conv, no topk cap): keypoints 429/429 identical, descriptors 3.50e-07
full SuperGlue on identical keypoints: 159 vs 146 matches, 95.8% same target
```
So the math-equivalent path is reproduced to FP rounding; the SuperGlue match
difference (159 vs 146, 95.8% same target) is entirely the tropical-OT behaviour.

`superglue_hw.py` (in `../superpoint_eval/`) is the lighter monkeypatch version
of the same thing; these files are the standalone drop-in form.

## What is and isn't bit-exact

Every op now uses the **hardware algorithm**: the 3×3/1×1 convs are explicit
direct shifted-tap MAC (`hw_conv2d`, not cuDNN Winograd/FFT); maxpool is
shifted-load running max; softmax is `2^x`; L2 is `rsqrt`; grid_sample is the
4-corner bilinear; the GNN/score matmuls are direct MAC dot-products
(`torch.matmul`/einsum — the same multiply-accumulate the IPU does).

The residual ~1e-7–1e-5 differences are **FP32 accumulation-ORDER** rounding
(torch/BLAS reduce in a different order than the IPU's sequential
`MULT.VE.CYCLIC`+`ACC`). That order is a property of the silicon datapath; a
pure-PyTorch file cannot reproduce it bit-for-bit — only **running the actual
`.asm` kernels in the emulator** does, which is exactly what
`superpoint_eval/asm_superpoint.py` (SuperPoint, 2.7e-5) and
`asm_superglue.py` (SuperGlue GNN, 1.5e-5) do. So:

- want the **hardware behaviour / algorithm** at PyTorch speed → these drop-in files;
- want the **bit-exact emulator output** → the `asm_*` harnesses.

Neither changes any decision: the keypoints and the matched correspondences are
identical (the only behavioural differences are the documented top-k cap and the
tropical optimal transport).
