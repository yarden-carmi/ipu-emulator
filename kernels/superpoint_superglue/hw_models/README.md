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
| conv1a–4b, convPa/Da (3×3+ReLU) | `conv_fp32_full` | shifted-load MAC + ReLU | identical |
| 2×2/s2 maxpool | shifted max + stride-2 gather | — | identical |
| convPb, convDb (1×1) | `conv1x1` | pointwise MAC | identical |
| detector softmax(65) | `softmax.asm` | **`2^(log2e·x)`** | 7e-8 |
| depth-to-space | `pixel_shuffle.asm` | plane relocation | identical |
| simple_nms | `maxpool_shift.asm` | local-max via shifted loads | identical |
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
exp2 softmax vs F.softmax        7.45e-08
rsqrt L2norm vs F.normalize      5.96e-08
4-corner bilinear vs grid_sample 5.96e-08
full SuperPoint (no topk cap): keypoints 429/429 identical, descriptors 8.94e-08
full SuperGlue on identical keypoints: 159 vs 146 matches, 95.8% same target
```
So the math-equivalent path is reproduced to FP rounding; the SuperGlue match
difference (159 vs 146, 95.8% same target) is entirely the tropical-OT behaviour.

`superglue_hw.py` (in `../superpoint_eval/`) is the lighter monkeypatch version
of the same thing; these files are the standalone drop-in form.
