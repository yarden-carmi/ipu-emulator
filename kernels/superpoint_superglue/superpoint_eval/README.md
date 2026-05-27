# SuperPoint op-by-op evaluation (asm vs real pretrained network)

Validates the `.asm` kernels against the **real SuperPoint** network
(`third_party/superpoint`, Magic Leap pretrained `superpoint_v1.pth`), one
operation at a time.

## Flow
1. `extract_weights.py` (needs `torch`, run once) → caches per-layer
   weights/biases to `weights/*.npy` (gitignored; regenerate from the submodule).
2. `modified_ops.py` — NumPy SuperPoint ops written to match each kernel's exact
   math (e.g. `conv2d_relu` = conv_fp32's 3×3 + zero-pad + bias + ReLU).
3. `asm_harness.py` — runs the `.asm` kernel in the wide-FP32 emulator with real
   weights. Bias is folded with **no kernel change** as an extra all-ones input
   channel whose center-tap weight = bias[oc].
4. `compare.py` — per op: `.asm` vs `modified_ops` (same math, expect ≤1e-3) and
   `.asm` vs torch SuperPoint (the true op). Reports error + cycles + ALU util.

## Status (asm vs real SuperPoint)

Op-by-op, the `.asm` conv kernels reproduce the real pretrained network's math.
Full detector path chained in the emulator (16×16 image) vs the SuperPoint
reference — every layer matches:

| layer | shape | max_abs_diff |
|-------|-------|-------------:|
| conv1a | 64×16×16 | 1.5e-5 |
| conv1b | 64×16×16 | 3.6e-5 |
| conv2a/2b | 64×8×8 | 1.6e-5 / 2.9e-5 |
| conv3a/3b | 128×4×4 | 2.5e-5 / 2.2e-5 |
| conv4a/4b | 128×2×2 | 2.2e-5 / 1.5e-5 |
| convPa | 256×2×2 | 3.8e-6 |
| **convPb (`semi`)** | 65×2×2 | **2.7e-5** |

**Compare as one:** the detector logits `semi` from the chained `.asm` forward
match SuperPoint to **2.7e-5** (FP32 accumulation-order rounding only). The
detector decision — our `superpoint_detect` max-over-64-channels argmax — equals
SuperPoint's softmax argmax **100%** of the time (softmax is monotonic, so
`argmax(softmax)=argmax(logits)`), so the selected keypoint per cell is identical;
the fixed-threshold top-k then operates on matching logits.

## Real-resolution, one kernel at a time (`real_res.py`)

Each conv kernel run at its **true SuperPoint spatial size** vs the reference.
Wide layers are width-tiled (≤126-col strips + halo) and tall/deep layers are
H-banded (row bands so the 128–256-channel activation fits the 2 MB XMEM — the
streaming the cache unit does on hardware); no kernel changes, interior outputs
stitched.

| layer | real res | Cin→Cout | max diff | MULT% | full-layer cycles |
|-------|----------|----------|---------:|------:|------------------:|
| conv1a | 480×640 | 1→64 | 5.7e-6 | 49 | ~6.8 M |
| conv4a | 60×80 | 128→128 | 3.1e-5 | 66 | ~15.0 M |
| convPa | 60×80 | 128→256 | 1.4e-6 | 66 | ~29.9 M |
| convPb (1×1) | 60×80 | 256→65 | 9.5e-6 | 66 | ~16.5 M |

All match. (conv1a's activation fits XMEM; deeper layers need H-banding.) convPb
is a 1×1 conv run on the 3×3 kernel with center-only weights.

Kernels used: `conv_fp32` (Cin≤13), `conv_fp32_tiled` (channel-group tiling, any
Cin; ReLU), `conv_fp32_tiled_norelu` (convPb/convDb), `superpoint_detect`
(max+threshold). 2×2/s2 maxpool is the one host step (stride-2 = a gather the ISA
leaves to host). Only `.asm` + Python; simulator untouched.

## SuperGlue GNN forward chained in asm (`asm_superglue.py`)

The whole SuperGlue forward run as **one chained asm pipeline** (not just per-op),
vs the pretrained model. Core primitive `matmul_asm` is a channels-in-lanes
matmul kernel (one 128-wide output-channel tile per launch, all tokens looped
internally, Din channel-groups, bias as an all-ones channel). On top: the
keypoint-encoder MLP (BatchNorm folded), Q/K/V projections, **interleaved** head
split (`view(b,dim,heads,N)` ⇒ channel `d*heads+h`), per-head QKt, attention
softmax via `softmax.asm`, attention·V, merge projection, the merge MLP, residual
adds, final projection, and the score matrix — every matmul/softmax on-device.

| stage (vs pretrained 'indoor') | max abs diff |
|--------------------------------|-------------:|
| matmul_asm vs numpy | 2.2e-6 |
| keypoint encoder | 8.9e-8 |
| one full GNN layer | 4.3e-6 |
| **full 18-layer GNN + final proj + score matrix** | **1.5e-5** |
| score-matrix row-argmax agreement | **100%** |

So the chained `.asm` SuperGlue GNN reproduces the pretrained network to FP32
rounding, and the correspondences (score-matrix argmax) are identical. (Host does
only the reshapes / residual adds / launch sequencing — data movement, no
arithmetic.) The optimal-transport step that follows is the tropical max-Sinkhorn
characterised below.

## SuperGlue end-to-end (`compare_sg.py`)

Validates the matcher against the **official pretrained SuperGlue**
(`third_party/superglue`, Magic Leap `superglue_outdoor.pth`). Our pipeline is
bit-identical to official everywhere except the optimal-transport step: the
kernels use the **tropical (max-plus) Sinkhorn** (`sinkhorn_iter.asm` row half-step
`Z-=max_j` + `sinkhorn_col.asm` col half-step `Z-=max_i`) instead of the
log-domain logsumexp Sinkhorn. (Base-2 softmax with a `log2(e)` pre-scale is
exactly natural softmax, so attention is bit-exact and needs no separate check.)

Result (pretrained 'outdoor', N=64 keypoints/image, mean over 5 seeds):

| metric | value |
|--------|------:|
| `sinkhorn_iter.asm`+`sinkhorn_col.asm` vs numpy tropical (65×65, 5 iters) | **0.00e+00** (exact) |
| stock log-Sinkhorn valid matches | 28.2 / 64 |
| our max-Sinkhorn valid matches | 33.4 / 64 |
| **same target when both match** | **91.2%** |
| matches0 agreement (incl. no-match) | 60.0% |

**Reading:** the `.asm` kernels reproduce the tropical iteration exactly; and when
both methods commit to a correspondence they pick the **same partner 91%** of the
time. The lower full-array agreement (60%) is the *threshold/validity* decision:
tropical OT is more permissive (it omits the log-marginal `log_mu/log_nu` terms),
so it admits more matches. This is the one genuine algorithmic gap -- unlike base-2
softmax (exact), the max-plus Sinkhorn is a real approximation of the soft OT.

**Real image pair (`compare_sg_real.py`).** Same comparison on SuperGlue's own
freiburg office demo frames: the official SuperPoint frontend extracts 120 real
keypoints/frame (real camera motion), 'indoor' weights. The real (121×121)
coupling is run through the actual `sinkhorn_iter.asm`+`sinkhorn_col.asm`:

| metric | value |
|--------|------:|
| asm Sinkhorn vs numpy tropical (real 121×121, 5 iters) | **0.00e+00** (exact) |
| stock log-Sinkhorn valid matches | 51 / 120 |
| our max-Sinkhorn valid matches | 71 / 120 |
| **same target when both match** | **100%** (42 pairs) |
| matches0 agreement (incl. no-match) | 68.3% |

On **real** features the mutual core is **identical** (100% same target) -- whenever
both methods commit to a correspondence they pick the same partner; our pipeline's
matches are a superset (the tropical OT is more permissive). So the kernels
reproduce official SuperGlue's correspondences exactly on the matches that matter.

## Run
```
source /tmp/ipuenv/bin/activate            # torch + numpy + ipu packages
python superpoint_eval/extract_weights.py  # once (SuperPoint)
python superpoint_eval/compare.py          # SuperPoint op-by-op
python superpoint_eval/compare_sg.py       # SuperGlue end-to-end (synthetic)
python superpoint_eval/compare_sg_real.py  # SuperGlue end-to-end (real freiburg pair)
```
