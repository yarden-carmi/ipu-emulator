# Kernels

A **kernel** is one assembly program plus the Python harness that feeds it: an
`.asm` file, an `IpuApp` subclass that lays out XMEM and reads results back, and
a declaration saying which computations it handles.

This section covers what exists, how the emulator knows what exists, and how to
add more.

## The pages

| Page | What it answers |
|---|---|
| [Softmax](softmax.md) | Which softmax kernels exist, which shape each handles, and what they cost |
| [Convolution](conv2d.md) | The pointwise and 3×3 FP32 kernels, their memory layout, and what the router refuses |
| [Pooling](pooling.md) | The stride-2 and stride-1 max-pools, and how `ACC.STRIDE` makes decimation host-free |
| [Normalization](normalize.md) | L2 normalization down the channel axis |
| [Reshape](reshape.md) | Depth-to-space, and how `ACC.RESHAPE` expresses an interleave |
| [Detector head](detect.md) | Channel peak and score thresholding, and what stays on the host |
| [SuperPoint layer map](superpoint.md) | Which kernel runs each SuperPoint operation, and which have none |
| [Application coverage](../app-coverage.md) | How the emulator knows which kernel implements a computation |
| [Adding applications](../adding-applications.md) | How to contribute a kernel |

## Finding a kernel for a computation

Rather than reading each kernel's docstring, ask the registry. It answers from
the kernels themselves, so it cannot drift out of date:

```bash
python -m ipu_apps.softmax --shape 32,300 --dim 1
python -m ipu_apps.softmax --catalog
```

```python
from ipu_apps.kernel_registry import lookup_layer, resolve

# PyTorch-native: the layer plus the shape it will receive
lookup_layer(nn.Softmax(dim=1), input_shape=(32, 300))

# framework-free
resolve("softmax", shape=(32, 300), dim=1)
```

Both return a verdict carrying the app class, its constructor arguments, why
that kernel was chosen, and any caveats that still apply — or, when nothing
covers the shape, what each candidate objected to.

## Currently registered

| Operation | Kernels |
|---|---|
| `softmax` | 5 — see [Softmax](softmax.md) |
| `conv2d` | 2 — see [Convolution](conv2d.md) |
| `maxpool2d` | 4 — see [Pooling](pooling.md) |
| `l2_normalize` | 1 — see [Normalization](normalize.md) |
| `depth_to_space` | 1 — see [Reshape](reshape.md) |
| `channel_peak` | 1 — see [Detector head](detect.md) |
| `score_threshold` | 1 — see [Detector head](detect.md) |

Together with the fully-connected app these cover SuperPoint's forward network
and the expressible part of its detector head; the
[SuperPoint layer map](superpoint.md) says which kernel runs each operation and
which stay on the host.

The depthwise and fully-connected applications exist in the tree but are not yet
registered; they are used directly rather than through the registry. See
[Building applications](../building-applications.md) for the general
application structure they follow.
