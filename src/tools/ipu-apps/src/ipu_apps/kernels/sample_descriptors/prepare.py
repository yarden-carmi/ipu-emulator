"""Geometry-only coefficient generation and HWC memory preparation."""
import numpy as np

from . import App, SeparableApp
from ipu_apps.kernel_registry.memory import positive_shape


def precompute_axis_weights(shape=(60, 80, 256)):
    """Stock-coordinate (Ky[H,3,8], Kx[W,3,8]), before any XY products."""
    h, w, c = positive_shape(shape, 3)
    if c != 256:
        raise ValueError("expected HWC with 256 channels")
    offsets = np.arange(-1, 2, dtype=np.float32)

    def axis(n):
        p = np.arange(8 * n, dtype=np.float32) - np.float32(3.5)
        p /= np.float32(8 * n - 4.5)
        p = p * np.float32(2) - np.float32(1)
        p = (p + np.float32(1)) / np.float32(2) * np.float32(n - 1)
        delta = p.reshape(n, 8)[:, None, :] - np.arange(n, dtype=np.float32)[:, None, None]
        return np.maximum(np.float32(0), np.float32(1) - np.abs(delta - offsets[None, :, None]))

    return axis(h), axis(w)


def precompute_weights(shape=(60, 80, 256), mode="shared"):
    """Return K[dy,dx,a,b] or K[i,j,dy,dx,a,b], with offsets dy/dx - 1.

    Stock mode follows the FP32 grid construction and align_corners=True
    unnormalization in third_party/superglue/models/superpoint.py. Only the
    geometry is evaluated here; descriptors never participate in preparation.
    """
    h, w, c = positive_shape(shape, 3)
    if c != 256 or mode not in ("shared", "stock"):
        raise ValueError("expected HWC with 256 channels and shared/stock mode")
    offsets = np.arange(-1, 2, dtype=np.float32)
    if mode == "shared":
        t = (np.arange(8, dtype=np.float32) - np.float32(3.5)) / np.float32(8)
        k = np.maximum(np.float32(0), np.float32(1) - np.abs(t[None, :] - offsets[:, None]))
        return k[:, None, :, None] * k[None, :, None, :]

    ky, kx = precompute_axis_weights(shape)
    return ky[:, None, :, None, :, None] * kx[None, :, None, :, None, :]


def pack_input(descriptors, *, mode="shared", weights=None,
               cell_row_start=0, cell_row_count=None, separable=False):
    """Return (preformatted FP32 XMEM input rows, MemoryLayout).

    Caller-supplied coefficients are supported in shared mode. Halo packing
    retains the whole coarse map even when a sub-band is requested.
    Set separable=True with mode="stock" for the compact stock kernel.
    """
    descriptors = np.asarray(descriptors, dtype='<f4')
    shape = descriptors.shape
    app = SeparableApp if separable else App
    layout = app.memory_layout(shape=shape, mode=mode,
                              cell_row_start=cell_row_start, cell_row_count=cell_row_count)
    if not np.isfinite(descriptors).all():
        raise ValueError("descriptors must be finite")
    if weights is not None and mode != "shared":
        raise ValueError("caller-provided weights require shared mode")
    h, w, _ = shape
    descriptor_rows = (h + 2) * (w + 2) * 2
    image = np.zeros((layout.input_rows, 128), dtype='<f4')
    image[:descriptor_rows].reshape(h + 2, w + 2, 256)[1:-1, 1:-1] = descriptors
    if separable:
        ky, kx = precompute_axis_weights(shape)
        image[descriptor_rows:descriptor_rows+h, :24] = ky.reshape(h, 24)
        image[descriptor_rows+h:descriptor_rows+h+w, :24] = kx.reshape(w, 24)
    else:
        if weights is None:
            weights = precompute_weights(shape, mode)
        weights = np.asarray(weights, dtype='<f4')
        expected = (3, 3, 8, 8) if mode == "shared" else (*shape[:2], 3, 3, 8, 8)
        if weights.shape != expected or not np.isfinite(weights).all():
            raise ValueError(f"weights must be finite with shape {expected}")
        image[descriptor_rows:, :64] = weights.reshape(-1, 64)
    return image, layout
