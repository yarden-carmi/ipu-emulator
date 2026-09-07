"""Independent four-corner reference for validation, never used by the kernel."""
import numpy as np


def reference_rows(descriptors, mode, y_start=0, y_count=None):
    h, w, _ = descriptors.shape
    if y_count is None:
        y_count = h * 8 - y_start
    x = np.arange(w * 8, dtype=np.float32)
    y = np.arange(y_start, y_start + y_count, dtype=np.float32)
    if mode == "shared":
        x = (x - 3.5) / 8
        y = (y - 3.5) / 8
    elif mode == "stock":
        # Deliberately construct the actual normalized grid, independently of
        # the phase/tap coefficient representation used by prepare.py.
        grid_x = ((x - 3.5) / np.float32(w * 8 - 4.5)) * 2 - 1
        grid_y = ((y - 3.5) / np.float32(h * 8 - 4.5)) * 2 - 1
        x = ((grid_x + 1) / 2) * (w - 1)
        y = ((grid_y + 1) / 2) * (h - 1)
    else:
        raise ValueError(mode)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = x - x0, y - y0
    result = np.zeros((len(y), len(x), 256), dtype=np.float32)
    for dy, wy in ((0, 1 - fy), (1, fy)):
        for dx, wx in ((0, 1 - fx), (1, fx)):
            ix, iy = x0 + dx, y0 + dy
            valid = ((iy[:, None] >= 0) & (iy[:, None] < h)
                     & (ix[None, :] >= 0) & (ix[None, :] < w))
            weight = (wy[:, None] * wx[None, :] * valid).astype(np.float32)
            values = descriptors[np.clip(iy, 0, h - 1)[:, None], np.clip(ix, 0, w - 1)[None, :]]
            result += values * weight[..., None]
    return result
