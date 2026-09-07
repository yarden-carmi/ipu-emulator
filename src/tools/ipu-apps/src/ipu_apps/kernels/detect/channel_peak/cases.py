"""Channel maxima and threshold checks, including the single-channel branch."""
import numpy as np
from . import App
from ipu_apps.kernel_registry.cases import KernelCase
from ipu_apps.kernel_registry.case_support import random_values, prepared_image, check_close


def prepare(workspace, *, channels, columns, threshold):
    if not np.isfinite(threshold):
        raise ValueError('threshold must be finite')
    params = dict(shape=(channels, columns))
    layout = App.memory_layout(**params)
    tiles = layout.crs[6]
    x = random_values(params['shape'])
    image = np.zeros((layout.input_rows, 128), dtype='<f4')
    image[:-1].reshape(channels, tiles*128)[:, :columns] = x
    image[-1] = threshold
    confidence = x.max(axis=0)
    expected = np.stack((confidence, np.maximum(confidence - np.float32(threshold), 0)))
    return prepared_image(workspace, params, image, layout,
                          lambda raw: check_close(raw.reshape(2, tiles*128)[:, :columns], expected))


CASES = {
    'default': KernelCase(prepare, dict(channels=4, columns=16, threshold=0.5)),
    'tile_boundary': KernelCase(prepare, dict(channels=3, columns=133, threshold=-0.25)),
    'single_channel': KernelCase(prepare, dict(channels=1, columns=129, threshold=0.0)),
}
