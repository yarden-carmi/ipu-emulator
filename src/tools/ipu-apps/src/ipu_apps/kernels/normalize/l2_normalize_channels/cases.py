"""Normalization cases including all-zero columns and partial tiles."""
import numpy as np
from . import App
from ipu_apps.kernel_registry.cases import KernelCase
from ipu_apps.kernel_registry.case_support import random_values, prepared_image, check_close


def prepare(workspace, *, channels, columns):
    params = dict(shape=(channels, columns))
    layout = App.memory_layout(**params)
    tiles = layout.crs[5]
    x = random_values(params['shape'])
    x[:, 0] = 0
    image = np.zeros((layout.input_rows, 128), dtype='<f4')
    image[:-1].reshape(channels, tiles*128)[:, :columns] = x
    norm = np.sqrt(np.sum(x*x, axis=0))
    expected = np.divide(x, norm, out=np.zeros_like(x), where=norm > 0)
    return prepared_image(workspace, params, image, layout,
                          lambda raw: check_close(raw.reshape(channels, tiles*128)[:, :columns], expected))


CASES = {
    'default': KernelCase(prepare, dict(channels=4, columns=16)),
    'tile_boundary': KernelCase(prepare, dict(channels=3, columns=133)),
    'single_channel': KernelCase(prepare, dict(channels=1, columns=129)),
}
