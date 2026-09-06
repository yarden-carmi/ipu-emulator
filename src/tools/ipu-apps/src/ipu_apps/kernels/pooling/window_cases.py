"""Cases shared by runtime-window and unrolled max-pool kernels."""
import numpy as np
from ipu_apps.kernel_registry.cases import KernelCase
from ipu_apps.kernel_registry.case_support import (
    random_values, tile_input, untile_output, prepared_image, check_close,
)


def make_cases(app, default_k):
    def prepare(workspace, *, channels, height, width, kernel_size):
        params = dict(shape=(channels, height, width), kernel_size=kernel_size,
                      stride=1, padding=kernel_size // 2)
        layout = app.memory_layout(**params)
        x = random_values(params['shape']) - 2
        floor = np.finfo(np.float32).min
        packed = tile_input(x, halo=kernel_size // 2, fill=floor)
        image = np.full((layout.input_rows, 128), floor, dtype='<f4')
        image[:len(packed)] = packed
        padded = np.pad(x, ((0, 0), (kernel_size//2,)*2, (kernel_size//2,)*2),
                        constant_values=floor)
        windows = np.lib.stride_tricks.sliding_window_view(padded, (kernel_size, kernel_size), axis=(1, 2))
        expected = windows.max(axis=(-1, -2))
        return prepared_image(workspace, params, image, layout,
                              lambda raw: check_close(untile_output(raw, x.shape, columns=129-kernel_size), expected))

    def case(c, h, w, k=default_k):
        return KernelCase(prepare, dict(channels=c, height=h, width=w, kernel_size=k))

    cases = {'default': case(1, 4, 16), 'tile_boundary': case(2, 3, 133),
             'single_pixel': case(1, 1, 1)}
    if app.fixed_kernel is None:
        cases.update(identity_window=case(2, 2, 129, 1), large_window=case(1, 1, 3, 127),
                     nms7=case(1, 3, 125, 7), nms9=case(1, 3, 123, 9))
    return cases
