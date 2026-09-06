"""Shared fixtures for both exact stride-2 kernel selections."""
import numpy as np
from ipu_apps.kernel_registry.cases import KernelCase, PreparedCase
from ipu_apps.kernel_registry.registry import kernel_spec
from ipu_apps.kernel_registry.case_support import random_values, tile_input, untile_output, check_close


def make_cases(name):
    def prepare(workspace, *, channels, height, width):
        params = dict(shape=(channels, height, width), kernel_size=2, stride=2, padding=0)
        kernel_spec(name).guard(**params)
        x = random_values(params['shape'])
        inp, out = workspace / 'input.bin', workspace / 'output.bin'
        inp.write_bytes(tile_input(x, fill=np.finfo(np.float32).min).tobytes())
        oh, ow = height//2, width//2
        expected = x[:, :2*oh, :2*ow].reshape(channels, oh, 2, ow, 2).max(axis=(2, 4))

        def check():
            raw = np.fromfile(out, dtype='<f4')
            check_close(untile_output(raw, expected.shape), expected)

        return PreparedCase(params, dict(input_path=inp, output_path=out), check)

    tail = name.endswith('_tail')
    return {
        'default': KernelCase(prepare, dict(channels=1, height=4, width=16 if tail else 256)),
        'tile_boundary': KernelCase(prepare, dict(channels=2, height=5, width=260 if tail else 511)),
    }


CASES_BY_KERNEL = {name: make_cases(name) for name in (
    'maxpool2d_stride2', 'maxpool2d_stride2_tail',
)}
