"""Descriptor fixtures and row-wise checks for the standard registry runner."""
from functools import partial

import numpy as np

from ipu_apps.kernel_registry.cases import KernelCase
from ipu_apps.kernel_registry.case_support import prepared_image
from .prepare import pack_input
from .reference import reference_rows


def prepare(workspace, *, height=2, width=3, mode="shared", amplitude=1.0,
            start=0, count=0, compare_stock=False, separable=False):
    if not np.isfinite(amplitude) or not 0 < amplitude <= np.finfo(np.float32).max:
        raise ValueError("amplitude must be positive, finite, and representable in FP32")
    values = np.random.default_rng(31).uniform(-amplitude, amplitude, (height, width, 256)).astype('<f4')
    params = dict(shape=values.shape, mode=mode, cell_row_start=start,
                  cell_row_count=count or None)
    image, layout = pack_input(values, mode=mode, separable=separable,
                              cell_row_start=start, cell_row_count=count or None)

    def check(raw):
        actual = raw.reshape(-1, width * 8, 256)
        error = stock_max = absolute_sum = squared_sum = 0.0
        for row, output in enumerate(actual):
            y = start * 8 + row
            expected = reference_rows(values, mode, y, 1)[0]
            np.testing.assert_allclose(output, expected, atol=1e-5 * amplitude, rtol=1e-5)
            error = max(error, float(np.max(np.abs(output - expected))))
            if compare_stock:
                difference = output.astype(np.float64) - reference_rows(values, "stock", y, 1)[0]
                stock_max = max(stock_max, float(np.max(np.abs(difference))))
                absolute_sum += float(np.sum(np.abs(difference)))
                squared_sum += float(np.sum(difference * difference))
        print(f"{mode}: shape={actual.shape}, {actual.nbytes} bytes, seed=31, "
              f"uniform[-{amplitude:g},{amplitude:g}], max_abs_error={error:.9g}")
        if compare_stock:
            print(f"{mode} vs stock reference: max_abs_error={stock_max:.9g}, "
                  f"mean_abs_error={absolute_sum / actual.size:.9g}, "
                  f"rmse={np.sqrt(squared_sum / actual.size):.9g}")

    return prepared_image(workspace, params, image, layout, check)


DEFAULTS = dict(height=2, width=3, mode="shared", amplitude=1.0,
                start=0, count=0, compare_stock=False)


CASES_BY_KERNEL = {
    "sample_descriptors": {
        "default": KernelCase(prepare, DEFAULTS),
        "stock": KernelCase(prepare, dict(DEFAULTS, mode="stock")),
        "single_cell": KernelCase(prepare, dict(DEFAULTS, height=1, width=1)),
        "stock_single_cell": KernelCase(prepare, dict(DEFAULTS, height=1, width=1, mode="stock")),
    },
    "sample_descriptors_separable": {
        name: KernelCase(partial(prepare, separable=True), dict(DEFAULTS, mode="stock", **options))
        for name, options in {
            "default": {},
            "single_cell": dict(height=1, width=1),
            "single_row": dict(height=1, width=5),
            "single_column": dict(height=5, width=1),
            "interior_band": dict(height=4, width=3, start=1, count=2),
            "last_band": dict(height=3, width=2, start=2, count=1),
            "large_values": dict(amplitude=100.0),
        }.items()
    },
}
