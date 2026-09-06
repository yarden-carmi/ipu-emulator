"""Cycle benchmark for the 2x2, stride-2 max-pool kernels.

Run from the repository root with::

    bazel run //src/tools/ipu-apps:benchmark_maxpool2d_stride2
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import numpy as np

from ipu_as.lark_tree import assemble_to_bin_file
from ipu_emu.debug_cli import debug_prompt
from ipu_emu.ipu import LANES
from ipu_apps.kernel_registry import resolve


KERNEL_DIR = Path(__file__).resolve().parent
CONFIGS = [
    (1, 2, 256),
    (1, 2, 260),
    (64, 2, 256),
    (64, 2, 260),
    (1, 480, 640),
    (64, 480, 640),
]


def _expected_output(values: np.ndarray) -> np.ndarray:
    channels, height, width = values.shape
    trimmed = values[:, : height // 2 * 2, : width // 2 * 2]
    return trimmed.reshape(channels, height // 2, 2, width // 2, 2).max(
        axis=(2, 4)
    )


def _run_shape(
    shape: tuple[int, int, int], tmp: Path, assembled: dict[str, Path]
) -> tuple[str, int, int, float]:
    channels, height, width = shape
    input_rows_per_matrix_row = (width + LANES - 1) // LANES
    output_width = width // 2
    output_rows_per_matrix_row = (output_width + LANES - 1) // LANES

    verdict = resolve(
        "maxpool2d",
        shape=shape,
        kernel_size=2,
        stride=2,
        padding=0,
    )
    if not verdict.supported:
        raise ValueError(verdict.reason)

    if verdict.kernel.asm not in assembled:
        binary = tmp / f"{verdict.app_name}.bin"
        assembly = KERNEL_DIR / verdict.kernel.asm
        assemble_to_bin_file(assembly.read_text(), str(binary))
        assembled[verdict.kernel.asm] = binary

    rng = np.random.default_rng(7)
    logical_input = rng.standard_normal(shape, dtype=np.float32)
    tiled_input = np.full(
        (channels, height, input_rows_per_matrix_row * LANES),
        np.finfo(np.float32).min,
        dtype="<f4",
    )
    tiled_input[:, :, :width] = logical_input

    input_path = tmp / f"input-{channels}-{height}-{width}.bin"
    output_path = tmp / f"output-{channels}-{height}-{width}.bin"
    input_path.write_bytes(tiled_input.tobytes())

    app = verdict.kernel.app_class(
        inst_path=assembled[verdict.kernel.asm],
        input_path=input_path,
        output_path=output_path,
        **verdict.kwargs,
    )
    started = time.perf_counter()
    _state, cycles = app.run(
        max_cycles=20_000_000,
        debug_callback=debug_prompt,
    )
    duration_seconds = time.perf_counter() - started

    output = np.frombuffer(output_path.read_bytes(), dtype="<f4").reshape(
        channels, height // 2, output_rows_per_matrix_row * LANES
    )
    expected = _expected_output(logical_input)
    if not np.array_equal(output[:, :, :output_width], expected):
        raise AssertionError(f"incorrect output for shape {shape}")

    output_elements = channels * (height // 2) * output_width
    return verdict.app_name, cycles, output_elements, duration_seconds


def main() -> None:
    header = (
        f"{'shape':>16}  {'kernel':<26}  {'cycles':>10}  "
        f"{'output elements':>15}  {'cycles/element':>14}  {'duration (ms)':>13}"
    )
    print(header)
    print("-" * len(header))

    with tempfile.TemporaryDirectory() as directory:
        tmp = Path(directory)
        assembled: dict[str, Path] = {}
        for shape in CONFIGS:
            kernel, cycles, output_elements, duration_seconds = _run_shape(
                shape, tmp, assembled
            )
            print(
                f"{str(shape):>16}  {kernel:<26}  {cycles:>10}  "
                f"{output_elements:>15}  {cycles / output_elements:>14.4f}  "
                f"{duration_seconds * 1000:>13.3f}"
            )


if __name__ == "__main__":
    main()
