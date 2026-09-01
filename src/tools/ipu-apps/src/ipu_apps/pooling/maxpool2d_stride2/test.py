"""Kernel-specific check imported by the generic registry test suite."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from ipu_as.lark_tree import assemble_to_bin_file
from ipu_emu.ipu import LANES
from ipu_apps.kernel_registry import resolve


def assert_maxpool2d_stride2_kernel(app_src: Path) -> None:
    """Resolve, assemble, run, and verify a tiled max-pool memory image."""
    shape = (2, 4, 260)
    channels, height, width = shape
    out_height, out_width = height // 2, width // 2
    out_tiles = (out_width + LANES - 1) // LANES
    in_row_stride = 2 * out_tiles + 1

    verdict = resolve(
        "maxpool2d",
        shape=shape,
        kernel_size=2,
        stride=2,
        padding=0,
    )
    assert verdict.supported
    assert verdict.app_name == "maxpool2d_stride2"

    rng = np.random.default_rng(7)
    logical_input = rng.standard_normal(shape, dtype=np.float32)
    tiled_input = np.full(
        (channels, height, in_row_stride * LANES),
        np.finfo(np.float32).min,
        dtype="<f4",
    )
    tiled_input[:, :, :width] = logical_input

    trimmed = logical_input[:, : 2 * out_height, : 2 * out_width]
    expected = trimmed.reshape(
        channels, out_height, 2, out_width, 2
    ).max(axis=(2, 4))

    asm = next(app_src.rglob(verdict.kernel.asm))
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        inst_path = tmp_path / "maxpool2d_stride2.bin"
        input_path = tmp_path / "input.bin"
        output_path = tmp_path / "output.bin"
        assemble_to_bin_file(asm.read_text(), str(inst_path))
        input_path.write_bytes(tiled_input.tobytes())

        app = verdict.kernel.app_class(
            inst_path=inst_path,
            input_path=input_path,
            output_path=output_path,
            **verdict.kwargs,
        )
        _, cycles = app.run(max_cycles=2_000_000)

        raw_output = np.frombuffer(output_path.read_bytes(), dtype="<f4")
        tiled_output = raw_output.reshape(
            channels, out_height, out_tiles * LANES
        )

    assert cycles > 0
    assert np.array_equal(tiled_output[:, :, :out_width], expected)
