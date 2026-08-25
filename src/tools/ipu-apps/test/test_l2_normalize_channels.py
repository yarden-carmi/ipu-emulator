"""Numpy-parity tests for the dense L2 normalization app."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from ipu_as.lark_tree import assemble_to_bin_file
from ipu_apps.kernel_registry import MalformedQuery, resolve
from ipu_apps.normalize.l2_normalize_channels import SPEC, L2NormalizeChannelsApp

ASM_PATH = (
    Path(__file__).resolve().parents[1]
    / "src/ipu_apps/normalize/l2_normalize_channels"
    / "l2_normalize_channels.asm"
)

TOL = 1e-5


def _reference(x: np.ndarray) -> np.ndarray:
    """x / ||x||_2 down axis 0, with a zero column normalizing to zeros."""
    n2 = (x.astype(np.float64) ** 2).sum(axis=0, keepdims=True)
    inv = np.where(n2 > 0, 1.0 / np.sqrt(np.where(n2 > 0, n2, 1.0)), 0.0)
    return x.astype(np.float64) * inv


def _naive_reference(x: np.ndarray) -> np.ndarray:
    """The same thing column by column, mirroring ``l2_normalize_ref``.

    ``kernels/superpoint_superglue/reference.py``'s pure-Python version, applied
    to each column. Kept as an independent anchor so ``_reference`` cannot drift
    into agreeing with the kernel for the wrong reason.
    """
    import math

    rows, cols = x.shape
    out = np.zeros((rows, cols), dtype=np.float64)
    for n in range(cols):
        col = [float(x[c][n]) for c in range(rows)]
        norm = math.sqrt(sum(v * v for v in col))
        if norm == 0:
            continue
        for c in range(rows):
            out[c][n] = col[c] / norm
    return out


@pytest.fixture(scope="module")
def inst_file():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "l2_normalize_channels.bin"
        assemble_to_bin_file(ASM_PATH.read_text(), str(path))
        yield path


def _run(inst_file, tmp_path, x: np.ndarray) -> np.ndarray:
    rows, cols = x.shape
    xp, op = tmp_path / "x.bin", tmp_path / "y.bin"
    x.astype("<f4").tofile(xp)
    app = L2NormalizeChannelsApp(
        inst_path=inst_file, input_path=xp, output_path=op, rows=rows, cols=cols
    )
    app.run(max_cycles=200_000_000)
    return np.frombuffer(op.read_bytes(), dtype="<f4").reshape(rows, cols)


@pytest.mark.parametrize(
    "rows,cols",
    [
        (1, 1),        # degenerate: a length-1 vector normalizes to +/-1
        (2, 3),        # tiny
        (4, 128),      # exactly one full column tile
        (4, 129),      # spills to a second tile by one column
        (4, 100),      # one partly-filled tile
        (64, 300),     # three tiles
        (256, 64),     # SuperPoint's descriptor depth, few columns
        (256, 200),    # convDb descriptors over two tiles
        (3, 512),      # four exact tiles
    ],
)
def test_matches_numpy(inst_file, tmp_path, rows, cols):
    rng = np.random.default_rng(rows * 131 + cols)
    x = rng.standard_normal((rows, cols), dtype=np.float32)
    out = _run(inst_file, tmp_path, x)
    ref = _reference(x)
    assert out.shape == ref.shape
    assert np.abs(out - ref).max() < TOL


def test_numpy_reference_matches_the_naive_column_loop(inst_file, tmp_path):
    """Anchor ``_reference`` to an independent implementation, then the kernel."""
    rng = np.random.default_rng(99)
    x = rng.standard_normal((6, 20), dtype=np.float32)
    ref = _reference(x)
    assert np.abs(ref - _naive_reference(x)).max() < TOL
    assert np.abs(_run(inst_file, tmp_path, x) - ref).max() < TOL


def test_every_column_has_unit_norm(inst_file, tmp_path):
    """The defining property, checked directly rather than via the reference."""
    rng = np.random.default_rng(4)
    x = rng.standard_normal((32, 300), dtype=np.float32)
    out = _run(inst_file, tmp_path, x)
    norms = np.linalg.norm(out.astype(np.float64), axis=0)
    assert np.abs(norms - 1.0).max() < TOL


def test_zero_column_normalizes_to_zeros(inst_file, tmp_path):
    """rsqrt is guarded at zero, so a zero vector must not become inf or NaN.

    This is the one numerical edge the descriptor head actually hits: a
    convDb output can be identically zero wherever the input image is flat.
    """
    rng = np.random.default_rng(8)
    x = rng.standard_normal((16, 200), dtype=np.float32)
    x[:, 5] = 0.0
    x[:, 199] = 0.0
    out = _run(inst_file, tmp_path, x)
    assert np.isfinite(out).all()
    assert (out[:, 5] == 0.0).all()
    assert (out[:, 199] == 0.0).all()
    # ...and the neighbouring columns are untouched by the guard.
    assert np.abs(out[:, 6] - _reference(x)[:, 6]).max() < TOL


def test_scale_invariance(inst_file, tmp_path):
    """Scaling a column by any positive factor must not change its direction."""
    rng = np.random.default_rng(12)
    x = rng.standard_normal((8, 130), dtype=np.float32)
    scaled = x.copy()
    scaled[:, 3] *= np.float32(1000.0)
    scaled[:, 70] *= np.float32(0.001)
    a, b = _run(inst_file, tmp_path, x), _run(inst_file, tmp_path, scaled)
    assert np.abs(a - b).max() < 1e-4


def test_padding_lanes_do_not_change_real_columns(inst_file, tmp_path):
    """A width that does not fill its tile must behave exactly like one that does.

    The padding lanes are their own all-zero columns, never part of a real
    column's reduction -- this is what says so.
    """
    rng = np.random.default_rng(21)
    x = rng.standard_normal((10, 100), dtype=np.float32)
    out = _run(inst_file, tmp_path, x)
    assert np.abs(out - _reference(x)).max() < TOL


def test_output_file_layout_is_dense(inst_file, tmp_path):
    """The output file is exactly rows*cols FP32 in reshape order."""
    rows, cols = 12, 300
    rng = np.random.default_rng(5)
    x = rng.standard_normal((rows, cols), dtype=np.float32)
    xp, op = tmp_path / "x.bin", tmp_path / "y.bin"
    x.astype("<f4").tofile(xp)
    app = L2NormalizeChannelsApp(
        inst_path=inst_file, input_path=xp, output_path=op, rows=rows, cols=cols
    )
    app.run(max_cycles=200_000_000)
    assert op.stat().st_size == rows * cols * 4
    out = np.frombuffer(op.read_bytes(), dtype="<f4").reshape(rows, cols)
    assert np.abs(out - _reference(x)).max() < TOL


# -- registry conformance ---------------------------------------------------


def test_registry_resolves_a_feature_map(inst_file, tmp_path):
    """A (C, H, W) descriptor tensor with dim=0 flattens to this kernel."""
    c, h, w = 16, 4, 5
    verdict = resolve("l2_normalize", shape=(c, h, w), dim=0)
    assert verdict.supported, verdict.reason
    assert verdict.app_name == "l2_normalize_channels"
    assert verdict.kwargs == {"rows": c, "cols": h * w}
    # The flatten is a reinterpretation, so it must be disclosed.
    assert verdict.shapes.notes, "the flatten was not disclosed"
    assert any("flattened" in n for n in verdict.shapes.notes), verdict.shapes.notes

    rng = np.random.default_rng(17)
    x = rng.standard_normal((c, h * w), dtype=np.float32)
    xp, op = tmp_path / "x.bin", tmp_path / "y.bin"
    x.astype("<f4").tofile(xp)
    app = verdict.kernel.app_class(
        inst_path=inst_file, input_path=xp, output_path=op, **verdict.kwargs
    )
    app.run(max_cycles=200_000_000)
    out = np.frombuffer(op.read_bytes(), dtype="<f4").reshape(c, h * w)
    assert np.abs(out - _reference(x)).max() < TOL


def test_rank4_batch_axis_is_split_off_not_flattened():
    """(1, C, H, W) with dim=1 is the shape a descriptor head actually has.

    ``flatten_to_matrix`` alone refuses dim=1 of a rank-4 shape as an interior
    axis; splitting the batch axis first is what makes the natural torch layout
    routable without transposing anything.
    """
    verdict = resolve("l2_normalize", shape=(1, 256, 4, 5), dim=1)
    assert verdict.supported, verdict.reason
    assert verdict.kwargs == {"rows": 256, "cols": 20}
    assert any("batch of 1" in n for n in verdict.shapes.notes), verdict.shapes.notes


def test_row_wise_normalization_is_refused_not_transposed():
    """dim=1 of a 2-D input reduces along rows -- a different kernel, not this one."""
    verdict = resolve("l2_normalize", shape=(8, 64), dim=1)
    assert not verdict.supported
    assert "along each row" in verdict.reason, verdict.reason


@pytest.mark.parametrize(
    "shape,dim,expected",
    [
        ((4, 8, 8), 2, "along each row"),
        ((3, 4, 5, 6), 1, "batch of 3"),
        ((0, 8), 0, "rows (0)"),
        ((8, 0), 0, "columns (0)"),
    ],
)
def test_router_refuses_what_no_kernel_implements(shape, dim, expected):
    verdict = resolve("l2_normalize", shape=shape, dim=dim)
    assert not verdict.supported
    assert expected in verdict.reason, verdict.reason


@pytest.mark.parametrize(
    "shape,ctor",
    [
        # 8192 channels x 8192 columns: two copies plus the scale row overflow.
        ((8192, 8192), {"rows": 8192, "cols": 8192}),
        ((0, 8), {"rows": 0, "cols": 8}),
        ((8, 0), {"rows": 8, "cols": 0}),
    ],
)
def test_constructor_guard_matches_spec(shape, ctor):
    """Whatever ``supports`` refuses, the constructor must refuse too."""
    assert not SPEC.check(shape=shape, dim=0).ok
    assert not resolve("l2_normalize", shape=shape, dim=0).supported

    with pytest.raises(ValueError):
        L2NormalizeChannelsApp(
            inst_path="unused.bin", input_path="unused.bin", **ctor
        )


def test_over_budget_refusal_points_at_the_column_axis():
    """Banding the reduction axis would compute partial norms, so never suggest it.

    A 256-channel descriptor map at a real resolution is the case that actually
    arises: 256 x 640*480 needs far more than the budget, and the advice has to
    send the caller at the columns rather than at the 256 channels.
    """
    verdict = resolve("l2_normalize", shape=(256, 640 * 480), dim=0)
    assert not verdict.supported
    assert "independent columns into chunks" in verdict.reason, verdict.reason
    assert "cannot be split" in verdict.reason, verdict.reason


def test_a_reduction_axis_too_tall_to_band_says_so():
    """When not even one column tile fits, saying 'tile the columns' would be a lie."""
    verdict = resolve("l2_normalize", shape=(8192, 8192), dim=0)
    assert not verdict.supported
    assert "Even one column tile does not fit" in verdict.reason, verdict.reason


@pytest.mark.parametrize(
    "shape,dim",
    [
        ((8, 8), 2),
        ((8, 8), -3),
        ((2, 4, 8, 8), 0),
        ((4, 8, 8, 8), 2),
    ],
)
def test_malformed_queries_propagate(shape, dim):
    with pytest.raises(MalformedQuery):
        resolve("l2_normalize", shape=shape, dim=dim)
