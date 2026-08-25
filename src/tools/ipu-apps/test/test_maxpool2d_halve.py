"""Numpy-parity tests for the 2x2 stride-2 max-pool app."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from ipu_as.lark_tree import assemble_to_bin_file
from ipu_apps.kernel_registry import MalformedQuery, resolve
from ipu_apps.pooling.maxpool2d_halve import SPEC, MaxPool2dHalveApp

ASM_PATH = (
    Path(__file__).resolve().parents[1]
    / "src/ipu_apps/pooling/maxpool2d_halve"
    / "maxpool2d_halve.asm"
)


def _reference(x: np.ndarray) -> np.ndarray:
    """max over each disjoint 2x2 block, trailing odd row/column dropped.

    Written as a reshape-and-reduce rather than a library pool so it carries no
    padding or ceil-mode convention of its own.
    """
    c, h, w = x.shape
    oh, ow = h // 2, w // 2
    trimmed = x[:, : 2 * oh, : 2 * ow]
    return trimmed.reshape(c, oh, 2, ow, 2).max(axis=(2, 4))


def _naive_reference(x: np.ndarray) -> np.ndarray:
    """The same thing as an explicit loop nest over the four taps.

    Mirrors ``maxpool_shift(kernel=2, stride=2, pad=0)`` in
    ``kernels/superpoint_superglue/hw_models/superpoint.py``, the running
    element-wise max over shifted taps that the hardware actually performs.
    Kept as an independent anchor so ``_reference`` cannot drift into agreeing
    with the kernel for the wrong reason.
    """
    c, h, w = x.shape
    oh, ow = h // 2, w // 2
    out = np.full((c, oh, ow), -np.inf, dtype=np.float64)
    for dy in range(2):
        for dx in range(2):
            tap = x[:, dy : dy + 2 * (oh - 1) + 1 : 2, dx : dx + 2 * (ow - 1) + 1 : 2]
            out = np.maximum(out, tap.astype(np.float64))
    return out


@pytest.fixture(scope="module")
def inst_file():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "maxpool2d_halve.bin"
        assemble_to_bin_file(ASM_PATH.read_text(), str(path))
        yield path


def _run(inst_file, tmp_path, x: np.ndarray) -> np.ndarray:
    c, h, w = x.shape
    xp, op = tmp_path / "x.bin", tmp_path / "y.bin"
    x.astype("<f4").tofile(xp)
    app = MaxPool2dHalveApp(
        inst_path=inst_file,
        input_path=xp,
        output_path=op,
        channels=c,
        height=h,
        width=w,
    )
    app.run(max_cycles=200_000_000)
    return np.frombuffer(op.read_bytes(), dtype="<f4").reshape(c, h // 2, w // 2)


@pytest.mark.parametrize(
    "c,h,w",
    [
        (1, 2, 2),        # degenerate: one output element
        (1, 2, 3),        # odd width: the last column is dropped
        (1, 3, 2),        # odd height: the last row is dropped
        (2, 6, 5),        # both extents odd
        (3, 4, 100),      # one partly-filled output tile
        (4, 4, 256),      # exactly one full output tile (128 output columns)
        (2, 4, 258),      # spills to a second output tile by one column
        (2, 2, 512),      # two exact output tiles
        (8, 8, 260),      # ragged, multi-channel
        (16, 6, 80),      # SuperPoint-ish channel count at a tiny resolution
    ],
)
def test_matches_numpy(inst_file, tmp_path, c, h, w):
    rng = np.random.default_rng(c * 131 + h * 17 + w)
    x = rng.standard_normal((c, h, w), dtype=np.float32)
    out = _run(inst_file, tmp_path, x)
    ref = _reference(x)
    assert out.shape == ref.shape
    # A maximum selects an input element unchanged, so this is exact, not
    # approximate -- a tolerance here would hide a wrong-element bug.
    assert np.array_equal(out, ref)


def test_numpy_reference_matches_the_naive_tap_loop(inst_file, tmp_path):
    """Anchor ``_reference`` to an independent implementation, then the kernel."""
    rng = np.random.default_rng(99)
    x = rng.standard_normal((3, 6, 10), dtype=np.float32)
    ref = _reference(x)
    assert np.array_equal(ref.astype(np.float64), _naive_reference(x))
    assert np.array_equal(_run(inst_file, tmp_path, x), ref)


def test_all_negative_input(inst_file, tmp_path):
    """Every value below zero, so a zero-valued accumulator seed would win.

    This is what pins ``ACC.MAX.FIRST`` on the first tap rather than a running
    ``ACC.MAX``: with a running max, R_ACC would still hold the previous tile's
    result (or its zero-initialised state), and on all-negative data that stale
    value beats every real input.
    """
    rng = np.random.default_rng(7)
    x = -rng.uniform(1.0, 100.0, size=(3, 6, 300)).astype(np.float32)
    assert (x < 0).all()
    assert np.array_equal(_run(inst_file, tmp_path, x), _reference(x))


def test_each_tap_reaches_its_own_neighbour(inst_file, tmp_path):
    """One impulse per 2x2 position, so a swapped or dropped tap shows up.

    Random data would let a kernel that read, say, ``(2y, 2x)`` twice instead of
    ``(2y, 2x+1)`` still look right most of the time. Placing the only positive
    value at each of the four window positions in turn makes each tap
    individually load-bearing.
    """
    for dy in range(2):
        for dx in range(2):
            x = np.full((1, 4, 260), -1.0, dtype=np.float32)
            x[0, dy::2, dx::2] = 5.0
            out = _run(inst_file, tmp_path, x)
            assert np.array_equal(out, _reference(x)), (dy, dx)
            assert (out == 5.0).all(), (dy, dx)


def test_pooling_is_per_channel(inst_file, tmp_path):
    """Channel c must never see channel c+1's data.

    Channel planes are adjacent in XMEM, and the kernel walks the input row
    pointer across a plane boundary once per channel, so a stride that is one
    row wrong silently mixes them.
    """
    c, h, w = 5, 4, 200
    x = np.zeros((c, h, w), dtype=np.float32)
    for ch in range(c):
        x[ch] = float(ch + 1)
    out = _run(inst_file, tmp_path, x)
    for ch in range(c):
        assert (out[ch] == float(ch + 1)).all(), ch


def test_padding_lanes_do_not_leak(inst_file, tmp_path):
    """A huge value in the columns past W must not reach a real output.

    The harness fills those lanes with -FLT_MAX; poisoning them with +inf
    instead would let any tap that strayed past the image win its maximum.
    """
    c, h, w = 2, 4, 100
    rng = np.random.default_rng(11)
    x = rng.standard_normal((c, h, w), dtype=np.float32)
    xp, op = tmp_path / "x.bin", tmp_path / "y.bin"
    x.astype("<f4").tofile(xp)

    app = MaxPool2dHalveApp(
        inst_path=inst_file, input_path=xp, output_path=op,
        channels=c, height=h, width=w,
    )
    original_setup = app.setup

    def poisoned_setup(state):
        original_setup(state)
        planes = np.full(
            (c, h, app.in_row_stride * 128), np.float32(1e30), dtype="<f4"
        )
        planes[:, :, :w] = x
        state.xmem.write_address(app.input_base, planes.tobytes())

    app.setup = poisoned_setup
    app.run(max_cycles=200_000_000)
    out = np.frombuffer(op.read_bytes(), dtype="<f4").reshape(c, h // 2, w // 2)
    assert np.array_equal(out, _reference(x))


def test_output_file_layout_is_dense(inst_file, tmp_path):
    """The output file is exactly C*(H//2)*(W//2) FP32 in reshape order."""
    c, h, w = 3, 6, 300
    rng = np.random.default_rng(5)
    x = rng.standard_normal((c, h, w), dtype=np.float32)
    xp, op = tmp_path / "x.bin", tmp_path / "y.bin"
    x.astype("<f4").tofile(xp)
    app = MaxPool2dHalveApp(
        inst_path=inst_file, input_path=xp, output_path=op,
        channels=c, height=h, width=w,
    )
    app.run(max_cycles=200_000_000)
    assert op.stat().st_size == c * (h // 2) * (w // 2) * 4
    out = np.frombuffer(op.read_bytes(), dtype="<f4").reshape(c, h // 2, w // 2)
    assert np.array_equal(out, _reference(x))


def test_tile_geometry_is_what_the_kernel_assumes():
    """128 output columns per row, two input tiles each, plus one guard tile."""
    app = MaxPool2dHalveApp(
        inst_path="unused.bin", input_path="unused.bin",
        channels=1, height=4, width=300,
    )
    assert (app.out_height, app.out_width) == (2, 150)
    # 150 output columns -> two output tiles -> four input tiles read, +1 guard.
    assert app.out_tiles_per_row == 2
    assert app.in_row_stride == 5
    # ...and that covers the ceil(300 / 128) = 3 tiles holding real columns.
    assert app.geometry.tiles_holding_width == 3
    assert app.in_plane_stride == 4 * 5
    assert app.out_plane_stride == 2 * 2


# -- registry conformance ---------------------------------------------------


def _params(**overrides):
    return {
        "shape": (4, 8, 8),
        "kernel_size": 2,
        "stride": 2,
        "padding": 0,
        **overrides,
    }


def test_registry_resolves_to_this_kernel(inst_file, tmp_path):
    c, h, w = 4, 6, 40
    verdict = resolve("maxpool2d", shape=(c, h, w), kernel_size=2, stride=2, padding=0)
    assert verdict.supported, verdict.reason
    assert verdict.app_name == "maxpool2d_halve"
    assert verdict.shapes["output"] == (c, h // 2, w // 2)

    rng = np.random.default_rng(17)
    x = rng.standard_normal((c, h, w), dtype=np.float32)
    xp, op = tmp_path / "x.bin", tmp_path / "y.bin"
    x.astype("<f4").tofile(xp)
    app = verdict.kernel.app_class(
        inst_path=inst_file, input_path=xp, output_path=op, **verdict.kwargs
    )
    app.run(max_cycles=200_000_000)
    out = np.frombuffer(op.read_bytes(), dtype="<f4").reshape(c, h // 2, w // 2)
    assert np.array_equal(out, _reference(x))


def test_odd_extent_is_disclosed_as_a_caveat():
    """Dropping a row is legal but surprising, so it must be said out loud."""
    verdict = resolve("maxpool2d", shape=(1, 7, 7), kernel_size=2, stride=2, padding=0)
    assert verdict.supported, verdict.reason
    assert any("odd extent" in c for c in verdict.caveats), verdict.caveats


@pytest.mark.parametrize(
    "overrides,expected",
    [
        ({"stride": 1}, "stride 1"),
        ({"kernel_size": 3, "stride": 3}, "3x3"),
        ({"padding": 1}, "padding 1"),
        ({"shape": (2, 4, 8, 8)}, "batch of 2"),
        ({"shape": (1, 1, 1)}, "does not fit"),
    ],
)
def test_router_refuses_geometry_no_kernel_implements(overrides, expected):
    verdict = resolve("maxpool2d", **_params(**overrides))
    assert not verdict.supported
    assert expected in verdict.reason, verdict.reason


@pytest.mark.parametrize(
    "overrides,ctor",
    [
        # Full-resolution SuperPoint conv1b output: far over the XMEM budget.
        ({"shape": (64, 480, 640)},
         {"channels": 64, "height": 480, "width": 640}),
        ({"shape": (0, 8, 8)}, {"channels": 0, "height": 8, "width": 8}),
        ({"shape": (4, 0, 8)}, {"channels": 4, "height": 0, "width": 8}),
        ({"shape": (4, 8, 0)}, {"channels": 4, "height": 8, "width": 0}),
        ({"shape": (4, 1, 8)}, {"channels": 4, "height": 1, "width": 8}),
    ],
)
def test_constructor_guard_matches_spec(overrides, ctor):
    """Whatever ``supports`` refuses, the constructor must refuse too."""
    params = _params(**overrides)
    assert not SPEC.check(**params).ok
    assert not resolve("maxpool2d", **params).supported

    with pytest.raises(ValueError):
        MaxPool2dHalveApp(inst_path="unused.bin", input_path="unused.bin", **ctor)


def test_over_budget_refusal_says_how_to_tile():
    """A refusal a caller cannot act on is only marginally better than a crash."""
    verdict = resolve(
        "maxpool2d", shape=(64, 480, 640), kernel_size=2, stride=2, padding=0
    )
    assert not verdict.supported
    assert "XMEM rows" in verdict.reason, verdict.reason
    assert "row bands of at most" in verdict.reason, verdict.reason


@pytest.mark.parametrize(
    "overrides",
    [
        {"shape": (8, 8)},
        {"kernel_size": 0},
        {"stride": 0},
        {"padding": -1},
        {"kernel_size": (2, 3)},
    ],
)
def test_malformed_queries_propagate(overrides):
    with pytest.raises(MalformedQuery):
        resolve("maxpool2d", **_params(**overrides))
