"""Numpy-parity tests for the stride-1 windowed max-pool app."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from ipu_as.lark_tree import assemble_to_bin_file
from ipu_apps.kernel_registry import MalformedQuery, resolve
from ipu_apps.pooling.maxpool2d_window import (
    SPEC,
    MaxPool2dWindowApp,
    out_tile_cols,
)

ASM_PATH = (
    Path(__file__).resolve().parents[1]
    / "src/ipu_apps/pooling/maxpool2d_window"
    / "maxpool2d_window.asm"
)

# The border value the window sees outside the image. Only ever a maximum's
# identity -- it must never appear in an output.
NEG = -3.4028234663852886e38


def _reference(x: np.ndarray, k: int) -> np.ndarray:
    """Centred stride-1 KxK max with a -FLT_MAX border, as a tap loop.

    Mirrors ``maxpool_shift(kernel=k, stride=1, pad=k//2)`` in
    ``kernels/superpoint_superglue/hw_models/superpoint.py``: a running
    element-wise max over the K*K shifted taps, which is what the kernel does.
    """
    c, h, w = x.shape
    p = k // 2
    xp = np.full((c, h + 2 * p, w + 2 * p), NEG, dtype=np.float64)
    xp[:, p : p + h, p : p + w] = x
    out = np.full((c, h, w), NEG, dtype=np.float64)
    for dy in range(k):
        for dx in range(k):
            out = np.maximum(out, xp[:, dy : dy + h, dx : dx + w])
    return out


def _naive_reference(x: np.ndarray, k: int) -> np.ndarray:
    """The same thing as an explicit loop nest, with no numpy shifting.

    An independent anchor, so ``_reference`` cannot drift into agreeing with the
    kernel for the wrong reason.
    """
    c, h, w = x.shape
    p = k // 2

    def g(ch, y, col):
        return float(x[ch][y][col]) if 0 <= y < h and 0 <= col < w else NEG

    out = np.full((c, h, w), NEG, dtype=np.float64)
    for ch in range(c):
        for y in range(h):
            for col in range(w):
                out[ch][y][col] = max(
                    g(ch, y + dy - p, col + dx - p)
                    for dy in range(k)
                    for dx in range(k)
                )
    return out


@pytest.fixture(scope="module")
def inst_file():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "maxpool2d_window.bin"
        assemble_to_bin_file(ASM_PATH.read_text(), str(path))
        yield path


def _run(inst_file, tmp_path, x: np.ndarray, k: int) -> np.ndarray:
    c, h, w = x.shape
    xp, op = tmp_path / "x.bin", tmp_path / "y.bin"
    x.astype("<f4").tofile(xp)
    app = MaxPool2dWindowApp(
        inst_path=inst_file,
        input_path=xp,
        output_path=op,
        channels=c,
        height=h,
        width=w,
        kernel_size=k,
    )
    app.run(max_cycles=200_000_000)
    return np.frombuffer(op.read_bytes(), dtype="<f4").reshape(c, h, w)


@pytest.mark.parametrize(
    "c,h,w,k",
    [
        (1, 1, 1, 1),      # degenerate: a 1x1 window is the identity
        (1, 1, 1, 3),      # a single pixel whose whole window is border
        (1, 3, 3, 3),      # smallest case exercising all nine taps
        (2, 4, 5, 3),      # rectangular, tiny
        (1, 4, 126, 3),    # exactly one full tile at K=3 (TC = 126)
        (1, 4, 127, 3),    # spills to a second tile by one column
        (2, 3, 200, 3),    # two tiles, halo crosses a tile boundary
        (1, 5, 5, 5),      # a wider window on a tiny image
        (1, 12, 40, 9),    # the SuperPoint simple_nms window (radius 4)
        (2, 6, 130, 9),    # K=9 across a tile boundary (TC = 120)
        (3, 6, 30, 7),     # a bit of everything
    ],
)
def test_matches_numpy(inst_file, tmp_path, c, h, w, k):
    rng = np.random.default_rng(c * 131 + h * 17 + w * 3 + k)
    x = rng.standard_normal((c, h, w), dtype=np.float32)
    out = _run(inst_file, tmp_path, x, k)
    ref = _reference(x, k)
    assert out.shape == ref.shape
    # A maximum selects an input element unchanged, so this is exact.
    assert np.array_equal(out.astype(np.float64), ref)


def test_numpy_reference_matches_the_naive_loop_nest(inst_file, tmp_path):
    """Anchor ``_reference`` to an independent implementation, then the kernel."""
    rng = np.random.default_rng(99)
    x = rng.standard_normal((2, 5, 9), dtype=np.float32)
    ref = _reference(x, 5)
    assert np.array_equal(ref, _naive_reference(x, 5))
    assert np.array_equal(_run(inst_file, tmp_path, x, 5).astype(np.float64), ref)


def test_border_never_wins(inst_file, tmp_path):
    """All-negative input: a zero-filled border would beat every real value.

    This is what makes the ``-FLT_MAX`` fill load-bearing. A centred window at
    the image edge genuinely reads outside the image, so unlike the halving
    kernel this one cannot be correct with any non-identity border.
    """
    rng = np.random.default_rng(7)
    x = -rng.uniform(1.0, 100.0, size=(2, 8, 140)).astype(np.float32)
    assert (x < 0).all()
    out = _run(inst_file, tmp_path, x, 9)
    assert (out < 0).all(), "a border value leaked into the output"
    assert np.array_equal(out.astype(np.float64), _reference(x, 9))


def test_single_impulse_spreads_over_the_window(inst_file, tmp_path):
    """One bright pixel must dominate exactly the KxK block centred on it.

    A dilated, off-centre or transposed window still produces plausible numbers
    on random data; an impulse pins the covered region exactly.
    """
    k, p = 9, 4
    c, h, w = 1, 14, 140
    x = np.zeros((c, h, w), dtype=np.float32)
    x[0, 7, 70] = 1.0
    out = _run(inst_file, tmp_path, x, k)
    hot = out[0] > 0
    expected = np.zeros((h, w), dtype=bool)
    expected[7 - p : 7 + p + 1, 70 - p : 70 + p + 1] = True
    assert np.array_equal(hot, expected)


def test_local_max_property(inst_file, tmp_path):
    """The pooled map must dominate the input everywhere.

    ``simple_nms`` reads this kernel's output only through
    ``scores == max_pool(scores)``, so ``pooled >= scores`` pointwise is the
    property that step depends on.
    """
    rng = np.random.default_rng(3)
    x = rng.standard_normal((2, 10, 90), dtype=np.float32)
    out = _run(inst_file, tmp_path, x, 5)
    assert (out >= x).all()
    # ...and the global maximum is a fixed point of the pool.
    for ch in range(2):
        peak = np.unravel_index(np.argmax(x[ch]), x[ch].shape)
        assert out[ch][peak] == x[ch][peak]


def test_pooling_is_per_channel(inst_file, tmp_path):
    """Channel c must never see channel c+1's data."""
    c, h, w = 5, 4, 150
    x = np.zeros((c, h, w), dtype=np.float32)
    for ch in range(c):
        x[ch] = float(ch + 1)
    out = _run(inst_file, tmp_path, x, 3)
    for ch in range(c):
        assert (out[ch] == float(ch + 1)).all(), ch


def test_output_file_layout_is_dense(inst_file, tmp_path):
    """The output file is exactly C*H*W FP32 in reshape order."""
    c, h, w = 3, 5, 200
    rng = np.random.default_rng(5)
    x = rng.standard_normal((c, h, w), dtype=np.float32)
    xp, op = tmp_path / "x.bin", tmp_path / "y.bin"
    x.astype("<f4").tofile(xp)
    app = MaxPool2dWindowApp(
        inst_path=inst_file, input_path=xp, output_path=op,
        channels=c, height=h, width=w, kernel_size=3,
    )
    app.run(max_cycles=200_000_000)
    assert op.stat().st_size == c * h * w * 4
    out = np.frombuffer(op.read_bytes(), dtype="<f4").reshape(c, h, w)
    assert np.array_equal(out.astype(np.float64), _reference(x, 3))


def test_tile_geometry_is_what_the_kernel_assumes():
    """K-1 halo elements per row, and input and output share one tile grid."""
    assert out_tile_cols(3) == 126
    assert out_tile_cols(9) == 120
    app = MaxPool2dWindowApp(
        inst_path="unused.bin", input_path="unused.bin",
        channels=1, height=4, width=121, kernel_size=9,
    )
    assert app.tile_cols == 120
    assert app.tiles_per_row == 2      # 121 columns spill past the first tile
    assert app.padded_height == 4 + 2 * 4
    assert app.in_plane_stride == 12 * 2
    # The largest element a valid lane reads is (TC-1) + (K-1) = 127.
    assert (app.tile_cols - 1) + (app.kernel_size - 1) == 127


# -- registry conformance ---------------------------------------------------


def _params(**overrides):
    return {
        "shape": (2, 8, 8),
        "kernel_size": 3,
        "stride": 1,
        "padding": 1,
        **overrides,
    }


def test_registry_resolves_to_this_kernel(inst_file, tmp_path):
    # K=9 and K=7 now belong to the unrolled maxpool2d_nms9/nms7, which win on
    # cost; this kernel is the general fallback for every other odd window.
    c, h, w = 2, 8, 40
    verdict = resolve("maxpool2d", shape=(c, h, w), kernel_size=11, stride=1, padding=5)
    assert verdict.supported, verdict.reason
    assert verdict.app_name == "maxpool2d_window"
    assert verdict.shapes["output"] == (c, h, w)

    rng = np.random.default_rng(17)
    x = rng.standard_normal((c, h, w), dtype=np.float32)
    xp, op = tmp_path / "x.bin", tmp_path / "y.bin"
    x.astype("<f4").tofile(xp)
    app = verdict.kernel.app_class(
        inst_path=inst_file, input_path=xp, output_path=op, **verdict.kwargs
    )
    app.run(max_cycles=200_000_000)
    out = np.frombuffer(op.read_bytes(), dtype="<f4").reshape(c, h, w)
    assert np.array_equal(out.astype(np.float64), _reference(x, 11))


def test_the_two_pooling_kernels_do_not_overlap():
    """Disjoint domains, and each refusal names the other kernel.

    A caller who asks the wrong one must be routed, not merely rejected.
    """
    stride2 = resolve("maxpool2d", shape=(2, 8, 8), kernel_size=2, stride=2, padding=0)
    assert stride2.app_name == "maxpool2d_stride2"
    assert stride2.alternatives == (), halve.alternatives

    window = resolve("maxpool2d", shape=(2, 8, 8), kernel_size=3, stride=1, padding=1)
    assert window.app_name == "maxpool2d_window"
    assert window.alternatives == (), window.alternatives

    # K=9 is the one window where a second kernel also claims it, on purpose.
    nms = resolve("maxpool2d", shape=(2, 60, 80), kernel_size=9, stride=1, padding=4)
    assert nms.app_name == "maxpool2d_nms9"
    assert nms.alternatives == ("maxpool2d_window",), nms.alternatives

    stride_2 = SPEC.check(**_params(kernel_size=2, stride=2, padding=0))
    assert not stride_2.ok
    assert "maxpool2d_stride2" in stride_2.reason, stride_2.reason


def test_wide_window_lane_cost_is_disclosed():
    """K=9 costs 8 lanes of every 128; a caller sizing a band needs to know."""
    verdict = resolve("maxpool2d", shape=(1, 8, 300), kernel_size=9, stride=1, padding=4)
    assert verdict.supported, verdict.reason
    assert any("120 usable" in c for c in verdict.caveats), verdict.caveats


@pytest.mark.parametrize(
    "overrides,expected",
    [
        ({"stride": 2}, "stride 2"),
        ({"kernel_size": 4, "padding": 2}, "even"),
        ({"padding": 0}, "padding 0"),
        ({"kernel_size": 129, "padding": 64}, "no usable output columns"),
        ({"shape": (2, 2, 8, 8)}, "batch of 2"),
    ],
)
def test_router_refuses_geometry_no_kernel_implements(overrides, expected):
    verdict = resolve("maxpool2d", **_params(**overrides))
    assert not verdict.supported
    assert expected in verdict.reason, verdict.reason


def test_a_full_resolution_nms_map_fits_in_one_launch():
    """The single-plane 480x640 score map SuperPoint runs simple_nms over.

    Worth pinning as a fact rather than assumed: every convolution in the
    network has to be banded to fit XMEM, so it would be easy to expect the NMS
    pool to need banding too. It does not -- one plane at K=9 is 2928 input rows
    and 2880 output rows against a 16384-row budget -- and the difference is
    that this map has one channel, not 64.
    """
    verdict = resolve(
        "maxpool2d", shape=(1, 480, 640), kernel_size=9, stride=1, padding=4
    )
    assert verdict.supported, verdict.reason
    # The unrolled maxpool2d_nms9 wins K=9 on cost; this kernel still claims it
    # (its supports states the true domain) and its layout is what is checked.
    assert verdict.app_name == "maxpool2d_nms9"
    assert "maxpool2d_window" in verdict.alternatives, verdict.alternatives
    app = MaxPool2dWindowApp(
        inst_path="unused.bin", input_path="unused.bin",
        channels=1, height=480, width=640, kernel_size=9,
    )
    assert app.geometry.total_rows == 64 + 488 * 6 + 1 + 480 * 6


@pytest.mark.parametrize(
    "overrides,ctor",
    [
        # 64 full-resolution planes at K=9: far over the XMEM budget.
        ({"shape": (64, 480, 640), "kernel_size": 9, "padding": 4},
         {"channels": 64, "height": 480, "width": 640, "kernel_size": 9}),
        ({"shape": (0, 8, 8)},
         {"channels": 0, "height": 8, "width": 8, "kernel_size": 3}),
        ({"shape": (2, 0, 8)},
         {"channels": 2, "height": 0, "width": 8, "kernel_size": 3}),
        ({"shape": (2, 8, 0)},
         {"channels": 2, "height": 8, "width": 0, "kernel_size": 3}),
        ({"kernel_size": 4, "padding": 2},
         {"channels": 2, "height": 8, "width": 8, "kernel_size": 4}),
        ({"kernel_size": 129, "padding": 64},
         {"channels": 2, "height": 8, "width": 8, "kernel_size": 129}),
    ],
)
def test_constructor_guard_matches_spec(overrides, ctor):
    """Whatever ``supports`` refuses, the constructor must refuse too."""
    params = _params(**overrides)
    assert not SPEC.check(**params).ok

    with pytest.raises(ValueError):
        MaxPool2dWindowApp(inst_path="unused.bin", input_path="unused.bin", **ctor)


@pytest.mark.parametrize(
    "overrides",
    [
        {"shape": (8, 8)},
        {"kernel_size": 0},
        {"stride": 0},
        {"padding": -1},
        {"kernel_size": (3, 5)},
    ],
)
def test_malformed_queries_propagate(overrides):
    with pytest.raises(MalformedQuery):
        resolve("maxpool2d", **_params(**overrides))
