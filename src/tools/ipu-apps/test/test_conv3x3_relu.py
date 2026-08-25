"""Numpy-parity tests for the 3x3 FP32 convolution + ReLU app."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from ipu_as.lark_tree import assemble_to_bin_file
from ipu_apps.convolutions_universal.conv.conv3x3_relu import (
    SPEC,
    TILE_COLS,
    Conv3x3ReluApp,
)
from ipu_apps.kernel_registry import MalformedQuery, resolve

ASM_PATH = (
    Path(__file__).resolve().parents[1]
    / "src/ipu_apps/convolutions_universal/conv/conv3x3_relu"
    / "conv3x3_relu.asm"
)

TOL = 1e-4


def _reference(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    """relu(bias + 3x3 zero-padded convolution), as a sum over the nine taps.

    Written as nine shifted whole-plane multiply-accumulates -- the same
    decomposition the kernel uses -- rather than a library convolution, so the
    two agree on tap order and padding by construction.
    """
    cin, h, width = x.shape
    cout = w.shape[0]
    xp = np.zeros((cin, h + 2, width + 2), dtype=np.float64)
    xp[:, 1 : h + 1, 1 : width + 1] = x
    out = np.broadcast_to(b[:, None, None].astype(np.float64), (cout, h, width)).copy()
    for kr in range(3):
        for kc in range(3):
            tap = xp[:, kr : kr + h, kc : kc + width]
            out += np.einsum("oc,chw->ohw", w[:, :, kr, kc].astype(np.float64), tap)
    return np.maximum(out, 0.0)


def _naive_reference(x, w, b):
    """The same thing as an explicit loop nest, with no numpy broadcasting.

    Mirrors ``conv3x3_relu_ref`` in ``kernels/superpoint_superglue/reference.py``,
    the checked-in ground truth these kernels were developed against. Kept as an
    independent anchor so ``_reference`` cannot drift into agreeing with the
    kernel for the wrong reason.
    """
    cin, h, width = x.shape
    cout = w.shape[0]

    def g(ci, y, p):
        return float(x[ci][y][p]) if 0 <= y < h and 0 <= p < width else 0.0

    out = np.zeros((cout, h, width), dtype=np.float64)
    for o in range(cout):
        for y in range(h):
            for p in range(width):
                s = float(b[o])
                for ci in range(cin):
                    for dy in range(3):
                        for dx in range(3):
                            s += float(w[o][ci][dy][dx]) * g(ci, y + dy - 1, p + dx - 1)
                out[o][y][p] = max(0.0, s)
    return out


@pytest.fixture(scope="module")
def inst_file():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "conv3x3_relu.bin"
        assemble_to_bin_file(ASM_PATH.read_text(), str(path))
        yield path


def _write(tmp_path, x, w, b):
    xp, wp, bp, op = (tmp_path / n for n in ("x.bin", "w.bin", "b.bin", "y.bin"))
    x.astype("<f4").tofile(xp)
    w.astype("<f4").tofile(wp)
    b.astype("<f4").tofile(bp)
    return xp, wp, bp, op


def _run(inst_file, tmp_path, x, w, b) -> np.ndarray:
    cin, h, width = x.shape
    cout = w.shape[0]
    xp, wp, bp, op = _write(tmp_path, x, w, b)
    app = Conv3x3ReluApp(
        inst_path=inst_file,
        input_path=xp,
        weight_path=wp,
        bias_path=bp,
        output_path=op,
        in_channels=cin,
        out_channels=cout,
        height=h,
        width=width,
    )
    app.run(max_cycles=200_000_000)
    return np.frombuffer(op.read_bytes(), dtype="<f4").reshape(cout, h, width)


def _random_case(cin, cout, h, w, seed, bias_shift=0.0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((cin, h, w), dtype=np.float32)
    weights = rng.standard_normal((cout, cin, 3, 3), dtype=np.float32) / np.float32(
        np.sqrt(9 * cin)
    )
    # Shift the bias positive by default so a good fraction of outputs survive
    # the ReLU -- an all-zero output would pass almost any comparison.
    bias = rng.standard_normal(cout, dtype=np.float32) + np.float32(bias_shift)
    assert x.dtype == weights.dtype == bias.dtype == np.float32
    return x, weights, bias


@pytest.mark.parametrize(
    "cin,cout,h,w",
    [
        (1, 1, 1, 1),        # degenerate: every tap is padding except the centre
        (1, 1, 3, 3),        # smallest case exercising all nine taps
        (2, 3, 4, 5),        # rectangular, tiny
        (4, 2, 3, 126),      # exactly one full tile
        (4, 2, 3, 127),      # spills to a second tile by one column
        (3, 2, 2, 200),      # two tiles, halo crosses a tile boundary
        (14, 2, 3, 20),      # exactly one full channel group
        (15, 2, 3, 20),      # two groups, last holds 1 channel (the min() path)
        (28, 2, 2, 16),      # two exact groups
        (8, 5, 5, 32),       # a bit of everything
    ],
)
def test_matches_numpy(inst_file, tmp_path, cin, cout, h, w):
    x, weights, bias = _random_case(cin, cout, h, w, seed=cin * 31 + cout)
    out = _run(inst_file, tmp_path, x, weights, bias)
    ref = _reference(x, weights, bias)
    assert out.shape == ref.shape
    assert np.abs(out - ref).max() < TOL


def test_numpy_reference_matches_the_naive_loop_nest(inst_file, tmp_path):
    """Anchor ``_reference`` to an independent implementation, then the kernel."""
    x, weights, bias = _random_case(3, 2, 4, 6, seed=99)
    ref = _reference(x, weights, bias)
    assert np.abs(ref - _naive_reference(x, weights, bias)).max() < TOL
    assert np.abs(_run(inst_file, tmp_path, x, weights, bias) - ref).max() < TOL


def test_relu_actually_clamps(inst_file, tmp_path):
    """A strongly negative bias must drive the whole output to exactly zero.

    Without the fused ReLU this would come back negative, so it pins that
    ACTIVATE.QUANTIZE relu is really in the store path.
    """
    cin, cout, h, w = 3, 2, 4, 10
    x, weights, _ = _random_case(cin, cout, h, w, seed=4)
    bias = np.full(cout, -1000.0, dtype=np.float32)
    out = _run(inst_file, tmp_path, x, weights, bias)
    assert np.all(out == 0.0)


def test_output_is_not_trivially_zero(inst_file, tmp_path):
    """Guard the ReLU tests: a normal case must have a real mix of values."""
    x, weights, bias = _random_case(4, 3, 5, 12, seed=8, bias_shift=0.5)
    out = _run(inst_file, tmp_path, x, weights, bias)
    assert (out > 0).mean() > 0.2, "too few surviving activations to be a real test"
    assert (out == 0).any(), "nothing was clamped, so the ReLU is untested here"


def test_borders_are_zero_padded(inst_file, tmp_path):
    """A single centre-tap weight shifts the image; the border must read zero.

    With W[o, 0] set to a single 1.0 at (kr, kc), the output is the input
    shifted by (-kr+1, -kc+1) with zeros shifted in. That isolates the vertical
    border rows and the horizontal halo from the accumulation logic entirely.
    """
    cin, cout, h, w = 1, 1, 4, 6
    rng = np.random.default_rng(21)
    # Strictly positive input, so the ReLU cannot mask a padding error.
    x = rng.uniform(1.0, 2.0, size=(cin, h, w)).astype(np.float32)
    bias = np.zeros(cout, dtype=np.float32)
    for kr in range(3):
        for kc in range(3):
            weights = np.zeros((cout, cin, 3, 3), dtype=np.float32)
            weights[0, 0, kr, kc] = 1.0
            out = _run(inst_file, tmp_path, x, weights, bias)
            assert np.abs(out - _reference(x, weights, bias)).max() < TOL, (kr, kc)


def test_tap_order_matches_weight_layout(inst_file, tmp_path):
    """Each of the nine weights must land on its own (kr, kc) neighbour.

    A transposed or rotated tap order still produces plausible numbers on
    random weights; giving the nine taps nine distinct magnitudes and comparing
    against the reference pins the mapping exactly.
    """
    cin, cout, h, w = 1, 1, 5, 5
    x = np.zeros((cin, h, w), dtype=np.float32)
    x[0, 2, 2] = 1.0  # a single impulse in the middle
    weights = (np.arange(9, dtype=np.float32) + 1.0).reshape(1, 1, 3, 3)
    bias = np.zeros(cout, dtype=np.float32)
    out = _run(inst_file, tmp_path, x, weights, bias)
    # An impulse convolves to the flipped kernel, which the reference computes
    # the same way -- what matters is that all nine distinct values appear in
    # the right places.
    assert np.abs(out - _reference(x, weights, bias)).max() < TOL
    assert set(np.unique(out)) >= set(range(1, 10)), np.unique(out)


def test_zero_bias_when_no_bias_file(inst_file, tmp_path):
    cin, cout, h, w = 2, 2, 3, 8
    x, weights, _ = _random_case(cin, cout, h, w, seed=3)
    xp, wp, _, op = _write(tmp_path, x, weights, np.zeros(cout, dtype=np.float32))
    app = Conv3x3ReluApp(
        inst_path=inst_file,
        input_path=xp,
        weight_path=wp,
        output_path=op,
        in_channels=cin,
        out_channels=cout,
        height=h,
        width=w,
    )
    app.run(max_cycles=200_000_000)
    out = np.frombuffer(op.read_bytes(), dtype="<f4").reshape(cout, h, w)
    ref = _reference(x, weights, np.zeros(cout, dtype=np.float32))
    assert np.abs(out - ref).max() < TOL


def test_output_file_layout_is_dense(inst_file, tmp_path):
    """The output file is exactly Cout*H*W FP32 in reshape order."""
    cin, cout, h, w = 3, 4, 3, 130   # 130 -> two tiles, partially filled
    x, weights, bias = _random_case(cin, cout, h, w, seed=5)
    xp, wp, bp, op = _write(tmp_path, x, weights, bias)
    app = Conv3x3ReluApp(
        inst_path=inst_file,
        input_path=xp,
        weight_path=wp,
        bias_path=bp,
        output_path=op,
        in_channels=cin,
        out_channels=cout,
        height=h,
        width=w,
    )
    app.run(max_cycles=200_000_000)
    assert op.stat().st_size == cout * h * w * 4
    out = np.frombuffer(op.read_bytes(), dtype="<f4").reshape(cout, h, w)
    assert np.abs(out - _reference(x, weights, bias)).max() < TOL


def test_tile_geometry_is_what_the_kernel_assumes():
    """126 usable columns per 128-element row, one halo element at each end."""
    assert TILE_COLS == 126
    app_layout = SPEC.app_class(
        inst_path="unused.bin",
        input_path="unused.bin",
        weight_path="unused.bin",
        in_channels=1,
        out_channels=1,
        height=4,
        width=127,
    )
    assert app_layout.tiles_per_row == 2
    assert app_layout.padded_height == 6
    assert app_layout.in_plane_stride == 12
    # lr_addr walks (y+2, t) -> (y, t) of the next channel by H*TPR.
    assert app_layout.chan_advance == 4 * 2
    assert app_layout.in_plane_stride - 2 * app_layout.tiles_per_row == (
        app_layout.chan_advance
    )


# -- registry conformance ---------------------------------------------------


def _params(**overrides):
    return {
        "shape": (4, 8, 8),
        "weight_shape": (4, 4, 3, 3),
        "stride": 1,
        "padding": 1,
        "dilation": 1,
        "groups": 1,
        "activation": "relu",
        **overrides,
    }


def test_registry_resolves_to_this_kernel(inst_file, tmp_path):
    cin, cout, h, w = 4, 3, 4, 16
    verdict = resolve(
        "conv2d",
        shape=(cin, h, w),
        weight_shape=(cout, cin, 3, 3),
        stride=1,
        padding=1,
        dilation=1,
        groups=1,
        activation="relu",
    )
    assert verdict.supported, verdict.reason
    assert verdict.app_name == "conv3x3_relu"
    assert verdict.shapes["output"] == (cout, h, w)
    assert any("ReLU is fused" in c for c in verdict.caveats), verdict.caveats

    x, weights, bias = _random_case(cin, cout, h, w, seed=17)
    xp, wp, bp, op = _write(tmp_path, x, weights, bias)
    app = verdict.kernel.app_class(
        inst_path=inst_file,
        input_path=xp,
        weight_path=wp,
        bias_path=bp,
        output_path=op,
        **verdict.kwargs,
    )
    app.run(max_cycles=200_000_000)
    out = np.frombuffer(op.read_bytes(), dtype="<f4").reshape(cout, h, w)
    assert np.abs(out - _reference(x, weights, bias)).max() < TOL


def test_plain_3x3_conv_is_refused_rather_than_given_a_relu_kernel():
    """The whole reason ``activation`` is a query parameter.

    A caller asking for a bare 3x3 convolution must not be handed a kernel that
    also applies ReLU -- that is a confidently wrong answer, not a near miss.
    """
    verdict = resolve("conv2d", **_params(activation="none"))
    assert not verdict.supported
    assert "relu" in verdict.reason, verdict.reason
    assert "immediate" in verdict.reason, verdict.reason


def test_pointwise_kernel_does_not_claim_a_relu_query():
    """The reverse direction: conv1x1 must refuse a fused-ReLU request."""
    verdict = resolve(
        "conv2d",
        shape=(4, 8, 8),
        weight_shape=(4, 4, 1, 1),
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        activation="relu",
    )
    assert not verdict.supported
    assert "no activation" in verdict.reason, verdict.reason


@pytest.mark.parametrize(
    "overrides,expected",
    [
        ({"weight_shape": (4, 4, 5, 5)}, "5x5"),
        ({"stride": 2}, "stride 2"),
        ({"padding": 0}, "padding 0"),
        ({"dilation": 2}, "dilation 2"),
        ({"groups": 2}, "groups=2"),
        ({"shape": (2, 4, 8, 8)}, "batch of 2"),
    ],
)
def test_router_refuses_geometry_no_kernel_implements(overrides, expected):
    verdict = resolve("conv2d", **_params(**overrides))
    assert not verdict.supported
    assert expected in verdict.reason, verdict.reason


@pytest.mark.parametrize(
    "overrides,ctor",
    [
        ({"shape": (256, 512, 512), "weight_shape": (256, 256, 3, 3)},
         {"in_channels": 256, "out_channels": 256, "height": 512, "width": 512}),
        ({"shape": (0, 8, 8), "weight_shape": (4, 0, 3, 3)},
         {"in_channels": 0, "out_channels": 4, "height": 8, "width": 8}),
        ({"shape": (4, 0, 8)},
         {"in_channels": 4, "out_channels": 4, "height": 0, "width": 8}),
        ({"shape": (4, 8, 0)},
         {"in_channels": 4, "out_channels": 4, "height": 8, "width": 0}),
        ({"weight_shape": (0, 4, 3, 3)},
         {"in_channels": 4, "out_channels": 0, "height": 8, "width": 8}),
    ],
)
def test_constructor_guard_matches_spec(overrides, ctor):
    """Whatever ``supports`` refuses, the constructor must refuse too."""
    params = _params(**overrides)
    assert not SPEC.check(**params).ok
    assert not resolve("conv2d", **params).supported

    with pytest.raises(ValueError):
        Conv3x3ReluApp(
            inst_path="unused.bin",
            input_path="unused.bin",
            weight_path="unused.bin",
            **ctor,
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"shape": (8, 8)},
        {"weight_shape": (4, 4, 3)},
        {"stride": 0},
        {"activation": "gelu"},
    ],
)
def test_malformed_queries_propagate(overrides):
    with pytest.raises(MalformedQuery):
        resolve("conv2d", **_params(**overrides))
