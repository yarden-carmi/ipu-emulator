"""Numpy-parity tests for the pointwise FP32 convolution app."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from ipu_as.lark_tree import assemble_to_bin_file
from ipu_apps.convolutions_universal.conv.conv1x1_fp32 import SPEC, Conv1x1Fp32App
from ipu_apps.kernel_registry import MalformedQuery, resolve

ASM_PATH = (
    Path(__file__).resolve().parents[1]
    / "src/ipu_apps/convolutions_universal/conv/conv1x1_fp32/conv1x1_fp32.asm"
)

TOL = 1e-4


def _reference(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    """out[o, y, p] = b[o] + SUM_ci w[o, ci] * x[ci, y, p]."""
    return np.einsum("oc,chw->ohw", w, x) + b[:, None, None]


def _naive_reference(x, w, b):
    """The same thing written as the loop nest, with no numpy machinery.

    Mirrors ``conv1x1_ref`` in ``kernels/superpoint_superglue/reference.py``,
    which is the checked-in ground truth these kernels were developed against.
    Kept as an independent anchor so ``_reference`` cannot drift into agreeing
    with the kernel for the wrong reason.
    """
    cin, h, w_ = x.shape
    cout = w.shape[0]
    out = np.zeros((cout, h, w_), dtype=np.float64)
    for o in range(cout):
        for y in range(h):
            for p in range(w_):
                out[o][y][p] = float(b[o]) + sum(
                    float(w[o][ci]) * float(x[ci][y][p]) for ci in range(cin)
                )
    return out


@pytest.fixture(scope="module")
def inst_file():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "conv1x1_fp32.bin"
        assemble_to_bin_file(ASM_PATH.read_text(), str(path))
        yield path


def _run(inst_file, tmp_path, x, w, b) -> np.ndarray:
    """Run the app on the given arrays and return the ``(Cout, H, W)`` output."""
    cin, h, width = x.shape
    cout = w.shape[0]
    xp = tmp_path / "x.bin"
    wp = tmp_path / "w.bin"
    bp = tmp_path / "b.bin"
    op = tmp_path / "y.bin"
    x.astype("<f4").tofile(xp)
    w.astype("<f4").tofile(wp)
    b.astype("<f4").tofile(bp)

    app = Conv1x1Fp32App(
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
    app.run(max_cycles=20_000_000)
    return np.frombuffer(op.read_bytes(), dtype="<f4").reshape(cout, h, width)


def _random_case(cin, cout, h, w, seed):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((cin, h, w), dtype=np.float32)
    # Scale in float32: dividing by a np.float64 scalar would silently promote
    # the array, and the files these arrays are written to are read back as f4.
    weights = rng.standard_normal((cout, cin), dtype=np.float32) / np.float32(
        np.sqrt(cin)
    )
    bias = rng.standard_normal(cout, dtype=np.float32)
    assert x.dtype == weights.dtype == bias.dtype == np.float32
    return x, weights, bias


@pytest.mark.parametrize(
    "cin,cout,h,w",
    [
        (1, 1, 1, 1),        # degenerate: one channel, one pixel
        (4, 3, 2, 128),      # exactly one column tile
        (4, 3, 2, 100),      # partial tile -- padding lanes must not leak
        (4, 3, 2, 200),      # two column tiles
        (128, 8, 3, 64),     # exactly one full channel group
        (129, 4, 2, 64),     # two groups, last holds 1 channel (the min() path)
        (256, 4, 2, 64),     # two exact groups (SuperPoint convDb's Cin)
        (257, 2, 2, 32),     # three groups, ragged
        (16, 65, 2, 32),     # SuperPoint convPb's Cout
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
    x, weights, bias = _random_case(6, 5, 3, 7, seed=99)
    ref = _reference(x, weights, bias)
    assert np.abs(ref - _naive_reference(x, weights, bias)).max() < TOL
    assert np.abs(_run(inst_file, tmp_path, x, weights, bias) - ref).max() < TOL


def test_bias_only(inst_file, tmp_path):
    """Zero weights => every output element is its channel's bias.

    Isolates the MULT.EE + ACC.ADD.FIRST word that both resets the accumulator
    and seeds it with the bias.
    """
    cin, cout, h, w = 8, 4, 3, 40
    rng = np.random.default_rng(7)
    x = rng.standard_normal((cin, h, w), dtype=np.float32)
    weights = np.zeros((cout, cin), dtype=np.float32)
    bias = rng.standard_normal(cout, dtype=np.float32)
    out = _run(inst_file, tmp_path, x, weights, bias)
    expected = np.broadcast_to(bias[:, None, None], (cout, h, w))
    assert np.abs(out - expected).max() < TOL


def test_zero_bias_when_no_bias_file(inst_file, tmp_path):
    """An absent bias_path leaves the bias region zero rather than unset."""
    cin, cout, h, w = 4, 3, 2, 32
    x, weights, _ = _random_case(cin, cout, h, w, seed=3)
    xp, wp, op = tmp_path / "x.bin", tmp_path / "w.bin", tmp_path / "y.bin"
    x.tofile(xp)
    weights.tofile(wp)
    app = Conv1x1Fp32App(
        inst_path=inst_file,
        input_path=xp,
        weight_path=wp,
        output_path=op,
        in_channels=cin,
        out_channels=cout,
        height=h,
        width=w,
    )
    app.run(max_cycles=20_000_000)
    out = np.frombuffer(op.read_bytes(), dtype="<f4").reshape(cout, h, w)
    ref = _reference(x, weights, np.zeros(cout, dtype=np.float32))
    assert np.abs(out - ref).max() < TOL


def test_padding_lanes_are_computed_but_excluded(inst_file, tmp_path):
    """The kernel writes the pad lanes; the harness must not hand them back.

    A store always drains a full 128-lane row, so lanes W..NCT*128-1 really are
    written -- with bias[o], since the input padding is zero. This pins both
    halves: XMEM holds those lanes, and the output file does not.
    """
    cin, cout, h, w = 6, 5, 3, 70
    x, weights, bias = _random_case(cin, cout, h, w, seed=11)
    xp, wp, bp, op = (tmp_path / n for n in ("x.bin", "w.bin", "b.bin", "y.bin"))
    x.tofile(xp)
    weights.tofile(wp)
    bias.tofile(bp)
    app = Conv1x1Fp32App(
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
    state, _ = app.run(max_cycles=20_000_000)

    padded_w = app.tiles_per_row * 128
    assert padded_w > w, "this case must actually pad, or it tests nothing"
    raw = state.xmem.read_address(app.output_base, app.output_rows * 512)
    planes = np.frombuffer(raw, dtype="<f4").reshape(cout, h, padded_w)
    # Pad lanes saw a zero input, so they carry the bias -- proof they were
    # written rather than left untouched.
    pad = planes[:, :, w:]
    assert np.abs(pad - np.broadcast_to(bias[:, None, None], pad.shape)).max() < TOL

    out = np.frombuffer(op.read_bytes(), dtype="<f4").reshape(cout, h, w)
    assert np.abs(out - _reference(x, weights, bias)).max() < TOL


def test_output_file_layout_is_dense(inst_file, tmp_path):
    """The output file is exactly Cout*H*W FP32 in reshape order.

    The conv analogue of the softmax layout round-trip: a caller must be able
    to np.frombuffer(...).reshape(Cout, H, W) with no app-specific unpacking.
    """
    cin, cout, h, w = 5, 4, 3, 90
    x, weights, bias = _random_case(cin, cout, h, w, seed=5)
    xp, wp, bp, op = (tmp_path / n for n in ("x.bin", "w.bin", "b.bin", "y.bin"))
    x.tofile(xp)
    weights.tofile(wp)
    bias.tofile(bp)
    app = Conv1x1Fp32App(
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
    app.run(max_cycles=20_000_000)
    assert op.stat().st_size == cout * h * w * 4
    out = np.frombuffer(op.read_bytes(), dtype="<f4").reshape(cout, h, w)
    assert np.abs(out - _reference(x, weights, bias)).max() < TOL


# -- registry conformance ---------------------------------------------------
# The generic suites in test_kernel_registry.py are scoped to op="softmax";
# these are the conv2d parallels.


def test_registry_resolves_to_this_kernel(inst_file, tmp_path):
    """resolve("conv2d", ...) picks this kernel and its kwargs actually run."""
    cin, cout, h, w = 8, 6, 2, 64
    verdict = resolve(
        "conv2d",
        shape=(cin, h, w),
        weight_shape=(cout, cin, 1, 1),
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
    )
    assert verdict.supported, verdict.reason
    assert verdict.app_name == "conv1x1_fp32"
    assert verdict.shapes["output"] == (cout, h, w)
    assert "output" in verdict.shapes.derived_roles

    x, weights, bias = _random_case(cin, cout, h, w, seed=17)
    xp, wp, bp, op = (tmp_path / n for n in ("x.bin", "w.bin", "b.bin", "y.bin"))
    x.tofile(xp)
    weights.tofile(wp)
    bias.tofile(bp)
    app = verdict.kernel.app_class(
        inst_path=inst_file,
        input_path=xp,
        weight_path=wp,
        bias_path=bp,
        output_path=op,
        **verdict.kwargs,
    )
    app.run(max_cycles=20_000_000)
    out = np.frombuffer(op.read_bytes(), dtype="<f4").reshape(cout, h, w)
    assert np.abs(out - _reference(x, weights, bias)).max() < TOL


def _query(**overrides):
    return {
        "shape": (4, 8, 8),
        "weight_shape": (4, 4, 1, 1),
        "stride": 1,
        "padding": 0,
        "dilation": 1,
        "groups": 1,
        **overrides,
    }


@pytest.mark.parametrize(
    "overrides,expected",
    [
        ({"weight_shape": (4, 4, 3, 3)}, "3x3"),      # not pointwise
        ({"stride": 2}, "stride 2"),                  # strided
        ({"padding": 1}, "padding 1"),                # padded
        ({"dilation": 2}, "dilation 2"),              # dilated
        ({"groups": 2}, "groups=2"),                  # grouped
        ({"shape": (2, 4, 8, 8)}, "batch of 2"),      # batched
    ],
)
def test_router_refuses_geometry_no_kernel_implements(overrides, expected):
    """Geometry the constructor cannot even express must still be refused.

    These are router-only refusals: the constructor pins kh=kw=1, stride 1,
    padding 0, dilation 1, groups 1 by construction, so it can never be asked
    for them. The registry is the layer that has to say no, and say why -- the
    reason must name the actual value, not just fail.
    """
    verdict = resolve("conv2d", **_query(**overrides))
    assert not verdict.supported
    assert expected in verdict.reason, verdict.reason


@pytest.mark.parametrize(
    "overrides",
    [
        {"shape": (8, 8)},                    # rank-2: not a conv activation
        {"weight_shape": (4, 4, 1)},          # rank-3 weight
        {"stride": 0},                        # would divide by zero
        {"dilation": 0},
        {"groups": 0},
        {"padding": -1},
        {"stride": (1, 2)},                   # asymmetric: not modelled
    ],
)
def test_malformed_queries_propagate(overrides):
    """A broken *question* raises rather than being reported as missing coverage.

    ``resolve`` re-raises ``MalformedQuery`` on purpose: no amount of adding
    kernels would make a rank-2 "convolution" answerable, so reporting it as a
    coverage gap would send the caller looking for the wrong fix. It also keeps
    a zero stride from escaping as a ZeroDivisionError.
    """
    with pytest.raises(MalformedQuery):
        resolve("conv2d", **_query(**overrides))


@pytest.mark.parametrize(
    "overrides,ctor",
    [
        # Over the XMEM budget: 512 channels at 512x512 needs ~1M rows.
        ({"shape": (512, 512, 512), "weight_shape": (512, 512, 1, 1)},
         {"in_channels": 512, "out_channels": 512, "height": 512, "width": 512}),
        ({"shape": (0, 8, 8), "weight_shape": (4, 0, 1, 1)},
         {"in_channels": 0, "out_channels": 4, "height": 8, "width": 8}),
        ({"shape": (4, 0, 8), "weight_shape": (4, 4, 1, 1)},
         {"in_channels": 4, "out_channels": 4, "height": 0, "width": 8}),
        ({"shape": (4, 8, 0), "weight_shape": (4, 4, 1, 1)},
         {"in_channels": 4, "out_channels": 4, "height": 8, "width": 0}),
        ({"shape": (4, 8, 8), "weight_shape": (0, 4, 1, 1)},
         {"in_channels": 4, "out_channels": 0, "height": 8, "width": 8}),
    ],
)
def test_constructor_guard_matches_spec(overrides, ctor):
    """Whatever ``supports`` refuses, the constructor must refuse too.

    Restricted to refusals the constructor can actually express -- the ones
    driven by the shape it is handed. Both paths go through SPEC, so this pins
    the delegation rather than the bounds: a hand-written guard could drift
    from the router, which is exactly what SPEC.guard exists to prevent.
    """
    query = _query(**overrides)
    assert not SPEC.check(**query).ok
    assert not resolve("conv2d", **query).supported

    with pytest.raises(ValueError):
        Conv1x1Fp32App(
            inst_path="unused.bin",
            input_path="unused.bin",
            weight_path="unused.bin",
            **ctor,
        )
