"""The Cin=1 convolution: same result as the general kernel, 1.77x cheaper."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from ipu_as.lark_tree import assemble_to_bin_file
from ipu_apps.convolutions_universal.conv.conv3x3_relu import Conv3x3ReluApp
from ipu_apps.convolutions_universal.conv.conv3x3_relu_cin1 import (
    SPEC,
    Conv3x3ReluCin1App,
)
from ipu_apps.kernel_registry import resolve

SRC = Path(__file__).resolve().parents[1] / "src/ipu_apps/convolutions_universal/conv"
TOL = 1e-4


def _reference(x, w, b):
    """relu(bias + 3x3 zero-padded conv), as nine shifted plane MACs."""
    cin, h, width = x.shape
    cout = w.shape[0]
    xp = np.zeros((cin, h + 2, width + 2), dtype=np.float64)
    xp[:, 1 : h + 1, 1 : width + 1] = x
    out = np.broadcast_to(b[:, None, None].astype(np.float64), (cout, h, width)).copy()
    for kr in range(3):
        for kc in range(3):
            out += np.einsum("oc,chw->ohw",
                             w[:, :, kr, kc].astype(np.float64),
                             xp[:, kr : kr + h, kc : kc + width])
    return np.maximum(out, 0.0)


@pytest.fixture(scope="module")
def inst():
    with tempfile.TemporaryDirectory() as tmp:
        built = {}
        for name in ("conv3x3_relu_cin1", "conv3x3_relu"):
            out = Path(tmp) / f"{name}.bin"
            assemble_to_bin_file((SRC / name / f"{name}.asm").read_text(), str(out))
            built[name] = out
        yield built


def _case(cout, h, w, seed):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((1, h, w), dtype=np.float32)
    weights = rng.standard_normal((cout, 1, 3, 3), dtype=np.float32) / np.float32(3.0)
    bias = rng.standard_normal(cout, dtype=np.float32) + np.float32(0.3)
    return x, weights, bias


def _run(app_cls, inst_path, tmp_path, x, weights, bias, **kw):
    cout, h, w = weights.shape[0], x.shape[1], x.shape[2]
    xp, wp, bp, op = (tmp_path / n for n in ("x.bin", "w.bin", "b.bin", "y.bin"))
    x.astype("<f4").tofile(xp)
    weights.astype("<f4").tofile(wp)
    bias.astype("<f4").tofile(bp)
    app = app_cls(inst_path=inst_path, input_path=xp, weight_path=wp,
                  bias_path=bp, output_path=op, height=h, width=w,
                  out_channels=cout, **kw)
    _s, cycles = app.run(max_cycles=200_000_000)
    return np.frombuffer(op.read_bytes(), dtype="<f4").reshape(cout, h, w), cycles


@pytest.mark.parametrize(
    "cout,h,w",
    [
        (1, 1, 1),      # degenerate: every tap is padding except the centre
        (1, 3, 3),      # smallest case exercising all nine taps
        (2, 4, 5),      # rectangular, tiny
        (4, 3, 126),    # exactly one full tile
        (4, 3, 127),    # spills to a second tile by one column
        (3, 2, 200),    # two tiles, halo crossing a boundary
        (64, 4, 80),    # conv1a's channel count on a small band
    ],
)
def test_matches_numpy(inst, tmp_path, cout, h, w):
    x, weights, bias = _case(cout, h, w, seed=cout * 31 + h + w)
    got, _ = _run(Conv3x3ReluCin1App, inst["conv3x3_relu_cin1"], tmp_path,
                  x, weights, bias)
    assert np.abs(got - _reference(x, weights, bias)).max() < TOL


def test_identical_to_the_general_kernel_and_cheaper(inst, tmp_path):
    """Same numbers, materially fewer cycles -- the whole justification."""
    x, weights, bias = _case(16, 8, 200, seed=5)
    fast, fast_cyc = _run(Conv3x3ReluCin1App, inst["conv3x3_relu_cin1"],
                          tmp_path, x, weights, bias)
    slow, slow_cyc = _run(Conv3x3ReluApp, inst["conv3x3_relu"], tmp_path,
                          x, weights, bias, in_channels=1)
    assert np.abs(fast - slow).max() < 1e-6, "the unrolled kernel changed the result"
    assert fast_cyc < slow_cyc * 0.62, (fast_cyc, slow_cyc)


def test_tap_order_matches_weight_layout(inst, tmp_path):
    """Nine distinct weights on an impulse: pins each tap to its own neighbour."""
    x = np.zeros((1, 5, 5), dtype=np.float32)
    x[0, 2, 2] = 1.0
    weights = (np.arange(9, dtype=np.float32) + 1.0).reshape(1, 1, 3, 3)
    bias = np.zeros(1, dtype=np.float32)
    got, _ = _run(Conv3x3ReluCin1App, inst["conv3x3_relu_cin1"], tmp_path,
                  x, weights, bias)
    assert np.abs(got - _reference(x, weights, bias)).max() < TOL
    assert set(np.unique(got)) >= set(range(1, 10)), np.unique(got)


def test_borders_are_zero_padded(inst, tmp_path):
    """One centre-tap weight at a time; the border must read zero."""
    rng = np.random.default_rng(21)
    x = rng.uniform(1.0, 2.0, size=(1, 4, 6)).astype(np.float32)
    bias = np.zeros(1, dtype=np.float32)
    for kr in range(3):
        for kc in range(3):
            weights = np.zeros((1, 1, 3, 3), dtype=np.float32)
            weights[0, 0, kr, kc] = 1.0
            got, _ = _run(Conv3x3ReluCin1App, inst["conv3x3_relu_cin1"],
                          tmp_path, x, weights, bias)
            assert np.abs(got - _reference(x, weights, bias)).max() < TOL, (kr, kc)


def test_relu_actually_clamps(inst, tmp_path):
    x, weights, _ = _case(2, 4, 10, seed=4)
    bias = np.full(2, -1000.0, dtype=np.float32)
    got, _ = _run(Conv3x3ReluCin1App, inst["conv3x3_relu_cin1"], tmp_path,
                  x, weights, bias)
    assert np.all(got == 0.0)


def test_no_guard_plane_is_allocated():
    """The unrolled kernel has no channel prefetch, so it needs one plane.

    The general kernel reserves Cin+1 planes for its one-past prefetch; at
    Cin=1 that is double the input region. A 40-row band of conv1a's real
    width, since full 480x640 is over the XMEM budget either way.
    """
    app = Conv3x3ReluCin1App(
        inst_path="unused.bin", input_path="unused.bin", weight_path="unused.bin",
        out_channels=64, height=40, width=640,
    )
    assert app.tiles_per_row == 6
    assert app.padded_height == 42
    # one plane, not two
    assert app.weight_base_row - app.input_base_row == 42 * 6


def test_registry_prefers_the_specialised_kernel_at_cin1():
    v = resolve("conv2d", shape=(1, 8, 40), weight_shape=(64, 1, 3, 3),
                stride=1, padding=1, dilation=1, groups=1, activation="relu")
    assert v.supported, v.reason
    assert v.app_name == "conv3x3_relu_cin1"
    assert "conv3x3_relu" in v.alternatives, v.alternatives
    assert any("Cin is fixed at 1" in c for c in v.caveats), v.caveats


@pytest.mark.parametrize("cin", [2, 14, 64, 128])
def test_more_channels_route_to_the_general_kernel(cin):
    v = resolve("conv2d", shape=(cin, 8, 40), weight_shape=(8, cin, 3, 3),
                stride=1, padding=1, dilation=1, groups=1, activation="relu")
    assert v.supported, v.reason
    assert v.app_name == "conv3x3_relu"


def test_spec_refuses_more_channels():
    verdict = SPEC.check(shape=(64, 8, 40), weight_shape=(8, 64, 3, 3),
                         stride=1, padding=1, dilation=1, groups=1,
                         activation="relu")
    assert not verdict.ok
    assert "exactly 1 input channel" in verdict.reason, verdict.reason
    assert "conv3x3_relu" in verdict.reason, verdict.reason


def test_imem_footprint_fits_one_bank(inst):
    """25 words. Unrolling Cin=2 would be 34, Cin=14 would be 130 -- the wall."""
    assert inst["conv3x3_relu_cin1"].stat().st_size // 24 == 25
