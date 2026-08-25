"""Numpy-parity tests for the detector confidence + gate app."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from ipu_as.lark_tree import assemble_to_bin_file
from ipu_apps.detect.channel_peak import SPEC, ChannelPeakApp
from ipu_apps.kernel_registry import MalformedQuery, resolve

ASM_PATH = (
    Path(__file__).resolve().parents[1]
    / "src/ipu_apps/detect/channel_peak"
    / "channel_peak.asm"
)

# The confidence is a maximum, so it is bit-exact. `keep` is `conf - tau`
# computed in FP32 against a float64 reference, so it carries FP32's ~1e-7
# relative error -- a purely absolute tolerance would fail on large scores.
RTOL, ATOL = 1e-6, 1e-6


def _close(got, want):
    return np.allclose(got, want, rtol=RTOL, atol=ATOL)


def _reference(x: np.ndarray, tau: float):
    conf = x.astype(np.float64).max(axis=0)
    return conf, np.maximum(conf - tau, 0.0)


@pytest.fixture(scope="module")
def inst_file():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "channel_peak.bin"
        assemble_to_bin_file(ASM_PATH.read_text(), str(path))
        yield path


def _run(inst_file, tmp_path, x: np.ndarray, tau: float):
    c, n = x.shape
    xp = tmp_path / "x.bin"
    keep_p, conf_p = tmp_path / "keep.bin", tmp_path / "conf.bin"
    x.astype("<f4").tofile(xp)
    app = ChannelPeakApp(
        inst_path=inst_file,
        input_path=xp,
        output_path=keep_p,
        confidence_path=conf_p,
        channels=c,
        cells=n,
        threshold=tau,
    )
    app.run(max_cycles=200_000_000)
    return (
        np.frombuffer(conf_p.read_bytes(), dtype="<f4"),
        np.frombuffer(keep_p.read_bytes(), dtype="<f4"),
    )


@pytest.mark.parametrize(
    "c,n",
    [
        (1, 1),        # degenerate: one plane, one cell -- the peeled path only
        (1, 200),      # one plane, so the channel loop is skipped entirely
        (2, 5),        # tiny
        (64, 128),     # exactly one full tile
        (64, 129),     # spills to a second tile
        (65, 100),     # SuperPoint's detector channel count, partial tile
        (64, 4800),    # a 60x80 cell grid: the real detector map
        (3, 300),      # three tiles
    ],
)
def test_matches_numpy(inst_file, tmp_path, c, n):
    rng = np.random.default_rng(c * 131 + n)
    x = rng.standard_normal((c, n), dtype=np.float32)
    conf, keep = _run(inst_file, tmp_path, x, 0.25)
    ref_conf, ref_keep = _reference(x, 0.25)
    assert conf.shape == ref_conf.shape
    # A maximum selects an input element unchanged, so the confidence is exact.
    assert np.array_equal(conf.astype(np.float64), ref_conf)
    assert _close(keep, ref_keep)


def test_one_plane_skips_the_channel_loop(inst_file, tmp_path):
    """C == 1: the peeled plane 0 is the whole answer, and BGE must skip the loop.

    Without that branch the loop would fold in whatever plane 1's address
    happens to point at -- the tau row, in this layout.
    """
    rng = np.random.default_rng(3)
    x = rng.standard_normal((1, 300), dtype=np.float32)
    conf, _ = _run(inst_file, tmp_path, x, 0.0)
    assert np.array_equal(conf, x[0])


def test_all_negative_input(inst_file, tmp_path):
    """A zero accumulator seed would beat every real value here.

    This is what pins ``ACC.MAX.FIRST`` on the peeled plane rather than a
    running ``ACC.MAX`` against a stale R_ACC.
    """
    rng = np.random.default_rng(7)
    x = -rng.uniform(1.0, 100.0, size=(16, 500)).astype(np.float32)
    conf, keep = _run(inst_file, tmp_path, x, -50.0)
    ref_conf, ref_keep = _reference(x, -50.0)
    assert np.array_equal(conf.astype(np.float64), ref_conf)
    assert _close(keep, ref_keep)


def test_gate_selects_exactly_the_survivors(inst_file, tmp_path):
    """{keep > 0} must equal {conf > tau} -- the property the detector relies on."""
    rng = np.random.default_rng(11)
    x = rng.standard_normal((8, 400), dtype=np.float32)
    tau = 0.5
    conf, keep = _run(inst_file, tmp_path, x, tau)
    assert np.array_equal(keep > 0, conf > np.float32(tau))
    assert (keep > 0).any() and (keep == 0).any(), "degenerate: nothing to compare"


def test_argmax_matches_the_softmax_path(inst_file, tmp_path):
    """The claim the docstring makes: argmax(softmax(x)) == argmax(x).

    This is why the detector may skip the softmax when only the ranking
    matters, and it is worth testing rather than asserting in prose.
    """
    rng = np.random.default_rng(13)
    x = rng.standard_normal((65, 500), dtype=np.float32)
    conf, _ = _run(inst_file, tmp_path, x, 0.0)
    shifted = x.astype(np.float64) - x.astype(np.float64).max(axis=0, keepdims=True)
    probs = np.exp(shifted) / np.exp(shifted).sum(axis=0, keepdims=True)
    assert np.array_equal(np.argmax(probs, axis=0), np.argmax(x, axis=0))
    assert np.array_equal(conf.astype(np.float64), x.astype(np.float64).max(axis=0))


def test_padding_cells_do_not_change_real_cells(inst_file, tmp_path):
    """A cell count that does not fill its tile must behave like one that does."""
    rng = np.random.default_rng(21)
    x = rng.standard_normal((6, 100), dtype=np.float32)
    conf, keep = _run(inst_file, tmp_path, x, 0.1)
    ref_conf, ref_keep = _reference(x, 0.1)
    assert np.array_equal(conf.astype(np.float64), ref_conf)
    assert _close(keep, ref_keep)


def test_output_file_layout_is_dense(inst_file, tmp_path):
    """Both output files are exactly `cells` FP32."""
    c, n = 8, 300
    rng = np.random.default_rng(5)
    x = rng.standard_normal((c, n), dtype=np.float32)
    xp = tmp_path / "x.bin"
    keep_p, conf_p = tmp_path / "keep.bin", tmp_path / "conf.bin"
    x.astype("<f4").tofile(xp)
    app = ChannelPeakApp(
        inst_path=inst_file, input_path=xp, output_path=keep_p,
        confidence_path=conf_p, channels=c, cells=n, threshold=0.0,
    )
    app.run(max_cycles=200_000_000)
    assert keep_p.stat().st_size == n * 4
    assert conf_p.stat().st_size == n * 4


# -- registry conformance ---------------------------------------------------


def test_registry_resolves_a_detector_map(inst_file, tmp_path):
    """A (C, H, W) logit map flattens to cells, and the flatten is disclosed."""
    c, h, w = 8, 4, 5
    verdict = resolve("channel_peak", shape=(c, h, w), threshold=0.1)
    assert verdict.supported, verdict.reason
    assert verdict.app_name == "channel_peak"
    assert verdict.kwargs == {"channels": c, "cells": h * w, "threshold": 0.1}
    assert verdict.shapes["confidence"] == (h * w,)
    assert any("flattened" in n for n in verdict.shapes.notes), verdict.shapes.notes

    rng = np.random.default_rng(17)
    x = rng.standard_normal((c, h * w), dtype=np.float32)
    conf, keep = _run(inst_file, tmp_path, x, 0.1)
    ref_conf, ref_keep = _reference(x, 0.1)
    assert np.array_equal(conf.astype(np.float64), ref_conf)
    assert _close(keep, ref_keep)


def test_the_logit_caveat_is_stated():
    """Using this where the value matters is the failure mode; say so."""
    verdict = resolve("channel_peak", shape=(65, 4800), threshold=0.005)
    assert verdict.supported, verdict.reason
    assert any("not a probability" in c for c in verdict.caveats), verdict.caveats


@pytest.mark.parametrize(
    "shape,expected",
    [
        ((0, 8), "channels (0)"),
        ((8, 0), "cells (0)"),
        ((4096, 4096), "XMEM rows"),
    ],
)
def test_router_refuses_what_no_kernel_implements(shape, expected):
    verdict = resolve("channel_peak", shape=shape, threshold=0.0)
    assert not verdict.supported
    assert expected in verdict.reason, verdict.reason


@pytest.mark.parametrize(
    "shape,ctor",
    [
        ((4096, 4096), {"channels": 4096, "cells": 4096, "threshold": 0.0}),
        ((0, 8), {"channels": 0, "cells": 8, "threshold": 0.0}),
        ((8, 0), {"channels": 8, "cells": 0, "threshold": 0.0}),
    ],
)
def test_constructor_guard_matches_spec(shape, ctor):
    """Whatever ``supports`` refuses, the constructor must refuse too."""
    assert not SPEC.check(shape=shape, threshold=0.0).ok
    with pytest.raises(ValueError):
        ChannelPeakApp(inst_path="unused.bin", input_path="unused.bin", **ctor)


def test_over_budget_refusal_says_how_to_split():
    verdict = resolve("channel_peak", shape=(4096, 4096), threshold=0.0)
    assert not verdict.supported
    assert "channels per launch" in verdict.reason, verdict.reason


@pytest.mark.parametrize(
    "shape,tau",
    [
        ((8,), 0.0),
        ((2, 3, 4, 5), 0.0),
        ((8, 8), float("nan")),
        ((8, 8), float("inf")),
    ],
)
def test_malformed_queries_propagate(shape, tau):
    with pytest.raises(MalformedQuery):
        resolve("channel_peak", shape=shape, threshold=tau)
