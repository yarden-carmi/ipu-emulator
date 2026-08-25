"""Numpy-parity tests for the keypoint selection gate + soft survivor count."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from ipu_as.lark_tree import assemble_to_bin_file
from ipu_apps.detect.score_threshold import (
    DEFAULT_TEMPERATURE,
    SPEC,
    ScoreThresholdApp,
)
from ipu_apps.kernel_registry import MalformedQuery, resolve

ASM_PATH = (
    Path(__file__).resolve().parents[1]
    / "src/ipu_apps/detect/score_threshold"
    / "score_threshold.asm"
)

# `selected` is `s - tau` in FP32 against a float64 reference, so it carries
# FP32's ~1e-7 relative error; the count is a sum of up to N sigmoids, so its
# error grows with N.
RTOL, ATOL = 1e-6, 1e-6


def _sel_reference(s: np.ndarray, tau: float) -> np.ndarray:
    return np.maximum(s.astype(np.float64) - tau, 0.0)


def _count_reference(s: np.ndarray, tau: float, t: float) -> float:
    z = t * (s.astype(np.float64) - tau)
    # The stable sigmoid the emulator uses, so this measures the kernel and not
    # numpy's overflow behaviour.
    pos = np.where(z >= 0, 1.0 / (1.0 + np.exp(-np.abs(z))), 0.0)
    neg = np.where(z < 0, np.exp(-np.abs(z)) / (1.0 + np.exp(-np.abs(z))), 0.0)
    return float((pos + neg).sum())


@pytest.fixture(scope="module")
def inst_file():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "score_threshold.bin"
        assemble_to_bin_file(ASM_PATH.read_text(), str(path))
        yield path


def _run(inst_file, tmp_path, s: np.ndarray, tau: float, t=DEFAULT_TEMPERATURE):
    sp = tmp_path / "s.bin"
    selp, cntp = tmp_path / "sel.bin", tmp_path / "cnt.bin"
    s.astype("<f4").tofile(sp)
    app = ScoreThresholdApp(
        inst_path=inst_file,
        input_path=sp,
        output_path=selp,
        count_path=cntp,
        shape=s.shape,
        threshold=tau,
        temperature=t,
    )
    app.run(max_cycles=200_000_000)
    sel = np.frombuffer(selp.read_bytes(), dtype="<f4").reshape(s.shape)
    count = float(np.frombuffer(cntp.read_bytes(), dtype="<f4")[0])
    return sel, count


@pytest.mark.parametrize(
    "shape",
    [
        (1,),           # degenerate: one score
        (7,),           # one partly-filled row
        (128,),         # exactly one full row
        (129,),         # spills to a second row
        (300,),         # three rows
        (8, 40),        # rank 2: the shape is preserved but does not matter
        (4, 5, 6),      # rank 3
        (60, 80),       # a SuperPoint cell grid
    ],
)
def test_matches_numpy(inst_file, tmp_path, shape):
    rng = np.random.default_rng(sum(shape) * 31 + len(shape))
    s = rng.standard_normal(shape, dtype=np.float32)
    tau = 0.25
    sel, count = _run(inst_file, tmp_path, s, tau)
    assert sel.shape == s.shape
    assert np.allclose(sel, _sel_reference(s, tau), rtol=RTOL, atol=ATOL)
    ref = _count_reference(s, tau, DEFAULT_TEMPERATURE)
    assert abs(count - ref) < max(1e-3, 1e-5 * s.size)


def test_gate_selects_exactly_the_survivors(inst_file, tmp_path):
    """{selected > 0} must equal {s > tau} -- the exact part of this kernel."""
    rng = np.random.default_rng(11)
    s = rng.standard_normal(1000, dtype=np.float32)
    tau = 0.5
    sel, _ = _run(inst_file, tmp_path, s, tau)
    assert np.array_equal(sel > 0, s > np.float32(tau))
    assert (sel > 0).any() and (sel == 0).any(), "degenerate: nothing to compare"


def test_count_tracks_the_true_survivor_count(inst_file, tmp_path):
    """At T=64 the sigmoid is sharp enough that the count is nearly integral.

    This is the property the host's bisection depends on: the count has to move
    with the real number of survivors, not merely be monotone.
    """
    rng = np.random.default_rng(3)
    s = rng.uniform(-1.0, 1.0, size=600).astype(np.float32)
    for tau in (-0.5, 0.0, 0.5):
        _, count = _run(inst_file, tmp_path, s, tau)
        true = int((s > np.float32(tau)).sum())
        assert abs(count - true) < 0.05 * s.size, (tau, count, true)


def test_count_is_monotone_in_tau(inst_file, tmp_path):
    """A bisection on tau is only well-founded if the count decreases with tau."""
    rng = np.random.default_rng(5)
    s = rng.standard_normal(500, dtype=np.float32)
    counts = [_run(inst_file, tmp_path, s, tau)[1] for tau in (-1.0, -0.5, 0.0, 0.5, 1.0)]
    assert counts == sorted(counts, reverse=True), counts


def test_padding_lanes_contribute_nothing_to_the_count(inst_file, tmp_path):
    """The whole point of the tau - 800/T fill.

    A count over 100 scores in a 128-lane row must not pick up the 28 padding
    lanes. With a zero fill and a negative tau they would each contribute
    nearly 1, inflating the count by 28.
    """
    rng = np.random.default_rng(8)
    s = rng.uniform(0.5, 1.5, size=100).astype(np.float32)
    # Every real score clears this, so the true count is exactly 100.
    _, count = _run(inst_file, tmp_path, s, -1.0)
    assert abs(count - 100.0) < 0.5, count


def test_padding_lanes_do_not_appear_in_the_output(inst_file, tmp_path):
    """The gate output is trimmed to the real element count."""
    rng = np.random.default_rng(9)
    s = rng.standard_normal(100, dtype=np.float32)
    sel, _ = _run(inst_file, tmp_path, s, -10.0)
    assert sel.shape == (100,)
    assert (sel > 0).all(), "every score clears this threshold"


def test_a_higher_temperature_sharpens_the_count(inst_file, tmp_path):
    """T controls how tightly the soft count hugs the integral one."""
    rng = np.random.default_rng(15)
    s = rng.uniform(-1.0, 1.0, size=400).astype(np.float32)
    true = int((s > np.float32(0.0)).sum())
    _, soft = _run(inst_file, tmp_path, s, 0.0, t=1.0)
    _, sharp = _run(inst_file, tmp_path, s, 0.0, t=512.0)
    assert abs(sharp - true) < abs(soft - true)


def test_output_file_layout_is_dense(inst_file, tmp_path):
    """The output file is exactly prod(shape) FP32; the count is one FP32."""
    shape = (30, 40)
    rng = np.random.default_rng(5)
    s = rng.standard_normal(shape, dtype=np.float32)
    sp = tmp_path / "s.bin"
    selp, cntp = tmp_path / "sel.bin", tmp_path / "cnt.bin"
    s.astype("<f4").tofile(sp)
    app = ScoreThresholdApp(
        inst_path=inst_file, input_path=sp, output_path=selp, count_path=cntp,
        shape=shape, threshold=0.0,
    )
    app.run(max_cycles=200_000_000)
    assert selp.stat().st_size == 30 * 40 * 4
    assert cntp.stat().st_size == 4


def test_pad_value_is_what_the_kernel_assumes():
    """T * (pad - tau) must be exactly -800, whatever tau and T are."""
    for tau, t in ((0.005, 64.0), (-3.0, 1.0), (0.0, 512.0)):
        app = ScoreThresholdApp(
            inst_path="unused.bin", input_path="unused.bin",
            shape=(10,), threshold=tau, temperature=t,
        )
        assert abs(t * (app.pad_value - tau) + 800.0) < 1e-6, (tau, t)


# -- registry conformance ---------------------------------------------------


def test_registry_resolves_to_this_kernel(inst_file, tmp_path):
    verdict = resolve("score_threshold", shape=(8, 40), threshold=0.25)
    assert verdict.supported, verdict.reason
    assert verdict.app_name == "score_threshold"
    assert verdict.shapes["selected"] == (8, 40)
    assert verdict.shapes["count"] == (1,)
    # temperature is not required, so it must come back defaulted.
    assert verdict.kwargs["temperature"] == DEFAULT_TEMPERATURE

    rng = np.random.default_rng(17)
    s = rng.standard_normal((8, 40), dtype=np.float32)
    sp, selp = tmp_path / "s.bin", tmp_path / "sel.bin"
    s.astype("<f4").tofile(sp)
    app = verdict.kernel.app_class(
        inst_path=inst_file, input_path=sp, output_path=selp, **verdict.kwargs
    )
    app.run(max_cycles=200_000_000)
    sel = np.frombuffer(selp.read_bytes(), dtype="<f4").reshape(8, 40)
    assert np.allclose(sel, _sel_reference(s, 0.25), rtol=RTOL, atol=ATOL)


def test_the_no_ranked_indices_limit_is_stated():
    """The one behavioural approximation in the detector path; say it out loud."""
    verdict = resolve("score_threshold", shape=(4800,), threshold=0.005)
    assert verdict.supported, verdict.reason
    assert any("ranked indices" in c for c in verdict.caveats), verdict.caveats


@pytest.mark.parametrize(
    "params,expected",
    [
        ({"shape": (0,), "threshold": 0.0}, "no elements"),
        ({"shape": (10,), "threshold": 0.0, "temperature": 1e-12}, "at least 1e-06"),
        ({"shape": (10,), "threshold": 0.0, "temperature": -1.0}, "BELOW the"),
        ({"shape": (4_000_000,), "threshold": 0.0}, "XMEM rows"),
    ],
)
def test_router_refuses_what_no_kernel_implements(params, expected):
    verdict = resolve("score_threshold", **params)
    assert not verdict.supported
    assert expected in verdict.reason, verdict.reason


@pytest.mark.parametrize(
    "params,ctor",
    [
        ({"shape": (4_000_000,), "threshold": 0.0},
         {"shape": (4_000_000,), "threshold": 0.0}),
        ({"shape": (0,), "threshold": 0.0}, {"shape": (0,), "threshold": 0.0}),
        ({"shape": (10,), "threshold": 0.0, "temperature": -1.0},
         {"shape": (10,), "threshold": 0.0, "temperature": -1.0}),
    ],
)
def test_constructor_guard_matches_spec(params, ctor):
    """Whatever ``supports`` refuses, the constructor must refuse too."""
    assert not SPEC.check(**params).ok
    with pytest.raises(ValueError):
        ScoreThresholdApp(inst_path="unused.bin", input_path="unused.bin", **ctor)


def test_over_budget_refusal_says_how_to_chunk():
    verdict = resolve("score_threshold", shape=(4_000_000,), threshold=0.0)
    assert not verdict.supported
    assert "the counts add" in verdict.reason, verdict.reason


@pytest.mark.parametrize(
    "params",
    [
        {"shape": (), "threshold": 0.0},
        {"shape": (10,), "threshold": float("nan")},
        {"shape": (10,), "threshold": 0.0, "temperature": float("inf")},
    ],
)
def test_malformed_queries_propagate(params):
    with pytest.raises(MalformedQuery):
        resolve("score_threshold", **params)
