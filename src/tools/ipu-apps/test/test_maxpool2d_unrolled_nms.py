"""The unrolled NMS kernels (K=9, K=7): same result as the general one, cheaper."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from ipu_as.lark_tree import assemble_to_bin_file
from ipu_apps.kernel_registry import resolve
from ipu_apps.pooling.maxpool2d_nms7 import SPEC as SPEC7
from ipu_apps.pooling.maxpool2d_nms7 import MaxPool2dNms7App
from ipu_apps.pooling.maxpool2d_nms9 import SPEC, MaxPool2dNms9App
from ipu_apps.pooling.maxpool2d_window import MaxPool2dWindowApp

SRC = Path(__file__).resolve().parents[1] / "src/ipu_apps/pooling"
NEG = -3.4028234663852886e38


def _reference(x: np.ndarray, k: int = 9) -> np.ndarray:
    """Centred stride-1 KxK max with a -FLT_MAX border, as a tap loop."""
    c, h, w = x.shape
    p = k // 2
    xp = np.full((c, h + 2 * p, w + 2 * p), NEG, dtype=np.float64)
    xp[:, p : p + h, p : p + w] = x
    out = np.full((c, h, w), NEG, dtype=np.float64)
    for dy in range(k):
        for dx in range(k):
            out = np.maximum(out, xp[:, dy : dy + h, dx : dx + w])
    return out


@pytest.fixture(scope="module")
def inst():
    with tempfile.TemporaryDirectory() as tmp:
        built = {}
        for name in ("maxpool2d_nms7", "maxpool2d_nms9", "maxpool2d_window"):
            out = Path(tmp) / f"{name}.bin"
            assemble_to_bin_file((SRC / name / f"{name}.asm").read_text(), str(out))
            built[name] = out
        yield built


def _run(app_cls, inst_path, tmp_path, x, **kw):
    c, h, w = x.shape
    xp, op = tmp_path / "x.bin", tmp_path / "y.bin"
    x.astype("<f4").tofile(xp)
    app = app_cls(inst_path=inst_path, input_path=xp, output_path=op,
                  channels=c, height=h, width=w, **kw)
    _s, cycles = app.run(max_cycles=200_000_000)
    return np.frombuffer(op.read_bytes(), dtype="<f4").reshape(c, h, w), cycles


@pytest.mark.parametrize(
    "c,h,w",
    [
        (1, 1, 1),      # a single pixel whose whole window is border
        (1, 9, 9),      # exactly the window size
        (1, 12, 40),    # the SuperPoint NMS window on a small map
        (2, 6, 120),    # exactly one full tile (TC = 120)
        (2, 6, 121),    # spills to a second tile by one column
        (1, 14, 260),   # three tiles, halo crossing both boundaries
        (3, 10, 90),    # multi-channel
    ],
)
def test_matches_numpy(inst, tmp_path, c, h, w):
    rng = np.random.default_rng(c * 131 + h * 17 + w)
    x = rng.standard_normal((c, h, w), dtype=np.float32)
    got, _ = _run(MaxPool2dNms9App, inst["maxpool2d_nms9"], tmp_path, x)
    assert np.array_equal(got.astype(np.float64), _reference(x))


def test_identical_to_the_general_kernel_and_cheaper(inst, tmp_path):
    """The whole justification: same numbers, materially fewer cycles.

    If the outputs ever diverge the specialisation is a bug, and if the cycle
    count stops winning the specialisation has no reason to exist.
    """
    rng = np.random.default_rng(5)
    x = rng.standard_normal((1, 24, 300), dtype=np.float32)
    fast, fast_cyc = _run(MaxPool2dNms9App, inst["maxpool2d_nms9"], tmp_path, x)
    slow, slow_cyc = _run(MaxPool2dWindowApp, inst["maxpool2d_window"], tmp_path,
                          x, kernel_size=9)
    assert np.array_equal(fast, slow), "the unrolled kernel changed the result"
    assert fast_cyc < slow_cyc * 0.80, (fast_cyc, slow_cyc)


def test_border_never_wins(inst, tmp_path):
    """All-negative input: a zero-filled border would beat every real value."""
    rng = np.random.default_rng(7)
    x = -rng.uniform(1.0, 100.0, size=(1, 14, 130)).astype(np.float32)
    got, _ = _run(MaxPool2dNms9App, inst["maxpool2d_nms9"], tmp_path, x)
    assert (got < 0).all()
    assert np.array_equal(got.astype(np.float64), _reference(x))


def test_single_impulse_covers_exactly_the_window(inst, tmp_path):
    """One bright pixel must dominate exactly the 9x9 block centred on it.

    Pins the three-slot rotation: a mis-rotated slot reads the wrong row and
    the hot region comes out shifted or the wrong height.
    """
    x = np.zeros((1, 20, 140), dtype=np.float32)
    x[0, 10, 70] = 1.0
    got, _ = _run(MaxPool2dNms9App, inst["maxpool2d_nms9"], tmp_path, x)
    expected = np.zeros((20, 140), dtype=bool)
    expected[10 - 4 : 10 + 5, 70 - 4 : 70 + 5] = True
    assert np.array_equal(got[0] > 0, expected)


def test_every_row_of_the_window_is_read(inst, tmp_path):
    """One impulse per dy offset, so a dropped or duplicated row shows up.

    The unrolled kernel emits nine separate row loads through three rotating
    slots; a row that is never loaded, or loaded into the slot being read,
    would still look plausible on random data.
    """
    for dy in range(9):
        x = np.full((1, 20, 130), -1.0, dtype=np.float32)
        x[0, 6 + dy, 60] = 5.0
        got, _ = _run(MaxPool2dNms9App, inst["maxpool2d_nms9"], tmp_path, x)
        assert np.array_equal(got.astype(np.float64), _reference(x)), dy


def test_registry_prefers_the_specialised_kernel_at_k9():
    """At K=9 the unrolled kernel wins on cost; the general one is the fallback."""
    v = resolve("maxpool2d", shape=(1, 60, 80), kernel_size=9, stride=1, padding=4)
    assert v.supported, v.reason
    assert v.app_name == "maxpool2d_nms9"
    # The general kernel must still CLAIM K=9 -- supports states the true
    # domain -- and appear as the alternative.
    assert "maxpool2d_window" in v.alternatives, v.alternatives
    assert any("fixed at 9" in c for c in v.caveats), v.caveats


@pytest.mark.parametrize("k", [3, 5, 11, 13])
def test_other_windows_still_route_to_the_general_kernel(k):
    v = resolve("maxpool2d", shape=(1, 60, 80), kernel_size=k, stride=1,
                padding=k // 2)
    assert v.supported, v.reason
    assert v.app_name == "maxpool2d_window"


@pytest.mark.parametrize(
    "kw,expect",
    [
        ({"kernel_size": 7, "padding": 3}, "7x7"),
        ({"stride": 2, "kernel_size": 2, "padding": 0}, "stride 2"),
        ({"padding": 0}, "padding 0"),
    ],
)
def test_spec_refuses_anything_but_the_unrolled_window(kw, expect):
    params = {"shape": (1, 60, 80), "kernel_size": 9, "stride": 1, "padding": 4}
    params.update(kw)
    verdict = SPEC.check(**params)
    assert not verdict.ok
    assert expect in verdict.reason, verdict.reason


def test_constructor_guard_matches_spec():
    """An over-budget query must be refused by the constructor too."""
    params = {"shape": (256, 2000, 2000), "kernel_size": 9, "stride": 1,
              "padding": 4}
    assert not SPEC.check(**params).ok
    with pytest.raises(ValueError):
        MaxPool2dNms9App(inst_path="unused.bin", input_path="unused.bin",
                         channels=256, height=2000, width=2000)


def test_imem_footprint_fits_one_bank(inst):
    """96 words of 24 bytes. Unrolling K=11 would not fit, which is the limit."""
    assert inst["maxpool2d_nms9"].stat().st_size // 24 == 96
    assert inst["maxpool2d_nms9"].stat().st_size // 24 <= 128


# -- the K=7 twin ------------------------------------------------------------


@pytest.mark.parametrize("c,h,w", [(1, 7, 7), (1, 12, 40), (2, 6, 122), (1, 14, 260)])
def test_nms7_matches_numpy(inst, tmp_path, c, h, w):
    rng = np.random.default_rng(c * 41 + h * 7 + w)
    x = rng.standard_normal((c, h, w), dtype=np.float32)
    got, _ = _run(MaxPool2dNms7App, inst["maxpool2d_nms7"], tmp_path, x)
    assert np.array_equal(got.astype(np.float64), _reference(x, 7))


def test_nms7_identical_to_the_general_kernel_and_cheaper(inst, tmp_path):
    rng = np.random.default_rng(15)
    x = rng.standard_normal((1, 24, 300), dtype=np.float32)
    fast, fast_cyc = _run(MaxPool2dNms7App, inst["maxpool2d_nms7"], tmp_path, x)
    slow, slow_cyc = _run(MaxPool2dWindowApp, inst["maxpool2d_window"], tmp_path,
                          x, kernel_size=7)
    assert np.array_equal(fast, slow)
    assert fast_cyc < slow_cyc * 0.78, (fast_cyc, slow_cyc)


def test_nms7_every_row_of_the_window_is_read(inst, tmp_path):
    """Pins the three-slot rotation at an odd K, where dy%3 wraps differently."""
    for dy in range(7):
        x = np.full((1, 18, 130), -1.0, dtype=np.float32)
        x[0, 5 + dy, 60] = 5.0
        got, _ = _run(MaxPool2dNms7App, inst["maxpool2d_nms7"], tmp_path, x)
        assert np.array_equal(got.astype(np.float64), _reference(x, 7)), dy


def test_registry_prefers_the_specialised_kernel_at_k7():
    v = resolve("maxpool2d", shape=(1, 60, 80), kernel_size=7, stride=1, padding=3)
    assert v.supported, v.reason
    assert v.app_name == "maxpool2d_nms7"
    assert "maxpool2d_window" in v.alternatives, v.alternatives


def test_nms7_imem_footprint_fits_one_bank(inst):
    """64 words -- less than half a bank, since K=7 emits 49 taps not 81."""
    assert inst["maxpool2d_nms7"].stat().st_size // 24 == 64
