"""``cell_nms`` as a composition of three registered kernels.

The hand-written ``kernels/superpoint_superglue/cell_nms.asm`` computes, over
one detector cell's 64 sub-grid channels::

    peak  = max_c p_c
    a     = SUM_c softmax(T*p)_c * (c // 8)      row sub-coordinate
    b     = SUM_c softmax(T*p)_c * (c %  8)      column sub-coordinate

It is not ported as a fourth kernel, because every piece of it already exists
and a monolithic port would carry a second copy of the softmax:

    peak        channel_peak        max down the channel planes
    softmax     softmax_columns     softmax down the channel axis of (64, N)
    a and b     conv1x1             a 1x1 convolution IS a per-channel weighted
                                    sum, so the two coordinate dots are two
                                    output channels with W[0,c] = c//8 and
                                    W[1,c] = c%8

Two further differences from the original are improvements, not compromises:
it processes **many cells per launch** (cells in lanes) rather than one, and it
returns real XMEM planes rather than the ``aaq0..aaq3`` scalar registers the
current ISA no longer has.

This file exists so that claim is tested rather than asserted in prose. If any
of the three kernels' layouts drift apart, the composition breaks here.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from ipu_as.lark_tree import assemble_to_bin_file
from ipu_apps.convolutions_universal.conv.conv1x1 import Conv1x1App
from ipu_apps.detect.channel_peak import ChannelPeakApp
from ipu_apps.kernel_registry import resolve

SRC = Path(__file__).resolve().parents[1] / "src/ipu_apps"
ASM = {
    "channel_peak": SRC / "detect/channel_peak/channel_peak.asm",
    "conv1x1": SRC / "convolutions_universal/conv/conv1x1/conv1x1.asm",
    # The softmax step is routed, not hard-coded: which column kernel wins
    # depends on the cell count, and picking one by hand is exactly the drift
    # the registry exists to stop. Below 65 cells several whole rows fit in one
    # vector and softmax_columns_packed takes over.
    "softmax_columns": SRC / "softmax/softmax_columns/softmax_columns.asm",
    "softmax_columns_packed": (
        SRC / "softmax/softmax_columns_packed/softmax_columns_packed.asm"
    ),
}

CHANNELS = 64          # the 8x8 sub-grid of one detector cell
GRID = 8
TEMPERATURE = 64.0     # the reference's T; large T -> the soft argmax hardens
TOL = 2e-4

# cell_nms_ref is a BASE-2 softmax -- 2^(T*p) -- while the softmax kernels
# compute the natural one via the base-2 reformulation 2^(log2(e)*x). Feeding
# them T*p would therefore run at an effective temperature of T/ln2, about 1.44x
# too sharp. Scaling the logits by ln2 makes the two exactly equal.
LOG2 = float(np.log(2.0))


def _cell_nms_reference(p: np.ndarray, t: float):
    """(peak, a, b) per cell, mirroring ``reference.py: cell_nms_ref``.

    ``p`` is ``(64, N)``: one column per cell. The reference takes a single
    cell, so this is it applied column-wise.
    """
    x = p.astype(np.float64)
    peak = x.max(axis=0)
    e = np.exp2(t * (x - peak))
    s = e / e.sum(axis=0, keepdims=True)
    rows = np.arange(CHANNELS) // GRID
    cols = np.arange(CHANNELS) % GRID
    return peak, s.T @ rows, s.T @ cols


@pytest.fixture(scope="module")
def inst_files():
    with tempfile.TemporaryDirectory() as tmp:
        built = {}
        for name, path in ASM.items():
            out = Path(tmp) / f"{name}.bin"
            assemble_to_bin_file(path.read_text(), str(out))
            built[name] = out
        yield built


def _peak(inst, tmp_path, p: np.ndarray) -> np.ndarray:
    """Step 1: max down the channel planes."""
    c, n = p.shape
    xp, conf = tmp_path / "p_in.bin", tmp_path / "p_conf.bin"
    p.astype("<f4").tofile(xp)
    ChannelPeakApp(
        inst_path=inst["channel_peak"],
        input_path=xp,
        confidence_path=conf,
        channels=c,
        cells=n,
        # The gate is unused here; a threshold below every score keeps it inert.
        threshold=float(p.min()) - 1.0,
    ).run(max_cycles=200_000_000)
    return np.frombuffer(conf.read_bytes(), dtype="<f4")


def _softmax(inst, tmp_path, p: np.ndarray) -> np.ndarray:
    """Step 2: softmax down the channel axis, i.e. down each column.

    ``p`` is the raw per-channel score; the temperature and the base conversion
    are folded into the logits here, so this returns exactly the ``s`` of
    ``cell_nms_ref``.
    """
    logits = (np.float32(TEMPERATURE * LOG2) * p).astype(np.float32)
    rows, width = logits.shape
    verdict = resolve("softmax", shape=(rows, width), dim=0)
    assert verdict.supported, verdict.reason
    xp, op = tmp_path / "s_in.bin", tmp_path / "s_out.bin"
    logits.tofile(xp)
    verdict.kernel.app_class(
        inst_path=inst[verdict.kernel.name],
        input_path=xp,
        output_path=op,
        **verdict.kwargs,
    ).run(max_cycles=200_000_000)
    return np.frombuffer(op.read_bytes(), dtype="<f4").reshape(rows, width)


def _coord_dots(inst, tmp_path, probs: np.ndarray) -> np.ndarray:
    """Step 3: the two coordinate dots, as a 1x1 convolution with Cout=2."""
    c, n = probs.shape
    weights = np.zeros((2, c, 1, 1), dtype=np.float32)
    weights[0, :, 0, 0] = np.arange(c) // GRID
    weights[1, :, 0, 0] = np.arange(c) % GRID
    xp, wp, op = (tmp_path / f"c_{k}.bin" for k in ("in", "w", "out"))
    probs.astype("<f4").tofile(xp)
    weights.tofile(wp)
    Conv1x1App(
        inst_path=inst["conv1x1"],
        input_path=xp,
        weight_path=wp,
        output_path=op,
        in_channels=c,
        out_channels=2,
        height=1,
        width=n,
        bias=False,
    ).run(max_cycles=200_000_000)
    return np.frombuffer(op.read_bytes(), dtype="<f4").reshape(2, n)


@pytest.mark.parametrize("cells", [1, 40, 128, 200])
def test_composition_matches_the_cell_nms_reference(inst_files, tmp_path, cells):
    rng = np.random.default_rng(cells * 7 + 1)
    p = rng.uniform(0.0, 1.0, size=(CHANNELS, cells)).astype(np.float32)

    peak = _peak(inst_files, tmp_path, p)
    probs = _softmax(inst_files, tmp_path, p)
    coords = _coord_dots(inst_files, tmp_path, probs)

    ref_peak, ref_a, ref_b = _cell_nms_reference(p, TEMPERATURE)
    assert np.array_equal(peak.astype(np.float64), ref_peak)
    assert np.abs(coords[0] - ref_a).max() < TOL
    assert np.abs(coords[1] - ref_b).max() < TOL


def test_the_soft_argmax_hardens_to_the_true_sub_position(inst_files, tmp_path):
    """As T grows, (a, b) approaches the integer argmax coordinates.

    That is the property the detector actually uses: the soft argmax is a
    stand-in for the hard one, so it has to land on the right sub-pixel.
    """
    cells = 30
    rng = np.random.default_rng(4)
    p = rng.uniform(0.0, 1.0, size=(CHANNELS, cells)).astype(np.float32)
    # Make each cell's winner unambiguous so the soft argmax has a clear target.
    winners = rng.integers(0, CHANNELS, size=cells)
    p[winners, np.arange(cells)] += np.float32(2.0)

    probs = _softmax(inst_files, tmp_path, p)
    coords = _coord_dots(inst_files, tmp_path, probs)

    assert np.abs(coords[0] - winners // GRID).max() < 1e-3
    assert np.abs(coords[1] - winners % GRID).max() < 1e-3


def test_the_probabilities_the_composition_passes_along_sum_to_one(inst_files, tmp_path):
    """Guards the middle step: a broken softmax would still give plausible dots."""
    rng = np.random.default_rng(6)
    p = rng.uniform(0.0, 1.0, size=(CHANNELS, 50)).astype(np.float32)
    probs = _softmax(inst_files, tmp_path, p)
    assert np.abs(probs.astype(np.float64).sum(axis=0) - 1.0).max() < 1e-5


def test_the_detector_softmax_needs_no_kernel_of_its_own():
    """SuperPoint's 65-channel softmax is softmax_columns' dim=0 case.

    The layer map claims this, and it is the one row there that asserts an
    operation is covered by a kernel written for a different family. A 60x80
    cell grid gives 4800 columns.
    """
    verdict = resolve("softmax", shape=(65, 4800), dim=0)
    assert verdict.supported, verdict.reason
    assert verdict.app_name == "softmax_columns"
