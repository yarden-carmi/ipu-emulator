"""Numpy-parity tests for the depth-to-space (pixel shuffle) app."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from ipu_as.lark_tree import assemble_to_bin_file
from ipu_apps.kernel_registry import MalformedQuery, resolve
from ipu_apps.reshape.depth_to_space import SPEC, DepthToSpaceApp

ASM_PATH = (
    Path(__file__).resolve().parents[1]
    / "src/ipu_apps/reshape/depth_to_space"
    / "depth_to_space.asm"
)


def _reference(x: np.ndarray, r: int) -> np.ndarray:
    """out[r*h+a, r*w+b] = x[r*a+b, h, w] -- the r=... case of nn.PixelShuffle.

    Written as the reshape/transpose torch itself performs, so it carries the
    channel-ordering convention rather than inventing one.
    """
    c, h, w = x.shape
    assert c == r * r
    return x.reshape(r, r, h, w).transpose(2, 0, 3, 1).reshape(h * r, w * r)


def _naive_reference(x: np.ndarray, r: int) -> np.ndarray:
    """The same thing as an explicit index loop, with no reshaping at all.

    Mirrors ``depth_to_space_ref`` in
    ``kernels/superpoint_superglue/reference.py``. An independent anchor, so
    ``_reference``'s transpose order cannot drift into agreeing with the kernel
    for the wrong reason.
    """
    c, h, w = x.shape
    out = np.zeros((h * r, w * r), dtype=np.float64)
    for a in range(r):
        for b in range(r):
            for y in range(h):
                for col in range(w):
                    out[r * y + a][r * col + b] = float(x[r * a + b][y][col])
    return out


@pytest.fixture(scope="module")
def inst_file():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "depth_to_space.bin"
        assemble_to_bin_file(ASM_PATH.read_text(), str(path))
        yield path


def _run(inst_file, tmp_path, x: np.ndarray, r: int) -> np.ndarray:
    c, h, w = x.shape
    xp, op = tmp_path / "x.bin", tmp_path / "y.bin"
    x.astype("<f4").tofile(xp)
    app = DepthToSpaceApp(
        inst_path=inst_file,
        input_path=xp,
        output_path=op,
        channels=c,
        height=h,
        width=w,
        upscale_factor=r,
    )
    app.run(max_cycles=400_000_000)
    return np.frombuffer(op.read_bytes(), dtype="<f4").reshape(h * r, w * r)


@pytest.mark.parametrize(
    "r,h,w",
    [
        (1, 3, 5),      # degenerate: a factor-1 shuffle is a copy
        (2, 1, 1),      # smallest real interleave
        (2, 3, 5),      # tiny
        (2, 2, 128),    # exactly one full input tile
        (2, 2, 130),    # spills to a second input tile
        (4, 3, 7),      # 16 planes
        (8, 2, 3),      # the SuperPoint factor, tiny
        (8, 4, 16),     # the SuperPoint factor, a real block
        (8, 3, 130),    # two input tiles at r=8 -> 16 output tiles per row
    ],
)
def test_matches_numpy(inst_file, tmp_path, r, h, w):
    rng = np.random.default_rng(r * 131 + h * 17 + w)
    x = rng.standard_normal((r * r, h, w), dtype=np.float32)
    out = _run(inst_file, tmp_path, x, r)
    ref = _reference(x, r)
    assert out.shape == ref.shape
    # Values are moved verbatim, so this is exact.
    assert np.array_equal(out.astype(np.float64), ref)


def test_numpy_reference_matches_the_naive_index_loop(inst_file, tmp_path):
    """Anchor ``_reference``'s transpose order, then the kernel."""
    rng = np.random.default_rng(99)
    x = rng.standard_normal((16, 3, 6), dtype=np.float32)
    ref = _reference(x, 4)
    assert np.array_equal(ref, _naive_reference(x, 4))
    assert np.array_equal(_run(inst_file, tmp_path, x, 4).astype(np.float64), ref)


def test_every_element_lands_exactly_once(inst_file, tmp_path):
    """Give every input element a unique value and check the output is a permutation.

    A shuffle that dropped, duplicated or zeroed a lane still looks plausible on
    random data -- distinct values make every one of the 128 destination indices
    individually load-bearing.
    """
    r, h, w = 8, 3, 20
    x = np.arange(r * r * h * w, dtype=np.float32).reshape(r * r, h, w)
    out = _run(inst_file, tmp_path, x, r)
    assert np.array_equal(np.sort(out.ravel()), np.sort(x.ravel()))
    assert np.array_equal(out.astype(np.float64), _reference(x, r))


def test_plane_lands_on_its_own_sub_position(inst_file, tmp_path):
    """One plane at a time: plane r*a+b must fill exactly the (a, b) sub-lattice.

    This is the test that pins the *mapping*, not just the multiset. A
    transposed convention (b*r+a) would pass the permutation test above.
    """
    r, h, w = 4, 3, 9
    for a in range(r):
        for b in range(r):
            x = np.zeros((r * r, h, w), dtype=np.float32)
            x[r * a + b] = 1.0
            out = _run(inst_file, tmp_path, x, r)
            expected = np.zeros((h * r, w * r), dtype=np.float32)
            expected[a::r, b::r] = 1.0
            assert np.array_equal(out, expected), (a, b)


def test_padding_lanes_do_not_reach_real_columns(inst_file, tmp_path):
    """A width that does not fill its input tile must still be exact.

    The zero columns past W fan out to output lanes past W*r, and trimming them
    on read-back is the whole reason the output file is dense.
    """
    r, h, w = 8, 2, 80   # SuperPoint's cell grid: 80 of 128 input lanes used
    rng = np.random.default_rng(21)
    x = rng.standard_normal((r * r, h, w), dtype=np.float32)
    out = _run(inst_file, tmp_path, x, r)
    assert out.shape == (h * r, w * r)
    assert np.array_equal(out.astype(np.float64), _reference(x, r))


def test_output_file_layout_is_dense(inst_file, tmp_path):
    """The output file is exactly (H*r)*(W*r) FP32 in reshape order."""
    r, h, w = 8, 3, 40
    rng = np.random.default_rng(5)
    x = rng.standard_normal((r * r, h, w), dtype=np.float32)
    xp, op = tmp_path / "x.bin", tmp_path / "y.bin"
    x.astype("<f4").tofile(xp)
    app = DepthToSpaceApp(
        inst_path=inst_file, input_path=xp, output_path=op,
        channels=r * r, height=h, width=w, upscale_factor=r,
    )
    app.run(max_cycles=400_000_000)
    assert op.stat().st_size == (h * r) * (w * r) * 4
    out = np.frombuffer(op.read_bytes(), dtype="<f4").reshape(h * r, w * r)
    assert np.array_equal(out.astype(np.float64), _reference(x, r))


def test_index_arithmetic_is_what_the_kernel_assumes():
    """The three derived counts the ACC.RESHAPE schedule is built on."""
    app = DepthToSpaceApp(
        inst_path="unused.bin", input_path="unused.bin",
        channels=64, height=60, width=80, upscale_factor=8,
    )
    assert app.elements_per_plane == 16        # 128 / r
    assert app.reshapes_per_plane == 2         # 16 / 8 elements per instruction
    assert app.in_tiles_per_row == 1
    assert app.out_tiles_per_row == 8          # each input tile fans out to r
    # The destination drift the -127 rewind undoes is always exactly 128.
    assert app.reshapes_per_plane * 8 * app.r == 128


def test_the_superpoint_heatmap_fits_in_one_launch():
    """60x80x64 -> 480x640 is the real detector-head shuffle.

    Worth pinning: unlike every convolution in the network, this runs unbanded.
    """
    verdict = resolve("depth_to_space", shape=(64, 60, 80), upscale_factor=8)
    assert verdict.supported, verdict.reason
    assert verdict.shapes["output"] == (1, 480, 640)
    app = DepthToSpaceApp(
        inst_path="unused.bin", input_path="unused.bin", **verdict.kwargs
    )
    assert app.query.total_rows == 64 + 64 * 60 + 480 * 8


# -- registry conformance ---------------------------------------------------


def test_registry_resolves_to_this_kernel(inst_file, tmp_path):
    r, h, w = 8, 2, 10
    verdict = resolve("depth_to_space", shape=(r * r, h, w), upscale_factor=r)
    assert verdict.supported, verdict.reason
    assert verdict.app_name == "depth_to_space"
    assert verdict.shapes["output"] == (1, h * r, w * r)

    rng = np.random.default_rng(17)
    x = rng.standard_normal((r * r, h, w), dtype=np.float32)
    xp, op = tmp_path / "x.bin", tmp_path / "y.bin"
    x.astype("<f4").tofile(xp)
    app = verdict.kernel.app_class(
        inst_path=inst_file, input_path=xp, output_path=op, **verdict.kwargs
    )
    app.run(max_cycles=400_000_000)
    out = np.frombuffer(op.read_bytes(), dtype="<f4").reshape(h * r, w * r)
    assert np.array_equal(out.astype(np.float64), _reference(x, r))


def test_idle_output_lanes_are_disclosed():
    """At r=8 a 48-lane input shortfall becomes 384 idle output lanes."""
    verdict = resolve("depth_to_space", shape=(64, 60, 80), upscale_factor=8)
    assert verdict.supported, verdict.reason
    assert any("fans out" in c for c in verdict.caveats), verdict.caveats


@pytest.mark.parametrize(
    "shape,r,expected",
    [
        ((64, 4, 4), 3, "not a multiple of 9"),
        ((128, 4, 4), 8, "would emit 2"),
        ((1024, 2, 2), 32, "not a whole number"),
        ((256, 2, 2), 16, "destination step"),
        ((2, 4, 8, 8), 2, "batch of 2"),
        ((0, 4, 4), 2, "channels (0)"),
    ],
)
def test_router_refuses_what_no_kernel_implements(shape, r, expected):
    verdict = resolve("depth_to_space", shape=shape, upscale_factor=r)
    assert not verdict.supported
    assert expected in verdict.reason, verdict.reason


@pytest.mark.parametrize(
    "shape,r,ctor",
    [
        # 64 planes of 480x640: the shuffled output alone is 30720 rows.
        ((64, 480, 640), 8,
         {"channels": 64, "height": 480, "width": 640, "upscale_factor": 8}),
        ((64, 4, 4), 3, {"channels": 64, "height": 4, "width": 4, "upscale_factor": 3}),
        ((128, 4, 4), 8,
         {"channels": 128, "height": 4, "width": 4, "upscale_factor": 8}),
        ((0, 4, 4), 2, {"channels": 0, "height": 4, "width": 4, "upscale_factor": 2}),
        ((4, 0, 4), 2, {"channels": 4, "height": 0, "width": 4, "upscale_factor": 2}),
    ],
)
def test_constructor_guard_matches_spec(shape, r, ctor):
    """Whatever ``supports`` refuses, the constructor must refuse too."""
    assert not SPEC.check(shape=shape, upscale_factor=r).ok
    assert not resolve("depth_to_space", shape=shape, upscale_factor=r).supported

    with pytest.raises(ValueError):
        DepthToSpaceApp(inst_path="unused.bin", input_path="unused.bin", **ctor)


@pytest.mark.parametrize(
    "shape,r",
    [
        ((8, 8), 2),
        ((64, 4, 4), 0),
        ((64, 4, 4), -2),
    ],
)
def test_malformed_queries_propagate(shape, r):
    with pytest.raises(MalformedQuery):
        resolve("depth_to_space", shape=shape, upscale_factor=r)
