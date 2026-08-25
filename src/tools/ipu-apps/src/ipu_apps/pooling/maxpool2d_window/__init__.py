"""Stride-1 KxK windowed max-pool harness (FP32 wide-vector mode).

Computes, per channel, with ``K`` odd and ``P = K // 2``::

    out[c, y, x] = max over dy, dx in [0, K) of pad(in)[c, y+dy-P, x+dx-P]

so the output is the same ``H x W`` as the input and every output is the
maximum of the window *centred* on it. That is the pooling step of SuperPoint's
``simple_nms`` (``K = 2 * nms_radius + 1``, so ``K = 9`` at the default radius
4), and any other stride-1 local-maximum pool.

**This is the pool, not the whole NMS.** ``simple_nms`` then compares the pooled
map against the original (``scores == max_pool(scores)``), builds a boolean
mask, and iterates. The ISA has no vector compare and no boolean vector, so
those steps stay on the host; what runs here is the part that is expressible,
which is also the expensive part.

Why the taps are a run-time loop
--------------------------------
``K`` is a CR value read by two nested loops, so one assembled binary serves
every window size. Unrolling is not merely unnecessary here, it is impossible:
R_CYCLIC holds four 128-element slots, so the ``K`` rows a ``K x K`` window
needs cannot all be resident once ``K > 4``. The kernel streams one row at a
time through slot 0 and keeps a resident ``-FLT_MAX`` row in slot 3 to seed
each output tile's maximum.

Horizontal shifts still cost nothing: XMEM addressing is row-granular, but
``MULT.RC.*`` reads R_CYCLIC at an arbitrary element index, so ``dx`` is a
``+1`` step on the read index within the one loaded row.

Memory layout (all sizes in 128-element rows; one row = 512 bytes here):

    input   C planes, each (H + 2P) spatial rows x TPR rows
    seed    1 row of -FLT_MAX (resident in R_CYCLIC slot 3)
    output  C planes, each H spatial rows x TPR rows

**Halo tiling.** A spatial row is cut into tiles of ``TC = 128 - (K - 1)``
output columns, each stored as one 128-element row whose leading ``P`` and
trailing ``P`` elements are the horizontal halo::

    element e of tile t = input column (t*TC + e - P)

Output lane ``j`` (0..TC-1) is column ``t*TC + j``, and tap ``dx`` reads element
``j + dx``, so every tap of every valid lane is satisfied from inside the one
row. The largest element a valid lane reads is ``(TC-1) + (K-1) = 127``,
exactly the last element of the slot.

Columns outside the image, and the ``P`` border rows above and below each
plane, hold ``-FLT_MAX``. Unlike the halving kernel, that fill is genuinely
load-bearing here: a centred window at the image edge really does read outside
the image, and the identity of a maximum is what makes those reads harmless.

The window widens the tile's halo, so a large ``K`` costs lanes: ``K = 9``
leaves 120 usable columns of every 128.

Usage::

    from ipu_apps.pooling.maxpool2d_window import MaxPool2dWindowApp

    app = MaxPool2dWindowApp(
        inst_path="maxpool2d_window.bin",
        input_path="x.bin",       # C*H*W FP32, channel-major
        output_path="y.bin",      # C*H*W FP32
        channels=1, height=60, width=80, kernel_size=9,
    )
    state, cycles = app.run()
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ipu_emu.ipu_math import DType
from ipu_emu.ipu_state import IpuState, WideVectorArithmetic

from ipu_apps.base import IpuApp
from ipu_apps.kernel_registry import KernelSpec, no, yes
from ipu_apps.pooling._spec_support import (
    BASE_ROW,
    LANES,
    NEG_FILL,
    ROW_BYTES,
    WIDE_VECTOR_ONLY,
    geometry_refusal,
    headroom_caveat,
    lane_caveat,
    pool_query,
    xmem_refusal,
)

# The stride this app implements. Fixed by the tiling: input and output share
# one tile grid, which only holds when every input column is an output column.
STRIDE = 1
# One input XMEM row per output XMEM row, and no guard tile: the widened halo
# already supplies every element a valid lane reads.
IN_TILES_PER_OUT_TILE = 1
GUARD_TILES = 0
# The resident -FLT_MAX row that seeds each output tile's running maximum.
SCRATCH_ROWS = 1
# R_CYCLIC slot 3 (elements 384..511) holds that seed row for the whole run.
SEED_SLOT = 3 * LANES


def out_tile_cols(kernel: int) -> int:
    """Usable output columns per 128-element row for a ``kernel``-wide window.

    ``K - 1`` elements of every row are the horizontal halo: ``P`` at each end
    for an odd ``K``. Derived rather than stored, since it is a pure function
    of the window and both the harness and the spec need it.
    """
    return LANES - (kernel - 1)


class MaxPool2dWindowApp(IpuApp):
    """Stride-1 KxK max-pool over a ``(C, H, W)`` activation, output ``(C, H, W)``.

    Args:
        inst_path:   Path to the assembled instruction binary.
        input_path:  Activation, ``C*H*W`` FP32, channel-major.
        output_path: Optional path for the ``C*H*W`` FP32 result.
        channels:    C.
        height:      H.
        width:       W.
        kernel_size: K, odd. Padding is always ``K // 2`` -- that is what makes
                     the window centred and the output the same size as the
                     input, which this kernel's shared tile grid requires.
    """

    def __init__(
        self,
        *,
        channels: int,
        height: int,
        width: int,
        kernel_size: int,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.input_path = Path(self.input_path)

        self.channels = int(channels)
        self.height = int(height)
        self.width = int(width)
        self.kernel_size = int(kernel_size)
        self.padding = self.kernel_size // 2

        # Delegate to the registry declaration rather than restating the
        # bounds: SPEC.supports is the single source of truth for this kernel's
        # domain, including the XMEM budget.
        shape = (self.channels, self.height, self.width)
        SPEC.guard(
            shape=shape,
            kernel_size=self.kernel_size,
            stride=STRIDE,
            padding=self.padding,
        )
        self.query = pool_query(
            shape,
            kernel_size=self.kernel_size,
            stride=STRIDE,
            padding=self.padding,
        )
        self._layout()

    def _layout(self) -> None:
        """Place the three regions back-to-back, in rows and in bytes.

        Row numbers drive the CR registers (the .asm's XMEM operands are rows);
        the byte addresses drive the direct ``state.xmem`` calls in setup and
        teardown, which bypass row translation.
        """
        self.tile_cols = out_tile_cols(self.kernel_size)
        lay = self.query.layout(
            out_tile_cols=self.tile_cols,
            in_tiles_per_out_tile=IN_TILES_PER_OUT_TILE,
            guard_tiles=GUARD_TILES,
            pad_rows=self.padding,
            scratch_rows=SCRATCH_ROWS,
        )
        self.geometry = lay
        self.tiles_per_row = lay.out_tiles_per_row
        self.padded_height = lay.padded_height
        self.in_plane_stride = lay.in_plane_stride
        self.output_rows = lay.output_rows

        self.input_base_row = BASE_ROW
        self.seed_base_row = self.input_base_row + lay.input_rows
        self.output_base_row = self.seed_base_row + SCRATCH_ROWS

        self.input_base = self.input_base_row * ROW_BYTES
        self.seed_base = self.seed_base_row * ROW_BYTES
        self.output_base = self.output_base_row * ROW_BYTES

    # -- wide-vector FP32 state ---------------------------------------------

    @staticmethod
    def make_state() -> IpuState:
        """Build the FP32 wide-vector state this app requires.

        ``wide_vector_quantize_output=False`` keeps elements 4-byte FP32
        through AAQ, so ACTIVATE.QUANTIZE is a true pass-through (the INT8 clamp
        is skipped entirely) and STR_POST_AAQ_REG drains the full 512 bytes.
        """
        state = IpuState(
            wide_vector_debug=True,
            wide_vector_arithmetic=WideVectorArithmetic.FP32,
            wide_vector_quantize_output=False,
        )
        # dtype is otherwise unused on the FP32 wide path, but several helpers
        # branch on it; INT8 matches the existing wide-vector apps.
        state.dtype = DType.INT8
        return state

    # -- host-side data marshalling -----------------------------------------

    def _read_f32(self, path: Path, count: int, what: str) -> np.ndarray:
        data = np.fromfile(path, dtype="<f4")
        if data.size != count:
            raise ValueError(
                f"{what} file {path} holds {data.size} FP32 values; this pool "
                f"needs {count}"
            )
        return data

    def _pack_input(self, x: np.ndarray) -> np.ndarray:
        """Lay out ``(C, H, W)`` as halo-tiled, vertically bordered planes.

        Returns ``(C, H + 2P, TPR, 128)``. Everything not covered by a real
        image column stays at ``-FLT_MAX``: the P border rows top and bottom,
        the P halo elements at each end of a row, and the columns past W in the
        last tile.
        """
        c, h, w = x.shape
        p, tpr = self.padding, self.tiles_per_row
        planes = np.full(
            (c, h + 2 * p, tpr, LANES), NEG_FILL, dtype="<f4"
        )
        for t in range(tpr):
            # Element e of tile t is input column t*TC + e - P, so the slice
            # starts P columns early and runs P columns past the tile.
            lo = t * self.tile_cols - p
            hi = lo + LANES
            src_lo, src_hi = max(lo, 0), min(hi, w)
            if src_hi > src_lo:
                planes[:, p : p + h, t, src_lo - lo : src_hi - lo] = x[
                    :, :, src_lo:src_hi
                ]
        return planes

    def setup(self, state: "IpuState") -> None:
        c, h, w = self.channels, self.height, self.width
        x = self._read_f32(self.input_path, c * h * w, "input").reshape(c, h, w)
        state.xmem.write_address(self.input_base, self._pack_input(x).tobytes())

        # The seed row: one full row of the maximum's identity, loaded into
        # R_CYCLIC slot 3 at startup and never reloaded.
        seed = np.full(LANES, NEG_FILL, dtype="<f4")
        state.xmem.write_address(self.seed_base, seed.tobytes())

        # CR0 and CR1 are READ-ONLY: CR0 == 0 always, CR1 == 1 always. Both are
        # exploited directly -- CR0 as the zero source and CR1 as both the 1.0
        # scalar for the identity multiply and every +1 increment. All writable
        # CRs below hold row numbers, row strides or loop bounds (.asm XMEM
        # operands are rows, not bytes -- see issue #179).
        state.regfile.set_cr(2, self.input_base_row)
        state.regfile.set_cr(3, self.output_base_row)
        state.regfile.set_cr(4, self.seed_base_row)
        # CR5 is both the tile-loop bound and the row step between dy taps:
        # consecutive padded rows of one tile column are TPR rows apart.
        state.regfile.set_cr(5, self.tiles_per_row)
        state.regfile.set_cr(6, self.in_plane_stride)
        state.regfile.set_cr(7, h)
        state.regfile.set_cr(8, c)
        state.regfile.set_cr(9, self.kernel_size)
        state.regfile.set_cr(10, SEED_SLOT)
        # CR11 = K-1: the dx loop's BLT reads its counter pre-increment (it is
        # incremented in the same word), so its bound is one less than the dy
        # loop's, whose BLT sits several words after the increment.
        state.regfile.set_cr(11, self.kernel_size - 1)

        # MULT.*/ACTIVATE.QUANTIZE read the active-element count from the named
        # dstructure CR's valid_elements field. The asm names CR15 throughout,
        # so set CR15.valid_elements = 128 to process the full FP32 row.
        state.set_cr_dstructure(valid_elements=LANES)

    def teardown(self, state: "IpuState") -> None:
        if self.output_path is None:
            return
        raw = state.xmem.read_address(self.output_base, self.output_rows * ROW_BYTES)
        planes = np.frombuffer(raw, dtype="<f4").reshape(
            self.channels, self.height, self.tiles_per_row, LANES
        )
        # Only the first TC lanes of each tile are real output columns; the rest
        # read past their slot. Concatenating the usable lanes and trimming to W
        # is what makes the output file a dense (C, H, W) array matching the
        # input's element order.
        cols = planes[:, :, :, : self.tile_cols].reshape(
            self.channels, self.height, self.tiles_per_row * self.tile_cols
        )
        np.ascontiguousarray(cols[:, :, : self.width]).tofile(self.output_path)

    def run(self, **kwargs):
        # Always run on the FP32 wide-vector state unless the caller supplied one.
        kwargs.setdefault("state", self.make_state())
        return super().run(**kwargs)


# -- registry declaration ---------------------------------------------------


def _query(params):
    return pool_query(
        params["shape"],
        kernel_size=params["kernel_size"],
        stride=params["stride"],
        padding=params["padding"],
    )


def _geometry(params):
    q = _query(params)
    return q.layout(
        out_tile_cols=out_tile_cols(q.kernel),
        in_tiles_per_out_tile=IN_TILES_PER_OUT_TILE,
        guard_tiles=GUARD_TILES,
        pad_rows=q.padding,
        scratch_rows=SCRATCH_ROWS,
    )


def _supports(**params):
    q = _query(params)
    bad = geometry_refusal(q)
    if bad:
        return no(bad)
    if q.stride != STRIDE:
        return no(
            f"pools with stride {STRIDE}; this query asks for stride {q.stride}. "
            f"A stride-2 pool decimates its result and has its own kernel "
            f"(maxpool2d_halve)."
        )
    if q.kernel % 2 == 0:
        return no(
            f"pools an odd-sized centred window; this query asks for an even "
            f"{q.kernel}x{q.kernel}, which has no centre and cannot keep the "
            f"output the same size as the input"
        )
    if q.padding != q.kernel // 2:
        return no(
            f"pads by exactly kernel//2 = {q.kernel // 2} to keep the output "
            f"H x W; this query asks for padding {q.padding}, which would make "
            f"the output {q.out_height} x {q.out_width}. Input and output share "
            f"one tile grid here, so a different extent is not expressible."
        )
    if out_tile_cols(q.kernel) < 1:
        return no(
            f"a {q.kernel}x{q.kernel} window needs {q.kernel - 1} halo elements "
            f"of every {LANES}-element row, leaving no usable output columns; "
            f"the largest window this tiling supports is {LANES - 1}x{LANES - 1}"
        )
    budget = xmem_refusal(_geometry(params))
    if budget:
        return no(budget)
    return yes()


def _build(**params):
    q = _query(params)
    return {
        "channels": q.channels,
        "height": q.height,
        "width": q.width,
        "kernel_size": q.kernel,
    }


def _explain(**params):
    lay = _geometry(params)
    q = lay.query
    return (
        f"a stride-1 {q.kernel}x{q.kernel} centred max; the {q.kernel} rows are "
        f"streamed through one R_CYCLIC slot and the {q.kernel} horizontal taps "
        f"are a +1 element step, so the output stays {q.height}x{q.width} in "
        f"{lay.out_tiles_per_row} row(s) per spatial row of "
        f"{lay.out_tile_cols} usable columns"
    )


def _caveats(**params):
    lay = _geometry(params)
    notes = [WIDE_VECTOR_ONLY]
    lanes = lane_caveat(lay)
    if lanes:
        notes.append(lanes)
    notes.append(headroom_caveat(lay))
    return tuple(notes)


SPEC = KernelSpec(
    name="maxpool2d_window",
    op="maxpool2d",
    variant="window",
    app_class=MaxPool2dWindowApp,
    asm="maxpool2d_window.asm",
    # The window parameters are required rather than defaulted on purpose: a
    # pooling spec that silently assumed stride 1 would answer for an operation
    # no kernel here computes.
    requires=("shape", "kernel_size", "stride", "padding"),
    tags=("fp32-wide", "windowed"),
    supports=_supports,
    build=_build,
    explain=_explain,
    caveats=_caveats,
    bundle=lambda **params: _query(params).bundle,
    # Disjoint from maxpool2d_halve -- that kernel is stride 2, this one stride
    # 1 -- so cost never actually decides between them.
    cost=lambda **params: 0.0,
)
