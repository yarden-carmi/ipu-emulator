"""9x9 stride-1 centred max-pool, fully unrolled (FP32 wide-vector mode).

The bespoke twin of :mod:`~ipu_apps.pooling.maxpool2d_window`, fixed at
``K = 9`` -- SuperPoint's ``simple_nms`` window at its default
``nms_radius = 4``. Identical result, 24% fewer cycles.

Where the saving comes from
---------------------------
A run-time ``K`` forces a uniform loop body, and three costs follow from that
which unrolling simply removes:

======================  ==========================  ==================
cost                    general kernel              unrolled here
======================  ==========================  ==================
row load latency        1 dead word per row (9)     folded into the taps
accumulator seed        resident -FLT_MAX row + 1   tap (0,0) is
                        word per tile               ``ACC.MAX.FIRST``
one-past prefetch       needs a guard row           last two rows skip
                                                    their load
======================  ==========================  ==================

    per output tile   K^2 + 3K + 5 = 113   ->   K^2 + 5 = 86 words
    480x640, 1 plane        326,887        ->   ~249,000 cycles

The assembled program is 96 words, inside the 128-word IMEM bank.

Three rotating R_CYCLIC slots
-----------------------------
Row ``dy`` is read from slot ``dy % 3`` while row ``dy + 2`` loads into slot
``(dy + 2) % 3`` -- the slot holding row ``dy - 1``, whose taps are finished.
**Two slots cannot do this**: the one not being read holds the row needed next.
Rows 0 and 1 are preloaded before the tap stream.

Everything else -- the halo tiling (``TC = 120``), the ``-FLT_MAX`` border, the
memory layout -- is ``maxpool2d_window``'s, unchanged, so the two are
interchangeable and the registry picks this one on cost at ``K = 9``.

Usage::

    from ipu_apps.pooling.maxpool2d_nms9 import MaxPool2dNms9App

    app = MaxPool2dNms9App(
        inst_path="maxpool2d_nms9.bin",
        input_path="scores.bin",   # C*H*W FP32
        output_path="pooled.bin",  # C*H*W FP32
        channels=1, height=480, width=640,
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

# Fixed by the unroll: every tap is emitted, so K is not a run-time value.
KERNEL_SIZE = 9
PADDING = KERNEL_SIZE // 2
STRIDE = 1
# K-1 elements of every 128-element row are the horizontal halo.
TILE_COLS = LANES - (KERNEL_SIZE - 1)
# No seed row: tap (0,0) carries ACC.MAX.FIRST, so nothing resident is needed.
SCRATCH_ROWS = 0
IN_TILES_PER_OUT_TILE = 1
GUARD_TILES = 0


class MaxPool2dNms9App(IpuApp):
    """9x9 stride-1 max over a ``(C, H, W)`` activation, output ``(C, H, W)``.

    Args:
        inst_path:   Path to the assembled instruction binary.
        input_path:  Activation, ``C*H*W`` FP32, channel-major.
        output_path: Optional path for the ``C*H*W`` FP32 result.
        channels:    C.
        height:      H.
        width:       W.
    """

    def __init__(self, *, channels: int, height: int, width: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.input_path = Path(self.input_path)
        self.channels = int(channels)
        self.height = int(height)
        self.width = int(width)
        self.kernel_size = KERNEL_SIZE
        self.padding = PADDING

        shape = (self.channels, self.height, self.width)
        SPEC.guard(shape=shape, kernel_size=KERNEL_SIZE, stride=STRIDE,
                   padding=PADDING)
        self.query = pool_query(shape, kernel_size=KERNEL_SIZE, stride=STRIDE,
                                padding=PADDING)
        self._layout()

    def _layout(self) -> None:
        lay = self.query.layout(
            out_tile_cols=TILE_COLS,
            in_tiles_per_out_tile=IN_TILES_PER_OUT_TILE,
            guard_tiles=GUARD_TILES,
            pad_rows=PADDING,
            scratch_rows=SCRATCH_ROWS,
        )
        self.geometry = lay
        self.tile_cols = TILE_COLS
        self.tiles_per_row = lay.out_tiles_per_row
        self.padded_height = lay.padded_height
        self.in_plane_stride = lay.in_plane_stride
        self.output_rows = lay.output_rows

        self.input_base_row = BASE_ROW
        self.output_base_row = self.input_base_row + lay.input_rows
        self.input_base = self.input_base_row * ROW_BYTES
        self.output_base = self.output_base_row * ROW_BYTES

    @staticmethod
    def make_state() -> IpuState:
        """Build the FP32 wide-vector state this app requires."""
        state = IpuState(
            wide_vector_debug=True,
            wide_vector_arithmetic=WideVectorArithmetic.FP32,
            wide_vector_quantize_output=False,
        )
        state.dtype = DType.INT8
        return state

    def _pack_input(self, x: np.ndarray) -> np.ndarray:
        """Halo-tiled, vertically bordered planes -- identical to maxpool2d_window."""
        c, h, w = x.shape
        p, tpr = self.padding, self.tiles_per_row
        planes = np.full((c, h + 2 * p, tpr, LANES), NEG_FILL, dtype="<f4")
        for t in range(tpr):
            lo = t * TILE_COLS - p
            hi = lo + LANES
            src_lo, src_hi = max(lo, 0), min(hi, w)
            if src_hi > src_lo:
                planes[:, p : p + h, t, src_lo - lo : src_hi - lo] = x[
                    :, :, src_lo:src_hi
                ]
        return planes

    def setup(self, state: "IpuState") -> None:
        c, h, w = self.channels, self.height, self.width
        data = np.fromfile(self.input_path, dtype="<f4")
        if data.size != c * h * w:
            raise ValueError(
                f"input file {self.input_path} holds {data.size} FP32 values; "
                f"this pool needs {c * h * w}"
            )
        state.xmem.write_address(
            self.input_base, self._pack_input(data.reshape(c, h, w)).tobytes()
        )

        state.regfile.set_cr(2, self.input_base_row)
        state.regfile.set_cr(3, self.output_base_row)
        state.regfile.set_cr(4, LANES)          # R_CYCLIC slot stride
        state.regfile.set_cr(5, self.tiles_per_row)
        state.regfile.set_cr(6, self.in_plane_stride)
        state.regfile.set_cr(7, h)
        state.regfile.set_cr(8, c)
        state.set_cr_dstructure(valid_elements=LANES)

    def teardown(self, state: "IpuState") -> None:
        if self.output_path is None:
            return
        raw = state.xmem.read_address(self.output_base, self.output_rows * ROW_BYTES)
        planes = np.frombuffer(raw, dtype="<f4").reshape(
            self.channels, self.height, self.tiles_per_row, LANES
        )
        cols = planes[:, :, :, :TILE_COLS].reshape(
            self.channels, self.height, self.tiles_per_row * TILE_COLS
        )
        np.ascontiguousarray(cols[:, :, : self.width]).tofile(self.output_path)

    def run(self, **kwargs):
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
    return _query(params).layout(
        out_tile_cols=TILE_COLS,
        in_tiles_per_out_tile=IN_TILES_PER_OUT_TILE,
        guard_tiles=GUARD_TILES,
        pad_rows=PADDING,
        scratch_rows=SCRATCH_ROWS,
    )


def _supports(**params):
    q = _query(params)
    bad = geometry_refusal(q)
    if bad:
        return no(bad)
    if q.kernel != KERNEL_SIZE or q.stride != STRIDE or q.padding != PADDING:
        return no(
            f"is the unrolled {KERNEL_SIZE}x{KERNEL_SIZE} stride-{STRIDE} "
            f"padding-{PADDING} window and nothing else; this query asks for "
            f"{q.kernel}x{q.kernel} stride {q.stride} padding {q.padding}. "
            f"Every tap is emitted, so K is not a run-time value here -- "
            f"maxpool2d_window takes any odd window."
        )
    budget = xmem_refusal(_geometry(params))
    if budget:
        return no(budget)
    return yes()


def _build(**params):
    q = _query(params)
    return {"channels": q.channels, "height": q.height, "width": q.width}


def _explain(**params):
    lay = _geometry(params)
    q = lay.query
    return (
        f"the unrolled {KERNEL_SIZE}x{KERNEL_SIZE} NMS window: all "
        f"{KERNEL_SIZE * KERNEL_SIZE} taps are emitted, so the row loads hide "
        f"in the tap stream and tap (0,0) seeds the maximum -- "
        f"{KERNEL_SIZE ** 2 + 5} words per output tile against "
        f"maxpool2d_window's {KERNEL_SIZE ** 2 + 3 * KERNEL_SIZE + 5}, about "
        f"24% fewer cycles for the same {q.height}x{q.width} result"
    )


def _caveats(**params):
    lay = _geometry(params)
    notes = [
        WIDE_VECTOR_ONLY,
        "K is fixed at 9 by the unroll: this kernel is SuperPoint's default "
        "nms_radius=4 window and cannot be asked for another size.",
    ]
    lanes = lane_caveat(lay)
    if lanes:
        notes.append(lanes)
    notes.append(headroom_caveat(lay))
    return tuple(notes)


SPEC = KernelSpec(
    name="maxpool2d_nms9",
    op="maxpool2d",
    variant="nms9",
    app_class=MaxPool2dNms9App,
    asm="maxpool2d_nms9.asm",
    requires=("shape", "kernel_size", "stride", "padding"),
    tags=("fp32-wide", "windowed", "unrolled"),
    supports=_supports,
    build=_build,
    explain=_explain,
    caveats=_caveats,
    bundle=lambda **params: _query(params).bundle,
    # Beats maxpool2d_window (1.0) at K=9, which is the whole point of it
    # existing; every other window still routes to the general kernel.
    cost=lambda **params: 0.0,
)
