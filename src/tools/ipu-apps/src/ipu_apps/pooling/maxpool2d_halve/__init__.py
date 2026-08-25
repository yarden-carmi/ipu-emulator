"""2x2 stride-2 max-pool harness (FP32 wide-vector mode).

Computes, per channel::

    out[c, y, x] = max( in[c, 2y,   2x], in[c, 2y,   2x+1],
                        in[c, 2y+1, 2x], in[c, 2y+1, 2x+1] )

which is ``nn.MaxPool2d(kernel_size=2, stride=2)`` -- the layer SuperPoint
applies after ``conv1b``, ``conv2b`` and ``conv3b``. One launch produces the
whole ``(C, H//2, W//2)`` output. An odd ``H`` or ``W`` drops the last row or
column, exactly as torch does.

How the four taps are read
--------------------------
XMEM addressing is row-granular: a load reaches a whole 128-element row and
cannot shift by one element, so the four shifted *loads* a byte-addressed
kernel would use are not expressible. ``MULT.RC.*`` reads R_CYCLIC at an
arbitrary element index and may cross slot boundaries, so the horizontal shift
moves into the register instead -- the two vertically-neighbouring spatial rows
occupy R_CYCLIC slots 0/1 and 2/3, and ``dx`` becomes a ``+1`` step on the read
index.

Why there is a round trip
-------------------------
Four taps give the *stride-1* 2x2 maximum at every column; a stride-2 pool
wants only the even ones. ``ACC.STRIDE`` performs exactly that decimation, but
it writes MULT_RES into R_ACC **overwriting**, so it cannot take a maximum and
the maximum has to be finished first. The kernel therefore stages the stride-1
result to a scratch row, reloads it, and decimates on the way back in.

Both halves of an output row are staged before either is decimated: half B's
``ACC.MAX.FIRST`` overwrites all 128 R_ACC elements and would destroy half A's
decimated result. Decimating afterwards works because ``ACC.STRIDE`` leaves the
R_ACC indices it does not write untouched -- half A lands at base 0 and half B
at base 64.

Memory layout (all sizes in 128-element rows; one row = 512 bytes here):

    input    C planes, each H spatial rows x IN_ROW_STRIDE rows
    scratch  2 rows (one per half of the output row being built)
    output   C planes, each (H//2) spatial rows x OUT_TILES_PER_ROW rows

One output XMEM row is 128 output columns and so spans 256 input columns --
two full-width input tiles. ``IN_ROW_STRIDE`` is sized from what the kernel
*reads* (``2 * OUT_TILES_PER_ROW``) rather than from ``ceil(W / 128)``, because
a partly-filled last output tile still needs its input tiles to exist, plus one
**guard tile**: the ``dx=1`` tap reads element 128 of its slot pair, i.e. the
first element of the next tile of the same spatial row, so the last half of the
last output tile needs a tile after it.

Every lane that is not a real image column -- the columns past ``W`` in a
partly-filled tile, and the guard tiles -- holds ``-FLT_MAX``, the identity of a
maximum. Those lanes never reach a kept output (output lane ``j < W//2`` reads
input columns ``2j`` and ``2j+1``, both below ``W``), so this is about keeping
the discarded lanes finite and debuggable rather than about correctness.

Usage::

    from ipu_apps.pooling.maxpool2d_halve import MaxPool2dHalveApp

    app = MaxPool2dHalveApp(
        inst_path="maxpool2d_halve.bin",
        input_path="x.bin",       # C*H*W FP32, channel-major
        output_path="y.bin",      # C*(H//2)*(W//2) FP32
        channels=64, height=8, width=80,
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

# The window this app implements. Both are fixed in the .asm: the four taps are
# unrolled, and ACC.STRIDE's decimation phase is an immediate.
KERNEL_SIZE = 2
STRIDE = 2
PADDING = 0

# Every lane of an output XMEM row is a usable output column -- this kernel
# spends no lanes on a halo, because the +1 shift is satisfied by the next
# input tile rather than by reserved lanes.
OUT_TILE_COLS = LANES
# 128 output columns span 256 input columns.
IN_TILES_PER_OUT_TILE = 2
# One tile past the last one a half reads, so the +1 element shift stays inside
# the spatial row.
GUARD_TILES = 1
# Padding is 0, so rows 2y and 2y+1 are always real image rows.
PAD_ROWS = 0
# One staging row per half of the output row being built.
SCRATCH_ROWS = 2


class MaxPool2dHalveApp(IpuApp):
    """2x2 stride-2 max-pool over a ``(C, H, W)`` activation.

    Args:
        inst_path:   Path to the assembled instruction binary.
        input_path:  Activation, ``C*H*W`` FP32, channel-major.
        output_path: Optional path for the ``C*(H//2)*(W//2)`` FP32 result.
        channels:    C.
        height:      H.
        width:       W.
    """

    def __init__(
        self,
        *,
        channels: int,
        height: int,
        width: int,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.input_path = Path(self.input_path)

        self.channels = int(channels)
        self.height = int(height)
        self.width = int(width)

        # Delegate to the registry declaration rather than restating the
        # bounds: SPEC.supports is the single source of truth for this kernel's
        # domain, including the XMEM budget.
        shape = (self.channels, self.height, self.width)
        SPEC.guard(
            shape=shape,
            kernel_size=KERNEL_SIZE,
            stride=STRIDE,
            padding=PADDING,
        )
        self.query = pool_query(
            shape, kernel_size=KERNEL_SIZE, stride=STRIDE, padding=PADDING
        )
        self._layout()

    def _layout(self) -> None:
        """Place the three regions back-to-back, in rows and in bytes.

        Row numbers drive the CR registers (the .asm's XMEM operands are rows);
        the byte addresses drive the direct ``state.xmem`` calls in setup and
        teardown, which bypass row translation. Keeping both explicit is what
        stops the two units being confused.
        """
        lay = self.query.layout(
            out_tile_cols=OUT_TILE_COLS,
            in_tiles_per_out_tile=IN_TILES_PER_OUT_TILE,
            guard_tiles=GUARD_TILES,
            pad_rows=PAD_ROWS,
            scratch_rows=SCRATCH_ROWS,
        )
        self.geometry = lay
        self.out_height = self.query.out_height
        self.out_width = self.query.out_width
        self.in_row_stride = lay.in_tiles_per_row
        self.out_tiles_per_row = lay.out_tiles_per_row
        self.in_plane_stride = lay.in_plane_stride
        self.out_plane_stride = lay.out_plane_stride
        self.output_rows = lay.output_rows

        # The input tiling is sized from what the halves read, so it must cover
        # every tile that actually holds image columns. This has always held for
        # a stride-2 window, but it is the one relation the .asm cannot check.
        assert lay.in_tiles_per_row >= lay.tiles_holding_width, (
            f"input tiling {lay.in_tiles_per_row} does not cover the "
            f"{lay.tiles_holding_width} tile(s) holding width {self.width}"
        )

        self.input_base_row = BASE_ROW
        self.scratch_base_row = self.input_base_row + lay.input_rows
        self.output_base_row = self.scratch_base_row + SCRATCH_ROWS

        self.input_base = self.input_base_row * ROW_BYTES
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

    def setup(self, state: "IpuState") -> None:
        c, h, w = self.channels, self.height, self.width
        x = self._read_f32(self.input_path, c * h * w, "input").reshape(c, h, w)

        # Every lane the kernel may read but that holds no image column is the
        # maximum's identity, so a stray tap can never win.
        planes = np.full(
            (c, h, self.in_row_stride * LANES), NEG_FILL, dtype="<f4"
        )
        planes[:, :, :w] = x
        state.xmem.write_address(self.input_base, planes.tobytes())

        # CR0 and CR1 are READ-ONLY: CR0 == 0 always, CR1 == 1 always. Both are
        # exploited directly -- CR0 as the zero source and CR1 as both the 1.0
        # scalar for the identity multiply and every +1 increment. All writable
        # CRs below hold row numbers, row strides or loop bounds (.asm XMEM
        # operands are rows, not bytes -- see issue #179).
        state.regfile.set_cr(2, self.input_base_row)
        state.regfile.set_cr(3, self.output_base_row)
        state.regfile.set_cr(4, self.scratch_base_row)
        state.regfile.set_cr(5, self.in_row_stride)
        # CR6 walks the input row base from spatial row 2y to 2(y+1). Precomputed
        # because the LR slot has no multiply.
        state.regfile.set_cr(6, 2 * self.in_row_stride)
        state.regfile.set_cr(7, self.out_tiles_per_row)
        state.regfile.set_cr(8, self.out_height)
        state.regfile.set_cr(9, c)
        state.regfile.set_cr(10, LANES)
        state.regfile.set_cr(11, self.in_plane_stride)

        # MULT.*/ACTIVATE.QUANTIZE read the active-element count from the named
        # dstructure CR's valid_elements field. The asm names CR15 throughout,
        # so set CR15.valid_elements = 128 to process the full FP32 row.
        state.set_cr_dstructure(valid_elements=LANES)

    def teardown(self, state: "IpuState") -> None:
        if self.output_path is None:
            return
        raw = state.xmem.read_address(self.output_base, self.output_rows * ROW_BYTES)
        planes = np.frombuffer(raw, dtype="<f4").reshape(
            self.channels, self.out_height, self.out_tiles_per_row * LANES
        )
        # Lanes past the output width are the decimated image of the -FLT_MAX
        # fill; trimming them here is what makes the output file a dense
        # (C, H//2, W//2) array matching the input's element order.
        np.ascontiguousarray(planes[:, :, : self.out_width]).tofile(self.output_path)

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
    return _query(params).layout(
        out_tile_cols=OUT_TILE_COLS,
        in_tiles_per_out_tile=IN_TILES_PER_OUT_TILE,
        guard_tiles=GUARD_TILES,
        pad_rows=PAD_ROWS,
        scratch_rows=SCRATCH_ROWS,
    )


def _supports(**params):
    q = _query(params)
    bad = geometry_refusal(q)
    if bad:
        return no(bad)
    if q.kernel != KERNEL_SIZE or q.stride != STRIDE:
        return no(
            f"pools a {KERNEL_SIZE}x{KERNEL_SIZE} window with stride {STRIDE}; "
            f"this query asks for {q.kernel}x{q.kernel} with stride {q.stride}. "
            f"Both are fixed in the .asm -- the four taps are unrolled and "
            f"ACC.STRIDE's decimation phase is an immediate. A stride-1 "
            f"windowed max has its own kernel (maxpool2d_window)."
        )
    if q.padding != PADDING:
        return no(
            f"pools without padding; this query asks for padding {q.padding}. "
            f"A pad would shift which input columns are even, and ACC.STRIDE "
            f"always keeps the even ones."
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
        f"a {KERNEL_SIZE}x{KERNEL_SIZE} stride-{STRIDE} max-pool; the four taps "
        f"are read out of R_CYCLIC at a +1 element step and ACC.STRIDE keeps "
        f"the even columns, so {q.height}x{q.width} becomes "
        f"{q.out_height}x{q.out_width} in {lay.out_tiles_per_row} output "
        f"row(s) per spatial row"
    )


def _caveats(**params):
    lay = _geometry(params)
    notes = [WIDE_VECTOR_ONLY]
    if lay.query.height % 2 or lay.query.width % 2:
        notes.append(
            f"an odd extent drops the trailing row/column: {lay.query.height}x"
            f"{lay.query.width} pools to {lay.query.out_height}x"
            f"{lay.query.out_width}, so input row {lay.query.height - 1} "
            f"and/or column {lay.query.width - 1} is never read. This matches "
            f"nn.MaxPool2d(2, 2, ceil_mode=False)."
        )
    lanes = lane_caveat(lay)
    if lanes:
        notes.append(lanes)
    notes.append(headroom_caveat(lay))
    return tuple(notes)


SPEC = KernelSpec(
    name="maxpool2d_halve",
    op="maxpool2d",
    variant="halve",
    app_class=MaxPool2dHalveApp,
    asm="maxpool2d_halve.asm",
    # The window parameters are required rather than defaulted on purpose: a
    # pooling spec that silently assumed stride 2 would answer for an operation
    # no kernel here computes.
    requires=("shape", "kernel_size", "stride", "padding"),
    tags=("fp32-wide", "strided"),
    supports=_supports,
    build=_build,
    explain=_explain,
    caveats=_caveats,
    bundle=lambda **params: _query(params).bundle,
    # The only stride-2 pooling kernel, and an exact match for its domain.
    cost=lambda **params: 0.0,
)
