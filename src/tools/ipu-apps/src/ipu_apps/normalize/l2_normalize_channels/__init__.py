"""L2 normalization down the leading axis (FP32 wide-vector mode).

Computes, over a ``(ROWS, COLS)`` matrix whose leading axis is the reduction
axis::

    out[c, n] = x[c, n] / sqrt( SUM_c x[c, n]^2 )

SuperPoint's dense descriptor normalization is this with ``ROWS = 256``
(``convDb``'s channels) and ``COLS = H*W``, so a ``(256, H, W)`` tensor with
``dim=0`` routes straight here.

An all-zero column normalizes to zeros rather than producing ``inf`` or
``NaN``: the ``rsqrt`` activation is guarded at zero, matching
``l2_normalize_ref``'s ``||x|| == 0 -> zeros``.

Why the leading axis
--------------------
Every column is an independent normalization and the datapath is 128 lanes
wide, so one pass down the rows reduces 128 columns at once with ``ACC.ADD``:
the running sum of squares stays a full 128-element vector and never collapses
to a scalar. There is no ``AGG``, no fan-out, and no per-row bookkeeping
vector -- the same structural saving ``softmax_columns`` has over
``softmax_rows``. A row-wise L2 norm would be a genuinely different kernel and
does not exist yet.

``1/||x||`` is an activation here, not a reduction post-function: the old
``AGG sum inv_sqrt`` post-fn is gone, and ``rsqrt`` is one of
``ACTIVATE.QUANTIZE``'s twelve activations.

Memory layout (all sizes in 128-element rows; one row = 512 bytes here):

    input   ROWS x TPR rows,  row (c, t) at c*TPR + t
    rvec    1 row, holding 1/||x|| for the 128 columns of the current tile
    output  ROWS x TPR rows,  same offsets against a different base

``TPR = ceil(COLS / 128)``. Input and output share one offset, so a single
walking register addresses both. Columns past ``COLS`` are zero; they form
their own all-zero columns, which the guard sends to zero, and are trimmed on
read-back.

Usage::

    from ipu_apps.normalize.l2_normalize_channels import L2NormalizeChannelsApp

    app = L2NormalizeChannelsApp(
        inst_path="l2_normalize_channels.bin",
        input_path="d.bin",       # ROWS*COLS FP32, row-major
        output_path="dn.bin",
        rows=256, cols=4800,
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
from ipu_apps.normalize._spec_support import (
    BASE_ROW,
    LANES,
    ROW_BYTES,
    WIDE_VECTOR_ONLY,
    geometry_refusal,
    headroom_caveat,
    lane_caveat,
    norm_query,
    xmem_refusal,
)


class L2NormalizeChannelsApp(IpuApp):
    """Unit-length normalization down the leading axis of a ``(rows, cols)`` matrix.

    Args:
        inst_path:   Path to the assembled instruction binary.
        input_path:  Matrix, ``rows*cols`` FP32, row-major.
        output_path: Optional path for the ``rows*cols`` FP32 result.
        rows:        Reduction length (the channel count).
        cols:        Independent columns.
    """

    def __init__(self, *, rows: int, cols: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.input_path = Path(self.input_path)

        self.rows = int(rows)
        self.cols = int(cols)

        # Delegate to the registry declaration rather than restating the
        # bounds: SPEC.supports is the single source of truth for this kernel's
        # domain, including the XMEM budget.
        SPEC.guard(shape=(self.rows, self.cols), dim=0)
        self.query = norm_query((self.rows, self.cols), 0)
        self._layout()

    def _layout(self) -> None:
        """Place the three regions back-to-back, in rows and in bytes.

        Row numbers drive the CR registers (the .asm's XMEM operands are rows);
        the byte addresses drive the direct ``state.xmem`` calls in setup and
        teardown, which bypass row translation.
        """
        q = self.query
        self.tiles_per_channel = q.tiles_per_channel
        self.region_rows = q.region_rows

        self.input_base_row = BASE_ROW
        self.rvec_row = self.input_base_row + self.region_rows
        self.output_base_row = self.rvec_row + 1

        self.input_base = self.input_base_row * ROW_BYTES
        self.output_base = self.output_base_row * ROW_BYTES

    # -- wide-vector FP32 state ---------------------------------------------

    @staticmethod
    def make_state() -> IpuState:
        """Build the FP32 wide-vector state this app requires.

        ``wide_vector_quantize_output=False`` keeps elements 4-byte FP32
        through AAQ, so ACTIVATE.QUANTIZE applies rsqrt in FP32 and writes
        floats into POST_AAQ_REG (the INT8 clamp is skipped entirely) and
        STR_POST_AAQ_REG drains the full 512 bytes.
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

    def setup(self, state: "IpuState") -> None:
        data = np.fromfile(self.input_path, dtype="<f4")
        want = self.rows * self.cols
        if data.size != want:
            raise ValueError(
                f"input file {self.input_path} holds {data.size} FP32 values; "
                f"this normalization needs {want}"
            )
        # Columns past `cols` stay zero: they are their own all-zero columns,
        # which rsqrt's guard sends to zero rather than to inf.
        padded = np.zeros(
            (self.rows, self.tiles_per_channel * LANES), dtype="<f4"
        )
        padded[:, : self.cols] = data.reshape(self.rows, self.cols)
        state.xmem.write_address(self.input_base, padded.tobytes())

        # CR0 and CR1 are READ-ONLY: CR0 == 0 always, CR1 == 1 always. Both are
        # exploited directly -- CR0 doubles as the 0.0 scalar that clears R_ACC
        # before each reduction, and CR1 as every +1 increment. All writable CRs
        # below hold row numbers, row strides or loop bounds (.asm XMEM operands
        # are rows, not bytes -- see issue #179).
        state.regfile.set_cr(2, self.input_base_row)
        state.regfile.set_cr(3, self.output_base_row)
        state.regfile.set_cr(4, self.rvec_row)
        # CR5 is both the row stride between matrix rows and the tile-loop
        # bound: a matrix row occupies TPR XMEM rows, and there are TPR tiles.
        state.regfile.set_cr(5, self.tiles_per_channel)
        state.regfile.set_cr(6, self.rows)

        # MULT.*/ACTIVATE.QUANTIZE read the active-element count from the named
        # dstructure CR's valid_elements field. The asm names CR15 throughout,
        # so set CR15.valid_elements = 128 to process the full FP32 row.
        state.set_cr_dstructure(valid_elements=LANES)

    def teardown(self, state: "IpuState") -> None:
        if self.output_path is None:
            return
        raw = state.xmem.read_address(self.output_base, self.region_rows * ROW_BYTES)
        padded = np.frombuffer(raw, dtype="<f4").reshape(
            self.rows, self.tiles_per_channel * LANES
        )
        np.ascontiguousarray(padded[:, : self.cols]).tofile(self.output_path)

    def run(self, **kwargs):
        # Always run on the FP32 wide-vector state unless the caller supplied one.
        kwargs.setdefault("state", self.make_state())
        return super().run(**kwargs)


# -- registry declaration ---------------------------------------------------


def _query(params):
    return norm_query(params["shape"], params["dim"])


def _supports(**params):
    q = _query(params)
    bad = geometry_refusal(q)
    if bad:
        return no(bad)
    if not q.down_columns:
        return no(
            f"normalizes down the leading axis, one scalar per column held in a "
            f"full 128-element vector; this query normalizes along each row of "
            f"{q.cols} elements, which needs an AGG reduction and a per-row "
            f"fan-out. No kernel implements that yet."
        )
    budget = xmem_refusal(q)
    if budget:
        return no(budget)
    return yes()


def _build(**params):
    q = _query(params)
    return {"rows": q.rows, "cols": q.cols}


def _explain(**params):
    q = _query(params)
    return (
        f"reduces down the leading axis, so the {q.cols} column(s) normalize "
        f"independently in {q.tiles_per_channel} tile(s) of at most {LANES} "
        f"lanes; the {q.rows}-element sum of squares stays a full vector, so "
        f"there is no AGG and no fan-out"
    )


def _caveats(**params):
    q = _query(params)
    notes = [
        WIDE_VECTOR_ONLY,
        "an all-zero vector normalizes to zeros: the rsqrt activation is "
        "guarded at zero rather than returning inf.",
    ]
    lanes = lane_caveat(q)
    if lanes:
        notes.append(lanes)
    notes.append(headroom_caveat(q))
    return tuple(notes)


SPEC = KernelSpec(
    name="l2_normalize_channels",
    op="l2_normalize",
    variant="channels",
    app_class=L2NormalizeChannelsApp,
    asm="l2_normalize_channels.asm",
    requires=("shape", "dim"),
    tags=("fp32-wide", "columns"),
    supports=_supports,
    build=_build,
    explain=_explain,
    caveats=_caveats,
    bundle=lambda **params: _query(params).bundle,
    # The only l2_normalize kernel, and an exact match for its domain.
    cost=lambda **params: 0.0,
)
