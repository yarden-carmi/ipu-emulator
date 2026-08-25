"""Detector confidence + fixed-threshold gate (FP32 wide-vector mode).

Computes, over a ``(C, N)`` logit map with cells in lanes::

    confidence[n] = max over c of logits[c, n]
    keep[n]       = relu(confidence[n] - tau)

SuperPoint's detector read-out. Both planes are written: the confidence is
useful on its own, and the gate keeps a surviving cell's shifted score while
zeroing the rest.

**Argmax-equivalent to the softmax path, not a softmax substitute.**
``argmax(softmax(x)) == argmax(x)``, so which cell wins is unchanged by taking
the maximum over raw logits instead of over probabilities. The *value* differs:
it is a logit, not a probability. Use this where the ranking matters; use
``softmax`` (the ``softmax_columns`` kernel handles a ``(65, H*W)`` detector
map) where the value does.

Why there is no AGG
-------------------
Every cell is an independent maximum and the datapath is 128 lanes wide, so one
pass down the channel planes reduces 128 cells at once with ``ACC.MAX``. The
running maximum stays a full 128-element vector and never collapses to a
scalar.

Plane 0 is issued before the loop so ``ACC.MAX.FIRST`` can seed R_ACC without a
``-inf`` vector; the channel count is a run-time bound, so the first iteration
cannot carry a different accumulate mode.

The threshold is a resident vector
----------------------------------
CR scalars are integer-only in wide-vector mode, so a fractional ``tau`` cannot
ride in a CR. It lives in R_CYCLIC slot 1 for the whole run and is subtracted
with ``ACC.SUB``. ``ACTIVATE.QUANTIZE`` does not modify R_ACC, so the
confidence is stored and then subtracted from in place -- the maximum is
computed once.

Memory layout (all sizes in 128-element rows; one row = 512 bytes here):

    input       C planes x TILES rows, plane c at c*TILES
    tau         1 row, resident in R_CYCLIC slot 1
    confidence  TILES rows
    keep        TILES rows

Usage::

    from ipu_apps.detect.channel_peak import ChannelPeakApp

    app = ChannelPeakApp(
        inst_path="channel_peak.bin",
        input_path="logits.bin",       # C*N FP32, channel-major
        output_path="keep.bin",        # N FP32
        confidence_path="conf.bin",    # N FP32 (optional)
        channels=64, cells=4800, threshold=0.005,
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
from ipu_apps.detect._spec_support import (
    BASE_ROW,
    LANES,
    ROW_BYTES,
    WIDE_VECTOR_ONLY,
    headroom_caveat,
    lane_caveat,
    peak_query,
    peak_refusal,
)


class ChannelPeakApp(IpuApp):
    """Per-cell maximum over ``channels`` planes, plus a threshold gate.

    Args:
        inst_path:       Path to the assembled instruction binary.
        input_path:      Logits, ``channels*cells`` FP32, channel-major.
        output_path:     Optional path for the ``cells`` FP32 gate output.
        confidence_path: Optional path for the ``cells`` FP32 confidence.
        channels:        Planes reduced over.
        cells:           Independent cells.
        threshold:       tau.
    """

    def __init__(
        self,
        *,
        channels: int,
        cells: int,
        threshold: float,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.input_path = Path(self.input_path)
        conf_path = getattr(self, "confidence_path", None)
        self.confidence_path = Path(conf_path) if conf_path is not None else None

        self.channels = int(channels)
        self.cells = int(cells)
        self.threshold = float(threshold)

        # Delegate to the registry declaration rather than restating the bounds.
        SPEC.guard(
            shape=(self.channels, self.cells), threshold=self.threshold
        )
        self.query = peak_query(
            (self.channels, self.cells), threshold=self.threshold
        )
        self._layout()

    def _layout(self) -> None:
        """Place the four regions back-to-back, in rows and in bytes."""
        q = self.query
        self.tiles = q.tiles

        self.input_base_row = BASE_ROW
        self.tau_row = self.input_base_row + q.input_rows
        self.confidence_base_row = self.tau_row + 1
        self.keep_base_row = self.confidence_base_row + self.tiles

        self.input_base = self.input_base_row * ROW_BYTES
        self.tau_base = self.tau_row * ROW_BYTES
        self.confidence_base = self.confidence_base_row * ROW_BYTES
        self.keep_base = self.keep_base_row * ROW_BYTES

    # -- wide-vector FP32 state ---------------------------------------------

    @staticmethod
    def make_state() -> IpuState:
        """Build the FP32 wide-vector state this app requires.

        ``wide_vector_quantize_output=False`` keeps elements 4-byte FP32
        through AAQ, so ACTIVATE.QUANTIZE applies the ReLU in FP32 and writes
        floats into POST_AAQ_REG (the INT8 clamp is skipped entirely).
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
        c, n = self.channels, self.cells
        data = np.fromfile(self.input_path, dtype="<f4")
        if data.size != c * n:
            raise ValueError(
                f"input file {self.input_path} holds {data.size} FP32 values; "
                f"this reduction needs {c * n}"
            )
        # Cells past `cells` stay zero. They are their own columns -- never part
        # of a real cell's maximum -- and are trimmed on read-back.
        planes = np.zeros((c, self.tiles * LANES), dtype="<f4")
        planes[:, :n] = data.reshape(c, n)
        state.xmem.write_address(self.input_base, planes.tobytes())

        # tau as a resident 128-element vector: CR scalars are integer-only in
        # wide-vector mode, so a fractional threshold cannot ride in a CR.
        tau = np.full(LANES, self.threshold, dtype="<f4")
        state.xmem.write_address(self.tau_base, tau.tobytes())

        # CR0 and CR1 are READ-ONLY: CR0 == 0 always, CR1 == 1 always, and CR1
        # doubles as the 1.0 scalar for the identity multiply. All writable CRs
        # below hold row numbers, row strides or loop bounds (.asm XMEM operands
        # are rows, not bytes -- see issue #179).
        state.regfile.set_cr(2, self.input_base_row)
        state.regfile.set_cr(3, self.confidence_base_row)
        state.regfile.set_cr(4, self.keep_base_row)
        state.regfile.set_cr(5, self.tau_row)
        # CR6 is both the row stride between channel planes and the tile-loop
        # bound: a plane occupies TILES rows, and there are TILES tiles.
        state.regfile.set_cr(6, self.tiles)
        state.regfile.set_cr(7, c)
        state.regfile.set_cr(10, LANES)

        state.set_cr_dstructure(valid_elements=LANES)

    def teardown(self, state: "IpuState") -> None:
        for path, base in (
            (self.output_path, self.keep_base),
            (self.confidence_path, self.confidence_base),
        ):
            if path is None:
                continue
            raw = state.xmem.read_address(base, self.tiles * ROW_BYTES)
            flat = np.frombuffer(raw, dtype="<f4")
            np.ascontiguousarray(flat[: self.cells]).tofile(path)

    def run(self, **kwargs):
        # Always run on the FP32 wide-vector state unless the caller supplied one.
        kwargs.setdefault("state", self.make_state())
        return super().run(**kwargs)


# -- registry declaration ---------------------------------------------------


def _query(params):
    return peak_query(params["shape"], threshold=params["threshold"])


def _supports(**params):
    q = _query(params)
    bad = peak_refusal(q)
    if bad:
        return no(bad)
    return yes()


def _build(**params):
    q = _query(params)
    return {"channels": q.channels, "cells": q.cells, "threshold": q.threshold}


def _explain(**params):
    q = _query(params)
    return (
        f"reduces {q.channels} channel plane(s) to one confidence per cell with "
        f"a running ACC.MAX over {q.tiles} tile(s) of at most {LANES} cells, "
        f"then gates against tau in the same R_ACC -- no AGG, no fan-out"
    )


def _caveats(**params):
    q = _query(params)
    notes = [
        WIDE_VECTOR_ONLY,
        "argmax-equivalent to softmax-then-argmax, but the confidence is a raw "
        "logit and not a probability: use a softmax kernel where the value "
        "itself matters.",
    ]
    lanes = lane_caveat(q.cells, q.tiles)
    if lanes:
        notes.append(lanes)
    notes.append(headroom_caveat(q.total_rows))
    return tuple(notes)


SPEC = KernelSpec(
    name="channel_peak",
    op="channel_peak",
    variant="gated",
    app_class=ChannelPeakApp,
    asm="channel_peak.asm",
    requires=("shape", "threshold"),
    tags=("fp32-wide", "columns"),
    supports=_supports,
    build=_build,
    explain=_explain,
    caveats=_caveats,
    bundle=lambda **params: _query(params).bundle,
    cost=lambda **params: 0.0,
)
