"""Keypoint selection gate + soft survivor count (FP32 wide-vector mode).

Computes, over a score map of any shape::

    selected[i] = relu(s[i] - tau)                 exact set {s > tau}
    count       = SUM_i sigmoid(T * (s[i] - tau))  approximate |{s > tau}|

SuperPoint's keypoint selection. The gate is exact -- a surviving score keeps a
positive shifted value, a rejected one becomes 0. The count is what the **host**
bisects on to hit a target ``k``: it re-launches with a new ``tau``, reads the
count, and narrows.

**This does not produce ranked indices.** Exact ranked top-k needs a
per-element compare and lane-index extraction, and the ISA has neither. The
host selects the final ``k`` from the thresholded scores. That is the one
behavioural approximation in SuperPoint's detector path: the cap keeps
about ``k`` keypoints rather than exactly ``k``.

The gate is element-wise, so the input's rank carries no meaning -- only how
many elements there are. The shape is preserved in the verdict so it still
reports what the caller asked about, and ``selected`` comes back in the same
shape.

Two resident threshold vectors, not one
---------------------------------------
CR scalars are integer-only in wide-vector mode, so ``tau`` rides in a
128-element XMEM row. The count needs ``T*(s - tau)``, and scaling R_ACC by
``T`` would cost an XMEM round trip -- so a *second* resident vector holds
``T*tau``, and the count path computes ``T*s`` (a vector-vector multiply
against a resident ``T`` vector in R0) and subtracts it. Both subtractions are
``ACC.SUB``.

The padding lanes are suppressed, not ignored
---------------------------------------------
Elements past the real count fill the last row, and ``sigmoid(0 - tau)`` is not
zero -- padding would inflate the count. Those lanes are filled with
``tau - 800/T``, so ``T*(pad - tau)`` is exactly ``-800`` and the sigmoid
underflows to a true zero. ``relu`` of the same value is zero too, so the
padding costs nothing in either output.

Why the count needs its own pass
--------------------------------
``AGG.SUM`` writes one R_ACC slot, and the next row's ``ACC.ADD.FIRST``
overwrites all 128 -- including that slot. The sigmoid plane is therefore
staged, reduced with ``ACC.ADD`` down the rows into 128 partial sums, and
collapsed once at the end.

Memory layout (all sizes in 128-element rows; one row = 512 bytes here):

    scores    ROWS rows        selected  ROWS rows
    staged    ROWS rows        count     1 row (partials, then the scalar)
    tau, T*tau, T             1 row each, resident

Usage::

    from ipu_apps.detect.score_threshold import ScoreThresholdApp

    app = ScoreThresholdApp(
        inst_path="score_threshold.bin",
        input_path="scores.bin",     # N FP32
        output_path="selected.bin",  # N FP32
        count_path="count.bin",      # 1 FP32 (optional)
        shape=(480, 640), threshold=0.005, temperature=64.0,
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
    NO_RANKED_INDICES,
    PAD_LOGIT,
    ROW_BYTES,
    WIDE_VECTOR_ONLY,
    headroom_caveat,
    lane_caveat,
    threshold_query,
    threshold_refusal,
)

# Default sharpness of the soft count. 64 is what the reference bisection uses:
# sharp enough that the count tracks the true survivor count closely, flat
# enough that the bisection sees a gradient.
DEFAULT_TEMPERATURE = 64.0


class ScoreThresholdApp(IpuApp):
    """``relu(s - tau)`` over a score map, plus a soft survivor count.

    Args:
        inst_path:   Path to the assembled instruction binary.
        input_path:  Scores, ``prod(shape)`` FP32.
        output_path: Optional path for the thresholded scores, same layout.
        count_path:  Optional path for the single-FP32 soft survivor count.
        shape:       The caller's score-map shape. Only its element count
                     affects the computation; it is kept so the output file
                     matches the input file's layout.
        threshold:   tau.
        temperature: T, the sharpness of the soft count.
    """

    def __init__(
        self,
        *,
        shape,
        threshold: float,
        temperature: float = DEFAULT_TEMPERATURE,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.input_path = Path(self.input_path)
        count_path = getattr(self, "count_path", None)
        self.count_path = Path(count_path) if count_path is not None else None

        self.shape = tuple(int(d) for d in shape)
        self.threshold = float(threshold)
        self.temperature = float(temperature)

        # Delegate to the registry declaration rather than restating the bounds.
        SPEC.guard(
            shape=self.shape,
            threshold=self.threshold,
            temperature=self.temperature,
        )
        self.query = threshold_query(
            self.shape, threshold=self.threshold, temperature=self.temperature
        )
        self.elements = self.query.elements
        self._layout()

    def _layout(self) -> None:
        """Place the regions back-to-back, in rows and in bytes."""
        self.rows = self.query.rows

        self.scores_base_row = BASE_ROW
        self.selected_base_row = self.scores_base_row + self.rows
        self.staged_base_row = self.selected_base_row + self.rows
        self.count_row = self.staged_base_row + self.rows
        self.tau_row = self.count_row + 1
        self.ttau_row = self.tau_row + 1
        self.tvec_row = self.ttau_row + 1

        self.scores_base = self.scores_base_row * ROW_BYTES
        self.selected_base = self.selected_base_row * ROW_BYTES
        self.count_base = self.count_row * ROW_BYTES
        self.tau_base = self.tau_row * ROW_BYTES
        self.ttau_base = self.ttau_row * ROW_BYTES
        self.tvec_base = self.tvec_row * ROW_BYTES

    @property
    def pad_value(self) -> float:
        """The score written into lanes past the real element count.

        Chosen so ``T * (pad - tau) == -PAD_LOGIT`` exactly: the sigmoid then
        underflows to a true zero and the padding adds nothing to the count,
        while ``relu(pad - tau)`` is zero for the same reason.
        """
        return self.threshold - PAD_LOGIT / self.temperature

    # -- wide-vector FP32 state ---------------------------------------------

    @staticmethod
    def make_state() -> IpuState:
        """Build the FP32 wide-vector state this app requires."""
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
        if data.size != self.elements:
            raise ValueError(
                f"input file {self.input_path} holds {data.size} FP32 values; "
                f"this shape needs {self.elements}"
            )
        scores = np.full(self.rows * LANES, self.pad_value, dtype="<f4")
        scores[: self.elements] = data
        state.xmem.write_address(self.scores_base, scores.tobytes())

        for base, value in (
            (self.tau_base, self.threshold),
            (self.ttau_base, self.temperature * self.threshold),
            (self.tvec_base, self.temperature),
        ):
            state.xmem.write_address(
                base, np.full(LANES, value, dtype="<f4").tobytes()
            )

        # CR0 and CR1 are READ-ONLY: CR0 == 0 always, CR1 == 1 always. CR0
        # doubles as the 0.0 scalar that clears R_ACC before pass 2, and CR1 as
        # the 1.0 scalar for every identity multiply. All writable CRs below
        # hold row numbers or loop bounds (.asm XMEM operands are rows, not
        # bytes -- see issue #179).
        state.regfile.set_cr(2, self.scores_base_row)
        state.regfile.set_cr(3, self.selected_base_row)
        state.regfile.set_cr(4, self.staged_base_row)
        state.regfile.set_cr(5, self.count_row)
        state.regfile.set_cr(6, self.tau_row)
        state.regfile.set_cr(7, self.ttau_row)
        state.regfile.set_cr(8, self.tvec_row)
        state.regfile.set_cr(9, self.rows)
        state.regfile.set_cr(10, LANES)

        state.set_cr_dstructure(valid_elements=LANES)

    def teardown(self, state: "IpuState") -> None:
        if self.output_path is not None:
            raw = state.xmem.read_address(
                self.selected_base, self.rows * ROW_BYTES
            )
            flat = np.frombuffer(raw, dtype="<f4")
            np.ascontiguousarray(flat[: self.elements]).tofile(self.output_path)
        if self.count_path is not None:
            raw = state.xmem.read_address(self.count_base, ROW_BYTES)
            np.ascontiguousarray(
                np.frombuffer(raw, dtype="<f4")[:1]
            ).tofile(self.count_path)

    def run(self, **kwargs):
        # Always run on the FP32 wide-vector state unless the caller supplied one.
        kwargs.setdefault("state", self.make_state())
        return super().run(**kwargs)


# -- registry declaration ---------------------------------------------------


def _query(params):
    return threshold_query(
        params["shape"],
        threshold=params["threshold"],
        temperature=params.get("temperature", DEFAULT_TEMPERATURE),
    )


def _supports(**params):
    q = _query(params)
    bad = threshold_refusal(q)
    if bad:
        return no(bad)
    return yes()


def _build(**params):
    q = _query(params)
    return {
        "shape": q.dims,
        "threshold": q.threshold,
        "temperature": q.temperature,
    }


def _explain(**params):
    q = _query(params)
    return (
        f"an element-wise gate over {q.elements} score(s) in {q.rows} row(s): "
        f"relu(s - tau) is exact, and the soft count sums "
        f"sigmoid({q.temperature:g}*(s - tau)) for the host's bisection on tau"
    )


def _caveats(**params):
    q = _query(params)
    notes = [WIDE_VECTOR_ONLY, NO_RANKED_INDICES]
    lanes = lane_caveat(q.elements, q.rows)
    if lanes:
        notes.append(lanes)
    notes.append(headroom_caveat(q.total_rows))
    return tuple(notes)


SPEC = KernelSpec(
    name="score_threshold",
    op="score_threshold",
    variant="relu_gate",
    app_class=ScoreThresholdApp,
    asm="score_threshold.asm",
    # `temperature` is deliberately NOT required: it has a meaningful default
    # and only affects the count, whereas omitting `threshold` would leave the
    # kernel with no gate at all.
    requires=("shape", "threshold"),
    tags=("fp32-wide", "elementwise"),
    supports=_supports,
    build=_build,
    explain=_explain,
    caveats=_caveats,
    bundle=lambda **params: _query(params).bundle,
    cost=lambda **params: 0.0,
)
