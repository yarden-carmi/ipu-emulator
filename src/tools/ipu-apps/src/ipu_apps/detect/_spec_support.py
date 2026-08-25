"""Shared helpers for the detector-head kernels' registry declarations.

Two operations live here -- ``channel_peak`` (reduce the channel planes to a
per-cell confidence, then gate it) and ``score_threshold`` (gate a score map and
report a soft survivor count). They share a datapath convention worth stating
once:

**A threshold is a resident vector, not a CR scalar.** CR scalars are
integer-only in wide-vector mode -- ``MULT.*``'s CR operand supplies its *low
byte* -- so a fractional threshold cannot ride in a CR. It is written into a
128-element XMEM row instead and subtracted with ``ACC.SUB``, exactly as
``softmax_rows`` keeps ``log2(e)`` in a resident vector. The older kernels
negated the threshold and added it (``AGG max value_cr(-1)`` then
``ACC.ADD_AAQ``); ``ACC.SUB`` does it directly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ipu_emu.xmem import XMEM_SIZE_BYTES

from ipu_apps.kernel_registry import MalformedQuery, ShapeBundle

# -- Constants --------------------------------------------------------------

LANES = 128            # elements per XMEM row (fixed by the 128-element datapath)
ROW_BYTES = LANES * 4  # 512 bytes per FP32 row in wide-vector debug mode

XMEM_ROWS = XMEM_SIZE_BYTES // ROW_BYTES  # 16384

# Row 0 is deliberately left outside every region: an address that defaulted to
# zero is then detectably wrong rather than silently landing in the input.
BASE_ROW = 64

WIDE_VECTOR_ONLY = (
    "Wide-vector FP32 debug mode only (wide_vector_debug=True). These apps run "
    "the whole datapath in FP32 and have no narrow (INT8/FP8) variant."
)

# Below this the sigmoid is flat across any realistic score range, so the soft
# count is ~N/2 regardless of tau and cannot drive a bisection. It is also the
# bound that keeps the padding suppressor (tau - PAD_LOGIT/T) finite.
MIN_TEMPERATURE = 1e-6

# T * (pad - tau) is forced to exactly this, so the padding lanes' sigmoid
# underflows to a true zero and contributes nothing to the count. Large enough
# that exp() underflows, small enough that nothing overflows FP32 on the way.
PAD_LOGIT = 800.0

NO_RANKED_INDICES = (
    "returns thresholded values, not ranked indices: the ISA has no vector "
    "compare and no lane-index extraction, so selecting the k largest is host "
    "work on this output."
)


def check_finite(name: str, value) -> float:
    """Reject a non-finite threshold or temperature before it reaches XMEM.

    A NaN threshold makes every comparison false and every output zero, which
    looks like a working kernel on a quiet image; an infinite one saturates the
    subtract. Both are mistakes in the question rather than shapes no kernel
    covers, so they raise rather than becoming a refusal.
    """
    try:
        f = float(value)
    except (TypeError, ValueError) as exc:
        raise MalformedQuery(f"{name} must be a real number; got {value!r}") from exc
    if not math.isfinite(f):
        raise MalformedQuery(f"{name} must be finite; got {f!r}")
    return f


def _flat_extent(shape) -> tuple[tuple[int, ...], int]:
    """Total element count of ``shape``, with the dims normalised."""
    dims = tuple(int(d) for d in shape)
    if not dims:
        raise MalformedQuery("a shape must have at least one dimension")
    total = 1
    for d in dims:
        total *= d
    return dims, total


# -- score_threshold --------------------------------------------------------


@dataclass(frozen=True)
class ThresholdQuery:
    """A score_threshold query reduced to what the kernel routes on.

    The gate is element-wise, so the input's rank carries no meaning -- only
    how many elements there are. The shape is preserved in the bundle so the
    verdict still reports what the caller asked about.

    Attributes:
        dims:        The shape as given.
        elements:    Total element count.
        threshold:   tau.
        temperature: T, the sharpness of the soft survivor count.
        bundle:      The role-keyed shape bundle for the verdict.
    """

    dims: tuple[int, ...]
    elements: int
    threshold: float
    temperature: float
    bundle: ShapeBundle

    @property
    def rows(self) -> int:
        """XMEM rows the score map occupies: ``ceil(elements / 128)``."""
        return (self.elements + LANES - 1) // LANES

    @property
    def total_rows(self) -> int:
        # scores + selected + the staged sigmoid plane, plus three resident
        # vectors (tau, T*tau, T) and one row for the reduced count.
        return BASE_ROW + 3 * self.rows + 4

    @property
    def max_elements(self) -> int:
        """Largest element count that would fit the budget."""
        return max(0, (XMEM_ROWS - BASE_ROW - 4) // 3) * LANES


def threshold_query(shape, *, threshold, temperature) -> ThresholdQuery:
    """Normalise a score_threshold query.

    Raises:
        MalformedQuery: for an empty shape, or a non-finite threshold or
            temperature.
    """
    dims, total = _flat_extent(shape)
    tau = check_finite("threshold", threshold)
    temp = check_finite("temperature", temperature)
    bundle = ShapeBundle.of(input=dims).with_shapes(
        derived={"selected": dims, "count": (1,)}
    )
    return ThresholdQuery(
        dims=dims,
        elements=total,
        threshold=tau,
        temperature=temp,
        bundle=bundle,
    )


def threshold_refusal(q: ThresholdQuery) -> str | None:
    """Refuse anything outside the score_threshold kernel's domain."""
    if q.elements < 1:
        return f"shape {q.dims} holds no elements"
    if q.temperature < MIN_TEMPERATURE:
        return (
            f"the soft survivor count needs a temperature of at least "
            f"{MIN_TEMPERATURE:g}; got {q.temperature:g}. Below that the "
            f"sigmoid is flat across any realistic score range, so the count "
            f"is about half the element count whatever tau is and cannot drive "
            f"a bisection. A negative T would count the scores BELOW the "
            f"threshold instead."
        )
    if q.total_rows > XMEM_ROWS:
        return (
            f"needs {q.total_rows} XMEM rows (scores {q.rows} + selected "
            f"{q.rows} + staged sigmoid {q.rows} + 4 resident + {BASE_ROW} "
            f"reserved); wide-vector XMEM holds {XMEM_ROWS} rows of "
            f"{ROW_BYTES} B. Split the map into chunks of at most "
            f"{q.max_elements} elements; the gate is element-wise, so a chunk "
            f"needs no context, and the counts add."
        )
    return None


# -- channel_peak -----------------------------------------------------------


@dataclass(frozen=True)
class PeakQuery:
    """A channel_peak query reduced to what the kernel routes on.

    Attributes:
        dims:      The shape as given.
        channels:  Planes reduced over (the leading axis).
        cells:     Independent cells, i.e. the trailing axes flattened.
        threshold: tau.
        bundle:    The role-keyed shape bundle for the verdict.
    """

    dims: tuple[int, ...]
    channels: int
    cells: int
    threshold: float
    bundle: ShapeBundle

    @property
    def tiles(self) -> int:
        """XMEM rows one channel plane occupies: ``ceil(cells / 128)``."""
        return (self.cells + LANES - 1) // LANES

    @property
    def input_rows(self) -> int:
        return self.channels * self.tiles

    @property
    def total_rows(self) -> int:
        # input + confidence + keep + one resident tau row.
        return BASE_ROW + self.input_rows + 2 * self.tiles + 1

    @property
    def max_channels(self) -> int:
        """Largest channel count that would fit the budget at this cell count."""
        if self.tiles < 1:
            return 0
        return max(0, (XMEM_ROWS - BASE_ROW - 1 - 2 * self.tiles) // self.tiles)


def peak_query(shape, *, threshold) -> PeakQuery:
    """Normalise a channel_peak query.

    Accepts a rank-2 ``(C, N)`` or rank-3 ``(C, H, W)`` input; the trailing
    axes are flattened into the cell count, which is recorded as a note rather
    than done silently.

    Raises:
        MalformedQuery: for a rank outside 2..3, or a non-finite threshold.
    """
    dims, _ = _flat_extent(shape)
    if len(dims) not in (2, 3):
        raise MalformedQuery(
            f"channel_peak expects a rank-2 (C, N) or rank-3 (C, H, W) input; "
            f"got rank-{len(dims)} {dims}"
        )
    tau = check_finite("threshold", threshold)

    channels = dims[0]
    cells = 1
    for d in dims[1:]:
        cells *= d

    notes = ()
    if len(dims) == 3:
        notes = (
            f"flattened {dims} -> {(channels, cells)}: the reduction runs down "
            f"dim 0 and is independent across every other axis, so the spatial "
            f"axes are just cells",
        )
    bundle = ShapeBundle.of(input=dims).with_shapes(
        derived={"confidence": (cells,), "keep": (cells,)}, notes=notes
    )
    return PeakQuery(
        dims=dims, channels=channels, cells=cells, threshold=tau, bundle=bundle
    )


def peak_refusal(q: PeakQuery) -> str | None:
    """Refuse anything outside the channel_peak kernel's domain."""
    if q.channels < 1:
        return f"channels ({q.channels}) must be >= 1"
    if q.cells < 1:
        return f"cells ({q.cells}) must be >= 1"
    if q.total_rows > XMEM_ROWS:
        band = q.max_channels
        advice = (
            f"Reduce at most {band} channels per launch and combine the "
            f"partial maxima on the host, or split the cells."
            if band >= 1
            else "Even one channel plane does not fit; split the cells."
        )
        return (
            f"needs {q.total_rows} XMEM rows (input {q.input_rows} + "
            f"confidence {q.tiles} + keep {q.tiles} + 1 resident + {BASE_ROW} "
            f"reserved); wide-vector XMEM holds {XMEM_ROWS} rows of "
            f"{ROW_BYTES} B. {advice}"
        )
    return None


def headroom_caveat(total_rows: int) -> str:
    """Report the XMEM rows a query leaves unused."""
    return (
        f"uses {total_rows} of {XMEM_ROWS} XMEM rows "
        f"({XMEM_ROWS - total_rows} free)"
    )


def lane_caveat(elements: int, tiles: int) -> str | None:
    """Quantify the lanes an element count that does not fill its tiles wastes."""
    total = tiles * LANES
    if total == elements:
        return None
    return (
        f"{elements} element(s) occupy {tiles} XMEM row(s) ({total} lanes); "
        f"{total - elements} lanes idle ({elements / total:.0%} datapath "
        f"utilisation)"
    )
