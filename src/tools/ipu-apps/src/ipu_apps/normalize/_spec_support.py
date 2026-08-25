"""Shared helpers for the l2_normalize kernels' registry declarations.

An ``l2_normalize`` query is a shape plus the axis whose vectors are scaled to
unit length -- the same two parameters a softmax query carries, and normalised
the same way, so :func:`~ipu_apps.kernel_registry.flatten_to_matrix` does the
rank reduction here too. What differs is that the reduction axis must be the
*leading* one: the kernel keeps one scalar per independent column in a full
128-element vector, which is what removes the fan-out a row-wise reduction
would need.

Unlike softmax there are no framework-layer adapters. ``torch`` exposes L2
normalization as ``nn.functional.normalize`` -- a function, not a layer class --
so there is no class name for :func:`~ipu_apps.kernel_registry.register_layer`
to dispatch on. Registering an adapter for ``LocalResponseNorm`` or
``BatchNorm2d`` would be worse than none: they are not this operation.
"""

from __future__ import annotations

from dataclasses import dataclass

from ipu_emu.xmem import XMEM_SIZE_BYTES

from ipu_apps.kernel_registry import (
    MalformedQuery,
    ShapeBundle,
    flatten_to_matrix,
)

# -- Constants --------------------------------------------------------------

LANES = 128            # elements per XMEM row (fixed by the 128-element datapath)
ROW_BYTES = LANES * 4  # 512 bytes per FP32 row in wide-vector debug mode

# Wide-vector debug mode addresses the whole XMEM allocation as 512-byte rows.
# Derived from the emulator's own constant rather than restated.
XMEM_ROWS = XMEM_SIZE_BYTES // ROW_BYTES  # 16384

# Row 0 is deliberately left outside every region: an address that defaulted to
# zero is then detectably wrong rather than silently landing in the input.
BASE_ROW = 64

WIDE_VECTOR_ONLY = (
    "Wide-vector FP32 debug mode only (wide_vector_debug=True). These apps "
    "build on the rsqrt activation over an FP32 vector path and have no narrow "
    "(INT8/FP8) variant."
)


@dataclass(frozen=True)
class NormQuery:
    """An l2_normalize query reduced to what the kernels route on.

    Attributes:
        batch:        Leading batch extent (1 unless a rank-4 input was given).
        rows, cols:   The (possibly flattened) 2-D problem.
        down_columns: True when the norm reduces down each column (the leading
            axis), False when it reduces along each row.
        bundle:       The shape bundle, carrying any flatten or batch note.
    """

    batch: int
    rows: int
    cols: int
    down_columns: bool
    bundle: ShapeBundle

    @property
    def reduction_length(self) -> int:
        """How many elements each norm sums the squares of."""
        return self.rows if self.down_columns else self.cols

    @property
    def vectors(self) -> int:
        """How many independent vectors are normalised."""
        return self.cols if self.down_columns else self.rows

    @property
    def tiles_per_channel(self) -> int:
        """XMEM rows one row of the 2-D problem occupies: ``ceil(cols / 128)``."""
        return (self.cols + LANES - 1) // LANES

    @property
    def region_rows(self) -> int:
        """Rows in one full copy of the matrix (input and output are equal)."""
        return self.rows * self.tiles_per_channel

    @property
    def total_rows(self) -> int:
        """Every row a kernel touches: input + one scale vector + output."""
        return BASE_ROW + 2 * self.region_rows + 1

    @property
    def max_band_rows(self) -> int:
        """Largest ``rows`` that would fit the budget at this width.

        Reported in the over-budget refusal so the caller learns how to tile.
        Note this bands the *reduction* axis, which an L2 norm cannot simply be
        run over independently -- see :func:`xmem_refusal`.
        """
        per_row = 2 * self.tiles_per_channel
        if per_row < 1:
            return 0
        return max(0, (XMEM_ROWS - BASE_ROW - 1) // per_row)


def norm_query(shape, dim: int) -> NormQuery:
    """Normalise ``(shape, dim)`` into the form every l2_normalize kernel routes on.

    Accepts any rank. A rank-4 ``(N, C, H, W)`` input has its batch axis split
    off first, so ``dim=1`` -- the channel axis, and the one a descriptor
    normalization uses -- reduces to the leading axis of a rank-3 problem
    rather than hitting ``flatten_to_matrix``'s interior-axis refusal.

    Raises:
        MalformedQuery: if ``dim`` is out of range for ``shape``, names the
            batch axis of a rank-4 input, or is an interior axis that could
            only be flattened by transposing.
    """
    dims = tuple(int(d) for d in shape)
    ndim = len(dims)
    if ndim < 1:
        raise MalformedQuery("l2_normalize needs a shape with at least one axis")
    if not -ndim <= int(dim) < ndim:
        raise MalformedQuery(
            f"dim {dim} is out of range for a rank-{ndim} shape {dims}"
        )
    axis = int(dim) % ndim

    batch, inner, inner_axis, batch_note = 1, dims, axis, None
    if ndim == 4:
        if axis == 0:
            raise MalformedQuery(
                f"dim 0 of a rank-4 (N, C, H, W) shape {dims} is the batch "
                f"axis; scaling across images is not an L2 normalization of a "
                f"feature map"
            )
        batch, inner, inner_axis = dims[0], dims[1:], axis - 1
        batch_note = (
            f"read {dims} as a batch of {batch} rank-3 {inner} problem(s); dim "
            f"{axis} is dim {inner_axis} of each"
        )

    try:
        shape_2d, dim_2d, flat_note = flatten_to_matrix(inner, inner_axis)
    except ValueError as exc:
        # flatten_to_matrix refuses an interior axis rather than transposing.
        # That is a mistake in the question, so it propagates rather than
        # becoming "no kernel covers this".
        raise MalformedQuery(str(exc)) from exc

    rows, cols = shape_2d
    notes = tuple(n for n in (batch_note, flat_note) if n)
    bundle = ShapeBundle.of(input=dims).with_shapes(
        derived={"output": dims}, notes=notes
    )
    return NormQuery(
        batch=batch,
        rows=rows,
        cols=cols,
        down_columns=(dim_2d == 0),
        bundle=bundle,
    )


def geometry_refusal(q: NormQuery) -> str | None:
    """Refuse anything structurally outside every l2_normalize kernel's domain."""
    if q.batch != 1:
        return (
            f"normalizes one image per launch; this query has a batch of "
            f"{q.batch}. Run the kernel once per image."
        )
    if q.rows < 1:
        return f"rows ({q.rows}) must be >= 1"
    if q.cols < 1:
        return f"columns ({q.cols}) must be >= 1"
    return None


def xmem_refusal(q: NormQuery) -> str | None:
    """Refuse a query whose regions do not fit the wide-vector XMEM budget.

    The advice deliberately points at the *column* axis, not the reduction
    axis. Splitting the reduction would give each band its own partial sum of
    squares, which is not a normalization of anything -- the columns are the
    independent problems, so they are what may be tiled.
    """
    if q.total_rows <= XMEM_ROWS:
        return None
    fits = max(0, (XMEM_ROWS - BASE_ROW - 1) // (2 * q.rows)) * LANES
    advice = (
        f"Split the {q.cols} independent columns into chunks of at most "
        f"{fits}; the {q.rows}-element reduction axis cannot be split, because "
        f"a partial sum of squares does not normalize anything."
        if fits >= 1
        else "Even one column tile does not fit; reduce the reduction length."
    )
    return (
        f"needs {q.total_rows} XMEM rows (input {q.region_rows} + output "
        f"{q.region_rows} + 1 scale row + {BASE_ROW} reserved); wide-vector "
        f"XMEM holds {XMEM_ROWS} rows of {ROW_BYTES} B. {advice}"
    )


def lane_caveat(q: NormQuery) -> str | None:
    """Quantify the lanes a width that does not fill its tiles leaves idle."""
    total = q.tiles_per_channel * LANES
    if total == q.cols:
        return None
    return (
        f"width {q.cols} occupies {q.tiles_per_channel} XMEM row(s) per matrix "
        f"row ({total} lanes); {total - q.cols} lanes idle "
        f"({q.cols / total:.0%} datapath utilisation)"
    )


def headroom_caveat(q: NormQuery) -> str:
    """Report the XMEM rows this query leaves unused."""
    return (
        f"uses {q.total_rows} of {XMEM_ROWS} XMEM rows "
        f"({XMEM_ROWS - q.total_rows} free)"
    )
