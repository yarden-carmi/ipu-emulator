"""Shared helpers for the maxpool2d kernels' registry declarations.

Every ``maxpool2d`` kernel answers the same query -- an activation shape plus
the three window parameters (``kernel_size``/``stride``/``padding``) -- so the
parameter unpacking, the XMEM budget arithmetic and the constants they reason
about live here rather than being repeated per kernel.

Two dataclasses, deliberately separate, for the same reason the conv family
splits them:

``PoolQuery``   what the caller asked for. Shapes and window geometry, nothing
                about how a kernel would lay it out in XMEM.
``PoolLayout``  how one kernel would lay that out. Built with
                :meth:`PoolQuery.layout`, which takes the four things the
                pooling kernels actually differ on.

The split earns its keep here because the two kernels genuinely disagree about
every one of those four numbers. A stride-1 window pool reads one input XMEM
row per output row and spends ``k-1`` lanes on a horizontal halo; a stride-2
halving pool reads *two* full-width input rows per output row, spends no lanes
on a halo, and needs staging rows for the ``ACC.STRIDE`` decimation. Sharing
the budget arithmetic while parameterising those numbers keeps one copy of the
XMEM accounting.

The framework-layer adapters live here too, for the same reason the specs live
beside their kernels: the op-agnostic registry should carry no pooling
vocabulary. They register on import, and discovery imports this module, so
:func:`~ipu_apps.kernel_registry.lookup_layer` sees them.
"""

from __future__ import annotations

from dataclasses import dataclass

from ipu_emu.xmem import XMEM_SIZE_BYTES

from ipu_apps.kernel_registry import (
    MalformedQuery,
    ShapeBundle,
    UnsupportedLayer,
    register_layer,
)

# -- Constants --------------------------------------------------------------

LANES = 128            # elements per XMEM row (fixed by the 128-element datapath)
ROW_BYTES = LANES * 4  # 512 bytes per FP32 row in wide-vector debug mode

# Wide-vector debug mode addresses the whole XMEM allocation as 512-byte rows.
# Derived from the emulator's own constant rather than restated: a second copy
# of the memory size would silently make every refusal below wrong if XMEM were
# ever resized.
XMEM_ROWS = XMEM_SIZE_BYTES // ROW_BYTES  # 16384

# Row 0 is deliberately left outside every region: an address that defaulted to
# zero is then detectably wrong rather than silently landing in the input.
BASE_ROW = 64

# The identity element of a maximum, as an FP32 value that survives the
# ``x * 1.0`` identity multiply the taps use. Written into every lane a kernel
# reads but that holds no real input -- the border of a padded plane, the
# columns past W in a partly-filled tile, and the guard tiles that make a
# horizontal shift legal at the end of a spatial row.
#
# -FLT_MAX rather than -inf: -inf * 1.0 is -inf, which is fine, but -inf would
# also propagate through any later arithmetic on a pooled plane, and a pool
# whose unused lanes hold -inf is harder to debug than one whose unused lanes
# hold a merely very negative number.
NEG_FILL = -3.4028234663852886e38

WIDE_VECTOR_ONLY = (
    "Wide-vector FP32 debug mode only (wide_vector_debug=True). These apps "
    "run the whole datapath in FP32 and have no narrow (INT8/FP8) variant."
)


@dataclass(frozen=True)
class PoolLayout:
    """How one kernel would place a query's tensors in XMEM, in 128-element rows.

    Attributes:
        query: The query being laid out.
        out_tile_cols: Valid output columns per output XMEM row. 128 when every
            lane is a usable output; fewer when the kernel spends lanes on a
            horizontal halo.
        in_tiles_per_out_tile: Input XMEM rows, within one spatial row, that one
            output tile reads. 1 for a halo-tiled stride-1 kernel; 2 for a
            stride-2 kernel, whose 128 output columns span 256 input columns.
        guard_tiles: Extra input XMEM rows appended to each spatial row so that
            a read shifted by one element past the last real tile still lands
            inside that spatial row. Filled with :data:`NEG_FILL`.
        pad_rows: Rows of :data:`NEG_FILL` added above *and* below each input
            plane, so a kernel reading a vertical neighbourhood needs no border
            special case. 0 for a kernel that does not pad.
        scratch_rows: Staging rows the kernel needs on top of its regions.
    """

    query: "PoolQuery"
    out_tile_cols: int
    in_tiles_per_out_tile: int
    guard_tiles: int
    pad_rows: int
    scratch_rows: int

    @property
    def out_tiles_per_row(self) -> int:
        """Output XMEM rows per output spatial row."""
        return (
            self.query.out_width + self.out_tile_cols - 1
        ) // self.out_tile_cols

    @property
    def in_tiles_per_row(self) -> int:
        """Input XMEM rows per input spatial row, guard tiles included.

        Sized from what the kernel *reads*, not from ``ceil(W / 128)``: the last
        output tile of a row may be only partly filled, and the input tiles
        feeding its unused lanes still have to exist.
        """
        return (
            self.in_tiles_per_out_tile * self.out_tiles_per_row + self.guard_tiles
        )

    @property
    def tiles_holding_width(self) -> int:
        """Input XMEM rows that carry real image columns: ``ceil(W / 128)``."""
        return (self.query.width + LANES - 1) // LANES

    @property
    def padded_height(self) -> int:
        """Input plane rows including the border above and below."""
        return self.query.height + 2 * self.pad_rows

    @property
    def in_plane_stride(self) -> int:
        """Rows per input channel plane (border included)."""
        return self.padded_height * self.in_tiles_per_row

    @property
    def out_plane_stride(self) -> int:
        """Rows per output channel plane."""
        return self.query.out_height * self.out_tiles_per_row

    @property
    def input_rows(self) -> int:
        return self.query.channels * self.in_plane_stride

    @property
    def output_rows(self) -> int:
        return self.query.channels * self.out_plane_stride

    @property
    def total_rows(self) -> int:
        """Every row the kernel touches, including the reserved low region."""
        return (
            BASE_ROW + self.input_rows + self.scratch_rows + self.output_rows
        )


@dataclass(frozen=True)
class PoolQuery:
    """A maxpool2d query reduced to what the kernels route on.

    Attributes:
        batch:      Leading batch extent (1 for a rank-3 ``(C, H, W)`` query).
        channels:   Channel count. Pooling is per-channel, so this is only a
                    loop bound -- but it drives the XMEM budget.
        height, width: Input spatial extent.
        kernel:     Window extent (square).
        stride:     Window step (square).
        padding:    Border added on all four sides (square).
        out_height, out_width: Derived output extent.
        bundle:     The role-keyed shape bundle for the verdict.
    """

    batch: int
    channels: int
    height: int
    width: int
    kernel: int
    stride: int
    padding: int
    out_height: int
    out_width: int
    bundle: ShapeBundle

    def layout(
        self,
        *,
        out_tile_cols: int,
        in_tiles_per_out_tile: int,
        guard_tiles: int,
        pad_rows: int,
        scratch_rows: int,
    ) -> PoolLayout:
        """Describe how a kernel with this tiling would place the tensors."""
        return PoolLayout(
            query=self,
            out_tile_cols=out_tile_cols,
            in_tiles_per_out_tile=in_tiles_per_out_tile,
            guard_tiles=guard_tiles,
            pad_rows=pad_rows,
            scratch_rows=scratch_rows,
        )


def _spatial_pair(value, name: str) -> tuple[int, int]:
    """Normalise a torch-style int-or-pair geometry attribute."""
    if isinstance(value, int):
        return int(value), int(value)
    try:
        pair = tuple(int(v) for v in value)
    except TypeError as exc:
        raise MalformedQuery(f"{name} must be an int or a pair; got {value!r}") from exc
    if len(pair) != 2:
        raise MalformedQuery(f"{name} must be an int or a pair; got {value!r}")
    return pair


def pool_query(shape, *, kernel_size, stride, padding) -> PoolQuery:
    """Normalise a maxpool2d query into the form every pooling kernel routes on.

    Accepts a rank-3 ``(C, H, W)`` activation or a rank-4 ``(N, C, H, W)`` one;
    the batch extent is carried through rather than folded away, so a kernel
    that cannot batch refuses it explicitly instead of silently pooling the
    first image only.

    Raises:
        MalformedQuery: if the query is structurally not a 2-D pool -- wrong
            rank, a non-pair geometry attribute, a non-positive kernel or
            stride, a negative padding, or an asymmetric window. Those are
            mistakes in the question, which no kernel could answer.
    """
    dims = tuple(int(d) for d in shape)
    if len(dims) == 3:
        batch, (channels, height, width) = 1, dims
    elif len(dims) == 4:
        batch, channels, height, width = dims
    else:
        raise MalformedQuery(
            f"maxpool2d expects a rank-3 (C, H, W) or rank-4 (N, C, H, W) "
            f"input; got rank-{len(dims)} {dims}"
        )

    kh, kw = _spatial_pair(kernel_size, "kernel_size")
    sh, sw = _spatial_pair(stride, "stride")
    ph, pw = _spatial_pair(padding, "padding")

    # Checked before the output extent is computed: a zero stride would divide
    # by zero, and a ZeroDivisionError is not a ValueError, so it would escape
    # `resolve` as a crash instead of a refusal.
    for label, value in (("kernel_size", kh), ("stride", sh)):
        if value < 1:
            raise MalformedQuery(f"{label} must be >= 1; got {value}")
    if ph < 0:
        raise MalformedQuery(f"padding must be >= 0; got {padding!r}")
    if kh != kw or sh != sw or ph != pw:
        raise MalformedQuery(
            f"asymmetric geometry (kernel_size={kernel_size!r}, stride="
            f"{stride!r}, padding={padding!r}) is not modelled; no pooling "
            f"kernel distinguishes the two spatial axes"
        )

    out_h = (height + 2 * ph - kh) // sh + 1
    out_w = (width + 2 * pw - kw) // sw + 1

    out_shape = (
        (channels, out_h, out_w)
        if len(dims) == 3
        else (batch, channels, out_h, out_w)
    )
    bundle = ShapeBundle.of(input=dims).with_shapes(
        derived={"output": tuple(max(d, 0) for d in out_shape)}
    )

    return PoolQuery(
        batch=batch,
        channels=channels,
        height=height,
        width=width,
        kernel=kh,
        stride=sh,
        padding=ph,
        out_height=out_h,
        out_width=out_w,
        bundle=bundle,
    )


def geometry_refusal(q: PoolQuery) -> str | None:
    """Refuse anything structurally outside every current pooling kernel's domain.

    These are the checks shared by both maxpool2d kernels -- non-positive
    extents, a batch, and a window that does not fit the padded image. A
    kernel's own ``supports`` adds its specific limits (its stride, its kernel
    size, its XMEM budget) on top.
    """
    if q.batch != 1:
        return (
            f"pools one image per launch; this query has a batch of {q.batch}. "
            f"Run the kernel once per image."
        )
    for label, value in (
        ("channels", q.channels),
        ("height", q.height),
        ("width", q.width),
    ):
        if value < 1:
            return f"{label} ({value}) must be >= 1"
    if q.out_height < 1 or q.out_width < 1:
        return (
            f"a {q.kernel}x{q.kernel} window with padding {q.padding} does not "
            f"fit a {q.height}x{q.width} image: the output would be "
            f"{q.out_height}x{q.out_width}"
        )
    return None


def xmem_refusal(layout: PoolLayout) -> str | None:
    """Refuse a query whose regions do not fit the wide-vector XMEM budget.

    A backstop, not a routine limit. XMEM is sized so every layer runs in one
    launch; this exists so a shape that genuinely cannot fit is refused with the
    arithmetic rather than crashing inside a store instruction thousands of
    cycles in. There is no row-band tiling to fall back on -- the kernels
    process whatever extent they are given, in one launch.
    """
    if layout.total_rows <= XMEM_ROWS:
        return None
    return (
        f"needs {layout.total_rows} XMEM rows (input {layout.input_rows} + "
        f"output {layout.output_rows} + scratch {layout.scratch_rows} + "
        f"{BASE_ROW} reserved); wide-vector XMEM holds {XMEM_ROWS} rows of "
        f"{ROW_BYTES} B."
    )


def lane_caveat(layout: PoolLayout) -> str | None:
    """Quantify the lanes an output width that does not fill its tiles leaves idle."""
    total = layout.out_tiles_per_row * LANES
    if total == layout.query.out_width:
        return None
    return (
        f"output width {layout.query.out_width} occupies "
        f"{layout.out_tiles_per_row} XMEM row(s) per spatial row ({total} "
        f"lanes, {layout.out_tile_cols} usable each); "
        f"{total - layout.query.out_width} lanes idle "
        f"({layout.query.out_width / total:.0%} datapath utilisation)"
    )


def headroom_caveat(layout: PoolLayout) -> str:
    """Report the XMEM rows this query leaves unused."""
    return (
        f"uses {layout.total_rows} of {XMEM_ROWS} XMEM rows "
        f"({XMEM_ROWS - layout.total_rows} free)"
    )


# -- framework-layer adapters -----------------------------------------------


@register_layer("MaxPool2d")
def _maxpool2d_layer(layer, input_shape):
    """``nn.MaxPool2d(...)`` -> the ``maxpool2d`` operation.

    Enumerates every attribute that changes what is computed and refuses the
    ones it cannot express, rather than ignoring them. ``ceil_mode`` changes the
    output extent, ``dilation`` changes which pixels the window covers, and
    ``return_indices`` asks for a second output no kernel produces -- answering
    for any of them would be confidently wrong.

    ``stride=None`` means "same as kernel_size" in torch, which is a real
    default rather than an unmodelled case, so it is resolved here.
    """
    expected = ("kernel_size", "stride", "padding")
    missing = [name for name in expected if not hasattr(layer, name)]
    if missing:
        raise UnsupportedLayer(
            f"{type(layer).__name__} is missing expected attribute(s) "
            f"{', '.join(missing)}; it does not look like the layer this "
            f"adapter was written for"
        )
    if getattr(layer, "return_indices", False):
        raise UnsupportedLayer(
            "MaxPool2d(return_indices=True) also returns the argmax positions; "
            "the ISA has no lane-index extraction, so no kernel produces them."
        )
    if getattr(layer, "ceil_mode", False):
        raise UnsupportedLayer(
            "MaxPool2d(ceil_mode=True) rounds the output extent up and pools "
            "partial windows at the edge; no kernel implements that."
        )
    dilation = getattr(layer, "dilation", 1)
    dh, _ = _spatial_pair(dilation, "dilation")
    if dh != 1:
        raise UnsupportedLayer(
            f"MaxPool2d(dilation={dilation!r}) spreads the window out; no "
            f"kernel implements a dilated pool."
        )
    stride = layer.stride if layer.stride is not None else layer.kernel_size
    return "maxpool2d", {
        "shape": input_shape,
        "kernel_size": layer.kernel_size,
        "stride": stride,
        "padding": layer.padding,
    }


@register_layer("AvgPool2d", "MaxPool1d", "MaxPool3d", "AdaptiveMaxPool2d")
def _unsupported_pool_relatives(layer, input_shape):
    """Refuse near-neighbours of MaxPool2d explicitly.

    These sit beside ``MaxPool2d`` in ``torch.nn`` and expose the same attribute
    names, so a permissive adapter would route them to a maxpool2d kernel and
    return confidently wrong numbers.
    """
    name = type(layer).__name__
    detail = {
        "AvgPool2d": "averages its window instead of taking the maximum",
        "MaxPool1d": "pools over one spatial axis, not two",
        "MaxPool3d": "pools over three spatial axes, not two",
        "AdaptiveMaxPool2d": (
            "derives its window from the requested output size at run time, so "
            "it has no fixed kernel_size or stride to route on"
        ),
    }[name]
    raise UnsupportedLayer(
        f"{name} {detail}; no kernel implements it. Using a maxpool2d kernel "
        f"here would return confidently wrong values."
    )
