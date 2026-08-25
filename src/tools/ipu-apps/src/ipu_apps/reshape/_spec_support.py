"""Shared helpers for the depth_to_space kernel's registry declaration.

A ``depth_to_space`` query is an activation shape plus the upscale factor. The
output shape follows from both, so it is *derived* and marked as such in the
bundle -- a wrong derivation must never read as something the caller asserted.

The ``PixelShuffle`` layer adapter lives here, and ``PixelUnshuffle`` is refused
by name: it is the exact inverse, shares the single ``upscale_factor``-shaped
attribute, and routing it here would return confidently wrong values.
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

XMEM_ROWS = XMEM_SIZE_BYTES // ROW_BYTES  # 16384

# Row 0 is deliberately left outside every region: an address that defaulted to
# zero is then detectably wrong rather than silently landing in the input.
BASE_ROW = 64

# ACC.RESHAPE moves exactly eight elements per instruction.
RESHAPE_ELEMENTS = 8

WIDE_VECTOR_ONLY = (
    "Wide-vector FP32 debug mode only (wide_vector_debug=True). Values are "
    "moved verbatim, but the 512-byte row granularity this relies on is a "
    "wide-vector property."
)


@dataclass(frozen=True)
class ShuffleQuery:
    """A depth_to_space query reduced to what the kernel routes on.

    Attributes:
        batch:      Leading batch extent (1 for a rank-3 ``(C, H, W)`` query).
        channels:   Input channels. Must be a multiple of ``r*r``.
        height, width: Input spatial extent.
        r:          Upscale factor.
        out_channels, out_height, out_width: Derived output extent.
        bundle:     The role-keyed shape bundle for the verdict.
    """

    batch: int
    channels: int
    height: int
    width: int
    r: int
    out_channels: int
    out_height: int
    out_width: int
    bundle: ShapeBundle

    @property
    def elements_per_plane(self) -> int:
        """Source elements one input plane contributes to one output tile.

        An output XMEM row is 128 output columns, which span ``128 / r`` input
        columns of each of the ``r`` planes that interleave into it.
        """
        return LANES // self.r

    @property
    def reshapes_per_plane(self) -> int:
        """``ACC.RESHAPE`` instructions needed to place one plane's elements."""
        return self.elements_per_plane // RESHAPE_ELEMENTS

    @property
    def in_tiles_per_row(self) -> int:
        """Input XMEM rows per input spatial row: ``ceil(W / 128)``."""
        return (self.width + LANES - 1) // LANES

    @property
    def out_tiles_per_row(self) -> int:
        """Output XMEM rows per output spatial row.

        Each input tile fans out to exactly ``r`` output tiles, so this follows
        the *padded* input width rather than the real output width -- which is
        why a width that does not fill its input tile wastes ``r`` times as many
        output lanes.
        """
        return self.in_tiles_per_row * self.r

    @property
    def in_plane_stride(self) -> int:
        return self.height * self.in_tiles_per_row

    @property
    def input_rows(self) -> int:
        return self.channels * self.in_plane_stride

    @property
    def output_rows(self) -> int:
        return self.out_channels * self.out_height * self.out_tiles_per_row

    @property
    def total_rows(self) -> int:
        return BASE_ROW + self.input_rows + self.output_rows

    @property
    def max_band_height(self) -> int:
        """Largest input ``height`` that would fit the budget at this width."""
        per_row = self.channels * self.in_tiles_per_row + (
            self.out_channels * self.r * self.out_tiles_per_row
        )
        if per_row < 1:
            return 0
        return max(0, (XMEM_ROWS - BASE_ROW) // per_row)


def shuffle_query(shape, *, upscale_factor) -> ShuffleQuery:
    """Normalise a depth_to_space query into the form the kernel routes on.

    Accepts a rank-3 ``(C, H, W)`` activation or a rank-4 ``(N, C, H, W)`` one;
    the batch extent is carried through rather than folded away.

    Raises:
        MalformedQuery: if the query is structurally not a pixel shuffle --
            wrong rank, or a non-positive upscale factor. Those are mistakes in
            the question, which no kernel could answer.
    """
    dims = tuple(int(d) for d in shape)
    if len(dims) == 3:
        batch, (channels, height, width) = 1, dims
    elif len(dims) == 4:
        batch, channels, height, width = dims
    else:
        raise MalformedQuery(
            f"depth_to_space expects a rank-3 (C, H, W) or rank-4 (N, C, H, W) "
            f"input; got rank-{len(dims)} {dims}"
        )

    r = int(upscale_factor)
    if r < 1:
        raise MalformedQuery(f"upscale_factor must be >= 1; got {r}")

    out_c = channels // (r * r) if r * r else 0
    out_shape = (
        (out_c, height * r, width * r)
        if len(dims) == 3
        else (batch, out_c, height * r, width * r)
    )
    bundle = ShapeBundle.of(input=dims).with_shapes(
        derived={"output": tuple(max(d, 0) for d in out_shape)}
    )

    return ShuffleQuery(
        batch=batch,
        channels=channels,
        height=height,
        width=width,
        r=r,
        out_channels=out_c,
        out_height=height * r,
        out_width=width * r,
        bundle=bundle,
    )


def geometry_refusal(q: ShuffleQuery) -> str | None:
    """Refuse anything structurally outside the depth_to_space kernel's domain."""
    if q.batch != 1:
        return (
            f"shuffles one image per launch; this query has a batch of "
            f"{q.batch}. Run the kernel once per image."
        )
    for label, value in (
        ("channels", q.channels),
        ("height", q.height),
        ("width", q.width),
    ):
        if value < 1:
            return f"{label} ({value}) must be >= 1"
    if q.channels % (q.r * q.r) != 0:
        return (
            f"a factor-{q.r} shuffle consumes {q.r * q.r} channels per output "
            f"channel; {q.channels} is not a multiple of {q.r * q.r}"
        )
    return None


def xmem_refusal(q: ShuffleQuery) -> str | None:
    """Refuse a query whose regions do not fit the wide-vector XMEM budget."""
    if q.total_rows <= XMEM_ROWS:
        return None
    band = q.max_band_height
    advice = (
        f"Tile the input into row bands of at most {band} rows; the shuffle is "
        f"independent across input rows, so a band needs no halo."
        if band >= 1
        else "Even a single row does not fit; reduce the channel count or width."
    )
    return (
        f"needs {q.total_rows} XMEM rows (input {q.input_rows} + output "
        f"{q.output_rows} + {BASE_ROW} reserved); wide-vector XMEM holds "
        f"{XMEM_ROWS} rows of {ROW_BYTES} B. {advice}"
    )


def lane_caveat(q: ShuffleQuery) -> str | None:
    """Quantify the output lanes a width that does not fill its input tile wastes.

    Worth stating loudly: the waste is multiplied by ``r``. Each input tile fans
    out to ``r`` output tiles whether or not it is full, so 48 idle input lanes
    become 384 idle output lanes at ``r = 8``.
    """
    total = q.out_tiles_per_row * LANES
    if total == q.out_width:
        return None
    padded_in = q.in_tiles_per_row * LANES
    return (
        f"input width {q.width} pads to {padded_in}, and each input tile fans "
        f"out to {q.r} output tiles, so the output row holds {total} lanes for "
        f"{q.out_width} real columns -- {total - q.out_width} idle "
        f"({q.out_width / total:.0%} datapath utilisation). Padding the input "
        f"width to a multiple of {LANES} costs nothing extra here."
    )


def headroom_caveat(q: ShuffleQuery) -> str:
    """Report the XMEM rows this query leaves unused."""
    return (
        f"uses {q.total_rows} of {XMEM_ROWS} XMEM rows "
        f"({XMEM_ROWS - q.total_rows} free)"
    )


# -- framework-layer adapters -----------------------------------------------


@register_layer("PixelShuffle")
def _pixel_shuffle_layer(layer, input_shape):
    """``nn.PixelShuffle(r)`` -> the ``depth_to_space`` operation."""
    if not hasattr(layer, "upscale_factor"):
        raise UnsupportedLayer(
            f"{type(layer).__name__} is missing expected attribute(s) "
            f"upscale_factor; it does not look like the layer this adapter was "
            f"written for"
        )
    return "depth_to_space", {
        "shape": input_shape,
        "upscale_factor": int(layer.upscale_factor),
    }


@register_layer("PixelUnshuffle")
def _unsupported_shuffle_relatives(layer, input_shape):
    """Refuse the exact inverse explicitly.

    ``PixelUnshuffle`` sits beside ``PixelShuffle`` in ``torch.nn`` and exposes
    a single downscale factor in the same shape, so a permissive adapter would
    route it here and return confidently wrong values -- it moves space into
    depth, not depth into space.
    """
    raise UnsupportedLayer(
        "PixelUnshuffle moves space into depth, the exact inverse of "
        "depth_to_space; no kernel implements it. Using a depth_to_space "
        "kernel here would return confidently wrong values."
    )
