"""Shared helpers for the conv2d kernels' registry declarations.

Every ``conv2d`` kernel answers the same query -- an activation shape, a weight
shape, and the four geometry parameters (``stride``/``padding``/``dilation``/
``groups``) -- so the parameter unpacking, the XMEM budget arithmetic and the
constants they reason about live here rather than being repeated per kernel.

Unlike softmax, a conv query is genuinely multi-tensor: the input, the weight,
the optional bias and the derived output all travel in one
:class:`~ipu_apps.kernel_registry.shapes.ShapeBundle`. Note that
``flatten_to_matrix`` is *not* used and must not be -- it is built around a
single reduction axis and raises on an interior one, which is every conv.

The framework-layer adapters live here too, for the same reason the specs live
beside their kernels: the op-agnostic registry should carry no conv vocabulary.
They register on import, and discovery imports this module, so
:func:`~ipu_apps.kernel_registry.lookup_layer` sees them.
"""

from __future__ import annotations

from dataclasses import dataclass

from ipu_apps.kernel_registry import (
    MalformedQuery,
    ShapeBundle,
    UnsupportedLayer,
    register_layer,
)

# -- Constants --------------------------------------------------------------

LANES = 128            # elements per XMEM row (fixed by the 128-element datapath)
ROW_BYTES = LANES * 4  # 512 bytes per FP32 row in wide-vector debug mode

# XMEM is allocated 8 MiB and wide-vector debug mode addresses all of it as
# 512-byte rows. Every conv region is sized in rows, so this is the single
# budget every kernel's `supports` checks against.
XMEM_ROWS = (8 << 20) // ROW_BYTES  # 16384

# Row 0 is deliberately left outside every region: an address that defaulted to
# zero is then detectably wrong rather than silently landing in the input.
BASE_ROW = 64

WIDE_VECTOR_ONLY = (
    "Wide-vector FP32 debug mode only (wide_vector_debug=True). These apps "
    "accumulate in FP32 through ACTIVATE.QUANTIZE and have no narrow "
    "(INT8/FP8) variant."
)


@dataclass(frozen=True)
class ConvQuery:
    """A conv2d query reduced to what the kernels route on.

    Attributes:
        batch:   Leading batch extent (1 for a rank-3 ``(Cin, H, W)`` query).
        cin:     Input channels.
        cout:    Output channels.
        height:  Input rows.
        width:   Input columns.
        kh, kw:  Weight spatial extent.
        stride, padding, dilation, groups: Convolution geometry.
        bias:    Whether a bias vector is present.
        weight_cin: The weight's declared input-channel extent, which must
            equal ``cin // groups``. Kept separate so a mismatch is reported
            rather than assumed away.
        bundle:  The role-keyed shape bundle for the verdict.
    """

    batch: int
    cin: int
    cout: int
    height: int
    width: int
    kh: int
    kw: int
    stride: int
    padding: int
    dilation: int
    groups: int
    bias: bool
    weight_cin: int
    bundle: ShapeBundle

    # -- derived XMEM geometry (all in 128-element rows) --------------------

    @property
    def tiles_per_row(self) -> int:
        """Column tiles per spatial row: ``ceil(width / 128)``.

        A spatial row occupies whole XMEM rows because addressing is
        row-granular, so a width that is not a multiple of 128 pads with idle
        lanes rather than packing tightly.
        """
        return (self.width + LANES - 1) // LANES

    @property
    def plane_stride(self) -> int:
        """Rows per channel plane -- the same for input and output."""
        return self.height * self.tiles_per_row

    @property
    def num_groups(self) -> int:
        """Channel groups: one ``LDR_MULT_REG`` row of weights covers 128 channels."""
        return (self.cin + LANES - 1) // LANES

    @property
    def input_rows(self) -> int:
        """Input region size, including the one guard plane the prefetch reads.

        The channel loop is software-pipelined: the last iteration prefetches
        channel ``cin``, whose data is never consumed but whose row must still
        be in bounds.
        """
        return (self.cin + 1) * self.plane_stride

    @property
    def weight_rows(self) -> int:
        return self.cout * self.num_groups

    @property
    def bias_rows(self) -> int:
        return self.cout

    @property
    def output_rows(self) -> int:
        return self.cout * self.plane_stride

    @property
    def total_rows(self) -> int:
        """Every row the kernel touches, including the reserved low region."""
        return (
            BASE_ROW
            + self.input_rows
            + self.weight_rows
            + self.bias_rows
            + self.output_rows
        )

    @property
    def max_band_height(self) -> int:
        """Largest ``height`` that would fit the XMEM budget at this width/channels.

        Reported in the over-budget refusal so the caller learns how to tile
        rather than only that it failed. Zero when even a single row overflows.
        """
        per_row = self.tiles_per_row * ((self.cin + 1) + self.cout)
        if per_row < 1:
            return 0
        fixed = BASE_ROW + self.weight_rows + self.bias_rows
        return max(0, (XMEM_ROWS - fixed) // per_row)


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


def conv_query(
    shape,
    weight_shape,
    *,
    stride,
    padding,
    dilation,
    groups,
    bias: bool = True,
) -> ConvQuery:
    """Normalise a conv2d query into the form every conv kernel routes on.

    Accepts a rank-3 ``(Cin, H, W)`` activation or a rank-4 ``(N, Cin, H, W)``
    one; the batch extent is carried through rather than folded away, so a
    kernel that cannot batch refuses it explicitly instead of silently
    computing the first image only.

    Raises:
        MalformedQuery: if the shapes are structurally not a convolution --
            wrong rank, or a non-pair geometry attribute. Those are mistakes in
            the question, which no kernel could answer.
    """
    dims = tuple(int(d) for d in shape)
    if len(dims) == 3:
        batch, (cin, height, width) = 1, dims
    elif len(dims) == 4:
        batch, cin, height, width = dims
    else:
        raise MalformedQuery(
            f"conv2d expects a rank-3 (Cin, H, W) or rank-4 (N, Cin, H, W) "
            f"input; got rank-{len(dims)} {dims}"
        )

    wdims = tuple(int(d) for d in weight_shape)
    if len(wdims) != 4:
        raise MalformedQuery(
            f"conv2d expects a rank-4 (Cout, Cin/groups, kh, kw) weight; got "
            f"rank-{len(wdims)} {wdims}"
        )
    cout, weight_cin, kh, kw = wdims

    sh, sw = _spatial_pair(stride, "stride")
    ph, pw = _spatial_pair(padding, "padding")
    dh, dw = _spatial_pair(dilation, "dilation")
    # Checked before the output extent is computed: a zero stride would divide
    # by zero, and a ZeroDivisionError is not a ValueError, so it would escape
    # `resolve` as a crash instead of a refusal.
    for label, value in (("stride", sh), ("dilation", dh), ("groups", int(groups))):
        if value < 1:
            raise MalformedQuery(f"{label} must be >= 1; got {value}")
    if ph < 0 or pw < 0:
        raise MalformedQuery(f"padding must be >= 0; got {padding!r}")
    if sh != sw or ph != pw or dh != dw:
        raise MalformedQuery(
            f"asymmetric geometry (stride={stride!r}, padding={padding!r}, "
            f"dilation={dilation!r}) is not modelled; no conv kernel "
            f"distinguishes the two spatial axes"
        )

    out_h = (height + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    out_w = (width + 2 * pw - dw * (kw - 1) - 1) // sw + 1

    in_shape = dims
    out_shape = (cout, out_h, out_w) if len(dims) == 3 else (batch, cout, out_h, out_w)
    roles = {"input": in_shape, "weight": wdims}
    if bias:
        roles["bias"] = (cout,)
    bundle = ShapeBundle.of(**roles).with_shapes(derived={"output": out_shape})

    return ConvQuery(
        batch=batch,
        cin=cin,
        cout=cout,
        height=height,
        width=width,
        kh=kh,
        kw=kw,
        stride=sh,
        padding=ph,
        dilation=dh,
        groups=int(groups),
        bias=bool(bias),
        weight_cin=weight_cin,
        bundle=bundle,
    )


def geometry_refusal(q: ConvQuery) -> str | None:
    """Refuse anything structurally outside every current conv kernel's domain.

    These are the checks shared by all conv2d kernels -- non-positive extents,
    a weight that disagrees with the input, and the geometry parameters none of
    them implement. A kernel's own ``supports`` adds its specific limits (its
    kernel size, its XMEM budget) on top.

    Enumerating stride/padding/dilation/groups here rather than assuming them
    is the point: a conv spec that silently ignores ``stride=2`` would answer
    confidently for an operation no kernel computes.
    """
    if q.batch != 1:
        return (
            f"processes one image per launch; this query has a batch of "
            f"{q.batch}. Run the kernel once per image."
        )
    for label, value in (
        ("input channels", q.cin),
        ("output channels", q.cout),
        ("height", q.height),
        ("width", q.width),
    ):
        if value < 1:
            return f"{label} ({value}) must be >= 1"
    if q.groups != 1:
        return (
            f"computes a dense convolution; this query asks for groups="
            f"{q.groups}. Grouped and depthwise convolutions are a different "
            f"dataflow and have no kernel yet."
        )
    if q.weight_cin != q.cin // q.groups:
        return (
            f"weight declares {q.weight_cin} input channels per group, but the "
            f"input has {q.cin} channels across {q.groups} group(s) "
            f"({q.cin // q.groups} per group)"
        )
    if q.stride != 1:
        return f"computes stride 1 only; this query asks for stride {q.stride}"
    if q.dilation != 1:
        return f"computes dilation 1 only; this query asks for dilation {q.dilation}"
    return None


def xmem_refusal(q: ConvQuery) -> str | None:
    """Refuse a query whose regions do not fit the wide-vector XMEM budget."""
    if q.total_rows <= XMEM_ROWS:
        return None
    band = q.max_band_height
    advice = (
        f"Tile the input into row bands of at most {band} rows."
        if band >= 1
        else "Even a single row does not fit; reduce the channel count or width."
    )
    return (
        f"needs {q.total_rows} XMEM rows (input {q.input_rows} incl. 1 guard "
        f"plane + weights {q.weight_rows} + bias {q.bias_rows} + output "
        f"{q.output_rows} + {BASE_ROW} reserved); wide-vector XMEM holds "
        f"{XMEM_ROWS} rows of {ROW_BYTES} B. {advice}"
    )


def lane_caveat(q: ConvQuery) -> str | None:
    """Quantify the idle lanes a non-multiple-of-128 width costs."""
    padded = q.tiles_per_row * LANES
    if padded == q.width:
        return None
    return (
        f"width {q.width} pads to {padded} elements per spatial row; "
        f"{padded - q.width} of every {padded} lanes idle "
        f"({q.width / padded:.0%} datapath utilisation)"
    )


def headroom_caveat(q: ConvQuery) -> str:
    """Report the XMEM rows this query leaves unused."""
    return (
        f"uses {q.total_rows} of {XMEM_ROWS} XMEM rows "
        f"({XMEM_ROWS - q.total_rows} free)"
    )


# -- framework-layer adapters -----------------------------------------------


@register_layer("Conv2d")
def _conv2d_layer(layer, input_shape):
    """``nn.Conv2d(...)`` -> the ``conv2d`` operation.

    Enumerates every attribute that changes what is computed and refuses the
    ones it cannot express, rather than ignoring them: a layer with
    ``padding='same'`` or a reflect padding mode computes something no conv
    kernel here implements, and answering for it would be confidently wrong.
    """
    expected = (
        "in_channels",
        "out_channels",
        "kernel_size",
        "stride",
        "padding",
        "dilation",
        "groups",
    )
    missing = [name for name in expected if not hasattr(layer, name)]
    if missing:
        raise UnsupportedLayer(
            f"{type(layer).__name__} is missing expected attribute(s) "
            f"{', '.join(missing)}; it does not look like the layer this "
            f"adapter was written for"
        )
    if isinstance(layer.padding, str):
        raise UnsupportedLayer(
            f"Conv2d(padding={layer.padding!r}) states padding as a policy "
            f"rather than an extent; resolve it to an integer padding first."
        )
    if getattr(layer, "padding_mode", "zeros") != "zeros":
        raise UnsupportedLayer(
            f"Conv2d(padding_mode={layer.padding_mode!r}) is not zero padding; "
            f"no conv kernel implements a non-zero border."
        )
    kh, kw = _spatial_pair(layer.kernel_size, "kernel_size")
    return "conv2d", {
        "shape": input_shape,
        "weight_shape": (
            int(layer.out_channels),
            int(layer.in_channels) // int(layer.groups),
            kh,
            kw,
        ),
        "stride": layer.stride,
        "padding": layer.padding,
        "dilation": layer.dilation,
        "groups": int(layer.groups),
        "bias": getattr(layer, "bias", None) is not None,
    }


@register_layer("ConvTranspose2d", "Conv1d", "Conv3d")
def _unsupported_conv_relatives(layer, input_shape):
    """Refuse near-neighbours of Conv2d explicitly.

    These sit beside ``Conv2d`` in ``torch.nn`` and expose the same attribute
    names, so a permissive adapter would route them to a conv2d kernel and
    return confidently wrong numbers.
    """
    name = type(layer).__name__
    detail = {
        "ConvTranspose2d": (
            "computes a transposed (fractionally strided) convolution, which "
            "scatters each input into the output rather than gathering"
        ),
        "Conv1d": "convolves over one spatial axis, not two",
        "Conv3d": "convolves over three spatial axes, not two",
    }[name]
    raise UnsupportedLayer(
        f"{name} {detail}; no kernel implements it. Using a conv2d kernel here "
        f"would return confidently wrong values."
    )
