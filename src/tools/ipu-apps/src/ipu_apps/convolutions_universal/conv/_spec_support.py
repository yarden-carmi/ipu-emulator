"""Shared helpers for the conv2d kernels' registry declarations.

Every ``conv2d`` kernel answers the same query -- an activation shape, a weight
shape, the four geometry parameters (``stride``/``padding``/``dilation``/
``groups``) and the fused activation function -- so the parameter unpacking,
the XMEM budget arithmetic and the constants they reason about live here rather
than being repeated per kernel.

Unlike softmax, a conv query is genuinely multi-tensor: the input, the weight,
the optional bias and the derived output all travel in one
:class:`~ipu_apps.kernel_registry.shapes.ShapeBundle`. Note that
``flatten_to_matrix`` is *not* used and must not be -- it is built around a
single reduction axis and raises on an interior one, which is every conv.

Two dataclasses, deliberately separate:

``ConvQuery``   what the caller asked for. Shapes and geometry, nothing about
                how a kernel would lay it out in XMEM.
``ConvLayout``  how one kernel would lay that out. Built with
                :meth:`ConvQuery.layout`, which takes the two things kernels
                actually differ on -- how many of a 128-element XMEM row are
                valid output columns, and how many zero rows border a plane.

The split matters because the pointwise and shifted-tap kernels genuinely
disagree: 1x1 reads one element per output element, so all 128 lanes are
usable and no border is needed; 3x3 spends one element at each end of a row on
the horizontal halo and needs a zero row above and below. Sharing the budget
arithmetic while parameterising those two numbers keeps one copy of the
XMEM accounting.

The framework-layer adapters live here too, for the same reason the specs live
beside their kernels: the op-agnostic registry should carry no conv vocabulary.
They register on import, and discovery imports this module, so
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

# Activation functions a conv kernel can fuse into its ACTIVATE.QUANTIZE.
# ACTIVATE.QUANTIZE takes the function as an *immediate*, not a CR, so it is
# fixed when the .asm is assembled -- a kernel implements exactly one, and the
# query has to say which is wanted or the registry would hand a caller asking
# for a plain convolution one that also applies ReLU.
NO_ACTIVATION = "none"
KNOWN_ACTIVATIONS = (NO_ACTIVATION, "relu")

WIDE_VECTOR_ONLY = (
    "Wide-vector FP32 debug mode only (wide_vector_debug=True). These apps "
    "accumulate in FP32 through ACTIVATE.QUANTIZE and have no narrow "
    "(INT8/FP8) variant."
)


@dataclass(frozen=True)
class ConvLayout:
    """How one kernel would place a query's tensors in XMEM, in 128-element rows.

    Attributes:
        query:     The query being laid out.
        tile_cols: Valid output columns per XMEM row. 128 when every lane is a
            usable output; fewer when the kernel spends lanes on a halo.
        pad_rows:  Zero rows added above *and* below each input plane, so a
            kernel reading a vertical neighbourhood needs no border special
            case. 0 for a kernel with no vertical window.
        weights_per_channel: Weight elements one input channel contributes to
            one output channel (``kh*kw``). Sets the channel-group size, since
            one ``LDR_MULT_REG`` row holds 128 of them.
    """

    query: "ConvQuery"
    tile_cols: int
    pad_rows: int
    weights_per_channel: int

    @property
    def tiles_per_row(self) -> int:
        """XMEM rows per spatial row: ``ceil(W / tile_cols)``."""
        return (self.query.width + self.tile_cols - 1) // self.tile_cols

    @property
    def padded_height(self) -> int:
        """Input plane rows including the zero border above and below."""
        return self.query.height + 2 * self.pad_rows

    @property
    def in_plane_stride(self) -> int:
        """Rows per input channel plane (border included)."""
        return self.padded_height * self.tiles_per_row

    @property
    def out_plane_stride(self) -> int:
        """Rows per output channel plane (no border)."""
        return self.query.height * self.tiles_per_row

    @property
    def group_cap(self) -> int:
        """Input channels whose weights fit one 128-element ``R0`` row."""
        return LANES // self.weights_per_channel

    @property
    def num_groups(self) -> int:
        return (self.query.cin + self.group_cap - 1) // self.group_cap

    @property
    def input_rows(self) -> int:
        """Input region, including the one guard plane the prefetch reads.

        The channel loop is software-pipelined or reads one channel ahead, so
        its last iteration touches channel ``cin``, whose data is never
        consumed but whose row must still be in bounds.
        """
        return (self.query.cin + 1) * self.in_plane_stride

    @property
    def weight_rows(self) -> int:
        return self.query.cout * self.num_groups

    @property
    def bias_rows(self) -> int:
        return self.query.cout

    @property
    def output_rows(self) -> int:
        return self.query.cout * self.out_plane_stride

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


@dataclass(frozen=True)
class ConvQuery:
    """A conv2d query reduced to what the kernels route on.

    Attributes:
        batch:      Leading batch extent (1 for a rank-3 ``(Cin, H, W)`` query).
        cin, cout:  Input and output channels.
        height, width: Input spatial extent.
        kh, kw:     Weight spatial extent.
        stride, padding, dilation, groups: Convolution geometry.
        bias:       Whether a bias vector is present.
        activation: Activation fused into the kernel's store, or ``"none"``.
        weight_cin: The weight's declared input-channel extent, which must
            equal ``cin // groups``. Kept separate so a mismatch is reported
            rather than assumed away.
        bundle:     The role-keyed shape bundle for the verdict.
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
    activation: str
    weight_cin: int
    bundle: ShapeBundle

    def layout(self, *, tile_cols: int, pad_rows: int) -> ConvLayout:
        """Describe how a kernel with this tiling would place the tensors."""
        return ConvLayout(
            query=self,
            tile_cols=tile_cols,
            pad_rows=pad_rows,
            weights_per_channel=self.kh * self.kw,
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


def conv_query(
    shape,
    weight_shape,
    *,
    stride,
    padding,
    dilation,
    groups,
    bias: bool = True,
    activation: str = NO_ACTIVATION,
) -> ConvQuery:
    """Normalise a conv2d query into the form every conv kernel routes on.

    Accepts a rank-3 ``(Cin, H, W)`` activation or a rank-4 ``(N, Cin, H, W)``
    one; the batch extent is carried through rather than folded away, so a
    kernel that cannot batch refuses it explicitly instead of silently
    computing the first image only.

    Raises:
        MalformedQuery: if the query is structurally not a convolution --
            wrong rank, a non-pair geometry attribute, a non-positive stride,
            or an unknown activation name. Those are mistakes in the question,
            which no kernel could answer.
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

    if activation not in KNOWN_ACTIVATIONS:
        raise MalformedQuery(
            f"unknown activation {activation!r}; conv2d kernels fuse one of "
            f"{', '.join(repr(a) for a in KNOWN_ACTIVATIONS)}"
        )

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
        activation=activation,
        weight_cin=weight_cin,
        bundle=bundle,
    )


def geometry_refusal(q: ConvQuery) -> str | None:
    """Refuse anything structurally outside every current conv kernel's domain.

    These are the checks shared by all conv2d kernels -- non-positive extents,
    a weight that disagrees with the input, and the geometry parameters none of
    them implement. A kernel's own ``supports`` adds its specific limits (its
    kernel size, its activation, its XMEM budget) on top.

    Enumerating stride/dilation/groups here rather than assuming them is the
    point: a conv spec that silently ignored ``stride=2`` would answer
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


def activation_refusal(q: ConvQuery, implemented: str) -> str | None:
    """Refuse a query whose fused activation is not the one this kernel applies.

    ``ACTIVATE.QUANTIZE`` takes its function as an immediate, so a kernel
    implements exactly one and cannot be asked for another at run time. Routing
    across that difference would return confidently wrong values -- a caller
    asking for a plain convolution would get one with ReLU folded in.
    """
    if q.activation == implemented:
        return None
    if implemented == NO_ACTIVATION:
        return (
            f"applies no activation; this query asks for {q.activation!r} fused "
            f"into the store. ACTIVATE.QUANTIZE takes its function as an "
            f"immediate, so it is fixed when the .asm is assembled."
        )
    return (
        f"fuses {implemented!r} into its store and cannot be asked for "
        f"{q.activation!r}; ACTIVATE.QUANTIZE takes its function as an "
        f"immediate, so it is fixed when the .asm is assembled."
    )


def xmem_refusal(layout: ConvLayout) -> str | None:
    """Refuse a query whose regions do not fit the wide-vector XMEM budget.

    A backstop, not a routine limit. XMEM is sized so every SuperPoint layer
    runs in one launch; this exists so a shape that genuinely cannot fit is
    refused with the arithmetic rather than crashing inside a store instruction
    thousands of cycles in. There is no row-band tiling to fall back on -- the
    kernels process whatever height they are given, in one launch.
    """
    if layout.total_rows <= XMEM_ROWS:
        return None
    return (
        f"needs {layout.total_rows} XMEM rows (input {layout.input_rows} incl. "
        f"1 guard plane + weights {layout.weight_rows} + bias "
        f"{layout.bias_rows} + output {layout.output_rows} + {BASE_ROW} "
        f"reserved); wide-vector XMEM holds {XMEM_ROWS} rows of {ROW_BYTES} B."
    )


def lane_caveat(layout: ConvLayout) -> str | None:
    """Quantify the lanes a width that does not fill its tiles leaves idle."""
    usable = layout.tiles_per_row * layout.tile_cols
    total = layout.tiles_per_row * LANES
    if usable == layout.query.width and total == usable:
        return None
    return (
        f"width {layout.query.width} occupies {layout.tiles_per_row} XMEM row(s) "
        f"per spatial row ({total} lanes, {layout.tile_cols} usable each); "
        f"{total - layout.query.width} lanes idle "
        f"({layout.query.width / total:.0%} datapath utilisation)"
    )


def headroom_caveat(layout: ConvLayout) -> str:
    """Report the XMEM rows this query leaves unused."""
    return (
        f"uses {layout.total_rows} of {XMEM_ROWS} XMEM rows "
        f"({XMEM_ROWS - layout.total_rows} free)"
    )


# -- framework-layer adapters -----------------------------------------------


@register_layer("Conv2d")
def _conv2d_layer(layer, input_shape):
    """``nn.Conv2d(...)`` -> the ``conv2d`` operation.

    Enumerates every attribute that changes what is computed and refuses the
    ones it cannot express, rather than ignoring them: a layer with
    ``padding='same'`` or a reflect padding mode computes something no conv
    kernel here implements, and answering for it would be confidently wrong.

    Emits ``activation="none"``, because a bare ``nn.Conv2d`` applies none. A
    fused conv+ReLU is two layers in torch, and this adapter sees one at a
    time -- it cannot know a ReLU follows, so it must not assume one does.
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
        "activation": NO_ACTIVATION,
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
