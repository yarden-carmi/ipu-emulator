"""3x3 FP32 convolution + bias + ReLU harness (wide-vector mode).

Computes, with zero padding on all four sides::

    out[o, y, x] = relu( bias[o]
                         + SUM_ci SUM_kr SUM_kc W[o,ci,kr,kc] * in[ci, y+kr, x+kc] )

for kr, kc in {-1, 0, +1}. This is SuperPoint's conv1a..conv4b / convPa /
convDa shape.

**The ReLU is fused and cannot be turned off.** ``ACTIVATE.QUANTIZE`` takes its
activation as an immediate, so it is fixed when the ``.asm`` is assembled. The
registry therefore treats it as part of the query (``activation="relu"``), and
a caller asking for a plain 3x3 convolution is refused rather than handed this
kernel -- see :mod:`~ipu_apps.convolutions_universal.conv._spec_support`.

Where the nine taps come from
-----------------------------
XMEM addressing is row-granular: a load reaches a whole 128-element row and
cannot shift by one element. The nine shifted *loads* the older byte-addressed
kernel used are simply not expressible. ``MULT.RC.*`` reads R_CYCLIC at an
arbitrary element index and may cross slot boundaries, so the horizontal shift
moves into the register instead -- three vertically-neighbouring rows occupy
three of R_CYCLIC's four slots, and kc becomes a +/-1 step on the read index.

Memory layout (all sizes in 128-element rows; one row = 512 bytes here):

    input   Cin planes (+1 reserved), each (H + 2) * TPR rows
    weight  NGROUPS rows per output channel, NGROUPS = ceil(Cin / 14)
    bias    one row per output channel, bias[o] in element 0
    output  Cout planes, H * TPR rows each

**Halo tiling.** A spatial row is cut into tiles of 126 output columns, each
stored as one 128-element row whose first and last elements are the horizontal
halo::

    element  0        = input column (t*126 - 1)
    elements 1..126   = input columns t*126 .. t*126+125
    element  127      = input column (t*126 + 126)

Output lane j (0..125) is column ``t*126 + j``, and tap kc reads element
``j + kc + 1`` -- so every tap of every valid lane is satisfied from inside the
one row, with no neighbouring-tile dependency. Columns outside the image are
stored as zero, which *is* the convolution's zero padding. Lanes 126 and 127
read past the slot and are discarded.

**The vertical border costs nothing.** Each plane carries an all-zero row band
above and below (H + 2 spatial rows), so output row y reads padded rows y, y+1,
y+2 unconditionally. There is no top/bottom special case in the kernel.

Nine weights per channel and 128 per ``LDR_MULT_REG`` row gives 14 channels per
group -- that limit is taps-per-row, and unlike the pointwise kernel's it is
not lifted by FP32. Group size is exact (``min(14, Cin - done)``).

Usage::

    from ipu_apps.convolutions_universal.conv.conv3x3_relu import (
        Conv3x3ReluApp,
    )

    app = Conv3x3ReluApp(
        inst_path="conv3x3_relu.bin",
        input_path="x.bin",        # Cin*H*W FP32, channel-major
        weight_path="w.bin",       # Cout*Cin*3*3 FP32, (kr, kc) row-major
        bias_path="b.bin",         # Cout FP32 (optional; zeros when absent)
        output_path="y.bin",       # Cout*H*W FP32
        in_channels=64, out_channels=64, height=8, width=80,
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
from ipu_apps.convolutions_universal.conv._spec_support import (
    BASE_ROW,
    LANES,
    ROW_BYTES,
    WIDE_VECTOR_ONLY,
    activation_refusal,
    conv_query,
    geometry_refusal,
    headroom_caveat,
    lane_caveat,
    xmem_refusal,
)

# The kernel size this app implements.
KERNEL_SIZE = 3

# One element at each end of a 128-element row is the horizontal halo, leaving
# 126 usable output columns per tile.
TILE_COLS = LANES - 2
# One all-zero spatial row above and below each plane supplies the vertical
# zero padding, so the kernel needs no top/bottom special case.
PAD_ROWS = 1

# Fused into the store as ACTIVATE.QUANTIZE relu, fixed at assembly time.
ACTIVATION = "relu"


class Conv3x3ReluApp(IpuApp):
    """3x3 FP32 convolution + bias + ReLU over a ``(Cin, H, W)`` activation.

    Args:
        inst_path:    Path to the assembled instruction binary.
        input_path:   Activation, ``Cin*H*W`` FP32, channel-major.
        weight_path:  Weights, ``Cout*Cin*3*3`` FP32, ``(kr, kc)`` row-major --
                      i.e. exactly ``W[o, ci].ravel()`` per channel.
        bias_path:    Optional bias, ``Cout`` FP32. Absent means a zero bias;
                      the region still exists, so the kernel's reset-and-seed
                      word is unconditional.
        output_path:  Optional path for the ``Cout*H*W`` FP32 result.
        in_channels:  Cin.
        out_channels: Cout.
        height:       H.
        width:        W.
        bias:         Whether this convolution has a bias, for the registry
                      declaration. Independent of ``bias_path``, which only
                      supplies the values.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        height: int,
        width: int,
        bias: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.input_path = Path(self.input_path)
        self.weight_path = Path(self.weight_path)
        bias_path = getattr(self, "bias_path", None)
        self.bias_path = Path(bias_path) if bias_path is not None else None

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.height = int(height)
        self.width = int(width)
        self.has_bias = bool(bias)

        # Delegate to the registry declaration rather than restating the
        # bounds: SPEC.supports is the single source of truth for this kernel's
        # domain, including the XMEM budget.
        SPEC.guard(
            shape=(self.in_channels, self.height, self.width),
            weight_shape=(
                self.out_channels,
                self.in_channels,
                KERNEL_SIZE,
                KERNEL_SIZE,
            ),
            stride=1,
            padding=1,
            dilation=1,
            groups=1,
            bias=self.has_bias,
            activation=ACTIVATION,
        )
        self.query = conv_query(
            (self.in_channels, self.height, self.width),
            (self.out_channels, self.in_channels, KERNEL_SIZE, KERNEL_SIZE),
            stride=1,
            padding=1,
            dilation=1,
            groups=1,
            bias=self.has_bias,
            activation=ACTIVATION,
        )
        self._layout()

    def _layout(self) -> None:
        """Place the four regions back-to-back, in rows and in bytes.

        Row numbers drive the CR registers (the .asm's XMEM operands are rows);
        the byte addresses drive the direct ``state.xmem`` calls in setup and
        teardown, which bypass row translation. Keeping both explicit is what
        stops the two units being confused.
        """
        lay = self.query.layout(tile_cols=TILE_COLS, pad_rows=PAD_ROWS)
        self.geometry = lay
        self.tiles_per_row = lay.tiles_per_row
        self.padded_height = lay.padded_height
        self.in_plane_stride = lay.in_plane_stride
        self.chan_advance = self.height * self.tiles_per_row
        self.num_groups = lay.num_groups
        self.group_cap = lay.group_cap
        self.output_rows = lay.output_rows

        self.input_base_row = BASE_ROW
        self.weight_base_row = self.input_base_row + lay.input_rows
        self.bias_base_row = self.weight_base_row + lay.weight_rows
        self.output_base_row = self.bias_base_row + lay.bias_rows

        self.input_base = self.input_base_row * ROW_BYTES
        self.weight_base = self.weight_base_row * ROW_BYTES
        self.bias_base = self.bias_base_row * ROW_BYTES
        self.output_base = self.output_base_row * ROW_BYTES

    # -- wide-vector FP32 state ---------------------------------------------

    @staticmethod
    def make_state() -> IpuState:
        """Build the FP32 wide-vector state this app requires.

        ``wide_vector_quantize_output=False`` keeps elements 4-byte FP32
        through AAQ, so ACTIVATE.QUANTIZE applies the ReLU in FP32 and writes
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

    def _read_f32(self, path: Path, count: int, what: str) -> np.ndarray:
        data = np.fromfile(path, dtype="<f4")
        if data.size != count:
            raise ValueError(
                f"{what} file {path} holds {data.size} FP32 values; this "
                f"convolution needs {count}"
            )
        return data

    def _pack_input(self, x: np.ndarray) -> np.ndarray:
        """Lay out ``(Cin, H, W)`` as halo-tiled, vertically bordered planes.

        Returns ``(Cin + 1, H + 2, TPR, 128)`` -- the trailing plane is the
        reserved guard plane and stays zero, and rows 0 and H+1 of each plane
        are the zero border.
        """
        cin, h, w = x.shape
        tpr = self.tiles_per_row
        planes = np.zeros((cin + 1, h + 2, tpr, LANES), dtype="<f4")
        for t in range(tpr):
            # Element e of tile t is input column t*TILE_COLS + e - 1, so the
            # slice starts one column early and runs one column past the tile.
            lo = t * TILE_COLS - 1
            hi = lo + LANES
            src_lo, src_hi = max(lo, 0), min(hi, w)
            if src_hi > src_lo:
                planes[:cin, 1 : h + 1, t, src_lo - lo : src_hi - lo] = x[
                    :, :, src_lo:src_hi
                ]
        return planes

    def setup(self, state: "IpuState") -> None:
        cin, cout = self.in_channels, self.out_channels
        h, w = self.height, self.width
        taps = KERNEL_SIZE * KERNEL_SIZE

        x = self._read_f32(self.input_path, cin * h * w, "input").reshape(cin, h, w)
        state.xmem.write_address(self.input_base, self._pack_input(x).tobytes())

        # Weights: NGROUPS rows of 128 per output channel; channel c occupies
        # nine consecutive elements at (c - g*14)*9, in (kr, kc) row-major
        # order -- which is exactly W[o, ci].ravel().
        weights = self._read_f32(
            self.weight_path, cout * cin * taps, "weight"
        ).reshape(cout, cin, taps)
        wrows = np.zeros((cout, self.num_groups, LANES), dtype="<f4")
        for c in range(cin):
            g, within = divmod(c, self.group_cap)
            wrows[:, g, within * taps : (within + 1) * taps] = weights[:, c, :]
        state.xmem.write_address(self.weight_base, wrows.tobytes())

        # Bias: one row per output channel, the value in element 0 (the kernel
        # selects it as Ra element 128, i.e. R1[0]).
        brows = np.zeros((cout, LANES), dtype="<f4")
        if self.bias_path is not None:
            brows[:, 0] = self._read_f32(self.bias_path, cout, "bias")
        state.xmem.write_address(self.bias_base, brows.tobytes())

        # CR0 and CR1 are READ-ONLY: CR0 == 0 always, CR1 == 1 always. Both are
        # exploited directly -- CR0 as the zero source and CR1 as both the 1.0
        # scalar for the bias broadcast and every +1 increment. All writable
        # CRs below hold row numbers, row strides, loop bounds or the two
        # walking-index constants (.asm XMEM operands are rows -- issue #179).
        state.regfile.set_cr(2, self.input_base_row)
        state.regfile.set_cr(3, self.output_base_row)
        state.regfile.set_cr(4, self.weight_base_row)
        state.regfile.set_cr(5, self.bias_base_row)
        # CR6 walks lr_addr from padded row (y+2, t) of one channel to (y, t)
        # of the next: (H+2)*TPR - 2*TPR. Precomputed because the LR slot has
        # no multiply.
        state.regfile.set_cr(6, self.chan_advance)
        state.regfile.set_cr(7, self.tiles_per_row)
        state.regfile.set_cr(8, h)
        state.regfile.set_cr(9, cin)
        state.regfile.set_cr(10, cout)
        # CR11 doubles as the per-output-channel weight-row advance.
        state.regfile.set_cr(11, self.num_groups)
        # CR12 = 128: the R_CYCLIC slot-1 index AND the Ra element index that
        # selects R1[0] = bias[o] (both just need the constant 128).
        state.regfile.set_cr(12, LANES)
        # CR13 = cap-1: added to lr_done it yields the group's LAST channel
        # index, which is what tap 9's BLT compares against (it reads lr_done
        # pre-increment).
        state.regfile.set_cr(13, self.group_cap - 1)
        # CR14 = 126: the tap walk's slot-to-slot step (128 - 2, since the
        # three kc taps within a slot have already advanced the index by 2).
        state.regfile.set_cr(14, LANES - 2)

        # MULT.*/ACTIVATE.QUANTIZE read the active-element count from the named
        # dstructure CR's valid_elements field. The asm names CR15 throughout,
        # so set CR15.valid_elements = 128 to process the full FP32 row.
        state.set_cr_dstructure(valid_elements=LANES)

    def teardown(self, state: "IpuState") -> None:
        if self.output_path is None:
            return
        raw = state.xmem.read_address(self.output_base, self.output_rows * ROW_BYTES)
        planes = np.frombuffer(raw, dtype="<f4").reshape(
            self.out_channels, self.height, self.tiles_per_row, LANES
        )
        # Only the first TILE_COLS lanes of each tile are real output columns;
        # lanes 126 and 127 read past their slot. Concatenating the usable
        # lanes and trimming to W is what makes the output file a dense
        # (Cout, H, W) array matching the input's element order.
        cols = planes[:, :, :, :TILE_COLS].reshape(
            self.out_channels, self.height, self.tiles_per_row * TILE_COLS
        )
        np.ascontiguousarray(cols[:, :, : self.width]).tofile(self.output_path)

    def run(self, **kwargs):
        # Always run on the FP32 wide-vector state unless the caller supplied one.
        kwargs.setdefault("state", self.make_state())
        return super().run(**kwargs)


# -- registry declaration ---------------------------------------------------


def _query(params):
    return conv_query(
        params["shape"],
        params["weight_shape"],
        stride=params["stride"],
        padding=params["padding"],
        dilation=params["dilation"],
        groups=params["groups"],
        bias=params.get("bias", True),
        activation=params.get("activation", "none"),
    )


def _geometry(params):
    return _query(params).layout(tile_cols=TILE_COLS, pad_rows=PAD_ROWS)


def _supports(**params):
    q = _query(params)
    bad = geometry_refusal(q)
    if bad:
        return no(bad)
    if q.kh != KERNEL_SIZE or q.kw != KERNEL_SIZE:
        return no(
            f"is {KERNEL_SIZE}x{KERNEL_SIZE}; this weight is {q.kh}x{q.kw}. "
            f"A 1x1 weight has its own kernel (conv1x1); other window "
            f"sizes have none."
        )
    bad = activation_refusal(q, ACTIVATION)
    if bad:
        return no(bad)
    if q.padding != 1:
        # 'same' padding is what keeps the output H x W; anything else changes
        # the output extent, which this kernel's store layout cannot express.
        return no(
            f"zero-pads by exactly 1 to keep the output H x W; this query asks "
            f"for padding {q.padding}, which would make the output "
            f"{q.height + 2 * q.padding - 2} x {q.width + 2 * q.padding - 2}"
        )
    budget = xmem_refusal(_geometry(params))
    if budget:
        return no(budget)
    return yes()


def _build(**params):
    q = _query(params)
    return {
        "in_channels": q.cin,
        "out_channels": q.cout,
        "height": q.height,
        "width": q.width,
        "bias": q.bias,
    }


def _explain(**params):
    lay = _geometry(params)
    return (
        f"a {KERNEL_SIZE}x{KERNEL_SIZE} zero-padded convolution with ReLU "
        f"fused into the store; the nine taps are read out of R_CYCLIC at a "
        f"walking element index, so {lay.query.cin} input channels run as "
        f"{lay.num_groups} exact group(s) of at most {lay.group_cap} "
        f"({KERNEL_SIZE * KERNEL_SIZE} weights each)"
    )


def _caveats(**params):
    lay = _geometry(params)
    notes = [
        WIDE_VECTOR_ONLY,
        "ReLU is fused into the store and cannot be disabled: "
        "ACTIVATE.QUANTIZE takes its activation as an immediate.",
    ]
    lanes = lane_caveat(lay)
    if lanes:
        notes.append(lanes)
    notes.append(headroom_caveat(lay))
    return tuple(notes)


SPEC = KernelSpec(
    name="conv3x3_relu",
    op="conv2d",
    variant="dense3x3_relu",
    app_class=Conv3x3ReluApp,
    asm="conv3x3_relu.asm",
    # The geometry parameters are required rather than defaulted on purpose: a
    # conv spec that silently assumed stride 1 would answer for an operation no
    # kernel here computes. `activation` is likewise part of the query, because
    # this kernel's ReLU is not optional.
    requires=("shape", "weight_shape", "stride", "padding", "dilation", "groups"),
    tags=("fp32-wide", "dense", "relu"),
    supports=_supports,
    build=_build,
    explain=_explain,
    caveats=_caveats,
    bundle=lambda **params: _query(params).bundle,
    # The general 3x3 kernel. It genuinely handles Cin=1 too and must say so,
    # but conv3x3_relu_cin1 unrolls the channel loop away and is 1.77x faster
    # there, so this steps aside on cost.
    cost=lambda **params: 1.0,
)
