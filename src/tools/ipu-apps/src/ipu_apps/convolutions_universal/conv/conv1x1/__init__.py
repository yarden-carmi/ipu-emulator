"""Pointwise (1x1) FP32 convolution harness (wide-vector mode).

Computes, for every output channel, spatial row and 128-column tile::

    out[o, y, x] = bias[o] + SUM_ci W[o, ci] * in[ci, y, x]

A 1x1 convolution has no spatial window, so each output element is a pure
channel-space dot product and all 128 lanes of a tile reduce independently.
The kernel is therefore one accumulation loop over input channels -- no taps,
no border, no mask -- which is what makes it the right first kernel to carry
this package's shared machinery (row-granular addressing, the FP32
wide-vector state, the pipelined channel loop, the registry declaration).

Memory layout (see the .asm for the cycle-level detail). XMEM operands are ROW
numbers, one row = 128 elements = 512 bytes in wide-vector debug mode, so
everything below is sized in rows:

    input   Cin planes + 1 guard, each H*NCT rows, NCT = ceil(W/128)
    weight  NGROUPS rows per output channel, NGROUPS = ceil(Cin/128)
    bias    one row per output channel, bias[o] in element 0
    output  Cout planes, the same H*NCT rows each

The guard plane exists because the channel loop is software-pipelined: its
last iteration prefetches channel index ``Cin``, whose data is never consumed
but whose row must still be in bounds.

A width that is not a multiple of 128 pads with idle lanes. Addressing is
row-granular, so there is no tighter packing available -- and correspondingly
none of the byte-level "tight packing", guard row and last-tile spill the older
byte-addressed kernel needed. ``teardown`` slices the padding back off, so the
output file is a dense ``(Cout, H, W)`` FP32 array.

Usage::

    from ipu_apps.convolutions_universal.conv.conv1x1 import Conv1x1App

    app = Conv1x1App(
        inst_path="conv1x1.bin",
        input_path="x.bin",        # Cin*H*W FP32, channel-major
        weight_path="w.bin",       # Cout*Cin FP32
        bias_path="b.bin",         # Cout FP32 (optional; zeros when absent)
        output_path="y.bin",       # Cout*H*W FP32
        in_channels=256, out_channels=65, height=8, width=80,
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
    NO_ACTIVATION,
    ROW_BYTES,
    WIDE_VECTOR_ONLY,
    activation_refusal,
    conv_query,
    geometry_refusal,
    headroom_caveat,
    lane_caveat,
    xmem_refusal,
)

# The kernel size this app implements. A 1x1 weight is what makes the whole
# thing a channel-space dot product; a larger window needs the shifted-tap
# kernel, not a parameter here.
KERNEL_SIZE = 1

# Every lane of a 128-element row is a usable output column: a 1x1 kernel reads
# exactly one input element per output element, so there is no halo to spend
# lanes on and no vertical border to add.
TILE_COLS = LANES
PAD_ROWS = 0

# This kernel stores through ACTIVATE.QUANTIZE identity -- a plain FP32
# pass-through, no activation folded in.
ACTIVATION = NO_ACTIVATION


class Conv1x1App(IpuApp):
    """Pointwise FP32 convolution over a ``(Cin, H, W)`` activation.

    Args:
        inst_path:    Path to the assembled instruction binary.
        input_path:   Activation, ``Cin*H*W`` FP32, channel-major.
        weight_path:  Weights, ``Cout*Cin`` FP32.
        bias_path:    Optional bias, ``Cout`` FP32. Absent means a zero bias --
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
            weight_shape=(self.out_channels, self.in_channels, KERNEL_SIZE, KERNEL_SIZE),
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=self.has_bias,
            activation=ACTIVATION,
        )
        # The same query object the spec reasoned about, so the harness and the
        # registry cannot disagree about region sizes.
        self.query = conv_query(
            (self.in_channels, self.height, self.width),
            (self.out_channels, self.in_channels, KERNEL_SIZE, KERNEL_SIZE),
            stride=1,
            padding=0,
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
        self.plane_stride = lay.in_plane_stride
        self.num_groups = lay.num_groups
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
        through AAQ, so ACTIVATE.QUANTIZE writes FP32 into POST_AAQ_REG (the
        INT8 clamp is skipped entirely) and STR_POST_AAQ_REG drains the full
        512 bytes.
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

    def setup(self, state: "IpuState") -> None:
        cin, cout = self.in_channels, self.out_channels
        h, w = self.height, self.width
        padded_w = self.tiles_per_row * LANES

        # Input: Cin planes of H x padded_w, plus one zero guard plane the
        # pipelined channel loop's final prefetch reads but never consumes.
        x = self._read_f32(self.input_path, cin * h * w, "input").reshape(cin, h, w)
        planes = np.zeros((cin + 1, h, padded_w), dtype="<f4")
        planes[:cin, :, :w] = x
        state.xmem.write_address(self.input_base, planes.tobytes())

        # Weights: NGROUPS rows of 128 per output channel, zero-padded in the
        # last group so the kernel's exact group sizing never reads garbage.
        weights = self._read_f32(self.weight_path, cout * cin, "weight").reshape(cout, cin)
        wrows = np.zeros((cout, self.num_groups * LANES), dtype="<f4")
        wrows[:, :cin] = weights
        state.xmem.write_address(self.weight_base, wrows.tobytes())

        # Bias: one row per output channel, the value in element 0 (the kernel
        # selects it as Ra element 128, i.e. R1[0]).
        brows = np.zeros((cout, LANES), dtype="<f4")
        if self.bias_path is not None:
            brows[:, 0] = self._read_f32(self.bias_path, cout, "bias")
        state.xmem.write_address(self.bias_base, brows.tobytes())

        # CR0 and CR1 are READ-ONLY: CR0 == 0 always, CR1 == 1 always. Both are
        # exploited directly -- CR0 as the zero source (cyclic slot index,
        # initialisation) and CR1 as both the 1.0 scalar for the bias broadcast
        # and every +1 increment. All writable CRs below hold row numbers, row
        # strides or loop bounds (.asm XMEM operands are rows -- issue #179).
        state.regfile.set_cr(2, self.input_base_row)
        state.regfile.set_cr(3, self.output_base_row)
        state.regfile.set_cr(4, self.weight_base_row)
        state.regfile.set_cr(5, self.bias_base_row)
        # CR6 = H*NCT is precomputed here because the LR slot has no multiply.
        state.regfile.set_cr(6, self.plane_stride)
        state.regfile.set_cr(7, self.tiles_per_row)
        state.regfile.set_cr(8, h)
        state.regfile.set_cr(9, cin)
        state.regfile.set_cr(10, cout)
        # CR11 doubles as the per-output-channel weight-row advance.
        state.regfile.set_cr(11, self.num_groups)
        # CR12 = 128: the channel-group cap AND the Ra element index that
        # selects R1[0] = bias[o] (both just need the constant 128).
        state.regfile.set_cr(12, LANES)

        # MULT.*/ACTIVATE.QUANTIZE read the active-element count from the named
        # dstructure CR's valid_elements field. The asm names CR15 throughout,
        # so set CR15.valid_elements = 128 to process the full FP32 row.
        state.set_cr_dstructure(valid_elements=LANES)

    def teardown(self, state: "IpuState") -> None:
        if self.output_path is None:
            return
        padded_w = self.tiles_per_row * LANES
        raw = state.xmem.read_address(self.output_base, self.output_rows * ROW_BYTES)
        planes = np.frombuffer(raw, dtype="<f4").reshape(
            self.out_channels, self.height, padded_w
        )
        # Lanes W..padded_w-1 carry whatever the input padding produced; the
        # store always drains a full row. Slicing them off here is what makes
        # the output file a dense (Cout, H, W) array matching the input's
        # element order.
        np.ascontiguousarray(planes[:, :, : self.width]).tofile(self.output_path)

    def run(self, **kwargs):
        # Always run on the FP32 wide-vector state unless the caller supplied one.
        kwargs.setdefault("state", self.make_state())
        return super().run(**kwargs)


# -- registry declaration ---------------------------------------------------
# Declared beside the kernel so the registry needs no central list. `supports`
# is the single source of truth for this kernel's domain -- the constructor
# guard delegates to it rather than restating the bounds.


def _query(params):
    return conv_query(
        params["shape"],
        params["weight_shape"],
        stride=params["stride"],
        padding=params["padding"],
        dilation=params["dilation"],
        groups=params["groups"],
        bias=params.get("bias", True),
        activation=params.get("activation", NO_ACTIVATION),
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
            f"is pointwise ({KERNEL_SIZE}x{KERNEL_SIZE}); this weight is "
            f"{q.kh}x{q.kw}. The shifted-tap kernel for larger windows is not "
            f"migrated yet."
        )
    if q.padding != 0:
        # A 1x1 convolution reads exactly one input element per output element,
        # so padding only widens the output -- it never feeds the reduction.
        return no(
            f"writes an H x W output; padding {q.padding} on a "
            f"{KERNEL_SIZE}x{KERNEL_SIZE} kernel would grow it to "
            f"{q.height + 2 * q.padding} x {q.width + 2 * q.padding} with a "
            f"border this kernel does not write"
        )
    bad = activation_refusal(q, ACTIVATION)
    if bad:
        return no(bad)
    budget = xmem_refusal(q.layout(tile_cols=TILE_COLS, pad_rows=PAD_ROWS))
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
        f"a {KERNEL_SIZE}x{KERNEL_SIZE} convolution is a channel-space dot "
        f"product at every position, so all {LANES} lanes of a tile reduce "
        f"independently; {lay.query.cin} input channels run as "
        f"{lay.num_groups} exact group(s) of at most {lay.group_cap}"
    )


def _caveats(**params):
    lay = _geometry(params)
    notes = [WIDE_VECTOR_ONLY]
    lanes = lane_caveat(lay)
    if lanes:
        notes.append(lanes)
    notes.append(headroom_caveat(lay))
    return tuple(notes)


SPEC = KernelSpec(
    name="conv1x1",
    op="conv2d",
    variant="pointwise",
    app_class=Conv1x1App,
    asm="conv1x1.asm",
    # Every callback below indexes these, so the registry checks them first: an
    # omitted parameter is then a refusal naming what is missing. The geometry
    # parameters are required rather than defaulted on purpose -- a conv spec
    # that silently assumed stride 1 would answer for an operation no kernel
    # here computes.
    requires=("shape", "weight_shape", "stride", "padding", "dilation", "groups"),
    tags=("fp32-wide", "pointwise"),
    supports=_supports,
    build=_build,
    explain=_explain,
    caveats=_caveats,
    bundle=lambda **params: _query(params).bundle,
    # The only conv2d kernel so far, and an exact match for its domain.
    cost=lambda **params: 0.0,
)
