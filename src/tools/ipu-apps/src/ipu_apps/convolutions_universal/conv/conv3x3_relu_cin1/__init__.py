"""3x3 conv + bias + ReLU specialised on a SINGLE input channel.

SuperPoint's ``conv1a`` (1 -> 64) and nothing else. The bespoke twin of
:mod:`~ipu_apps.convolutions_universal.conv.conv3x3_relu`, identical result,
**1.77x fewer cycles**.

Why only Cin = 1
----------------
``conv3x3_relu`` is already ~95% MULT-occupied at Cin = 64: its nine taps are
unrolled and its channel loop is software-pipelined to 9 words per 9 MACs, so
specialising it buys ~5%. At Cin = 1 that inverts -- there is a single channel
to amortise the per-tile and per-group bookkeeping against, and only 9 of 23
words multiply::

    conv3x3_relu at Cin=1              conv3x3_relu_cin1
    2  tile head + bias                3  row loads (bias rides in the third)
    2  prime two rows                  9  taps (tap 9 carries ACTIVATE + STR)
    5  group sizing (partial group)    1  advance + branch
    9  taps
    2  group advance + branch         13 words per output tile
    3  drain
   23 words per output tile            -> 1.77x

Cin = 1 is also what makes it *possible*. A fully unrolled channel loop costs
``9*Cin + 4`` words, so it fits the 128-word IMEM bank only while
``Cin <= 13``. Every other SuperPoint layer has ``Cin >= 64`` and would need
580 to 1156 words -- more than a bank, and at Cin=128 more than the whole
1024-word instruction memory. They keep the general kernel.

What the unroll removes
-----------------------
* the channel loop, and the exact-group-size computation a partial final group
  forces (``ADD gend`` / ``BLT`` / ``SET`` / ``DEC``);
* the **guard input plane** -- the general kernel's pipelined loop prefetches
  one channel past the end; this one has no next channel to prefetch, so the
  input region is exactly one plane;
* the separate drain word -- tap 9 co-issues ``ACTIVATE.QUANTIZE`` and the
  store, since ACC runs before AAQ runs before STORE inside a word.

Halo tiling, the zero border, and the weight layout are ``conv3x3_relu``'s,
unchanged, so the two are interchangeable and the registry picks this one on
cost when ``Cin == 1``.

Usage::

    from ipu_apps.convolutions_universal.conv.conv3x3_relu_cin1 import (
        Conv3x3ReluCin1App,
    )

    app = Conv3x3ReluCin1App(
        inst_path="conv3x3_relu_cin1.bin",
        input_path="image.bin",   # 1*H*W FP32
        weight_path="w.bin",      # Cout*1*3*3 FP32, (kr, kc) row-major
        bias_path="b.bin",        # Cout FP32
        output_path="y.bin",      # Cout*H*W FP32
        out_channels=64, height=480, width=640,
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
    XMEM_ROWS,
    activation_refusal,
    conv_query,
    geometry_refusal,
)

KERNEL_SIZE = 3
TAPS = KERNEL_SIZE * KERNEL_SIZE
TILE_COLS = LANES - 2
PAD_ROWS = 1
ACTIVATION = "relu"
IN_CHANNELS = 1          # fixed by the unroll


class Conv3x3ReluCin1App(IpuApp):
    """3x3 convolution + bias + ReLU over a single-channel ``(1, H, W)`` image.

    Args:
        inst_path:    Path to the assembled instruction binary.
        input_path:   Activation, ``H*W`` FP32.
        weight_path:  Weights, ``Cout*1*3*3`` FP32, ``(kr, kc)`` row-major.
        bias_path:    Optional bias, ``Cout`` FP32; absent means a zero bias.
        output_path:  Optional path for the ``Cout*H*W`` FP32 result.
        out_channels: Cout.
        height:       H.
        width:        W.
        bias:         Whether this convolution has a bias, for the registry.
    """

    def __init__(self, *, out_channels: int, height: int, width: int,
                 bias: bool = True, **kwargs) -> None:
        super().__init__(**kwargs)
        self.input_path = Path(self.input_path)
        self.weight_path = Path(self.weight_path)
        bias_path = getattr(self, "bias_path", None)
        self.bias_path = Path(bias_path) if bias_path is not None else None

        self.in_channels = IN_CHANNELS
        self.out_channels = int(out_channels)
        self.height = int(height)
        self.width = int(width)
        self.has_bias = bool(bias)

        SPEC.guard(
            shape=(IN_CHANNELS, self.height, self.width),
            weight_shape=(self.out_channels, IN_CHANNELS, KERNEL_SIZE, KERNEL_SIZE),
            stride=1, padding=1, dilation=1, groups=1,
            bias=self.has_bias, activation=ACTIVATION,
        )
        self.query = conv_query(
            (IN_CHANNELS, self.height, self.width),
            (self.out_channels, IN_CHANNELS, KERNEL_SIZE, KERNEL_SIZE),
            stride=1, padding=1, dilation=1, groups=1,
            bias=self.has_bias, activation=ACTIVATION,
        )
        self._layout()

    def _layout(self) -> None:
        """Row math, done here rather than via ConvLayout.

        ConvLayout reserves a guard input plane for the general kernel's
        one-past prefetch. This kernel has no channel loop and so no prefetch,
        and its input region is exactly one plane -- which is most of why a
        Cin=1 layer is cheap in XMEM as well as in cycles.
        """
        self.tiles_per_row = (self.width + TILE_COLS - 1) // TILE_COLS
        self.padded_height = self.height + 2 * PAD_ROWS
        self.in_plane_stride = self.padded_height * self.tiles_per_row
        self.output_rows = self.out_channels * self.height * self.tiles_per_row

        self.input_base_row = BASE_ROW
        self.weight_base_row = self.input_base_row + self.in_plane_stride
        self.bias_base_row = self.weight_base_row + self.out_channels
        self.output_base_row = self.bias_base_row + self.out_channels
        self.total_rows = self.output_base_row + self.output_rows

        self.input_base = self.input_base_row * ROW_BYTES
        self.weight_base = self.weight_base_row * ROW_BYTES
        self.bias_base = self.bias_base_row * ROW_BYTES
        self.output_base = self.output_base_row * ROW_BYTES

    @staticmethod
    def make_state() -> IpuState:
        state = IpuState(
            wide_vector_debug=True,
            wide_vector_arithmetic=WideVectorArithmetic.FP32,
            wide_vector_quantize_output=False,
        )
        state.dtype = DType.INT8
        return state

    def _read_f32(self, path: Path, count: int, what: str) -> np.ndarray:
        data = np.fromfile(path, dtype="<f4")
        if data.size != count:
            raise ValueError(
                f"{what} file {path} holds {data.size} FP32 values; this "
                f"convolution needs {count}"
            )
        return data

    def setup(self, state: "IpuState") -> None:
        cout, h, w = self.out_channels, self.height, self.width
        x = self._read_f32(self.input_path, h * w, "input").reshape(h, w)

        # One plane, halo-tiled, with a zero row above and below.
        plane = np.zeros((self.padded_height, self.tiles_per_row, LANES), dtype="<f4")
        for t in range(self.tiles_per_row):
            lo = t * TILE_COLS - 1
            hi = lo + LANES
            src_lo, src_hi = max(lo, 0), min(hi, w)
            if src_hi > src_lo:
                plane[1 : h + 1, t, src_lo - lo : src_hi - lo] = x[:, src_lo:src_hi]
        state.xmem.write_address(self.input_base, plane.tobytes())

        # Nine weights per output channel, one row each, (kr, kc) row-major.
        weights = self._read_f32(self.weight_path, cout * TAPS, "weight")
        wrows = np.zeros((cout, LANES), dtype="<f4")
        wrows[:, :TAPS] = weights.reshape(cout, TAPS)
        state.xmem.write_address(self.weight_base, wrows.tobytes())

        brows = np.zeros((cout, LANES), dtype="<f4")
        if self.bias_path is not None:
            brows[:, 0] = self._read_f32(self.bias_path, cout, "bias")
        state.xmem.write_address(self.bias_base, brows.tobytes())

        state.regfile.set_cr(2, self.input_base_row)
        state.regfile.set_cr(3, self.output_base_row)
        state.regfile.set_cr(4, self.weight_base_row)
        state.regfile.set_cr(5, self.bias_base_row)
        state.regfile.set_cr(6, self.tiles_per_row)
        state.regfile.set_cr(7, h)
        state.regfile.set_cr(8, cout)
        state.regfile.set_cr(9, TILE_COLS)      # rc slot-to-slot step
        state.regfile.set_cr(10, LANES)
        # CR11 = TPR-1: the tile branch reads its counter pre-increment.
        state.regfile.set_cr(11, self.tiles_per_row - 1)
        state.set_cr_dstructure(valid_elements=LANES)

    def teardown(self, state: "IpuState") -> None:
        if self.output_path is None:
            return
        raw = state.xmem.read_address(self.output_base, self.output_rows * ROW_BYTES)
        planes = np.frombuffer(raw, dtype="<f4").reshape(
            self.out_channels, self.height, self.tiles_per_row, LANES
        )
        cols = planes[:, :, :, :TILE_COLS].reshape(
            self.out_channels, self.height, self.tiles_per_row * TILE_COLS
        )
        np.ascontiguousarray(cols[:, :, : self.width]).tofile(self.output_path)

    def run(self, **kwargs):
        kwargs.setdefault("state", self.make_state())
        return super().run(**kwargs)


# -- registry declaration ---------------------------------------------------


def _query(params):
    return conv_query(
        params["shape"], params["weight_shape"],
        stride=params["stride"], padding=params["padding"],
        dilation=params["dilation"], groups=params["groups"],
        bias=params.get("bias", True),
        activation=params.get("activation", "none"),
    )


def _rows(q):
    tpr = (q.width + TILE_COLS - 1) // TILE_COLS
    return (BASE_ROW + (q.height + 2) * tpr + 2 * q.cout
            + q.cout * q.height * tpr)


def _supports(**params):
    q = _query(params)
    bad = geometry_refusal(q)
    if bad:
        return no(bad)
    if q.kh != KERNEL_SIZE or q.kw != KERNEL_SIZE:
        return no(
            f"is {KERNEL_SIZE}x{KERNEL_SIZE}; this weight is {q.kh}x{q.kw}"
        )
    bad = activation_refusal(q, ACTIVATION)
    if bad:
        return no(bad)
    if q.cin != IN_CHANNELS:
        return no(
            f"unrolls the channel loop and so handles exactly {IN_CHANNELS} "
            f"input channel; this query has {q.cin}. A fully unrolled loop "
            f"costs 9*Cin+4 words and only fits the 128-word IMEM bank while "
            f"Cin <= 13 -- use conv3x3_relu, which loses only about 5% at "
            f"these channel counts."
        )
    if q.padding != 1:
        return no(
            f"zero-pads by exactly 1 to keep the output H x W; this query asks "
            f"for padding {q.padding}"
        )
    rows = _rows(q)
    if rows > XMEM_ROWS:
        return no(
            f"needs {rows} XMEM rows; wide-vector XMEM holds {XMEM_ROWS} rows "
            f"of {ROW_BYTES} B."
        )
    return yes()


def _build(**params):
    q = _query(params)
    return {"out_channels": q.cout, "height": q.height, "width": q.width,
            "bias": q.bias}


def _explain(**params):
    q = _query(params)
    return (
        f"a single-input-channel {KERNEL_SIZE}x{KERNEL_SIZE} convolution: the "
        f"channel loop is unrolled away, so an output tile is 13 words against "
        f"conv3x3_relu's 23 -- about 1.77x fewer cycles for the same "
        f"{q.cout}x{q.height}x{q.width} result"
    )


def _caveats(**params):
    q = _query(params)
    return (
        WIDE_VECTOR_ONLY,
        "ReLU is fused into the store and cannot be disabled: "
        "ACTIVATE.QUANTIZE takes its activation as an immediate.",
        f"Cin is fixed at {IN_CHANNELS} by the unroll; this kernel is "
        f"SuperPoint's conv1a and cannot be asked for more channels.",
        f"uses {_rows(q)} of {XMEM_ROWS} XMEM rows "
        f"({XMEM_ROWS - _rows(q)} free)",
    )


SPEC = KernelSpec(
    name="conv3x3_relu_cin1",
    op="conv2d",
    variant="dense3x3_relu_cin1",
    app_class=Conv3x3ReluCin1App,
    asm="conv3x3_relu_cin1.asm",
    requires=("shape", "weight_shape", "stride", "padding", "dilation", "groups"),
    tags=("fp32-wide", "dense", "relu", "unrolled"),
    supports=_supports,
    build=_build,
    explain=_explain,
    caveats=_caveats,
    bundle=lambda **params: _query(params).bundle,
    # Beats conv3x3_relu (which must move to 1.0) at Cin=1.
    cost=lambda **params: 0.0,
)
