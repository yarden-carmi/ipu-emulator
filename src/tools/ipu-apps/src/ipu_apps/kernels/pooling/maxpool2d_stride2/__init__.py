"""Memory-only harness for the 2x2, stride-2 FP32 max-pool kernel.

This is the identity boilerplate adapted to the CR map and XMEM geometry in
``maxpool2d_stride2.asm``. The harness does not pool, pad, tile, or reshape
data. ``input_path`` must already contain the kernel's XMEM rows:

* channel-major, then spatial-row-major;
* ``ceil(width / LANES)`` XMEM rows per spatial row;
* real columns first, then ``-FLT_MAX`` padding in the final XMEM row.

The output file is the raw tiled XMEM result. Each logical output row occupies
``ceil((width // 2) / LANES)`` XMEM rows; unused positions remain in the file.
"""

from __future__ import annotations

from pathlib import Path

from ipu_emu.emulator import dump_xmem_to_binary, load_binary_to_xmem
from ipu_emu.ipu import LANES, R_ACC_SIZE
from ipu_emu.ipu_math import DType
from ipu_emu.ipu_state import IpuState, WideVectorArithmetic
from ipu_emu.xmem import XMEM_SIZE_BYTES

from ipu_apps.base import IpuApp
from ipu_apps.kernel_registry import OUTPUT, KernelSpec, ShapeBundle, no, yes

KERNEL_SIZE = 2
STRIDE = 2
PADDING = 0
SCRATCH_ROWS = 2


def _shape(params) -> tuple[int, ...]:
    return tuple(int(dim) for dim in params["shape"])


def _geometry(shape: tuple[int, ...]) -> dict[str, int]:
    channels, height, width = shape
    out_height = height // STRIDE
    out_width = width // STRIDE
    input_tiles_per_row = (width + LANES - 1) // LANES
    out_tiles_per_row = (out_width + LANES - 1) // LANES
    in_row_stride = input_tiles_per_row
    in_plane_stride = height * in_row_stride
    input_rows = channels * in_plane_stride
    output_rows = channels * out_height * out_tiles_per_row
    scratch_base_row = input_rows
    output_base_row = scratch_base_row + SCRATCH_ROWS
    return {
        "out_height": out_height,
        "out_width": out_width,
        "input_tiles_per_row": input_tiles_per_row,
        "out_tiles_per_row": out_tiles_per_row,
        "in_row_stride": in_row_stride,
        "in_plane_stride": in_plane_stride,
        "input_rows": input_rows,
        "output_rows": output_rows,
        "scratch_base_row": scratch_base_row,
        "output_base_row": output_base_row,
        "total_rows": output_base_row + output_rows,
    }


class _MaxPool2dStride2Base(IpuApp):
    """Shared memory contract for the two stride-2 assembly kernels."""

    def __init__(
        self,
        *,
        input_path: str | Path,
        shape=(1, 2, 2),
        _spec: KernelSpec,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.input_path = Path(input_path)
        self.shape = tuple(int(dim) for dim in shape)
        _spec.guard(
            shape=self.shape,
            kernel_size=KERNEL_SIZE,
            stride=STRIDE,
            padding=PADDING,
        )

        self.channels, self.height, self.width = self.shape
        for name, value in _geometry(self.shape).items():
            setattr(self, name, value)

        expected_size = self.input_rows * R_ACC_SIZE
        size = self.input_path.stat().st_size
        if size != expected_size:
            raise ValueError(
                f"preformatted input for shape {self.shape} must contain "
                f"{self.input_rows} XMEM rows ({expected_size} bytes); got {size} bytes"
            )

    @staticmethod
    def make_state() -> IpuState:
        state = IpuState(
            wide_vector_debug=True,
            wide_vector_arithmetic=WideVectorArithmetic.FP32,
            wide_vector_quantize_output=False,
        )
        state.dtype = DType.INT8
        return state

    def _load_input(self, state: IpuState) -> None:
        load_binary_to_xmem(
            state,
            self.input_path,
            base_addr=0,
            chunk_size=R_ACC_SIZE,
            max_chunks=self.input_rows,
        )

    def teardown(self, state: IpuState) -> None:
        if self.output_path is not None:
            dump_xmem_to_binary(
                state,
                self.output_path,
                base_addr=self.output_base_row * R_ACC_SIZE,
                chunk_size=R_ACC_SIZE,
                num_chunks=self.output_rows,
            )

    def run(self, **kwargs):
        kwargs.setdefault("state", self.make_state())
        return super().run(**kwargs)


class MaxPool2dStride2App(_MaxPool2dStride2Base):
    """Initialization for the kernel whose input XMEM rows form complete pairs."""

    def __init__(self, *, input_path: str | Path, shape=(1, 2, 2), **kwargs) -> None:
        super().__init__(
            input_path=input_path,
            shape=shape,
            _spec=FULL_SPEC,
            **kwargs,
        )

    def setup(self, state: IpuState) -> None:
        self._load_input(state)
        state.regfile.set_cr(2, 0)
        state.regfile.set_cr(3, self.output_base_row)
        state.regfile.set_cr(4, self.scratch_base_row)
        state.regfile.set_cr(5, self.in_row_stride)
        state.regfile.set_cr(6, STRIDE * self.in_row_stride)
        state.regfile.set_cr(7, self.out_tiles_per_row)
        state.regfile.set_cr(8, self.out_height)
        state.regfile.set_cr(9, self.channels)
        state.regfile.set_cr(10, LANES)
        state.regfile.set_cr(11, self.in_plane_stride)
        state.set_cr_dstructure(valid_elements=LANES)


class MaxPool2dStride2TailApp(_MaxPool2dStride2Base):
    """Initialization for the kernel with one final unpaired input XMEM row."""

    def __init__(self, *, input_path: str | Path, shape=(1, 2, 2), **kwargs) -> None:
        super().__init__(
            input_path=input_path,
            shape=shape,
            _spec=TAIL_SPEC,
            **kwargs,
        )

    def setup(self, state: IpuState) -> None:
        self._load_input(state)
        state.regfile.set_cr(2, 0)
        state.regfile.set_cr(3, self.output_base_row)
        state.regfile.set_cr(4, self.scratch_base_row)
        state.regfile.set_cr(5, self.in_row_stride)
        state.regfile.set_cr(6, STRIDE * self.in_row_stride)
        state.regfile.set_cr(7, self.out_tiles_per_row)
        state.regfile.set_cr(8, self.out_height)
        state.regfile.set_cr(9, self.channels)
        state.regfile.set_cr(10, LANES)
        state.regfile.set_cr(11, self.in_plane_stride)
        state.regfile.set_cr(12, self.out_tiles_per_row - 1)
        state.set_cr_dstructure(valid_elements=LANES)


def _supports_common(**params):
    shape = _shape(params)
    if len(shape) != 3:
        return no(f"expects a rank-3 (channels, height, width) tensor; got {shape}")
    channels, height, width = shape
    if min(shape) < 1:
        return no(f"all shape dimensions must be positive; got {shape}")
    if height < KERNEL_SIZE or width < KERNEL_SIZE:
        return no(f"a {KERNEL_SIZE}x{KERNEL_SIZE} window does not fit shape {shape}")
    if params["kernel_size"] != KERNEL_SIZE or params["stride"] != STRIDE:
        return no(
            f"implements kernel_size={KERNEL_SIZE}, stride={STRIDE}; got "
            f"kernel_size={params['kernel_size']}, stride={params['stride']}"
        )
    if params["padding"] != PADDING:
        return no(f"implements padding={PADDING}; got padding={params['padding']}")
    geometry = _geometry(shape)
    if geometry["total_rows"] * R_ACC_SIZE > XMEM_SIZE_BYTES:
        return no(
            f"memory image needs {geometry['total_rows']} XMEM rows, exceeding "
            f"the IPU capacity of {XMEM_SIZE_BYTES // R_ACC_SIZE} rows"
        )
    return yes()


def _has_single_half_tail(shape: tuple[int, ...]) -> bool:
    geometry = _geometry(shape)
    return geometry["input_tiles_per_row"] < 2 * geometry["out_tiles_per_row"]


def _supports_full(**params):
    support = _supports_common(**params)
    if not support.ok:
        return support
    shape = _shape(params)
    if _has_single_half_tail(shape):
        return no("the final output XMEM row has only one input XMEM row")
    return yes()


def _supports_tail(**params):
    support = _supports_common(**params)
    if not support.ok:
        return support
    shape = _shape(params)
    if not _has_single_half_tail(shape):
        return no("every output XMEM row has two input XMEM rows")
    return yes()


def _bundle(**params):
    shape = _shape(params)
    channels, height, width = shape
    return ShapeBundle.of(input=shape).with_shapes(
        derived={OUTPUT: (channels, height // STRIDE, width // STRIDE)}
    )


FULL_SPEC = KernelSpec(
    name="maxpool2d_stride2",
    op="maxpool2d",
    variant="stride2_full_pairs",
    app_class=MaxPool2dStride2App,
    asm="maxpool2d_stride2.asm",
    requires=("shape", "kernel_size", "stride", "padding"),
    supports=_supports_full,
    build=lambda **params: {"shape": _shape(params)},
    explain=lambda **params: (
        f"a 2x2 stride-2 max-pool maps {_shape(params)} to "
        f"{_bundle(**params)[OUTPUT]}"
    ),
    bundle=_bundle,
    caveats=lambda **params: (
        "input and output files use the kernel's raw tiled XMEM layout",
    ),
    tags=("fp32-wide", "strided", "full-pairs"),
)

TAIL_SPEC = KernelSpec(
    name="maxpool2d_stride2_tail",
    op="maxpool2d",
    variant="stride2_single_half_tail",
    app_class=MaxPool2dStride2TailApp,
    asm="maxpool2d_stride2_tail.asm",
    requires=("shape", "kernel_size", "stride", "padding"),
    supports=_supports_tail,
    build=lambda **params: {"shape": _shape(params)},
    explain=lambda **params: (
        f"a 2x2 stride-2 max-pool maps {_shape(params)} to "
        f"{_bundle(**params)[OUTPUT]} with a one-XMEM-row final input tail"
    ),
    bundle=_bundle,
    caveats=lambda **params: (
        "input and output files use the kernel's raw tiled XMEM layout",
    ),
    tags=("fp32-wide", "strided", "single-half-tail"),
)

SPECS = (FULL_SPEC, TAIL_SPEC)
