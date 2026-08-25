"""Depth-to-space (pixel shuffle) harness, FP32 wide-vector mode.

Computes::

    out[r*h + a, r*w + b] = in[r*a + b, h, w]      for a, b in [0, r)

which is ``nn.PixelShuffle(r)`` with one output channel. SuperPoint's detector
head is this at ``r = 8``: the 64 sub-grid channels left after the softmax
drops the dustbin become the full-resolution ``(H*8, W*8)`` heatmap.

Why this needs `ACC.RESHAPE`
----------------------------
One output row interleaves ``r`` input planes at stride ``r`` --
``out_row[r*w + b] = plane_b[w]``. There is no scatter-store, no vector shuffle,
and no inverse of ``ACC.STRIDE`` (which decimates; it does not expand).
``ACC.RESHAPE`` is the only instruction that writes MULT_RES elements to
*arbitrary* R_ACC indices: eight per instruction, addressed by two ``LRDn``
register pairs read as eight source and eight destination byte indices.

The source indices stay the constant ``[0..7]`` stepped by ``+8``, because the
per-tile element offset rides in the ``rc_idx`` of the ``MULT.RC.VE`` that
stages the row instead -- the same place every other kernel here puts a
horizontal shift. Encoding it in the source array would need a fresh pair of CR
constants per output tile.

The destination array is seeded ``[0, r, 2r, ..., 7r]`` from two CRs, stepped
by ``8r`` per instruction, and rewound by ``127`` per plane -- 128 to undo the
``r * (128/r)`` total drift, minus one for the next plane's ``+b``. Across all
planes and instructions the destinations cover 0..127 exactly once, which is
why R_ACC never has to be cleared.

The original hand-written kernel did only the plane-granular part of this and
left the interleave to the host. That is no longer necessary.

Memory layout (all sizes in 128-element rows; one row = 512 bytes here):

    input   C = r*r planes, each H spatial rows x IN_TILES_PER_ROW rows
    output  1 plane, (H*r) spatial rows x (IN_TILES_PER_ROW * r) rows

**One output channel.** A multi-channel shuffle is an outer loop offsetting the
plane index by ``c' * r * r``; it is refused rather than silently computing the
first channel only.

**Idle output lanes are multiplied by r.** Each input tile fans out to exactly
``r`` output tiles whether or not it is full, so an input width of 80 (48 idle
lanes of 128) becomes an output row of 1024 lanes holding 640 real columns.
Padding the input width to a multiple of 128 costs nothing extra.

Usage::

    from ipu_apps.reshape.depth_to_space import DepthToSpaceApp

    app = DepthToSpaceApp(
        inst_path="depth_to_space.bin",
        input_path="s.bin",       # (r*r)*H*W FP32, channel-major
        output_path="heat.bin",   # (H*r)*(W*r) FP32
        channels=64, height=60, width=80, upscale_factor=8,
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
from ipu_apps.reshape._spec_support import (
    BASE_ROW,
    LANES,
    RESHAPE_ELEMENTS,
    ROW_BYTES,
    WIDE_VECTOR_ONLY,
    geometry_refusal,
    headroom_caveat,
    lane_caveat,
    shuffle_query,
    xmem_refusal,
)

# Upscale factors this kernel's ACC.RESHAPE schedule can express. Two limits
# bound it, and r = 16 fails the second:
#
#   * 128/r elements per plane per output tile must be a whole number of
#     8-element ACC.RESHAPE instructions, so r must divide 16; and
#   * the per-instruction destination step is 8r, applied with ADDB, whose
#     source byte is reinterpreted as SIGNED. 8r must therefore be at most 127,
#     so r <= 15. At r = 16 the step 128 reads as -128 and walks the index array
#     down to zero instead of up.
SUPPORTED_FACTORS = (1, 2, 4, 8)

# This kernel emits one output channel; see the module docstring.
OUT_CHANNELS = 1


def _pack_bytes(values) -> int:
    """Pack four byte indices into one LR word, little-endian (element 0 low)."""
    lo, b1, b2, hi = values
    return lo | (b1 << 8) | (b2 << 16) | (hi << 24)


class DepthToSpaceApp(IpuApp):
    """Factor-``r`` pixel shuffle of an ``(r*r, H, W)`` activation.

    Args:
        inst_path:      Path to the assembled instruction binary.
        input_path:     Activation, ``C*H*W`` FP32, channel-major.
        output_path:    Optional path for the ``(H*r)*(W*r)`` FP32 result.
        channels:       C, which must equal ``r*r``.
        height:         H.
        width:          W.
        upscale_factor: r.
    """

    def __init__(
        self,
        *,
        channels: int,
        height: int,
        width: int,
        upscale_factor: int,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.input_path = Path(self.input_path)

        self.channels = int(channels)
        self.height = int(height)
        self.width = int(width)
        self.r = int(upscale_factor)

        # Delegate to the registry declaration rather than restating the
        # bounds: SPEC.supports is the single source of truth for this kernel's
        # domain, including the XMEM budget.
        shape = (self.channels, self.height, self.width)
        SPEC.guard(shape=shape, upscale_factor=self.r)
        self.query = shuffle_query(shape, upscale_factor=self.r)
        self._layout()

    def _layout(self) -> None:
        """Place the two regions back-to-back, in rows and in bytes."""
        q = self.query
        self.elements_per_plane = q.elements_per_plane
        self.reshapes_per_plane = q.reshapes_per_plane
        self.in_tiles_per_row = q.in_tiles_per_row
        self.out_tiles_per_row = q.out_tiles_per_row
        self.in_plane_stride = q.in_plane_stride
        self.out_height = q.out_height
        self.out_width = q.out_width
        self.output_rows = q.output_rows

        self.input_base_row = BASE_ROW
        self.output_base_row = self.input_base_row + q.input_rows

        self.input_base = self.input_base_row * ROW_BYTES
        self.output_base = self.output_base_row * ROW_BYTES

    # -- wide-vector FP32 state ---------------------------------------------

    @staticmethod
    def make_state() -> IpuState:
        """Build the FP32 wide-vector state this app requires.

        ``wide_vector_quantize_output=False`` keeps elements 4-byte FP32
        through AAQ, so ACTIVATE.QUANTIZE is a true pass-through -- values are
        moved verbatim -- and STR_POST_AAQ_REG drains the full 512 bytes.
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

    def setup(self, state: "IpuState") -> None:
        c, h, w = self.channels, self.height, self.width
        data = np.fromfile(self.input_path, dtype="<f4")
        if data.size != c * h * w:
            raise ValueError(
                f"input file {self.input_path} holds {data.size} FP32 values; "
                f"this shuffle needs {c * h * w}"
            )
        # Columns past W are zero. They are copied to real output lanes past
        # W*r, which teardown trims -- no real output column reads them.
        planes = np.zeros((c, h, self.in_tiles_per_row * LANES), dtype="<f4")
        planes[:, :, :w] = data.reshape(c, h, w)
        state.xmem.write_address(self.input_base, planes.tobytes())

        r = self.r
        # CR0 and CR1 are READ-ONLY: CR0 == 0 always, CR1 == 1 always. Both are
        # exploited directly -- CR0 as the zero source, CR1 as the 1.0 scalar
        # for the identity multiply and every +1 increment. All writable CRs
        # below hold row numbers, row strides, loop bounds, or the packed byte
        # index arrays the two LRD pairs are seeded from.
        state.regfile.set_cr(2, self.input_base_row)
        state.regfile.set_cr(3, self.output_base_row)
        # CR4 doubles as the h-loop bound: hbase steps by IN_TILES_PER_ROW and
        # reaches exactly H * IN_TILES_PER_ROW = IN_PLANE_STRIDE.
        state.regfile.set_cr(4, self.in_plane_stride)
        state.regfile.set_cr(5, r * self.in_plane_stride)
        state.regfile.set_cr(6, self.in_tiles_per_row)
        # CR7/CR14: the destination index array [0, r, 2r, ..., 7r], split
        # across the two LRs backing LRD14 (elements 0-3 low, 4-7 high).
        state.regfile.set_cr(7, _pack_bytes([4 * r, 5 * r, 6 * r, 7 * r]))
        # CR8 = r: the a-loop, sub-loop and plane-loop all run exactly r times.
        state.regfile.set_cr(8, r)
        state.regfile.set_cr(9, self.elements_per_plane)
        state.regfile.set_cr(10, RESHAPE_ELEMENTS * r)
        # CR11 = reshapes-per-plane minus one: that loop's BLT reads its counter
        # pre-increment (it is incremented in the same word).
        state.regfile.set_cr(11, self.reshapes_per_plane - 1)
        state.regfile.set_cr(12, _pack_bytes([0, 1, 2, 3]))
        state.regfile.set_cr(13, _pack_bytes([4, 5, 6, 7]))
        state.regfile.set_cr(14, _pack_bytes([0, r, 2 * r, 3 * r]))

        # MULT.*/ACTIVATE.QUANTIZE read the active-element count from the named
        # dstructure CR's valid_elements field. The asm names CR15 throughout,
        # so set CR15.valid_elements = 128 to process the full FP32 row.
        state.set_cr_dstructure(valid_elements=LANES)

    def teardown(self, state: "IpuState") -> None:
        if self.output_path is None:
            return
        raw = state.xmem.read_address(self.output_base, self.output_rows * ROW_BYTES)
        rows = np.frombuffer(raw, dtype="<f4").reshape(
            self.out_height, self.out_tiles_per_row * LANES
        )
        # Output lanes past W*r are the shuffled image of the input's zero
        # padding; trimming them is what makes the output file a dense
        # (H*r, W*r) array.
        np.ascontiguousarray(rows[:, : self.out_width]).tofile(self.output_path)

    def run(self, **kwargs):
        # Always run on the FP32 wide-vector state unless the caller supplied one.
        kwargs.setdefault("state", self.make_state())
        return super().run(**kwargs)


# -- registry declaration ---------------------------------------------------


def _query(params):
    return shuffle_query(params["shape"], upscale_factor=params["upscale_factor"])


def _supports(**params):
    q = _query(params)
    bad = geometry_refusal(q)
    if bad:
        return no(bad)
    if q.r not in SUPPORTED_FACTORS:
        detail = (
            f"its {RESHAPE_ELEMENTS * q.r} destination step exceeds the signed "
            f"byte ADDB reads, so the index array would walk backwards"
            if LANES % (RESHAPE_ELEMENTS * q.r) == 0
            else (
                f"it leaves {LANES}/{q.r} elements per plane per output tile, "
                f"which is not a whole number of {RESHAPE_ELEMENTS}-element "
                f"ACC.RESHAPE instructions"
            )
        )
        return no(
            f"cannot use upscale factor {q.r}: {detail}. Supported factors are "
            f"{', '.join(str(f) for f in SUPPORTED_FACTORS)}."
        )
    if q.out_channels != OUT_CHANNELS:
        return no(
            f"emits one output channel; a factor-{q.r} shuffle of "
            f"{q.channels} channels would emit {q.out_channels}. A "
            f"multi-channel shuffle needs an outer loop offsetting the plane "
            f"index by c'*{q.r * q.r}, which this kernel does not have."
        )
    budget = xmem_refusal(q)
    if budget:
        return no(budget)
    return yes()


def _build(**params):
    q = _query(params)
    return {
        "channels": q.channels,
        "height": q.height,
        "width": q.width,
        "upscale_factor": q.r,
    }


def _explain(**params):
    q = _query(params)
    return (
        f"a factor-{q.r} depth-to-space; each output row interleaves {q.r} "
        f"input planes at stride {q.r}, placed {RESHAPE_ELEMENTS} elements at a "
        f"time by {q.reshapes_per_plane} ACC.RESHAPE(s) per plane, turning "
        f"{q.channels}x{q.height}x{q.width} into {q.out_height}x{q.out_width}"
    )


def _caveats(**params):
    q = _query(params)
    notes = [WIDE_VECTOR_ONLY]
    lanes = lane_caveat(q)
    if lanes:
        notes.append(lanes)
    notes.append(headroom_caveat(q))
    return tuple(notes)


SPEC = KernelSpec(
    name="depth_to_space",
    op="depth_to_space",
    variant="planes",
    app_class=DepthToSpaceApp,
    asm="depth_to_space.asm",
    requires=("shape", "upscale_factor"),
    tags=("fp32-wide", "movement"),
    supports=_supports,
    build=_build,
    explain=_explain,
    caveats=_caveats,
    bundle=lambda **params: _query(params).bundle,
    # The only depth_to_space kernel.
    cost=lambda **params: 0.0,
)
