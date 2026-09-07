# Dense descriptor interpolation

The descriptor kernels compute every image pixel's descriptor from an HWC
FP32 coarse map: `[60,80,256] -> [480,640,256]`. Neither normalizes its input
or output. The direct `sample_descriptors.asm` supports shared and stock
weights; `sample_descriptors_separable.asm` optimizes stock interpolation.
The registry resolves `mode="stock"` to the separable implementation and
`mode="shared"` to the direct implementation by default.

## Two coordinate mappings

At pixel `(y,x)=(8i+a,8j+b)`, with `a,b` in `0..7`, the kernel computes

\[
O[y,x,c]=\sum_{d_y,d_x=-1}^{1}
K_{i,j}[d_y+1,d_x+1,a,b]D[i+d_y,j+d_x,c].
\]

Out-of-map descriptors are zero. In the direct implementation all nine
coefficients are consumed. Each coefficient multiplies all 256 channels;
channels are never mixed.

| Mode | Coarse sampling coordinate | Logical coefficient shape |
|---|---|---|
| `shared` (default) | `u=(x-3.5)/8`, `v=(y-3.5)/8` | `[3,3,8,8]` |
| `stock` | `u=(x-3.5)*(W-1)/(8W-4.5)`, likewise for `v,H` | separable `[H,3,8]` + `[W,3,8]`; direct `[H,W,3,3,8,8]` |

Both generators use the bilinear tent function:

\[
K_{i,j}=\max(0,1-|v-i-d_y|)\max(0,1-|u-j-d_x|).
\]

Generated kernels have at most four nonzero taps. Shared mode also accepts
arbitrary caller-provided nine-tap weights, including negative coefficients.
It is equivalent to applying 64 shared spatial filters independently to every
descriptor channel, then rearranging their outputs into each `8x8` image cell.
The assembly performs that rearrangement through its store addresses.

Stock mode follows the FP32 normalized-grid construction and
`align_corners=True` unnormalization in the checked-in
`third_party/superglue/models/superpoint.py`. This reproduces that function's
**interpolation stage**, within FP32 rounding; the final L2 normalization is
intentionally excluded. The mapping differs from shared mode because the
stock weights depend on the cell position as well as the within-cell phase.
The precomputation preserves the original sequence of FP32 operations instead
of evaluating only the simplified coordinate formula above.

## Separable stock implementation

Stock's 2D coefficients factor as `K[i,j,dy,dx,a,b] = Ky[i,dy,a] * Kx[j,dx,b]`.
The optimized assembly uses this factorization directly:

```text
T[a,j,c]          = sum_dy Ky[i,dy,a] * D[i+dy-1,j,c]
O[8i+a,8j+b,c]    = sum_dx Kx[j,dx,b] * T[a,j+dx-1,c]
```

For each coarse row, the vertical pass produces an eight-row strip, then the
horizontal pass consumes it. Only that strip is kept in scratch; it is reused
for the next coarse row. Horizontal neighbor vectors stay in three cyclic
register slots across all eight output phases, avoiding repeated loads.

The coefficient shapes are `[60,3,8]` and `[80,3,8]`: **3,360 floats, or
13.125 KiB logically**. Packing one axis cell's 24 coefficients into each XMEM
row takes **70 KiB**, replacing the direct stock path's 21.094 MiB tables.
The stock sampling coordinates and zero padding are identical. FP32 rounding
can differ because vertical sums are rounded before horizontal interpolation.
Custom shared tables need not be separable and continue to use all nine taps
of the direct implementation.

## Memory and assembly

The current emulator allocates **512 MiB XMEM**. Both full-resolution variants
run in **one launch**, including the entire **300 MiB output**.

| Region | Shared, direct | Stock, direct | Stock, separable (default) |
|---|---:|---:|---:|
| Padded coarse descriptors | 4.965 MiB | 4.965 MiB | 4.965 MiB |
| Packed coefficients | 4.5 KiB | 21.094 MiB | 70 KiB |
| Scratch | 0 | 0 | 656 KiB |
| Dense output | 300 MiB | 300 MiB | 300 MiB |
| Total | 304.969 MiB | 326.059 MiB | 305.674 MiB |

Each XMEM row contains 128 FP32 values (512 bytes). A descriptor occupies two
consecutive rows. Input descriptors have a one-cell zero halo. In the direct
path each coefficient table occupies nine rows in spatial tap order, with
phases `8*a+b` in lanes `0..63`. In the separable path, one row stores one
axis cell's `[tap,phase]` coefficients in lanes `0..23`; unused lanes are zero.
Its scratch layout is `[8,W+2,256]`, including zero horizontal halos. The output
is contiguous HWC data without channel padding or a separate shuffle buffer.

`MULT.RC.VE` broadcasts one coefficient over a 128-channel tile;
`ACC.ADD.FIRST` and `ACC.ADD` accumulate the taps. Each assembly header specifies
every register, stride, and pipeline dependency. Optional coarse-row start/count
parameters support subregion tests and callers; the default case processes
the entire map and does not split it into bands.

## Calling it

Both assembly files live in the `sample_descriptors` package, with two registry
entries (`SPECS`), one shared memory harness, and one preparation/case/test suite.
They use the standard registry `KernelCase` runner for assembly,
execution, validation, output export, and measured cycle reporting. There is
no descriptor-specific runner or benchmark CLI. The harness is a `MemoryApp`
with a memory layout and CR map; coefficient generation and reference checks
stay outside it.

Callers with their own descriptors use `sample_descriptors.prepare.pack_input` and the registered
memory harness: `sample_descriptors` for direct or `sample_descriptors_separable`
for separable. Pass `separable=True, mode="stock"` to `pack_input` for the latter.
Both harnesses accept `shape=(H,W,256)`, `mode`, and optional
`cell_row_start` / `cell_row_count`. The shape always describes the complete
coarse map, so stock coordinates and halo neighbors remain correct for
subregions. Registry resolution of the operation with `mode="stock"` prefers
`sample_descriptors_separable`; naming the direct kernel explicitly retains
its original memory contract.

```python
from ipu_apps.kernel_registry import create_harness
from ipu_apps.kernels.sample_descriptors.prepare import pack_input

# descriptors: FP32 HWC array; instructions.bin is the assembled separable kernel.
image, layout = pack_input(descriptors, mode="stock", separable=True)
image.tofile("input.bin")
app = create_harness(
    "sample_descriptors_separable",
    params=dict(shape=descriptors.shape, mode="stock"),
    bindings=dict(input_path="input.bin", inst_path="instructions.bin",
                  output_path="dense.bin"),
)
state, cycles = app.run(max_cycles=20_000_000)
assert state.is_halted
print(f"Completed: {cycles} cycles")
```

`dense.bin` contains raw little-endian FP32 HWC output. Shared callers instead
import `sample_descriptors.prepare.pack_input`, optionally supply a
`weights=` table, and select the `sample_descriptors` harness with `mode="shared"`.

```bash
bazel test //src/tools/ipu-apps:sample_descriptors //src/tools/ipu-apps:sample_descriptors_separable
bazel run //src/tools/ipu-apps:sample_descriptors_separable

# Full-size registry case through Bazel; the runner prints measured cycles:
bazel run //src/tools/ipu-apps:sample_descriptors_separable -- \
  --height 60 --width 80 --amplitude 100 --max-cycles 20000000 \
  --output /tmp/stock-descriptors.bin

# Shared output, including its difference from stock coordinates:
bazel run //src/tools/ipu-apps:sample_descriptors -- \
  --height 60 --width 80 --amplitude 100 --compare-stock --max-cycles 20000000

# Original nine-tap stock implementation:
bazel run //src/tools/ipu-apps:sample_descriptors -- \
  --case stock --height 60 --width 80 --max-cycles 20000000
```

The full checks execute all 78,643,200 output values and compare one image row
at a time against an independent four-corner reference, with `rtol=1e-5` and
`atol=1e-5 * amplitude` (the default amplitude is 1). The absolute tolerance
scales with the input values; all reported errors remain in the original units.
They take substantially longer than the small cases in the Python emulator.
The registry runner prints the actual cycle counter returned by the emulator
after the program halts. Cycle counts are
not inferred from elapsed time or calculated from loop bounds.

The `--compare-stock` option also reports the output's maximum absolute error, mean
absolute error, and RMSE against the **stock-coordinate** reference. These
cross-mapping errors are separate from the shared kernel's correctness error
against its own reference. The fixture uses seed 31, uniform values in
`[-1,1]`, and no L2 normalization; different descriptor maps give different
cross-mapping errors. Use `--amplitude A` to sample uniformly from `[-A,A]`;
the default is `1`, and the log records the selected range. Full-size runs are
opt-in CLI overrides; the registered defaults remain small for CI.
