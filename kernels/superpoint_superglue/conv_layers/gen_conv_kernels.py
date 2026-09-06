#!/usr/bin/env python3
"""Generate one .asm kernel per SuperPoint convolution layer.

Each layer maps to one of two base kernels (their bodies are emitted verbatim
with a per-layer header documenting Cin->Cout, kernel size, ReLU, and the
channel-group config the host must set):
  3x3 + ReLU  -> conv_fp32_full.asm   (conv1a..conv4b, convPa, convDa)
  1x1 no ReLU -> conv1x1.asm          (convPb, convDb)
Spatial size (H,W) and addresses remain CR runtime params (resolution varies);
the per-layer channel config (G, ngroups) is documented per file.

Run: python gen_conv_kernels.py   (writes <layer>.asm next to it)
"""
import os
HERE = os.path.dirname(__file__)
KD = os.path.join(HERE, "..")
G = 13  # channels per group (G*9 <= 128 for 3x3; G*1 for 1x1)

# (name, Cin, Cout, ksize, relu)
LAYERS = [
    ("conv1a", 1, 64, 3, True), ("conv1b", 64, 64, 3, True),
    ("conv2a", 64, 64, 3, True), ("conv2b", 64, 64, 3, True),
    ("conv3a", 64, 128, 3, True), ("conv3b", 128, 128, 3, True),
    ("conv4a", 128, 128, 3, True), ("conv4b", 128, 128, 3, True),
    ("convPa", 128, 256, 3, True), ("convPb", 256, 65, 1, False),
    ("convDa", 128, 256, 3, True), ("convDb", 256, 256, 1, False),
]

def body(path):
    """Return the kernel body (drop the leading // header comment block)."""
    lines = open(path).read().splitlines()
    # header runs from the first '// ===' to the matching closing '// ===' line
    i = 0
    ends = [k for k, l in enumerate(lines) if l.startswith("// ===")]
    start = ends[1] + 1 if len(ends) >= 2 else 0
    return "\n".join(lines[start:]).strip("\n")

BODY_3x3 = body(os.path.join(KD, "conv_fp32_full.asm"))
BODY_1x1 = body(os.path.join(KD, "conv1x1.asm"))

for name, cin, cout, k, relu in LAYERS:
    cinp = cin + 1                      # +1 ones channel (bias)
    ngroups = (cinp + G - 1) // G
    base = BODY_3x3 if k == 3 else BODY_1x1
    gws = G * (9 if k == 3 else 1) * 4
    hdr = (
        f"// ============================================================================\n"
        f"// {name}.asm  --  SuperPoint {name}: {cin}->{cout}, {k}x{k}, "
        f"{'ReLU' if relu else 'no ReLU'}\n"
        f"// ----------------------------------------------------------------------------\n"
        f"// One full output channel per launch (loops rows x col-tiles x channel-groups).\n"
        f"// Base kernel: {'conv_fp32_full' if k==3 else 'conv1x1'}. Run {cout} times (one per\n"
        f"// output channel), advancing OUT (CR3) and weights_base (CR9) by the host.\n"
        f"// Channel config: Cin'={cinp} (incl. bias ones-channel), G={G}, ngroups={ngroups},\n"
        f"//   group_weight_stride={gws} (CR13), G(CR11)={G}, ngroups(CR12)={ngroups}.\n"
        f"// Set CR2=IN, CR3=OUT, CR4=in_chan_stride, CR5=RW_in, CR6=H, CR7=512,\n"
        f"//   CR8=RW_out, CR9=weights_base, CR10=128, CR14=num_col_tiles, CR15=dtype.\n"
        f"// ============================================================================\n"
    )
    out = os.path.join(HERE, f"{name}.asm")
    open(out, "w").write(hdr + "\n" + base + "\n")
    print(f"wrote {name}.asm  ({cin}->{cout} {k}x{k} {'relu' if relu else 'norelu'}, ngroups={ngroups})")
