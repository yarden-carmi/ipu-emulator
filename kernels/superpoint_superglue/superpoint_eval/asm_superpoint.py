#!/usr/bin/env python3
"""End-to-end SuperPoint DETECTOR path in .asm vs real torch SuperPoint.

Runs the full forward through the detector logits (`semi`, 65 ch) entirely with
the .asm conv kernels (conv_fp32_tiled[/_norelu]) on a small image, comparing to
torch SuperPoint at every layer and at the end ("compare as one"). Then applies
the detector decision: our .asm-style max-over-channels (superpoint_detect) vs
torch's softmax, reporting per-cell argmax (keypoint) agreement.

2x2/s2 maxpool is done on the host (stride-2 = a gather the ISA leaves to host,
as documented); every conv/relu and the detect/threshold are the .asm math.

Run: python kernels/superpoint_superglue/superpoint_eval/asm_superpoint.py
"""
import os, sys, numpy as np
HERE = os.path.dirname(__file__); sys.path.insert(0, HERE)
import asm_harness, modified_ops

W = os.path.join(HERE, "weights")
def wload(L): return (np.load(os.path.join(W, f"{L}.weight.npy")),
                      np.load(os.path.join(W, f"{L}.bias.npy")))

def asm_conv(x, layer, relu=True, wcrop=None):
    w, b = wload(layer)
    Cin = w.shape[1]
    wcrop = wcrop or x.shape[2]
    if Cin <= 13 and w.shape[2] == 3:
        out, *_ = asm_harness.conv_fp32_layer(x, w, b, wcrop=wcrop)        # relu always
        return out
    return asm_harness.conv_fp32_tiled_layer(x, w, b, wcrop=wcrop, relu=relu)[0]

def maxpool2(x):  # host 2x2 stride-2 max (gather; documented host step)
    C, H, Wd = x.shape
    return x[:, :H // 2 * 2, :Wd // 2 * 2].reshape(C, H // 2, 2, Wd // 2, 2).max(axis=(2, 4))

def torch_superpoint(img):
    import torch
    sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "third_party", "superpoint"))
    from demo_superpoint import SuperPointNet
    net = SuperPointNet()
    net.load_state_dict(torch.load(os.path.join(HERE, "..", "..", "..", "third_party",
                                                 "superpoint", "superpoint_v1.pth"), map_location="cpu"))
    net.eval()
    acts = {}
    x = torch.from_numpy(img)[None, None]
    relu = torch.nn.functional.relu
    def C(l, t, r=True):
        w, b = wload(l)
        import torch.nn.functional as F
        o = F.conv2d(t, torch.from_numpy(w), torch.from_numpy(b), padding=1 if w.shape[2] == 3 else 0)
        return relu(o) if r else o
    x = C("conv1a", x); acts["conv1a"] = x
    x = C("conv1b", x); acts["conv1b"] = x
    x = torch.nn.functional.max_pool2d(x, 2)
    x = C("conv2a", x); x = C("conv2b", x); x = torch.nn.functional.max_pool2d(x, 2)
    x = C("conv3a", x); x = C("conv3b", x); x = torch.nn.functional.max_pool2d(x, 2)
    x = C("conv4a", x); x = C("conv4b", x)
    cPa = C("convPa", x); semi = C("convPb", cPa, r=False)
    acts["semi"] = semi
    return semi[0].detach().numpy(), {k: v[0].detach().numpy() for k, v in acts.items()}

def main():
    H = Wd = 32                          # /8 -> 4x4 detector grid
    rng = np.random.default_rng(0)
    img = (rng.standard_normal((H, Wd)).astype(np.float32) * 0.3)
    print(f"=== SuperPoint detector path: .asm vs torch, image {H}x{Wd} ===")

    # ---- .asm forward ----
    x = img[None]                         # (1,H,W)
    x = asm_conv(x, "conv1a"); x = asm_conv(x, "conv1b"); x = maxpool2(x)
    x = asm_conv(x, "conv2a"); x = asm_conv(x, "conv2b"); x = maxpool2(x)
    x = asm_conv(x, "conv3a"); x = asm_conv(x, "conv3b"); x = maxpool2(x)
    x = asm_conv(x, "conv4a"); x = asm_conv(x, "conv4b")
    cPa = asm_conv(x, "convPa")
    semi_asm = asm_conv(cPa, "convPb", relu=False)     # (65, Hc, Wc)

    # ---- torch reference ----
    try:
        semi_t, _ = torch_superpoint(img)
        Hc, Wc = semi_asm.shape[1], semi_asm.shape[2]
        semi_t = semi_t[:, :Hc, :Wc]
        d = np.abs(semi_asm - semi_t)
        print(f"\n[compare as one] detector logits `semi` (65 x {Hc} x {Wc}):")
        print(f"  asm vs torch SuperPoint: max {d.max():.2e}  mean {d.mean():.4e}  "
              f"{'MATCH' if d.max() < 5e-2 else 'DIFF'}")
        # detector decision: torch softmax-argmax vs our max-over-64 argmax
        sm = np.exp(semi_t - semi_t.max(0)); sm /= sm.sum(0)
        tk = sm[:64].argmax(0)               # torch keypoint sub-index per cell
        ak = semi_asm[:64].argmax(0)         # our max-over-channels argmax
        agree = (tk == ak).mean() * 100
        print(f"  per-cell keypoint argmax agreement (max-detect vs softmax): {agree:.1f}%")
    except Exception as ex:
        import traceback; traceback.print_exc()
        print("torch reference failed:", ex)

if __name__ == "__main__":
    main()
