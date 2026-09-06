"""NumPy SuperPoint ops written to match each .asm kernel's exact math.

conv2d_relu matches conv_fp32.asm: 3x3, zero-pad=1, + bias, + ReLU.
(SuperPoint's encoder/head convs are exactly conv+bias+ReLU, so this is also
the true op for those layers.)
"""
import numpy as np

def conv2d_relu(x, w, b, relu=True):
    """x:(Cin,H,W) w:(Cout,Cin,3,3) b:(Cout,) -> (Cout,H,W), zero-pad=1."""
    Cin, H, W = x.shape
    Cout = w.shape[0]
    xp = np.zeros((Cin, H + 2, W + 2), np.float32)
    xp[:, 1:H + 1, 1:W + 1] = x
    y = np.zeros((Cout, H, W), np.float32)
    for oc in range(Cout):
        acc = np.full((H, W), b[oc], np.float32)
        for ci in range(Cin):
            for kr in range(3):
                for kc in range(3):
                    acc += w[oc, ci, kr, kc] * xp[ci, kr:kr + H, kc:kc + W]
        y[oc] = np.maximum(acc, 0.0) if relu else acc
    return y
