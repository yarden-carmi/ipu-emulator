#!/usr/bin/env python3
"""Hardware-faithful SuperPoint -- DROP-IN replacement for the Magic Leap
`models/superpoint.py`. Same class/API/weights; every operation is computed the
way our IPU `.asm` kernels do it, INCLUDING mathematically-equivalent steps
(softmax via 2^x, descriptor L2 via rsqrt, grid_sample as an explicit 4-corner
bilinear), so a side-by-side run isolates exactly where hardware diverges.

Op -> kernel:
  conv1a..conv4b, convPa, convDa  3x3 conv + ReLU   conv_fp32_full.asm
  pool                            2x2/s2 max        shifted max + stride-2 gather
  convPb, convDb                  1x1 conv          conv1x1.asm
  detector softmax(65)            base-2 softmax    softmax.asm   (exp2)
  depth-to-space reshape          plane relocation  pixel_shuffle.asm
  simple_nms                      3x3/Rx max NMS    maxpool_shift.asm
  score threshold                 relu(s-tau)       topk.asm
  top-k cap                       calibrated tau    topk_mt.asm + host bisection  [*behavioural]
  descriptor L2 normalize         x*rsqrt(sum x^2)  l2_normalize.asm  (inv_sqrt)
  sample_descriptors grid_sample  4-corner bilinear grid_sample (precomputed weights)

[*] the only non-equivalent step: the hardware top-k cap is a calibrated
    threshold (no sort), so it keeps ~k keypoints (k +/- the bisection precision)
    rather than exactly k. Everything else is mathematically identical to stock.
"""
from pathlib import Path
import torch
from torch import nn

LOG2E = 1.4426950408889634


def hw_conv2d(x, weight, bias, pad: int):
    """Direct convolution as the hardware computes it (conv_fp32_full.asm): a
    sum over the k*k spatial taps, each a constant-offset shifted contiguous load
    times a 1x1 weight (a per-tap channel MAC), accumulated in FP32 -- NOT
    cuDNN's Winograd/FFT path. Zero border padding. weight:(Cout,Cin,kh,kw)."""
    Cout, Cin, kh, kw = weight.shape
    b, c, H, W = x.shape
    xp = torch.nn.functional.pad(x, (pad, pad, pad, pad))         # zero border
    out = bias.view(1, Cout, 1, 1).expand(b, Cout, H, W).clone()  # bias (ones-channel)
    for dy in range(kh):
        for dx in range(kw):                                      # tap = shifted load
            tap = xp[:, :, dy:dy + H, dx:dx + W]
            out = out + torch.einsum('oc,bchw->bohw', weight[:, :, dy, dx], tap)  # MAC + ACC
    return out


def maxpool_shift(x, kernel: int, stride: int, pad: int):
    """Window max as the hardware computes it (maxpool_shift.asm): a running
    element-wise max over the kernel*kernel shifted taps (each a constant-offset
    contiguous load), with a -3.4e38 border seed. Bit-identical to
    F.max_pool2d(kernel, stride, pad)."""
    nd = x.dim()
    if nd == 3:
        x = x[:, None]                                        # (b,H,W) -> (b,1,H,W)
    NEG = -3.4e38
    H, W = x.shape[-2:]
    xp = torch.nn.functional.pad(x, (pad, pad, pad, pad), value=NEG)
    Ho = (H + 2 * pad - kernel) // stride + 1
    Wo = (W + 2 * pad - kernel) // stride + 1
    out = x.new_full((*x.shape[:2], Ho, Wo), NEG)
    for dy in range(kernel):
        for dx in range(kernel):                              # tap (dy,dx) = shifted load
            tap = xp[:, :, dy:dy + stride * (Ho - 1) + 1:stride,
                          dx:dx + stride * (Wo - 1) + 1:stride]
            out = torch.maximum(out, tap)                     # ACC.MAX
    return out[:, 0] if nd == 3 else out


def exp2_softmax(x, dim):
    """softmax via 2^x (softmax.asm): 2^(log2e*(x-m)) / sum.  == F.softmax."""
    s = (x - x.max(dim=dim, keepdim=True).values) * LOG2E
    e = torch.pow(2.0, s)
    return e / e.sum(dim=dim, keepdim=True)


def l2_normalize(x, dim):
    """x * rsqrt(sum x^2) (l2_normalize.asm uses inv_sqrt; guarded at 0)."""
    n2 = (x * x).sum(dim=dim, keepdim=True)
    return x * torch.where(n2 > 0, torch.rsqrt(n2), torch.zeros_like(n2))


def simple_nms(scores, nms_radius: int):
    """Maxpool-based NMS (maxpool_shift.asm local max via shifted loads)."""
    assert nms_radius >= 0
    def max_pool(x):
        return maxpool_shift(x, kernel=nms_radius * 2 + 1, stride=1, pad=nms_radius)
    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)
    for _ in range(2):
        supp_mask = max_pool(max_mask.float()) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == max_pool(supp_scores)
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    return torch.where(max_mask, scores, zeros)


def remove_borders(keypoints, scores, border: int, height: int, width: int):
    mask_h = (keypoints[:, 0] >= border) & (keypoints[:, 0] < (height - border))
    mask_w = (keypoints[:, 1] >= border) & (keypoints[:, 1] < (width - border))
    mask = mask_h & mask_w
    return keypoints[mask], scores[mask]


def top_k_keypoints(keypoints, scores, k: int):
    """Hardware top-k cap: no sort. topk_mt.asm emits the soft survivor count
    sum sigmoid(T(s-tau)); the HOST binary-searches tau so count ~ k, then keeps
    {s > tau} (topk.asm relu gate). tau -> the k-th largest score, so the set
    equals torch.topk's set up to the bisection precision + ties."""
    if k >= len(keypoints):
        return keypoints, scores
    T = 64.0
    lo, hi = float(scores.min()), float(scores.max())
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        cnt = torch.sigmoid(T * (scores - mid)).sum().item()
        lo, hi = (mid, hi) if cnt > k else (lo, mid)
    tau = 0.5 * (lo + hi)
    mask = scores > tau
    return keypoints[mask], scores[mask]


def sample_descriptors(keypoints, descriptors, s: int = 8):
    """Bilinear descriptor sampling as the EXPLICIT 4-corner weighted sum the
    kernels compute (align_corners=True, zero padding) + L2(rsqrt), instead of
    F.grid_sample/F.normalize. Matches stock to FP rounding."""
    b, c, h, w = descriptors.shape
    kp = keypoints - s / 2 + 0.5
    kp = kp / torch.tensor([(w * s - s / 2 - 0.5), (h * s - s / 2 - 0.5)]).to(kp)[None]
    kp = kp * 2 - 1                                       # grid in [-1, 1], (b, K, 2)
    gx, gy = kp[..., 0], kp[..., 1]                       # (b, K)
    ix = (gx + 1) / 2 * (w - 1)
    iy = (gy + 1) / 2 * (h - 1)
    x0 = torch.floor(ix); y0 = torch.floor(iy)
    wx = ix - x0; wy = iy - y0
    out = descriptors.new_zeros(b, c, kp.shape[1])
    for dy, wyk in ((0, 1 - wy), (1, wy)):
        for dx, wxk in ((0, 1 - wx), (1, wx)):
            xi = (x0 + dx).long(); yi = (y0 + dy).long()
            valid = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
            xc = xi.clamp(0, w - 1); yc = yi.clamp(0, h - 1)
            wgt = (wyk * wxk) * valid.to(descriptors.dtype)   # (b, K)
            for bi in range(b):
                gathered = descriptors[bi, :, yc[bi], xc[bi]]  # (c, K)
                out[bi] += gathered * wgt[bi][None]
    return l2_normalize(out, dim=1)


class SuperPoint(nn.Module):
    """Hardware-faithful SuperPoint (drop-in for models/superpoint.py)."""
    default_config = {
        'descriptor_dim': 256, 'nms_radius': 4, 'keypoint_threshold': 0.005,
        'max_keypoints': -1, 'remove_borders': 4,
    }

    def __init__(self, config):
        super().__init__()
        self.config = {**self.default_config, **config}
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5 = 64, 64, 128, 128, 256
        self.conv1a = nn.Conv2d(1, c1, 3, 1, 1); self.conv1b = nn.Conv2d(c1, c1, 3, 1, 1)
        self.conv2a = nn.Conv2d(c1, c2, 3, 1, 1); self.conv2b = nn.Conv2d(c2, c2, 3, 1, 1)
        self.conv3a = nn.Conv2d(c2, c3, 3, 1, 1); self.conv3b = nn.Conv2d(c3, c3, 3, 1, 1)
        self.conv4a = nn.Conv2d(c3, c4, 3, 1, 1); self.conv4b = nn.Conv2d(c4, c4, 3, 1, 1)
        self.convPa = nn.Conv2d(c4, c5, 3, 1, 1); self.convPb = nn.Conv2d(c5, 65, 1, 1, 0)
        self.convDa = nn.Conv2d(c4, c5, 3, 1, 1)
        self.convDb = nn.Conv2d(c5, self.config['descriptor_dim'], 1, 1, 0)
        for p in [Path(__file__).parent / 'weights/superpoint_v1.pth',
                  Path(__file__).parents[3] / 'third_party/superglue/models/weights/superpoint_v1.pth']:
            if p.exists():
                self.load_state_dict(torch.load(str(p))); break
        mk = self.config['max_keypoints']
        if mk == 0 or mk < -1:
            raise ValueError('"max_keypoints" must be positive or "-1"')
        print('Loaded SuperPoint model [hardware-faithful]')

    def forward(self, data):
        cv = lambda t, conv, p=1: self.relu(hw_conv2d(t, conv.weight, conv.bias, p))  # direct MAC
        mp = lambda t: maxpool_shift(t, kernel=2, stride=2, pad=0)   # 2x2/s2 = max of 4 taps
        x = cv(data['image'], self.conv1a); x = cv(x, self.conv1b); x = mp(x)
        x = cv(x, self.conv2a); x = cv(x, self.conv2b); x = mp(x)
        x = cv(x, self.conv3a); x = cv(x, self.conv3b); x = mp(x)
        x = cv(x, self.conv4a); x = cv(x, self.conv4b)

        cPa = cv(x, self.convPa)
        scores = hw_conv2d(cPa, self.convPb.weight, self.convPb.bias, 0)   # 1x1, no ReLU
        scores = exp2_softmax(scores, 1)[:, :-1]                 # softmax.asm (exp2)
        b, _, h, w = scores.shape
        scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8)   # pixel_shuffle.asm
        scores = scores.permute(0, 1, 3, 2, 4).reshape(b, h * 8, w * 8)
        scores = simple_nms(scores, self.config['nms_radius'])       # maxpool_shift.asm

        keypoints = [torch.nonzero(s > self.config['keypoint_threshold']) for s in scores]
        scores = [s[tuple(k.t())] for s, k in zip(scores, keypoints)]
        keypoints, scores = list(zip(*[
            remove_borders(k, s, self.config['remove_borders'], h * 8, w * 8)
            for k, s in zip(keypoints, scores)]))
        if self.config['max_keypoints'] >= 0:
            keypoints, scores = list(zip(*[
                top_k_keypoints(k, s, self.config['max_keypoints'])
                for k, s in zip(keypoints, scores)]))
        keypoints = [torch.flip(k, [1]).float() for k in keypoints]

        cDa = cv(x, self.convDa)
        descriptors = hw_conv2d(cDa, self.convDb.weight, self.convDb.bias, 0)   # 1x1
        descriptors = l2_normalize(descriptors, 1)               # l2_normalize.asm (rsqrt)
        descriptors = [sample_descriptors(k[None], d[None], 8)[0]
                       for k, d in zip(keypoints, descriptors)]
        return {'keypoints': keypoints, 'scores': scores, 'descriptors': descriptors}
