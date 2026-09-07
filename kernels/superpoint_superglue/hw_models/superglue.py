#!/usr/bin/env python3
"""Hardware-faithful SuperGlue -- DROP-IN replacement for the Magic Leap
`models/superglue.py`. Same class/API/weights. Every op is computed as our IPU
`.asm` kernels do it, INCLUDING math-equivalent steps:

  Q/K/V/out projections, merge MLP, final proj, score    matmul   conv1x1.asm
  attention QKt, attn@V                                   matmul   (activation x activation)
  attention softmax                                       base-2   softmax.asm  (exp2)
  optimal transport (Sinkhorn)                            tropical sinkhorn_iter.asm +
                                                                   sinkhorn_col.asm   [*behavioural]
  matching argmax / mutual-NN / threshold                 host     argmax_match.asm, topk.asm

[*] the only non-equivalent step: the hardware OT is the tropical (max-plus)
    Sinkhorn -- row half-step Z-=max_j, col half-step Z-=max_i, on the
    dustbin-augmented coupling, NO log-marginal (log_mu/log_nu) terms -- instead
    of the official log-domain logsumexp Sinkhorn. The attention softmax base
    change (2^x vs e^x) is exact, so the GNN is mathematically identical.
"""
from copy import deepcopy
from pathlib import Path
from typing import List, Tuple
import torch
from torch import nn

LOG2E = 1.4426950408889634


def MLP(channels: List[int], do_bn: bool = True) -> nn.Module:
    n = len(channels); layers = []
    for i in range(1, n):
        layers.append(nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n - 1):
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def normalize_keypoints(kpts, image_shape):
    _, _, height, width = image_shape
    one = kpts.new_tensor(1)
    size = torch.stack([one * width, one * height])[None]
    center = size / 2
    scaling = size.max(1, keepdim=True).values * 0.7
    return (kpts - center[:, None, :]) / scaling[:, None, :]


class KeypointEncoder(nn.Module):
    def __init__(self, feature_dim, layers):
        super().__init__()
        self.encoder = MLP([3] + layers + [feature_dim])
        nn.init.constant_(self.encoder[-1].bias, 0.0)

    def forward(self, kpts, scores):
        inputs = [kpts.transpose(1, 2), scores.unsqueeze(1)]
        return self.encoder(torch.cat(inputs, dim=1))


def attention(query, key, value):
    """Scaled dot-product attention; softmax via 2^x (softmax.asm). == stock."""
    dim = query.shape[1]
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim ** .5
    s = (scores - scores.max(dim=-1, keepdim=True).values) * LOG2E
    e = torch.pow(2.0, s)
    prob = e / e.sum(dim=-1, keepdim=True)
    return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob


class MultiHeadedAttention(nn.Module):
    def __init__(self, num_heads, d_model):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(self, query, key, value):
        b = query.size(0)
        query, key, value = [l(x).view(b, self.dim, self.num_heads, -1)
                             for l, x in zip(self.proj, (query, key, value))]
        x, _ = attention(query, key, value)
        return self.merge(x.contiguous().view(b, self.dim * self.num_heads, -1))


class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim, num_heads):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim * 2, feature_dim * 2, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x, source):
        message = self.attn(x, source, source)
        return self.mlp(torch.cat([x, message], dim=1))


class AttentionalGNN(nn.Module):
    def __init__(self, feature_dim, layer_names):
        super().__init__()
        self.layers = nn.ModuleList([AttentionalPropagation(feature_dim, 4)
                                     for _ in range(len(layer_names))])
        self.names = layer_names

    def forward(self, desc0, desc1):
        for layer, name in zip(self.layers, self.names):
            src0, src1 = (desc1, desc0) if name == 'cross' else (desc0, desc1)
            delta0, delta1 = layer(desc0, src0), layer(desc1, src1)
            desc0, desc1 = (desc0 + delta0), (desc1 + delta1)
        return desc0, desc1


def log_optimal_transport(scores, alpha, iters: int):
    """Tropical (max-plus) Sinkhorn -- sinkhorn_iter.asm (row Z-=max_j) +
    sinkhorn_col.asm (col Z-=max_i), T iters on the dustbin-augmented coupling.
    No log-marginal terms (the hardware's behaviour)."""
    b, m, n = scores.shape
    bins0 = alpha.expand(b, m, 1); bins1 = alpha.expand(b, 1, n); a = alpha.expand(b, 1, 1)
    Z = torch.cat([torch.cat([scores, bins0], -1), torch.cat([bins1, a], -1)], 1)
    for _ in range(iters):
        Z = Z - Z.max(dim=2, keepdim=True).values
        Z = Z - Z.max(dim=1, keepdim=True).values
    return Z


def arange_like(x, dim):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1


class SuperGlue(nn.Module):
    """Hardware-faithful SuperGlue (drop-in for models/superglue.py)."""
    default_config = {
        'descriptor_dim': 256, 'weights': 'indoor',
        'keypoint_encoder': [32, 64, 128, 256],
        'GNN_layers': ['self', 'cross'] * 9,
        'sinkhorn_iterations': 100, 'match_threshold': 0.2,
    }

    def __init__(self, config):
        super().__init__()
        self.config = {**self.default_config, **config}
        self.kenc = KeypointEncoder(self.config['descriptor_dim'], self.config['keypoint_encoder'])
        self.gnn = AttentionalGNN(self.config['descriptor_dim'], self.config['GNN_layers'])
        self.final_proj = nn.Conv1d(self.config['descriptor_dim'], self.config['descriptor_dim'], 1, bias=True)
        self.register_parameter('bin_score', torch.nn.Parameter(torch.tensor(1.)))
        assert self.config['weights'] in ['indoor', 'outdoor']
        for base in [Path(__file__).parent / 'weights',
                     Path(__file__).parents[3] / 'third_party/superglue/models/weights']:
            p = base / 'superglue_{}.pth'.format(self.config['weights'])
            if p.exists():
                self.load_state_dict(torch.load(str(p))); break
        print('Loaded SuperGlue model ("{}" weights) [hardware-faithful]'.format(self.config['weights']))

    def forward(self, data):
        desc0, desc1 = data['descriptors0'], data['descriptors1']
        kpts0, kpts1 = data['keypoints0'], data['keypoints1']
        if kpts0.shape[1] == 0 or kpts1.shape[1] == 0:
            shape0, shape1 = kpts0.shape[:-1], kpts1.shape[:-1]
            return {'matches0': kpts0.new_full(shape0, -1, dtype=torch.int),
                    'matches1': kpts1.new_full(shape1, -1, dtype=torch.int),
                    'matching_scores0': kpts0.new_zeros(shape0),
                    'matching_scores1': kpts1.new_zeros(shape1)}
        kpts0 = normalize_keypoints(kpts0, data['image0'].shape)
        kpts1 = normalize_keypoints(kpts1, data['image1'].shape)
        desc0 = desc0 + self.kenc(kpts0, data['scores0'])
        desc1 = desc1 + self.kenc(kpts1, data['scores1'])
        desc0, desc1 = self.gnn(desc0, desc1)
        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)
        scores = torch.einsum('bdn,bdm->bnm', mdesc0, mdesc1)
        scores = scores / self.config['descriptor_dim'] ** .5
        scores = log_optimal_transport(scores, self.bin_score, iters=self.config['sinkhorn_iterations'])
        max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
        indices0, indices1 = max0.indices, max1.indices
        mutual0 = arange_like(indices0, 1)[None] == indices1.gather(1, indices0)
        mutual1 = arange_like(indices1, 1)[None] == indices0.gather(1, indices1)
        zero = scores.new_tensor(0)
        mscores0 = torch.where(mutual0, max0.values.exp(), zero)
        mscores1 = torch.where(mutual1, mscores0.gather(1, indices1), zero)
        valid0 = mutual0 & (mscores0 > self.config['match_threshold'])
        valid1 = mutual1 & valid0.gather(1, indices1)
        indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
        indices1 = torch.where(valid1, indices1, indices1.new_tensor(-1))
        return {'matches0': indices0, 'matches1': indices1,
                'matching_scores0': mscores0, 'matching_scores1': mscores1}
