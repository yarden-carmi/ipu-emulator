"""Convolution applications.

One sub-package per convolution family -- :mod:`~ipu_apps.convolutions_universal.conv`
for standard (dense) convolution, with depthwise and pointwise families to
follow. Each family holds one directory per kernel, matching the softmax tree's
``softmax/softmax_rows/`` shape.

Nothing is re-exported here: discovery walks the tree and collects each
kernel's module-level ``SPEC``, so importing this package must stay cheap and
side-effect-free.
"""

from __future__ import annotations

__all__: list[str] = []
