"""Standard (dense) convolution kernels -- the ``conv2d`` operation.

Every kernel under this package implements ``op="conv2d"``: a dense
convolution reading an ``(Cin, H, W)`` activation and a ``(Cout, Cin, kh, kw)``
weight. Depthwise and grouped convolutions are a different operation with a
different dataflow and will live in sibling families, not here.

Shared query normalisation and the framework-layer adapters live in
:mod:`~ipu_apps.convolutions_universal.conv._spec_support`.
"""

from __future__ import annotations

__all__: list[str] = []
