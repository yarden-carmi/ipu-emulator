"""Normalization applications -- the ``l2_normalize`` operation.

One directory per kernel, matching the softmax, convolution and pooling trees.

Shared query normalisation and the XMEM budget arithmetic live in
:mod:`~ipu_apps.normalize._spec_support`. There are no framework-layer adapters
here: ``torch`` exposes L2 normalization as ``nn.functional.normalize``, a
function rather than a layer class, so there is no class name to dispatch on.

Nothing is re-exported here: discovery walks the tree and collects each
kernel's module-level ``SPEC``, so importing this package must stay cheap and
side-effect-free.
"""

from __future__ import annotations

__all__: list[str] = []
