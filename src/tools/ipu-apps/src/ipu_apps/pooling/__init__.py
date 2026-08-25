"""Pooling applications -- the ``maxpool2d`` operation.

One directory per kernel, matching the softmax and convolution trees. The
kernels here differ on *stride*, which is what makes them separate programs
rather than one parameterised one: a stride-2 pool has to decimate its result
(``ACC.STRIDE``) and a stride-1 pool must not, and neither the decimation phase
nor the halo width can be chosen at run time.

Shared query normalisation, the XMEM budget arithmetic and the framework-layer
adapters live in :mod:`~ipu_apps.pooling._spec_support`.

Nothing is re-exported here: discovery walks the tree and collects each
kernel's module-level ``SPEC``, so importing this package must stay cheap and
side-effect-free.
"""

from __future__ import annotations

__all__: list[str] = []
