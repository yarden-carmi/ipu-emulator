"""Reshape applications -- the ``depth_to_space`` operation.

Data movement rather than arithmetic, but not therefore free: the ISA has no
scatter-store, so a rearrangement that a CPU does with a view has to be built
out of ``ACC.RESHAPE``'s eight-element scatter into R_ACC.

Shared query normalisation and the ``PixelShuffle`` layer adapter live in
:mod:`~ipu_apps.reshape._spec_support`.

Nothing is re-exported here: discovery walks the tree and collects each
kernel's module-level ``SPEC``, so importing this package must stay cheap and
side-effect-free.
"""

from __future__ import annotations

__all__: list[str] = []
