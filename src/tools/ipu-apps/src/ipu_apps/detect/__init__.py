"""Detector-head applications.

The read-out steps of SuperPoint's keypoint detector: reducing the channel
planes to a per-cell confidence, and gating scores against a threshold.

These are the operations the ISA *can* express. The rest of ``simple_nms`` and
the top-k cap cannot be -- there is no vector compare, no boolean vector and no
lane-index extraction -- and are documented as host work rather than left as an
apparent coverage hole. See ``docs/content/kernels/superpoint.md``.

Shared query normalisation and the resident-threshold convention live in
:mod:`~ipu_apps.detect._spec_support`. There are no framework-layer adapters:
neither operation is a ``torch.nn`` layer.

Nothing is re-exported here: discovery walks the tree and collects each
kernel's module-level ``SPEC``, so importing this package must stay cheap and
side-effect-free.
"""

from __future__ import annotations

__all__: list[str] = []
