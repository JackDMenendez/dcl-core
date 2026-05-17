"""dcl_core.core3d -- Integer-token A=1 framework with Bresenham residual.

This submodule is the active design point.  Where :mod:`dcl_core.core`
inherits Paper~I's continuous-amplitude formulation, ``core3d`` re-
implements the same physics with **integer-token probability accounting**:

- ``N_R``, ``N_L`` are ``int64`` arrays of indistinguishable
  probability tokens (one per chirality field).
- ``phi_R``, ``phi_L`` are continuous ``float64`` phase arrays.
- A=1 is the exact integer identity

      sum_x (N_R(x) + N_L(x)) == n_units            (no float tolerance)

- The hop operator's analytical update produces fractional new token
  counts ``N_target(x) = n_units * |psi_new(x)|^2``; the
  :class:`BresenhamResidual` accumulator carries the fractional bits
  tick-to-tick so the global integer identity holds without per-tick
  renormalisation.

CPU (NumPy) and GPU (CuPy) backends are addressable through a single
API surface; the backend is selected at
:class:`BipartiteLattice`-construction time.

State of implementation (v0.1.0-dev)
------------------------------------
All implementations in this submodule are deliberate stubs and
raise :class:`NotImplementedError`.  The architectural decisions
(interfaces, public API, naming, documentation conventions) are
deliberately separated from the implementation work; see
``CLAUDE.md``'s ``CURRENT STATUS`` for the implementation roadmap.

Public API contract
-------------------
- Adding a re-export here is a MINOR version bump.
- Removing or changing a re-export signature is a MAJOR version
  bump.
- Pure internal refactoring is a PATCH.

Pre-1.0 (``0.X.Y``) signals "API still unstable" -- minor bumps may
break callers.
"""

from dcl_core.core3d.lattice import BipartiteLattice
from dcl_core.core3d.session import DiscreteCausalSession
from dcl_core.core3d.hop import HopOperator
from dcl_core.core3d.remainder import BresenhamResidual
from dcl_core.core3d.scheduler import TickScheduler

__all__ = [
    "BipartiteLattice",
    "DiscreteCausalSession",
    "HopOperator",
    "BresenhamResidual",
    "TickScheduler",
]
