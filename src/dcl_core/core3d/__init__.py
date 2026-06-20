"""dcl_core.core3d -- Integer-token A=1 framework with token-residual carry.

This submodule is the active design point.  Where :mod:`dcl_core.core`
inherits Paper~I's continuous-amplitude formulation, ``core3d`` re-
implements the same physics with **integer-token probability accounting**:

- ``N_RGB``, ``N_CMY`` are ``int64`` arrays of indistinguishable
  probability tokens (one per chirality field).
- ``phi_RGB``, ``phi_CMY`` are continuous ``float64`` phase arrays.
- A=1 is the exact integer identity

      sum_x (N_RGB(x) + N_CMY(x)) == n_units            (no float tolerance)

- The hop operator's analytical update produces fractional new token
  counts ``N_target(x) = n_units * |psi_new(x)|^2``; the
  :class:`TokenResidual` accumulator carries the fractional bits
  tick-to-tick so the global integer identity holds without per-tick
  renormalisation.

CPU (NumPy) and GPU (CuPy) backends are addressable through a single
API surface; the backend is selected at
:class:`BipartiteLattice`-construction time.

State of implementation (v0.1.0)
--------------------------------
All five public operators are implemented and pass their unit
suites under ``tests/core3d/`` (lattice, session, hop, remainder,
scheduler, clifford, continuum-limit).  The remaining v1.0 work is
the cross-submodule convergence layer in
``tests/test_cross_validation.py`` (``core`` <-> ``core3d`` in the
``n_units -> infinity`` limit); see ``CLAUDE.md``'s ``CURRENT
STATUS`` for the v1.0 roadmap.

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
from dcl_core.core3d.remainder import TokenResidual, BresenhamResidual
from dcl_core.core3d.scheduler import TickScheduler
from dcl_core.core3d.gauge import uniform_B_potential

__all__ = [
    "BipartiteLattice",
    "DiscreteCausalSession",
    "HopOperator",
    "TokenResidual",
    "TickScheduler",
    # Gauge-field construction helper (v0.3.0; Peierls coupling).
    "uniform_B_potential",
    # Deprecated alias (renamed TokenResidual); kept for backward-compat.
    "BresenhamResidual",
]
