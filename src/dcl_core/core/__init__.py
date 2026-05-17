"""dcl_core.core -- Paper~I's continuous-amplitude framework engine.

Provenance
----------
This subpackage is a verbatim copy of Paper~I's ``src/core/``
(doi:10.5281/zenodo.20078529).  The six module files
(``OctahedralLattice``, ``PhaseOscillator``, ``UnityConstraint``,
``CausalSession``, ``TickScheduler``, ``CompositeCausalSession``)
are reproduced unmodified; only this ``__init__.py`` is new (it
mirrors Paper~I's own ``src/core/__init__.py``).

The intent is backwards-compatible reproducibility.  Experiments
written against Paper~I's

    from src.core import OctahedralLattice, CausalSession, enforce_unity_spinor

migrate to ``dcl_core`` by changing the import to

    from dcl_core.core import OctahedralLattice, CausalSession, enforce_unity_spinor

with no behavioural change.  The migration is a one-liner per
import site.

State of A=1 conservation in this submodule
-------------------------------------------
Paper~I's continuous-amplitude formulation uses float arrays
(``psi_R``, ``psi_L``) and enforces A=1 via
``enforce_unity_spinor`` (renormalisation to unit total probability
density at the end of every tick).  This is a *float-tolerance*
enforcement, not an exact integer identity.  The
:mod:`dcl_core.core3d` submodule re-implements the same physics
with integer tokens so that A=1 becomes an exact equality
``sum_x (N_R + N_L) == n_units``.

The two submodules implement the same physics at different
resolutions.  As ``n_units -> infinity`` in ``core3d``, its
dynamics should converge to ``core``'s.  The
``tests/test_cross_validation.py`` skeleton documents the
convergence test that links the two.

Public API
----------
Lifted from Paper~I's ``src/core/__init__.py``.
"""

from dcl_core.core.OctahedralLattice import (
    OctahedralLattice,
    RGB_VECTORS,
    CMY_VECTORS,
    ALL_VECTORS,
    SUBLATTICE_SIZE,
    COORDINATION_NUMBER,
    active_vectors,
    EVEN_TICK,
    ODD_TICK,
)
from dcl_core.core.PhaseOscillator import PhaseOscillator
from dcl_core.core.UnityConstraint import (
    enforce_unity,
    unity_residual,
    is_unity,
    enforce_unity_spinor,
    unity_residual_spinor,
)
from dcl_core.core.CausalSession import CausalSession
from dcl_core.core.TickScheduler import TickScheduler, ShuffleScheme
from dcl_core.core.CompositeCausalSession import CompositeCausalSession

__all__ = [
    "OctahedralLattice",
    "RGB_VECTORS",
    "CMY_VECTORS",
    "ALL_VECTORS",
    "SUBLATTICE_SIZE",
    "COORDINATION_NUMBER",
    "active_vectors",
    "EVEN_TICK",
    "ODD_TICK",
    "PhaseOscillator",
    "enforce_unity",
    "unity_residual",
    "is_unity",
    "enforce_unity_spinor",
    "unity_residual_spinor",
    "CausalSession",
    "TickScheduler",
    "ShuffleScheme",
    "CompositeCausalSession",
]
