"""dcl_core -- Core library for the A=1 Discrete Causal Lattice framework.

The package exposes two submodules with complementary roles:

- :mod:`dcl_core.core` -- Paper~I's continuous-amplitude engine
  (``psi_R`` / ``psi_L`` complex arrays), ported verbatim from the
  Paper~I v1.0 deposit (doi:10.5281/zenodo.20078529).  Backwards-
  compatible API for existing experiments; A=1 is enforced via
  float-tolerance renormalisation (``enforce_unity_spinor``).
- :mod:`dcl_core.core3d` -- the new design with integer-token
  probability accounting (``N_RGB`` / ``N_CMY`` integer counts +
  ``phi_RGB`` / ``phi_CMY`` phase fields), a ``TokenResidual``
  fractional-bit carry, and CPU/GPU backend split.  A=1 is an exact
  integer identity: ``sum_x (N_RGB + N_CMY) == n_units``.

**No top-level shortcuts are re-exported.**  Callers pick their
submodule explicitly -- e.g.

    >>> from dcl_core.core import OctahedralLattice, CausalSession
    >>> from dcl_core.core3d import BipartiteLattice, DiscreteCausalSession

This is deliberate: the two submodules have different shapes
(continuous amplitude vs integer tokens), and a top-level shortcut
would hide which engine is in play.  Sharing only ``__version__``
at the top level keeps the choice visible at every import site.

Public API contract:
    - Adding a re-export from either submodule's ``__init__.py`` is
      a MINOR version bump (backwards compatible).
    - Removing or changing a re-export signature is a MAJOR version
      bump.
    - Pure internal refactoring is a PATCH.

The ``.claude/agents/api-stability-reviewer.md`` agent enforces this
on diffs.
"""

from dcl_core._version import __version__
from dcl_core import core, core3d

__all__ = [
    "__version__",
    "core",
    "core3d",
]
