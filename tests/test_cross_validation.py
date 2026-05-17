"""Cross-validation: dcl_core.core3d converges to dcl_core.core as N -> infinity.

The integer-token (``core3d``) and continuous-amplitude (``core``)
submodules implement the same physics at different resolutions.  The
integer-token formulation should reproduce the continuous-amplitude
dynamics exactly in the limit ``n_units -> infinity``.

This file is the de-facto spec linking the two submodules: any
divergence between them, at any sufficiently large ``n_units``, is
a bug in one of the two implementations.

STUB
----
These tests are skeletons.  They are marked ``@pytest.mark.skip``
pending the integer-token implementation in :mod:`dcl_core.core3d`.
Remove the skip once ``core3d`` has enough machinery to run the
named scenarios.

Why a separate test file
------------------------
Tests in ``tests/core/`` and ``tests/core3d/`` exercise each
submodule in isolation.  This file exercises the *relationship*
between the two -- a cross-cutting concern that needs its own
home so the convergence claim is documented, runnable, and
flagged if it regresses.
"""

from __future__ import annotations

import pytest


@pytest.mark.skip(reason="cross-validation pending core3d implementation")
def test_free_propagation_matches_in_large_N_limit() -> None:
    """Free-particle Gaussian wavepacket evolves identically in both cores.

    Initialise the same Gaussian on a small lattice in both
    submodules, propagate for K ticks at successively larger
    ``n_units``, and check that the per-site amplitude difference
    shrinks as ``1 / sqrt(n_units)`` (Poisson scaling of the
    token-count noise).
    """
    ...


@pytest.mark.skip(reason="cross-validation pending core3d implementation")
def test_two_body_orbit_locks_in_both_cores() -> None:
    """An exp_12-style two-body Coulomb orbit settles to the Bohr radius in both.

    Run the inherited exp_12 initial condition in both submodules at
    matched parameters; assert both find an ``r_pdf`` within
    tolerance of ``R_1 = 10.3`` after ``N`` ticks.

    This is the load-bearing convergence test: it ties the
    integer-token implementation's correctness to Paper~I's
    published 4-significant-figure hydrogen result.
    """
    ...


@pytest.mark.skip(reason="cross-validation pending core3d implementation")
def test_conservation_invariants_agree() -> None:
    """Both submodules conserve total probability over K ticks.

    ``core`` conserves it to float tolerance (post-renormalisation);
    ``core3d`` conserves it as an exact integer identity.  At any
    ``n_units``, ``core3d``'s totals must equal ``n_units`` exactly,
    and ``core``'s totals must equal ``1.0`` to within
    ``len(lattice) * eps``.
    """
    ...
