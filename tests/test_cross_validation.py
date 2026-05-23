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


@pytest.mark.skip(reason="cross-validation pending core3d implementation + coupling machinery")
def test_arnold_tongue_locations_agree() -> None:
    """Coupled-session frequency-locking tongues sit in the same places.

    Set up two coupled sessions (or one session + a periodic external
    potential) at intrinsic frequency ``omega_A`` and driver
    ``omega_B``.  Sweep ``(omega_B / omega_A, coupling_strength K)``
    in 2D and identify the rational-ratio plateaus where the
    sessions phase-lock -- the Arnold tongues.

    For the framework's continuum-limit claim to hold, the tongue
    locations (rational ratios) and widths must agree between
    ``core`` (continuous baseline) and ``core3d`` (integer-token)
    at sufficiently large ``n_units``.  This is a sharp test:
    tongues are a *structural* feature of the dynamics, not a
    local quantity, so shifts or width changes would expose any
    silent break in the integer-token formulation -- including
    phase-information loss in :class:`BresenhamResidual`'s carry
    (see ``notes/bresenham_residual_design.md`` for why a real
    carry might shift tongues that a complex carry would preserve).

    This test is part of Paper III's discrete-vs-continuous
    comparison; the scan logic likely lives downstream in
    ``external/dcl-paper-03-tidal-ionization`` and this entry runs
    a minimal smoke version against a fixed (omega_A, omega_B, K)
    grid corner.
    """
    ...
