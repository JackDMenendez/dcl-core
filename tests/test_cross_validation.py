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


def test_free_propagation_matches_in_large_N_limit() -> None:
    """A propagating core3d wavepacket converges to its own continuous limit.

    Reframed self-convergence test (2026-06-02).  The original intent
    -- "the same Gaussian evolves *identically* in ``core`` and
    ``core3d``" -- is not well-posed at a fixed lattice: the two
    submodules implement structurally different hop kernels (core3d:
    uniform average, periodic boundaries, one chirality per tick;
    core: directed momentum-weighted, open boundaries, both
    chiralities per massive tick), so they pin different observables
    at fixed spacing and agree only in the *double* limit ``a -> 0``
    AND ``n_units -> infinity`` (see
    ``dcl-delta-p-min/notes/cross_engine_equivalence.md``).  The
    cross-engine match therefore lives in the (still-skipped)
    two-body / Arnold-tongue tests; this test isolates the piece that
    IS clean at fixed lattice: that core3d's integer-token density
    tracks the continuous amplitude it is a quantisation of, with the
    error vanishing as ``n_units -> infinity``.

    Mechanism.  Each tick the engine forms the renormalised analytical
    density ``d_cont = |psi_R_new|^2 + |psi_L_new|^2`` and the
    BresenhamResidual quantises it to integer counts ``N`` summing
    exactly to ``n_units`` (see ``scheduler.step``).  After ``K``
    ticks of genuine propagation (nonzero ``momentum``), the integer
    density ``d_int = (N_R + N_L) / n_units`` must approach ``d_cont``.
    Because Bresenham is a *deterministic* error-diffusion (not random
    rounding), the per-site error is ``O(1 / n_units)`` -- the L2 error
    over the lattice scales as ``sqrt(n_sites) / n_units``, faster than
    the ``1 / sqrt(n_units)`` a Poisson-noise model would give.
    """
    import numpy as np

    from dcl_core.core3d import (
        BipartiteLattice,
        DiscreteCausalSession,
        HopOperator,
        TickScheduler,
    )

    shape = (16, 16, 16)
    n_sites = shape[0] * shape[1] * shape[2]
    center = (8, 8, 8)
    sigma = 2.0
    omega = 0.1
    momentum = (0.15, 0.0, 0.0)  # nonzero -> the packet actually propagates
    n_ticks = 8

    def final_tick_error(n_units: int) -> float:
        """L2(d_int - d_cont) for the last of ``n_ticks`` propagation steps.

        ``d_cont`` is computed by replaying the final tick's hop +
        renormalisation on the tick-(K-1) state -- exactly the target
        the scheduler then hands to ``BresenhamResidual.quantise`` --
        so ``d_int`` is precisely the integer quantisation of
        ``d_cont`` with no feedback from earlier ticks contaminating
        the difference.  This is white-box by design: it mirrors
        ``scheduler.step``'s renorm formula on purpose.
        """
        lattice = BipartiteLattice(shape=shape)
        session = DiscreteCausalSession.wavepacket(
            lattice,
            n_units=n_units,
            omega=omega,
            center=center,
            sigma=sigma,
            momentum=momentum,
        )
        scheduler = TickScheduler(
            lattice=lattice, hop=HopOperator(lattice=lattice)
        )
        scheduler.register(session)
        for _ in range(n_ticks - 1):
            scheduler.step()

        # Continuous target the final tick will quantise (same parity,
        # same unit-norm renormalisation as scheduler.step).
        parity = scheduler.parity_now()
        psi_R_new, psi_L_new = scheduler.hop.step(session, parity)
        norm = np.sqrt(
            np.sum(np.abs(psi_R_new) ** 2) + np.sum(np.abs(psi_L_new) ** 2)
        )
        psi_R_new, psi_L_new = psi_R_new / norm, psi_L_new / norm
        d_cont = np.abs(psi_R_new) ** 2 + np.abs(psi_L_new) ** 2

        # Integer density after the final tick is the quantisation of d_cont.
        scheduler.step()
        d_int = (session.N_R + session.N_L).astype(np.float64) / n_units

        # Sanity: both are probability densities (exact for the integer one).
        assert session.total_tokens() == n_units
        np.testing.assert_allclose(d_cont.sum(), 1.0, atol=1e-9)

        return float(np.sqrt(np.sum((d_int - d_cont) ** 2)))

    n_units_values = [10**3, 10**4, 10**5, 10**6]
    errors = [final_tick_error(n) for n in n_units_values]

    # (1) Absolute bound: deterministic Bresenham keeps the L2 error within
    # O(sqrt(n_sites) / n_units).  The 3x prefactor leaves >5x headroom over
    # the empirically observed constant (~0.5) across this range.
    for n_units, l2 in zip(n_units_values, errors):
        bound = 3.0 * np.sqrt(n_sites) / n_units
        assert l2 <= bound, (
            f"L2 error {l2:.3e} exceeds O(1/n_units) bound {bound:.3e} "
            f"at n_units={n_units}"
        )

    # (2) Convergence rate: each 10x in n_units cuts the error by at least
    # 3x.  Poisson noise would give only ~3.16x and so could fail to clear
    # a 3x bar consistently; the deterministic engine is observed at >=5x,
    # so this both confirms convergence and that it beats Poisson scaling.
    for coarse, fine in zip(errors[:-1], errors[1:]):
        assert fine <= coarse / 3.0, (
            f"error did not fall >=3x per decade: {coarse:.3e} -> {fine:.3e}"
        )


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


def test_conservation_invariants_agree() -> None:
    """Both submodules conserve total probability over K ticks.

    ``core`` conserves it to float tolerance (post-renormalisation);
    ``core3d`` conserves it as an exact integer identity.  At any
    ``n_units``, ``core3d``'s totals must equal ``n_units`` exactly,
    and ``core``'s totals must equal ``1.0`` to within the framework's
    own ``< 1e-10`` "well-behaved" bound (see
    ``dcl_core.core.UnityConstraint.unity_residual``).

    This is the one cross-validation invariant that does **not**
    require the two engines' *dynamics* to coincide: each conserves
    its own A=1 invariant under its own hop kernel.  The
    convergence-of-dynamics tests (free propagation, two-body lock-in)
    are gated on reconciling the engines' distinct discretisations --
    core3d hops a uniform average with periodic boundaries and one
    chirality per tick, while core hops a directed momentum-weighted
    kernel with open boundaries and both chiralities per (massive)
    tick.  See ``dcl-delta-p-min/notes/cross_engine_equivalence.md``.
    """
    import numpy as np

    from dcl_core.core import CausalSession, OctahedralLattice
    from dcl_core.core3d import (
        BipartiteLattice,
        DiscreteCausalSession,
        HopOperator,
        TickScheduler,
    )

    shape = (12, 12, 12)
    omega = 0.3
    n_ticks = 25
    center = (shape[0] // 2, shape[1] // 2, shape[2] // 2)

    # --- core (continuous amplitude): A=1 to float tolerance ---------------
    # enforce_unity_spinor renormalises the *global* sum to 1.0 after each
    # tick; the residual is the framework's documented "well-behaved" bound.
    core_tol = 1e-10
    core_session = CausalSession(
        lattice=OctahedralLattice(*shape),
        initial_node=center,
        instruction_frequency=omega,
    )
    for _ in range(n_ticks):
        core_session.tick()
        core_session.advance_tick_counter()
        core_total = float(core_session.probability_density().sum())
        assert abs(core_total - 1.0) <= core_tol, (
            f"core A=1 drift {abs(core_total - 1.0):.3e} exceeds "
            f"{core_tol:.0e} after a tick"
        )

    # --- core3d (integer tokens): A=1 as an EXACT integer identity ---------
    # No tolerance: sum_x (N_R + N_L) == n_units, bit for bit, every tick.
    n_units = 1_000_000
    c3_lattice = BipartiteLattice(shape=shape)
    c3_session = DiscreteCausalSession.delta_at(
        c3_lattice, n_units=n_units, omega=omega, position=center
    )
    scheduler = TickScheduler(
        lattice=c3_lattice, hop=HopOperator(lattice=c3_lattice)
    )
    scheduler.register(c3_session)
    for _ in range(n_ticks):
        scheduler.step()
        assert c3_session.total_tokens() == n_units, (
            f"core3d A=1 violated at tick {scheduler.tick}: "
            f"{c3_session.total_tokens()} != {n_units}"
        )


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
