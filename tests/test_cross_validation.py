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
    TokenResidual quantises it to integer counts ``N`` summing
    exactly to ``n_units`` (see ``scheduler.step``).  After ``K``
    ticks of genuine propagation (nonzero ``momentum``), the integer
    density ``d_int = (N_RGB + N_CMY) / n_units`` must approach ``d_cont``.
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
        the scheduler then hands to ``TokenResidual.quantise`` --
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
        d_int = (session.N_RGB + session.N_CMY).astype(np.float64) / n_units

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


@pytest.mark.slow
def test_two_body_orbit_locks_in_both_cores() -> None:
    """A hydrogen-like orbit locks in to a *stable* radius in both engines.

    Respec (2026-06-03).  The original spec asserted both engines
    settle to ``r_pdf ~ R_1 = 10.3`` -- but that is false at a fixed
    lattice.  As ``dcl-delta-p-min`` Phase 2 established, core3d's hop
    kernel (uniform average, periodic boundaries, one chirality per
    tick) is a *different* discretisation of the Dirac dynamics than
    Paper~I/core's (directed momentum-weighted, open boundaries, both
    chiralities per massive tick), so at fixed spacing the two engines
    pin **different** ground-state radii (core ~12, core3d ~21 on this
    33^3 / strength-30 well); they agree only in the *double* limit
    ``a -> 0`` AND ``n_units -> infinity`` (see
    ``dcl-delta-p-min/notes/cross_engine_equivalence.md`` and
    ``exp_12_dp_min_sweep.py``).

    What IS true at fixed lattice -- and what this test certifies -- is
    the qualitative lock-in: started as an orbiting wavepacket in the
    same fixed Coulomb well, **each engine settles to a stable,
    stationary radial scale** (the per-tick peak radius stops drifting
    and holds with low variance over the scoring window), at a finite
    interior radius (no collapse to the centre, no dispersal to the
    box).  The settled radii differ between engines by design -- the
    test also pins that divergence so a future change that
    accidentally made them coincide at fixed spacing would be flagged
    for review rather than silently accepted.

    Setup mirrors Paper~I ``exp_10_standalone`` (continuous) and
    ``exp_12_dp_min_sweep`` (integer): single session in a fixed
    ``V(r) = -30 / (r + 0.5)`` well, Gaussian packet of width 1.5 with
    tangential momentum ``k = 1/R_1`` along ``V2 = (1, -1, -1)``.
    core3d has no pairwise coupling in v0.1.0, so the "two-body" orbit
    IS one session in the proton's fixed Coulomb well (the standard
    single-particle-hydrogen reduction).
    """
    import numpy as np

    # --- shared physics (Paper~I exp_10 / exp_12 parameters) ---------------
    STRENGTH, SOFTENING, OMEGA, WIDTH = 30.0, 0.5, 0.1019, 1.5
    R1 = 10.3  # only sets the launch offset + tangential k, NOT an assert target
    GRID, TICKS, BURN_IN = 33, 130, 50
    N_UNITS = 10**5
    V2_DIR = np.array([1.0, -1.0, -1.0])
    K_TANG = 1.0 / R1

    wc = (GRID // 2, GRID // 2, GRID // 2)
    dr = int(round(R1 / np.sqrt(3.0)))
    start = tuple(min(wc[i] + dr, GRID - 3) for i in range(3))

    xx, yy, zz = np.meshgrid(*(np.arange(GRID),) * 3, indexing="ij")
    r_field = np.sqrt(
        (xx - wc[0]) ** 2 + (yy - wc[1]) ** 2 + (zz - wc[2]) ** 2
    )
    r_max = float(r_field.max())
    V = -STRENGTH / (r_field + SOFTENING)

    def radial_peak(prob: np.ndarray, n_bins: int = 60) -> float:
        """Peak of the volume-binned radial PDF (matches exp_12)."""
        edges = np.linspace(0.0, r_max, n_bins + 1)
        pdf, _ = np.histogram(r_field.ravel(), bins=edges, weights=prob.ravel())
        centres = 0.5 * (edges[:-1] + edges[1:])
        return float(centres[int(np.argmax(pdf))])

    def settled(traj: list[float]) -> tuple[float, float]:
        """Return (mean radius, coefficient of variation) over the window."""
        a = np.asarray(traj)
        return float(a.mean()), float(a.std() / a.mean())

    # --- core3d (integer tokens): one session + external Coulomb well -------
    # core3d v0.1.0's TickScheduler does not thread external_potential, so we
    # drive the documented tick body directly (exactly exp_12's run_one_tick).
    from dcl_core.core3d import (
        BipartiteLattice,
        TokenResidual,
        DiscreteCausalSession,
        HopOperator,
    )

    c3_lattice = BipartiteLattice(shape=(GRID, GRID, GRID), backend="cpu")
    c3_hop = HopOperator(lattice=c3_lattice)
    c3_res = TokenResidual(lattice=c3_lattice)
    c3_session = DiscreteCausalSession.wavepacket(
        c3_lattice, n_units=N_UNITS, omega=OMEGA, center=start, sigma=WIDTH,
        momentum=tuple(K_TANG * V2_DIR),
    )
    c3_traj: list[float] = []
    for tick in range(TICKS):
        parity = "even" if tick % 2 == 0 else "odd"
        psi_R_new, psi_L_new = c3_hop.step(
            c3_session, parity, external_potential=V
        )
        norm_sq = float(
            np.sum(np.abs(psi_R_new) ** 2) + np.sum(np.abs(psi_L_new) ** 2)
        )
        if norm_sq > 0.0:
            inv = 1.0 / np.sqrt(norm_sq)
            psi_R_new, psi_L_new = psi_R_new * inv, psi_L_new * inv
        N_RGB_int, N_CMY_int = c3_res.quantise(
            np.abs(psi_R_new) ** 2 * N_UNITS,
            np.abs(psi_L_new) ** 2 * N_UNITS,
            N_UNITS,
        )
        c3_session.N_RGB[...] = N_RGB_int
        c3_session.N_CMY[...] = N_CMY_int
        c3_session.phi_RGB[...] = np.angle(psi_R_new)
        c3_session.phi_CMY[...] = np.angle(psi_L_new)
        # A=1 is an exact integer identity at every tick.
        assert c3_session.total_tokens() == N_UNITS
        if tick >= BURN_IN:
            density = (c3_session.N_RGB + c3_session.N_CMY).astype(np.float64) / N_UNITS
            c3_traj.append(radial_peak(density))
    r_c3, cv_c3 = settled(c3_traj)

    # --- core (continuous amplitude): same well via topological_potential ---
    from dcl_core.core import CausalSession, OctahedralLattice
    from dcl_core.core.UnityConstraint import enforce_unity_spinor

    core_lattice = OctahedralLattice(GRID, GRID, GRID)
    core_lattice.topological_potential[...] = V
    core_session = CausalSession(
        lattice=core_lattice, initial_node=start, instruction_frequency=OMEGA
    )
    # Inject exp_10's Gaussian packet with tangential momentum (the engine's
    # ctor seeds a delta; overwrite psi to match the integer engine's launch).
    envelope = np.exp(-0.5 * ((xx - start[0]) ** 2 + (yy - start[1]) ** 2
                              + (zz - start[2]) ** 2) / WIDTH**2)
    phase = K_TANG * (xx - yy - zz)  # k . x along V2 = (1, -1, -1)
    packet = (envelope * np.exp(1j * phase)).astype(complex)
    core_session.psi_R = packet / np.sqrt(2.0)
    core_session.psi_L = packet / np.sqrt(2.0)
    enforce_unity_spinor(core_session.psi_R, core_session.psi_L)
    core_traj: list[float] = []
    for tick in range(TICKS):
        core_session.tick()
        core_session.advance_tick_counter()
        density = core_session.probability_density()
        # A=1 holds to float tolerance (post-renormalisation).
        assert abs(float(density.sum()) - 1.0) <= 1e-9
        if tick >= BURN_IN:
            core_traj.append(radial_peak(density))
    r_core, cv_core = settled(core_traj)

    # --- assertions: both lock in to a stable interior radius --------------
    for name, r_settled, cv in (
        ("core", r_core, cv_core),
        ("core3d", r_c3, cv_c3),
    ):
        # Finite, interior: not collapsed to the centre, not dispersed to the
        # box face/corner.  (r_max is the corner distance.)
        assert 2.0 < r_settled < 0.92 * r_max, (
            f"{name} settled radius {r_settled:.2f} not interior "
            f"(0 < r < {0.92 * r_max:.1f})"
        )
        # Locked in: the peak radius holds steady over the scoring window.
        assert cv < 0.20, (
            f"{name} radius did not stabilise: cv={cv:.3f} over the window"
        )

    # --- the engine-protocol divergence is real at fixed lattice -----------
    # Tripwire on the documented mismatch: core ~12, core3d ~21 here.  If a
    # future change makes them coincide at fixed spacing, that is a genuine
    # convergence result -- revisit this test rather than loosening it.
    rel_divergence = abs(r_c3 - r_core) / r_core
    assert rel_divergence > 0.2, (
        f"engines unexpectedly agree at fixed lattice: core={r_core:.2f}, "
        f"core3d={r_c3:.2f} (rel {rel_divergence:.2f}); convergence at fixed "
        f"spacing would contradict the double-limit finding -- review needed"
    )


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
    # No tolerance: sum_x (N_RGB + N_CMY) == n_units, bit for bit, every tick.
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


@pytest.mark.slow
def test_arnold_tongue_locations_agree() -> None:
    """Both engines show frequency-dependent orbital lock-in in a fixed well.

    Respec (2026-06-03).  Frequency-locking shows up in this framework
    in two forms (Paper~I ``exp_09_harmonics``):

    - **Part D -- inter-session locking:** several *dynamical* sessions
      at incommensurate ``omega`` drift toward rational frequency
      ratios.  This is the 2-parameter Arnold tongue (frequency ratio
      x coupling strength) and it needs inter-session coupling, which
      ``core3d`` v0.1.0's ``TickScheduler`` does not have (pairwise
      interaction is deferred to v0.2.0).  That richer tongue is the
      v0.2.0 follow-on.
    - **Part B -- orbital resonance (single session):** one packet in a
      fixed clock-density / Coulomb well, sweeping ``omega``, locks
      into a stable orbit whose tightness depends on ``omega``.  This
      is the form ``dcl-delta-p-min``'s ``exp_09_dp_min_floor`` uses
      for its ``core3d`` column (proton = fixed well, the standard
      single-particle reduction), and it runs on **both** engines at
      v0.1.0 via the same single-session + ``external_potential`` loop
      as the two-body test.

    This test certifies the Part~B form.  It sweeps ``omega`` for a
    packet launched on a tangential orbit in the same fixed
    ``V = -30/(r + 0.5)`` well, and measures the orbital lock-in via
    the swing (max - min) of the centre-of-mass radius over the run:
    a tight lock-in is a small swing, a loose / detuned orbit a large
    one.  Each engine must show a **frequency-dependent** orbital
    response (the swing varies materially across ``omega`` -- i.e.
    ``omega`` tunes the orbit, the lock-in prerequisite) while staying
    bound throughout.

    As with the two-body test, it does NOT assert the engines lock at
    the same ``omega``: at fixed lattice they pin lock-in at different
    places (core tightest near the physical electron ``omega ~ 0.1``,
    core3d tightest at higher ``omega``), converging only in the double
    limit.  The test pins that the tightest-lock ``omega`` *differs*
    between engines, so a future change that made them coincide at
    fixed spacing is flagged for review.

    NOTE: this is the single-session orbital-resonance reduction, not
    the full coupled-oscillator Arnold tongue -- the latter waits on
    core3d v0.2.0 inter-session coupling, after which a minimal
    two-session ``(omega_A, omega_B, K)`` smoke can replace / augment
    this, with the PASS tolerance set by Paper~III.
    """
    import numpy as np

    # --- shared physics / well (matches the two-body test) -----------------
    STRENGTH, SOFTENING, WIDTH, R1 = 30.0, 0.5, 1.5, 10.3
    GRID, TICKS, TRANSIENT = 33, 70, 5
    N_UNITS = 10**5
    V2_DIR = np.array([1.0, -1.0, -1.0])
    K_TANG = 1.0 / R1
    OMEGAS = [0.1019, 0.5, 1.0]  # low (physical e-) -> high; the sweep axis

    wc = (GRID // 2, GRID // 2, GRID // 2)
    dr = int(round(R1 / np.sqrt(3.0)))
    start = tuple(min(wc[i] + dr, GRID - 3) for i in range(3))
    xx, yy, zz = np.meshgrid(*(np.arange(GRID),) * 3, indexing="ij")
    r_field = np.sqrt(
        (xx - wc[0]) ** 2 + (yy - wc[1]) ** 2 + (zz - wc[2]) ** 2
    )
    r_max = float(r_field.max())
    V = -STRENGTH / (r_field + SOFTENING)

    def com_radius(prob: np.ndarray) -> float:
        """Distance of the probability centre-of-mass from the well centre."""
        total = float(prob.sum())
        if total < 1e-12:
            return 0.0
        cx = float((xx * prob).sum() / total)
        cy = float((yy * prob).sum() / total)
        cz = float((zz * prob).sum() / total)
        return float(
            np.sqrt((cx - wc[0]) ** 2 + (cy - wc[1]) ** 2 + (cz - wc[2]) ** 2)
        )

    def swing_and_mean(traj: list[float]) -> tuple[float, float]:
        a = np.asarray(traj[TRANSIENT:])
        return float(a.max() - a.min()), float(a.mean())

    # --- core3d: single session in the well, swept over omega --------------
    from dcl_core.core3d import (
        BipartiteLattice,
        TokenResidual,
        DiscreteCausalSession,
        HopOperator,
    )

    c3_lattice = BipartiteLattice(shape=(GRID, GRID, GRID), backend="cpu")
    c3_hop = HopOperator(lattice=c3_lattice)

    def run_core3d(omega: float) -> list[float]:
        residual = TokenResidual(lattice=c3_lattice)
        session = DiscreteCausalSession.wavepacket(
            c3_lattice, n_units=N_UNITS, omega=omega, center=start,
            sigma=WIDTH, momentum=tuple(K_TANG * V2_DIR),
        )
        traj: list[float] = []
        for tick in range(TICKS):
            parity = "even" if tick % 2 == 0 else "odd"
            psi_R, psi_L = c3_hop.step(session, parity, external_potential=V)
            norm_sq = float(np.sum(np.abs(psi_R) ** 2) + np.sum(np.abs(psi_L) ** 2))
            if norm_sq > 0.0:
                inv = 1.0 / np.sqrt(norm_sq)
                psi_R, psi_L = psi_R * inv, psi_L * inv
            N_RGB_int, N_CMY_int = residual.quantise(
                np.abs(psi_R) ** 2 * N_UNITS, np.abs(psi_L) ** 2 * N_UNITS, N_UNITS
            )
            session.N_RGB[...], session.N_CMY[...] = N_RGB_int, N_CMY_int
            session.phi_RGB[...], session.phi_CMY[...] = np.angle(psi_R), np.angle(psi_L)
            traj.append(
                com_radius((session.N_RGB + session.N_CMY).astype(np.float64) / N_UNITS)
            )
        return traj

    # --- core: same well via topological_potential, swept over omega -------
    from dcl_core.core import CausalSession, OctahedralLattice
    from dcl_core.core.UnityConstraint import enforce_unity_spinor

    def run_core(omega: float) -> list[float]:
        lattice = OctahedralLattice(GRID, GRID, GRID)
        lattice.topological_potential[...] = V
        session = CausalSession(
            lattice=lattice, initial_node=start, instruction_frequency=omega
        )
        envelope = np.exp(-0.5 * ((xx - start[0]) ** 2 + (yy - start[1]) ** 2
                                  + (zz - start[2]) ** 2) / WIDTH**2)
        packet = (envelope * np.exp(1j * K_TANG * (xx - yy - zz))).astype(complex)
        session.psi_R = packet / np.sqrt(2.0)
        session.psi_L = packet / np.sqrt(2.0)
        enforce_unity_spinor(session.psi_R, session.psi_L)
        traj: list[float] = []
        for _ in range(TICKS):
            session.tick()
            session.advance_tick_counter()
            traj.append(com_radius(session.probability_density()))
        return traj

    # --- sweep both engines and characterise the lock-in ------------------
    results: dict[str, tuple[list[float], list[float]]] = {}
    for name, runner in (("core", run_core), ("core3d", run_core3d)):
        swings, means = [], []
        for omega in OMEGAS:
            swing, mean = swing_and_mean(runner(omega))
            swings.append(swing)
            means.append(mean)
        results[name] = (swings, means)

        # (1) Bound throughout: the orbit stays interior at every omega.
        for omega, mean in zip(OMEGAS, means):
            assert 2.0 < mean < 0.85 * r_max, (
                f"{name} unbound at omega={omega}: mean radius {mean:.2f} "
                f"outside (2, {0.85 * r_max:.1f})"
            )
        # (2) The orbit always moves (never frozen): nonzero swing.
        assert min(swings) > 0.3, f"{name} orbit frozen: min swing {min(swings):.2f}"
        # (3) Frequency-dependent lock-in: omega materially tunes the orbit.
        assert max(swings) / min(swings) > 2.0, (
            f"{name} orbital swing is flat across omega "
            f"(factor {max(swings) / min(swings):.2f}); no lock-in structure"
        )

    # (4) The lock-in sits at a different omega in each engine (fixed-lattice
    # protocol mismatch).  argmin(swing) is the tightest-lock omega.  Tripwire:
    # if they ever coincide at fixed spacing that is a real convergence result
    # -- revisit this test rather than loosening it.
    tightest_core = int(np.argmin(results["core"][0]))
    tightest_c3 = int(np.argmin(results["core3d"][0]))
    assert tightest_core != tightest_c3, (
        f"engines lock tightest at the same omega index "
        f"({tightest_core}); convergence at fixed spacing contradicts the "
        f"double-limit finding -- review needed"
    )
