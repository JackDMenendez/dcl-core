"""Test A=1 conservation as integer equality (no float tolerance).

The framework's foundational claim is that probability is conserved
exactly, by integer arithmetic. These tests guard that claim: every
tick of every operator must leave `sum(N(x)) == n_units` exactly.

If any test here fails, the regression is structural -- something in
the hop / remainder / scheduler chain has slipped back into floating-
point renormalisation. Fix the underlying code, do not loosen these
tests.

Tests are unlocked incrementally as core modules land:
- Construction (vacuum, delta_at, from_arrays), corruption detection,
  amplitude consistency, and momentum-gradient initialisation run
  today against `DiscreteCausalSession` alone.
- The "invariant across N ticks" tests remain skipped until
  `HopOperator` and `TickScheduler` are concrete (steps 4-5 in
  CLAUDE.md's roadmap).
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Bare-constructor (vacuum) behaviour
# ---------------------------------------------------------------------------


def test_bare_init_is_vacuum(small_shape: tuple[int, int, int]) -> None:
    """The bare constructor allocates zeros: 0 tokens, vacuum state.

    A vacuum session deliberately violates A=1 -- it is a *building
    block* for the factory methods, not a usable physical state.
    Callers must populate via a factory or by direct array writes
    before stepping the engine.
    """
    from dcl_core.core3d import BipartiteLattice, DiscreteCausalSession

    lattice = BipartiteLattice(shape=small_shape)
    session = DiscreteCausalSession(lattice=lattice, n_units=100, omega=0.1)
    assert session.total_tokens() == 0
    with pytest.raises(AssertionError):
        session.assert_unity()


def test_init_rejects_non_positive_n_units(small_shape: tuple[int, int, int]) -> None:
    """`n_units` must be a positive integer; 0 and negative values raise."""
    from dcl_core.core3d import BipartiteLattice, DiscreteCausalSession

    lattice = BipartiteLattice(shape=small_shape)
    with pytest.raises(ValueError):
        DiscreteCausalSession(lattice=lattice, n_units=0, omega=0.1)
    with pytest.raises(ValueError):
        DiscreteCausalSession(lattice=lattice, n_units=-5, omega=0.1)


# ---------------------------------------------------------------------------
# delta_at factory
# ---------------------------------------------------------------------------


def test_delta_at_places_all_tokens_at_position(
    small_shape: tuple[int, int, int],
) -> None:
    """`delta_at` puts all tokens at the named site, 50/50 R/L (extra to R).

    Pin down the contract so future changes are deliberate.  Uses an
    odd ``n_units`` to exercise the R-gets-the-extra tiebreaker, and
    a non-origin position to verify the site is not hard-coded.
    """
    from dcl_core.core3d import BipartiteLattice, DiscreteCausalSession

    lattice = BipartiteLattice(shape=small_shape)
    position = (3, 5, 7)
    session = DiscreteCausalSession.delta_at(
        lattice, n_units=101, omega=0.1, position=position
    )

    # Origin site empty (and every site that isn't `position`).
    n_R_off = int(session.N_RGB.sum()) - int(session.N_RGB[position])
    n_L_off = int(session.N_CMY.sum()) - int(session.N_CMY[position])
    assert n_R_off == 0
    assert n_L_off == 0

    # `position` holds the full budget, split with extra going to R.
    assert int(session.N_RGB[position]) == (101 + 1) // 2  # 51
    assert int(session.N_CMY[position]) == 101 // 2  # 50
    assert session.total_tokens() == 101


def test_delta_at_rejects_out_of_bounds_position(
    small_shape: tuple[int, int, int],
) -> None:
    """`delta_at` validates the position lies inside the lattice."""
    from dcl_core.core3d import BipartiteLattice, DiscreteCausalSession

    lattice = BipartiteLattice(shape=small_shape)
    nx, _, _ = small_shape

    with pytest.raises(ValueError):
        DiscreteCausalSession.delta_at(
            lattice, n_units=100, omega=0.1, position=(nx, 0, 0)
        )
    with pytest.raises(ValueError):
        DiscreteCausalSession.delta_at(
            lattice, n_units=100, omega=0.1, position=(-1, 0, 0)
        )


# ---------------------------------------------------------------------------
# from_arrays factory
# ---------------------------------------------------------------------------


def test_from_arrays_round_trips_population(
    small_shape: tuple[int, int, int],
) -> None:
    """`from_arrays` writes the caller's exact arrays into the session."""
    import numpy as np

    from dcl_core.core3d import BipartiteLattice, DiscreteCausalSession

    lattice = BipartiteLattice(shape=small_shape)
    N_RGB = np.zeros(small_shape, dtype=np.int64)
    N_CMY = np.zeros(small_shape, dtype=np.int64)
    N_RGB[1, 2, 3] = 40
    N_CMY[4, 5, 6] = 60
    session = DiscreteCausalSession.from_arrays(
        lattice, n_units=100, omega=0.1, N_RGB=N_RGB, N_CMY=N_CMY
    )
    np.testing.assert_array_equal(session.N_RGB, N_RGB)
    np.testing.assert_array_equal(session.N_CMY, N_CMY)
    assert session.total_tokens() == 100


def test_from_arrays_validates_total(small_shape: tuple[int, int, int]) -> None:
    """`from_arrays` raises if the array sum doesn't equal `n_units`."""
    import numpy as np

    from dcl_core.core3d import BipartiteLattice, DiscreteCausalSession

    lattice = BipartiteLattice(shape=small_shape)
    N_RGB = np.zeros(small_shape, dtype=np.int64)
    N_CMY = np.zeros(small_shape, dtype=np.int64)
    N_RGB[0, 0, 0] = 50
    N_CMY[0, 0, 0] = 60  # total = 110, but n_units = 100 below
    with pytest.raises(ValueError):
        DiscreteCausalSession.from_arrays(
            lattice, n_units=100, omega=0.1, N_RGB=N_RGB, N_CMY=N_CMY
        )


def test_from_arrays_validates_shape(small_shape: tuple[int, int, int]) -> None:
    """`from_arrays` raises if N_RGB / N_CMY shapes don't match the lattice.

    The shape check fires before the sum-equals-n_units check, so the
    test uses ``n_units=1`` (any value works -- we never reach the
    sum check).
    """
    import numpy as np

    from dcl_core.core3d import BipartiteLattice, DiscreteCausalSession

    lattice = BipartiteLattice(shape=small_shape)
    wrong_shape = tuple(s + 1 for s in small_shape)
    N_RGB_wrong = np.zeros(wrong_shape, dtype=np.int64)
    N_CMY_ok = np.zeros(small_shape, dtype=np.int64)
    with pytest.raises(ValueError):
        DiscreteCausalSession.from_arrays(
            lattice, n_units=1, omega=0.1, N_RGB=N_RGB_wrong, N_CMY=N_CMY_ok
        )
    # And symmetrically for N_CMY.
    N_RGB_ok = np.zeros(small_shape, dtype=np.int64)
    N_CMY_wrong = np.zeros(wrong_shape, dtype=np.int64)
    with pytest.raises(ValueError):
        DiscreteCausalSession.from_arrays(
            lattice, n_units=1, omega=0.1, N_RGB=N_RGB_ok, N_CMY=N_CMY_wrong
        )


def test_from_arrays_accepts_custom_phases(
    small_shape: tuple[int, int, int],
) -> None:
    """`from_arrays` accepts user-supplied phi_RGB / phi_CMY verbatim."""
    import numpy as np

    from dcl_core.core3d import BipartiteLattice, DiscreteCausalSession

    lattice = BipartiteLattice(shape=small_shape)
    N_RGB = np.zeros(small_shape, dtype=np.int64)
    N_CMY = np.zeros(small_shape, dtype=np.int64)
    N_RGB[0, 0, 0] = 100
    phi_RGB = np.full(small_shape, 0.5, dtype=np.float64)
    phi_CMY = np.full(small_shape, -0.25, dtype=np.float64)
    session = DiscreteCausalSession.from_arrays(
        lattice,
        n_units=100,
        omega=0.1,
        N_RGB=N_RGB,
        N_CMY=N_CMY,
        phi_RGB=phi_RGB,
        phi_CMY=phi_CMY,
    )
    np.testing.assert_array_equal(session.phi_RGB, phi_RGB)
    np.testing.assert_array_equal(session.phi_CMY, phi_CMY)


# ---------------------------------------------------------------------------
# wavepacket factory (min-Δp Gaussian)
# ---------------------------------------------------------------------------


def test_wavepacket_total_equals_n_units(small_shape: tuple[int, int, int]) -> None:
    """The Gaussian wavepacket factory still satisfies A=1 exactly.

    The integer quantisation reuses ``TokenResidual``; the
    `from_arrays`-style sum check is not applicable since the
    factory constructs via the bare ctor, so this test stands in
    as the A=1 contract at the wavepacket entry point.
    """
    from dcl_core.core3d import BipartiteLattice, DiscreteCausalSession

    lattice = BipartiteLattice(shape=small_shape)
    n_units = 10_000
    session = DiscreteCausalSession.wavepacket(
        lattice,
        n_units=n_units,
        omega=0.1,
        center=(4, 4, 4),
        sigma=1.5,
    )
    assert session.total_tokens() == n_units


def test_wavepacket_localised_near_center(small_shape: tuple[int, int, int]) -> None:
    """Mass concentrates around `center`; far-corner sites get zero tokens.

    For ``sigma = 1.0`` on a (8,8,8) lattice the far corner sits at
    distance √48 ≈ 6.93 σ -- the Gaussian value there is ``exp(-24)``,
    which quantises to 0 for any reasonable ``n_units``.  The peak
    should sit at the centre site and dominate the distribution.
    """
    from dcl_core.core3d import BipartiteLattice, DiscreteCausalSession

    lattice = BipartiteLattice(shape=small_shape)
    center = (4, 4, 4)
    session = DiscreteCausalSession.wavepacket(
        lattice,
        n_units=10_000,
        omega=0.1,
        center=center,
        sigma=1.0,
    )
    N_total = session.N_RGB + session.N_CMY
    # Centre site IS the peak.
    assert int(N_total[center]) == int(N_total.max())
    # Far corner sites are empty for this sigma / n_units.
    assert int(N_total[0, 0, 0]) == 0
    assert int(N_total[7, 7, 7]) == 0


def test_wavepacket_momentum_imprints_phase_gradient(
    small_shape: tuple[int, int, int],
) -> None:
    """`momentum` passes through to the underlying `phi(x) = k.x` gradient."""
    import numpy as np

    from dcl_core.core3d import BipartiteLattice, DiscreteCausalSession

    lattice = BipartiteLattice(shape=small_shape)
    k = (0.2, -0.1, 0.05)
    session = DiscreteCausalSession.wavepacket(
        lattice,
        n_units=10_000,
        omega=0.1,
        center=(4, 4, 4),
        sigma=2.0,
        momentum=k,
    )
    coords = np.indices(small_shape, dtype=np.float64)
    expected = k[0] * coords[0] + k[1] * coords[1] + k[2] * coords[2]
    np.testing.assert_allclose(session.phi_RGB, expected)
    np.testing.assert_allclose(session.phi_CMY, expected)


def test_wavepacket_accepts_anisotropic_sigma(
    small_shape: tuple[int, int, int],
) -> None:
    """A 3-tuple `sigma` gives a per-axis width; total still equals n_units."""
    from dcl_core.core3d import BipartiteLattice, DiscreteCausalSession

    lattice = BipartiteLattice(shape=small_shape)
    session = DiscreteCausalSession.wavepacket(
        lattice,
        n_units=10_000,
        omega=0.1,
        center=(4, 4, 4),
        sigma=(1.0, 2.0, 3.0),
    )
    assert session.total_tokens() == 10_000


def test_wavepacket_rejects_bad_sigma(small_shape: tuple[int, int, int]) -> None:
    """Negative / zero / wrong-shape sigma values raise ValueError."""
    from dcl_core.core3d import BipartiteLattice, DiscreteCausalSession

    lattice = BipartiteLattice(shape=small_shape)
    with pytest.raises(ValueError):
        DiscreteCausalSession.wavepacket(
            lattice, n_units=100, omega=0.1, center=(0, 0, 0), sigma=-1.0
        )
    with pytest.raises(ValueError):
        DiscreteCausalSession.wavepacket(
            lattice, n_units=100, omega=0.1, center=(0, 0, 0), sigma=0.0
        )
    with pytest.raises(ValueError):
        DiscreteCausalSession.wavepacket(
            lattice,
            n_units=100,
            omega=0.1,
            center=(0, 0, 0),
            sigma=(1.0, -1.0, 1.0),
        )
    with pytest.raises(ValueError):
        DiscreteCausalSession.wavepacket(
            lattice, n_units=100, omega=0.1, center=(0, 0, 0), sigma=(1.0, 1.0)
        )


def test_wavepacket_rejects_bad_center(small_shape: tuple[int, int, int]) -> None:
    """Out-of-bounds `center` raises ValueError (validated by `_validate_position`)."""
    from dcl_core.core3d import BipartiteLattice, DiscreteCausalSession

    lattice = BipartiteLattice(shape=small_shape)
    nx, _, _ = small_shape
    with pytest.raises(ValueError):
        DiscreteCausalSession.wavepacket(
            lattice, n_units=100, omega=0.1, center=(nx, 0, 0), sigma=1.0
        )


# ---------------------------------------------------------------------------
# Momentum gradient
# ---------------------------------------------------------------------------


def test_momentum_applies_linear_phase_gradient(
    small_shape: tuple[int, int, int],
) -> None:
    """A nonzero `momentum` imprints `phi(x) = k . x` on both chiralities.

    The plane-wave gradient lives on the phase fields whether or not
    a site has tokens; sites with N(x)=0 just contribute nothing to
    the amplitude.  Tested via the bare constructor so the gradient
    is in isolation, with no population.
    """
    import numpy as np

    from dcl_core.core3d import BipartiteLattice, DiscreteCausalSession

    lattice = BipartiteLattice(shape=small_shape)
    k = (0.1, -0.2, 0.05)
    session = DiscreteCausalSession(
        lattice=lattice, n_units=100, omega=0.1, momentum=k
    )

    coords = np.indices(small_shape, dtype=np.float64)
    expected = k[0] * coords[0] + k[1] * coords[1] + k[2] * coords[2]
    np.testing.assert_allclose(session.phi_RGB, expected)
    np.testing.assert_allclose(session.phi_CMY, expected)


def test_zero_momentum_leaves_phases_at_zero(
    small_shape: tuple[int, int, int],
) -> None:
    """Default ``momentum = (0, 0, 0)`` does NOT touch ``phi_RGB`` / ``phi_CMY``."""
    import numpy as np

    from dcl_core.core3d import BipartiteLattice, DiscreteCausalSession

    lattice = BipartiteLattice(shape=small_shape)
    session = DiscreteCausalSession(lattice=lattice, n_units=100, omega=0.1)
    np.testing.assert_array_equal(session.phi_RGB, np.zeros(small_shape))
    np.testing.assert_array_equal(session.phi_CMY, np.zeros(small_shape))


# ---------------------------------------------------------------------------
# Corruption detection
# ---------------------------------------------------------------------------


def test_assert_unity_raises_on_corruption(small_shape: tuple[int, int, int]) -> None:
    """`session.assert_unity()` raises if state is hand-corrupted.

    Sanity check on the sanity check.  Uses ``delta_at(..., (0,0,0))``
    so that ``N_RGB.flat[0]`` carries tokens and the decrement actually
    reduces the total.
    """
    from dcl_core.core3d import BipartiteLattice, DiscreteCausalSession

    lattice = BipartiteLattice(shape=small_shape)
    session = DiscreteCausalSession.delta_at(
        lattice, n_units=100, omega=0.1, position=(0, 0, 0)
    )
    session.N_RGB.flat[0] = max(0, int(session.N_RGB.flat[0]) - 1)
    with pytest.raises(AssertionError):
        session.assert_unity()


# ---------------------------------------------------------------------------
# Amplitude consistency
# ---------------------------------------------------------------------------


def test_amplitude_is_consistent_with_token_field(
    small_shape: tuple[int, int, int],
) -> None:
    """`round(|amplitude("R")|^2 * n_units) == N_RGB` bitwise; same for L.

    The complex amplitude is the implicit derivation
    `psi = sqrt(N/n_units) * exp(i*phi)`; squaring + rescaling must
    invert it (modulo float64 rounding handled by `.round()`).  Uses
    `delta_at` so there is something to recover.
    """
    import numpy as np

    from dcl_core.core3d import BipartiteLattice, DiscreteCausalSession

    lattice = BipartiteLattice(shape=small_shape)
    session = DiscreteCausalSession.delta_at(
        lattice, n_units=1_000_000, omega=0.1, position=(0, 0, 0)
    )

    psi_R = session.amplitude("R")
    psi_L = session.amplitude("L")
    assert psi_R.dtype.kind == "c"
    assert psi_L.dtype.kind == "c"

    recovered_R = (np.abs(psi_R) ** 2 * session.n_units).round().astype(np.int64)
    recovered_L = (np.abs(psi_L) ** 2 * session.n_units).round().astype(np.int64)
    np.testing.assert_array_equal(recovered_R, session.N_RGB)
    np.testing.assert_array_equal(recovered_L, session.N_CMY)


def test_amplitude_rejects_unknown_chirality(small_shape: tuple[int, int, int]) -> None:
    """`amplitude("X")` (anything but "R" or "L") raises ValueError."""
    from dcl_core.core3d import BipartiteLattice, DiscreteCausalSession

    lattice = BipartiteLattice(shape=small_shape)
    session = DiscreteCausalSession(lattice=lattice, n_units=100, omega=0.1)
    with pytest.raises(ValueError):
        session.amplitude("X")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Multi-tick conservation -- blocked on HopOperator + TickScheduler
# ---------------------------------------------------------------------------


def test_session_total_invariant_under_hop(small_shape: tuple[int, int, int]) -> None:
    """One hop preserves the token total exactly."""
    from dcl_core.core3d import (
        BipartiteLattice,
        DiscreteCausalSession,
        HopOperator,
        TickScheduler,
    )

    lattice = BipartiteLattice(shape=small_shape)
    session = DiscreteCausalSession.delta_at(
        lattice, n_units=1_000_000, omega=0.1, position=(0, 0, 0)
    )
    scheduler = TickScheduler(lattice=lattice, hop=HopOperator(lattice=lattice))
    scheduler.register(session)

    initial = session.total_tokens()
    scheduler.step()
    assert session.total_tokens() == initial, (
        f"A=1 violated after 1 tick: {session.total_tokens()} vs initial {initial}"
    )


@pytest.mark.parametrize("n_ticks", [10, 100, 1000])
def test_session_total_invariant_over_many_ticks(
    small_shape: tuple[int, int, int], n_ticks: int
) -> None:
    """A=1 holds exactly across `n_ticks` consecutive ticks.

    No accumulating drift, no slow integer-loss bug.
    """
    from dcl_core.core3d import (
        BipartiteLattice,
        DiscreteCausalSession,
        HopOperator,
        TickScheduler,
    )

    lattice = BipartiteLattice(shape=small_shape)
    session = DiscreteCausalSession.delta_at(
        lattice, n_units=1_000_000, omega=0.1, position=(0, 0, 0)
    )
    scheduler = TickScheduler(lattice=lattice, hop=HopOperator(lattice=lattice))
    scheduler.register(session)

    initial = session.total_tokens()
    for _ in range(n_ticks):
        scheduler.step()
        assert session.total_tokens() == initial, (
            f"A=1 violated at tick {scheduler.tick}"
        )
