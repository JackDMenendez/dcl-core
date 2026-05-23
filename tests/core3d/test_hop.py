"""Tests for the bipartite Dirac hop operator.

Hop is the framework's central kinematic claim. These tests guard:

- bipartite parity (RGB hops only act when even, CMY only when odd)
- structure-factor expansion has the right small-k behaviour
- the operator is identity in the trivial omega=0, V=0 case with a
  uniform amplitude (no flow)
- the session is not mutated by `step` (the residual + scheduler do
  the writeback, this operator is pure)

The Clifford algebra anti-commutator and the continuum-limit dispersion
live in `test_clifford.py` / `test_continuum_limit.py`; both are still
stubbed pending exposed gamma matrices and an FFT-based extractor.
"""

from __future__ import annotations

import pytest


def test_hop_uniform_amplitude_is_stationary(small_shape: tuple[int, int, int]) -> None:
    """For a uniform amplitude with omega=0, V=0, the hop should not move tokens.

    A constant function is the lowest-energy mode; in the absence of
    on-site mass or potential, it sits still.
    """
    from dcl_core.core3d import BipartiteLattice, DiscreteCausalSession, HopOperator

    lattice = BipartiteLattice(shape=small_shape)
    session = DiscreteCausalSession(lattice=lattice, n_units=8 * 8 * 8, omega=0.0)
    # Initialise: one token per site, zero phase.
    session.N_R[...] = 1
    session.N_L[...] = 0
    session.phi_R[...] = 0.0
    session.phi_L[...] = 0.0

    hop = HopOperator(lattice=lattice)
    psi_R_new, psi_L_new = hop.step(session, parity="even", external_potential=None)

    # Uniform amplitude is an eigenstate of a free hop with omega=0.
    # Token redistribution should be zero or trivial.
    import numpy as np

    np.testing.assert_allclose(np.abs(psi_R_new), np.abs(psi_R_new.flat[0]))


def test_massless_even_tick_replaces_R_with_hopped_L(
    small_shape: tuple[int, int, int],
) -> None:
    """omega = 0 -> psi_R_new = hop_RGB(psi_L); psi_L unchanged.

    Sharper than the "is stationary" smoke test: both R and L are
    uniformly populated, so the mass term contribution (sin(0)*psi_R)
    is zero and the only thing left is the hop of L into R.
    """
    import numpy as np

    from dcl_core.core3d import BipartiteLattice, DiscreteCausalSession, HopOperator

    lattice = BipartiteLattice(shape=small_shape)
    n_units = 2 * 8 * 8 * 8  # split 50/50 across chiralities
    session = DiscreteCausalSession(lattice=lattice, n_units=n_units, omega=0.0)
    session.N_R[...] = 1
    session.N_L[...] = 1
    session.phi_R[...] = 0.0
    session.phi_L[...] = 0.0

    hop = HopOperator(lattice=lattice)
    psi_R_new, psi_L_new = hop.step(session, parity="even")

    psi_L = session.amplitude("L")
    # With uniform psi_L, hop_RGB(psi_L) IS psi_L (mean of constants).
    np.testing.assert_allclose(psi_R_new, psi_L)
    # psi_L is passive on even tick.
    np.testing.assert_array_equal(psi_L_new, psi_L)


def test_odd_tick_leaves_R_amplitude_unchanged(
    small_shape: tuple[int, int, int],
) -> None:
    """On odd (CMY) tick, ``psi_R_new`` is the bitwise-equal psi_R snapshot.

    Mirror of the L-passive-on-even invariant: the docstring promises
    ``psi_R_new = psi_R`` on odd ticks, full stop.
    """
    import numpy as np

    from dcl_core.core3d import BipartiteLattice, DiscreteCausalSession, HopOperator

    lattice = BipartiteLattice(shape=small_shape)
    session = DiscreteCausalSession.delta_at(
        lattice, n_units=1_000, omega=0.5, position=(4, 4, 4)
    )

    hop = HopOperator(lattice=lattice)
    psi_R_new, _ = hop.step(session, parity="odd")
    np.testing.assert_array_equal(psi_R_new, session.amplitude("R"))


def test_step_does_not_mutate_session(small_shape: tuple[int, int, int]) -> None:
    """``step`` is pure: the session's state arrays are untouched by the call."""
    import numpy as np

    from dcl_core.core3d import BipartiteLattice, DiscreteCausalSession, HopOperator

    lattice = BipartiteLattice(shape=small_shape)
    session = DiscreteCausalSession.delta_at(
        lattice, n_units=1_000, omega=0.5, position=(4, 4, 4)
    )

    N_R_before = session.N_R.copy()
    N_L_before = session.N_L.copy()
    phi_R_before = session.phi_R.copy()
    phi_L_before = session.phi_L.copy()

    hop = HopOperator(lattice=lattice)
    _ = hop.step(session, parity="even")

    np.testing.assert_array_equal(session.N_R, N_R_before)
    np.testing.assert_array_equal(session.N_L, N_L_before)
    np.testing.assert_array_equal(session.phi_R, phi_R_before)
    np.testing.assert_array_equal(session.phi_L, phi_L_before)


def test_step_rejects_wrong_external_potential_shape(
    small_shape: tuple[int, int, int],
) -> None:
    """An ``external_potential`` of wrong shape raises ValueError."""
    import numpy as np

    from dcl_core.core3d import BipartiteLattice, DiscreteCausalSession, HopOperator

    lattice = BipartiteLattice(shape=small_shape)
    session = DiscreteCausalSession.delta_at(
        lattice, n_units=100, omega=0.1, position=(0, 0, 0)
    )

    hop = HopOperator(lattice=lattice)
    bad_potential = np.zeros((small_shape[0] + 1,) + small_shape[1:])
    with pytest.raises(ValueError):
        hop.step(session, parity="even", external_potential=bad_potential)


def test_fourier_kernel_small_k_is_linear(small_shape: tuple[int, int, int]) -> None:
    """In the small-k limit, the structure factor approaches `i k . gamma`."""
    import numpy as np

    from dcl_core.core3d import BipartiteLattice, HopOperator

    lattice = BipartiteLattice(shape=small_shape)
    hop = HopOperator(lattice=lattice)

    # Pick a small wavevector; verify linearity in k.
    k1 = np.array([0.01, 0.0, 0.0])
    k2 = 2 * k1
    K1 = hop.fourier_kernel(k1)
    K2 = hop.fourier_kernel(k2)
    # Leading-order: K(k) is linear in k. Check ratio.
    np.testing.assert_allclose(K2, 2 * K1, rtol=1e-2)


def test_fourier_kernel_vanishes_at_zero(small_shape: tuple[int, int, int]) -> None:
    """``fourier_kernel(0) == 0``: the kinetic operator has no constant piece.

    Equivalent statement: the bipartite-antisymmetric structure factor
    integrates to zero by construction, so the continuum limit IS
    purely the linear ``i k . gamma`` piece with no static offset.
    """
    import numpy as np

    from dcl_core.core3d import BipartiteLattice, HopOperator

    lattice = BipartiteLattice(shape=small_shape)
    hop = HopOperator(lattice=lattice)
    K0 = hop.fourier_kernel(np.zeros(3))
    np.testing.assert_allclose(K0, 0.0, atol=1e-15)


def test_fourier_kernel_is_antisymmetric(small_shape: tuple[int, int, int]) -> None:
    """``fourier_kernel(-k) == -fourier_kernel(k)``.

    Follows from ``CMY = -RGB``: swapping ``k -> -k`` swaps the role
    of RGB and CMY contributions in the bipartite difference, which
    negates the kernel.
    """
    import numpy as np

    from dcl_core.core3d import BipartiteLattice, HopOperator

    lattice = BipartiteLattice(shape=small_shape)
    hop = HopOperator(lattice=lattice)
    k = np.array([0.3, -0.2, 0.15])
    np.testing.assert_allclose(hop.fourier_kernel(-k), -hop.fourier_kernel(k))


def test_fourier_kernel_rejects_wrong_k_shape() -> None:
    """``fourier_kernel`` requires a 3-vector; other shapes raise."""
    import numpy as np

    from dcl_core.core3d import BipartiteLattice, HopOperator

    lattice = BipartiteLattice(shape=(8, 8, 8))
    hop = HopOperator(lattice=lattice)
    with pytest.raises(ValueError):
        hop.fourier_kernel(np.array([0.0, 0.0]))
    with pytest.raises(ValueError):
        hop.fourier_kernel(np.array([[0.0, 0.0, 0.0]]))
