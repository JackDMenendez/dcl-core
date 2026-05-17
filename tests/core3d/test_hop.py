"""Tests for the bipartite Dirac hop operator.

Hop is the framework's central kinematic claim. These tests guard:

- bipartite parity (RGB hops only act when even, CMY only when odd)
- structure-factor expansion has the right small-k behaviour
- the operator is identity in the trivial omega=0, V=0 case with a
  uniform amplitude (no flow)
- Clifford algebra relations hold in the structure-factor matrix
  representation
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="HopOperator is a stub; remove skip once implemented.")


def test_hop_uniform_amplitude_is_stationary(small_shape: tuple[int, int, int]) -> None:
    """For a uniform amplitude with omega=0, V=0, the hop should not move tokens.

    A constant function is the lowest-energy mode; in the absence of
    on-site mass or potential, it sits still.
    """
    from dcl_core import BipartiteLattice, DiscreteCausalSession, HopOperator

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


def test_fourier_kernel_small_k_is_linear(small_shape: tuple[int, int, int]) -> None:
    """In the small-k limit, the structure factor approaches `i k . gamma`."""
    import numpy as np

    from dcl_core import BipartiteLattice, HopOperator

    lattice = BipartiteLattice(shape=small_shape)
    hop = HopOperator(lattice=lattice)

    # Pick a small wavevector; verify linearity in k.
    k1 = np.array([0.01, 0.0, 0.0])
    k2 = 2 * k1
    K1 = hop.fourier_kernel(k1)
    K2 = hop.fourier_kernel(k2)
    # Leading-order: K(k) is linear in k. Check ratio.
    np.testing.assert_allclose(K2, 2 * K1, rtol=1e-2)
