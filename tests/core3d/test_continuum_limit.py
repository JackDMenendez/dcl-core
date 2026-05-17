"""Tests for continuum-limit recovery as `N -> infinity` and `a -> 0`.

The framework's claim is that v1.0-style continuous dynamics is the
`N -> infinity` limit of the integer-token theory, and that standard
Dirac dispersion `E^2 = m^2 + |p|^2` is the `a -> 0` limit of the
lattice. These tests parametrise over N (or a) and check that the
target quantity converges at the expected rate.

Marked `slow` because larger N is required to see clean convergence;
exclude with `make tests-fast` in normal dev.
"""

from __future__ import annotations

import pytest

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skip(reason="Core operators are stubs; remove once implemented."),
]


@pytest.mark.parametrize("n_units", [10**3, 10**4, 10**5, 10**6])
def test_token_distribution_approaches_continuous_amplitude(n_units: int) -> None:
    """As N grows, the discrete N(x)/N_units profile converges to |psi|^2.

    Initialise a session whose continuous amplitude is a known
    Gaussian; check that the L2 error between the integer-token
    profile and the analytical |psi|^2 shrinks as 1/sqrt(N).
    """
    pytest.skip("Implement once the core is concrete.")


@pytest.mark.parametrize("lattice_size", [16, 32, 64])
def test_dirac_dispersion_in_continuum_limit(lattice_size: int) -> None:
    """The dispersion `E^2 = m^2 + |p|^2` is recovered as `a -> 0`."""
    pytest.skip("Implement via HopOperator.fourier_kernel and FFT.")
