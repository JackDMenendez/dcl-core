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


pytestmark = pytest.mark.slow


@pytest.mark.parametrize("n_units", [10**3, 10**4, 10**5, 10**6])
def test_token_distribution_approaches_continuous_amplitude(n_units: int) -> None:
    """As N grows, the discrete N(x)/N_units profile converges to |psi|^2.

    Build the same analytical Gaussian via :meth:`wavepacket` at each
    ``n_units`` and compare the integer-token density to the closed-
    form Gaussian.  Per-site error is O(1/N) from the
    Bresenham quantisation; summed over ``n_sites``, the L2 error
    scales as ``sqrt(n_sites) / n_units``.  The threshold below uses
    10x slack on that bound to absorb the rebalance step's
    deficit-deposit asymmetry.
    """
    import numpy as np

    from dcl_core.core3d import BipartiteLattice, DiscreteCausalSession

    shape = (16, 16, 16)
    n_sites = shape[0] * shape[1] * shape[2]
    centre = (shape[0] // 2, shape[1] // 2, shape[2] // 2)
    sigma = 2.0

    lattice = BipartiteLattice(shape=shape)
    session = DiscreteCausalSession.wavepacket(
        lattice,
        n_units=n_units,
        omega=0.1,
        center=centre,
        sigma=sigma,
    )

    # Closed-form normalised Gaussian profile (what the factory was
    # asked to approximate).
    coords = np.indices(shape, dtype=np.float64)
    r2 = (
        (coords[0] - centre[0]) ** 2
        + (coords[1] - centre[1]) ** 2
        + (coords[2] - centre[2]) ** 2
    )
    g = np.exp(-r2 / (2.0 * sigma**2))
    g /= g.sum()

    # Integer-token density.  Total density is the marginal
    # `(N_R + N_L) / n_units`; this IS what |psi|^2 should be.
    density = (session.N_R + session.N_L).astype(np.float64) / n_units

    l2_error = float(np.sqrt(np.sum((density - g) ** 2)))
    threshold = 10.0 * np.sqrt(n_sites) / n_units
    assert l2_error <= threshold, (
        f"L2 error {l2_error:.3e} exceeds threshold {threshold:.3e} "
        f"at n_units={n_units}"
    )


@pytest.mark.parametrize("lattice_size", [16, 32, 64])
def test_dirac_dispersion_in_continuum_limit(lattice_size: int) -> None:
    """The dispersion's linear-in-k coefficient is recovered as ``a -> 0``.

    The structure factor's small-k expansion IS the lattice analogue
    of the continuum Dirac operator ``i k . gamma``.  For ``k`` along
    the x-axis on the RGB sublattice, the closed-form is

        K(k_x . x_hat) = i * sin(k_x) / 3

    so ``K / k_x -> i / 3`` as ``k_x -> 0``.  This test evaluates at
    the smallest representable wavevector on a periodic lattice,
    ``k_min = 2 * pi / L``, and verifies the ratio approaches the
    continuum value ``i / 3`` with deviation bounded by the leading
    O(k_x^3) term in the sine expansion.
    """
    import numpy as np

    from dcl_core.core3d import BipartiteLattice, HopOperator

    lattice = BipartiteLattice(shape=(lattice_size, lattice_size, lattice_size))
    hop = HopOperator(lattice=lattice)

    k_min = 2.0 * np.pi / lattice_size
    K_at_kmin = complex(hop.fourier_kernel(np.array([k_min, 0.0, 0.0])))

    # Continuum expectation along x: K -> i * k_x / 3.
    expected = 1j * k_min / 3.0

    # Leading deviation in K is the k^3 term in sin(k_x): error in
    # K(k_min) is k_min^3 / 18; allow 3x slack for floating-point
    # arithmetic and higher-order terms.
    tolerance = (k_min**3 / 18.0) * 3.0
    np.testing.assert_allclose(
        K_at_kmin, expected, atol=tolerance, rtol=0
    )
