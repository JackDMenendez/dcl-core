"""Acceptance tests for the Peierls gauge coupling in the core3d hop.

Phase 1 of the v0.3.0 gauge-field work (requirements:
`docs/design/04_gauge_field_and_vacuum_response.md`; plan:
`notes/gauge_field_v030_plan.md`). This file covers the three Phase-1
acceptance criteria; the magnetic `Q`-tensor cross-check (#4), the
B->0 quadratic limit (#3), and the E+B birefringence-order verdict (#5)
arrive with Phases 2-3.

Covered here:

1. **Zero-field regression** -- `vector_potential=None` and an all-zeros
   field reproduce the gauge-free hop bit-for-bit (and a nonzero field
   actually changes the output, so the coupling is live).
2. **Gauge covariance / U(1) Ward identity** -- a pure-gauge potential
   leaves token densities `|psi|^2` invariant tick-for-tick.
6. **A=1 exactness under the field** -- total token count is conserved
   exactly every tick with the Peierls phase present.

The pure gauge used for #2 is a **commensurate linear** gauge
`Lambda(x) = k . x` with `k_i = 2*pi*n_i / L_i`: its `exp(i*Lambda)` is
single-valued on the periodic torus (no boundary artifact) and its
constant `A = grad(Lambda)` makes the mid-point Peierls rule *exact*, so
the Ward identity holds to float roundoff rather than only to `O(a^2)`.
"""

from __future__ import annotations

import numpy as np
import pytest

from dcl_core.core3d import (
    BipartiteLattice,
    DiscreteCausalSession,
    HopOperator,
    TokenResidual,
)


def _wavepacket(lattice: BipartiteLattice, omega: float = 0.3) -> DiscreteCausalSession:
    """A generic, non-uniform, A=1 session for gauge tests."""
    c = lattice.shape[0] // 2
    return DiscreteCausalSession.wavepacket(
        lattice, n_units=100_000, omega=omega, center=(c, c, c), sigma=1.5
    )


# ---------------------------------------------------------------------------
# #1 -- zero-field regression
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("parity", ["even", "odd"])
def test_zero_field_none_and_zeros_match_gauge_free(
    small_shape: tuple[int, int, int], parity: str
) -> None:
    """``vector_potential=None`` and an all-zeros field are bit-for-bit equal."""
    lattice = BipartiteLattice(shape=small_shape)
    session = _wavepacket(lattice)
    hop = HopOperator(lattice=lattice)

    r_none, l_none = hop.step(session, parity=parity, vector_potential=None)
    zeros = np.zeros((3,) + small_shape)
    r_zero, l_zero = hop.step(session, parity=parity, vector_potential=zeros)

    np.testing.assert_array_equal(r_none, r_zero)
    np.testing.assert_array_equal(l_none, l_zero)


@pytest.mark.parametrize("parity", ["even", "odd"])
def test_nonzero_field_changes_the_hop(
    small_shape: tuple[int, int, int], parity: str
) -> None:
    """A nonzero gauge field actually perturbs the hop (coupling is live).

    Guards against the regression test passing trivially because the
    argument were silently ignored.
    """
    lattice = BipartiteLattice(shape=small_shape)
    session = _wavepacket(lattice)
    hop = HopOperator(lattice=lattice)

    r_free, l_free = hop.step(session, parity=parity, vector_potential=None)

    coords = np.indices(small_shape).astype(np.float64)
    a_field = np.zeros((3,) + small_shape)
    a_field[1] = 0.3 * coords[0]  # A_y = 0.3 x -> uniform B_z, genuinely non-trivial
    r_a, l_a = hop.step(session, parity=parity, vector_potential=a_field)

    # The active component must change; the passive one is untouched anyway.
    active_free, active_a = (r_free, r_a) if parity == "even" else (l_free, l_a)
    assert not np.allclose(active_free, active_a)


def test_step_rejects_wrong_vector_potential_shape(
    small_shape: tuple[int, int, int],
) -> None:
    """A ``vector_potential`` of wrong shape raises ValueError."""
    lattice = BipartiteLattice(shape=small_shape)
    session = _wavepacket(lattice)
    hop = HopOperator(lattice=lattice)

    bad = np.zeros((2,) + small_shape)  # leading axis must be 3
    with pytest.raises(ValueError):
        hop.step(session, parity="even", vector_potential=bad)


# ---------------------------------------------------------------------------
# #2 -- gauge covariance / U(1) Ward identity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("parity", ["even", "odd"])
def test_pure_gauge_leaves_densities_invariant(
    small_shape: tuple[int, int, int], parity: str
) -> None:
    """U(1) Ward identity: a pure-gauge ``A = grad(Lambda)`` does not move tokens.

    Construct a commensurate linear gauge ``Lambda = k . x`` (so
    ``exp(i*Lambda)`` is periodic and the constant ``A`` makes the
    mid-point rule exact).  Evolving the gauge-transformed state under
    that ``A`` must give the **same** ``|psi|^2`` as evolving the
    original state with no field.
    """
    lattice = BipartiteLattice(shape=small_shape)
    nx, ny, nz = small_shape
    hop = HopOperator(lattice=lattice)

    # Commensurate wavevector: one full period along each axis.
    k = np.array([2.0 * np.pi / nx, 2.0 * np.pi / ny, 2.0 * np.pi / nz])

    session = _wavepacket(lattice)
    r0, l0 = hop.step(session, parity=parity, vector_potential=None)

    # Gauge-transform the state: phi -> phi + Lambda, Lambda(x) = k . x.
    coords = np.indices(small_shape).astype(np.float64)
    lam = k[0] * coords[0] + k[1] * coords[1] + k[2] * coords[2]
    gauged = DiscreteCausalSession.from_arrays(
        lattice,
        n_units=session.n_units,
        omega=session.omega,
        N_RGB=session.N_RGB.copy(),
        N_CMY=session.N_CMY.copy(),
        phi_RGB=session.phi_RGB + lam,
        phi_CMY=session.phi_CMY + lam,
    )
    # Constant A = grad(Lambda) = k everywhere.
    a_field = np.empty((3,) + small_shape)
    a_field[0], a_field[1], a_field[2] = k[0], k[1], k[2]

    rg, lg = hop.step(gauged, parity=parity, vector_potential=a_field)

    # Densities are gauge-invariant; only the phase rotates by exp(i*Lambda).
    np.testing.assert_allclose(np.abs(rg), np.abs(r0), rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(np.abs(lg), np.abs(l0), rtol=1e-9, atol=1e-12)


# ---------------------------------------------------------------------------
# #6 -- A=1 exactness under the gauge field
# ---------------------------------------------------------------------------

def test_a1_exact_under_gauge_field(small_shape: tuple[int, int, int]) -> None:
    """Total token count is conserved exactly each tick with Peierls present.

    Drives the same hop -> renorm -> quantise -> write-back cycle the
    scheduler uses, with a non-trivial gauge field on, and asserts the
    integer A=1 identity holds every tick.
    """
    lattice = BipartiteLattice(shape=small_shape)
    session = _wavepacket(lattice)
    hop = HopOperator(lattice=lattice)
    residual = TokenResidual(lattice=lattice)
    n_units = session.n_units

    coords = np.indices(small_shape).astype(np.float64)
    a_field = np.zeros((3,) + small_shape)
    a_field[1] = 0.05 * coords[0]  # A_y = 0.05 x -> uniform B_z

    for tick in range(6):
        parity = "even" if tick % 2 == 0 else "odd"
        psi_r, psi_l = hop.step(session, parity=parity, vector_potential=a_field)

        norm_sq = float(np.sum(np.abs(psi_r) ** 2) + np.sum(np.abs(psi_l) ** 2))
        if norm_sq > 0.0:
            inv = 1.0 / np.sqrt(norm_sq)
            psi_r, psi_l = psi_r * inv, psi_l * inv

        n_rgb = np.abs(psi_r) ** 2 * n_units
        n_cmy = np.abs(psi_l) ** 2 * n_units
        new_rgb, new_cmy = residual.quantise(n_rgb, n_cmy, n_units)
        session.N_RGB[...] = new_rgb
        session.N_CMY[...] = new_cmy
        session.phi_RGB[...] = np.angle(psi_r)
        session.phi_CMY[...] = np.angle(psi_l)

        assert session.total_tokens() == n_units, f"A=1 broke at tick {tick}"
