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
    TickScheduler,
    TokenResidual,
    uniform_B_potential,
    uniform_E_potential,
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


# ---------------------------------------------------------------------------
# R3 -- uniform_B_potential (symmetric gauge)
# ---------------------------------------------------------------------------

def test_uniform_B_potential_curl_is_B(small_shape: tuple[int, int, int]) -> None:
    """curl(A) == B everywhere for the symmetric-gauge uniform-B potential.

    `A` is linear in position, so np.gradient (central differences) is
    exact -- the discrete curl reproduces `B` to float roundoff.
    """
    B = np.array([0.3, -0.2, 0.15])
    A = uniform_B_potential(small_shape, B)
    assert A.shape == (3, *small_shape)

    curl_x = np.gradient(A[2], axis=1) - np.gradient(A[1], axis=2)
    curl_y = np.gradient(A[0], axis=2) - np.gradient(A[2], axis=0)
    curl_z = np.gradient(A[1], axis=0) - np.gradient(A[0], axis=1)

    np.testing.assert_allclose(curl_x, B[0], atol=1e-12)
    np.testing.assert_allclose(curl_y, B[1], atol=1e-12)
    np.testing.assert_allclose(curl_z, B[2], atol=1e-12)


def test_uniform_B_potential_origin_is_pure_gauge(
    small_shape: tuple[int, int, int],
) -> None:
    """Two origins differ by a constant (uniform) shift of A: a pure gauge."""
    B = np.array([0.1, 0.2, -0.3])
    a_centre = uniform_B_potential(small_shape, B)  # default = centre
    a_corner = uniform_B_potential(small_shape, B, origin=(0, 0, 0))
    diff = a_corner - a_centre
    # The difference is spatially constant (same 3-vector at every site).
    for c in range(3):
        np.testing.assert_allclose(diff[c], diff[c].flat[0], atol=1e-12)


def test_uniform_B_potential_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError):
        uniform_B_potential((8, 8), np.array([1.0, 0.0, 0.0]))
    with pytest.raises(ValueError):
        uniform_B_potential((8, 8, 8), np.array([1.0, 0.0]))


# ---------------------------------------------------------------------------
# R4 electric sector -- uniform_E_potential (A_0 = -E.(r-origin))
# ---------------------------------------------------------------------------

def test_uniform_E_potential_grad_is_E(small_shape: tuple[int, int, int]) -> None:
    """-grad(A_0) == E everywhere for the uniform-E scalar potential."""
    E = np.array([0.2, -0.1, 0.3])
    A0 = uniform_E_potential(small_shape, E)
    assert A0.shape == small_shape
    # A_0 is linear so np.gradient is exact; -grad recovers E.
    np.testing.assert_allclose(-np.gradient(A0, axis=0), E[0], atol=1e-12)
    np.testing.assert_allclose(-np.gradient(A0, axis=1), E[1], atol=1e-12)
    np.testing.assert_allclose(-np.gradient(A0, axis=2), E[2], atol=1e-12)


def test_a1_exact_under_E_and_B(small_shape: tuple[int, int, int]) -> None:
    """A=1 holds every tick under the **combined E+B** background.

    Confirms the engine supports both gauge sectors at once (R4): the
    electric A_0 via external_potential and the magnetic A via
    vector_potential, with token conservation exact.
    """
    lattice = BipartiteLattice(shape=small_shape)
    session = _wavepacket(lattice)
    A0 = uniform_E_potential(small_shape, np.array([0.05, 0.0, 0.0]))
    A = uniform_B_potential(small_shape, np.array([0.0, 0.0, 0.08]))
    sched = TickScheduler(
        lattice=lattice,
        hop=HopOperator(lattice=lattice),
        external_potential=A0,
        vector_potential=A,
    )
    sched.register(session)
    n_units = session.n_units
    for tick in range(6):
        sched.step()
        assert session.total_tokens() == n_units, f"A=1 broke at tick {tick}"


# ---------------------------------------------------------------------------
# #5 (machinery + interplay) -- the E+B response is live, both sectors
# contribute, and there is no destructive E/B cancellation.
# ---------------------------------------------------------------------------

def test_E_plus_B_interplay_is_live_and_non_cancelling() -> None:
    """The combined E+B induced response is well-defined and does not cancel.

    The requirements' #5 *verdict* -- classify the leading photon
    birefringence order (O(1) dim-4 vs (ka)^2-suppressed vs null) -- is a
    **vacuum-averaged, N-converged** research result and is the job of Paper
    IV's `exp_03` (it needs the token-ensemble induced action, large N, and
    the full orientation sweep; see `04_*.md` R4/R5). It is NOT decided in a
    unit test.

    What this test pins is the **engine machinery exp_03 builds on**, plus
    the part of the verdict that IS robust at CPU scale:

    - both gauge sectors contribute (B-only and E-only responses are live),
    - the combined response adds **roughly in quadrature** (within 20%), so
      there is **no destructive E/B cancellation** -- the *null* branch of
      the #5 verdict is disfavoured already here, and
    - the response is finite (no NaN/inf), consistent with the A=1-exact
      evolution (`test_a1_exact_under_E_and_B`).

    The exact order (O(1) vs (ka)^2) and the final sign close at `exp_03`.
    """
    shape = (12, 12, 12)
    lattice = BipartiteLattice(shape=shape)
    c = shape[0] // 2
    session = DiscreteCausalSession.wavepacket(
        lattice, n_units=1_000_000, omega=0.3, center=(c, c, c), sigma=2.0
    )
    hop = HopOperator(lattice=lattice)
    axis = np.array([1.0, 1.0, -1.0]) / np.sqrt(3.0)  # the optical axis
    B = uniform_B_potential(shape, 0.05 * axis)
    A0 = uniform_E_potential(shape, 0.05 * axis)

    def even_response(external_potential, vector_potential) -> float:
        # Even in the field amplitude (symmetrise over flipping the field):
        # isolates the second-order induced response of the token vacuum.
        rp = hop.step(
            session, "even",
            external_potential=external_potential, vector_potential=vector_potential,
        )
        rm = hop.step(
            session, "even",
            external_potential=None if external_potential is None else -external_potential,
            vector_potential=None if vector_potential is None else -vector_potential,
        )
        r0 = hop.step(session, "even", vector_potential=None)
        rho = lambda r: np.abs(r[0]) ** 2 + np.abs(r[1]) ** 2
        even = 0.5 * (rho(rp) + rho(rm)) - rho(r0)
        return float(np.linalg.norm(even))

    r_b = even_response(None, B)        # magnetic sector
    r_e = even_response(A0, None)       # electric sector
    r_eb = even_response(A0, B)         # combined

    # Both sectors live.
    assert r_b > 0.0 and r_e > 0.0, f"sector(s) dead: r_b={r_b}, r_e={r_e}"
    assert np.isfinite(r_eb)

    # No destructive cancellation: E+B adds ~ in quadrature (within 20%).
    quadrature = float(np.hypot(r_b, r_e))
    assert r_eb > 0.8 * quadrature, (
        f"E+B response collapsed (possible cancellation): r_eb={r_eb}, "
        f"quadrature={quadrature}"
    )


# ---------------------------------------------------------------------------
# Scheduler threading (D1d: static field on TickScheduler -> hop.step)
# ---------------------------------------------------------------------------

def _run_scheduler(
    shape: tuple[int, int, int], vector_potential, n_ticks: int = 4
) -> DiscreteCausalSession:
    """Build a fresh wavepacket session, evolve `n_ticks` under the field."""
    lattice = BipartiteLattice(shape=shape)
    session = _wavepacket(lattice)
    sched = TickScheduler(
        lattice=lattice, hop=HopOperator(lattice=lattice),
        vector_potential=vector_potential,
    )
    sched.register(session)
    for _ in range(n_ticks):
        sched.step()
    return session


def test_scheduler_zero_field_matches_none(
    small_shape: tuple[int, int, int],
) -> None:
    """A scheduler with vector_potential=zeros evolves bit-for-bit as None."""
    s_none = _run_scheduler(small_shape, None)
    s_zero = _run_scheduler(small_shape, np.zeros((3, *small_shape)))
    np.testing.assert_array_equal(s_none.N_RGB, s_zero.N_RGB)
    np.testing.assert_array_equal(s_none.N_CMY, s_zero.N_CMY)
    np.testing.assert_array_equal(s_none.phi_RGB, s_zero.phi_RGB)
    np.testing.assert_array_equal(s_none.phi_CMY, s_zero.phi_CMY)


def test_scheduler_a1_exact_under_field(small_shape: tuple[int, int, int]) -> None:
    """A=1 holds every tick when the scheduler threads a real B field."""
    lattice = BipartiteLattice(shape=small_shape)
    session = _wavepacket(lattice)
    A = uniform_B_potential(small_shape, np.array([0.1, 0.0, 0.0]))
    sched = TickScheduler(
        lattice=lattice, hop=HopOperator(lattice=lattice), vector_potential=A
    )
    idx = sched.register(session)
    n_units = session.n_units
    for tick in range(6):
        sched.step()
        assert session.total_tokens() == n_units, f"A=1 broke at tick {tick}"
    assert idx == 0


# ---------------------------------------------------------------------------
# #3 -- induced response -> 0 quadratically as |B| -> 0
# ---------------------------------------------------------------------------

def test_induced_response_vanishes_quadratically(
    small_shape: tuple[int, int, int],
) -> None:
    """The (even-in-B) induced density response scales as |B|^2.

    The induced response IS the second-order susceptibility: the
    **even-in-B** part of the density change, isolated by symmetrising
    over +-B (the odd/paramagnetic part needs a pre-existing current and
    is not the "induced" piece). For a symmetric zero-momentum probe it
    scales as |B|^2, so doubling B quadruples the response.

    Measured at the analytical (pre-quantisation) level on a single hop so
    the quadratic scaling is not masked by integer-token quantisation.
    """
    lattice = BipartiteLattice(shape=small_shape)
    session = _wavepacket(lattice)  # real, zero-momentum, centred
    hop = HopOperator(lattice=lattice)
    parity = "even"

    def response(scale: float) -> float:
        # B along the optical axis (1, 1, -1).
        B = scale * np.array([1.0, 1.0, -1.0])
        A = uniform_B_potential(small_shape, B)
        rp_R, rp_L = hop.step(session, parity, vector_potential=A)
        rm_R, rm_L = hop.step(session, parity, vector_potential=-A)
        r0_R, r0_L = hop.step(session, parity, vector_potential=None)
        rho_plus = np.abs(rp_R) ** 2 + np.abs(rp_L) ** 2
        rho_minus = np.abs(rm_R) ** 2 + np.abs(rm_L) ** 2
        rho_0 = np.abs(r0_R) ** 2 + np.abs(r0_L) ** 2
        even = 0.5 * (rho_plus + rho_minus) - rho_0  # even-in-B => O(|B|^2)
        return float(np.linalg.norm(even))

    r1 = response(0.01)
    r2 = response(0.02)
    r4 = response(0.04)

    assert r1 > 0.0  # response is live
    # Quadratic: response(2B) / response(B) -> 4.
    np.testing.assert_allclose(r2 / r1, 4.0, rtol=0.05)
    np.testing.assert_allclose(r4 / r2, 4.0, rtol=0.05)


# ---------------------------------------------------------------------------
# #4 (structural) -- magnetic induced response is uniaxial about (1,1,-1),
# axial-large -- the Paper I Q-tensor {4,4,16} eigenstructure.
# ---------------------------------------------------------------------------

def test_induced_response_uniaxial_about_optical_axis() -> None:
    """The magnetic susceptibility is uniaxial about (1,1,-1), axis-large.

    Paper I's induced gauge action gives a `Q`-tensor with eigenvalues
    `{4,4,16}`, the large (16) eigenvalue along the optical axis `(1,1,-1)`
    (App. B). This test confirms the **eigenstructure** the Peierls coupling
    must reproduce:

    - **uniaxial:** the susceptibility is equal for directions perpendicular
      to `(1,1,-1)` (the two `4`s are degenerate), and
    - **axial-large:** `(1,1,-1)` carries the *larger* response (the `16` is
      axial, i.e. prolate not oblate).

    SCOPE / HONEST LIMIT: this pins the *structure* (axis + uniaxial +
    prolate sign), which is what a wrong Peierls/midpoint convention would
    break. It does **not** pin the exact `16/4 = 4` eigenvalue ratio -- that
    needs the **vacuum-averaged** (token-ensemble) induced action and Paper
    I's `induced_gauge_action.py` normalisation, deferred to the exp_03
    cross-check (Phase 3). The backward-midpoint sign flag
    (`acceptance-test-4-confirms-backward-midpoint-sign`) is therefore
    *partially* discharged here (structure confirmed) and fully closed by
    exp_03.
    """
    shape = (12, 12, 12)
    lattice = BipartiteLattice(shape=shape)
    c = shape[0] // 2
    session = DiscreteCausalSession.wavepacket(
        lattice, n_units=1_000_000, omega=0.3, center=(c, c, c), sigma=2.0
    )
    hop = HopOperator(lattice=lattice)

    def chi(direction: tuple[float, float, float], scale: float = 0.02) -> float:
        n = np.asarray(direction, dtype=np.float64)
        n = n / np.linalg.norm(n)
        A = uniform_B_potential(shape, scale * n)
        rp = hop.step(session, "even", vector_potential=A)
        rm = hop.step(session, "even", vector_potential=-A)
        r0 = hop.step(session, "even", vector_potential=None)
        rho = lambda r: np.abs(r[0]) ** 2 + np.abs(r[1]) ** 2
        even = 0.5 * (rho(rp) + rho(rm)) - rho(r0)
        return float(np.linalg.norm(even)) / scale**2  # /|B|^2

    chi_axis = chi((1, 1, -1))
    # Three independent directions perpendicular to (1, 1, -1) (a+b-c = 0).
    chi_perp = [chi((1, -1, 0)), chi((1, 1, 2)), chi((0, 1, 1))]

    # Uniaxial: all perpendicular directions are degenerate.
    for cp in chi_perp[1:]:
        np.testing.assert_allclose(cp, chi_perp[0], rtol=0.02)

    # Axial-large: the optical axis carries a clearly larger response.
    assert chi_axis > 1.2 * max(chi_perp), (
        f"expected axial-large (prolate); chi_axis={chi_axis}, "
        f"chi_perp={chi_perp}"
    )
