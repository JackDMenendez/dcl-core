"""
tests/test_causal_session.py

Unit tests for CausalSession — a particle as a persistent causal session.

Theory correspondence:
  psi_R, psi_L         IS the two-component Dirac spinor (RGB/CMY sublattices).
  tick()               IS the bipartite Dirac evolution step.
  probability_density()IS the Born probability |psi_R|^2 + |psi_L|^2.
  A=1                  IS the unity constraint enforced after every tick.
  is_massless=True     IS the photon (massless, strict alternating parity).
"""

import numpy as np
import pytest
from dcl_core.core.OctahedralLattice import OctahedralLattice
from dcl_core.core.CausalSession import CausalSession
from dcl_core.core.UnityConstraint import unity_residual_spinor


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_lattice(size=11):
    return OctahedralLattice(size, size, size)


def make_session(size=11, omega=0.3, center=None, momentum=(0., 0., 0.),
                 is_massless=False):
    lat = make_lattice(size)
    if center is None:
        center = (size // 2, size // 2, size // 2)
    return CausalSession(lat, center, omega, momentum=momentum,
                         is_massless=is_massless)


# ── Construction ──────────────────────────────────────────────────────────────

class TestCausalSessionConstruction:

    def test_spinor_shapes(self):
        s = make_session(9)
        assert s.psi_R.shape == (9, 9, 9)
        assert s.psi_L.shape == (9, 9, 9)

    def test_a1_at_construction(self):
        """A=1 is enforced on construction."""
        s = make_session()
        res = unity_residual_spinor(s.psi_R, s.psi_L)
        assert res < 1e-12

    def test_initial_probability_at_center(self):
        """All probability starts at the initial node."""
        s = make_session(11, center=(5, 5, 5))
        total = float(s.probability_density().sum())
        assert total == pytest.approx(1.0, abs=1e-12)

    def test_initial_amplitude_localised(self):
        s = make_session(11, center=(5, 5, 5))
        p = s.probability_density()
        assert p[5, 5, 5] == pytest.approx(1.0, abs=1e-12)
        # All other sites are zero
        mask = np.ones((11, 11, 11), dtype=bool)
        mask[5, 5, 5] = False
        assert np.all(p[mask] < 1e-20)

    def test_tick_counter_zero(self):
        s = make_session()
        assert s.tick_counter == 0

    def test_is_massless_flag(self):
        s_photon = make_session(is_massless=True)
        assert s_photon.is_massless is True
        s_massive = make_session(is_massless=False)
        assert s_massive.is_massless is False

    def test_psi_property_returns_psi_r(self):
        s = make_session()
        assert s.psi is s.psi_R

    def test_psi_setter_distributes_evenly(self):
        """Setting psi via the backwards-compat setter distributes sqrt(2) equally."""
        s = make_session(9)
        shape = (9, 9, 9)
        new_psi = np.zeros(shape, dtype=complex)
        new_psi[4, 4, 4] = 1.0
        s.psi = new_psi
        # Both components should have magnitude 1/sqrt(2) at (4,4,4)
        amp_R = abs(s.psi_R[4, 4, 4])
        amp_L = abs(s.psi_L[4, 4, 4])
        assert amp_R == pytest.approx(1.0 / np.sqrt(2), abs=1e-12)
        assert amp_L == pytest.approx(1.0 / np.sqrt(2), abs=1e-12)


# ── A=1 conservation across ticks ────────────────────────────────────────────

class TestA1Conservation:
    """A=1 must be maintained at every tick for both massive and massless sessions."""

    def _check_a1_after_ticks(self, session, n_ticks=20):
        for _ in range(n_ticks):
            session.tick()
            res = unity_residual_spinor(session.psi_R, session.psi_L)
            assert res < 1e-10, f"A=1 violated: residual={res}"

    def test_massive_flat_lattice(self):
        s = make_session(omega=0.5)
        self._check_a1_after_ticks(s)

    def test_massless_flat_lattice(self):
        s = make_session(omega=0.0, is_massless=True)
        self._check_a1_after_ticks(s)

    def test_massive_with_coulomb_well(self):
        lat = make_lattice(11)
        lat.set_coulomb_well((5, 5, 5), strength=5.0, softening=0.5)
        s = CausalSession(lat, (5, 5, 5), 0.3)
        self._check_a1_after_ticks(s)

    def test_massive_with_momentum(self):
        s = make_session(omega=0.2, momentum=(0.1, 0.0, 0.0))
        self._check_a1_after_ticks(s)

    def test_high_mass(self):
        """High omega (near pi) should still conserve A=1."""
        s = make_session(omega=np.pi * 0.9)
        self._check_a1_after_ticks(s)

    def test_a1_with_normalize_false(self):
        """When normalize=False tick() skips enforce_unity_spinor.

        The kinetic hop redistributes amplitude so the norm changes; the key
        observable is that the result is NOT snapped back to exactly 1.0.
        We run one tick without normalisation and compare against a parallel
        tick with normalisation: they must differ.
        """
        import copy
        s = make_session()
        s_ref = CausalSession(s.lattice, (5, 5, 5), 0.3)
        # Give them identical state
        s_ref.psi_R[:] = s.psi_R
        s_ref.psi_L[:] = s.psi_L
        # Scale up before the unnormalised tick so there is drift to detect
        s.psi_R *= 2.0
        s.psi_L *= 2.0
        s.tick(normalize=False)
        total_unnorm = float(np.sum(np.abs(s.psi_R)**2 + np.abs(s.psi_L)**2))
        # Must NOT equal 1.0 — if it did, normalization was applied silently
        assert abs(total_unnorm - 1.0) > 0.05


# ── Probability density ───────────────────────────────────────────────────────

class TestProbabilityDensity:

    def test_sums_to_one(self):
        s = make_session()
        for _ in range(10):
            s.tick()
        assert float(s.probability_density().sum()) == pytest.approx(1.0, abs=1e-10)

    def test_non_negative(self):
        s = make_session()
        for _ in range(10):
            s.tick()
        assert np.all(s.probability_density() >= 0.0)

    def test_shape(self):
        s = make_session(9)
        assert s.probability_density().shape == (9, 9, 9)

    def test_equal_to_abs_squared_sum(self):
        s = make_session()
        for _ in range(5):
            s.tick()
        pd = s.probability_density()
        expected = np.abs(s.psi_R)**2 + np.abs(s.psi_L)**2
        np.testing.assert_allclose(pd, expected)


# ── Massless vs massive spreading ────────────────────────────────────────────

class TestSpreadingBehavior:
    """A massless photon spreads faster than a massive particle."""

    def test_massless_spreads_faster_than_massive(self):
        """After N ticks, massless session has smaller interior fraction.

        advance_tick_counter() must be called alongside tick() so the photon
        alternates between RGB and CMY hops (otherwise psi_L never updates and
        interior_fraction stays artificially high).
        """
        n = 13
        center = (n // 2, n // 2, n // 2)
        lat_m = OctahedralLattice(n, n, n)
        lat_p = OctahedralLattice(n, n, n)
        massive = CausalSession(lat_m, center, 0.8)
        photon = CausalSession(lat_p, center, 0.0, is_massless=True)
        ticks = 5
        for _ in range(ticks):
            massive.tick()
            massive.advance_tick_counter()
            photon.tick()
            photon.advance_tick_counter()
        r_test = 2.0
        frac_massive = massive.interior_fraction(center, r_test)
        frac_photon = photon.interior_fraction(center, r_test)
        assert frac_massive > frac_photon

    def test_high_mass_stays_localised(self):
        """Very heavy particle barely moves."""
        s = make_session(omega=np.pi * 0.95, size=13)
        center = (6, 6, 6)
        frac_before = s.interior_fraction(center, 1.5)
        for _ in range(10):
            s.tick()
        frac_after = s.interior_fraction(center, 1.5)
        # Should still have substantial fraction near centre
        assert frac_after > 0.3


# ── Tick counter ──────────────────────────────────────────────────────────────

class TestTickCounter:

    def test_advance_tick_counter_increments(self):
        s = make_session()
        s.advance_tick_counter()
        assert s.tick_counter == 1

    def test_advance_tick_counter_increments_phase_oscillator(self):
        s = make_session(omega=0.5)
        s.advance_tick_counter()
        assert s.phase_oscillator.phase == pytest.approx(0.5)

    def test_tick_does_not_advance_counter(self):
        """tick() evolves the wavefunction; advance_tick_counter() is separate."""
        s = make_session()
        s.tick()
        assert s.tick_counter == 0  # counter not changed by tick alone


# ── Cone properties ───────────────────────────────────────────────────────────

class TestConeProperties:

    def test_cone_half_angle_massless(self):
        """Massless photon has half-angle arcsin(cos(0)) = pi/2."""
        s = make_session(omega=0.0)
        assert s.cone_half_angle == pytest.approx(np.pi / 2)

    def test_cone_half_angle_decreases_with_mass(self):
        s_light = make_session(omega=0.1)
        s_heavy = make_session(omega=1.5)
        assert s_light.cone_half_angle > s_heavy.cone_half_angle

    def test_rgb_cmy_imbalance_balanced_at_init(self):
        """Initial state has equal R and L amplitude — zero imbalance."""
        s = make_session()
        assert s.rgb_cmy_imbalance == pytest.approx(0.0, abs=1e-12)


# ── Phase gradient and phase map ──────────────────────────────────────────────

class TestPhaseGradientAndMap:

    def test_phase_gradient_field_shape(self):
        s = make_session(9)
        grad = s.phase_gradient_field()
        assert grad.shape == (3, 9, 9, 9)

    def test_apply_phase_map_preserves_a1(self):
        s = make_session()
        delta = np.random.random((11, 11, 11)) * 0.5
        s.apply_phase_map(delta)
        res = unity_residual_spinor(s.psi_R, s.psi_L)
        assert res < 1e-12

    def test_apply_phase_map_changes_phase_not_amplitude(self):
        s = make_session(9)
        amp_before_R = np.abs(s.psi_R).copy()
        amp_before_L = np.abs(s.psi_L).copy()
        delta = np.ones((9, 9, 9)) * 0.3
        s.apply_phase_map(delta)
        np.testing.assert_allclose(np.abs(s.psi_R), amp_before_R, atol=1e-14)
        np.testing.assert_allclose(np.abs(s.psi_L), amp_before_L, atol=1e-14)


# ── Sublattice ratio ──────────────────────────────────────────────────────────

class TestImposeSublatticeRatio:

    def test_target_ratio_achieved(self):
        s = make_session()
        s.impose_sublattice_ratio(0.8)
        p_R = float(np.sum(np.abs(s.psi_R)**2))
        assert p_R == pytest.approx(0.8, abs=1e-6)

    def test_a1_preserved_after_sublattice_set(self):
        s = make_session()
        s.impose_sublattice_ratio(0.3)
        res = unity_residual_spinor(s.psi_R, s.psi_L)
        assert res < 1e-12

    def test_fully_right_handed(self):
        s = make_session()
        s.impose_sublattice_ratio(1.0)
        p_R = float(np.sum(np.abs(s.psi_R)**2))
        assert p_R == pytest.approx(1.0, abs=1e-6)

    def test_fully_left_handed(self):
        s = make_session()
        s.impose_sublattice_ratio(0.0)
        p_L = float(np.sum(np.abs(s.psi_L)**2))
        assert p_L == pytest.approx(1.0, abs=1e-6)


# ── Interior fraction and cone profile ───────────────────────────────────────

class TestInteriorFraction:

    def test_interior_fraction_range(self):
        s = make_session()
        for _ in range(5):
            s.tick()
        frac = s.interior_fraction((5, 5, 5), 3.0)
        assert 0.0 <= frac <= 1.0

    def test_large_radius_captures_all(self):
        s = make_session(11, center=(5, 5, 5))
        for _ in range(3):
            s.tick()
        frac = s.interior_fraction((5, 5, 5), 20.0)
        assert frac == pytest.approx(1.0, abs=1e-10)

    def test_cone_amplitude_profile_shapes(self):
        s = make_session(11, center=(5, 5, 5))
        radii, profile = s.cone_amplitude_profile((5, 5, 5), n_shells=10)
        assert radii.shape == (10,)
        assert profile.shape == (10,)

    def test_cone_amplitude_profile_sums_to_one(self):
        s = make_session(11, center=(5, 5, 5))
        _, profile = s.cone_amplitude_profile((5, 5, 5), n_shells=20)
        assert float(profile.sum()) == pytest.approx(1.0, abs=1e-10)


# ── Momentum initialisation ───────────────────────────────────────────────────

class TestMomentumInitialisation:

    def test_momentum_shifts_com_over_time(self):
        """Non-zero momentum gives a net CoM drift vs the zero-momentum case.

        A single-node initial state has no spatial phase gradient on tick 0
        (neighbours are empty, so delta_p = 0).  The gradient builds up once
        amplitude has spread.  We run enough ticks for the gradient to develop
        and compare +k vs -k: their CoMs must diverge in opposite x-directions.
        """
        n = 21
        center = (n // 2, n // 2, n // 2)

        lat_pos = OctahedralLattice(n, n, n)
        lat_neg = OctahedralLattice(n, n, n)
        s_pos = CausalSession(lat_pos, center, 0.1, momentum=( 0.4, 0.0, 0.0))
        s_neg = CausalSession(lat_neg, center, 0.1, momentum=(-0.4, 0.0, 0.0))

        x = np.arange(n)
        xx, _, _ = np.meshgrid(x, x, x, indexing='ij')

        for _ in range(20):
            s_pos.tick(); s_pos.advance_tick_counter()
            s_neg.tick(); s_neg.advance_tick_counter()

        com_pos = float(np.sum(xx * s_pos.probability_density()))
        com_neg = float(np.sum(xx * s_neg.probability_density()))
        # Opposite momenta → CoMs drift in opposite directions
        assert com_pos > com_neg
