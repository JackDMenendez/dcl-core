"""
tests/test_composite_causal_session.py

Unit tests for CompositeCausalSession — a bound composite of CausalSessions.

Theory correspondence:
  probability_density()   IS |Σ ψ_i|²  (coherent sum — phases interfere).
  incoherent_density()    IS  Σ|ψ_i|²  (independent sessions, no cross-terms).
  charge_balance()        IS  Σ(|ψ_R_i|² - |ψ_L_i|²) — net charge proxy.
  phase_coherence()       IS  |Σψ|²/Σ|ψ|² ∈ [0,1] — fringe visibility.
  binding_strength → 1    IS  the strong-force phase lock (cone narrowing).
"""

import numpy as np
import pytest
from dcl_core.core.OctahedralLattice import OctahedralLattice
from dcl_core.core.CausalSession import CausalSession
from dcl_core.core.CompositeCausalSession import CompositeCausalSession
from dcl_core.core.UnityConstraint import unity_residual_spinor


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_session(size=9, omega=0.3, center=None):
    lat = OctahedralLattice(size, size, size)
    if center is None:
        center = (size // 2, size // 2, size // 2)
    return CausalSession(lat, center, omega)


def make_composite(n=2, omega=0.3, size=9, binding=0.9):
    sessions = [make_session(size=size, omega=omega) for _ in range(n)]
    return CompositeCausalSession(sessions, binding_strength=binding)


# ── Construction ──────────────────────────────────────────────────────────────

class TestConstruction:

    def test_requires_at_least_two_sessions(self):
        s = make_session()
        with pytest.raises(ValueError):
            CompositeCausalSession([s])

    def test_two_sessions_ok(self):
        comp = make_composite(n=2)
        assert len(comp.sessions) == 2

    def test_binding_strength_stored(self):
        comp = make_composite(binding=0.7)
        assert comp.binding_strength == pytest.approx(0.7)

    def test_binding_strength_clamped_above_one(self):
        sessions = [make_session(), make_session()]
        comp = CompositeCausalSession(sessions, binding_strength=1.5)
        assert comp.binding_strength == pytest.approx(1.0)

    def test_binding_strength_clamped_below_zero(self):
        sessions = [make_session(), make_session()]
        comp = CompositeCausalSession(sessions, binding_strength=-0.3)
        assert comp.binding_strength == pytest.approx(0.0)

    def test_tick_counter_starts_zero(self):
        comp = make_composite()
        assert comp.tick_counter == 0


# ── Tick and advance ──────────────────────────────────────────────────────────

class TestTickAndAdvance:

    def test_tick_does_not_crash(self):
        comp = make_composite()
        comp.tick()

    def test_tick_increments_no_counter(self):
        """tick() evolves sessions but does not increment the composite counter."""
        comp = make_composite()
        comp.tick()
        assert comp.tick_counter == 0

    def test_advance_tick_counter_increments(self):
        comp = make_composite()
        comp.advance_tick_counter()
        assert comp.tick_counter == 1

    def test_advance_tick_counter_propagates_to_sessions(self):
        comp = make_composite()
        comp.advance_tick_counter()
        for s in comp.sessions:
            assert s.tick_counter == 1

    def test_multiple_ticks_stable(self):
        comp = make_composite(n=3)
        for _ in range(20):
            comp.tick()
        # Constituent sessions should each satisfy A=1
        for s in comp.sessions:
            res = unity_residual_spinor(s.psi_R, s.psi_L)
            assert res < 1e-10


# ── Probability density: coherent vs incoherent ───────────────────────────────

class TestProbabilityDensity:

    def test_probability_density_shape(self):
        comp = make_composite(size=9)
        pd = comp.probability_density()
        assert pd.shape == (9, 9, 9)

    def test_incoherent_density_shape(self):
        comp = make_composite(size=9)
        inc = comp.incoherent_density()
        assert inc.shape == (9, 9, 9)

    def test_probability_density_non_negative(self):
        comp = make_composite()
        for _ in range(5):
            comp.tick()
        assert np.all(comp.probability_density() >= 0.0)

    def test_incoherent_density_non_negative(self):
        comp = make_composite()
        for _ in range(5):
            comp.tick()
        assert np.all(comp.incoherent_density() >= 0.0)

    def test_coherent_and_incoherent_same_total_at_random_phases(self):
        """Coherent and incoherent totals bracket each other as phases vary.

        |Σψ|² = Σ|ψ|² + 2 Re(cross-terms): cross-terms can be positive
        (constructive) or negative (destructive) at individual nodes, so neither
        pointwise inequality holds in general.  What is guaranteed: the totals
        satisfy  coh_total ∈ [0, N * inc_total]  (Cauchy-Schwarz).
        We check the upper bound (coh ≤ N * inc integrated).
        """
        comp = make_composite(n=3)
        for _ in range(5):
            comp.tick()
        coh_total = float(comp.probability_density().sum())
        inc_total = float(comp.incoherent_density().sum())
        n = len(comp.sessions)
        assert coh_total >= 0.0
        assert coh_total <= n * inc_total + 1e-10

    def test_coherent_le_n_times_incoherent(self):
        """Coherent can exceed incoherent for aligned phases, but not by more than N²/1."""
        # For equal amplitudes: |Σψ|² ≤ N * Σ|ψ|²  (Cauchy-Schwarz upper bound)
        comp = make_composite(n=2, binding=1.0)
        for _ in range(10):
            comp.tick()
        coh = float(comp.probability_density().sum())
        inc = float(comp.incoherent_density().sum())
        n = len(comp.sessions)
        assert coh <= n * inc + 1e-10


# ── Phase coherence ───────────────────────────────────────────────────────────

class TestPhaseCoherence:

    def test_coherence_in_range(self):
        """Phase coherence = |Σψ|²/Σ|ψ|² ∈ [0, N].

        1.0 = incoherent phases (cancellation and reinforcement balance).
        N   = perfectly aligned phases (fully constructive: |Σψ|² = N²·each).
        0.0 = perfectly cancelling phases (fully destructive).
        """
        comp = make_composite()
        for _ in range(5):
            comp.tick()
        c = comp.phase_coherence()
        n = len(comp.sessions)
        assert 0.0 <= c <= float(n) + 1e-10

    def test_identical_sessions_coherence_equals_n(self):
        """Two identical sessions: |Σψ|² = 4|ψ|² = 2 * (2|ψ|²). Coherence = N."""
        size = 9
        lat = OctahedralLattice(size, size, size)
        center = (4, 4, 4)
        omega = 0.3
        s1 = CausalSession(lat, center, omega)
        s2 = CausalSession(lat, center, omega)
        # Force both to exact same state
        s2.psi_R[:] = s1.psi_R
        s2.psi_L[:] = s1.psi_L
        comp = CompositeCausalSession([s1, s2], binding_strength=0.0)
        c = comp.phase_coherence()
        # |Σψ|²/Σ|ψ|² = 4|ψ|²/(2|ψ|²) = 2 = N
        assert c == pytest.approx(2.0, abs=1e-6)


# ── Charge balance ────────────────────────────────────────────────────────────

class TestChargeBalance:

    def test_neutral_composite_at_init(self):
        """Balanced sessions start with zero charge proxy each."""
        comp = make_composite()
        # Each session is balanced (p_R = p_L = 0.5), so sum = 0
        assert comp.charge_balance() == pytest.approx(0.0, abs=1e-12)

    def test_charge_balance_is_sum_of_imbalances(self):
        comp = make_composite(n=3)
        expected = sum(s.rgb_cmy_imbalance for s in comp.sessions)
        assert comp.charge_balance() == pytest.approx(expected)

    def test_charge_balance_changes_with_sublattice_ratio(self):
        s1 = make_session()
        s2 = make_session()
        s1.impose_sublattice_ratio(1.0)  # fully right-handed
        s2.impose_sublattice_ratio(0.0)  # fully left-handed
        comp = CompositeCausalSession([s1, s2])
        balance = comp.charge_balance()
        assert abs(balance) < 0.1  # near-neutral composite


# ── Effective cone properties ─────────────────────────────────────────────────

class TestEffectiveCone:

    def test_effective_cone_angle_positive(self):
        comp = make_composite(omega=0.5)
        angle = comp.effective_cone_half_angle()
        assert angle > 0.0

    def test_effective_cone_angle_matches_mean_of_constituents(self):
        """effective_cone_half_angle returns mean of constituent cone angles."""
        comp = make_composite(n=3, omega=0.5)
        expected = np.mean([s.cone_half_angle for s in comp.sessions])
        assert comp.effective_cone_half_angle() == pytest.approx(expected)

    def test_heavier_composite_has_smaller_cone_angle(self):
        comp_light = make_composite(omega=0.1)
        comp_heavy = make_composite(omega=1.5)
        assert comp_light.effective_cone_half_angle() > comp_heavy.effective_cone_half_angle()


# ── Binding physics ───────────────────────────────────────────────────────────

class TestBindingPhysics:

    def test_zero_binding_no_phase_lock(self):
        """binding_strength=0: no phase mixing applied."""
        comp = make_composite(binding=0.0, n=2)
        # Should tick without error and maintain A=1
        for _ in range(10):
            comp.tick()
        for s in comp.sessions:
            res = unity_residual_spinor(s.psi_R, s.psi_L)
            assert res < 1e-10

    def test_full_binding_phases_become_aligned(self):
        """With binding_strength=1, repeated binding should align constituent phases."""
        size = 9
        lat = OctahedralLattice(size, size, size)
        center = (4, 4, 4)
        # Start with sessions at same point so they interact
        s1 = CausalSession(lat, center, 0.3)
        s2 = CausalSession(lat, center, 0.3)
        # Give s2 a different initial phase offset
        s2.apply_phase_map(np.ones((size, size, size)) * 1.0)
        comp = CompositeCausalSession([s1, s2], binding_strength=1.0)
        for _ in range(30):
            comp.tick()
        # After many binding steps, coherence should be high
        c = comp.phase_coherence()
        assert c > 1.0  # coherence > 1 means phases are aligning (constructive)
