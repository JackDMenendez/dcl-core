"""
tests/test_tick_scheduler.py

Unit tests for TickScheduler — the combinatorial clock scheduler.

Theory correspondence:
  register_session()      adds a clock to the combinatorial space.
  advance()               runs one macro-tick (all sessions in shuffle order).
  ShuffleScheme           IS the experimental variable (order of processing).
  bind_sessions()         sets coupling strength for pairwise phase locking.
  register_emission()     wires up an electron-photon radiation-reaction pair.
  clock_count()           IS the number of active clocks.
  combinatorial_state_space_size() IS n! -- arrow of time.
"""

import numpy as np
import pytest
from dcl_core.core.OctahedralLattice import OctahedralLattice
from dcl_core.core.CausalSession import CausalSession
from dcl_core.core.TickScheduler import TickScheduler, ShuffleScheme
from dcl_core.core.UnityConstraint import unity_residual_spinor


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_session(size=11, omega=0.3, center=None):
    lat = OctahedralLattice(size, size, size)
    if center is None:
        center = (size // 2, size // 2, size // 2)
    return CausalSession(lat, center, omega)


def make_photon(size=11, center=None):
    lat = OctahedralLattice(size, size, size)
    if center is None:
        center = (size // 2, size // 2, size // 2)
    return CausalSession(lat, center, 0.0, is_massless=True)


# ── Registration ──────────────────────────────────────────────────────────────

class TestRegistration:

    def test_register_returns_index(self):
        sched = TickScheduler()
        s = make_session()
        idx = sched.register_session(s)
        assert idx == 0

    def test_multiple_sessions_indexed_sequentially(self):
        sched = TickScheduler()
        s1 = make_session()
        s2 = make_session()
        assert sched.register_session(s1) == 0
        assert sched.register_session(s2) == 1

    def test_clock_count(self):
        sched = TickScheduler()
        assert sched.clock_count() == 0
        sched.register_session(make_session())
        assert sched.clock_count() == 1
        sched.register_session(make_session())
        assert sched.clock_count() == 2

    def test_combinatorial_state_space_size(self):
        import math
        sched = TickScheduler()
        sched.register_session(make_session())
        sched.register_session(make_session())
        sched.register_session(make_session())
        assert sched.combinatorial_state_space_size() == math.factorial(3)


# ── Macro-tick counter ────────────────────────────────────────────────────────

class TestMacroTick:

    def test_macro_tick_starts_at_zero(self):
        sched = TickScheduler()
        assert sched.macro_tick == 0

    def test_macro_tick_increments(self):
        sched = TickScheduler()
        sched.register_session(make_session())
        sched.advance()
        assert sched.macro_tick == 1
        sched.advance()
        assert sched.macro_tick == 2


# ── Session tick counters ─────────────────────────────────────────────────────

class TestSessionTickCounters:

    def test_session_tick_counter_increments_per_advance(self):
        sched = TickScheduler()
        s = make_session()
        sched.register_session(s)
        sched.advance()
        assert s.tick_counter == 1

    def test_multiple_sessions_all_advance(self):
        sched = TickScheduler()
        sessions = [make_session() for _ in range(3)]
        for s in sessions:
            sched.register_session(s)
        sched.advance()
        for s in sessions:
            assert s.tick_counter == 1


# ── A=1 after advance ─────────────────────────────────────────────────────────

class TestA1AfterAdvance:
    """All sessions must satisfy A=1 after each macro-tick."""

    def test_single_session_a1(self):
        sched = TickScheduler()
        s = make_session()
        sched.register_session(s)
        for _ in range(10):
            sched.advance()
        res = unity_residual_spinor(s.psi_R, s.psi_L)
        assert res < 1e-10

    def test_two_sessions_a1(self):
        sched = TickScheduler()
        s1 = make_session()
        s2 = make_session()
        sched.register_session(s1)
        sched.register_session(s2)
        for _ in range(10):
            sched.advance()
        assert unity_residual_spinor(s1.psi_R, s1.psi_L) < 1e-10
        assert unity_residual_spinor(s2.psi_R, s2.psi_L) < 1e-10


# ── Shuffle schemes ───────────────────────────────────────────────────────────

class TestShuffleSchemes:

    def _run_n_ticks(self, scheme, n=5):
        sched = TickScheduler(shuffle_scheme=scheme)
        s1 = make_session()
        s2 = make_session()
        sched.register_session(s1)
        sched.register_session(s2)
        for _ in range(n):
            sched.advance()
        return sched, s1, s2

    def test_sequential_runs_without_error(self):
        sched, s1, s2 = self._run_n_ticks(ShuffleScheme.SEQUENTIAL)
        assert sched.macro_tick == 5

    def test_random_runs_without_error(self):
        sched, s1, s2 = self._run_n_ticks(ShuffleScheme.RANDOM)
        assert sched.macro_tick == 5

    def test_priority_runs_without_error(self):
        sched, s1, s2 = self._run_n_ticks(ShuffleScheme.PRIORITY)
        assert sched.macro_tick == 5

    def test_clock_density_raises_not_implemented(self):
        sched = TickScheduler(shuffle_scheme=ShuffleScheme.CLOCK_DENSITY)
        sched.register_session(make_session())
        with pytest.raises(NotImplementedError):
            sched.advance()

    def test_sequential_processing_order_is_sorted(self):
        sched = TickScheduler(shuffle_scheme=ShuffleScheme.SEQUENTIAL)
        sched.register_session(make_session())
        sched.register_session(make_session())
        sched.register_session(make_session())
        order = sched._processing_order()
        assert order == [0, 1, 2]

    def test_priority_order_by_tick_counter(self):
        sched = TickScheduler(shuffle_scheme=ShuffleScheme.PRIORITY)
        s1 = make_session()
        s2 = make_session()
        s3 = make_session()
        sched.register_session(s1)
        sched.register_session(s2)
        sched.register_session(s3)
        # Manually set tick counters
        s1.tick_counter = 5
        s2.tick_counter = 1
        s3.tick_counter = 3
        order = sched._processing_order()
        # Should be sorted: s2 (1), s3 (3), s1 (5) → indices 1, 2, 0
        assert order == [1, 2, 0]


# ── Bind sessions ─────────────────────────────────────────────────────────────

class TestBindSessions:

    def test_bind_sets_coupling(self):
        sched = TickScheduler()
        sched.register_session(make_session())
        sched.register_session(make_session())
        sched.bind_sessions(0, 1, coupling=0.8)
        key = (0, 1)
        assert sched._bindings[key] == pytest.approx(0.8)

    def test_bind_clamps_above_one(self):
        sched = TickScheduler()
        sched.register_session(make_session())
        sched.register_session(make_session())
        sched.bind_sessions(0, 1, coupling=2.0)
        assert sched._bindings[(0, 1)] == pytest.approx(1.0)

    def test_bind_clamps_below_zero(self):
        sched = TickScheduler()
        sched.register_session(make_session())
        sched.register_session(make_session())
        sched.bind_sessions(0, 1, coupling=-0.5)
        assert sched._bindings[(0, 1)] == pytest.approx(0.0)

    def test_bind_key_order_canonical(self):
        """Key is always (min, max) regardless of argument order."""
        sched = TickScheduler()
        sched.register_session(make_session())
        sched.register_session(make_session())
        sched.bind_sessions(1, 0, coupling=0.5)
        assert (0, 1) in sched._bindings

    def test_strongly_bound_sessions_advance_without_error(self):
        sched = TickScheduler()
        s1 = make_session(center=(4, 5, 5))
        s2 = make_session(center=(6, 5, 5))
        sched.register_session(s1)
        sched.register_session(s2)
        sched.bind_sessions(0, 1, coupling=0.9)
        for _ in range(5):
            sched.advance()
        assert sched.macro_tick == 5

    def test_a1_preserved_with_strong_binding(self):
        sched = TickScheduler()
        s1 = make_session()
        s2 = make_session()
        sched.register_session(s1)
        sched.register_session(s2)
        sched.bind_sessions(0, 1, coupling=1.0)
        for _ in range(8):
            sched.advance()
        assert unity_residual_spinor(s1.psi_R, s1.psi_L) < 1e-10
        assert unity_residual_spinor(s2.psi_R, s2.psi_L) < 1e-10


# ── Emission registration ─────────────────────────────────────────────────────

class TestEmissionRegistration:

    def test_register_emission_stores_pair(self):
        sched = TickScheduler()
        electron = make_session(omega=0.1019)
        photon = make_photon()
        proton = make_session(omega=np.pi / 2)
        e_idx = sched.register_session(electron)
        p_idx = sched.register_session(photon)
        pr_idx = sched.register_session(proton)
        sched.register_emission(e_idx, p_idx, pr_idx, rate=0.005)
        assert len(sched._emission_pairs) == 1
        assert sched._emission_pairs[0][0] == e_idx
        assert sched._emission_pairs[0][1] == p_idx

    def test_emission_weight_initialised_to_one(self):
        sched = TickScheduler()
        e = make_session()
        p = make_photon()
        pr = make_session()
        e_idx = sched.register_session(e)
        p_idx = sched.register_session(p)
        pr_idx = sched.register_session(pr)
        sched.register_emission(e_idx, p_idx, pr_idx, rate=0.01)
        assert sched.emission_weight(e_idx) == pytest.approx(1.0)

    def test_advance_with_emission_pair_does_not_crash(self):
        """Emission coupling must not blow up on advance."""
        sched = TickScheduler(shuffle_scheme=ShuffleScheme.SEQUENTIAL)
        electron = make_session(omega=0.1019)
        photon = make_photon()
        proton = make_session(omega=np.pi / 2)
        e_idx = sched.register_session(electron)
        p_idx = sched.register_session(photon)
        pr_idx = sched.register_session(proton)
        sched.register_emission(e_idx, p_idx, pr_idx, rate=0.005)
        for _ in range(5):
            sched.advance()
        assert sched.macro_tick == 5

    def test_a1_preserved_with_emission_registered(self):
        sched = TickScheduler(shuffle_scheme=ShuffleScheme.SEQUENTIAL)
        electron = make_session(omega=0.1019)
        photon = make_photon()
        proton = make_session(omega=np.pi / 2)
        e_idx = sched.register_session(electron)
        p_idx = sched.register_session(photon)
        pr_idx = sched.register_session(proton)
        sched.register_emission(e_idx, p_idx, pr_idx, rate=0.005)
        for _ in range(5):
            sched.advance()
        for s in [electron, photon, proton]:
            assert unity_residual_spinor(s.psi_R, s.psi_L) < 1e-10

    def test_emission_grid_precomputed(self):
        sched = TickScheduler()
        e = make_session(size=9)
        p = make_photon(size=9)
        pr = make_session(size=9)
        e_idx = sched.register_session(e)
        p_idx = sched.register_session(p)
        pr_idx = sched.register_session(pr)
        sched.register_emission(e_idx, p_idx, pr_idx, rate=0.01)
        xx, yy, zz = sched._emission_grid[e_idx]
        assert xx.shape == (9, 9, 9)
        assert yy.shape == (9, 9, 9)
        assert zz.shape == (9, 9, 9)
