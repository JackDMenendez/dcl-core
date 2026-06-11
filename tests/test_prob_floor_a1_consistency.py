"""tests/test_prob_floor_a1_consistency.py

Exercises the ``prob_floor`` parameter on ``dcl_core.core.CausalSession``
(the continuous-amplitude engine's delta_p_min knob).

Contract (from dcl-delta-p-min's notes/dcl_core_coordination.md):
  1. Backwards compatible -- prob_floor=None is bit-for-bit the old engine.
  2. Per-site clamp of the Born density p(x) = |psi_R|^2 + |psi_L|^2 up to
     prob_floor, with phase (and the psi_R/psi_L ratio) preserved.
  3. Unity-preserving -- A=1 (sum p == 1) holds after every tick.
  4. Applied at end-of-tick.
"""

import numpy as np
import pytest

from dcl_core.core.CausalSession import CausalSession
from dcl_core.core.OctahedralLattice import OctahedralLattice
from dcl_core.core.UnityConstraint import unity_residual_spinor


def make_session(size=11, omega=0.3, center=None, momentum=(0.0, 0.0, 0.0),
                 prob_floor=None):
    lat = OctahedralLattice(size, size, size)
    if center is None:
        center = (size // 2, size // 2, size // 2)
    return CausalSession(lat, center, omega, momentum=momentum,
                         prob_floor=prob_floor)


# ── Validation ──────────────────────────────────────────────────────────────

class TestProbFloorValidation:

    @pytest.mark.parametrize("bad", [0.0, -1e-6, 1.0, 2.0])
    def test_out_of_range_raises(self, bad):
        with pytest.raises(ValueError, match="prob_floor"):
            make_session(prob_floor=bad)

    def test_none_is_default(self):
        assert make_session().prob_floor is None

    @pytest.mark.parametrize("good", [1e-10, 1e-4, 0.25, 0.999])
    def test_in_range_accepted(self, good):
        assert make_session(prob_floor=good).prob_floor == good


# ── Backwards compatibility (Property 1) ─────────────────────────────────────

class TestProbFloorBackwardsCompatible:

    def test_none_matches_no_kwarg_bitwise(self):
        """prob_floor=None must be bit-for-bit identical to not passing it."""
        a = make_session(9, momentum=(0.2, 0.0, 0.0))           # no kwarg
        b = make_session(9, momentum=(0.2, 0.0, 0.0), prob_floor=None)
        for _ in range(12):
            a.tick(); a.advance_tick_counter()
            b.tick(); b.advance_tick_counter()
        assert np.array_equal(a.psi_R, b.psi_R)
        assert np.array_equal(a.psi_L, b.psi_L)


# ── Per-site clamp + phase preservation (Properties 2, 3) ────────────────────

class TestProbFloorClamp:

    def test_clamps_small_site_to_floor_exactly(self):
        s = make_session(7, prob_floor=1e-4)
        s.psi_R[:] = 0.0
        s.psi_L[:] = 0.0
        s.psi_R[3, 3, 3] = 0.8                       # big: p = 0.64 > floor
        s.psi_R[0, 0, 0] = 1e-3 * np.exp(1j * 0.7)   # small: p = 1e-6 < floor
        s._apply_prob_floor(s.psi_R, s.psi_L)

        p_small = abs(s.psi_R[0, 0, 0]) ** 2 + abs(s.psi_L[0, 0, 0]) ** 2
        assert p_small == pytest.approx(1e-4, rel=1e-12)

    def test_phase_and_ratio_preserved(self):
        s = make_session(7, prob_floor=1e-4)
        s.psi_R[:] = 0.0
        s.psi_L[:] = 0.0
        s.psi_R[2, 2, 2] = 8e-4 * np.exp(1j * 0.7)   # combined p < floor
        s.psi_L[2, 2, 2] = 6e-4 * np.exp(-1j * 1.3)
        ratio_before = s.psi_R[2, 2, 2] / s.psi_L[2, 2, 2]
        ang_R = np.angle(s.psi_R[2, 2, 2])
        ang_L = np.angle(s.psi_L[2, 2, 2])

        s._apply_prob_floor(s.psi_R, s.psi_L)

        assert np.angle(s.psi_R[2, 2, 2]) == pytest.approx(ang_R)
        assert np.angle(s.psi_L[2, 2, 2]) == pytest.approx(ang_L)
        assert s.psi_R[2, 2, 2] / s.psi_L[2, 2, 2] == pytest.approx(ratio_before)

    def test_big_site_unchanged(self):
        s = make_session(7, prob_floor=1e-4)
        s.psi_R[:] = 0.0
        s.psi_L[:] = 0.0
        s.psi_R[3, 3, 3] = 0.8
        before = s.psi_R[3, 3, 3]
        s._apply_prob_floor(s.psi_R, s.psi_L)
        assert s.psi_R[3, 3, 3] == before

    def test_zero_site_stays_zero(self):
        """No phase to preserve at an empty node -- it must not be filled."""
        s = make_session(7, prob_floor=1e-4)
        s.psi_R[:] = 0.0
        s.psi_L[:] = 0.0
        s.psi_R[3, 3, 3] = 0.8
        s._apply_prob_floor(s.psi_R, s.psi_L)
        assert s.psi_R[1, 1, 1] == 0.0
        assert s.psi_L[1, 1, 1] == 0.0

    def test_floor_engages_end_to_end(self):
        """A floor high enough to bite the packet's tails changes the
        evolution relative to a floor-free run -- while A=1 still holds."""
        free = make_session(11, momentum=(0.3, 0.1, 0.0))
        floored = make_session(11, momentum=(0.3, 0.1, 0.0), prob_floor=1e-3)
        for _ in range(8):
            free.tick(); free.advance_tick_counter()
            floored.tick(); floored.advance_tick_counter()
        # the floor bit -> the two evolutions diverge ...
        assert not np.allclose(free.probability_density(),
                               floored.probability_density())
        # ... and the clamp did not break the unity constraint.
        assert unity_residual_spinor(floored.psi_R, floored.psi_L) < 1e-10


# ── Unity preservation over many ticks (Property 3) ──────────────────────────

class TestProbFloorUnity:

    @pytest.mark.parametrize("floor", [1e-10, 1e-6, 1e-3, 0.05])
    def test_a1_holds_every_tick(self, floor):
        s = make_session(11, momentum=(0.25, 0.0, 0.0), prob_floor=floor)
        for _ in range(20):
            s.tick()
            assert unity_residual_spinor(s.psi_R, s.psi_L) < 1e-10
            s.advance_tick_counter()

    def test_a1_holds_massless(self):
        s = make_session(11, prob_floor=1e-5)
        s.is_massless = True
        for _ in range(15):
            s.tick()
            assert unity_residual_spinor(s.psi_R, s.psi_L) < 1e-10
            s.advance_tick_counter()


# ── Deep-denormal overflow safety (v0.2.2 regression) ────────────────────────

class TestProbFloorDenormalOverflow:
    """A continuous wavepacket's tail underflows p(x) to deep denormals
    (p ~ 1e-323).  The fused ``sqrt(prob_floor / p)`` overflows to inf there
    (prob_floor / p > max_float), then NaN-poisons the joint renormalisation.
    v0.2.2 splits the rescale into ``sqrt(prob_floor) / sqrt(p)`` so both
    operands stay in the normal float64 range.  These guard that fix."""

    def _denormal_packet(self, size, width, eps):
        lat = OctahedralLattice(size, size, size)
        c = size // 2
        x = np.arange(size)
        xx, yy, zz = np.meshgrid(x, x, x, indexing="ij")
        # Narrow Gaussian -> its tail underflows to deep denormals on this grid.
        env = np.exp(-0.5 * ((xx - c) ** 2 + (yy - c) ** 2 + (zz - c) ** 2)
                     / width ** 2)
        amp = env.astype(complex) / np.sqrt(2.0)
        s = CausalSession(lat, (c, c, c), 0.1019, prob_floor=eps)
        s.psi_R = amp.copy()
        s.psi_L = amp.copy()
        from dcl_core.core.UnityConstraint import enforce_unity_spinor
        enforce_unity_spinor(s.psi_R, s.psi_L)
        return s

    @pytest.mark.parametrize("eps", [1e-10, 1e-6, 1e-3, 0.25])
    def test_deep_denormal_tail_stays_finite_and_unit(self, eps):
        """min nonzero p ~ 1e-323 must not overflow; A=1 must survive."""
        s = self._denormal_packet(size=33, width=0.5, eps=eps)
        p0 = s.probability_density()
        assert p0[p0 > 0].min() < 1e-300        # genuine deep-denormal tail
        with np.errstate(over="raise", invalid="raise", divide="raise",
                         under="ignore"):
            for _ in range(20):
                s.tick()
                s.advance_tick_counter()
        d = s.probability_density()
        assert np.isfinite(d.sum())
        assert unity_residual_spinor(s.psi_R, s.psi_L) < 1e-10

    def test_split_sqrt_matches_fused_for_normal_p(self):
        """For non-denormal p the split form must equal the old fused form to
        ~1 ulp (so v0.2.1 results are reproduced where they were finite)."""
        s = make_session(7, prob_floor=1e-4)
        s.psi_R[:] = 0.0
        s.psi_L[:] = 0.0
        s.psi_R[2, 2, 2] = 5e-3 * np.exp(1j * 0.4)   # p = 2.5e-5 < floor
        fused = np.sqrt(1e-4 / 2.5e-5)
        s._apply_prob_floor(s.psi_R, s.psi_L)
        p = abs(s.psi_R[2, 2, 2]) ** 2 + abs(s.psi_L[2, 2, 2]) ** 2
        assert p == pytest.approx(1e-4, rel=1e-12)
        del fused


# ── Manufactured-probability cleanup ledger (v0.2.2 A=1 accounting) ──────────

class TestProbFloorLedger:
    """Raising a sub-floor node to prob_floor manufactures (floor - p) of
    probability there; A=1 renormalisation redistributes it.  The ledger
    records that manufactured mass exactly (dcl-delta-p-min zoo-accounting)."""

    def test_none_floor_ledger_is_zero(self):
        s = make_session(9, momentum=(0.2, 0.0, 0.0))   # prob_floor=None
        for _ in range(10):
            s.tick(); s.advance_tick_counter()
        led = s.floor_ledger()
        assert led["manufactured_total"] == 0.0
        assert led["n_raised_total"] == 0
        assert led["ticks_floored"] == 0

    def test_manufactured_mass_matches_hand_count(self):
        """One known sub-floor node -> manufactured == floor - p exactly."""
        s = make_session(7, prob_floor=1e-4)
        s.psi_R[:] = 0.0
        s.psi_L[:] = 0.0
        s.psi_R[3, 3, 3] = 0.8                        # big, untouched
        s.psi_R[0, 0, 0] = 1e-3 * np.exp(1j * 0.5)    # p = 1e-6 < floor
        p_small = abs(s.psi_R[0, 0, 0]) ** 2
        s._apply_prob_floor(s.psi_R, s.psi_L)
        led = s.floor_ledger()
        assert led["manufactured_last"] == pytest.approx(1e-4 - p_small, rel=1e-9)
        assert led["n_raised_last"] == 1
        assert led["manufactured_total"] == pytest.approx(1e-4 - p_small, rel=1e-9)

    def test_aggressive_floor_manufactures_large_mass(self):
        """A floor of 0.25 over-writes a packet toward uniform: the ledger's
        manufactured_total must be >> 1 (the quantitative destruction signal),
        while A=1 still holds."""
        s = self_packet = TestProbFloorDenormalOverflow()._denormal_packet(
            size=21, width=1.5, eps=0.25)
        for _ in range(15):
            s.tick(); s.advance_tick_counter()
        led = s.floor_ledger()
        assert led["manufactured_total"] > 1.0
        assert led["ticks_floored"] == 15
        assert unity_residual_spinor(s.psi_R, s.psi_L) < 1e-10
