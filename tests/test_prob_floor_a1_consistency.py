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
