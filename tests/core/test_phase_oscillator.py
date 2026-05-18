"""
tests/test_phase_oscillator.py

Unit tests for PhaseOscillator — the U(1) internal clock.

Theory correspondence:
  phase_oscillator.omega  IS the rest mass (instruction overhead).
  phase_oscillator.phase  IS the accumulated internal clock angle.
  amplitude               IS the U(1) group element exp(i*phase).
  advance()               IS left-multiplication by exp(i*omega).
  phase_cost()            IS the local Hamiltonian H = omega + V.
"""

import numpy as np
import pytest
from dcl_core.core.PhaseOscillator import PhaseOscillator


class TestPhaseOscillatorInit:

    def test_omega_stored(self):
        osc = PhaseOscillator(frequency=0.5)
        assert osc.omega == pytest.approx(0.5)

    def test_phase_initialised_to_zero(self):
        osc = PhaseOscillator(frequency=0.3)
        assert osc.phase == pytest.approx(0.0)

    def test_amplitude_at_init_is_one(self):
        """exp(i*0) = 1, so |amplitude| = 1 at construction."""
        osc = PhaseOscillator(frequency=1.0)
        assert abs(osc.amplitude) == pytest.approx(1.0)

    def test_amplitude_at_init_value(self):
        osc = PhaseOscillator(frequency=0.7)
        assert osc.amplitude == pytest.approx(1.0 + 0.0j)


class TestPhaseOscillatorAmplitude:
    """amplitude IS the U(1) group element -- always on the unit circle."""

    def test_amplitude_on_unit_circle_after_many_advances(self):
        osc = PhaseOscillator(frequency=0.123)
        for _ in range(1000):
            osc.advance()
        assert abs(osc.amplitude) == pytest.approx(1.0, abs=1e-12)

    def test_amplitude_equals_exp_i_phase(self):
        osc = PhaseOscillator(frequency=0.5)
        for _ in range(7):
            osc.advance()
        expected = np.exp(1j * osc.phase)
        assert osc.amplitude == pytest.approx(expected)


class TestPhaseOscillatorAdvance:
    """advance() increments phase by omega and wraps at 2*pi."""

    def test_phase_increments_by_omega(self):
        osc = PhaseOscillator(frequency=0.4)
        osc.advance()
        assert osc.phase == pytest.approx(0.4)

    def test_phase_accumulates(self):
        osc = PhaseOscillator(frequency=0.2)
        for _ in range(5):
            osc.advance()
        assert osc.phase == pytest.approx(1.0)

    def test_phase_wraps_at_2pi(self):
        """Phase stays in [0, 2*pi)."""
        osc = PhaseOscillator(frequency=np.pi)
        osc.advance()   # phase = pi
        osc.advance()   # phase = 2*pi -> wraps to 0
        assert osc.phase == pytest.approx(0.0, abs=1e-12)

    def test_phase_stays_in_range(self):
        osc = PhaseOscillator(frequency=0.7)
        for _ in range(200):
            osc.advance()
        assert 0.0 <= osc.phase < 2 * np.pi

    def test_zero_frequency_phase_stays_zero(self):
        osc = PhaseOscillator(frequency=0.0)
        for _ in range(50):
            osc.advance()
        assert osc.phase == pytest.approx(0.0)


class TestPhaseCost:
    """phase_cost() IS the local Hamiltonian H = omega + V."""

    def test_zero_potential(self):
        osc = PhaseOscillator(frequency=0.3)
        assert osc.phase_cost(0.0) == pytest.approx(0.3)

    def test_nonzero_potential(self):
        osc = PhaseOscillator(frequency=0.3)
        assert osc.phase_cost(0.7) == pytest.approx(1.0)

    def test_negative_potential(self):
        """Attractive wells produce negative phase cost (bound-state case)."""
        osc = PhaseOscillator(frequency=0.5)
        cost = osc.phase_cost(-0.5)
        assert cost == pytest.approx(0.0)

    def test_linearity(self):
        osc = PhaseOscillator(frequency=0.4)
        assert osc.phase_cost(0.1) + osc.phase_cost(0.2) != osc.phase_cost(0.3)
        # Check the formula directly: omega + V
        v = 2.3
        assert osc.phase_cost(v) == pytest.approx(osc.omega + v)


class TestPhaseShift:
    """phase_shift() IS exp(i*H) -- the U(1) evolution operator."""

    def test_magnitude_is_one(self):
        osc = PhaseOscillator(frequency=0.5)
        ps = osc.phase_shift(0.3)
        assert abs(ps) == pytest.approx(1.0)

    def test_value(self):
        osc = PhaseOscillator(frequency=0.5)
        v = 0.3
        expected = np.exp(1j * (0.5 + 0.3))
        assert osc.phase_shift(v) == pytest.approx(expected)

    def test_zero_hamiltonian_returns_one(self):
        osc = PhaseOscillator(frequency=0.0)
        ps = osc.phase_shift(0.0)
        assert ps == pytest.approx(1.0 + 0.0j)

    def test_pi_hamiltonian_returns_neg_one(self):
        osc = PhaseOscillator(frequency=np.pi)
        ps = osc.phase_shift(0.0)
        assert ps == pytest.approx(-1.0 + 0.0j, abs=1e-10)
