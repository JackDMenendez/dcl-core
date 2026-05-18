"""
tests/test_unity_constraint.py

Unit tests for UnityConstraint — the A=1 probability conservation axiom.

Theory correspondence:
  enforce_unity()         normalises a scalar field so sum(|psi|^2) = 1.
  enforce_unity_spinor()  normalises a Dirac spinor so sum(|R|^2+|L|^2) = 1.
  unity_residual()        measures deviation from A=1.
  is_unity()              boolean A=1 test within tolerance.
  enforce_joint_unity()   normalises multiple spinors jointly (pair emission).
"""

import numpy as np
import pytest
from dcl_core.core.UnityConstraint import (
    enforce_unity,
    unity_residual,
    is_unity,
    enforce_unity_spinor,
    unity_residual_spinor,
    enforce_joint_unity,
)


# ── enforce_unity (scalar) ────────────────────────────────────────────────────

class TestEnforceUnity:

    def test_already_normalised_unchanged(self):
        psi = np.array([1.0 / np.sqrt(2), 1.0 / np.sqrt(2)], dtype=complex)
        result = enforce_unity(psi)
        np.testing.assert_allclose(np.sum(np.abs(result)**2), 1.0, atol=1e-14)

    def test_normalises_flat_array(self):
        psi = np.ones(4, dtype=complex)
        result = enforce_unity(psi)
        assert np.sum(np.abs(result)**2) == pytest.approx(1.0)

    def test_normalises_3d_array(self):
        psi = np.random.random((5, 5, 5)) + 1j * np.random.random((5, 5, 5))
        enforce_unity(psi)
        assert np.sum(np.abs(psi)**2) == pytest.approx(1.0)

    def test_modifies_in_place(self):
        """enforce_unity modifies the array in-place and returns the same object."""
        psi = np.array([2.0 + 0j, 0.0 + 0j])
        result = enforce_unity(psi)
        assert result is psi

    def test_shape_preserved(self):
        psi = np.ones((3, 4, 5), dtype=complex)
        enforce_unity(psi)
        assert psi.shape == (3, 4, 5)

    def test_zero_norm_raises(self):
        psi = np.zeros(5, dtype=complex)
        with pytest.raises(RuntimeError):
            enforce_unity(psi)

    def test_complex_field(self):
        psi = np.array([1.0 + 1.0j, 2.0 - 0.5j, 0.3 + 0.0j])
        enforce_unity(psi)
        assert np.sum(np.abs(psi)**2) == pytest.approx(1.0)


# ── unity_residual ────────────────────────────────────────────────────────────

class TestUnityResidual:

    def test_perfect_norm_zero_residual(self):
        psi = np.array([1.0 / np.sqrt(3), 1.0 / np.sqrt(3), 1.0 / np.sqrt(3)],
                       dtype=complex)
        assert unity_residual(psi) == pytest.approx(0.0, abs=1e-14)

    def test_unnormalised_positive_residual(self):
        psi = np.ones(4, dtype=complex)
        assert unity_residual(psi) > 0.0

    def test_residual_after_normalisation_zero(self):
        psi = np.random.random(10).astype(complex)
        enforce_unity(psi)
        assert unity_residual(psi) < 1e-14

    def test_zero_array_residual(self):
        psi = np.zeros(5, dtype=complex)
        assert unity_residual(psi) == pytest.approx(1.0)


# ── is_unity ──────────────────────────────────────────────────────────────────

class TestIsUnity:

    def test_true_for_normalised(self):
        psi = np.array([1.0 + 0j])
        assert bool(is_unity(psi)) is True

    def test_false_for_unnormalised(self):
        psi = np.ones(4, dtype=complex)  # sum(|psi|^2) = 4
        assert bool(is_unity(psi)) is False

    def test_tolerance_respected(self):
        psi = np.array([1.0 + 0j])
        psi[0] += 1e-6  # small violation
        assert bool(is_unity(psi, tolerance=1e-4)) is True
        assert bool(is_unity(psi, tolerance=1e-8)) is False


# ── enforce_unity_spinor ──────────────────────────────────────────────────────

class TestEnforceUnitySpinor:

    def _make_spinor(self, size=5):
        shape = (size, size, size)
        psi_R = np.random.random(shape) + 1j * np.random.random(shape)
        psi_L = np.random.random(shape) + 1j * np.random.random(shape)
        return psi_R, psi_L

    def test_normalised_spinor_has_unit_norm(self):
        psi_R, psi_L = self._make_spinor()
        enforce_unity_spinor(psi_R, psi_L)
        total = np.sum(np.abs(psi_R)**2 + np.abs(psi_L)**2)
        assert total == pytest.approx(1.0)

    def test_modifies_in_place(self):
        psi_R, psi_L = self._make_spinor()
        orig_id_R = id(psi_R)
        orig_id_L = id(psi_L)
        psi_R_out, psi_L_out = enforce_unity_spinor(psi_R, psi_L)
        assert id(psi_R_out) == orig_id_R
        assert id(psi_L_out) == orig_id_L

    def test_returns_both_components(self):
        psi_R, psi_L = self._make_spinor(3)
        result = enforce_unity_spinor(psi_R, psi_L)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_zero_spinor_raises(self):
        shape = (4, 4, 4)
        psi_R = np.zeros(shape, dtype=complex)
        psi_L = np.zeros(shape, dtype=complex)
        with pytest.raises(RuntimeError):
            enforce_unity_spinor(psi_R, psi_L)

    def test_shape_preserved(self):
        shape = (3, 4, 5)
        psi_R = np.ones(shape, dtype=complex)
        psi_L = np.ones(shape, dtype=complex)
        enforce_unity_spinor(psi_R, psi_L)
        assert psi_R.shape == shape
        assert psi_L.shape == shape

    def test_relative_ratio_preserved(self):
        """The ratio |psi_R|/|psi_L| is unchanged by normalisation."""
        shape = (3, 3, 3)
        psi_R = 2.0 * np.ones(shape, dtype=complex)
        psi_L = 1.0 * np.ones(shape, dtype=complex)
        enforce_unity_spinor(psi_R, psi_L)
        # Both divided by same norm; ratio preserved
        ratio = np.abs(psi_R[0, 0, 0]) / np.abs(psi_L[0, 0, 0])
        assert ratio == pytest.approx(2.0)

    def test_already_normalised_no_change(self):
        shape = (4, 4, 4)
        amp = 1.0 / np.sqrt(2 * 4**3)
        psi_R = amp * np.ones(shape, dtype=complex)
        psi_L = amp * np.ones(shape, dtype=complex)
        psi_R_copy = psi_R.copy()
        psi_L_copy = psi_L.copy()
        enforce_unity_spinor(psi_R, psi_L)
        np.testing.assert_allclose(psi_R, psi_R_copy, atol=1e-14)
        np.testing.assert_allclose(psi_L, psi_L_copy, atol=1e-14)


# ── unity_residual_spinor ─────────────────────────────────────────────────────

class TestUnityResidualSpinor:

    def test_zero_for_normalised_spinor(self):
        shape = (3, 3, 3)
        amp = 1.0 / np.sqrt(2 * 27)
        psi_R = amp * np.ones(shape, dtype=complex)
        psi_L = amp * np.ones(shape, dtype=complex)
        assert unity_residual_spinor(psi_R, psi_L) == pytest.approx(0.0, abs=1e-14)

    def test_positive_for_unnormalised_spinor(self):
        shape = (3, 3, 3)
        psi_R = np.ones(shape, dtype=complex)
        psi_L = np.ones(shape, dtype=complex)
        assert unity_residual_spinor(psi_R, psi_L) > 0.0

    def test_small_after_normalisation(self):
        shape = (5, 5, 5)
        psi_R = np.random.random(shape).astype(complex)
        psi_L = np.random.random(shape).astype(complex)
        enforce_unity_spinor(psi_R, psi_L)
        assert unity_residual_spinor(psi_R, psi_L) < 1e-14


# ── enforce_joint_unity ───────────────────────────────────────────────────────

class TestEnforceJointUnity:
    """Joint normalisation for a pair of spinors (emission/absorption case)."""

    def test_joint_norm_is_one(self):
        shape = (4, 4, 4)
        psi_R1 = np.random.random(shape).astype(complex)
        psi_L1 = np.random.random(shape).astype(complex)
        psi_R2 = np.random.random(shape).astype(complex)
        psi_L2 = np.random.random(shape).astype(complex)
        enforce_joint_unity([(psi_R1, psi_L1), (psi_R2, psi_L2)])
        total = (np.sum(np.abs(psi_R1)**2 + np.abs(psi_L1)**2) +
                 np.sum(np.abs(psi_R2)**2 + np.abs(psi_L2)**2))
        assert total == pytest.approx(1.0)

    def test_single_spinor_acts_like_enforce_unity_spinor(self):
        shape = (4, 4, 4)
        psi_R = np.random.random(shape).astype(complex) * 3.0
        psi_L = np.random.random(shape).astype(complex) * 2.0
        enforce_joint_unity([(psi_R, psi_L)])
        total = np.sum(np.abs(psi_R)**2 + np.abs(psi_L)**2)
        assert total == pytest.approx(1.0)

    def test_zero_joint_amplitude_raises(self):
        shape = (3, 3, 3)
        psi_R = np.zeros(shape, dtype=complex)
        psi_L = np.zeros(shape, dtype=complex)
        with pytest.raises(RuntimeError):
            enforce_joint_unity([(psi_R, psi_L)])

    def test_relative_weight_between_spinors_preserved(self):
        """After joint normalisation, the ratio of the two spinors' norms is unchanged."""
        shape = (4, 4, 4)
        psi_R1 = np.ones(shape, dtype=complex) * 3.0
        psi_L1 = np.ones(shape, dtype=complex) * 3.0
        psi_R2 = np.ones(shape, dtype=complex) * 1.0
        psi_L2 = np.ones(shape, dtype=complex) * 1.0
        norm1_before = np.sqrt(np.sum(np.abs(psi_R1)**2 + np.abs(psi_L1)**2))
        norm2_before = np.sqrt(np.sum(np.abs(psi_R2)**2 + np.abs(psi_L2)**2))
        ratio_before = norm1_before / norm2_before
        enforce_joint_unity([(psi_R1, psi_L1), (psi_R2, psi_L2)])
        norm1_after = np.sqrt(np.sum(np.abs(psi_R1)**2 + np.abs(psi_L1)**2))
        norm2_after = np.sqrt(np.sum(np.abs(psi_R2)**2 + np.abs(psi_L2)**2))
        ratio_after = norm1_after / norm2_after
        assert ratio_after == pytest.approx(ratio_before, rel=1e-12)
