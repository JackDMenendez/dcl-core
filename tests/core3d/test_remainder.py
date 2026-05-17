"""Tests for the Bresenham residual / token-quantisation rule.

These guard the integer-conservation property of the rounding step:
no matter what fractional targets the hop produces, the resulting
integer counts sum to exactly `n_units_total`.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="BresenhamResidual is a stub; remove skip once implemented.")


def test_quantise_preserves_total_exactly(small_shape: tuple[int, int, int]) -> None:
    """sum(N_R_int) + sum(N_L_int) == n_units_total for any fractional target."""
    import numpy as np

    from dcl_core import BipartiteLattice, BresenhamResidual

    lattice = BipartiteLattice(shape=small_shape)
    accumulator = BresenhamResidual(lattice=lattice)

    # Random fractional targets that don't sum to the integer total.
    rng = np.random.default_rng(seed=42)
    N_target_R = rng.uniform(0.0, 5.0, size=small_shape)
    N_target_L = rng.uniform(0.0, 5.0, size=small_shape)
    n_units_total = int(np.round(N_target_R.sum() + N_target_L.sum()))

    N_R_int, N_L_int = accumulator.quantise(N_target_R, N_target_L, n_units_total)
    assert int(N_R_int.sum() + N_L_int.sum()) == n_units_total


@pytest.mark.parametrize("n_ticks", [10, 100])
def test_carry_remains_bounded(small_shape: tuple[int, int, int], n_ticks: int) -> None:
    """Per-site carry stays in [0, 1) after each `quantise` call.

    Bresenham's invariant: the fractional residual never accumulates
    past one unit before being deposited.
    """
    import numpy as np

    from dcl_core import BipartiteLattice, BresenhamResidual

    lattice = BipartiteLattice(shape=small_shape)
    accumulator = BresenhamResidual(lattice=lattice)
    rng = np.random.default_rng(seed=123)

    for _ in range(n_ticks):
        N_target_R = rng.uniform(0.0, 5.0, size=small_shape)
        N_target_L = rng.uniform(0.0, 5.0, size=small_shape)
        total = int(np.round(N_target_R.sum() + N_target_L.sum()))
        accumulator.quantise(N_target_R, N_target_L, total)
        assert accumulator.carry.min() >= 0.0
        assert accumulator.carry.max() < 1.0
