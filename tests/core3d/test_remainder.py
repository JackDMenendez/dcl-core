"""Tests for the Bresenham residual / token-quantisation rule.

These guard the integer-conservation property of the rounding step:
no matter what fractional targets the hop produces, the resulting
integer counts sum to exactly `n_units_total`.
"""

from __future__ import annotations

import pytest


def test_quantise_preserves_total_exactly(small_shape: tuple[int, int, int]) -> None:
    """sum(N_RGB_int) + sum(N_CMY_int) == n_units_total for any fractional target."""
    import numpy as np

    from dcl_core.core3d import BipartiteLattice, TokenResidual

    lattice = BipartiteLattice(shape=small_shape)
    accumulator = TokenResidual(lattice=lattice)

    # Random fractional targets that don't sum to the integer total.
    rng = np.random.default_rng(seed=42)
    N_target_RGB = rng.uniform(0.0, 5.0, size=small_shape)
    N_target_CMY = rng.uniform(0.0, 5.0, size=small_shape)
    n_units_total = int(np.round(N_target_RGB.sum() + N_target_CMY.sum()))

    N_RGB_int, N_CMY_int = accumulator.quantise(N_target_RGB, N_target_CMY, n_units_total)
    assert int(N_RGB_int.sum() + N_CMY_int.sum()) == n_units_total


@pytest.mark.parametrize("n_ticks", [10, 100])
def test_carry_remains_bounded(small_shape: tuple[int, int, int], n_ticks: int) -> None:
    """Per-site carry stays in [0, 1) after each `quantise` call.

    Bresenham's invariant: the fractional residual never accumulates
    past one unit before being deposited.
    """
    import numpy as np

    from dcl_core.core3d import BipartiteLattice, TokenResidual

    lattice = BipartiteLattice(shape=small_shape)
    accumulator = TokenResidual(lattice=lattice)
    rng = np.random.default_rng(seed=123)

    for _ in range(n_ticks):
        N_target_RGB = rng.uniform(0.0, 5.0, size=small_shape)
        N_target_CMY = rng.uniform(0.0, 5.0, size=small_shape)
        total = int(np.round(N_target_RGB.sum() + N_target_CMY.sum()))
        accumulator.quantise(N_target_RGB, N_target_CMY, total)
        assert accumulator.carry.min() >= 0.0
        assert accumulator.carry.max() < 1.0


def test_quantise_returns_nonnegative_int_arrays(
    small_shape: tuple[int, int, int],
) -> None:
    """Output token counts are int64 arrays with no negative entries."""
    import numpy as np

    from dcl_core.core3d import BipartiteLattice, TokenResidual

    lattice = BipartiteLattice(shape=small_shape)
    accumulator = TokenResidual(lattice=lattice)

    rng = np.random.default_rng(seed=7)
    N_target_RGB = rng.uniform(0.0, 5.0, size=small_shape)
    N_target_CMY = rng.uniform(0.0, 5.0, size=small_shape)
    total = int(np.round(N_target_RGB.sum() + N_target_CMY.sum()))

    N_RGB_int, N_CMY_int = accumulator.quantise(N_target_RGB, N_target_CMY, total)
    assert N_RGB_int.dtype == np.int64
    assert N_CMY_int.dtype == np.int64
    assert int(N_RGB_int.min()) >= 0
    assert int(N_CMY_int.min()) >= 0


def test_first_tick_carry_starts_at_zero(small_shape: tuple[int, int, int]) -> None:
    """Fresh `TokenResidual` has carry == 0 everywhere before any call."""
    import numpy as np

    from dcl_core.core3d import BipartiteLattice, TokenResidual

    lattice = BipartiteLattice(shape=small_shape)
    accumulator = TokenResidual(lattice=lattice)
    # Shape is (2, *lattice.shape): R at index 0, L at index 1.
    assert accumulator.carry.shape == (2, *small_shape)
    np.testing.assert_array_equal(accumulator.carry, 0.0)
    # And drift_magnitude reports zero.
    assert accumulator.drift_magnitude() == 0.0


def test_carry_persists_between_ticks(small_shape: tuple[int, int, int]) -> None:
    """Successive `quantise` calls accumulate fractional bits into `carry`.

    Feed identical sub-1 fractional targets every tick; after a few
    ticks the carries should be measurably nonzero somewhere -- this
    is the whole point of the persistent residual.
    """
    import numpy as np

    from dcl_core.core3d import BipartiteLattice, TokenResidual

    lattice = BipartiteLattice(shape=small_shape)
    accumulator = TokenResidual(lattice=lattice)

    # Each site asks for 0.3 of a token per tick; floor is always 0.
    # Carry grows by 0.3 per tick until it crosses 1 and deposits.
    N_target_RGB = np.full(small_shape, 0.3, dtype=np.float64)
    N_target_CMY = np.full(small_shape, 0.3, dtype=np.float64)
    total = int(np.round(N_target_RGB.sum() + N_target_CMY.sum()))

    # After one tick the carry should be in [0, 0.3] roughly (small
    # adjustments from rebalance).  In particular, not identically zero.
    accumulator.quantise(N_target_RGB, N_target_CMY, total)
    assert float(accumulator.carry.max()) > 0.0
