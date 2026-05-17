"""Test A=1 conservation as integer equality (no float tolerance).

The framework's foundational claim is that probability is conserved
exactly, by integer arithmetic. These tests guard that claim: every
tick of every operator must leave `sum(N(x)) == n_units` exactly.

If any test here fails, the regression is structural -- something in
the hop / remainder / scheduler chain has slipped back into floating-
point renormalisation. Fix the underlying code, do not loosen these
tests.
"""

from __future__ import annotations

import pytest

# Tests are skipped at the module level until the core modules are
# implemented. Remove the skip once `lattice`, `session`, `hop`, and
# `remainder` are concrete.
pytestmark = pytest.mark.skip(reason="Core modules are stubs; remove this skip once implemented.")


def test_session_init_has_exact_token_total(small_shape: tuple[int, int, int]) -> None:
    """A newly-initialised session has exactly n_units tokens."""
    from dcl_core import BipartiteLattice, DiscreteCausalSession

    lattice = BipartiteLattice(shape=small_shape)
    n_units = 1_000_000
    session = DiscreteCausalSession(lattice=lattice, n_units=n_units, omega=0.1)
    assert session.total_tokens() == n_units


def test_session_total_invariant_under_hop(small_shape: tuple[int, int, int]) -> None:
    """One hop preserves the token total exactly."""
    from dcl_core import BipartiteLattice, DiscreteCausalSession, HopOperator, TickScheduler

    lattice = BipartiteLattice(shape=small_shape)
    session = DiscreteCausalSession(lattice=lattice, n_units=1_000_000, omega=0.1)
    scheduler = TickScheduler(lattice=lattice, hop=HopOperator(lattice=lattice))
    scheduler.register(session)

    initial = session.total_tokens()
    scheduler.step()
    assert session.total_tokens() == initial, (
        f"A=1 violated after 1 tick: {session.total_tokens()} vs initial {initial}"
    )


@pytest.mark.parametrize("n_ticks", [10, 100, 1000])
def test_session_total_invariant_over_many_ticks(
    small_shape: tuple[int, int, int], n_ticks: int
) -> None:
    """A=1 holds exactly across `n_ticks` consecutive ticks.

    No accumulating drift, no slow integer-loss bug.
    """
    from dcl_core import BipartiteLattice, DiscreteCausalSession, HopOperator, TickScheduler

    lattice = BipartiteLattice(shape=small_shape)
    session = DiscreteCausalSession(lattice=lattice, n_units=1_000_000, omega=0.1)
    scheduler = TickScheduler(lattice=lattice, hop=HopOperator(lattice=lattice))
    scheduler.register(session)

    initial = session.total_tokens()
    for _ in range(n_ticks):
        scheduler.step()
        assert session.total_tokens() == initial, (
            f"A=1 violated at tick {scheduler.tick}"
        )


def test_assert_unity_raises_on_corruption(small_shape: tuple[int, int, int]) -> None:
    """`session.assert_unity()` raises if state is hand-corrupted.

    Sanity check on the sanity check: if we delete a token from N_R,
    assert_unity should notice.
    """
    from dcl_core import BipartiteLattice, DiscreteCausalSession

    lattice = BipartiteLattice(shape=small_shape)
    session = DiscreteCausalSession(lattice=lattice, n_units=100, omega=0.1)
    # Manually deduct one token. assert_unity should now fail.
    session.N_R.flat[0] = max(0, int(session.N_R.flat[0]) - 1)
    with pytest.raises(AssertionError):
        session.assert_unity()
