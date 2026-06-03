"""Tests for the TickScheduler.

The conservation tests in `test_conservation.py` already exercise the
scheduler's *output* (token total invariant across many ticks).  This
file pins down the scheduler's *contract*: registration semantics,
tick-counter / parity behaviour, lattice-mismatch validation, and
multi-session bookkeeping.
"""

from __future__ import annotations

import pytest


def test_register_returns_increasing_indices(
    small_shape: tuple[int, int, int],
) -> None:
    """Each `register` call returns the next sequential index (0, 1, 2, ...)."""
    from dcl_core.core3d import (
        BipartiteLattice,
        DiscreteCausalSession,
        HopOperator,
        TickScheduler,
    )

    lattice = BipartiteLattice(shape=small_shape)
    scheduler = TickScheduler(lattice=lattice, hop=HopOperator(lattice=lattice))

    s0 = DiscreteCausalSession.delta_at(
        lattice, n_units=100, omega=0.1, position=(0, 0, 0)
    )
    s1 = DiscreteCausalSession.delta_at(
        lattice, n_units=200, omega=0.2, position=(1, 1, 1)
    )
    s2 = DiscreteCausalSession.delta_at(
        lattice, n_units=300, omega=0.3, position=(2, 2, 2)
    )
    assert scheduler.register(s0) == 0
    assert scheduler.register(s1) == 1
    assert scheduler.register(s2) == 2


def test_register_allocates_one_residual_per_session(
    small_shape: tuple[int, int, int],
) -> None:
    """Each registered session gets its own `TokenResidual` in `residuals`."""
    from dcl_core.core3d import (
        BipartiteLattice,
        TokenResidual,
        DiscreteCausalSession,
        HopOperator,
        TickScheduler,
    )

    lattice = BipartiteLattice(shape=small_shape)
    scheduler = TickScheduler(lattice=lattice, hop=HopOperator(lattice=lattice))

    session = DiscreteCausalSession.delta_at(
        lattice, n_units=100, omega=0.1, position=(0, 0, 0)
    )
    idx = scheduler.register(session)

    assert idx in scheduler.residuals
    assert isinstance(scheduler.residuals[idx], TokenResidual)
    # The residual lives on the same lattice as the scheduler.
    assert scheduler.residuals[idx].lattice is lattice


def test_register_rejects_lattice_mismatch(
    small_shape: tuple[int, int, int],
) -> None:
    """`register` raises if the session's lattice is a different object."""
    from dcl_core.core3d import (
        BipartiteLattice,
        DiscreteCausalSession,
        HopOperator,
        TickScheduler,
    )

    lattice_a = BipartiteLattice(shape=small_shape)
    lattice_b = BipartiteLattice(shape=small_shape)  # same shape, different object
    scheduler = TickScheduler(lattice=lattice_a, hop=HopOperator(lattice=lattice_a))
    session_on_b = DiscreteCausalSession.delta_at(
        lattice_b, n_units=100, omega=0.1, position=(0, 0, 0)
    )
    with pytest.raises(ValueError):
        scheduler.register(session_on_b)


def test_parity_now_alternates_with_tick(small_shape: tuple[int, int, int]) -> None:
    """`parity_now` reports `even` for tick%2==0 and `odd` otherwise."""
    from dcl_core.core3d import BipartiteLattice, HopOperator, TickScheduler

    lattice = BipartiteLattice(shape=small_shape)
    scheduler = TickScheduler(lattice=lattice, hop=HopOperator(lattice=lattice))
    # Sweep through a handful of ticks.
    for tick in range(6):
        scheduler.tick = tick
        expected = "even" if tick % 2 == 0 else "odd"
        assert scheduler.parity_now() == expected


def test_step_increments_tick(small_shape: tuple[int, int, int]) -> None:
    """Each `step()` advances `tick` by exactly one, even with no sessions."""
    from dcl_core.core3d import BipartiteLattice, HopOperator, TickScheduler

    lattice = BipartiteLattice(shape=small_shape)
    scheduler = TickScheduler(lattice=lattice, hop=HopOperator(lattice=lattice))
    assert scheduler.tick == 0
    scheduler.step()
    assert scheduler.tick == 1
    scheduler.step()
    assert scheduler.tick == 2


def test_step_with_no_sessions_is_a_noop(small_shape: tuple[int, int, int]) -> None:
    """An empty scheduler's `step()` just advances the tick counter.

    No sessions to evolve, no residuals to update; the per-session
    loop body is skipped entirely.  Cheap regression guard against
    bugs that would crash on the empty list.
    """
    from dcl_core.core3d import BipartiteLattice, HopOperator, TickScheduler

    lattice = BipartiteLattice(shape=small_shape)
    scheduler = TickScheduler(lattice=lattice, hop=HopOperator(lattice=lattice))
    for _ in range(5):
        scheduler.step()
    assert scheduler.tick == 5
    assert scheduler.sessions == []
    assert scheduler.residuals == {}


def test_run_calls_step_n_times(small_shape: tuple[int, int, int]) -> None:
    """`run(n)` is `step()` called `n` times."""
    from dcl_core.core3d import (
        BipartiteLattice,
        DiscreteCausalSession,
        HopOperator,
        TickScheduler,
    )

    lattice = BipartiteLattice(shape=small_shape)
    session = DiscreteCausalSession.delta_at(
        lattice, n_units=1_000, omega=0.1, position=(0, 0, 0)
    )
    scheduler = TickScheduler(lattice=lattice, hop=HopOperator(lattice=lattice))
    scheduler.register(session)

    scheduler.run(7)
    assert scheduler.tick == 7
    # A=1 still holds after the run.
    assert session.total_tokens() == 1_000


def test_on_tick_complete_fires_with_scheduler(
    small_shape: tuple[int, int, int],
) -> None:
    """The callback is called once per `step()`, with the scheduler as arg."""
    from dcl_core.core3d import (
        BipartiteLattice,
        DiscreteCausalSession,
        HopOperator,
        TickScheduler,
    )

    lattice = BipartiteLattice(shape=small_shape)
    session = DiscreteCausalSession.delta_at(
        lattice, n_units=100, omega=0.1, position=(0, 0, 0)
    )

    calls: list[int] = []

    def hook(sched: TickScheduler) -> None:
        # At the moment the callback fires, `tick` has already been
        # incremented to reflect the just-completed tick.
        calls.append(sched.tick)

    scheduler = TickScheduler(
        lattice=lattice, hop=HopOperator(lattice=lattice), on_tick_complete=hook
    )
    scheduler.register(session)
    scheduler.step()
    scheduler.step()
    scheduler.step()
    assert calls == [1, 2, 3]


def test_on_tick_complete_default_is_none_and_optional(
    small_shape: tuple[int, int, int],
) -> None:
    """Schedulers without a callback work exactly as before."""
    from dcl_core.core3d import BipartiteLattice, HopOperator, TickScheduler

    lattice = BipartiteLattice(shape=small_shape)
    scheduler = TickScheduler(lattice=lattice, hop=HopOperator(lattice=lattice))
    assert scheduler.on_tick_complete is None
    scheduler.step()  # no error
    assert scheduler.tick == 1


def test_step_evolves_multiple_independent_sessions(
    small_shape: tuple[int, int, int],
) -> None:
    """Multiple registered sessions each conserve their own ``n_units``.

    v0.1.0 has no pairwise interactions, so two sessions with
    different ``n_units`` evolve independently and each preserves
    its own A=1 contract.
    """
    from dcl_core.core3d import (
        BipartiteLattice,
        DiscreteCausalSession,
        HopOperator,
        TickScheduler,
    )

    lattice = BipartiteLattice(shape=small_shape)
    s_small = DiscreteCausalSession.delta_at(
        lattice, n_units=1_000, omega=0.1, position=(0, 0, 0)
    )
    s_big = DiscreteCausalSession.delta_at(
        lattice, n_units=10_000, omega=0.5, position=(4, 4, 4)
    )
    scheduler = TickScheduler(lattice=lattice, hop=HopOperator(lattice=lattice))
    scheduler.register(s_small)
    scheduler.register(s_big)

    for _ in range(20):
        scheduler.step()

    assert s_small.total_tokens() == 1_000
    assert s_big.total_tokens() == 10_000
