"""TickScheduler: orchestrates many sessions through coupled ticks.

A single session evolves under the hop + residual machinery. The
scheduler IS the multi-session glue: it advances every registered
session by one tick, handles pairwise interactions (gauge coupling,
recoil, emission, absorption), and enforces the global parity rule
that defines bipartite time evolution.

Each tick:

    1. Determine parity from tick index (even -> RGB active, odd -> CMY).
    2. For each session: HopOperator.step(...) -> analytical psi_new.
    3. For each session: BresenhamResidual.quantise(...) -> integer N_new.
    4. Apply pairwise interactions (Coulomb, gauge phase, emission pair).
    5. (Sanity) Each session: session.assert_unity().
    6. Advance tick counter.

The scheduler does **not** perform per-session A=1 enforcement of its
own -- A=1 is a consequence of integer arithmetic in the remainder
step, not a constraint applied after the fact. The sanity check in
step 5 is a debug aid; it must pass at every tick of a correct
implementation.

See:
    docs/reference/scheduler.md -- API and event types
    docs/design/02_remainder_strategy.md -- why no global renorm step
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .hop import HopOperator
from .lattice import BipartiteLattice, TickParity
from .remainder import BresenhamResidual
from .session import DiscreteCausalSession


@dataclass
class TickScheduler:
    """Drive the simulation forward in unit ticks.

    Parameters
    ----------
    lattice : BipartiteLattice
        Shared geometry for all registered sessions. Mixing lattices
        is not supported.
    hop : HopOperator
        The Dirac evolution operator.
    sessions : list[DiscreteCausalSession]
        Registered sessions. Order matters for pairwise-interaction
        determinism.

    Attributes
    ----------
    tick : int
        Current tick counter. Even -> RGB active, odd -> CMY.
    residuals : dict[int, BresenhamResidual]
        One residual per session, keyed by session index.
    """

    lattice: BipartiteLattice
    hop: HopOperator
    sessions: list[DiscreteCausalSession] = field(default_factory=list)
    tick: int = 0
    residuals: dict[int, BresenhamResidual] = field(default_factory=dict)

    def parity_now(self) -> TickParity:
        """Return the active sublattice parity for the current tick."""
        return "even" if (self.tick % 2 == 0) else "odd"

    def register(self, session: DiscreteCausalSession) -> int:
        """Add a session; return its scheduler-local index.

        The returned index is stable for the lifetime of the session
        and is used to address its residual in `self.residuals`.
        """
        raise NotImplementedError(
            "TickScheduler.register: append session, allocate matching BresenhamResidual"
        )

    def step(self) -> None:
        """Advance every registered session by one tick.

        After this call:
          - `self.tick` is incremented by 1.
          - Each session's `(N_R, N_L, phi_R, phi_L)` reflect one tick
            of bipartite Dirac evolution.
          - Each session's `assert_unity()` would pass (A=1 holds).
          - Pairwise interactions registered for this tick have fired.
        """
        raise NotImplementedError(
            "TickScheduler.step: see docstring algorithm; implement once "
            "HopOperator and BresenhamResidual are concrete."
        )

    def run(self, n_ticks: int) -> None:
        """Convenience wrapper: call `step()` `n_ticks` times.

        For long runs, prefer driving `step()` from a loop in the
        experiment script so progress / instrumentation / early-exit
        logic is explicit at the call site.
        """
        for _ in range(n_ticks):
            self.step()
