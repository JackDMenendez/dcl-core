"""TickScheduler: orchestrates many sessions through coupled ticks.

A single session evolves under the hop + residual machinery. The
scheduler IS the multi-session glue: it advances every registered
session by one tick, handles pairwise interactions (gauge coupling,
recoil, emission, absorption), and enforces the global parity rule
that defines bipartite time evolution.

Each tick:

    1. Determine parity from tick index (even -> RGB active, odd -> CMY).
    2. For each session: HopOperator.step(...) -> analytical psi_new.
    3. For each session: TokenResidual.quantise(...) -> integer N_new.
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
from typing import Callable

import numpy as np

from .hop import HopOperator
from .lattice import BipartiteLattice, TickParity
from .remainder import TokenResidual
from .session import DiscreteCausalSession

# Type alias for the post-tick callback.  Receives the scheduler so the
# callback can introspect `tick` (count of completed ticks), `parity_now()`,
# `sessions`, and `residuals` -- whatever it needs without us having to
# pre-decide the argument list.
TickCallback = Callable[["TickScheduler"], None]


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

    on_tick_complete : callable, optional
        Hook fired at the end of every :meth:`step` call, after the
        tick counter has been incremented.  Signature is
        ``on_tick_complete(scheduler) -> None``; the callback can
        introspect ``scheduler.tick`` (now the count of completed
        ticks), ``scheduler.parity_now()`` (the parity of the NEXT
        tick), ``scheduler.sessions`` and ``scheduler.residuals``.
        Intended for instrumentation, per-tick observable extraction,
        or live plotting -- the engine itself does no I/O.

    Attributes
    ----------
    tick : int
        Current tick counter. Even -> RGB active, odd -> CMY.
    residuals : dict[int, TokenResidual]
        One residual per session, keyed by session index.
    """

    lattice: BipartiteLattice
    hop: HopOperator
    sessions: list[DiscreteCausalSession] = field(default_factory=list)
    tick: int = 0
    residuals: dict[int, TokenResidual] = field(default_factory=dict)
    on_tick_complete: TickCallback | None = None
    # Static background fields forwarded to every session's hop each tick
    # (v0.3.0, Peierls coupling).  `external_potential` feeds the on-site
    # (temporal / electric) phase; `vector_potential` (shape
    # `(3, *lattice.shape)`) feeds the spatial Peierls link phase.  Both
    # `None` -> the hop is gauge-free and this is the pre-v0.3.0 scheduler
    # bit-for-bit.  These are the *static-background* case (exp_03); a
    # session-sourced / dynamic field is the deferred interaction-engine
    # work (see `docs/design/05_interaction_engine.md` D1d).
    # physics: A_0 -- the on-site temporal/electric gauge phase.
    external_potential: np.ndarray | None = None
    # physics: A_i -- the spatial U(1) vector potential (Peierls link phase).
    vector_potential: np.ndarray | None = None

    def parity_now(self) -> TickParity:
        """Return the active sublattice parity for the current tick."""
        return "even" if (self.tick % 2 == 0) else "odd"

    def register(self, session: DiscreteCausalSession) -> int:
        """Add a session; return its scheduler-local index.

        The returned index is stable for the lifetime of the session
        and is used to address its residual in `self.residuals`.

        Validates that ``session.lattice`` is the same object as the
        scheduler's lattice (identity check, not just shape match) --
        mixing lattices is not supported, and silently allowing it
        would surface as a confusing shape mismatch in the hop.
        """
        if session.lattice is not self.lattice:
            raise ValueError(
                "session.lattice must be the same object as the scheduler's "
                "lattice; mixing lattices is not supported"
            )
        idx = len(self.sessions)
        self.sessions.append(session)
        self.residuals[idx] = TokenResidual(lattice=self.lattice)
        return idx

    def step(self) -> None:
        """Advance every registered session by one tick.

        After this call:
          - `self.tick` is incremented by 1.
          - Each session's `(N_RGB, N_CMY, phi_RGB, phi_CMY)` reflect one tick
            of bipartite Dirac evolution.
          - Each session's `assert_unity()` would pass (A=1 holds).
          - Pairwise interactions registered for this tick have fired
            (none in v0.1.0 -- sessions evolve independently).

        Pairwise / multi-session interactions (Coulomb, gauge phase,
        emission pair) are deferred to v0.2.0; v0.1.0 ships the
        scheduler skeleton in which every registered session is
        advanced independently.  The hook for those interactions is
        between the hop and the quantisation step (so they see the
        analytical psi_new but write before integer counts are
        committed).
        """
        parity = self.parity_now()
        for idx, session in enumerate(self.sessions):
            # 1.  Hop -> analytical complex amplitudes (snapshot; no mutation).
            # Static background fields (None -> gauge-free, bit-for-bit).
            psi_R_new, psi_L_new = self.hop.step(
                session,
                parity,
                external_potential=self.external_potential,
                vector_potential=self.vector_potential,
            )

            # 2.  Renormalise `psi_new` to unit norm before the
            # integer quantisation.  The half-step hop formula in
            # `HopOperator.step` is NOT natively unitary -- only one
            # chirality moves per tick, so `sum(|psi|^2)` can drift
            # by O(0.3) per tick.  Paper~I handles this via its
            # `UnityConstraint.enforce_unity_spinor`; the integer
            # engine adopts the same fix at this boundary so the
            # residual's rebalance only absorbs O(eps) float
            # roundoff rather than O(n_units) non-unitarity (which
            # would overflow the lattice when the deficit exceeds
            # the site count).  A natively-unitary hop is a v0.2.0
            # design knob; for now this float-renorm preserves
            # Paper~I-equivalent dynamics so the discrete-vs-continuous
            # comparison in Paper III is clean.
            norm_sq = float(
                np.sum(np.abs(psi_R_new) ** 2)
                + np.sum(np.abs(psi_L_new) ** 2)
            )
            if norm_sq > 0.0:
                inv_norm = 1.0 / np.sqrt(norm_sq)
                psi_R_new = psi_R_new * inv_norm
                psi_L_new = psi_L_new * inv_norm

            # Fractional token targets sum to ~`n_units` (modulo eps).
            N_target_RGB = np.abs(psi_R_new) ** 2 * session.n_units
            N_target_CMY = np.abs(psi_L_new) ** 2 * session.n_units

            # 3.  Integer quantisation.
            residual = self.residuals[idx]
            N_RGB_int, N_CMY_int = residual.quantise(
                N_target_RGB, N_target_CMY, session.n_units
            )

            # 4.  (pairwise interactions go here; v0.2.0+).

            # 5.  Write back.  Phases are the analytical hop's angles;
            # the integer N supplies the amplitude.
            session.N_RGB[...] = N_RGB_int
            session.N_CMY[...] = N_CMY_int
            session.phi_RGB[...] = np.angle(psi_R_new)
            session.phi_CMY[...] = np.angle(psi_L_new)

            # 6.  Cheap sanity check.  This should never fire if
            # `quantise` is correct.
            session.assert_unity()

        self.tick += 1

        # 7.  Instrumentation hook.  Fires once per `step` (not once per
        # session), after the tick counter has advanced.
        if self.on_tick_complete is not None:
            self.on_tick_complete(self)

    def run(self, n_ticks: int) -> None:
        """Convenience wrapper: call `step()` `n_ticks` times.

        For long runs, prefer driving `step()` from a loop in the
        experiment script so progress / instrumentation / early-exit
        logic is explicit at the call site.
        """
        for _ in range(n_ticks):
            self.step()
