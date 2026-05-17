"""Discrete causal session: integer-token state on the bipartite lattice.

A session IS one quantum-mechanical object (electron, photon, proton, ...)
modelled as a budget of `N_units` indistinguishable probability tokens
distributed across lattice sites. A=1 is the integer equality:

    sum_x N(x) == N_units            (exact, no float tolerance)

The session's amplitude at a site is implicit:

    psi(x) = sqrt(N(x) / N_units) * exp(i * phi(x))

where `phi(x)` is a continuous U(1) phase carried per-site, per-chirality.
The Dirac spinor structure is realised by two chirality fields psi_R, psi_L
co-resident on the lattice.

The "Planck of probability" is `epsilon_P = 1 / N_units`. Continuum-limit
v1.0-style dynamics are recovered as `N_units -> infinity`.

See:
    docs/design/01_planck_of_probability.md -- rationale and consequences
    docs/reference/session.md -- API reference and examples
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .lattice import BipartiteLattice

Chirality = Literal["R", "L"]


@dataclass
class DiscreteCausalSession:
    """One causal session: integer tokens + continuous phase on a lattice.

    Parameters
    ----------
    lattice : BipartiteLattice
        The geometry this session lives on.
    n_units : int
        The probability budget. Sum of `N(x)` across all sites is
        exactly this integer at every tick (A=1).
    omega : float
        The session's internal clock frequency. This IS the rest-mass
        parameter via E = N_units * omega (in framework units).
        omega=0 -> massless (photon).
        omega=pi/2 -> maximal Zitterbewegung.

    Notes
    -----
    Token distributions `N_R(x)`, `N_L(x)` and phases `phi_R(x)`,
    `phi_L(x)` are stored as backend arrays. The lattice's `backend`
    field determines whether these are NumPy or CuPy.

    The session does **not** know about other sessions; multi-session
    interactions go through :class:`~dcl_core.scheduler.TickScheduler`.
    """

    lattice: BipartiteLattice
    n_units: int
    omega: float

    # State arrays -- shape = lattice.shape, dtype = int64 (counts) / float64 (phase).
    # Initialised lazily in __post_init__; do not assume present until after init.
    N_R: np.ndarray = None  # type: ignore[assignment]
    N_L: np.ndarray = None  # type: ignore[assignment]
    phi_R: np.ndarray = None  # type: ignore[assignment]
    phi_L: np.ndarray = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.n_units <= 0:
            raise ValueError(f"n_units must be positive, got {self.n_units}")
        self._allocate_state()

    def _allocate_state(self) -> None:
        """Initialise zeroed token / phase arrays.

        Concrete allocator dispatches on `lattice.backend`. CPU path
        uses `numpy.zeros`; GPU path uses `cupy.zeros`.
        """
        raise NotImplementedError(
            "DiscreteCausalSession._allocate_state: dispatch on lattice.backend"
        )

    @property
    def epsilon_P(self) -> float:
        """The Planck of probability for this session: 1 / N_units."""
        return 1.0 / self.n_units

    def total_tokens(self) -> int:
        """Return `sum(N_R) + sum(N_L)`.

        For an A=1-conserving session, this equals `n_units` exactly.
        Use this in tests to assert conservation with integer equality
        (not `np.allclose`).
        """
        raise NotImplementedError(
            "DiscreteCausalSession.total_tokens: implement via backend sum"
        )

    def assert_unity(self) -> None:
        """Raise if the session has drifted away from A=1.

        Cheap sanity check intended for tests, debug runs, and CI. In
        normal execution this should never fire; if it does, the bug
        is in the hop / remainder / scheduler interaction, not in the
        session itself.
        """
        total = self.total_tokens()
        if total != self.n_units:
            raise AssertionError(
                f"A=1 violated: total_tokens={total}, expected {self.n_units}"
            )

    def amplitude(self, chirality: Chirality) -> np.ndarray:
        """Return the implicit complex amplitude psi_{R|L}(x) for inspection.

        Computed as `sqrt(N(x) / n_units) * exp(i * phi(x))`. Provided
        for introspection / plotting only; the session's evolution
        operates on integer N and continuous phi directly, never on
        the derived complex field.
        """
        raise NotImplementedError(
            "DiscreteCausalSession.amplitude: implement via backend ops"
        )
