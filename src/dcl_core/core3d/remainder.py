"""Bresenham-style residual accumulator: integer tokens from fractional amplitude.

The hop operator produces an analytical target `N_target(x) = n_units * |psi_new(x)|^2`,
which is in general fractional. The remainder rule converts these
fractional targets back to integer token counts while preserving A=1
exactly.

Strategy: error-diffusion (Bresenham-style).

    For each site x:
        carry(x) += N_target(x) - floor(N_target(x))
        N_new(x)  = floor(N_target(x))
        if carry(x) >= 1:
            N_new(x) += 1
            carry(x) -= 1

    After the per-site pass, distribute any remaining tokens (caused
    by the global rounding step) to the highest-residual sites until
    sum(N_new) == n_units exactly.

The residual `carry(x)` is **persisted between ticks**. It IS the
sub-token-resolution amplitude that the framework cannot resolve at
this N but that should not be discarded. Across long runs, carries
accumulate and eventually deposit tokens at the right rate.

Open hypothesis (parked, not committed): `carry(x)` may need to be
**complex**, not real, to preserve phase interference. A real-residual
implementation discards phase information when squaring; a complex
residual carries it forward. The decision is forced by
`exp_03`-style interference experiments: if real-residual fringes
match the continuous core, real is sufficient; if not, residual is
genuinely amplitude.

See `docs/design/02_remainder_strategy.md` for the design rationale.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .lattice import BipartiteLattice


@dataclass
class BresenhamResidual:
    """Error-diffusion accumulator: fractional bits carried tick-to-tick.

    Parameters
    ----------
    lattice : BipartiteLattice
        Provides shape and backend. The carry field shadows the
        session's token field.

    Attributes
    ----------
    carry : array
        Per-site, per-chirality fractional residual in [0, 1). Same
        shape as the session's `N_R` / `N_L` fields, plus a leading
        chirality dimension of size 2. Initialised to zeros.

    Notes
    -----
    The carry field is **part of the session's state for
    reproducibility purposes**. When serialising / restoring a session
    at a checkpoint, the carry must be saved and restored alongside
    `N_R`, `N_L`, `phi_R`, `phi_L`. Otherwise the long-run dynamics
    will diverge from the original trajectory.
    """

    lattice: BipartiteLattice
    carry: np.ndarray = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self._allocate()

    def _allocate(self) -> None:
        """Allocate the per-chirality carry field."""
        raise NotImplementedError(
            "BresenhamResidual._allocate: dispatch on lattice.backend"
        )

    def quantise(
        self,
        N_target_R: np.ndarray,
        N_target_L: np.ndarray,
        n_units_total: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert fractional targets into integer counts, conserving total.

        Parameters
        ----------
        N_target_R, N_target_L : float arrays
            Analytical target counts at each site, from the hop. May
            be non-integer; may not sum exactly to `n_units_total`.
        n_units_total : int
            The session's `n_units`. The returned integer arrays
            satisfy `sum(N_R) + sum(N_L) == n_units_total` exactly.

        Returns
        -------
        (N_R_int, N_L_int) : tuple of int arrays
            Integer token counts at each site, after error-diffusion
            and global rebalancing. A=1 is satisfied exactly.

        Side effects
        ------------
        Updates `self.carry` in place. The new carry is in [0, 1)
        at every site after the call.
        """
        raise NotImplementedError(
            "BresenhamResidual.quantise: floor + carry, then top-k rebalance"
        )

    def drift_magnitude(self) -> float:
        """Total fractional residual currently held; a coarse health metric.

        Returns `sum(carry)`. Should remain `O(n_sites)` in steady
        state; runaway drift indicates either a non-unitarity bug in
        the hop or an N that is too small for the target physics.
        """
        raise NotImplementedError(
            "BresenhamResidual.drift_magnitude: implement via backend sum"
        )
