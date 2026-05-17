"""Hop operator: bipartite Dirac evolution on the lattice.

The hop operator IS the lattice realisation of the Dirac equation. At
each tick, one chirality propagates along the active sublattice's
basis vectors while the other carries phase forward via Zitterbewegung.

Even tick (RGB active):
    psi_R_new = cos(delta_phi/2) * hop_RGB(psi_L) + i * sin(delta_phi/2) * psi_R
    psi_L_new = psi_L

Odd tick (CMY active):
    psi_L_new = cos(delta_phi/2) * hop_CMY(psi_R) + i * sin(delta_phi/2) * psi_L
    psi_R_new = psi_R

where `delta_phi = omega + V(x)` is the on-site phase mismatch (mass
+ external potential) and `hop_X(psi)` is the average of `psi` over
the three X-basis-vector shifts.

In the integer-token framework, the analytical step computes a
fractional new token count `N_new(x) = n_units * |psi_new(x)|^2`. The
fractional residual is handled by :class:`BresenhamResidual`; this
operator just produces the analytical target.

Peierls substitution: when an external gauge potential A_mu is
present, each hop along basis vector v acquires a phase
`exp(i * A . v)`. This IS the U(1) gauge coupling at the lattice
level. The structure-factor expansion of the hop is documented in
`docs/design/02_remainder_strategy.md`.

See:
    docs/reference/hop.md -- API reference
    docs/design/01_planck_of_probability.md -- why integer tokens
    notes/structure_factor_derivation.md -- (when added)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .lattice import BipartiteLattice, TickParity
from .session import DiscreteCausalSession


@dataclass
class HopOperator:
    """Bipartite Dirac hop on a fixed lattice.

    Parameters
    ----------
    lattice : BipartiteLattice
        The lattice this operator hops on. The hop's parity sequence
        is determined by tick number (even/odd) and the lattice's
        sublattice vectors.

    Notes
    -----
    The operator is stateless; calling it on a session returns the
    new analytical amplitude target without mutating the session.
    The token-count update (rounding + residual carry) is done by
    :class:`~dcl_core.remainder.BresenhamResidual`.

    Backends:
      "cpu" -- NumPy broadcasting + np.roll for periodic shifts.
      "gpu" -- CuPy + RawKernel; one kernel per parity for coalesced
               memory access.
    """

    lattice: BipartiteLattice

    def step(
        self,
        session: DiscreteCausalSession,
        parity: TickParity,
        external_potential: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute the analytical (psi_R_new, psi_L_new) for one tick.

        Parameters
        ----------
        session : DiscreteCausalSession
            Current state. **Not mutated** by this call.
        parity : "even" | "odd"
            Which sublattice is active this tick.
        external_potential : array of shape `lattice.shape`, optional
            On-site contribution to `delta_phi`. Defaults to zeros.

        Returns
        -------
        (psi_R_new, psi_L_new) : tuple of complex arrays
            Analytical Dirac evolution for one tick. The
            BresenhamResidual then converts these to new integer
            token counts.

        Notes
        -----
        For testing, this method is the natural place to verify the
        Clifford algebra anti-commutator and continuum-limit dispersion.
        See `tests/test_hop.py` and `tests/test_continuum_limit.py`.
        """
        raise NotImplementedError(
            "HopOperator.step: implement bipartite Dirac update for the active sublattice"
        )

    def fourier_kernel(self, k: np.ndarray) -> np.ndarray:
        """Return the structure factor at wavevector k (for analysis / tests).

        The structure factor IS the Fourier transform of the hop
        average. In the small-k limit it reduces to `i k . gamma_RGB`
        (or `i k . gamma_CMY`); the Taylor expansion of this is what
        produces the continuum Dirac operator.

        Used by `tests/test_continuum_limit.py` to verify
        `E^2 -> m^2 + |p|^2` as `a -> 0`.
        """
        raise NotImplementedError(
            "HopOperator.fourier_kernel: implement the structure-factor expansion"
        )
