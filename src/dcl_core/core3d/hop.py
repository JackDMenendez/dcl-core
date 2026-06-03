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
fractional residual is handled by :class:`TokenResidual`; this
operator just produces the analytical target.

TODO(complex-carry, v0.2.0): squaring `psi_new` to `|psi|^2` at this
boundary discards the phase information that a complex
:class:`TokenResidual.carry` would need.  If the residual switches
to complex carry, `step` must hand the residual `psi_new` directly
(complex) instead of `N_target` (real).  See
`notes/bresenham_residual_design.md` and `remainder.py`'s module
docstring.

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

from .backends import get_backend
from .lattice import CMY_VECTORS, RGB_VECTORS, BipartiteLattice, TickParity
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
    :class:`~dcl_core.remainder.TokenResidual`.

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
            TokenResidual then converts these to new integer
            token counts.

        Notes
        -----
        For testing, this method is the natural place to verify the
        Clifford algebra anti-commutator and continuum-limit dispersion.
        See `tests/test_hop.py` and `tests/test_continuum_limit.py`.
        """
        backend = get_backend(self.lattice.backend)

        # Reconstruct the complex amplitudes from the integer-token state.
        # The session.amplitude() snapshot is fresh each call; we do not
        # mutate session state.
        psi_R = session.amplitude("R")
        psi_L = session.amplitude("L")

        # On-site phase mismatch.  `delta_phi = omega + V(x)`; mass +
        # external potential.  Scalar or array; downstream cos/sin
        # broadcast either way.
        delta_phi = session.omega
        if external_potential is not None:
            if external_potential.shape != self.lattice.shape:
                raise ValueError(
                    f"external_potential.shape {external_potential.shape} "
                    f"!= lattice.shape {self.lattice.shape}"
                )
            delta_phi = delta_phi + external_potential
        cos_half = backend.cos(delta_phi / 2.0)
        sin_half = backend.sin(delta_phi / 2.0)

        vectors = self.lattice.neighbour_offsets(parity)
        if parity == "even":
            # Even tick: V_3^+ (RGB) is the active sublattice.  The RGB
            # component takes the hopped CMY amplitude; CMY is passive.
            # physics: psi_R updates from hopped psi_L (Dirac kinetic term).
            hop_L = self._hop_average(psi_L, vectors)
            psi_R_new = cos_half * hop_L + 1j * sin_half * psi_R
            psi_L_new = psi_L
        else:  # odd tick: V_3^- (CMY) active.
            # physics: psi_L updates from hopped psi_R.
            hop_R = self._hop_average(psi_R, vectors)
            psi_L_new = cos_half * hop_R + 1j * sin_half * psi_L
            psi_R_new = psi_R

        return psi_R_new, psi_L_new

    def _hop_average(
        self,
        psi: np.ndarray,
        vectors: tuple[tuple[int, int, int], ...],
    ) -> np.ndarray:
        """Mean of ``psi`` shifted by each vector in ``vectors``.

        Matches the docstring's "average of `psi` over the three
        X-basis-vector shifts."  Uses ``backend.shift`` (periodic) so
        the result respects the lattice's boundary conventions.
        """
        backend = get_backend(self.lattice.backend)
        n = len(vectors)
        accumulator = backend.shift(psi, vectors[0])
        for v in vectors[1:]:
            accumulator = accumulator + backend.shift(psi, v)
        return accumulator / n

    def fourier_kernel(self, k: np.ndarray) -> np.ndarray:
        """Return the structure factor at wavevector k (for analysis / tests).

        The structure factor IS the Fourier transform of the hop
        average. In the small-k limit it reduces to `i k . gamma_RGB`
        (or `i k . gamma_CMY`); the Taylor expansion of this is what
        produces the continuum Dirac operator.

        Used by `tests/test_continuum_limit.py` to verify
        `E^2 -> m^2 + |p|^2` as `a -> 0`.

        Implementation
        --------------
        Returns the **bipartite antisymmetric** combination of the RGB
        and CMY structure factors, ``(S_RGB(k) - S_CMY(k)) / 2``.
        Since CMY vectors are the negatives of the RGB vectors, this
        equals ``i * (1/3) * sum_v sin(k . v)`` for ``v`` in
        ``RGB_VECTORS`` -- purely imaginary, vanishing at ``k = 0``,
        and linear in ``k`` for small ``k`` (which IS the
        ``i k . gamma`` form the continuum Dirac operator takes).
        """
        k_arr = np.asarray(k, dtype=np.float64)
        if k_arr.shape != (3,):
            raise ValueError(
                f"k must be a 3-vector (shape (3,)), got shape {k_arr.shape}"
            )
        s_rgb = sum(np.exp(1j * float(np.dot(k_arr, v))) for v in RGB_VECTORS) / 3
        s_cmy = sum(np.exp(1j * float(np.dot(k_arr, v))) for v in CMY_VECTORS) / 3
        return np.asarray((s_rgb - s_cmy) / 2)
