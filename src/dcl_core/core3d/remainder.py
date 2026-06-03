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

TODO(complex-carry, v0.2.0)
---------------------------

`carry(x)` is currently **real** ([0, 1) per site per chirality).  The
working hypothesis is that it should be **complex** -- a virtual /
off-shell amplitude that materialises as an integer token when it
crosses threshold.  See `notes/bresenham_residual_design.md` for the
full motivation (including the connection to the 51 extra generators
in Paper II's per-site automorphism algebra) and the three concrete
consequences of the switch:

1. Deposit rule changes: `|carry|**2 >= 1` (or equivalent) replaces
   `carry >= 1`.  Different choices give different virtual-mode
   lifetimes.
2. Conservation invariant becomes two-pool:
   `sum N(x) + sum |carry(x)|**2 == n_units`.  The manifest token
   count alone is no longer conserved tick-to-tick.
3. Upstream API: `HopOperator` currently hands the residual
   `N_target = |psi|**2 * n_units` (real).  With complex carry, the
   hop should hand `psi` (complex) instead; squaring at the boundary
   discards the phase that the carry would otherwise preserve.

Decision criterion: the **minimum-momentum-uncertainty (min-Δp)
experiments**, expected to land in Paper~III
(`external/dcl-paper-03-tidal-ionization`) or in their own series.
The determination is NOT made inside `dcl_core`; a future maintainer
working in that downstream context reaches the call when the
resolution-limit observables (fringes, momentum correlations) show
whether sub-token phase information has been lost.

Until that verdict arrives, do NOT inspect `carry.dtype` outside this
module: that lets the eventual v0.2.0+ bump swap the dtype without
breaking callers.

See `docs/design/02_remainder_strategy.md` for the (eventual)
polished rationale.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from .backends import get_backend
from .lattice import BipartiteLattice


@dataclass
class TokenResidual:
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
        shape as the session's `N_RGB` / `N_CMY` fields, plus a leading
        chirality dimension of size 2. Initialised to zeros.

    Notes
    -----
    The carry field is **part of the session's state for
    reproducibility purposes**. When serialising / restoring a session
    at a checkpoint, the carry must be saved and restored alongside
    `N_RGB`, `N_CMY`, `phi_RGB`, `phi_CMY`. Otherwise the long-run dynamics
    will diverge from the original trajectory.
    """

    lattice: BipartiteLattice
    # TODO(complex-carry, v0.2.0): `carry` is real today; the parked
    # hypothesis is that it should be complex (virtual-particle amplitude
    # tied to the 51 extras in Paper II's algebra).  See
    # `notes/bresenham_residual_design.md` and the module docstring above.
    # Do not inspect `carry.dtype` outside this module so the switch
    # stays internal.
    carry: np.ndarray = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self._allocate()

    def _allocate(self) -> None:
        """Allocate the per-chirality carry field as zeros.

        Carry shape is ``(2, *lattice.shape)``: index 0 holds the R
        residual, index 1 the L residual.  Dtype is float64 (real
        carry; see the module-level TODO for the complex variant).
        """
        backend = get_backend(self.lattice.backend)
        self.carry = backend.zeros((2,) + tuple(self.lattice.shape))

    def quantise(
        self,
        N_target_RGB: np.ndarray | None = None,
        N_target_CMY: np.ndarray | None = None,
        n_units_total: int | None = None,
        *,
        N_target_R: np.ndarray | None = None,  # deprecated alias -> N_target_RGB
        N_target_L: np.ndarray | None = None,  # deprecated alias -> N_target_CMY
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert fractional targets into integer counts, conserving total.

        Parameters
        ----------
        N_target_RGB, N_target_CMY : float arrays
            Analytical target counts at each site, from the hop. May
            be non-integer; may not sum exactly to `n_units_total`.
        n_units_total : int
            The session's `n_units`. The returned integer arrays
            satisfy `sum(N_RGB) + sum(N_CMY) == n_units_total` exactly.

        Returns
        -------
        (N_RGB_int, N_CMY_int) : tuple of int arrays
            Integer token counts at each site, after error-diffusion
            and global rebalancing. A=1 is satisfied exactly.

        Side effects
        ------------
        Updates `self.carry` in place. The new carry is in [0, 1)
        at every site after the call.

        TODO(complex-carry, v0.2.0): the deposit rule below
        (`carry >= 1` -> deposit +1, `carry -= 1`) IS the
        virtual-particle decay law in the real-carry interpretation.
        The complex-carry variant replaces it with a magnitude /
        phase-coherence rule; see `notes/bresenham_residual_design.md`.

        Algorithm
        ---------
        1. Add the persisted carry to the analytical target.  The sum
           is the running fractional accumulator at each site.
        2. Floor pass: `N_int = floor(accumulator)`, new carry =
           accumulator - N_int (in [0, 1) by construction).
        3. Global rebalance: compute `deficit = n_units_total - sum(N_int)`.
           If positive, the top-`deficit` sites by carry (across both
           chiralities) get one extra token each; their carry is reset
           to zero.  If negative, the bottom-`|deficit|` sites with
           ``N_int > 0`` give up one token each; their carry is reset
           to zero.

        Note (v0.1.0 simplification): the rebalance step drops at most
        `|deficit|` units of fractional mass per tick, because
        promoted / demoted carries are reset to zero rather than
        carrying a signed debt.  A future revision that conserves
        carry mass exactly (signed debt OR explicit redistribution)
        is one of the design knobs the complex-carry switch will
        revisit.
        """
        if N_target_R is not None or N_target_L is not None:
            warnings.warn(
                "quantise(N_target_R=, N_target_L=) are deprecated lattice-frame "
                "aliases; use N_target_RGB / N_target_CMY",
                DeprecationWarning,
                stacklevel=2,
            )
            N_target_RGB = N_target_R if N_target_R is not None else N_target_RGB
            N_target_CMY = N_target_L if N_target_L is not None else N_target_CMY
        if N_target_RGB is None or N_target_CMY is None or n_units_total is None:
            raise TypeError(
                "quantise requires N_target_RGB, N_target_CMY, and n_units_total"
            )

        carry_RGB = self.carry[0]
        carry_CMY = self.carry[1]

        # 1.  Running sub-token accumulator at each site: carry + target.
        # physics: fractional probability mass below the epsilon_P threshold,
        # carried forward from the previous tick (the off-shell residual).
        acc_RGB = N_target_RGB + carry_RGB
        acc_CMY = N_target_CMY + carry_CMY

        # 2.  Floor pass.
        backend = get_backend(self.lattice.backend)
        floor_RGB = backend.floor(acc_RGB)
        floor_CMY = backend.floor(acc_CMY)
        N_RGB_int = floor_RGB.astype(np.int64)
        N_CMY_int = floor_CMY.astype(np.int64)
        new_carry_RGB = acc_RGB - floor_RGB
        new_carry_CMY = acc_CMY - floor_CMY

        # 3.  Global rebalance to enforce sum == n_units_total exactly.
        sum_int = int(backend.sum_all(N_RGB_int)) + int(backend.sum_all(N_CMY_int))
        deficit = n_units_total - sum_int

        if deficit != 0:
            N_RGB_int, N_CMY_int, new_carry_RGB, new_carry_CMY = _rebalance(
                N_RGB_int, N_CMY_int, new_carry_RGB, new_carry_CMY, deficit
            )

        self.carry[0] = new_carry_RGB
        self.carry[1] = new_carry_CMY
        return N_RGB_int, N_CMY_int

    def drift_magnitude(self) -> float:
        """Total fractional residual currently held; a coarse health metric.

        Returns `sum(carry)`. Should remain `O(n_sites)` in steady
        state; runaway drift indicates either a non-unitarity bug in
        the hop or an N that is too small for the target physics.
        """
        backend = get_backend(self.lattice.backend)
        return float(backend.sum_all(self.carry))


def _rebalance(
    N_RGB: np.ndarray,
    N_CMY: np.ndarray,
    carry_RGB: np.ndarray,
    carry_CMY: np.ndarray,
    deficit: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Push `deficit` tokens across the (R, L) chiralities to enforce A=1.

    Positive `deficit` -> promote top-`deficit` sites by carry (one
    extra token each, carry reset to 0).  Negative `deficit` ->
    demote bottom-``|deficit|`` sites (sites with ``N_int > 0`` only)
    by one token, carry reset to 0.

    Returns rebalanced ``(N_RGB, N_CMY, carry_RGB, carry_CMY)``.  The
    chirality dimension is flattened internally so the top-k pick is
    across both R and L sites simultaneously -- there is no
    "promote R first, then L" asymmetry.
    """
    n_RGB_sites = carry_RGB.size
    flat_N = np.concatenate([N_RGB.ravel(), N_CMY.ravel()])
    flat_carry = np.concatenate([carry_RGB.ravel(), carry_CMY.ravel()])

    if deficit > 0:
        # Highest-carry sites get the extra tokens.
        top_idx = np.argpartition(-flat_carry, deficit - 1)[:deficit]
        flat_N[top_idx] += 1
        flat_carry[top_idx] = 0.0
    else:
        n_demote = -deficit
        # Sites with N_int == 0 cannot be demoted; mask them out by
        # setting their carry to +inf so argpartition never picks them.
        eligible_carry = np.where(flat_N > 0, flat_carry, np.inf)
        bot_idx = np.argpartition(eligible_carry, n_demote - 1)[:n_demote]
        flat_N[bot_idx] -= 1
        flat_carry[bot_idx] = 0.0

    N_RGB_new = flat_N[:n_RGB_sites].reshape(N_RGB.shape)
    N_CMY_new = flat_N[n_RGB_sites:].reshape(N_CMY.shape)
    carry_RGB_new = flat_carry[:n_RGB_sites].reshape(carry_RGB.shape)
    carry_CMY_new = flat_carry[n_RGB_sites:].reshape(carry_CMY.shape)
    return N_RGB_new, N_CMY_new, carry_RGB_new, carry_CMY_new


# Deprecated alias.  The class was renamed BresenhamResidual -> TokenResidual:
# "Bresenham" named the *algorithm* (error-diffusion), not the lattice object,
# which IS the sub-token fractional carry that keeps A=1 exact.  The
# error-diffusion strategy is unchanged; see this module's docstring.  Retained
# for backward-compatibility (downstream pins, old call sites).
BresenhamResidual = TokenResidual
