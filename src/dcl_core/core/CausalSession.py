"""
CausalSession.py
The Quantum Lantern: a particle as a persistent probability flux.

The bipartite Dirac spinor model:

  STRUCTURE:
    psi_R : amplitude on RGB sublattice (right-handed component)
    psi_L : amplitude on CMY sublattice (left-handed component)

    The bipartite lattice IS the Dirac structure.  RGB/CMY are psi_R/psi_L.

  DIRAC TICK RULE:
    Even tick (RGB active):
      new_psi_R = cos(delta_phi/2) * kinetic_hop(psi_L, RGB_VECTORS)
                + 1j * sin(delta_phi/2) * psi_R

    Odd tick (CMY active):
      new_psi_L = cos(delta_phi/2) * kinetic_hop(psi_R, CMY_VECTORS)
                + 1j * sin(delta_phi/2) * psi_L

    - Kinetic term hops the OPPOSITE component across the active sublattice
    - Mass term rotates each component IN PLACE (no hop, just phase rotation)
    - A=1: sum(|psi_R|^2 + |psi_L|^2) = 1 enforced after each tick

  SPECIAL CASES:
    Massless (delta_phi=0): cos=1, sin=0 -> full swap per tick (photon)
    Max mass (delta_phi=pi): cos=0, sin=1 -> stays in place

  MOMENTUM:
    Phase-alignment weighting biases the kinetic hop toward aligned neighbors,
    giving net drift.  Inertia scales with omega (1/(1+omega) damping).

  ZITTERBEWEGUNG:
    Now appears as amplitude trading between psi_R and psi_L each tick,
    rather than p_stay at a single site.  More physically accurate.

Paper reference: Section 3 (Dirac Spinor, Bipartite Tick Rule)
"""

import numpy as np
from typing import Tuple
from .OctahedralLattice import (OctahedralLattice, COORDINATION_NUMBER,
                                 SUBLATTICE_SIZE, active_vectors,
                                 ALL_VECTORS, RGB_VECTORS, CMY_VECTORS)
from .PhaseOscillator import PhaseOscillator
from .UnityConstraint import enforce_unity, enforce_unity_spinor


class CausalSession:
    """
    A particle as a persistent causal session on T^3_diamond.

    Uses a two-component Dirac spinor (psi_R, psi_L) matching the
    bipartite RGB/CMY sublattice structure.

    The is_massless flag is a performance shortcut for delta_phi=0;
    the physics is identical since cos(0/2)=1 and sin(0/2)=0.
    """

    def __init__(self,
                 lattice: OctahedralLattice,
                 initial_node: Tuple[int, int, int],
                 instruction_frequency: float,
                 momentum: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                 is_massless: bool = False,
                 *,
                 prob_floor: float | None = None):
        """
        prob_floor : float | None, default None
            Minimum per-site probability (= |psi_R|^2 + |psi_L|^2).  When a
            tick produces a per-site probability below prob_floor, both
            spinor components at that node are scaled so the per-site
            probability equals prob_floor exactly, with phase (and the R/L
            ratio) preserved; the joint state is then renormalised so
            sum(probabilities) == 1 (A=1 preserved).  Applied at
            end-of-tick.  None (default) clamps nothing -- behaviour is
            bit-for-bit the pre-prob_floor engine.  Nodes with exactly zero
            amplitude carry no phase and are left at zero (the floor is the
            minimum *non-zero* probability quantum, mirroring core3d's
            1/N = delta_p_min granularity).  This IS the delta_p_min knob
            for the continuous engine (see dcl-delta-p-min's 4-cell grid).
        """
        self.lattice          = lattice
        self.phase_rotor      = PhaseOscillator(frequency=instruction_frequency)
        self.tick_counter     = 0
        self.is_massless      = is_massless

        if prob_floor is not None and not (0.0 < prob_floor < 1.0):
            raise ValueError(
                f"prob_floor must be in the open interval (0, 1) or None, "
                f"got {prob_floor!r}"
            )
        self.prob_floor       = prob_floor

        # ── prob_floor cleanup ledger (A=1 probability accounting) ──────────
        # Raising a sub-floor node up to prob_floor *manufactures* probability
        # (floor - p) at that node; the end-of-tick joint renormalisation then
        # spreads that manufactured mass across the whole state so A=1 still
        # holds.  The manufactured mass is therefore not free -- it is a real
        # bookkeeping quantity (the "probability debt" the floor incurs each
        # tick).  These counters record it exactly so the displaced mass can be
        # audited downstream (dcl-delta-p-min's zoo-accounting hypothesis: the
        # debt is balanced by other generators in the SM zoo).  All stay 0.0
        # when prob_floor is None.
        self.floor_manufactured_total = 0.0   # cumulative manufactured prob
        self.floor_n_raised_total     = 0     # cumulative count of raised nodes
        self.floor_ticks_floored      = 0     # ticks on which the floor ran
        self._floor_manufactured_last = 0.0   # per-tick manufactured prob
        self._floor_n_raised_last     = 0     # per-tick count of raised nodes
        self._floor_sum_before_last   = 0.0   # sum(p) before the floor (pre-renorm)
        self._floor_sum_after_last    = 0.0   # sum(p) after the floor (pre-renorm)

        # Two-component Dirac spinor
        shape = (lattice.size_x, lattice.size_y, lattice.size_z)
        self.psi_R = np.zeros(shape, dtype=complex)
        self.psi_L = np.zeros(shape, dtype=complex)

        # Initialize at single node, amplitude split equally between components
        x0, y0, z0 = initial_node
        amp = 1.0 / np.sqrt(2.0)
        self.psi_R[x0, y0, z0] = amp
        self.psi_L[x0, y0, z0] = amp

        if any(p != 0.0 for p in momentum):
            self._apply_initial_momentum(initial_node, momentum)

        enforce_unity_spinor(self.psi_R, self.psi_L)

    # ── Backward-compatibility property ───────────────────────────────────────

    @property
    def psi(self) -> np.ndarray:
        """
        Backward-compatible accessor: returns psi_R.
        Use probability_density() for the physically correct total density.
        """
        return self.psi_R

    @psi.setter
    def psi(self, value: np.ndarray):
        """
        Backward-compatible setter: distributes a scalar field equally
        across both spinor components, preserving A=1.

        If value is normalized (sum |psi|^2 = 1), then
        psi_R = psi_L = value / sqrt(2) gives
        sum(|psi_R|^2 + |psi_L|^2) = sum(|psi|^2) = 1.
        """
        amp = value / np.sqrt(2.0)
        self.psi_R = amp.copy().astype(complex)
        self.psi_L = amp.copy().astype(complex)

    # ── Momentum initialization ────────────────────────────────────────────────

    def _apply_initial_momentum(self, center, momentum):
        """
        Momentum = phase gradient across the packet.
        Applied identically to both spinor components.
        """
        kx, ky, kz = momentum
        for x in range(self.lattice.size_x):
            for y in range(self.lattice.size_y):
                for z in range(self.lattice.size_z):
                    if (np.abs(self.psi_R[x, y, z]) > 1e-12 or
                            np.abs(self.psi_L[x, y, z]) > 1e-12):
                        phase = np.exp(1j * (kx*x + ky*y + kz*z))
                        self.psi_R[x, y, z] *= phase
                        self.psi_L[x, y, z] *= phase

    # ── Kinetic hop kernel ─────────────────────────────────────────────────────

    def _kinetic_hop(self, source: np.ndarray,
                     vectors: list) -> np.ndarray:
        """
        Directed kinetic hop: source amplitude propagates to neighbor sites.

        For each direction v, the phase advance is delta_p = phi(r+v) - phi(r).
        A positive delta_p means the neighbor is AHEAD in phase -- the momentum
        gradient points toward it.  We use delta_p as the directional weight
        (max(0, delta_p), so only momentum-aligned directions receive amplitude)
        and include exp(i*delta_p) as a per-direction phase correction so that
        emitted amplitude arrives at the destination with the correct plane-wave
        phase (i.e. constructive interference is preserved).

        For zero-momentum regions (all delta_p ≈ 0): falls back to uniform
        distribution with no phase correction.  Gravity continues to act via
        the sin/cos(delta_phi/2) mass-term coefficients, which are independent
        of this momentum bias.

        Parameters
        ----------
        source  : complex (X,Y,Z) -- the component being hopped
        vectors : list of (dx,dy,dz) -- active sublattice direction set

        Returns
        -------
        result : complex (X,Y,Z) -- accumulated hopped amplitude
        """
        n_vec = len(vectors)
        local_phase = np.angle(source)

        # Per-direction phase advance and directed weight
        delta_p_list = []
        weights = np.zeros((n_vec,) + source.shape, dtype=float)
        for i, (dx, dy, dz) in enumerate(vectors):
            nb = np.roll(np.roll(np.roll(source, -dx, 0), -dy, 1), -dz, 2)
            nb_abs = np.abs(nb)
            nb_phase = np.where(nb_abs > 1e-9, np.angle(nb), local_phase)
            delta_p = nb_phase - local_phase               # phase advance in dir v
            delta_p_list.append(delta_p)
            # Weight = positive phase advance only; inertia damps response to gradient
            weights[i] = np.maximum(0.0, delta_p) / (1.0 + self.phase_rotor.omega)

        # Normalize (fallback: uniform real weights when momentum ≈ zero)
        total_w  = weights.sum(axis=0)
        zero_mom = total_w < 1e-12
        total_w_safe = np.where(zero_mom, 1.0, total_w)
        uniform  = 1.0 / n_vec

        # Emit: per-direction complex weight = (real weight) * exp(i*delta_p)
        # This ensures amplitude arrives at destination with the correct phase.
        result = np.zeros_like(source)
        sx, sy, sz = source.shape
        for i, (dx, dy, dz) in enumerate(vectors):
            w_i = np.where(zero_mom, uniform, weights[i] / total_w_safe)
            # Phase correction: exp(i*delta_p) so emitted amp matches dest wave.
            # For zero-momentum fallback: no correction (exp(i*0)=1).
            phase_corr = np.where(zero_mom, 1.0+0j,
                                  np.exp(1j * delta_p_list[i]).astype(complex))
            emission = source * phase_corr * w_i

            mask = np.ones((sx, sy, sz), dtype=bool)
            if dx > 0: mask[sx-dx:, :, :]  = False
            if dx < 0: mask[:-dx,   :, :]  = False
            if dy > 0: mask[:, sy-dy:, :]  = False
            if dy < 0: mask[:, :-dy,   :]  = False
            if dz > 0: mask[:, :, sz-dz:]  = False
            if dz < 0: mask[:, :, :-dz  ]  = False
            emission = np.where(mask, emission, 0.0)

            result += np.roll(np.roll(np.roll(emission, dx, 0), dy, 1), dz, 2)

        return result

    # ── The Dirac tick ─────────────────────────────────────────────────────────

    def tick(self, normalize: bool = True):
        """
        The bipartite Dirac spinor update cycle.

        Even tick (RGB active):
          new_psi_R = cos(delta_phi/2) * kinetic_hop(psi_L, RGB)
                    + 1j * sin(delta_phi/2) * psi_R
          psi_L     unchanged

        Odd tick (CMY active):
          new_psi_L = cos(delta_phi/2) * kinetic_hop(psi_R, CMY)
                    + 1j * sin(delta_phi/2) * psi_L
          psi_R     unchanged

        A=1 enforced after each tick via joint normalization.

        Parameters
        ----------
        normalize : bool, default True
            If True (the default), enforce_unity_spinor is called after the
            tick, restoring |psi_R|^2 + |psi_L|^2 = 1 at every node. The
            standard A=1 contract.
            If False, normalization is skipped. Required by the photon
            emission experiments (exp_19, exp_19c) where amplitude is
            transferred between sessions between consecutive ticks; the
            caller is responsible for re-establishing the joint constraint
            across the full multi-session set.

        Paper reference: Section 3 (Dirac tick rule, bipartite structure)
        """
        tick_parity = self.tick_counter % 2

        delta_phi = (self.phase_rotor.omega
                     + self.lattice.topological_potential)       # (X,Y,Z)
        cos_half  = np.cos(delta_phi / 2.0)                      # (X,Y,Z)
        sin_half  = np.sin(delta_phi / 2.0)                      # (X,Y,Z)

        if self.is_massless:
            # Massless photon: strict bipartite alternation (chirality preserved).
            # Even tick: psi_L -> psi_R via RGB; psi_L unchanged.
            # Odd tick:  psi_R -> psi_L via CMY; psi_R unchanged.
            if tick_parity == 0:
                new_psi_R = self._kinetic_hop(self.psi_L, RGB_VECTORS)
                new_psi_L = self.psi_L
            else:
                new_psi_L = self._kinetic_hop(self.psi_R, CMY_VECTORS)
                new_psi_R = self.psi_R
        else:
            # Massive particle: both components updated simultaneously.
            # RGB hop: psi_L -> psi_R; CMY hop: psi_R -> psi_L.
            # Simultaneous update averages RGB and CMY displacements,
            # giving zero net CoM drift for zero-momentum states (symmetry
            # of V1+V2+V3 + CMY1+CMY2+CMY3 = 0).
            hop_R     = self._kinetic_hop(self.psi_L, RGB_VECTORS)
            hop_L     = self._kinetic_hop(self.psi_R, CMY_VECTORS)
            new_psi_R = cos_half * hop_R + 1j * sin_half * self.psi_R
            new_psi_L = cos_half * hop_L + 1j * sin_half * self.psi_L

        # A=1: normalize both components jointly (skip iff caller asks).
        # The probability floor (if set) is applied FIRST, then the joint
        # renormalisation restores sum(p)=1 -- so prob_floor never breaks A=1.
        if normalize:
            if self.prob_floor is not None:
                self._apply_prob_floor(new_psi_R, new_psi_L)
            enforce_unity_spinor(new_psi_R, new_psi_L)
        self.psi_R = new_psi_R
        self.psi_L = new_psi_L

    def _apply_prob_floor(self, psi_R: np.ndarray, psi_L: np.ndarray) -> None:
        """
        Clamp per-site probability up to self.prob_floor (in place) and record
        the manufactured-probability ledger.

        Per-site probability is p(x) = |psi_R(x)|^2 + |psi_L(x)|^2 -- this IS
        the Born density of the bipartite Dirac spinor at node x.  Any node
        with 0 < p(x) < prob_floor has BOTH components scaled by the real,
        positive factor sqrt(prob_floor)/sqrt(p(x)); afterward p(x) ==
        prob_floor exactly and every phase (and the psi_R/psi_L ratio) is
        unchanged.  Nodes with p(x) == 0 carry no amplitude to rescale and are
        left at zero (no phase to preserve; division-by-zero avoided).

        Overflow safety.  The rescale factor is computed as
        ``sqrt(prob_floor) / sqrt(p)``, NOT ``sqrt(prob_floor / p)``.  The two
        are algebraically identical, but the fused form forms the intermediate
        ``prob_floor / p`` which overflows to ``inf`` (then propagates NaN
        through the joint renormalisation) whenever a continuous wavepacket's
        tail has underflowed p to a deep denormal (p < prob_floor / max_float
        ~ 1e-308 * prob_floor).  Splitting into two square roots keeps both
        operands in the normal float64 range -- even the smallest subnormal
        p ~ 1e-323 has sqrt(p) ~ 3e-162, a normal double -- so the floor is
        numerically robust on the deep tails that arise on large lattices.

        A=1 accounting.  Raising a node from p to prob_floor manufactures
        (prob_floor - p) of probability there; the end-of-tick
        ``enforce_unity_spinor`` then renormalises sum(p) back to 1, spreading
        that manufactured mass across the state.  A=1 is never broken, but the
        manufactured mass is a real bookkeeping quantity and is recorded
        exactly in self.floor_* (see __init__).  Large manufactured totals are
        the quantitative signature of the floor over-writing a continuous
        packet toward uniform -- the cross-engine non-equivalence with
        core3d's token granularity (where sub-quantum mass rounds *down* to
        empty rather than *up* to the floor).

        Called at end-of-tick, BEFORE enforce_unity_spinor, per the
        dcl-delta-p-min coordination spec (notes/dcl_core_coordination.md):
        applying the floor mid-tick would leave the partial state non-unit.
        """
        floor = self.prob_floor
        p = np.abs(psi_R) ** 2 + np.abs(psi_L) ** 2
        below = (p > 0.0) & (p < floor)

        # ── ledger: probability this tick's floor manufactures ──────────────
        manufactured = float(np.sum(np.where(below, floor - p, 0.0)))
        sum_before = float(p.sum())

        # ── apply floor (overflow-safe split-sqrt; only `below` sites move) ─
        safe_p = np.where(below, p, 1.0)
        scale = np.where(below, np.sqrt(floor) / np.sqrt(safe_p), 1.0)
        psi_R *= scale
        psi_L *= scale

        # ── record ledger ───────────────────────────────────────────────────
        n_raised = int(below.sum())
        self._floor_manufactured_last = manufactured
        self._floor_n_raised_last = n_raised
        self._floor_sum_before_last = sum_before
        self._floor_sum_after_last = sum_before + manufactured
        self.floor_manufactured_total += manufactured
        self.floor_n_raised_total += n_raised
        self.floor_ticks_floored += 1

    def floor_ledger(self) -> dict:
        """Snapshot of the prob_floor cleanup ledger (A=1 accounting).

        manufactured_total : cumulative probability the floor has manufactured
            (sum over ticks of sum_x max(0, prob_floor - p(x)) on raised
            nodes), pre-renormalisation.  This is the displaced mass A=1
            conservation must attribute -- the dcl-delta-p-min zoo-accounting
            target.  0.0 when prob_floor is None or the floor never bit.
        n_raised_total     : cumulative count of node-raisings.
        ticks_floored      : number of ticks the floor ran.
        manufactured_last / n_raised_last / sum_before_last / sum_after_last :
            the most recent tick's values.
        """
        return {
            "prob_floor": self.prob_floor,
            "manufactured_total": self.floor_manufactured_total,
            "n_raised_total": self.floor_n_raised_total,
            "ticks_floored": self.floor_ticks_floored,
            "manufactured_last": self._floor_manufactured_last,
            "n_raised_last": self._floor_n_raised_last,
            "sum_before_last": self._floor_sum_before_last,
            "sum_after_last": self._floor_sum_after_last,
        }

    def probability_density(self) -> np.ndarray:
        """Total probability density: |psi_R|^2 + |psi_L|^2."""
        return np.abs(self.psi_R) ** 2 + np.abs(self.psi_L) ** 2

    def advance_tick_counter(self):
        self.tick_counter += 1
        self.phase_rotor.advance()
