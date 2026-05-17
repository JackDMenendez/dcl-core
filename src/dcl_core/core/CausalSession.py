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
                 is_massless: bool = False):
        self.lattice          = lattice
        self.phase_rotor      = PhaseOscillator(frequency=instruction_frequency)
        self.tick_counter     = 0
        self.is_massless      = is_massless

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
        if normalize:
            enforce_unity_spinor(new_psi_R, new_psi_L)
        self.psi_R = new_psi_R
        self.psi_L = new_psi_L

    def probability_density(self) -> np.ndarray:
        """Total probability density: |psi_R|^2 + |psi_L|^2."""
        return np.abs(self.psi_R) ** 2 + np.abs(self.psi_L) ** 2

    def advance_tick_counter(self):
        self.tick_counter += 1
        self.phase_rotor.advance()
