"""
CompositeCausalSession.py
A bound collection of CausalSessions evolving as a composite particle.

The key distinction from running multiple sessions independently in the
TickScheduler is the probability density:

  Independent sessions:   P = Σ |ψ_i|²          (incoherent sum — no cancellation)
  Composite session:      P = |Σ ψ_i|²           (coherent sum — phases interfere)

Phase cancellation only appears in the coherent sum.  For a neutral composite
(constituents with opposite rgb_cmy_imbalance), the constituent phase gradients
cancel in the coherent sum — this is the mechanism that narrows the effective
material cone.

See notes/material_cone_and_composites.md and notes/cone_modification_classes.md.

DOCUMENTATION CONVENTION:
  Every non-trivial line of physics code should say what it IS in the theory,
  not just what it does in the program.  Name the mathematical object, cite
  the paper equation where one exists, and state the correspondence explicitly:
  "this IS X" when exact, "this approximates X" in the continuum limit.
  The structure factor comment in CausalSession._kinetic_hop is the template.

Paper reference: future Section on composite particles and cone narrowing.
"""

import numpy as np
from .CausalSession import CausalSession
from .UnityConstraint import enforce_unity_spinor


class CompositeCausalSession:
    """
    A bound composite of N CausalSessions.

    Each constituent session ticks independently each macro-tick, then a
    binding step partially locks their phases according to binding_strength.
    The composite's probability density uses the coherent sum |Σ ψ_i|²,
    so constituent phase cancellation is physically realized.

    Parameters
    ----------
    sessions        : list of CausalSession  — the constituent particles
    binding_strength: float in [0, 1]
                      0.0 = no binding (free particles, coherent sum only)
                      0.9 = strongly bound (phases lock each tick)
                      1.0 = perfect lock (constituents move as one)
    """

    def __init__(self, sessions: list, binding_strength: float = 0.9):
        if len(sessions) < 2:
            raise ValueError("CompositeCausalSession requires at least 2 sessions.")
        self.sessions         = sessions
        self.binding_strength = float(np.clip(binding_strength, 0.0, 1.0))
        self.tick_counter     = 0

    # ── Evolution ─────────────────────────────────────────────────────────────

    def tick(self):
        """
        Advance all constituent sessions by one tick, then apply binding.
        """
        for s in self.sessions:
            s.tick()
        if self.binding_strength > 0:
            self._apply_binding()

    def advance_tick_counter(self):
        for s in self.sessions:
            s.advance_tick_counter()
        self.tick_counter += 1

    def _apply_binding(self):
        """
        Partially lock the phases of all constituent sessions toward their
        mean phase.  binding_strength=1 → all phases equal after one tick.

        This is the physical mechanism of the strong force: color-correlated
        sessions are prevented from drifting in phase relative to each other.
        The phase lock is what makes their cones interfere coherently and
        causes the phase gradient of the coherent sum to approach zero for
        a neutral composite.
        """
        # Σ ψ_R_i, Σ ψ_L_i: coherent sums -- the composite spinor components.
        # The phase of each sum IS the mean phase of the bound state.
        sum_R = sum(s.psi_R for s in self.sessions)
        sum_L = sum(s.psi_L for s in self.sessions)
        # φ_mean(x) = arg(Σ ψ_R_i): the mean phase of the composite at each node.
        # This IS the target phase that binding pulls constituents toward.
        mean_phase_R = np.angle(sum_R)
        mean_phase_L = np.angle(sum_L)

        for s in self.sessions:
            # Δφ_i(x) = φ_mean - φ_i: phase deficit of this constituent relative to the composite mean.
            # For a neutral composite at lock-in, Δφ_i → 0 (phases aligned).
            phase_diff_R = mean_phase_R - np.angle(s.psi_R)
            phase_diff_L = mean_phase_L - np.angle(s.psi_L)
            # exp(i * α * Δφ_i): partial U(1) rotation toward the mean phase.
            # α = binding_strength: α=0 → free, α=1 → fully locked after one tick.
            # This IS the discrete strong-coupling evolution U = exp(i α Δφ) in U(1).
            s.psi_R *= np.exp(1j * self.binding_strength * phase_diff_R)
            s.psi_L *= np.exp(1j * self.binding_strength * phase_diff_L)
            # A=1: group manifold constraint -- each constituent session persists at unit norm.
            enforce_unity_spinor(s.psi_R, s.psi_L)

    # ── Probability density (coherent sum) ────────────────────────────────────

    def probability_density(self) -> np.ndarray:
        """
        Coherent probability density: |Σ_i ψ_R_i|² + |Σ_i ψ_L_i|²

        This is the physically correct density for a bound composite.
        It includes cross-terms (interference) between constituents.

        For a neutral composite with cancelling phase gradients, this is
        MUCH more localized than the incoherent sum Σ|ψ_i|² — the phase
        cancellation suppresses the off-center amplitude.
        """
        # Σ ψ_R_i: coherent spinor sum -- the composite wavefunction right-component.
        # The cross-terms |Σ ψ_i|² - Σ|ψ_i|² = Σ_{i≠j} Re(ψ_i* ψ_j) ARE the interference.
        # For a neutral composite these cross-terms cancel the off-center amplitude.
        sum_R = sum(s.psi_R for s in self.sessions)
        sum_L = sum(s.psi_L for s in self.sessions)
        # |Σ ψ_R|² + |Σ ψ_L|²: Born probability density of the composite spinor.
        # This IS ρ_composite(x) -- the physical observable for a bound state.
        return np.abs(sum_R)**2 + np.abs(sum_L)**2

    def incoherent_density(self) -> np.ndarray:
        """
        Incoherent probability density: Σ_i (|ψ_R_i|² + |ψ_L_i|²)

        This is what you would measure if the sessions were independent.
        Compare with probability_density() to see the effect of phase
        cancellation: incoherent_density >= probability_density always,
        with equality only when all phases are aligned.
        """
        # Σ_i ρ_i(x): classical (incoherent) sum -- no cross-terms.
        # This IS the density of N independently prepared particles at the same location.
        # The ratio probability_density / incoherent_density measures phase alignment.
        return sum(s.probability_density() for s in self.sessions)

    # ── Composite cone properties ─────────────────────────────────────────────

    def effective_cone_half_angle(self) -> float:
        """
        Effective material cone half-angle of the composite.

        Computed from the coherent probability density's radial spread:
        a narrow distribution → small effective cone angle (appears massive).
        A neutral composite with cancelling phases should give a much smaller
        angle than any individual constituent.

        Returns the mean constituent angle as a baseline for comparison.
        (A full geometric computation requires knowing the lattice center.)
        """
        # arccos(√p_stay) for each constituent: the half-angle of the material cone.
        # Δ = arccos(√sin²(ω/2)) = π/2 - ω/2: smaller ω → wider cone → lighter (more relativistic).
        # Mean Δ̄ IS the effective cone angle of the composite before phase cancellation.
        # The coherent sum narrows this further; use cone_amplitude_profile for the full result.
        angles = [s.cone_half_angle for s in self.sessions]
        return float(np.mean(angles))   # baseline; use cone_amplitude_profile for true value

    def charge_balance(self) -> float:
        """
        Net charge proxy of the composite: Σ rgb_cmy_imbalance across constituents.

        Near zero → electrically neutral composite (phases tend to cancel).
        The magnitude of phase cancellation in probability_density() correlates
        with how close this is to zero.
        """
        # Σ_i (N_RGB_i - N_CMY_i): net sublattice imbalance across all constituents.
        # This IS the net phase chirality of the composite -- the discrete analogue of charge.
        # Neutral composite: Σ = 0 → phase gradients cancel in the coherent sum → narrow cone.
        return sum(s.rgb_cmy_imbalance for s in self.sessions)

    def phase_coherence(self) -> float:
        """
        Measure of phase alignment between constituent sessions.

        Returns the ratio: |Σ ψ_i|² / Σ|ψ_i|²  (integrated over lattice).

        1.0 → perfectly aligned phases (maximum constructive interference)
        0.0 → perfectly cancelling phases (maximum destructive interference)

        For a neutral composite, this approaches 0 — the constituents'
        phases cancel and the effective cone is narrow.  For a charged
        composite, this is larger — phases reinforce and the cone is wider.
        """
        # ∫|Σ ψ_i|² dV: total coherent probability (with interference terms).
        coherent   = float(self.probability_density().sum())
        # ∫Σ|ψ_i|² dV: total incoherent probability (no interference).
        incoherent = float(self.incoherent_density().sum())
        # Ratio = |Σψ|²/Σ|ψ|²: coherence measure in [0,1].
        # This IS the fringe visibility of a multi-path interference experiment:
        # 1.0 = fully constructive (all phases aligned, max cone width)
        # 0.0 = fully destructive (phases cancel, minimum effective cone)
        # For a neutral composite at binding_strength → 1: this approaches 0.
        return coherent / incoherent if incoherent > 1e-12 else 1.0
