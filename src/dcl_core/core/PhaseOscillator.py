"""
PhaseOscillator.py
The U(1) Internal Clock: the particle's phase state keeper.

The PhaseOscillator is the internal oscillator of a CausalSession.
Its frequency omega is the particle's instruction overhead -- rest mass.
Its current phase encodes the particle's internal time coordinate.

A high-frequency oscillator (heavy particle) has high phase stability:
it resists redirection (inertia). A low-frequency oscillator (light particle)
is easily scattered by minor potential gradients.

DOCUMENTATION CONVENTION:
  Every non-trivial line of physics code should say what it IS in the theory,
  not just what it does in the program.  Name the mathematical object, cite
  the paper equation where one exists, and state the correspondence explicitly:
  "this IS X" when exact, "this approximates X" in the continuum limit.
  The structure factor comment in CausalSession._kinetic_hop is the template.

Paper reference: Section 3 (Phase Oscillator, mass as instruction overhead)
"""

import numpy as np


class PhaseOscillator:
    """
    U(1) internal clock for a CausalSession.

    The oscillator lives on the complex unit circle: r = 1 always (A=1).
    Phase advances by omega at every tick.
    """

    def __init__(self, frequency: float):
        """
        Parameters
        ----------
        frequency : omega -- instruction overhead, encoding rest mass.
                    Higher omega = heavier particle = greater phase stability.
        """
        self.omega = frequency          # Instruction frequency (rest mass)
        self.phase = 0.0                # Current phase angle (radians)

    @property
    def amplitude(self) -> complex:
        """
        The oscillator state as a unit complex number.
        Always on the unit circle: |amplitude| = 1 (enforces A=1 locally).
        """
        return np.exp(1j * self.phase)

    def advance(self):
        """
        Advance the oscillator by one tick.
        Phase increments by omega, wrapping at 2*pi.

        This is left-multiplication by exp(i*omega) in U(1):
          r_{n+1} = exp(i*omega) * r_n
        The group element r_n tracks accumulated phase; omega is the
        algebra generator that produces the advance.
        Paper: eq. (oscillator_advance)
        """
        self.phase = (self.phase + self.omega) % (2 * np.pi)

    def phase_cost(self, local_potential: float) -> float:
        """
        The discrete Hamiltonian H at a node:  H = omega + V(x,y,z)

        This is the Lie algebra element delta_phi in u(1) that governs
        the evolution at this node.  It is the phase transition cost per tick:
          - omega alone: free-particle rest energy (mass term)
          - V(x,y,z):    local potential energy (Coulomb, gravity, EM)
          - omega + V:   total local Hamiltonian

        The Hamiltonian is an algebra element (a real number in u(1)).
        The evolution operator exp(i*H) is the corresponding group element.
        CausalSession.tick() computes cos(H/2) and sin(H/2), which are the
        matrix elements of exp(i*H/2*sigma_x) -- the spinor evolution operator.

        Paper: eq. (phase_mismatch)  delta_phi = omega + V
               Section 3.1 (phase mismatch as local Hamiltonian)
        """
        return self.omega + local_potential

    def phase_shift(self, local_potential: float) -> complex:
        """
        The unitary evolution operator U = exp(i*H) as a U(1) element.

        This is the Lie group element corresponding to the algebra element
        H = phase_cost(local_potential).  Exponentiates the Hamiltonian
        into a phase rotation.

        Note: CausalSession.tick() uses cos(H/2) and sin(H/2) rather than
        this scalar exp(iH), because the spinor evolution is a 2x2 SU(2)
        rotation exp(i*H/2*sigma_x), not a scalar U(1) phase.  This method
        is provided for scalar oscillator calculations and diagnostic use.
        """
        return np.exp(1j * self.phase_cost(local_potential))
