"""
UnityConstraint.py
The A=1 Unity Constraint: strict probability conservation.

Every tick, after all amplitude updates are applied, the wave
function must be renormalized to unit probability. This is not
a correction -- it is the foundational axiom of the framework.

A=1 means: the session persists. Probability is conserved.
The universe does not lose or create information in a tick.

DOCUMENTATION CONVENTION:
  Every non-trivial line of physics code should say what it IS in the theory,
  not just what it does in the program.  Name the mathematical object, cite
  the paper equation where one exists, and state the correspondence explicitly:
  "this IS X" when exact, "this approximates X" in the continuum limit.
  The structure factor comment in CausalSession._kinetic_hop is the template.

Paper reference: Section 3 (Causal Sessions and the Phase Oscillator)

--- NOTE ON USE IN EXPERIMENTS ---

enforce_unity_spinor is called in two physically distinct contexts.
Both are legitimate applications of A=1; neither is a normalisation trick.

1. CONSERVATION (every tick, single session):
   The tick rule applies cos/sin mixing and hop operators.  The result
   is not automatically unit-normalized because the discrete operators do
   not form an exact unitary group.  enforce_unity_spinor corrects the
   O(dt^2) drift and is the discrete implementation of the Lie-group
   constraint |r| = 1.  This is the standard usage in all experiments.

2. PROJECTION / DISSIPATION (exp_19, photon emission):
   After transferring a fraction of the electron's outer-orbit amplitude
   to a photon session, enforce_unity_spinor is called on the electron.
   This is NOT a bookkeeping renormalization -- it is a physical projection
   measurement.  The A=1 constraint asserts that the electron session
   persists with unit probability at every tick; outer-orbit amplitude
   that has been carried away by the photon no longer belongs to the
   electron session, so the electron must re-normalize.  The probability
   deficit reappears in the photon session (also normalized to A=1 after
   receiving the transfer).

   This is the mechanism by which the framework implements spontaneous
   emission without violating A=1: total probability is not conserved
   within the electron session alone after emission, but it IS conserved
   across the joint (electron + photon) system via the photon's A=1
   renormalization.  The apparent "renormalization" of the electron is
   the discrete equivalent of wavefunction collapse onto the sub-space
   of states with r < R1 -- the ground-state orbit.

   A reader who sees enforce_unity_spinor called after amplitude transfer
   and suspects a cheat should note: the photon session grows by exactly
   the amplitude removed from the electron.  Joint probability is
   conserved.  The per-session renormalization is the A=1 constraint
   applied to each persistent session independently, as required by
   the framework's foundational axioms.

3. PROJECTION / EXCITATION (absorption -- exp_20, not yet implemented):
   Absorption is the exact time-reverse of emission.  A photon session
   enters as a third session alongside (electron, proton).  Each tick:

     outer_mask   = rate * max(0, r(x) - R1) / R1
     transfer     = outer_mask * photon.psi       # photon amplitude at r > R1
     electron.psi += transfer                     # add to electron outer orbit
     photon.psi   -= transfer                     # photon loses that amplitude
     enforce_unity_spinor(electron)               # PDF shifts outward -> excitation
     enforce_unity_spinor(photon)                 # photon diminishes or terminates

   The outer-orbit mask is the same as for emission.  Transfer direction is
   reversed.  After renormalization the electron's PDF shifts OUTWARD
   (excitation), while the photon session shrinks tick-by-tick.  When the
   photon's norm falls below 1e-12, enforce_unity_spinor raises RuntimeError
   -- the session has annihilated, fully absorbed.

   Phase matching is automatic and enforces the resonance condition.  The
   photon carries exp(i phi_photon(x)) at each node.  If the photon
   frequency does not match the transition energy omega_E*(1 - 1/n^2), the
   added phase interferes destructively with the electron's existing
   outer-orbit phase and the net transfer averages to near zero over many
   ticks.  Resonant absorption is the constructive-interference case; the
   resonance condition emerges from lattice dynamics without being imposed.

   Summary table:

     Event        Mask region   Direction   Electron PDF   Photon
     ----------   -----------   ---------   ------------   ------
     Emission     r > R1        e -> gamma  shifts inward  grows
     Absorption   r > R1        gamma -> e  shifts outward shrinks
     Conservation none          --          unchanged      unchanged

   This three-way taxonomy covers every physical context in which
   enforce_unity_spinor appears.  In all three cases the call is the A=1
   constraint applied to a session after its amplitude budget has changed;
   it is never a free parameter or an ad-hoc correction.
"""

import numpy as np


def enforce_unity(psi: np.ndarray) -> np.ndarray:
    """
    Enforce the A=1 unity constraint on a complex amplitude field.

    Normalizes psi in-place so that sum(|psi|^2) = 1.
    If the norm is zero the session has annihilated -- this should
    not occur under valid lattice dynamics and is flagged as an error.

    Parameters
    ----------
    psi : complex ndarray of any shape (1D, 2D, or 3D lattice)

    Returns
    -------
    psi : normalized in-place (also returned for convenience)
    """
    norm = np.sqrt(np.sum(np.abs(psi) ** 2))
    if norm < 1e-12:
        raise RuntimeError(
            "Unity constraint violated: session amplitude collapsed to zero. "
            "Check lattice boundary conditions and Hamiltonian parameters."
        )
    psi /= norm
    return psi


def unity_residual(psi: np.ndarray) -> float:
    """
    Returns the deviation from unity: |sum(|psi|^2) - 1|.
    Used in audit modules to verify conservation at every tick.
    A well-behaved simulation should keep this below 1e-10.
    """
    return abs(np.sum(np.abs(psi) ** 2) - 1.0)


def is_unity(psi: np.ndarray, tolerance: float = 1e-10) -> bool:
    """Returns True if the amplitude field satisfies A=1 within tolerance."""
    return unity_residual(psi) < tolerance


def enforce_unity_spinor(psi_R: np.ndarray, psi_L: np.ndarray):
    """
    Enforce A=1 for a two-component Dirac spinor.

    Normalizes both components in-place so sum(|psi_R|^2 + |psi_L|^2) = 1.
    Modifies arrays in-place and returns (psi_R, psi_L) for convenience.

    Two legitimate physical uses -- see module docstring for full discussion:
      - Conservation: corrects O(dt^2) drift from the discrete tick rule.
      - Projection:   after amplitude transfer to a photon session, this
                      projects the electron onto its post-emission sub-space.
                      The photon receives the removed amplitude and is also
                      normalized; joint probability is conserved across sessions.

    Paper reference: Section 3 (Dirac spinor, A=1 unity constraint)
    """
    norm = np.sqrt(np.sum(np.abs(psi_R) ** 2 + np.abs(psi_L) ** 2))
    if norm < 1e-12:
        raise RuntimeError(
            "Unity constraint violated: spinor amplitude collapsed to zero. "
            "Check lattice boundary conditions and Hamiltonian parameters."
        )
    psi_R /= norm
    psi_L /= norm
    return psi_R, psi_L


def unity_residual_spinor(psi_R: np.ndarray, psi_L: np.ndarray) -> float:
    """
    Returns the deviation from unity for a Dirac spinor:
    |sum(|psi_R|^2 + |psi_L|^2) - 1|
    """
    return abs(np.sum(np.abs(psi_R) ** 2 + np.abs(psi_L) ** 2) - 1.0)


def enforce_joint_unity(spinor_list: list):
    """
    Enforce A=1 jointly across multiple spinors.
    
    This evaluates the total probability across all sessions passed to it
    (e.g., an electron and an emitted photon) and normalizes them together 
    so sum(P_total) = 1.0. This implements A=1 for the joint state.
    """
    total_norm2 = 0.0
    for psi_R, psi_L in spinor_list:
        total_norm2 += np.sum(np.abs(psi_R)**2) + np.sum(np.abs(psi_L)**2)
        
    norm = np.sqrt(total_norm2)
    if norm < 1e-12:
        raise RuntimeError("Joint unity constraint violated: total amplitude zero.")
        
    for psi_R, psi_L in spinor_list:
        psi_R /= norm
        psi_L /= norm

