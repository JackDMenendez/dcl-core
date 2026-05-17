"""
OctahedralLattice.py
The Primal 3D Topology: T^3_diamond (Bipartite Structure)

The substrate of the universe modeled as a bipartite 3D lattice.
The six basis vectors split into two chirally-opposite sets:

  RGB sublattice (even ticks):  V1=(1,1,1)  V2=(1,-1,-1)  V3=(-1,1,-1)
  CMY sublattice (odd ticks):  -V1=(-1,-1,-1) -V2=(-1,1,1) -V3=(1,-1,1)

CMY = -RGB exactly. The bipartite structure is the geometric origin of:
  - The speed of light (photons always move, never stay)
  - Chirality (photons see only 3 vectors per tick)
  - Mass (massive particles blur across both sublattices via Zitterbewegung)
  - The distinction between EM (curl of phase field) and gravity (div)

DOCUMENTATION CONVENTION:
  Every non-trivial line of physics code should say what it IS in the theory,
  not just what it does in the program.  Name the mathematical object, cite
  the paper equation where one exists, and state the correspondence explicitly:
  "this IS X" when exact, "this approximates X" in the continuum limit.
  The structure factor comment in CausalSession._kinetic_hop is the template.

Paper reference: Section 2 (The Octahedral Substrate, Bipartite Structure)
"""

import numpy as np
from typing import Tuple, List

# ── Bipartite basis vectors (from T3d diagram) ────────────────────────────────

# The six basis vectors of T^3_diamond: the causal adjacency structure.
# These ARE the gamma matrix directions of the Dirac equation --
# RGB encodes the spatial part of gamma^1, gamma^2, gamma^3 via the
# outer product structure sum_RGB v_i v_j (the frame matrix M).
# CMY = -RGB exactly: the two sublattices are the chiral partners psi_R / psi_L.
# Paper: Section 2 (octahedral_substrate.tex), Section 4 (emergent_kinematics.tex)
RGB_VECTORS = [
    ( 1,  1,  1),   # V1  RED   -- gamma direction 1
    ( 1, -1, -1),   # V2  GREEN -- gamma direction 2
    (-1,  1, -1),   # V3  BLUE  -- gamma direction 3
]

CMY_VECTORS = [
    (-1, -1, -1),   # -V1  CYAN    -- chiral partner of V1
    (-1,  1,  1),   # -V2  MAGENTA -- chiral partner of V2
    ( 1, -1,  1),   # -V3  YELLOW  -- chiral partner of V3
]

ALL_VECTORS         = RGB_VECTORS + CMY_VECTORS   # full causal neighbourhood of a node
SUBLATTICE_SIZE     = len(RGB_VECTORS)    # 3: one tick accesses 3 vectors (one sublattice)
COORDINATION_NUMBER = len(ALL_VECTORS)    # 6: total causal neighbours = denominator of Born rule (6^N paths)
EVEN_TICK = 0   # RGB sublattice active (psi_R updated)
ODD_TICK  = 1   # CMY sublattice active (psi_L updated)


def active_vectors(tick_parity: int) -> List[Tuple[int, int, int]]:
    """
    Returns the 3 active basis vectors for this tick's sublattice.
    Photons (massless) only ever see these 3 -- strict chirality.
    """
    return RGB_VECTORS if tick_parity == EVEN_TICK else CMY_VECTORS


class OctahedralLattice:
    """
    The discrete 3D causal substrate T^3_diamond with bipartite structure.

    Nodes are causal events. Edges are causal adjacency only.
    The topological_potential V(x,y,z) encodes clock density (gravity) --
    a divergence deformation of the phase field.
    The vector_potential A(x,y,z) encodes electromagnetism --
    a curl deformation of the phase field.
    """

    def __init__(self, size_x: int, size_y: int, size_z: int):
        self.size_x = size_x
        self.size_y = size_y
        self.size_z = size_z

        # V(x,y,z): the topological potential -- Lie algebra element V in u(1).
        # This IS the potential energy term in the discrete Hamiltonian H = omega + V.
        # Physically: clock density gradient (gravity) or Coulomb well (EM charge).
        # Geometrically: a DIV deformation of the phase field -- sources / sinks.
        # The phase mismatch delta_phi = omega + V determines p_stay and p_hop.
        # Paper: eq. (phase_mismatch), Section 7 (gravity_as_clock_density.tex)
        self.topological_potential = np.zeros(
            (size_x, size_y, size_z), dtype=float
        )

        # A(x,y,z): the electromagnetic vector potential -- shape (3, X, Y, Z).
        # This IS the gauge field A_mu of U(1) electromagnetism.
        # Geometrically: a CURL deformation of the phase field -- no sources/sinks.
        # Applied via the Peierls substitution: delta_p -> delta_p + A.v for each
        # hop direction v, biasing hops perpendicular to A (Lorentz force).
        # Paper: Section 3.3 (tick rule as U(1) gauge transformation)
        self.vector_potential = np.zeros(
            (3, size_x, size_y, size_z), dtype=float
        )

    def active_neighbors(self, node: Tuple[int,int,int],
                         tick_parity: int) -> List[Tuple[int,int,int]]:
        """
        Returns the 3 causally active neighbors for this tick's sublattice.
        Photons use only these 3. Massive particles use all 6.
        """
        x, y, z = node
        neighbors = []
        for dx, dy, dz in active_vectors(tick_parity):
            nx, ny, nz = x+dx, y+dy, z+dz
            if 0 <= nx < self.size_x and 0 <= ny < self.size_y and 0 <= nz < self.size_z:
                neighbors.append((nx, ny, nz))
        return neighbors

    def all_neighbors(self, node: Tuple[int,int,int]) -> List[Tuple[int,int,int]]:
        """
        All 6 causal neighbors (both sublattices).
        Used by massive particles whose Zitterbewegung blurs both parities.
        """
        x, y, z = node
        neighbors = []
        for dx, dy, dz in ALL_VECTORS:
            nx, ny, nz = x+dx, y+dy, z+dz
            if 0 <= nx < self.size_x and 0 <= ny < self.size_y and 0 <= nz < self.size_z:
                neighbors.append((nx, ny, nz))
        return neighbors

    def set_clock_density_well(self, center: Tuple, width: float, depth: float):
        """
        Gaussian clock density well -- gravitational source.
        DIV deformation of the phase field.
        """
        cx, cy, cz = center
        x = np.arange(self.size_x)
        y = np.arange(self.size_y)
        z = np.arange(self.size_z)
        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        r_sq = (xx - cx)**2 + (yy - cy)**2 + (zz - cz)**2
        # Gaussian clock density profile: rho_clock(r) = depth * exp(-r^2 / 2*width^2)
        # This IS a gravitational source in the clock-fluid picture.
        # div(grad V) = rho_clock via the Poisson equation (paper Section 7.3).
        self.topological_potential += depth * np.exp(-0.5 * r_sq / width**2)

    def set_coulomb_well(self, center: Tuple, strength: float,
                          softening: float = 1.0):
        """
        Creates a 1/r clock density well -- Coulombic gravitational source.

        V(r) = -strength / (r + softening)

        Unlike the Gaussian well (exp_02), this falls off as 1/r and
        supports an infinite series of bound states at E_n ~ -1/n^2.
        The softening parameter regularizes the singularity at r=0.

        This is the correct potential for deriving the hydrogen spectrum.
        The Gaussian well has only one characteristic depth; the Coulomb
        well has a ladder of quantized bound state energies.

        Paper reference: Section 11 (Hydrogen Spectrum from Lattice Geometry)
        """
        cx, cy, cz = center
        x = np.arange(self.size_x)
        y = np.arange(self.size_y)
        z = np.arange(self.size_z)
        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        r = np.sqrt((xx-cx)**2 + (yy-cy)**2 + (zz-cz)**2)
        # V(r) = -strength / (r + softening) -- the Coulomb potential.
        # This IS the discrete analogue of the hydrogen atom potential e^2/r.
        # Negative sign: well deepens toward center (attractive).
        # Softening regularises the r=0 singularity (lattice UV cutoff).
        # The 1/n^2 hydrogen spectrum arises from the infinite series of
        # bound states of this potential. Paper: Section 10 (hydrogen_spectrum.tex)
        self.topological_potential += -strength / (r + softening)

    def set_em_twist(self, center: Tuple, width: float,
                     strength: float, axis: int = 2):
        """
        Gaussian curl deformation -- electromagnetic source.
        CURL deformation of the phase field.
        Positive/negative strength = opposite charge signs.
        axis: rotation axis (0=x, 1=y, 2=z)
        """
        cx, cy, cz = center
        tang_axes = [a for a in [0,1,2] if a != axis]
        a1, a2 = tang_axes
        coords = np.mgrid[0:self.size_x, 0:self.size_y, 0:self.size_z]
        r_sq = (coords[0]-cx)**2 + (coords[1]-cy)**2 + (coords[2]-cz)**2
        # A(x) = strength * exp(-r^2/2w^2): Gaussian envelope of the EM field.
        # The curl A_a1 - A_a2 is the magnetic field B in the axis direction.
        # This IS the discrete gauge field A_mu; its curl is the Faraday tensor F_mu_nu.
        # Applied to hops via the Peierls substitution delta_p -> delta_p + A.v.
        envelope = strength * np.exp(-0.5 * r_sq / width**2)
        self.vector_potential[a1] += envelope   # +A component -> reinforces hops in a1
        self.vector_potential[a2] -= envelope   # -A component -> suppresses hops in a2
        # The asymmetry IS the Lorentz force: deflects amplitude perp. to both v and B.

    def clock_density_at(self, node: Tuple[int,int,int]) -> float:
        x, y, z = node
        return self.topological_potential[x, y, z]

    def vector_potential_at(self, node: Tuple[int,int,int]) -> np.ndarray:
        x, y, z = node
        return self.vector_potential[:, x, y, z]

    def causal_cone_nodes(self, origin: Tuple[int,int,int],
                          n_ticks: int) -> List[Tuple]:
        """
        BFS over actual diagonal step vectors.
        The causal cone with these basis vectors is a rhombic dodecahedron.
        """
        if n_ticks == 0:
            return [origin]
        reachable = {origin}
        frontier  = {origin}
        for _ in range(n_ticks):
            new_frontier = set()
            for node in frontier:
                x, y, z = node
                for dx, dy, dz in ALL_VECTORS:
                    nx, ny, nz = x+dx, y+dy, z+dz
                    c = (nx, ny, nz)
                    if (0 <= nx < self.size_x and
                            0 <= ny < self.size_y and
                            0 <= nz < self.size_z and
                            c not in reachable):
                        reachable.add(c)
                        new_frontier.add(c)
            frontier = new_frontier
        return list(reachable)
