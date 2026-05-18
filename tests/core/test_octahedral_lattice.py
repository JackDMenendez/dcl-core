"""
tests/test_octahedral_lattice.py

Unit tests for OctahedralLattice and the module-level constants / helpers.

Theory correspondence:
  OctahedralLattice IS the T^3_diamond bipartite substrate.
  RGB_VECTORS and CMY_VECTORS are the two chiral sublattices.
  The topological_potential V(x,y,z) encodes clock density / Coulomb wells.
  The vector_potential A_mu encodes EM (curl deformation of phase field).
"""

import numpy as np
import pytest
from dcl_core.core.OctahedralLattice import (
    OctahedralLattice,
    RGB_VECTORS, CMY_VECTORS, ALL_VECTORS,
    SUBLATTICE_SIZE, COORDINATION_NUMBER,
    EVEN_TICK, ODD_TICK,
    active_vectors,
)


# ── Module-level constants ────────────────────────────────────────────────────

class TestBasisVectors:
    """RGB and CMY sublattice vectors are chiral partners: CMY = -RGB exactly."""

    def test_rgb_has_three_vectors(self):
        assert len(RGB_VECTORS) == 3

    def test_cmy_has_three_vectors(self):
        assert len(CMY_VECTORS) == 3

    def test_all_vectors_is_six(self):
        assert len(ALL_VECTORS) == 6

    def test_cmy_equals_neg_rgb(self):
        """CMY = -RGB: the two sublattices are exact chiral partners."""
        for rgb, cmy in zip(RGB_VECTORS, CMY_VECTORS):
            assert tuple(-np.array(rgb)) == cmy

    def test_rgb_vectors_correct_values(self):
        assert RGB_VECTORS[0] == (1, 1, 1)
        assert RGB_VECTORS[1] == (1, -1, -1)
        assert RGB_VECTORS[2] == (-1, 1, -1)

    def test_cmy_vectors_correct_values(self):
        assert CMY_VECTORS[0] == (-1, -1, -1)
        assert CMY_VECTORS[1] == (-1, 1, 1)
        assert CMY_VECTORS[2] == (1, -1, 1)

    def test_sublattice_size(self):
        assert SUBLATTICE_SIZE == 3

    def test_coordination_number(self):
        assert COORDINATION_NUMBER == 6

    def test_all_vectors_length_unit(self):
        """Every basis vector has length sqrt(3) (diagonal unit steps)."""
        for v in ALL_VECTORS:
            assert np.isclose(np.linalg.norm(v), np.sqrt(3))

    def test_rgb_cmy_sum_is_zero(self):
        """RGB + CMY vectors sum to zero: the lattice has no preferred direction."""
        total = np.sum(ALL_VECTORS, axis=0)
        np.testing.assert_array_equal(total, [0, 0, 0])

    def test_tick_parity_constants(self):
        assert EVEN_TICK == 0
        assert ODD_TICK == 1


class TestActiveVectors:
    """active_vectors selects the correct sublattice for a given tick parity."""

    def test_even_tick_returns_rgb(self):
        assert active_vectors(EVEN_TICK) == RGB_VECTORS

    def test_odd_tick_returns_cmy(self):
        assert active_vectors(ODD_TICK) == CMY_VECTORS


# ── OctahedralLattice construction ────────────────────────────────────────────

class TestOctahedralLatticeConstruction:

    def test_size_stored(self):
        lat = OctahedralLattice(5, 7, 9)
        assert lat.size_x == 5
        assert lat.size_y == 7
        assert lat.size_z == 9

    def test_topological_potential_shape(self):
        lat = OctahedralLattice(4, 4, 4)
        assert lat.topological_potential.shape == (4, 4, 4)

    def test_vector_potential_shape(self):
        lat = OctahedralLattice(4, 4, 4)
        assert lat.vector_potential.shape == (3, 4, 4, 4)

    def test_potentials_initialised_to_zero(self):
        lat = OctahedralLattice(4, 4, 4)
        assert np.all(lat.topological_potential == 0.0)
        assert np.all(lat.vector_potential == 0.0)


# ── Neighbor queries ──────────────────────────────────────────────────────────

class TestNeighborQueries:

    def setup_method(self):
        self.lat = OctahedralLattice(10, 10, 10)
        self.center = (5, 5, 5)

    def test_active_neighbors_even_tick_count(self):
        nb = self.lat.active_neighbors(self.center, EVEN_TICK)
        assert len(nb) == 3

    def test_active_neighbors_odd_tick_count(self):
        nb = self.lat.active_neighbors(self.center, ODD_TICK)
        assert len(nb) == 3

    def test_all_neighbors_count(self):
        nb = self.lat.all_neighbors(self.center)
        assert len(nb) == 6

    def test_active_neighbors_are_displaced_by_rgb(self):
        nb = self.lat.active_neighbors(self.center, EVEN_TICK)
        cx, cy, cz = self.center
        for (nx, ny, nz), (dx, dy, dz) in zip(nb, RGB_VECTORS):
            assert nx == cx + dx
            assert ny == cy + dy
            assert nz == cz + dz

    def test_active_neighbors_boundary_clipping(self):
        """Nodes at the boundary lose some neighbors (no wrapping)."""
        nb = self.lat.active_neighbors((0, 0, 0), EVEN_TICK)
        # RGB vectors include (1,1,1) → valid; (1,-1,-1) → y=-1 invalid; (-1,1,-1) → x=-1 invalid
        assert len(nb) == 1  # only (1,1,1) is in bounds

    def test_all_neighbors_boundary_clipping(self):
        """Corner node (0,0,0) has fewer than 6 valid neighbours."""
        nb = self.lat.all_neighbors((0, 0, 0))
        # Check all returned nodes are in bounds
        for (nx, ny, nz) in nb:
            assert 0 <= nx < 10
            assert 0 <= ny < 10
            assert 0 <= nz < 10


# ── Topological potential setters ─────────────────────────────────────────────

class TestSetClockDensityWell:

    def setup_method(self):
        self.lat = OctahedralLattice(15, 15, 15)

    def test_peak_at_center(self):
        center = (7, 7, 7)
        self.lat.set_clock_density_well(center, width=2.0, depth=1.0)
        cx, cy, cz = center
        peak_val = self.lat.topological_potential[cx, cy, cz]
        assert peak_val == pytest.approx(1.0, abs=1e-10)

    def test_falls_off_from_center(self):
        center = (7, 7, 7)
        self.lat.set_clock_density_well(center, width=2.0, depth=1.0)
        cx, cy, cz = center
        nearby = self.lat.topological_potential[cx+1, cy, cz]
        assert nearby < self.lat.topological_potential[cx, cy, cz]

    def test_adds_to_existing_potential(self):
        """Two wells accumulate."""
        center = (7, 7, 7)
        self.lat.set_clock_density_well(center, width=2.0, depth=1.0)
        v1 = self.lat.topological_potential[7, 7, 7]
        self.lat.set_clock_density_well(center, width=2.0, depth=1.0)
        v2 = self.lat.topological_potential[7, 7, 7]
        assert np.isclose(v2, 2 * v1)


class TestSetCoulombWell:

    def setup_method(self):
        self.lat = OctahedralLattice(15, 15, 15)

    def test_negative_at_center(self):
        """Coulomb well is attractive (negative potential at centre)."""
        center = (7, 7, 7)
        self.lat.set_coulomb_well(center, strength=30.0, softening=0.5)
        v_center = self.lat.topological_potential[7, 7, 7]
        assert v_center < 0.0

    def test_value_at_center(self):
        center = (7, 7, 7)
        self.lat.set_coulomb_well(center, strength=30.0, softening=0.5)
        # r=0 at center, V = -30 / (0 + 0.5)
        expected = -30.0 / 0.5
        assert self.lat.topological_potential[7, 7, 7] == pytest.approx(expected)

    def test_falls_off_with_distance(self):
        """Potential rises (becomes less negative) with distance."""
        center = (7, 7, 7)
        self.lat.set_coulomb_well(center, strength=30.0, softening=0.5)
        v0 = self.lat.topological_potential[7, 7, 7]
        v1 = self.lat.topological_potential[8, 7, 7]
        v3 = self.lat.topological_potential[10, 7, 7]
        assert v0 < v1 < v3

    def test_approaches_zero_far_from_center(self):
        """Coulomb well is negligible far from the center."""
        center = (7, 7, 7)
        self.lat.set_coulomb_well(center, strength=1.0, softening=0.5)
        v_far = self.lat.topological_potential[0, 0, 0]
        assert abs(v_far) < 0.1


class TestSetEMTwist:

    def setup_method(self):
        self.lat = OctahedralLattice(15, 15, 15)

    def test_vector_potential_nonzero_after_twist(self):
        self.lat.set_em_twist((7, 7, 7), width=3.0, strength=1.0, axis=2)
        assert np.any(self.lat.vector_potential != 0.0)

    def test_asymmetry_of_tangential_components(self):
        """For axis=2, components 0 and 1 of A are equal and opposite at center."""
        center = (7, 7, 7)
        self.lat.set_em_twist(center, width=3.0, strength=1.0, axis=2)
        a0 = self.lat.vector_potential[0, 7, 7, 7]
        a1 = self.lat.vector_potential[1, 7, 7, 7]
        assert np.isclose(a0, -a1)

    def test_topological_potential_unchanged(self):
        """EM twist should not modify the scalar (gravitational) potential."""
        self.lat.set_em_twist((7, 7, 7), width=3.0, strength=1.0)
        assert np.all(self.lat.topological_potential == 0.0)


# ── Point accessors ───────────────────────────────────────────────────────────

class TestPointAccessors:

    def setup_method(self):
        self.lat = OctahedralLattice(10, 10, 10)

    def test_clock_density_at_default(self):
        assert self.lat.clock_density_at((5, 5, 5)) == 0.0

    def test_clock_density_at_after_well(self):
        self.lat.set_coulomb_well((5, 5, 5), strength=10.0, softening=1.0)
        assert self.lat.clock_density_at((5, 5, 5)) == pytest.approx(-10.0)

    def test_vector_potential_at_default(self):
        vp = self.lat.vector_potential_at((5, 5, 5))
        np.testing.assert_array_equal(vp, [0.0, 0.0, 0.0])

    def test_vector_potential_at_shape(self):
        vp = self.lat.vector_potential_at((3, 3, 3))
        assert vp.shape == (3,)


# ── Causal cone (BFS) ─────────────────────────────────────────────────────────

class TestCausalCone:

    def setup_method(self):
        self.lat = OctahedralLattice(20, 20, 20)
        self.origin = (10, 10, 10)

    def test_zero_ticks_returns_origin(self):
        nodes = self.lat.causal_cone_nodes(self.origin, n_ticks=0)
        assert nodes == [self.origin]

    def test_one_tick_contains_six_neighbors_plus_origin(self):
        nodes = self.lat.causal_cone_nodes(self.origin, n_ticks=1)
        # origin + 6 neighbors = 7 nodes (all in bounds for central node)
        assert len(nodes) == 7

    def test_one_tick_origin_included(self):
        nodes = self.lat.causal_cone_nodes(self.origin, n_ticks=1)
        assert self.origin in nodes

    def test_cone_grows_with_ticks(self):
        n1 = self.lat.causal_cone_nodes(self.origin, n_ticks=1)
        n2 = self.lat.causal_cone_nodes(self.origin, n_ticks=2)
        assert len(n2) > len(n1)

    def test_cone_bounded_by_lattice(self):
        """No node in the cone exceeds lattice bounds."""
        nodes = self.lat.causal_cone_nodes(self.origin, n_ticks=3)
        for (nx, ny, nz) in nodes:
            assert 0 <= nx < 20
            assert 0 <= ny < 20
            assert 0 <= nz < 20
