"""Tests for the bipartite lattice geometry.

These run against the lattice description only (no session state),
so they're cheap and live near the top of the test ordering.
"""

from __future__ import annotations

import pytest


def test_basis_vectors_are_chiral_opposites() -> None:
    """Each CMY vector is the negation of the corresponding RGB vector.

    This is a structural invariant: the bipartite split IS chirality.
    If anyone reorders the constants in `lattice.py`, this catches it.
    """
    from dcl_core.core3d.lattice import CMY_VECTORS, RGB_VECTORS

    assert len(RGB_VECTORS) == 3
    assert len(CMY_VECTORS) == 3
    for rgb, cmy in zip(RGB_VECTORS, CMY_VECTORS, strict=True):
        negated = tuple(-c for c in rgb)
        assert negated == cmy, (
            f"RGB {rgb} is not the negative of CMY {cmy}; "
            "the bipartite chiral structure is broken."
        )


def test_all_vectors_is_rgb_then_cmy() -> None:
    """`ALL_VECTORS == RGB + CMY`, in that order. Order is load-bearing."""
    from dcl_core.core3d.lattice import ALL_VECTORS, CMY_VECTORS, RGB_VECTORS

    assert ALL_VECTORS == RGB_VECTORS + CMY_VECTORS


def test_n_sites_is_product_of_shape(small_shape: tuple[int, int, int]) -> None:
    """`BipartiteLattice.n_sites` matches the product of `shape`."""
    from dcl_core.core3d import BipartiteLattice

    lattice = BipartiteLattice(shape=small_shape)
    nx, ny, nz = small_shape
    assert lattice.n_sites == nx * ny * nz


def test_neighbour_offsets_matches_parity() -> None:
    """RGB offsets returned for even tick; CMY offsets for odd tick."""
    from dcl_core.core3d import BipartiteLattice
    from dcl_core.core3d.lattice import CMY_VECTORS, RGB_VECTORS

    lattice = BipartiteLattice(shape=(4, 4, 4))
    assert lattice.neighbour_offsets("even") == RGB_VECTORS
    assert lattice.neighbour_offsets("odd") == CMY_VECTORS


def test_neighbour_offsets_rejects_bad_parity() -> None:
    """A parity other than 'even' or 'odd' raises ValueError."""
    from dcl_core.core3d import BipartiteLattice

    lattice = BipartiteLattice(shape=(4, 4, 4))
    with pytest.raises(ValueError):
        lattice.neighbour_offsets("middle")  # type: ignore[arg-type]
