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


def test_parity_field_shape_and_dtype(small_shape: tuple[int, int, int]) -> None:
    """parity_field returns an integer array of `shape`."""
    import numpy as np

    from dcl_core.core3d import BipartiteLattice

    parity = BipartiteLattice(shape=small_shape).parity_field()
    assert parity.shape == small_shape
    assert np.issubdtype(parity.dtype, np.integer)


def test_parity_field_origin_is_rgb() -> None:
    """Site (0, 0, 0) has x + y + z = 0 (even) -> RGB -> parity 0."""
    from dcl_core.core3d import BipartiteLattice

    parity = BipartiteLattice(shape=(4, 4, 4)).parity_field()
    assert int(parity[0, 0, 0]) == 0


def test_parity_field_values_are_binary(small_shape: tuple[int, int, int]) -> None:
    """Only 0 (RGB) and 1 (CMY) appear; no other integers leak through."""
    import numpy as np

    from dcl_core.core3d import BipartiteLattice

    parity = BipartiteLattice(shape=small_shape).parity_field()
    assert set(np.unique(parity).tolist()) == {0, 1}


def test_parity_flips_under_basis_hop(small_shape: tuple[int, int, int]) -> None:
    """Every RGB / CMY basis vector has odd component-sum, so hopping by one
    flips parity at every site.  This IS the bipartite chirality structure
    expressed at the lattice level; if it ever fails, either the basis
    vectors have changed or parity_field has decoupled from them.
    """
    import numpy as np

    from dcl_core.core3d import BipartiteLattice
    from dcl_core.core3d.lattice import ALL_VECTORS

    parity = BipartiteLattice(shape=small_shape).parity_field()
    for v in ALL_VECTORS:
        shifted = np.roll(parity, shift=v, axis=(0, 1, 2))
        # Every site of `shifted` should disagree with the corresponding
        # site of `parity` -- the XOR field is identically 1.
        assert np.all((parity ^ shifted) == 1), (
            f"parity did not flip under basis hop {v}: bipartite structure broken"
        )
