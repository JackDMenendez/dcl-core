"""Bipartite octahedral lattice geometry (T^3_diamond).

The lattice IS the substrate on which causal sessions live. Two
sublattices RGB and CMY alternate by tick parity; the six nearest-
neighbour basis vectors split chirally.

    RGB (even ticks): V1=(1,1,1)   V2=(1,-1,-1)   V3=(-1,1,-1)
    CMY (odd ticks):  -V1, -V2, -V3

The bipartite structure IS the Dirac structure -- gamma_5 acts as
multiplication by +/-1 on the RGB/CMY sublattices. The continuum limit
recovers (3+1)-dim Minkowski space with the standard Clifford algebra.

See `docs/reference/lattice.md` for the geometric construction and
`docs/design/03_naming_convention.md` for naming.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .backends import get_backend

# Sublattice basis vectors. These IS the framework's chirality structure.
# Do not reorder, do not renormalise -- downstream code identifies sublattice
# by sign of the first nonzero component.
RGB_VECTORS: tuple[tuple[int, int, int], ...] = (
    (1, 1, 1),
    (1, -1, -1),
    (-1, 1, -1),
)
CMY_VECTORS: tuple[tuple[int, int, int], ...] = (
    (-1, -1, -1),
    (-1, 1, 1),
    (1, -1, 1),
)
ALL_VECTORS: tuple[tuple[int, int, int], ...] = RGB_VECTORS + CMY_VECTORS

# Sublattice parity: at even ticks, sessions propagate along RGB; at odd
# ticks, along CMY. This IS gamma_5 acting on the bipartite state.
TickParity = Literal["even", "odd"]


@dataclass(frozen=True)
class BipartiteLattice:
    """A finite cube of the T^3_diamond lattice with periodic boundaries.

    Attributes
    ----------
    shape : tuple[int, int, int]
        Number of sites along each Cartesian axis. The total site
        count is the product of these; sublattice parity is set by
        (x + y + z) mod 2.
    backend : str
        Numerical backend, "cpu" (NumPy) or "gpu" (CuPy). Default "cpu".

    Notes
    -----
    This class is a thin, frozen description of geometry. State (a
    session's amplitude / token counts) lives in
    :class:`~dcl_core.session.DiscreteCausalSession`, never on the
    lattice object itself.
    """

    shape: tuple[int, int, int]
    backend: str = "cpu"

    @property
    def n_sites(self) -> int:
        """Total number of lattice sites (product of `shape`)."""
        nx, ny, nz = self.shape
        return nx * ny * nz

    def parity_field(self) -> np.ndarray:
        """Return a `shape`-sized int array with 0 on RGB sites, 1 on CMY.

        Site (x, y, z) is on RGB iff (x + y + z) is even. This IS the
        Z_2 grading that becomes gamma_5 in the continuum limit.
        """
        backend = get_backend(self.backend)
        coords = backend.indices(self.shape)
        return coords.sum(axis=0) % 2

    def neighbour_offsets(self, parity: TickParity) -> tuple[tuple[int, int, int], ...]:
        """Return the basis vectors used for hopping at the given tick parity.

        Even tick -> RGB vectors (`RGB_VECTORS`).
        Odd tick  -> CMY vectors (`CMY_VECTORS`).
        """
        if parity == "even":
            return RGB_VECTORS
        if parity == "odd":
            return CMY_VECTORS
        raise ValueError(f"parity must be 'even' or 'odd', got {parity!r}")
