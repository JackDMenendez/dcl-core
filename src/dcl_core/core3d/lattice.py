"""Bipartite octahedral lattice geometry (T^3_diamond).

Lattice frame.  This IS Lambda_3, the d=3 instance of the diamond
progression Lambda_d (``dcl-mathematics``).  Its vertices split into two
parity classes V_3^+ / V_3^- by (x+y+z) mod 2 -- the RGB and CMY
sublattices -- which alternate as the active sublattice by tick parity.
Each site has coordination ``coord(d) = 2d = 6``: the six nearest-
neighbour simplex-edge generators (E_3),

    RGB (even ticks, V_3^+): V1=(1,1,1)   V2=(1,-1,-1)   V3=(-1,1,-1)
    CMY (odd ticks,  V_3^-): -V1, -V2, -V3

physics: the bipartite structure IS the Dirac structure -- gamma_5 acts
as +/-1 on the RGB/CMY sublattices, and the six basis vectors are the
gamma-matrix directions.  The continuum limit recovers (3+1)-dim
Minkowski space with the standard Clifford algebra.

See `docs/reference/lattice.md` for the geometric construction and
`docs/design/03_naming_convention.md` for the two-frame naming rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .backends import get_backend

# Simplex-edge generators E_3 of the two parity classes (the protected
# RGB/CMY lattice geometry; see docs/design/03_naming_convention.md).
# physics: these IS the framework's chirality structure -- the gamma-matrix
# directions of the bipartite Dirac operator.
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

    @property
    def coordination(self) -> int:
        """Neighbour count per site: ``coord(d) = 2d = 6`` at d=3.

        physics: the six gamma-matrix directions; the denominator of the
        Born-rule path count (6^N paths over N ticks).
        """
        return len(ALL_VECTORS)

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
