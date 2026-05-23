"""Dirac gamma matrices and the Minkowski metric (symbolic, sympy).

Why this lives in its own module
--------------------------------
sympy is an **optional** dependency (declared in the ``[sympy]`` and
``[dev]`` extras in ``pyproject.toml``, not in ``project.dependencies``).
Importing it at the top of :mod:`dcl_core.core3d.hop` would force every
user of the engine to install sympy, which is overkill -- the engine
itself never touches it.  Keeping the gammas here means symbolic
verification machinery is opt-in: users do
``from dcl_core.core3d.clifford import gamma_0`` explicitly.

The module itself imports cleanly even when sympy is not installed;
the sympy import is deferred until one of the symbolic attributes
(``gamma_0``, ``gamma_1``, ..., ``ETA_MINKOWSKI``, ``GAMMAS``) is
actually accessed.  Missing-sympy gives a clear
``ImportError`` with an install hint at access time, not at module
load time.

Convention
----------
Standard Dirac representation, **mostly-minus** Minkowski metric
``eta = diag(+1, -1, -1, -1)``::

    gamma_0 = diag(I_2, -I_2) = diag(1, 1, -1, -1)
    gamma_i = (( 0,    sigma_i),
               (-sigma_i,  0  ))     for i in {1, 2, 3}

where ``sigma_i`` are the standard 2x2 Pauli matrices.  The matrices
satisfy the Clifford anticommutation relation::

    {gamma_mu, gamma_nu} = 2 * eta[mu, nu] * I_4

which :mod:`tests/core3d/test_clifford.py` verifies symbolically (no
float tolerance).

If the framework's downstream physics needs a different convention
(mostly-plus, chiral representation, Majorana, ...), add the
alternative matrices here next to these rather than mutating these.
The mostly-minus Dirac form is what Paper~I uses; consistency with
Paper~I matters for the discrete-vs-continuous comparison in Paper III.
"""

from __future__ import annotations

from typing import Any

_SYMBOLIC_NAMES = frozenset(
    {
        "gamma_0",
        "gamma_1",
        "gamma_2",
        "gamma_3",
        "ETA_MINKOWSKI",
        "GAMMAS",
    }
)

_CACHED: dict[str, Any] = {}


def _build_gammas() -> dict[str, Any]:
    """Construct and cache the gamma matrices using sympy.

    Imports sympy on first call.  Subsequent calls return the cached
    dict so we pay the sympy import / Matrix construction cost once.
    """
    if _CACHED:
        return _CACHED
    try:
        import sympy as sp
    except ImportError as e:
        raise ImportError(
            "dcl_core.core3d.clifford requires sympy.  Install with: "
            "pip install 'dcl_core[sympy]'  (or: pip install sympy)"
        ) from e

    # Standard 2x2 Pauli matrices.
    sigma_1 = sp.Matrix([[0, 1], [1, 0]])
    sigma_2 = sp.Matrix([[0, -sp.I], [sp.I, 0]])
    sigma_3 = sp.Matrix([[1, 0], [0, -1]])

    def gamma_spatial(sigma: sp.Matrix) -> sp.Matrix:
        """Build ``(( 0, sigma ), ( -sigma, 0 ))`` as a 4x4 matrix."""
        g = sp.zeros(4, 4)
        g[0:2, 2:4] = sigma
        g[2:4, 0:2] = -sigma
        return g

    gamma_0 = sp.diag(1, 1, -1, -1)
    gamma_1 = gamma_spatial(sigma_1)
    gamma_2 = gamma_spatial(sigma_2)
    gamma_3 = gamma_spatial(sigma_3)
    eta = sp.diag(1, -1, -1, -1)

    _CACHED.update(
        {
            "gamma_0": gamma_0,
            "gamma_1": gamma_1,
            "gamma_2": gamma_2,
            "gamma_3": gamma_3,
            "ETA_MINKOWSKI": eta,
            "GAMMAS": (gamma_0, gamma_1, gamma_2, gamma_3),
        }
    )
    return _CACHED


def __getattr__(name: str) -> Any:
    """Lazy attribute resolution for symbolic gamma matrices (PEP 562).

    ``dcl_core.core3d.clifford`` imports cleanly without sympy; the
    sympy dependency is paid at first access of one of the symbolic
    attributes.  Any other attribute name falls through to the normal
    ``AttributeError``.
    """
    if name in _SYMBOLIC_NAMES:
        return _build_gammas()[name]
    raise AttributeError(
        f"module 'dcl_core.core3d.clifford' has no attribute {name!r}"
    )
