"""Test the Clifford algebra `{gamma_mu, gamma_nu} = 2 eta_munu I`.

Uses sympy to verify the anti-commutator relations exactly on the
4x4 Dirac matrices in :mod:`dcl_core.core3d.clifford`.  This is an
algebraic check, not a numerical one -- no float tolerance.

Marked ``sympy`` so it skips automatically on hosts that did NOT
install the ``[sympy]`` (or ``[dev]``) extras.
"""

from __future__ import annotations

import pytest

# Skip the entire module if sympy is not installed (i.e. the `[sympy]`
# / `[dev]` extra was not selected at install time).  The `sympy`
# marker is preserved so `pytest -m "not sympy"` still deselects it
# even when sympy IS available.
pytest.importorskip("sympy")
pytestmark = pytest.mark.sympy


def test_clifford_anticommutators() -> None:
    """``{gamma_mu, gamma_nu} = 2 * eta_munu * I`` for all mu, nu in {0, 1, 2, 3}.

    eta is the Minkowski metric in mostly-minus convention,
    ``diag(+1, -1, -1, -1)`` (see :mod:`dcl_core.core3d.clifford` for
    why this convention; Paper I uses the same).  The equality is
    enforced as a symbolic match (``==`` between sympy Matrices), not
    a numerical ``allclose``.
    """
    import sympy as sp

    from dcl_core.core3d.clifford import ETA_MINKOWSKI, GAMMAS

    I4 = sp.eye(4)
    for mu in range(4):
        for nu in range(4):
            anticomm = GAMMAS[mu] * GAMMAS[nu] + GAMMAS[nu] * GAMMAS[mu]
            expected = 2 * ETA_MINKOWSKI[mu, nu] * I4
            assert anticomm == expected, (
                f"{{gamma_{mu}, gamma_{nu}}} = {anticomm} "
                f"does not equal 2 * eta[{mu},{nu}] * I = {expected}"
            )


def test_gamma_0_is_diagonal_plus_minus_one() -> None:
    """``gamma_0 = diag(+1, +1, -1, -1)`` in the Dirac representation."""
    import sympy as sp

    from dcl_core.core3d.clifford import gamma_0

    assert gamma_0 == sp.diag(1, 1, -1, -1)


def test_gammas_tuple_matches_indexed_names() -> None:
    """``GAMMAS[mu]`` agrees with the named symbols ``gamma_mu``."""
    from dcl_core.core3d.clifford import (
        GAMMAS,
        gamma_0,
        gamma_1,
        gamma_2,
        gamma_3,
    )

    assert GAMMAS == (gamma_0, gamma_1, gamma_2, gamma_3)
