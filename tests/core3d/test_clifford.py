"""Test the Clifford algebra `{gamma_mu, gamma_nu} = 2 eta_munu I`.

Uses sympy to verify the anti-commutator relations exactly on the
structure-factor's matrix representation. This is an algebraic check,
not a numerical one -- no float tolerance.

Marked `sympy` so it can be skipped on hosts without sympy installed
(though `dev` extras include sympy by default).
"""

from __future__ import annotations

import pytest

pytestmark = [
    pytest.mark.sympy,
    pytest.mark.skip(reason="Implement once HopOperator exposes its gamma matrices."),
]


def test_clifford_anticommutators() -> None:
    """`{gamma_mu, gamma_nu} = 2 * eta_munu * I` for all mu, nu.

    eta is the Minkowski metric diag(+1, -1, -1, -1) (mostly-minus
    convention; flip signs if the framework uses mostly-plus).
    """
    import sympy

    # Once concrete: import gamma matrices from dcl_core (e.g.
    # `from dcl_core.hop import gamma_0, gamma_1, gamma_2, gamma_3`).
    # Verify each {gamma_mu, gamma_nu} = 2 eta_munu I as a sympy
    # Matrix equality (.equals, not numerical .allclose).
    sympy.zeros(4)  # placeholder so the import is exercised
    pytest.skip("HopOperator.gamma_matrices not yet exposed.")
