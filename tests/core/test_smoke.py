"""Smoke tests for dcl_core.core (the continuous-amplitude submodule).

Mirrors the top-level ``test_smoke.py``'s API-stability checks but
targets the ``core`` submodule's public surface.  This file should
run successfully even before the integer-token side of the package
has any implementations -- ``core`` is Paper~I's engine, ported
verbatim, and is ready to use as soon as ``dcl_core`` is installed.
"""

from __future__ import annotations


def test_core_imports() -> None:
    """``import dcl_core.core`` succeeds without error."""
    import dcl_core.core  # noqa: F401


def test_core_public_api_is_what_we_expect() -> None:
    """``dcl_core.core.__all__`` matches the Paper~I engine surface.

    If this fails, the classic submodule's public API has changed --
    check whether a semver bump is needed and update this test in
    lockstep.
    """
    from dcl_core import core

    expected = {
        "OctahedralLattice",
        "RGB_VECTORS",
        "CMY_VECTORS",
        "ALL_VECTORS",
        "SUBLATTICE_SIZE",
        "COORDINATION_NUMBER",
        "active_vectors",
        "EVEN_TICK",
        "ODD_TICK",
        "PhaseOscillator",
        "enforce_unity",
        "unity_residual",
        "is_unity",
        "enforce_unity_spinor",
        "unity_residual_spinor",
        "CausalSession",
        "TickScheduler",
        "ShuffleScheme",
        "CompositeCausalSession",
    }
    assert set(core.__all__) == expected, (
        f"core submodule public API drift: __all__ = {set(core.__all__)}, "
        f"expected {expected}.  Bump semver and update this test."
    )
