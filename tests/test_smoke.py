"""Smoke tests: the package imports and exposes its public API.

Runs in milliseconds; first line of defence against import-time
regressions (typos in `__init__.py`, missing modules, syntax errors).
"""

from __future__ import annotations


def test_package_imports() -> None:
    """`import dcl_core` succeeds without error."""
    import dcl_core  # noqa: F401


def test_version_is_a_string() -> None:
    """`dcl_core.__version__` is a non-empty string."""
    import dcl_core

    assert isinstance(dcl_core.__version__, str)
    assert dcl_core.__version__  # non-empty


def test_public_api_is_what_we_expect() -> None:
    """The top-level `__all__` exposes only `__version__` + two submodules.

    The two submodules (`core`, `core3d`) carry their own public-API
    contracts; each submodule has its own smoke test
    (`tests/core/test_smoke.py`, `tests/core3d/test_smoke.py`)
    asserting its own `__all__`.  At the top level, the only
    contract is "the choice of engine is explicit at every import
    site" -- no top-level shortcuts.

    If this fails, either a new submodule has been added (then bump
    minor + update the expected set), or someone has re-exported a
    name through the top-level package (which we explicitly do not
    do; revert the change or escalate).
    """
    import dcl_core

    expected = {
        "__version__",
        "core",
        "core3d",
    }
    assert set(dcl_core.__all__) == expected, (
        f"Top-level public API drift: __all__ = {set(dcl_core.__all__)}, "
        f"expected {expected}.  Bump semver and update this test."
    )


def test_submodules_import() -> None:
    """Both submodules import cleanly.

    `core` is Paper~I's continuous-amplitude engine, ported verbatim;
    it should be fully importable as soon as `dcl_core` is installed.
    `core3d` is the integer-token design with stub implementations;
    importing the module surface must work even though calling into
    its operators raises NotImplementedError.
    """
    import dcl_core.core  # noqa: F401
    import dcl_core.core3d  # noqa: F401
