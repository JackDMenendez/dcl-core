"""Single source of truth for the package version.

Bump this on every release, in lockstep with `CITATION.cff`'s
`version` and `date-released` fields. The `pyproject.toml`
`[tool.hatch.version]` block reads this file so `pip install` and
`python -m build` pick it up automatically.

Semver:
  MAJOR.MINOR.PATCH
    MAJOR -- breaking change to anything re-exported from
             `src/dcl_core/__init__.py`
    MINOR -- backwards-compatible additions to the public API
    PATCH -- internal refactoring, bug fixes, no API surface change

Pre-1.0 releases (0.X.Y) signal "API still unstable" -- minor bumps
may break callers.
"""

__version__ = "0.1.0"
