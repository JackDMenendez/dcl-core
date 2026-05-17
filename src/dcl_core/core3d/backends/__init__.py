"""Numerical backends.

A backend is a thin module exposing the array-allocation and
array-operation primitives that the core modules call. The public-API
classes (lattice / session / hop / remainder / scheduler) accept a
`backend` string ("cpu" | "gpu") and dispatch to the corresponding
module here.

Adding a backend:
    1. Create `<name>.py` in this directory.
    2. Implement the same set of primitives as `cpu.py`.
    3. Register the name in `_BACKENDS` below.

The contract is currently informal -- if backend count grows past
two, formalise it as an `abc.ABC` protocol.
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType

_BACKENDS = {
    "cpu": "dcl_core.core3d.backends.cpu",
    "gpu": "dcl_core.core3d.backends.gpu",
}


def get_backend(name: str) -> ModuleType:
    """Return the backend module for the named target.

    Raises
    ------
    KeyError
        If `name` is not a registered backend.
    ImportError
        If the named backend's module fails to import (e.g. asking
        for "gpu" on a host without CuPy installed).
    """
    try:
        module_path = _BACKENDS[name]
    except KeyError as exc:
        raise KeyError(
            f"unknown backend {name!r}; known: {sorted(_BACKENDS)}"
        ) from exc
    return import_module(module_path)
