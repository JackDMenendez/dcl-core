"""Shared pytest fixtures.

Convention: fixtures here are **small, fast, deterministic**. Large
or slow setups are local to the test file that needs them.

Tests are parametrised over backend where applicable. The `backend`
fixture below yields each registered backend in turn; tests that
require a specific backend should use a narrower fixture
(`cpu_backend`, `gpu_backend`).
"""

from __future__ import annotations

import importlib
import pytest


# ---------------------------------------------------------------------------
# Backend fixtures
# ---------------------------------------------------------------------------

def _gpu_available() -> bool:
    """Return True iff CuPy imports and finds a CUDA device."""
    try:
        cupy = importlib.import_module("cupy")
        cupy.cuda.runtime.getDeviceCount()
        return True
    except Exception:
        return False


_BACKENDS_AVAILABLE = ["cpu"]
if _gpu_available():
    _BACKENDS_AVAILABLE.append("gpu")


@pytest.fixture(params=_BACKENDS_AVAILABLE)
def backend(request: pytest.FixtureRequest) -> str:
    """Parametrise a test across all available backends."""
    return request.param


@pytest.fixture
def cpu_backend() -> str:
    """Force a test to use the CPU backend."""
    return "cpu"


@pytest.fixture
def gpu_backend() -> str:
    """Force a test to use the GPU backend; skip if unavailable."""
    if not _gpu_available():
        pytest.skip("CuPy / CUDA not available")
    return "gpu"


# ---------------------------------------------------------------------------
# Small-lattice fixtures (cheap; fine for unit tests)
# ---------------------------------------------------------------------------

@pytest.fixture
def small_shape() -> tuple[int, int, int]:
    """A cheap lattice shape suitable for fast unit tests."""
    return (8, 8, 8)


@pytest.fixture
def tiny_shape() -> tuple[int, int, int]:
    """A 4^3 lattice for the cheapest possible smoke tests."""
    return (4, 4, 4)


# ---------------------------------------------------------------------------
# RNG / determinism
# ---------------------------------------------------------------------------

@pytest.fixture
def rng_seed() -> int:
    """Single seed used by every test that needs randomness.

    Hard-coded so test failures are reproducible: rerun the same
    test and you get the same bytes. If you need to vary the seed
    inside one test, parametrise locally.
    """
    return 20260509
