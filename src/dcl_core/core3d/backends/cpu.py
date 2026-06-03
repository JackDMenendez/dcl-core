"""CPU backend: NumPy-based reference implementation.

This module supplies the array primitives the core modules call when
`backend == "cpu"`. The CPU backend IS the reference implementation
-- the GPU backend's correctness is asserted against it in
`tests/test_gpu_matches_cpu.py` (once added).

Performance is secondary here; clarity is primary. The hop kernel
uses `np.roll` for periodic shifts; for large lattices, the GPU
backend should be used instead.
"""

from __future__ import annotations

import numpy as np

NAME = "cpu"


def zeros(shape: tuple[int, ...], dtype: type | str = np.float64) -> np.ndarray:
    """Allocate a zeroed array of the given shape on the CPU."""
    return np.zeros(shape, dtype=dtype)


def zeros_int(shape: tuple[int, ...]) -> np.ndarray:
    """Allocate a zeroed int64 array -- the standard token-count dtype."""
    return np.zeros(shape, dtype=np.int64)


def indices(shape: tuple[int, ...]) -> np.ndarray:
    """Coordinate grid of `shape`; same semantics as `numpy.indices`.

    Returns an int64 array of shape `(len(shape), *shape)` whose
    `axis=0` slices are the per-axis coordinate fields.  Used by
    `BipartiteLattice.parity_field` to compute the Z_2 grading
    `(x + y + z) mod 2` without an explicit Python-level triple loop.
    """
    return np.indices(shape, dtype=np.int64)


def shift(array: np.ndarray, offset: tuple[int, int, int]) -> np.ndarray:
    """Periodic shift of `array` by `offset` along the spatial axes.

    Returns `array[x - offset]` with wraparound. Pure NumPy
    implementation using `np.roll`. The "minus" sign aligns with the
    framework's hop convention: `hop_RGB(psi)` at site x averages
    `psi` from the three sites x - v (v in RGB_VECTORS).
    """
    return np.roll(array, shift=offset, axis=(0, 1, 2))


def sum_all(array: np.ndarray) -> np.generic:
    """Return the scalar sum of all elements (returns a NumPy scalar)."""
    return array.sum()


def floor(array: np.ndarray) -> np.ndarray:
    """Element-wise floor (stays float).  Used by `TokenResidual.quantise`
    to split an accumulator into integer + fractional parts.
    """
    return np.floor(array)


def sqrt(array: np.ndarray) -> np.ndarray:
    """Element-wise square root.  Promotes int -> float64."""
    return np.sqrt(array)


def exp(array: np.ndarray) -> np.ndarray:
    """Element-wise complex exponential.  Returns complex128 for complex input."""
    return np.exp(array)


def cos(array: np.ndarray) -> np.ndarray:
    """Element-wise cosine (scalar or array)."""
    return np.cos(array)


def sin(array: np.ndarray) -> np.ndarray:
    """Element-wise sine (scalar or array)."""
    return np.sin(array)
