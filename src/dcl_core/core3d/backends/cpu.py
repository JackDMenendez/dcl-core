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
