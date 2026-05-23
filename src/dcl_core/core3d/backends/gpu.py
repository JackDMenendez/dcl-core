"""GPU backend: CuPy + RawKernel for the hop operator.

This module supplies the array primitives the core modules call when
`backend == "gpu"`. It is an optional dependency: install with
`pip install -e .[gpu]` on a host with a CUDA-capable card and a
matching CuPy build.

The hop kernel is memory-bound (3 shifts + multiply + accumulate per
site). The CPU backend's `np.roll` is fine for prototyping but slow
on lattices > 64^3; the GPU `RawKernel` here does coalesced loads
with one thread per output site.

If CuPy is not installed, importing this module raises ImportError
at first use -- not at package import -- so a CPU-only install can
still import `dcl_core` cleanly.
"""

from __future__ import annotations

try:
    import cupy as cp

    _CUPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    cp = None  # type: ignore[assignment]
    _CUPY_AVAILABLE = False

NAME = "gpu"


def _require_cupy() -> None:
    if not _CUPY_AVAILABLE:
        raise ImportError(
            "GPU backend requested but CuPy is not installed. "
            "Install with: pip install -e .[gpu] (requires CUDA)."
        )


def zeros(shape: tuple[int, ...], dtype=None):  # type: ignore[no-untyped-def]
    """Allocate a zeroed CuPy array on the active GPU."""
    _require_cupy()
    return cp.zeros(shape, dtype=dtype or cp.float64)


def zeros_int(shape: tuple[int, ...]):  # type: ignore[no-untyped-def]
    """Allocate a zeroed int64 CuPy array on the active GPU."""
    _require_cupy()
    return cp.zeros(shape, dtype=cp.int64)


def indices(shape: tuple[int, ...]):  # type: ignore[no-untyped-def]
    """Coordinate grid of `shape`; CuPy analogue of `numpy.indices`.

    Returns an int64 CuPy array of shape `(len(shape), *shape)` whose
    `axis=0` slices are the per-axis coordinate fields.
    """
    _require_cupy()
    return cp.indices(shape, dtype=cp.int64)


def shift(array, offset: tuple[int, int, int]):  # type: ignore[no-untyped-def]
    """Periodic shift on the GPU via `cupy.roll`.

    For large lattices, prefer a hand-rolled `RawKernel` (see
    `hop_kernel_RGB` / `hop_kernel_CMY` below) over a chain of
    `cp.roll` calls; the kernel does one global-memory pass instead
    of three.
    """
    _require_cupy()
    return cp.roll(array, shift=offset, axis=(0, 1, 2))


def sum_all(array):  # type: ignore[no-untyped-def]
    """Return the scalar sum (CuPy scalar; cast on the host as needed)."""
    _require_cupy()
    return array.sum()


def floor(array):  # type: ignore[no-untyped-def]
    """Element-wise floor via `cupy.floor`."""
    _require_cupy()
    return cp.floor(array)


def sqrt(array):  # type: ignore[no-untyped-def]
    """Element-wise square root via `cupy.sqrt`."""
    _require_cupy()
    return cp.sqrt(array)


def exp(array):  # type: ignore[no-untyped-def]
    """Element-wise complex exponential via `cupy.exp`."""
    _require_cupy()
    return cp.exp(array)


def cos(array):  # type: ignore[no-untyped-def]
    """Element-wise cosine via `cupy.cos`."""
    _require_cupy()
    return cp.cos(array)


def sin(array):  # type: ignore[no-untyped-def]
    """Element-wise sine via `cupy.sin`."""
    _require_cupy()
    return cp.sin(array)


# ---------------------------------------------------------------------------
# RawKernel sources -- implement once the public API is stable.
# ---------------------------------------------------------------------------
# Sketch of a coalesced bipartite-hop kernel. Implementation deferred
# until the CPU backend is correct and the test harness can compare
# the two. Keep the source string here so the API surface is visible
# even before the implementation lands.

_HOP_KERNEL_RGB_SRC = r"""
extern "C" __global__
void hop_kernel_rgb(
    const double* psi_re, const double* psi_im,
    double* out_re, double* out_im,
    const int Nx, const int Ny, const int Nz
) {
    // TODO: average over the three RGB basis vectors with periodic
    // wrap on (Nx, Ny, Nz). One thread per output site.
}
"""

_HOP_KERNEL_CMY_SRC = r"""
extern "C" __global__
void hop_kernel_cmy(
    const double* psi_re, const double* psi_im,
    double* out_re, double* out_im,
    const int Nx, const int Ny, const int Nz
) {
    // TODO: same as hop_kernel_rgb but over the three CMY basis vectors.
}
"""
