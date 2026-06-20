"""Gauge-field construction helpers for the core3d Peierls coupling.

Lattice frame.  These helpers build the U(1) vector potential ``A`` that
:meth:`dcl_core.core3d.HopOperator.step` couples to via the Peierls link
phase (``vector_potential=`` argument; see ``hop.py``).  The hop consumes
``A`` as a real field of shape ``(3, *lattice.shape)``; this module is
where the physically-meaningful background configurations (a uniform
magnetic field, ...) are constructed.

physics: ``A`` IS the electromagnetic vector potential A_i (spatial
components); its lattice curl IS the magnetic field B (the Faraday
2-form restricted to the spatial block).

CPU-only at v0.3.0 (NumPy).  GPU array construction is deferred with the
rest of the GPU Peierls path (see ``notes/gauge_field_v030_plan.md``
Phase 4); an experiment on a GPU lattice converts the returned array.

See:
    docs/design/04_gauge_field_and_vacuum_response.md -- requirements (R3)
    docs/design/05_interaction_engine.md -- design context
"""

from __future__ import annotations

import numpy as np


def uniform_B_potential(
    shape: tuple[int, int, int],
    B_vec: tuple[float, float, float] | np.ndarray,
    origin: tuple[float, float, float] | np.ndarray | None = None,
) -> np.ndarray:
    """Symmetric-gauge vector potential for a **uniform** magnetic field ``B``.

    ``A(r) = 1/2 B x (r - origin)``  ⇒  ``curl A = B`` (uniform, exact on
    the lattice because ``A`` is linear in ``r`` so central differences are
    exact).

    Parameters
    ----------
    shape : (int, int, int)
        Lattice shape; the returned field has shape ``(3, *shape)``.
    B_vec : (float, float, float)
        The uniform magnetic field vector (lattice units).  Orient it along
        ``(1, 1, -1)`` for the optical-axis sweep, in the perpendicular
        plane, or obliquely.
    origin : (float, float, float), optional
        Gauge origin (the point where ``A = 0``).  Defaults to the lattice
        **centre** ``((nx-1)/2, (ny-1)/2, (nz-1)/2)`` -- the symmetric
        choice that minimises ``|A|`` across the box.  The choice is a gauge
        freedom; observables (``curl A``, densities) do not depend on it.

    Returns
    -------
    A : float64 array of shape ``(3, *shape)``
        physics: the symmetric-gauge vector potential; ``curl A = B``.

    Notes
    -----
    Different origins differ by a pure gauge (a uniform shift of ``A`` is
    ``A -> A + 1/2 B x c``, which is ``grad`` of a quadratic ``Lambda``);
    the Peierls Ward identity (``test_peierls.py`` #2) guarantees densities
    are invariant under that choice.
    """
    if len(shape) != 3:
        raise ValueError(f"shape must be a 3-tuple, got {shape!r}")
    B = np.asarray(B_vec, dtype=np.float64)
    if B.shape != (3,):
        raise ValueError(f"B_vec must be a 3-vector, got shape {B.shape}")

    if origin is None:
        origin = np.array([(n - 1) / 2.0 for n in shape], dtype=np.float64)
    else:
        origin = np.asarray(origin, dtype=np.float64)
        if origin.shape != (3,):
            raise ValueError(f"origin must be a 3-vector, got shape {origin.shape}")

    # Site-position field and displacement from the gauge origin.
    # physics: r_sites IS the lattice site position r_i (lattice units, a=1);
    #          delta_r = r - origin is the symmetric gauge's lever arm.
    r_sites = np.indices(shape, dtype=np.float64)
    delta_r = r_sites - origin[:, None, None, None]

    # A = 1/2 B x delta_r, component by component (the cross product).
    # physics: A_i = 1/2 epsilon_ijk B_j (r-origin)_k -- symmetric-gauge potential.
    A = np.empty((3, *shape), dtype=np.float64)
    A[0] = 0.5 * (B[1] * delta_r[2] - B[2] * delta_r[1])
    A[1] = 0.5 * (B[2] * delta_r[0] - B[0] * delta_r[2])
    A[2] = 0.5 * (B[0] * delta_r[1] - B[1] * delta_r[0])
    return A


def uniform_E_potential(
    shape: tuple[int, int, int],
    E_vec: tuple[float, float, float] | np.ndarray,
    origin: tuple[float, float, float] | np.ndarray | None = None,
) -> np.ndarray:
    """Scalar potential ``A_0(r) = -E . (r - origin)`` for a uniform static E.

    The **electric** sector of the gauge background (the magnetic sector is
    :func:`uniform_B_potential`).  A static, uniform electric field ``E`` is
    a spatially-linear on-site potential: ``-grad A_0 = E``.  Feed the
    returned array to ``HopOperator.step(external_potential=...)`` /
    ``TickScheduler.external_potential`` -- core3d's on-site
    ``delta_phi = omega + A_0`` IS the temporal/tick-direction gauge phase.

    physics: ``A_0`` IS the time-component of the U(1) gauge potential; the
    static electric field is ``E = -grad A_0`` (no ``-dA/dt`` since the
    background is static).

    Parameters
    ----------
    shape : (int, int, int)
        Lattice shape; the returned scalar field has shape ``shape``.
    E_vec : (float, float, float)
        Uniform electric field vector (lattice units).
    origin : (float, float, float), optional
        Zero of the potential.  Defaults to the lattice **centre**.  A shift
        of ``origin`` adds a constant to ``A_0`` -- a global phase offset
        with no effect on ``E`` or on densities.

    Returns
    -------
    A0 : float64 array of shape ``shape``
        physics: the electric scalar potential; ``-grad A0 = E``.

    Notes
    -----
    Like a linear gauge on a torus, ``A_0 = -E.r`` is **not periodic** (it
    jumps across the wrap).  Use a probe localised away from the boundary,
    or a commensurate setup, exactly as for a constant vector potential.
    This is an honest boundary limitation, not a bug.
    """
    if len(shape) != 3:
        raise ValueError(f"shape must be a 3-tuple, got {shape!r}")
    E = np.asarray(E_vec, dtype=np.float64)
    if E.shape != (3,):
        raise ValueError(f"E_vec must be a 3-vector, got shape {E.shape}")

    if origin is None:
        origin = np.array([(n - 1) / 2.0 for n in shape], dtype=np.float64)
    else:
        origin = np.asarray(origin, dtype=np.float64)
        if origin.shape != (3,):
            raise ValueError(f"origin must be a 3-vector, got shape {origin.shape}")

    # Site-position field and displacement from the gauge origin.
    # physics: r_sites IS the lattice site position r_i (lattice units, a=1);
    #          delta_r = r - origin is the scalar-potential lever arm.
    r_sites = np.indices(shape, dtype=np.float64)
    delta_r = r_sites - origin[:, None, None, None]
    # A_0(r) = -E . (r - origin).
    # physics: -grad A_0 = E (the static electric field).
    return -(E[0] * delta_r[0] + E[1] * delta_r[1] + E[2] * delta_r[2])
