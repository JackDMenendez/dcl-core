"""Discrete causal session: integer-token state on the bipartite lattice.

A session IS one quantum-mechanical object (electron, photon, proton, ...)
modelled as a budget of `N_units` indistinguishable probability tokens
distributed across lattice sites. A=1 is the integer equality:

    sum_x N(x) == N_units            (exact, no float tolerance)

The session's amplitude at a site is implicit:

    psi(x) = sqrt(N(x) / N_units) * exp(i * phi(x))

where `phi(x)` is a continuous U(1) phase carried per-site, per-chirality.
The Dirac spinor structure is realised by two chirality fields psi_R, psi_L
co-resident on the lattice.

The "Planck of probability" is `epsilon_P = 1 / N_units`. Continuum-limit
v1.0-style dynamics are recovered as `N_units -> infinity`.

Construction
------------
The bare constructor creates a **vacuum** session: all token counts are
zero, all phases are zero (or a linear `k.x` gradient if `momentum` is
nonzero).  A vacuum session deliberately violates A=1; users are
expected to populate it before stepping the engine.  The usual way is
to call a factory classmethod:

- :meth:`DiscreteCausalSession.delta_at` -- a localised point source at
  a chosen lattice site, split 50/50 between R and L.
- :meth:`DiscreteCausalSession.from_arrays` -- a fully custom initial
  state from explicit `N_R`, `N_L`, optional `phi_R`, `phi_L` arrays.

Experiments with multiple sessions (hydrogen emission, proton internals,
scattering) construct each session via its own factory call so the
starting placements / momenta are explicit at the experiment level
rather than baked into the engine.

See:
    docs/design/01_planck_of_probability.md -- rationale and consequences
    docs/reference/session.md -- API reference and examples
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .backends import get_backend
from .lattice import BipartiteLattice

Chirality = Literal["R", "L"]


def _validate_position(
    position: tuple[int, int, int], shape: tuple[int, int, int]
) -> None:
    """Raise ``ValueError`` if ``position`` is not a valid site of ``shape``."""
    if len(position) != 3:
        raise ValueError(f"position must be a 3-tuple, got {position!r}")
    for axis_name, coord, axis_len in zip("xyz", position, shape, strict=True):
        if not (0 <= int(coord) < axis_len):
            raise ValueError(
                f"position[{axis_name}] = {coord} out of range [0, {axis_len})"
            )


@dataclass
class DiscreteCausalSession:
    """One causal session: integer tokens + continuous phase on a lattice.

    Parameters
    ----------
    lattice : BipartiteLattice
        The geometry this session lives on.
    n_units : int
        The probability budget. Sum of `N(x)` across all sites is
        exactly this integer at every tick (A=1) once the session has
        been populated by a factory or by direct array writes.
    omega : float
        The session's internal clock frequency. This IS the rest-mass
        parameter via E = N_units * omega (in framework units).
        omega=0 -> massless (photon).
        omega=pi/2 -> maximal Zitterbewegung.
    momentum : (float, float, float), default (0, 0, 0)
        Plane-wave momentum in lattice units. When nonzero, the bare
        constructor imprints `phi_R(x) = phi_L(x) = k.x` across the
        whole lattice; combined with a population (delta_at / etc.)
        this IS a moving wavepacket. Matches Paper I's
        `CausalSession(initial_node, momentum=...)` semantics.

    Notes
    -----
    Token distributions `N_R(x)`, `N_L(x)` and phases `phi_R(x)`,
    `phi_L(x)` are stored as backend arrays of shape `lattice.shape`
    indexed `[x, y, z]`. The lattice's `backend` field determines
    whether these are NumPy or CuPy.

    The session does **not** know about other sessions; multi-session
    interactions go through :class:`~dcl_core.core3d.scheduler.TickScheduler`.
    """

    lattice: BipartiteLattice
    n_units: int
    omega: float
    momentum: tuple[float, float, float] = (0.0, 0.0, 0.0)

    # State arrays -- shape = lattice.shape, dtype = int64 (counts) / float64 (phase).
    # Initialised lazily in __post_init__; do not assume present until after init.
    N_R: np.ndarray = field(default=None, repr=False)  # type: ignore[assignment]
    N_L: np.ndarray = field(default=None, repr=False)  # type: ignore[assignment]
    phi_R: np.ndarray = field(default=None, repr=False)  # type: ignore[assignment]
    phi_L: np.ndarray = field(default=None, repr=False)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.n_units <= 0:
            raise ValueError(f"n_units must be positive, got {self.n_units}")
        self._allocate_state()

    def _allocate_state(self) -> None:
        """Allocate zeroed state arrays (vacuum) and imprint any momentum phase.

        Backend dispatched on ``self.lattice.backend``: CPU path uses
        ``numpy.zeros``, GPU path uses ``cupy.zeros``.  After
        allocation the session is in the **vacuum** state -- all
        ``N_R``/``N_L`` counts are zero and ``total_tokens()`` is 0,
        which deliberately violates A=1.  Populate the session via a
        factory (e.g. :meth:`delta_at`) or by writing the state
        arrays directly before handing it to the scheduler.

        If ``self.momentum`` is nonzero, ``phi_R`` and ``phi_L`` are
        initialised to the linear plane-wave phase ``k.x`` rather
        than to zero; that phase persists at empty sites where it
        does not affect the amplitude.
        """
        backend = get_backend(self.lattice.backend)
        shape = self.lattice.shape
        self.N_R = backend.zeros_int(shape)
        self.N_L = backend.zeros_int(shape)
        self.phi_R = backend.zeros(shape)
        self.phi_L = backend.zeros(shape)
        if any(p != 0.0 for p in self.momentum):
            self._apply_momentum_gradient()

    def _apply_momentum_gradient(self) -> None:
        """Imprint ``phi_R(x) = phi_L(x) += k.x`` from ``self.momentum``.

        `k` IS the plane-wave momentum (lattice units, `a = 1`).
        Paper I applies this multiplicatively to ``psi``; the integer
        formulation here applies it additively to the phase fields,
        which is equivalent under ``psi = sqrt(N/n) * exp(i*phi)``.
        """
        kx, ky, kz = self.momentum
        backend = get_backend(self.lattice.backend)
        coords = backend.indices(self.lattice.shape)
        gradient = kx * coords[0] + ky * coords[1] + kz * coords[2]
        self.phi_R += gradient
        self.phi_L += gradient

    # ------------------------------------------------------------------
    # Factory classmethods -- preferred construction sites for experiments
    # ------------------------------------------------------------------

    @classmethod
    def delta_at(
        cls,
        lattice: BipartiteLattice,
        n_units: int,
        omega: float,
        position: tuple[int, int, int],
        momentum: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> "DiscreteCausalSession":
        """Construct a session with all ``n_units`` tokens at one site.

        Parameters
        ----------
        lattice, n_units, omega, momentum
            As for the bare constructor.
        position : (int, int, int)
            The site that receives all the tokens. Must satisfy
            ``0 <= position[i] < lattice.shape[i]`` for each axis.

        Returns
        -------
        DiscreteCausalSession
            A session whose ``N_R[position]`` and ``N_L[position]``
            sum to ``n_units``, split 50/50 (extra token to R if
            ``n_units`` is odd; arbitrary tiebreaker, not a physical
            claim).  All other sites are empty.  Phases follow the
            ``momentum`` plane-wave gradient.

        Notes
        -----
        This IS Paper I's ``CausalSession(initial_node=...)`` idiom
        in integer-token form.  For a Gaussian wavepacket or a
        bound-state-shaped initial profile (electron 1s, etc.),
        compute the array offline and pass it to
        :meth:`from_arrays` instead.
        """
        _validate_position(position, lattice.shape)
        session = cls(
            lattice=lattice, n_units=n_units, omega=omega, momentum=momentum
        )
        n_R = (n_units + 1) // 2  # extra token to R when n_units is odd
        n_L = n_units // 2
        session.N_R[position] = n_R
        session.N_L[position] = n_L
        return session

    @classmethod
    def wavepacket(
        cls,
        lattice: BipartiteLattice,
        n_units: int,
        omega: float,
        center: tuple[int, int, int],
        sigma: float | tuple[float, float, float],
        momentum: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> "DiscreteCausalSession":
        """Minimum-uncertainty Gaussian wavepacket at the given position width.

        Parameters
        ----------
        lattice, n_units, omega, momentum
            As for the bare constructor.  `momentum` sets the mean k
            (group velocity / drift); the wavepacket envelope is
            centred on `center` with width `sigma`.
        center : (int, int, int)
            Lattice site at the centre of the Gaussian envelope.
        sigma : float or (float, float, float)
            Position-space standard deviation in lattice units.  A
            scalar is interpreted as an isotropic σ; a 3-tuple sets a
            per-axis σ (anisotropic Gaussian).

        Returns
        -------
        DiscreteCausalSession
            A session whose ``|psi|^2`` profile is a quantised Gaussian
            of width ``sigma`` at ``center``, with ``total_tokens() ==
            n_units`` exactly.  Phases follow the ``momentum``
            plane-wave gradient.

        Notes
        -----
        This IS the **minimum-Δp** state at the given σ_x: a Gaussian
        saturating the Heisenberg lower bound, so the momentum-space
        width is ``Δp = 1 / (2 σ_x)`` per axis in lattice units.
        Paper III's min-Δp scans iterate ``sigma`` to probe how the
        integer-token engine reproduces (or fails to reproduce) the
        continuous engine's behaviour at the lattice resolution limit::

            for sigma in [1.0, 2.0, 4.0, 8.0]:
                session = DiscreteCausalSession.wavepacket(
                    lattice, n_units=10**6, omega=0.1,
                    center=(L//2, L//2, L//2),
                    sigma=sigma,
                    momentum=(0.1, 0.0, 0.0),
                )
                # ... evolve, measure observable, log ...

        For large ``sigma`` relative to the lattice extent the periodic
        image of the Gaussian contributes; the construction does not
        sum images, so the realised envelope deviates from the ideal
        single-period Gaussian.  Keep ``sigma`` comfortably below
        ``min(lattice.shape) / 4`` if that matters for the experiment.

        The integer quantisation reuses
        :class:`~dcl_core.core3d.remainder.BresenhamResidual` for the
        floor + top-up rebalance, so the same algorithm that conserves
        A=1 tick-to-tick also conserves it at construction time.
        """
        _validate_position(center, lattice.shape)
        if isinstance(sigma, (int, float)):
            sigma_vec = (float(sigma), float(sigma), float(sigma))
        else:
            sigma_vec = tuple(float(s) for s in sigma)
            if len(sigma_vec) != 3:
                raise ValueError(
                    f"sigma must be a scalar or a 3-tuple, got {sigma!r}"
                )
        if any(s <= 0.0 for s in sigma_vec):
            raise ValueError(
                f"sigma components must be positive, got {sigma_vec!r}"
            )

        backend = get_backend(lattice.backend)
        coords = backend.indices(lattice.shape)
        cx, cy, cz = center
        sx, sy, sz = sigma_vec
        # IS exp(-0.5 * sum_axis ((x - c) / sigma)^2).  Real, positive,
        # peaked at `center`.
        exponent = -0.5 * (
            ((coords[0] - cx) / sx) ** 2
            + ((coords[1] - cy) / sy) ** 2
            + ((coords[2] - cz) / sz) ** 2
        )
        envelope = backend.exp(exponent)
        total = float(backend.sum_all(envelope))
        if total == 0.0:  # numerical underflow for absurdly small sigma
            raise ValueError(
                "Gaussian envelope underflowed to zero; sigma is too small "
                "relative to lattice spacing."
            )
        envelope = envelope / total

        # Per-chirality fractional targets.  Extra-token convention:
        # the (possibly nonzero) parity goes to R, matching delta_at.
        n_R = (n_units + 1) // 2
        n_L = n_units // 2
        N_target_R = envelope * n_R
        N_target_L = envelope * n_L

        # Reuse the residual's floor + top-up algorithm for the
        # integer quantisation.  A fresh residual starts with carry==0
        # so this is a one-shot quantise, not an evolving accumulator.
        from .remainder import BresenhamResidual

        residual = BresenhamResidual(lattice=lattice)
        N_R_int, N_L_int = residual.quantise(N_target_R, N_target_L, n_units)

        session = cls(
            lattice=lattice, n_units=n_units, omega=omega, momentum=momentum
        )
        session.N_R[...] = N_R_int
        session.N_L[...] = N_L_int
        return session

    @classmethod
    def from_arrays(
        cls,
        lattice: BipartiteLattice,
        n_units: int,
        omega: float,
        N_R: np.ndarray,
        N_L: np.ndarray,
        phi_R: np.ndarray | None = None,
        phi_L: np.ndarray | None = None,
    ) -> "DiscreteCausalSession":
        """Construct a session from fully-specified state arrays.

        Parameters
        ----------
        lattice, n_units, omega
            As for the bare constructor.
        N_R, N_L : int arrays of shape ``lattice.shape``
            Per-site token counts. Must satisfy
            ``sum(N_R) + sum(N_L) == n_units`` exactly; this is the
            A=1 contract the factory enforces at construction time.
        phi_R, phi_L : float arrays of shape ``lattice.shape``, optional
            Per-site phases for each chirality. ``None`` (the
            default) leaves them at zero.

        Returns
        -------
        DiscreteCausalSession
            A session with the supplied state.

        Notes
        -----
        This factory does NOT accept a ``momentum`` argument:
        plane-wave momentum should be baked into ``phi_R`` and
        ``phi_L`` by the caller (``phi[x,y,z] = kx*x + ky*y + kz*z``)
        so that the factory contract -- "the arrays you pass are
        exactly what the session uses" -- is unambiguous.
        """
        if N_R.shape != lattice.shape:
            raise ValueError(
                f"N_R.shape {N_R.shape} != lattice.shape {lattice.shape}"
            )
        if N_L.shape != lattice.shape:
            raise ValueError(
                f"N_L.shape {N_L.shape} != lattice.shape {lattice.shape}"
            )
        total = int(N_R.sum()) + int(N_L.sum())
        if total != n_units:
            raise ValueError(
                f"sum(N_R) + sum(N_L) = {total}, expected n_units = {n_units}"
            )
        if phi_R is not None and phi_R.shape != lattice.shape:
            raise ValueError(
                f"phi_R.shape {phi_R.shape} != lattice.shape {lattice.shape}"
            )
        if phi_L is not None and phi_L.shape != lattice.shape:
            raise ValueError(
                f"phi_L.shape {phi_L.shape} != lattice.shape {lattice.shape}"
            )
        session = cls(lattice=lattice, n_units=n_units, omega=omega)
        session.N_R[...] = N_R
        session.N_L[...] = N_L
        if phi_R is not None:
            session.phi_R[...] = phi_R
        if phi_L is not None:
            session.phi_L[...] = phi_L
        return session

    # ------------------------------------------------------------------
    # Inspection / invariants
    # ------------------------------------------------------------------

    @property
    def epsilon_P(self) -> float:
        """The Planck of probability for this session: 1 / N_units."""
        return 1.0 / self.n_units

    def total_tokens(self) -> int:
        """Return ``sum(N_R) + sum(N_L)``.

        For a populated, A=1-conserving session this equals
        ``n_units`` exactly.  A bare-constructed (vacuum) session
        returns 0 here -- the constructor does not enforce A=1; the
        factory methods or direct array writes do.

        On the GPU backend, the ``int()`` cast forces a
        device-to-host sync; that is the right semantic for
        "give me the number now."
        """
        backend = get_backend(self.lattice.backend)
        return int(backend.sum_all(self.N_R) + backend.sum_all(self.N_L))

    def assert_unity(self) -> None:
        """Raise if the session has drifted away from A=1.

        Cheap sanity check intended for tests, debug runs, and CI.  Also
        catches the case of an un-populated bare-constructed session
        being handed to the scheduler.  In normal execution this should
        never fire; if it does for a populated session, the bug is in
        the hop / remainder / scheduler interaction, not in the
        session itself.
        """
        total = self.total_tokens()
        if total != self.n_units:
            raise AssertionError(
                f"A=1 violated: total_tokens={total}, expected {self.n_units}"
            )

    def amplitude(self, chirality: Chirality) -> np.ndarray:
        """Return the implicit complex amplitude psi_{R|L}(x) for inspection.

        Computed as ``sqrt(N(x) / n_units) * exp(i * phi(x))``.  This
        is a freshly-allocated snapshot, not a view -- callers can
        mutate the result without affecting session state, and the
        session never reads back from it.  Provided for introspection
        / plotting only; the session's evolution operates on integer
        ``N`` and continuous ``phi`` directly, never on the derived
        complex field.
        """
        if chirality == "R":
            N, phi = self.N_R, self.phi_R
        elif chirality == "L":
            N, phi = self.N_L, self.phi_L
        else:
            raise ValueError(f"chirality must be 'R' or 'L', got {chirality!r}")
        backend = get_backend(self.lattice.backend)
        # psi(x) IS sqrt(N(x) / n_units) * exp(i * phi(x)).  The division
        # promotes int64 -> float64; the `1j * phi` promotes float64 ->
        # complex128; the product is complex128.
        return backend.sqrt(N / self.n_units) * backend.exp(1j * phi)
