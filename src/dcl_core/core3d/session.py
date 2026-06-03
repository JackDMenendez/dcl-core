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
  state from explicit `N_RGB`, `N_CMY`, optional `phi_RGB`, `phi_CMY` arrays.

Experiments with multiple sessions (hydrogen emission, proton internals,
scattering) construct each session via its own factory call so the
starting placements / momenta are explicit at the experiment level
rather than baked into the engine.

See:
    docs/design/01_planck_of_probability.md -- rationale and consequences
    docs/reference/session.md -- API reference and examples
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .backends import get_backend
from .lattice import BipartiteLattice

# Component selector for the bipartite token state.  "RGB"/"CMY" are the
# lattice-frame names (the two sublattice components, V_3^+ / V_3^-); "R"/"L"
# are accepted physics-frame aliases.
# physics: RGB/CMY component IS the right/left-chiral Dirac spinor psi_R/psi_L.
Component = Literal["RGB", "CMY", "R", "L"]
Chirality = Component  # deprecated alias for the old name


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
        constructor imprints `phi_RGB(x) = phi_CMY(x) = k.x` across the
        whole lattice; combined with a population (delta_at / etc.)
        this IS a moving wavepacket. Matches Paper I's
        `CausalSession(initial_node, momentum=...)` semantics.

    Notes
    -----
    Token distributions `N_RGB(x)`, `N_CMY(x)` and phases `phi_RGB(x)`,
    `phi_CMY(x)` are stored as backend arrays of shape `lattice.shape`
    indexed `[x, y, z]`. The lattice's `backend` field determines
    whether these are NumPy or CuPy.

    The session does **not** know about other sessions; multi-session
    interactions go through :class:`~dcl_core.core3d.scheduler.TickScheduler`.
    """

    lattice: BipartiteLattice
    n_units: int
    omega: float
    momentum: tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Lattice state -- shape = lattice.shape; int64 token counts on the two
    # sublattice components (V_3^+ / V_3^-), float64 per-site U(1) phases.
    # Initialised lazily in __post_init__; do not assume present until after init.
    # physics: N_RGB/N_CMY IS |psi_R|^2/|psi_L|^2 * n_units (Born-rule density);
    #          phi_RGB/phi_CMY IS the phase of the Dirac spinor psi_R/psi_L.
    N_RGB: np.ndarray = field(default=None, repr=False)  # type: ignore[assignment]
    N_CMY: np.ndarray = field(default=None, repr=False)  # type: ignore[assignment]
    phi_RGB: np.ndarray = field(default=None, repr=False)  # type: ignore[assignment]
    phi_CMY: np.ndarray = field(default=None, repr=False)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.n_units <= 0:
            raise ValueError(f"n_units must be positive, got {self.n_units}")
        self._allocate_state()

    def _allocate_state(self) -> None:
        """Allocate zeroed state arrays (vacuum) and imprint any momentum phase.

        Backend dispatched on ``self.lattice.backend``: CPU path uses
        ``numpy.zeros``, GPU path uses ``cupy.zeros``.  After
        allocation the session is in the **vacuum** state -- all
        ``N_RGB``/``N_CMY`` counts are zero and ``total_tokens()`` is 0,
        which deliberately violates A=1.  Populate the session via a
        factory (e.g. :meth:`delta_at`) or by writing the state
        arrays directly before handing it to the scheduler.

        If ``self.momentum`` is nonzero, ``phi_RGB`` and ``phi_CMY`` are
        initialised to the linear plane-wave phase ``k.x`` rather
        than to zero; that phase persists at empty sites where it
        does not affect the amplitude.
        """
        backend = get_backend(self.lattice.backend)
        shape = self.lattice.shape
        self.N_RGB = backend.zeros_int(shape)
        self.N_CMY = backend.zeros_int(shape)
        self.phi_RGB = backend.zeros(shape)
        self.phi_CMY = backend.zeros(shape)
        if any(p != 0.0 for p in self.momentum):
            self._apply_momentum_gradient()

    def _apply_momentum_gradient(self) -> None:
        """Imprint ``phi_RGB(x) = phi_CMY(x) += k.x`` from ``self.momentum``.

        `k` IS the plane-wave momentum (lattice units, `a = 1`).
        Paper I applies this multiplicatively to ``psi``; the integer
        formulation here applies it additively to the phase fields,
        which is equivalent under ``psi = sqrt(N/n) * exp(i*phi)``.
        """
        kx, ky, kz = self.momentum
        backend = get_backend(self.lattice.backend)
        coords = backend.indices(self.lattice.shape)
        gradient = kx * coords[0] + ky * coords[1] + kz * coords[2]
        self.phi_RGB += gradient
        self.phi_CMY += gradient

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
            A session whose ``N_RGB[position]`` and ``N_CMY[position]``
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
        session.N_RGB[position] = n_R
        session.N_CMY[position] = n_L
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
        :class:`~dcl_core.core3d.remainder.TokenResidual` for the
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
        N_target_RGB = envelope * n_R
        N_target_CMY = envelope * n_L

        # Reuse the residual's floor + top-up algorithm for the
        # integer quantisation.  A fresh residual starts with carry==0
        # so this is a one-shot quantise, not an evolving accumulator.
        from .remainder import TokenResidual

        residual = TokenResidual(lattice=lattice)
        N_RGB_int, N_CMY_int = residual.quantise(N_target_RGB, N_target_CMY, n_units)

        session = cls(
            lattice=lattice, n_units=n_units, omega=omega, momentum=momentum
        )
        session.N_RGB[...] = N_RGB_int
        session.N_CMY[...] = N_CMY_int
        return session

    @classmethod
    def from_arrays(
        cls,
        lattice: BipartiteLattice,
        n_units: int,
        omega: float,
        N_RGB: np.ndarray | None = None,
        N_CMY: np.ndarray | None = None,
        phi_RGB: np.ndarray | None = None,
        phi_CMY: np.ndarray | None = None,
        *,
        N_R: np.ndarray | None = None,  # deprecated alias -> N_RGB
        N_L: np.ndarray | None = None,  # deprecated alias -> N_CMY
        phi_R: np.ndarray | None = None,  # deprecated alias -> phi_RGB
        phi_L: np.ndarray | None = None,  # deprecated alias -> phi_CMY
    ) -> "DiscreteCausalSession":
        """Construct a session from fully-specified state arrays.

        Parameters
        ----------
        lattice, n_units, omega
            As for the bare constructor.
        N_RGB, N_CMY : int arrays of shape ``lattice.shape``
            Per-site token counts. Must satisfy
            ``sum(N_RGB) + sum(N_CMY) == n_units`` exactly; this is the
            A=1 contract the factory enforces at construction time.
        phi_RGB, phi_CMY : float arrays of shape ``lattice.shape``, optional
            Per-site phases for each component. ``None`` (the
            default) leaves them at zero.

        Returns
        -------
        DiscreteCausalSession
            A session with the supplied state.

        Notes
        -----
        This factory does NOT accept a ``momentum`` argument:
        plane-wave momentum should be baked into ``phi_RGB`` and
        ``phi_CMY`` by the caller (``phi[x,y,z] = kx*x + ky*y + kz*z``)
        so that the factory contract -- "the arrays you pass are
        exactly what the session uses" -- is unambiguous.

        The keyword names ``N_R``/``N_L``/``phi_R``/``phi_L`` are
        accepted as deprecated lattice-frame aliases.
        """
        if any(a is not None for a in (N_R, N_L, phi_R, phi_L)):
            warnings.warn(
                "from_arrays(N_R=, N_L=, phi_R=, phi_L=) are deprecated "
                "aliases; use N_RGB / N_CMY / phi_RGB / phi_CMY",
                DeprecationWarning,
                stacklevel=2,
            )
            N_RGB = N_R if N_R is not None else N_RGB
            N_CMY = N_L if N_L is not None else N_CMY
            phi_RGB = phi_R if phi_R is not None else phi_RGB
            phi_CMY = phi_L if phi_L is not None else phi_CMY
        if N_RGB is None or N_CMY is None:
            raise TypeError("from_arrays requires N_RGB and N_CMY")

        if N_RGB.shape != lattice.shape:
            raise ValueError(
                f"N_RGB.shape {N_RGB.shape} != lattice.shape {lattice.shape}"
            )
        if N_CMY.shape != lattice.shape:
            raise ValueError(
                f"N_CMY.shape {N_CMY.shape} != lattice.shape {lattice.shape}"
            )
        total = int(N_RGB.sum()) + int(N_CMY.sum())
        if total != n_units:
            raise ValueError(
                f"sum(N_RGB) + sum(N_CMY) = {total}, expected n_units = {n_units}"
            )
        if phi_RGB is not None and phi_RGB.shape != lattice.shape:
            raise ValueError(
                f"phi_RGB.shape {phi_RGB.shape} != lattice.shape {lattice.shape}"
            )
        if phi_CMY is not None and phi_CMY.shape != lattice.shape:
            raise ValueError(
                f"phi_CMY.shape {phi_CMY.shape} != lattice.shape {lattice.shape}"
            )
        session = cls(lattice=lattice, n_units=n_units, omega=omega)
        session.N_RGB[...] = N_RGB
        session.N_CMY[...] = N_CMY
        if phi_RGB is not None:
            session.phi_RGB[...] = phi_RGB
        if phi_CMY is not None:
            session.phi_CMY[...] = phi_CMY
        return session

    # ------------------------------------------------------------------
    # Inspection / invariants
    # ------------------------------------------------------------------

    @property
    def epsilon_P(self) -> float:
        """The minimum probability quantum for this session: ``1 / n_units``.

        physics: the "Planck of probability" -- the smallest representable
        probability mass.  Formal symbol ``delta p_min`` (see
        ``dcl-mathematics``); exposed under that name as :attr:`dp_min`.
        """
        return 1.0 / self.n_units

    @property
    def dp_min(self) -> float:
        """``delta p_min`` -- alias of :attr:`epsilon_P`, matching the
        ``dcl-mathematics`` formal symbol."""
        return self.epsilon_P

    # ---- deprecated chirality-frame aliases for the lattice token state ----
    # The fields were renamed N_R/N_L/phi_R/phi_L -> N_RGB/N_CMY/phi_RGB/phi_CMY
    # (lattice frame; see docs/design/03_naming_convention.md).  These
    # properties keep old callers (and downstream pins) working; in-place
    # mutation (`session.N_R[...] = x`) and rebinding both flow through.
    @property
    def N_R(self) -> np.ndarray:  # deprecated alias -> N_RGB
        return self.N_RGB

    @N_R.setter
    def N_R(self, value: np.ndarray) -> None:
        self.N_RGB = value

    @property
    def N_L(self) -> np.ndarray:  # deprecated alias -> N_CMY
        return self.N_CMY

    @N_L.setter
    def N_L(self, value: np.ndarray) -> None:
        self.N_CMY = value

    @property
    def phi_R(self) -> np.ndarray:  # deprecated alias -> phi_RGB
        return self.phi_RGB

    @phi_R.setter
    def phi_R(self, value: np.ndarray) -> None:
        self.phi_RGB = value

    @property
    def phi_L(self) -> np.ndarray:  # deprecated alias -> phi_CMY
        return self.phi_CMY

    @phi_L.setter
    def phi_L(self, value: np.ndarray) -> None:
        self.phi_CMY = value

    def total_tokens(self) -> int:
        """Return ``sum(N_RGB) + sum(N_CMY)``.

        For a populated, A=1-conserving session this equals
        ``n_units`` exactly.  A bare-constructed (vacuum) session
        returns 0 here -- the constructor does not enforce A=1; the
        factory methods or direct array writes do.

        On the GPU backend, the ``int()`` cast forces a
        device-to-host sync; that is the right semantic for
        "give me the number now."
        """
        backend = get_backend(self.lattice.backend)
        return int(backend.sum_all(self.N_RGB) + backend.sum_all(self.N_CMY))

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

    def amplitude(
        self, component: Component | None = None, *, chirality: Component | None = None
    ) -> np.ndarray:
        """Return the derived complex amplitude of a sublattice component.

        ``component`` selects the RGB or CMY sublattice component (the
        lattice frame); "R"/"L" are accepted physics-frame aliases. The
        old keyword ``chirality=`` is accepted as a deprecated alias.

        physics: the returned field IS the Dirac spinor component
        ``psi = sqrt(N(x) / n_units) * exp(i * phi(x))`` -- the physics
        reading of the lattice state ``(N, phi)``.  It is a derived,
        freshly-allocated snapshot (not a view): the engine evolves the
        integer ``N`` and continuous ``phi`` directly and never reads
        back from this complex field.  Provided for introspection /
        plotting only.
        """
        if chirality is not None:
            warnings.warn(
                "amplitude(chirality=) is a deprecated alias; use component=",
                DeprecationWarning,
                stacklevel=2,
            )
            component = chirality if component is None else component
        if component in ("RGB", "R"):
            N, phi = self.N_RGB, self.phi_RGB
        elif component in ("CMY", "L"):
            N, phi = self.N_CMY, self.phi_CMY
        else:
            raise ValueError(
                f"component must be 'RGB'/'CMY' (or alias 'R'/'L'), "
                f"got {component!r}"
            )
        backend = get_backend(self.lattice.backend)
        # psi(x) IS sqrt(N(x) / n_units) * exp(i * phi(x)).  The division
        # promotes int64 -> float64; the `1j * phi` promotes float64 ->
        # complex128; the product is complex128.
        return backend.sqrt(N / self.n_units) * backend.exp(1j * phi)
