# src/dcl_core/

The installable Python package.  Two submodules with complementary
roles:

## `dcl_core.core` -- continuous-amplitude engine (Paper~I)

Verbatim port of Paper~I's `src/core/`
([doi:10.5281/zenodo.20078529](https://doi.org/10.5281/zenodo.20078529)).
A=1 is enforced as a float-tolerance renormalisation via
`enforce_unity_spinor` at the end of every tick.

| Symbol | Module | Role |
|---|---|---|
| `OctahedralLattice` | `core/OctahedralLattice.py` | Bipartite lattice geometry |
| `CausalSession` | `core/CausalSession.py` | Continuous (`psi_R`, `psi_L`) state on the lattice |
| `CompositeCausalSession` | `core/CompositeCausalSession.py` | Multi-session composite |
| `TickScheduler`, `ShuffleScheme` | `core/TickScheduler.py` | Tick orchestration |
| `PhaseOscillator` | `core/PhaseOscillator.py` | Phase-clock primitive |
| `enforce_unity*`, `unity_residual*`, `is_unity` | `core/UnityConstraint.py` | Float-tolerance A=1 enforcement |
| `RGB_VECTORS`, `CMY_VECTORS`, `ALL_VECTORS`, `SUBLATTICE_SIZE`, `COORDINATION_NUMBER`, `active_vectors`, `EVEN_TICK`, `ODD_TICK` | `core/OctahedralLattice.py` | Geometric constants |

Use this submodule when:

- Reproducing a Paper~I experiment unchanged.
- Migrating an experiment from Paper~I's `from src.core import ...`
  pattern to a pip-installable dependency (one-line import rewrite).

## `dcl_core.core3d` -- integer-token engine (new design)

Active development.  Integer-token probability accounting:
`N_R`/`N_L` are `int64` counts and `phi_R`/`phi_L` are continuous
phase fields; A=1 is the exact integer equality
`sum_x (N_R + N_L) == n_units`.  The hop's fractional output is
absorbed by the Bresenham residual accumulator so the integer
identity holds without per-tick renormalisation.

| Symbol | Module | Role |
|---|---|---|
| `BipartiteLattice` | `core3d/lattice.py` | RGB/CMY sublattice geometry (frozen dataclass) |
| `DiscreteCausalSession` | `core3d/session.py` | Integer-token + phase state on the lattice |
| `HopOperator` | `core3d/hop.py` | Bipartite Dirac evolution (analytical output) |
| `BresenhamResidual` | `core3d/remainder.py` | Fractional-bit carry between ticks |
| `TickScheduler` | `core3d/scheduler.py` | Multi-session orchestration |

CPU (NumPy) and GPU (CuPy) backends via `core3d/backends/`; backend
selected at `BipartiteLattice` construction time.

State of implementation (v0.1.0-dev): all `core3d` operators raise
`NotImplementedError`.  The architectural decisions (interfaces,
public API, documentation conventions, CI matrix) are in place; the
implementation is the next phase.  See `CLAUDE.md`'s `CURRENT
STATUS` for the roadmap.

## Top-level package: no shortcuts

`dcl_core/__init__.py` re-exports `__version__` and the two
submodules (`core`, `core3d`), and nothing else.  This is
deliberate: the two submodules have different shapes (continuous
amplitude vs integer tokens), and a top-level shortcut would hide
which engine is in play.

```python
# Choose your engine explicitly:
from dcl_core.core import OctahedralLattice, CausalSession      # Paper I shape
from dcl_core.core3d import BipartiteLattice, DiscreteCausalSession  # new shape
```

Public API contract (per submodule):

- Adding a re-export from `<submodule>/__init__.py`'s `__all__`
  is a MINOR version bump (backwards compatible).
- Removing or changing a re-export signature is a MAJOR version
  bump.
- Pure internal refactoring is a PATCH.

The `.claude/agents/api-stability-reviewer.md` agent enforces this
on diffs.

## Documentation convention

Every non-trivial line of physics / framework code should say what
it **is** in the theory, not just what it does in the program.  Name
the mathematical object (e.g. `gamma_0`, `delta_phi`, `n_units`),
cite the design / reference doc where one exists, and state the
correspondence explicitly: "this IS X" when exact, "approximates X"
in the continuum limit.

When adding new physics code, follow the same pattern:

- Name the mathematical object.
- State what the variable **is**, not what you are doing with it.
- Use "IS" for exact correspondences, "approximates" for continuum
  limits.
- Cross-reference `docs/design/*.md`, `docs/reference/*.md`, or
  `notes/*.md` where one exists.

The `.claude/agents/physics-naming-reviewer.md` agent flags code
that slips into operational-role naming.

## Cross-validation

`tests/test_cross_validation.py` (STUB) holds the convergence test
linking the two submodules: as `n_units -> infinity` in `core3d`,
its dynamics must reproduce `core`'s.  Any divergence between the
two at large `n_units` is a bug in one of them.  The cross-
validation test is the de-facto unifying spec.

## What goes here

- Lattice geometry primitives (both submodules).
- Session state and lifecycle (`core` continuous, `core3d` integer-
  token).
- Bipartite Dirac evolution operators.
- The remainder / residual accumulator (`core3d` only).
- Tick schedulers.
- Backend-specific kernels (`core3d/backends/`).

## What does NOT go here

- Experiment scripts -- those live in `experiments/`.
- Plotting and figure generation -- separate utilities, downstream.
- Paper-section content -- belongs in a separate paper repo that
  depends on this one.
- One-off analysis scripts -- keep this package focused.
