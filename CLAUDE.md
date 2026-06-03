<!-- markdownlint-disable MD022 MD025 MD033 MD060 -->
# CLAUDE.md -- Working Brief for Claude Code

> Project: `dcl_core` -- Core library for the A=1 Discrete Causal
> Lattice framework (a research-software package).

This file is the project memory for Claude Code. Keep it updated so a
new conversation can continue work without the full chat history.

The structure below is a starting point: replace the placeholder
sections with project-specific content as the work develops, and keep
the **CURRENT STATUS** block at the top up to date.

---

## CURRENT STATUS (2026-06-02) -- v0.1.0, core3d modules landed; v1.0 gap is cross-validation only

Repository created on 2026-05-16 from the user's `dcl-core-template`
(a Python-package-shaped GitHub template, separate from the
`dcl-paper-experiment-template` used for the papers in the series).
The template lays out a substantively NEW design relative to
Paper~I's `src/core/`: **integer-token probability accounting**
(`N_units` integer counts vs Paper~I's continuous-amplitude
`psi_R`/`psi_L`), a Bresenham-style residual accumulator for
fractional-bit carry between ticks, a CPU/GPU backend split, and a
semver-enforced public-API surface.

The `core3d` integer-token engine is **implemented and unit-green**,
not stubbed (the earlier "all stubs raise `NotImplementedError`"
status was superseded once the modules landed).  Phase~1 triage
(issue `dcl-project/007`) on 2026-06-02:

- All five public operators (`BipartiteLattice`,
  `DiscreteCausalSession`, `HopOperator`, `BresenhamResidual`,
  `TickScheduler`) plus `clifford` are implemented (~1.6k LOC).
- `tests/core3d/*` -- **67 passed** (lattice, session, hop,
  remainder, conservation, continuum_limit, clifford, scheduler).
- `tests/test_smoke.py` -- pass.
- `tests/test_cross_validation.py` -- **4 tests still
  `@pytest.mark.skip`**.  This is the entire remaining v1.0 gap.

The work is now the **`core3d` -> v1.0 upgrade** tracked by issue
`dcl-project/007` (Phases 2-4): land the cross-validation layer,
stabilise the public API, then cut `dcl-core v1.0.0`.

**Cross-validation triage finding (2026-06-02).**  Of the 4
`test_cross_validation.py` tests, only `test_conservation_invariants_agree`
is genuinely cross-engine-unblocked -- each engine conserves its
*own* A=1 invariant (core3d: exact integer identity; core: float
renorm to `< 1e-10`), so the test needs no agreement of dynamics.
**DONE** -- implemented and passing.  The other three all require
the two engines' *dynamics* to converge, which is gated on the
**engine-protocol mismatch**: core3d hops a uniform average with
periodic boundaries and one chirality per tick, while core hops a
directed momentum-weighted kernel with open boundaries and both
chiralities per (massive) tick.  At fixed lattice spacing they pin
different observables (delta-p-min: core3d `r_peak = 19.63` vs
core/Paper~I `R_1 = 10.3`) and agree only in the *double* limit
(`a -> 0` AND `N -> infinity`).  So the `1/sqrt(N)` scaling in
`test_free_propagation_matches_in_large_N_limit` governs core3d's
deviation from its *own* analytic amplitude, NOT the gap to core --
the cross-engine difference does not vanish.

**Cross-validation progress.**  ALL 4 `test_cross_validation.py`
tests now PASS (suite: 232 passed, 26 xfailed, 0 skipped):
- `test_conservation_invariants_agree` (2026-06-02) -- each engine
  conserves its own A=1 invariant.
- `test_free_propagation_matches_in_large_N_limit` (2026-06-02) --
  **reframed** as a core3d *self-convergence* test: the integer-token
  density tracks the continuous amplitude it quantises, with L2 error
  `O(sqrt(n_sites)/n_units)` (deterministic Bresenham, faster than
  Poisson `1/sqrt(N)`).  The cross-engine "evolves identically"
  reading is not well-posed at fixed lattice.
- `test_two_body_orbit_locks_in_both_cores` (2026-06-03, `slow`) --
  **respec'd**: started as an orbiting wavepacket in the same fixed
  `V = -30/(r+0.5)` Coulomb well (single session + external potential
  -- core3d has no pairwise coupling in v0.1.0), each engine settles
  to a *stable, stationary, interior* radius (low late-window cv).
  Does NOT assert a shared `R_1 = 10.3`: at fixed 33^3 spacing core
  locks ~12, core3d ~21 (the engine-protocol mismatch), and the test
  pins that >0.2 divergence as a tripwire.  Mirrors Paper~I
  `exp_10_standalone` (core) and `exp_12_dp_min_sweep` (core3d).

- `test_arnold_tongue_locations_agree` (2026-06-03, `slow`) --
  **reframed** to the *single-session orbital resonance* (Paper~I
  `exp_09_harmonics` Part~B / the form `dcl-delta-p-min`'s
  `exp_09_dp_min_floor` uses for its core3d column).  Sweeps `omega`
  for a packet on a tangential orbit in the same fixed Coulomb well;
  each engine must show a *frequency-dependent* orbital lock-in (the
  centre-of-mass radial swing varies materially across `omega`) while
  staying bound.  Does NOT assert the engines lock at the same
  `omega` -- pins that the tightest-lock `omega` *differs* (fixed-
  lattice mismatch tripwire).  CORRECTION: an earlier note here called
  this test "blocked on v0.2.0 inter-session coupling" -- that was an
  over-claim.  Only the *2-parameter coupled-oscillator* tongue
  (Part~D: two dynamical sessions, frequency-ratio x coupling) needs
  v0.2.0 coupling; the single-session orbital-resonance form runs at
  v0.1.0, which is why delta-p-min's `exp_09` is blocked on `core`'s
  `prob_floor`, NOT on core3d.  The richer Part~D tongue remains a
  v0.2.0 follow-on.

**Next concrete actions, in dependency order (issue 007 Phase 2-4):**

1. Phase~3: audit/stabilise the public API re-exported from
   `src/dcl_core/core3d/__init__.py`; document the v1.0 contract.
   (All 4 cross-validation tests PASS; the v1.0 cut is unblocked.)
2. Phase~4: bump `_version.py` + `CITATION.cff`, write
   `release_notes/`, deposit `dcl-core v1.0.0` on Zenodo, then run
   the downstream pin-bump coordination.

**What is NOT in scope for the v1.0 cut:**

- GPU backend (CuPy).  Stub the `backends/gpu/` interface but
  defer the CUDA implementation to v0.2.0.
- Multi-session entanglement / pairwise emission rules from
  Paper~I's `TickScheduler`.  v0.1.0 ships the scheduler
  skeleton, not the full emission machinery.
- A full port of Paper~I's `exp_*.py` experiments.  Paper~I and
  Paper~II are unaffected by dcl_core's release (they remain
  self-contained with their own vendored / inline implementations).
  Paper~III (the first downstream consumer) and its migration are
  a separate decision -- see *Downstream papers* below.

**Downstream papers (migration outlook).**

The framework series uses `dcl_core` in three different ways:

- **Paper~I (`dcl`).**  Vendored copy of the original
  continuous-amplitude engine in its own `src/core/`.  Stays
  pinned at Paper~I's v1.0 release; not migrated.  Paper~I's
  Zenodo deposit is immutable.
- **Paper~II (`dcl-paper-02-sm-derivation`) and the generator zoo
  (`dcl-generator-zoo`).**  Pure symbolic / sympy work today; no
  dependency on the engine *yet*.  Caveat: Paper~II's `C^3`
  colour-memory algebra (the SU(3) factor) is exactly what a future
  `core3d` proton-internals implementation would import and make run
  numerically -- at that point the "no dependency" statement stops
  being true.  See `notes/color_structure.md`.
- **Paper~III (`dcl-paper-03-tidal-ionization`).**  The first
  downstream consumer.  Resolved (2026-05-16): Paper~III pins
  `dcl_core` via
  `dcl_core @ git+https://github.com/JackDMenendez/dcl-core@<TAG>`
  in its `virtual-env-requirements.txt`.  Its `src/experiments/exp_18_tidal_ionization.py`
  imports from `dcl_core.core` directly; no vendored copy.

**Bump-and-rebuild rule (READ THIS BEFORE EVERY DCL-CORE RELEASE).**

Every `dcl-core` release triggers a *bump-and-rebuild* workflow in
every downstream paper repo that pins `dcl_core`.  After tagging
and pushing a `dcl-core` release, walk each downstream consumer
listed above and **bump its pin from `@main` (or the previous
`@vX.Y.Z`) to the new tag** before the consumer's next release.

The detailed procedure -- including the co-released order
(dcl-core deposit FIRST, then paper-side pin bump, then paper
deposit) -- lives in two places:

- `release_notes/README.md` here in dcl-core (see *Downstream
  paper coordination* section).
- Each consumer paper's own `release_notes/README.md`
  (Paper~III's includes a *Pre-release: bump pinned dcl_core* section).

Do not skip the bump.  Do not reverse the order.  A downstream
paper that ships pinned to `@main` is non-reproducible by
construction; the bump is what restores its reproducibility
guarantee.

Update this block whenever the answer to "what is the next action"
changes.

---

## What This Project Is

`dcl_core` is a Python implementation of the A=1 Discrete Causal
Lattice framework with **integer-token probability accounting**.  A
session is a fixed budget of $N$ indistinguishable probability
tokens distributed over a bipartite octahedral lattice; A=1 means
$\sum_x N(x) = N$ exactly, by integer arithmetic.  The library
exposes lattice geometry (`BipartiteLattice`), the hop operator
(`HopOperator` -- bipartite Dirac evolution), the residual
accumulator (`BresenhamResidual` -- fractional-bit carry between
ticks), and the tick scheduler (`TickScheduler` -- multi-session
orchestration).  CPU (NumPy) and GPU (CuPy) backends are
addressable through a single API surface; the backend is selected
at lattice-construction time.

Papers in the A=1 Discrete Causal Lattice series that depend on
this core pin to a specific Zenodo-deposited version (e.g.\
`dcl_core==0.1.0` in `pyproject.toml` or
`virtual-env-requirements.txt`).  The core can evolve at its own
cadence; each paper's reproducibility is anchored to a specific
software release, not a moving target.

The relationship to Paper~I: Paper~I's `src/core/` is the
*original* (continuous-amplitude, float-tolerance A=1)
implementation.  `dcl_core` is a *re-implementation* with integer
tokens, separating the analytical hop step
(`HopOperator.step`, fractional output) from the integer-token
update (`BresenhamResidual`, the fractional-bit carry).  The
intent is that the continuum limit ($N \to \infty$) recovers
Paper~I's dynamics exactly while making A=1 a hard integer
identity at finite $N$.

---

## Package Layout

Two submodules, both installable via `pip install dcl_core`.  The
top-level `dcl_core/__init__.py` re-exports only `__version__` and
the submodules themselves; **no top-level shortcuts**, so the
choice of engine is explicit at every import site.

### `dcl_core.core` -- continuous-amplitude engine (Paper~I port)

| Module | Role |
|---|---|
| `src/dcl_core/core/OctahedralLattice.py` | Bipartite lattice geometry + constants (`RGB_VECTORS` etc.) |
| `src/dcl_core/core/CausalSession.py` | Continuous (`psi_R`, `psi_L`) state |
| `src/dcl_core/core/CompositeCausalSession.py` | Multi-session composite |
| `src/dcl_core/core/TickScheduler.py` | Tick orchestration + `ShuffleScheme` |
| `src/dcl_core/core/PhaseOscillator.py` | Phase-clock primitive |
| `src/dcl_core/core/UnityConstraint.py` | Float-tolerance A=1 enforcement |

Verbatim port of Paper~I's `src/core/`; the six module files are
unmodified, only `core/__init__.py` is new (re-exports the Paper~I
public surface).  Ready to use immediately.

### `dcl_core.core3d` -- integer-token engine (new design)

| Module | Role |
|---|---|
| `src/dcl_core/core3d/lattice.py` | `BipartiteLattice` (frozen dataclass) |
| `src/dcl_core/core3d/session.py` | `DiscreteCausalSession` (integer tokens + phase) |
| `src/dcl_core/core3d/hop.py` | `HopOperator` (bipartite Dirac, analytical output) |
| `src/dcl_core/core3d/remainder.py` | `BresenhamResidual` (fractional-bit carry) |
| `src/dcl_core/core3d/scheduler.py` | `TickScheduler` (multi-session) |
| `src/dcl_core/core3d/backends/` | CPU / GPU implementations |

All operators raise `NotImplementedError` at v0.1.0-dev; see
`CURRENT STATUS` for the implementation roadmap.

Public API per submodule is whatever its `__init__.py` re-exports;
anything not re-exported is internal and may change without a
semver bump.  Cross-validation between the two submodules lives in
`tests/test_cross_validation.py` (STUB).

---

## Conventions

- **Versioning.** Semver. Public API changes (re-exports from
  `__init__.py`) require a **minor** bump (backwards-compatible
  additions) or **major** bump (breaking changes). Internal
  refactoring is **patch**. Pre-1.0 majors signal "API still
  unstable, expect breakage."
- **File naming.** Modules: `<topic>.py` (lowercase, single noun).
  Tests: `tests/test_<topic>.py`. Experiments:
  `experiments/exp_NN_<short_name>.{py,md}`. Docs:
  `docs/<category>/<topic>.md`.
- **Documentation convention for code (two-frame, `core3d`).** Code in
  `dcl_core.core3d` lives in two frames: the **name** says what the
  object IS in the lattice's own mathematics (aligned with
  `dcl-mathematics`' formal symbols -- `Lambda_d`, `V_d^+/V_d^-`,
  `coord(d)`, `N_RGB/N_CMY`, `epsilon_P`/`dp_min`, `TokenResidual`), and
  a **`# physics:` comment** names the existing-physics correspondence
  (IS for exact, approximates for continuum limits). The physics-frame
  derived amplitude (`psi_R`/`psi_L`) and the protected `RGB`/`CMY`
  geometry stay as-is. Full rule + glossary:
  `docs/design/03_naming_convention.md`. `dcl_core.core` (Paper~I port)
  is frozen and exempt.
- **Test discipline.** Every conservation law and continuum-limit
  claim gets a `tests/test_*.py` entry. Integer-A=1 tests assert
  equality with no float tolerance. Tests for the continuum limit
  parametrize over N or lattice spacing and check convergence.
- **API stability.** Any change to a symbol re-exported from
  `src/dcl_core/__init__.py` must be deliberate: either a semver bump
  or a deprecation cycle. The `.claude/agents/api-stability-reviewer.md`
  agent enforces this on diffs.
- **Naming review.** The `.claude/agents/physics-naming-reviewer.md`
  agent enforces the two-frame convention on `core3d`: it flags
  operational-role names, lattice *state* named only in the physics
  frame where a `dcl-mathematics` term applies, and non-trivial lattice
  lines missing their `# physics:` correspondence comment.
- **core3d naming retrofit (2026-06-03).** `core3d` adopted the
  two-frame convention with a public rename, all carrying
  backward-compatible deprecation shims:
  `BresenhamResidual` -> `TokenResidual` (class alias kept);
  session state `N_R/N_L/phi_R/phi_L` -> `N_RGB/N_CMY/phi_RGB/phi_CMY`
  (property aliases kept); `amplitude` takes `component=` ("RGB"/"CMY",
  with "R"/"L" + the old `chirality=` kwarg accepted); `epsilon_P`
  gains a `dp_min` alias; `BipartiteLattice` gains `coordination`.
  Renamed *keyword* params (`from_arrays`, `amplitude`,
  `TokenResidual.quantise`) accept the old keywords with a
  `DeprecationWarning`. Net semver: the renames are breaking, but
  pre-1.0 + every old name still resolves -- treat this as the **v1.0
  API freeze** (new names canonical, aliases deprecated). Drop the
  aliases no earlier than the first MAJOR after 1.0.

---

## Test invariants (what the suite protects)

These tests should exist and run on every commit. Add to this list as
new invariants are formalised.

- `test_conservation.py` -- `sum(N(x))` exact integer equality across
  N ticks.
- `test_hop.py` -- hop operator preserves expected symmetries
  (lattice rotations, bipartite parity).
- `test_remainder.py` -- Bresenham accumulator carries fractional
  bits correctly; long-run drift bounded by epsilon_P.
- `test_continuum_limit.py` -- as `N -> infinity` and lattice spacing
  `a -> 0`, dispersion converges to `E^2 = m^2 + |p|^2`.
- `test_clifford.py` -- `{gamma_mu, gamma_nu} = 2 eta_{munu} I`
  (sympy verification of the Clifford algebra).
- `test_gauge_invariance.py` -- Peierls-coupled hop is invariant
  under `A_mu -> A_mu + d_mu Lambda`.

---

## Release flow

See `release_notes/README.md` for the full procedure. Summary:

1. CI green on `main`.
2. Bump `src/dcl_core/_version.py` and `CITATION.cff`
   (`version`, `date-released`).
3. Draft `release_notes/vX.Y.md` and `release_notes/vX.Y-release-message.md`.
4. **Deposit on Zenodo first** -- the DOI lands in `CITATION.cff`
   *before* the release commit.
5. Commit version bump (DOI included).
6. Tag `vX.Y`, push the tag.
7. Create the GitHub Release using the release-message body.
8. (Optional) Publish to PyPI.

Downstream paper repos pin to the released version
(`dcl_core==X.Y.Z` in their requirements). Once released, a version
is **immutable** -- never amend a tagged commit.

---

## What NOT to Change

- The bipartite RGB / CMY sublattice geometry: it IS the Dirac
  structure.
- The A=1 constraint (whatever its current implementation -- integer
  tokens or continuous amplitude with renormalisation): keep, just
  understand which mode you are in.
- The two-frame naming convention (`docs/design/03_naming_convention.md`)
  in `core3d`, and the math-analog naming in `core`: this discipline
  keeps the code legible to future readers (humans and Claude). The
  protected `RGB`/`CMY` geometry names stay.
- Public API symbols without a deprecation cycle.

---

## Cross-references to the A=1 series repositories

The series is decomposed across several repos.  For local
Claude / agent work, expose them as Windows directory junctions
under `external/`.  All `external/` paths are gitignored, so the
junctions are not part of the committed repo.

- **`external/dcl`** $\to$ `C:\dev\dcl` -- Paper~I's repo.
  Vendored copy of the original (continuous-amplitude) engine
  lives at `external/dcl/src/core/`.  Useful for cross-checking
  `dcl_core`'s integer-token implementation against the reference
  dynamics, and for lifting Paper~I's geometric conventions
  (lattice basis vectors, parity rule) directly into
  `BipartiteLattice`.
- **`external/dcl-paper-02-sm-derivation`** $\to$
  `C:\dev\dcl-paper-02-sm-derivation` -- Paper~II.  Pure sympy /
  symbolic; no engine dependency today (see colour-memory caveat in
  *Downstream papers*).  The SU(3)/colour derivation lives here:
  `src/utilities/automorphism_rgb_su3.py` (RGB -> only $\mathbb{Z}_3$)
  and `src/utilities/su3_generation_from_colour_memory.py` ($\mathbb{C}^3$
  memory -> full $\mathfrak{su}(3)$).
- **`external/dcl-generator-zoo`** $\to$ `C:\dev\dcl-zoo` --
  the generator zoo (catalogue of the 71-dim per-site
  automorphism algebra).  Pure sympy; no engine dependency.
- **`external/dcl-paper-03-tidal-ionization`** $\to$
  `C:\dev\dcl-paper-03-tidal-ionization` -- Paper~III, the first
  candidate downstream consumer of `dcl_core` (see *Downstream
  papers* above).
- **`external/research`** $\to$ `C:\dev\physics-research` --
  parallel formalisation effort (notation, algebra, topology,
  balanced $\mathcal{A}=1$ equations).  Findings during dcl_core
  work that touch notation should flow upstream to
  `external/research/Notes/`.

To (re)create on Windows:

```bat
mkdir external
mklink /J external\dcl C:\dev\dcl
mklink /J external\dcl-paper-02-sm-derivation C:\dev\dcl-paper-02-sm-derivation
mklink /J external\dcl-generator-zoo C:\dev\dcl-zoo
mklink /J external\dcl-paper-03-tidal-ionization C:\dev\dcl-paper-03-tidal-ionization
mklink /J external\research C:\dev\physics-research
```

---

## Notes Index (important theoretical / scratchpad files)

`notes/README.md` -- conventions for notes/

Landed notes:

- `notes/color_structure.md` (DRAFT) -- the corrected colour
  picture for proton internals: geometric RGB gives only
  $\mathbb{Z}_3$; real SU(3) colour lives on a separate $\mathbb{C}^3$
  colour-memory factor (Paper~II result), and implementing it in
  `core3d` is what would first couple `dcl_core` to Paper~II.

Notes expected to land during v0.1.0 development:

- `notes/structure_factor_derivation.md` (planned) -- the
  Fourier-space hop kernel that supports
  `HopOperator.fourier_kernel`.  Documents the small-$k$ Taylor
  expansion and the continuum-limit identification
  $\mathrm{hop} \to i \mathbf{k} \cdot \boldsymbol{\gamma}_{RGB}$.
- `notes/bresenham_residual_design.md` (planned, optional --
  may live in `docs/design/` instead) -- the fractional-bit-carry
  algorithm; how the analytical hop's continuous output is rounded
  to integer-token updates without long-run drift.

(List individual notes here as they accumulate.  Notes are
durable working documents; release-notes refer back to them for
context.)
