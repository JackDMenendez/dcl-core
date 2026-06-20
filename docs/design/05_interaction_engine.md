# 05 — core3d as an interaction engine: session / field architecture

> **Status: LIVING.** This document is **continuously maintained** — unlike
> the requirements docs (e.g. `04_gauge_field_and_vacuum_response.md`,
> frozen at proposal) and unlike release notes (frozen at tag), this one is
> updated as the design evolves. If a section here disagrees with the code,
> the code is right and this doc is stale — fix the doc.
>
> **Scope.** How `core3d` grows from *independent-session evolution*
> (v0.1.0: every session steps alone) into a *multi-session interaction
> engine* (particles that couple through shared fields). The U(1) gauge
> coupling for Paper IV `exp_03` is the **first concrete driver**
> (requirements: `04_gauge_field_and_vacuum_response.md`); the architecture
> here is meant to outlive it.
>
> **Companions:** class diagram `notes/figures/core3d-classes.md`;
> implementation plan + open-decision scratchpad `notes/gauge_field_v030_plan.md`.

---

## 0. Governing principle — the simple ideas are frozen

The original contracts stay fixed; new capability is **additive** and lands
in `core3d` (engine policy, `04_*.md` §1.1). Concretely:

- **A=1** is the integer token identity — never relaxed.
- **`N_RGB` / `N_CMY` semantics are frozen** (bipartite token densities).
  New physics (gauge field, colour memory) goes in **new arrays**, never by
  overloading these. This is what keeps the `.npy` contract and
  `dcl-lattice-viewer` reading correctly forever (§4).
- **The bipartite RGB/CMY geometry** IS the Dirac structure — untouched.
- **Honest limitations.** Every capability is paired with what it *cannot*
  do (§6). If we can't do something, we list it — we never imply it works.

---

## 1. The layered architecture

| Layer | Object | State | Shared? |
|---|---|---|---|
| Geometry | `BipartiteLattice` | frozen (immutable) | one shared instance, by reference |
| Particle | `*Session` (≤ `AbstractSession`) | token counts + phase, per session | one per particle |
| Interaction | `Field` (e.g. `GaugeFieldU1`) | `A0`/`A`, **mutable**, recomputed per tick | one shared instance **per kind** |
| Orchestration | `TickScheduler` | tick counter, residuals | the driver |
| Evolution | `HopOperator` | stateless | the operator |

**Fields are "global like the lattice".** A `Field` is *not* a process/module
global. Like `BipartiteLattice`, it is a single instance the experiment
constructs and **passes by reference**, identity-shared per simulation — so
each test/run builds its own and there is no cross-test state bleed. It
differs from the lattice in two ways the design must respect: there can be
**several fields, one lattice** (a session names which it couples to), and a
field is **mutable** (recomputes `A0`/`A` each tick) where the lattice is
frozen. A field carries a `BipartiteLattice` reference (it is lattice-shaped).

See `notes/figures/core3d-classes.md` for the class diagram (current code +
this proposed overlay).

---

## 2. The abstraction discipline (dependency inversion)

**The engine (`TickScheduler`, `HopOperator`) depends on `AbstractSession`
and `Field` only — it branches on no concrete particle type.** Type
knowledge lives in the session hierarchy, reached through two virtuals
(Python: every method is virtual; the root is an `abc.ABC`):

- **`field_specs() -> list[FieldKey]`** — *which* field(s) this session
  couples to. Overridden **high** in the hierarchy (the coupling group,
  e.g. an intermediate `ChargedSession` → the U(1) key). "Knows what field
  it modifies."
- **`deposit_source(field)`** — *how much* this session contributes (charge
  sign/magnitude; a composite proton's form factor). Overridden **low**
  (the species). "The more specific inheritor knows what factory to call."

*Coupling decided high, magnitude decided low* is the only structural
commitment. The **particle taxonomy itself is open** (§5) — and the same
virtual dispatch dissolves the "is a field keyed per-interaction or
per-species?" question: each class's override answers it for itself.

**Per-tick flow (scheduler branches on nothing):**
`register(session)` matches `session.field_specs()` to the passed-in fields
and `field.add_member(session)`. Each `step()`:
1. `field.recompute()` for every field — reads members' **start-of-tick**
   state (Jacobi; see D1b, §5).
2. per session: `A0, A = session.gauge_potential(parity)` — pulls
   `field.potential_for(self)` (self-source excluded; see D1c, §5).
3. `hop.step(session, parity, external_potential=A0, vector_potential=A)`.
4. renormalise ψ → `residual.quantise(...)` → write back `N_*`, `phi_*` →
   `assert_unity()`.

With **no field attached**, this is v0.1.0 bit-for-bit (§3).

**Concurrency — reentrant, lock-free (no mutexes).** Parallelism is coming
(R5: orientation sweeps, multi-process/-device fan-out). The engine must run
it **without locks** — achieved by structure, not synchronisation:

- **Pure operators.** `HopOperator.step` is stateless and does not mutate the
  session (guarded by `test_step_does_not_mutate_session`). Pure ⇒ reentrant:
  N orientations/sessions hop concurrently with zero shared mutable state.
- **Immutable shared data.** `BipartiteLattice` is frozen ⇒ lock-free shared
  by reference. Any precomputed static link-phase cache (Phase 4 perf) must be
  **built eagerly at construction and read-only thereafter** — never a
  lazily-filled mutable cache (the first-write race is exactly what forces a
  lock).
- **Per-worker ownership of mutable state.** Each session owns its `(N, phi)`
  arrays and its own `TokenResidual.carry` (scheduler keys residuals per
  session). Partition sessions across workers ⇒ disjoint writes, no sharing.
- **Double-buffered fields = Jacobi (D1b).** The one shared *mutable* object is
  the dynamic `Field`. Split each tick into an all-read phase (workers read the
  current, read-only field) and a recompute phase (write *next* from the
  start-of-tick snapshot), barrier between. That IS Jacobi double-buffering —
  so **D1b's Jacobi choice is also the concurrency choice**; Gauss-Seidel would
  force mid-tick read-after-write and a lock.
- **No mutable module globals.** Backends are stateless dispatch; the `Field`
  is passed by reference, not a process-global — the earlier "global like the
  lattice, not a module global" decision now also buys race-freedom.

Net: lock-free-ness falls out of **immutability + purity + partitioning +
Jacobi**. Keep those four and parallelism needs no mutexes; introducing shared
mutable state (a lazy cache, a Gauss-Seidel field) is the design smell that
would force a lock.

---

## 3. Physics vs architecture — the boundary that must hold

**The test:** a change is *architecture* if, with no field attached,
`TickScheduler` + `hop.step` reproduce v0.1.0 **bit-for-bit** (acceptance
test #1). It is *physics* if it can move an observable. Almost everything in
§1–§2 (abstract session, particle subclasses, field object model, the `.npy`
writer) is architecture. The physics is small and enumerated:

**`hop.step` — exactly one physics change:**
- **Magnetic Peierls coupling** — `exp(i·A_mid·v)` per shifted amplitude
  (`A_mid = ½(A(x)+A(x+v))`). Guarded by the Ward identity (test #2) and the
  `Q`-tensor cross-check (test #4).
- *Surprise:* the **electric / Coulomb** sector needs **no** hop change — a
  scalar potential is the on-site `delta_phi = omega + V(x)`, which
  `hop.step` already accepts via `external_potential`. The resonating-Coulomb
  multi-session case rides existing hop machinery; the hop's only new physics
  is magnetism.

**`TickScheduler` — the physics it gains:**
- **(1) Inter-session field sourcing — the core change.** v0.1.0 sessions
  evolve independently; the moment the scheduler builds a field from other
  sessions' densities and feeds it into their hops, particles **interact**.
  This is the actual "interaction engine".
- **(2) Threading the on-site potential.** Today `scheduler.step` drops
  `external_potential`; wiring it through is what lets the Coulomb case exist
  through the scheduler.
- **(3) Update ordering (D1b).** Jacobi vs Gauss-Seidel is a *discretisation
  of the interaction*, not a software choice — it can shift observables.
  Recommend **Jacobi** (order-independent ⇒ reproducible ⇒ clean `.npy`).
- **(4) Self-field (D1c).** Whether a charge feels its own field is a
  self-energy modelling choice; lives in `field.potential_for(session)`.
- **(5) Electric-sector sign/normalisation.** Pin `A0`'s convention against
  Paper I App. B — a silent sign error is invisible to software tests and
  fatal to tests #4/#5.

The existing scheduler renorm does **not** change: per-link
`|exp(i·A_mid·v)| = 1`, so the Peierls phase preserves the norm and does not
perturb the renorm (tests #1/#6 confirm).

---

## 4. Data (`.npy`) contract impact — additive, no break

The `.npy` output is derived from session (and now field) state; the contract
is `docs/data_deposit_and_provenance.md`.

- **Existing per-session stream survives untouched.** Particle subclasses
  *inherit* `N_RGB`/`N_CMY`/`phi_*`; a writer reading those keeps working and
  the viewer contract is unchanged. The abstraction is invisible to `.npy`.
- **The inversion makes the writer cleaner.** A writer depending on
  `AbstractSession` dumps any of the (≤71) kinds without knowing the type —
  argues for a `snapshot() -> dict[str, ndarray]` on the abstract base.
- **New dumpable stream: the gauge field.** `Field.A0`/`A` are new arrays
  worth writing for `exp_03` and visualisation — a new manifest entry, not a
  change to existing files.
- **Record the `A_mid` *direction*, not just the rule** (PM reverse-handoff,
  2026-06-19). The gauge manifest entry must state the midpoint **direction**
  — `"backward / gather-side"` (`½(A(x)+A(x−v))`), not merely `"half-sum"` —
  so a sign bug is traceable from the artefact alone. This is the durable
  trail for the `acceptance-test-4-confirms-backward-midpoint-sign` flag.
- **Identity extension.** Many sessions/kinds + M fields ⇒ the manifest's
  `arrays[]` gains a per-array `source` tag (session index/kind, or field
  key) + a filename convention; bump `schema_version`. Per-array axis/dtype
  unchanged.
- **Frozen rule (§0):** `N_RGB`/`N_CMY` semantics never change; new physics →
  new arrays. This is what guarantees existing manifests/viewer stay valid.

---

## 5. Decisions ledger

**Resolved:**
- Engine sees `AbstractSession` + `Field` only; no type branching.
- Two virtuals: `field_specs()` (which, high) / `deposit_source()` (how much,
  low). Taxonomy shape deferred to override level.
- `Field` is a lattice-like shared instance, one per kind, mutable, carrying
  a lattice ref.
- Physics is isolated and opt-in: no field ⇒ v0.1.0 bit-for-bit.
- Jacobi update ordering is the recommendation (D1b).
- `N_RGB`/`N_CMY` semantics frozen; new physics in new arrays.
- **D1d (2026-06-19): v0.3.0 ships minimal threading** — a static
  `vector_potential`/`external_potential` on `TickScheduler` → `hop.step`,
  NO `Field`/seam objects. exp_03 is single-session + static background; the
  whole interaction-engine architecture (§1–§2) stays forward (≈ v1.0). The
  static array is the seam's zero-source degenerate case, so no rework.

**Open:**
- **Particle taxonomy** — concrete inheritance shape (not flat); whether
  `DiscreteCausalSession` IS the abstract base or a concrete base under it;
  `omega`(mass)/charge as ctor params vs subclass identity. (≈ v1.0.)
- **71 kinds** (`dcl-generator-zoo`) — small parameterised hierarchy +
  registry vs subclass-per-kind. Lean parameterised.
- **D1b / D1c** — ordering and self-field are recommendations, not locked.
- **D1d — v0.3.0 scope** — ship the seam + static `Field` (degenerate, fixed
  `A0`/`A`, no members) for `exp_03`; defer the session-sourced provider to
  the inter-session-coupling release.
- **R4 readout** — experiment-side vs in-engine `chi(n̂)` (open in `04_*.md`).
- **Field creation** — up-front (like the lattice) vs lazy-on-first-coupling
  factory. Minor.

**Forward (not v0.3.0, but the design must not preclude):**
- **Intrinsic particle diameter** (user, 2026-06-18). **Primary motivation
  = Hilbert's 6th problem** (the axiomatisation of physics): both of its
  arms are live here — (i) the **axiomatisation of probability** (A=1 integer
  tokens ARE a discrete probability axiom), and (ii) the **atomistic →
  continuum limiting process** (the `n_units → ∞`, `a → 0` double limit that
  recovers continuous field dynamics; tested in `test_continuum_limit.py`
  and the cross-validation convergence). The diameter is the **finite-size
  control scale of that limit** — cf. the **Boltzmann–Grad** scaling, where
  particle size `d` → 0 with `N·d²` (3D) held fixed sets the collision
  cross-section that makes the discrete→continuum passage well-posed. So
  diameter governs the very derivation the framework is trying to make
  rigorous; it is **not** mere particle realism. This is **foundational**,
  not engine-local → flows upstream to `external/research` (the A=1
  formalisation effort).
  - *OPEN (load-bearing): how does the diameter scale against `a` and
    `n_units` in the double limit?* Is there a Boltzmann–Grad-analogue
    invariant held fixed (some `n_units · (D/a)^k = const`), or does it enter
    differently in the A=1 setting? To confirm with the user / upstream.
  - *Secondary (derived) engine role:* a low-level **species parameter**
    (units of `a`; alongside mass `omega`, charge) that shapes
    **`deposit_source`** (finite-size form factor over `R/a` sites) and
    **regulates D1c** (finite size softens the self-energy ⇒ "feels its own
    field, smeared over its diameter" — well-posed self-field). Distinct from
    the *emergent* wavefunction spread (`wavepacket(sigma)`), which already
    exists. First bites on **composite** particles (proton charge radius) —
    arrives with the `C^3` colour-memory coupling. Frozen contracts untouched
    (additive).

---

## 6. Limitations ledger (honest "can't do")

The interaction engine, stated honestly, **cannot**:

- **create or destroy particles** — sessions couple via shared fields
  (forces/phases) but exchange **no tokens**; emission, absorption, and pair
  production are not modelled.
- **propagate the field** — interactions are **instantaneous**, non-retarded
  (a per-tick background, not a dynamical photon). (`04_*.md` §5.)
- **do non-abelian forces** — U(1) only; no SU(2)/SU(3) (no real colour
  dynamics — the deferred `C^3` colour-memory coupling to Paper II).
- **claim a natively unitary hop** — one chirality moves per tick; unitarity
  is *patched* by the renorm, not fundamental.
- **resolve sub-token phase** — `TokenResidual` carry is real; the
  complex-carry hypothesis is parked (see memory + `remainder.py` TODO).
- **pin the birefringence order at arbitrary precision on CPU** — N-limited
  until the GPU Peierls path lands.

---

## 7. References

- Requirements: `docs/design/04_gauge_field_and_vacuum_response.md`.
- Naming: `docs/design/03_naming_convention.md` (two-frame rule).
- Class diagram: `notes/figures/core3d-classes.md`.
- Implementation plan + live decision scratchpad:
  `notes/gauge_field_v030_plan.md`.
- Data contract: `docs/data_deposit_and_provenance.md`.
