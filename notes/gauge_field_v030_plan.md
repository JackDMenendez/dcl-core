# Gauge field + Peierls coupling — v0.3.0 implementation plan (working note)

**Purpose:** capture the implementation plan for the core3d U(1) gauge field / Peierls coupling (the v0.3.0 requirements in `docs/design/04_gauge_field_and_vacuum_response.md`) while the open design decisions are still being thought through. **Status:** DRAFT — decisions D1–D4 (§1) are UNRESOLVED; do not start coding until they settle. Promote to `docs/design/` once they do.

> This is a *working note*, not a settled design doc. The sequencing in §2 is conditional on the §1 decisions. Findings in §0 are facts read from the current code (2026-06-17) and are not in question.
>
> **Design rationale now lives in the LIVING design doc**
> `docs/design/05_interaction_engine.md` (architecture, physics-vs-architecture
> boundary, decisions ledger, limitations ledger). This note is the v0.3.0
> *implementation plan / scratchpad*; promote stabilised decisions up into 05.

------------------------------------------------------------------------

## 0. Findings from the current code (facts — not up for decision)

1.  **The scheduler threads no potential to the hop.** `TickScheduler.step` calls `self.hop.step(session, parity)` with no `external_potential` (`src/dcl_core/core3d/scheduler.py`, the hop call). The existing on-site potential only reaches the hop when an experiment calls `HopOperator.step` directly (as `tests/core3d/test_hop.py` does). → delivering a *background* gauge field is a plumbing decision (D1), not just a new kwarg.

2.  **The hop documents Peierls but does plain shifts.** `_hop_average` (`src/dcl_core/core3d/hop.py`) averages `backend.shift(psi, v)` with no link phase. R2 is a localized change here.

3.  **`core`'s Peierls convention ≠ R2's.** `core` adds `A·v` into the on-site rotation angle `delta_p` (`src/dcl_core/core/OctahedralLattice.py`); R2 mandates a complex link-phase multiply `exp(i·A_mid·v)` on the shifted amplitude. core3d follows **R2** (gauge-covariant by construction); `core` is reference-for-intent only, not a port source.

4.  **The write-back gives the Ward identity for free.** The scheduler writes `phi = angle(psi_new)`, `N = |psi_new|^2`. A pure-gauge `A = ∇Λ` changes only the hop *phase*, never `|psi|` (each link factor has unit modulus), so token densities are invariant tick-for-tick — exactly acceptance test #2. The Peierls phase rides on `psi_new` *before* the `|psi|^2` reduction, so it is compatible with the current **real** `TokenResidual` carry (NFR). No complex-carry work is pulled in.

5.  **The "electric" sector mostly already exists.** core3d's on-site `delta_phi = omega + V(x)` IS the temporal/tick-direction gauge phase `A_0`. R4's electric sector = feed `A_0` through the existing `external_potential` path (with a gauge-correct sign/convention); the **magnetic** sector is the new spatial `vector_potential`. "Both sectors" = reuse `external_potential` as `A_0` + add `vector_potential` as `A`. The novel work is the **E+B readout** (exp_03), not two new engine mechanisms. *Trap: the `A_0` sign/normalization must be pinned against Paper I App. B so tests #4/#5 aren't off by a convention.*

6.  **The GPU RawKernel is a stub.** `gpu.shift` uses CuPy roll; the RawKernel sources are "implement once the API is stable" (`src/dcl_core/core3d/backends/gpu.py`). The GPU Peierls path (R2/R5) is a separate, larger workstream. CPU-first is sanctioned; the headline result (test #5) is then `N`-bound by CPU until GPU lands.

------------------------------------------------------------------------

## 1. Open decisions (THINK ABOUT THESE — coding is blocked on them)

**D1 — How does the field reach the hop? (= the generic multi-session
coupling seam)** *(blocking)*

REFRAMED 2026-06-18 (user). A static `vector_potential` array is only the
**degenerate case** (exp_03's imposed external background). The physically
central case — multiple sessions resonating in a Coulomb field — has the
field **sourced by the sessions themselves and sampled at the instant of
each `hop.step`**: it is not a fixed gradient stored anywhere; it is a
function of the current token state of the (other) sessions at that tick
("whatever it happens to be at that hop.step"). So D1 is really *what is
the generic multi-session field-coupling seam*, and the static background
is the zero-source special case of it.

*Note on scope:* this is NOT the "dynamical gauge propagation" deferred by
requirements §5. An **instantaneous** Coulomb potential sourced by charge
density (no own evolution / retardation) is the **pairwise interaction the
scheduler already promises** (step 4 of its docstring), and the
inter-session coupling CLAUDE.md still lists as ahead — not a propagating
photon field.

**The seam (proposed):** a per-tick *field provider* the scheduler
evaluates, before each session's hop, instead of a stored static array:

```
FieldProvider.potentials(scheduler, session_idx, parity) -> (A0, A)
    # A0 -> external_potential (temporal/electric), A -> vector_potential (magnetic)
  • StaticBackgroundField  -> returns fixed arrays            (exp_03; ship in v0.3.0)
  • SessionSourcedField    -> A0/A from sessions' densities   (Coulomb/multi-session; later)
```

`hop.step` stays as in R1/R2 (it just receives `external_potential` +
`vector_potential`); the provider decides what those arrays *are* each
tick. `None` provider ⇒ current behaviour bit-for-bit.

**Sub-decisions the seam forces (think about these):**
- **D1b — update ordering.** Build "the field session B sees" from all
  sessions' state at the *start of the tick* (synchronous / Jacobi —
  every session sees one snapshot, order-independent) or from whatever is
  *already updated this tick* (sequential / Gauss-Seidel — order-
  dependent; the scheduler docstring already warns "order matters for
  pairwise-interaction determinism"). Physics choice, not just plumbing.
- **D1c — self-field.** Does a session see the field it sources itself, or
  only the *other* sessions' field? (Coulomb force between distinct
  charges excludes self; an induced-vacuum-response reading might not.)
- **D1d — v0.3.0 scope.** Ship the *seam* + the **static** provider
  exp_03 needs; defer the **session-sourced** provider to the inter-
  session-coupling release — but design the seam now so it drops in
  without re-plumbing.

**REFINEMENT 2026-06-18 (user) — invert the seam: polymorphic session,
abstract scheduler.** A `FieldProvider` *on the scheduler* makes the
scheduler understand sources per particle type. Invert it: the engine
(`TickScheduler`, `HopOperator`) depends on an **`AbstractSession`** only
and branches on no concrete type. The type knowledge lives in the session
hierarchy via two virtuals (Python: all methods are virtual; root is an
`abc.ABC`):

- `field_specs() -> list[FieldKey]` — *which* field(s) this session
  couples to. Overridden at the **coupling-group** level (e.g. an
  intermediate `ChargedSession` → the U(1) key). ("knows what field it
  modifies.")
- `deposit_source(field)` — *how much* this session contributes (charge
  sign/magnitude; a composite proton's form factor). Overridden at the
  **species** level. ("the more specific inheritor knows what factory.")

So *coupling is decided high in the hierarchy, magnitude low* — the only
structural commitment. **The particle taxonomy itself is OPEN** (not flat;
`Electron`/`Photon`/`Proton`/… positions TBD). With 71 candidate kinds
(the `dcl-generator-zoo` automorphism algebra), prefer a small behavioural
hierarchy **parameterized by a registry/factory** over 71 hand-written
classes — subclass only where *behaviour*, not just numbers, differs.

**`Field` is a shared instance modelled on the lattice (user, 2026-06-18:
"global like the lattice").** Key terminology: the lattice is NOT a
process-level/module global — it is a single instance the experiment
constructs and **passes by reference**, identity-shared
(`session.lattice is self.lattice`, enforced in `scheduler.register`).
The `Field` adopts the same lifecycle: a first-class object constructed
alongside the lattice (it carries a `BipartiteLattice` ref — it is
lattice-shaped) and handed to the sessions/scheduler. This gives the
"one shared instance" semantics AND keeps test isolation (each simulation
builds its own — the lattice pattern, not a module global, so no
cross-test state bleed).

Two differences from the lattice the design must respect:
- **Several fields, one lattice.** "Global like the lattice" = *one shared
  instance per kind* (EM, …); `field_specs()` is how a session names which
  shared field(s) it attaches to.
- **Mutable shared state vs frozen geometry.** The lattice is frozen; the
  `Field` recomputes `A0`/`A` every tick, so lifecycle and the start-of-
  tick read order (D1b) matter in a way they never did for the lattice.

Wiring: fields are constructed up front (like the lattice) and the
scheduler holds them (`fields: list[Field]`, analogous to its `lattice`
ref); `register(session)` matches `session.field_specs()` to the available
fields and calls `field.add_member(session)`. Per tick: `field.recompute()`
(all fields) → each session pulls `gauge_potential(parity)` from its
attached field(s) → `hop.step(session, parity, A0, A)`. Scheduler branches
on nothing. *(A factory still has a role only if fields should be created
lazily on first coupling instead of up front — open, minor.)*

This **dissolves the (a)-vs-(b) keying question** (per-interaction vs
per-species field): the override decides per class — a charged particle's
`field_specs()` returns the shared U(1) key (reading (a)); a future
per-species field is a different override (reading (b)). No global choice.
It also **relocates D1b/D1c into `Field`**: D1b = when `recompute()` reads
members (start-of-tick ⇒ Jacobi); D1c = `field.potential_for(session)`
excluding that session's own source — both invisible to the scheduler.

*Open (taxonomy):* the concrete particle inheritance shape; whether
`DiscreteCausalSession` IS the abstract base or a concrete base under it;
where `omega`(mass)/charge become constructor params vs subclass identity.
Targeted at the particle-class work (≈ v1.0); the v0.3.0 seam must not
preclude it.

*(Superseded framing, kept for history: "Option A = store a static array
on the scheduler; Option B = exp_03 drives hop.step directly." Option B is
still available for a pure-static exp_03, but it forecloses the provider
seam, so it is no longer preferred.)*

**D2 — R4 readout: in-engine vs experiment-side?** (the requirements doc leaves this explicitly open). *Leaning experiment-side:* the engine already exposes `amplitude()`/`N_*`/`phi_*`; it just guarantees A=1-exact evolution and adds no `chi(n̂)` method. Revisit only if exp_03 wants engine-internal state.

**D3 — Where do the E/B helpers live?** *Leaning:* new module `src/dcl_core/core3d/gauge.py` (`uniform_B_potential` + E/B helpers), re-exported from `core3d/__init__.py` (MINOR bump). Run `physics-naming-reviewer` on it.

**D4 — `A_mid` convention.** *Leaning:* mid-point `A_mid = ½(A(x)+A(x+v))` with periodic wrap via `backend.shift` so CPU/GPU agree.

------------------------------------------------------------------------

## 2. Work breakdown (CPU-first; conditional on §1)

**Phase 1 — CPU Peierls core (R1 + R2) + correctness tests** - R1: `vector_potential: np.ndarray | None = None` (shape `(3, *lattice.shape)`) on `HopOperator.step`; validate shape; `None` ⇒ unchanged. Pass into `_hop_average`. - R2: in `_hop_average`, per `v`, multiply shifted amplitude by `exp(i·A_mid·v)`, `A_mid = ½(A + shift(A, v))`, `A·v = Σ_c A[c]·v[c]`. - Tests (`tests/test_peierls.py`): #1 zero-field regression, **#2 Ward identity** (load-bearing), #6 A=1 exactness. (#3 stubbed until Phase 2.)

**Phase 2 — Field parameterization (R3) + scheduler plumbing (D1)** - R3: `gauge.py::uniform_B_potential(shape, B_vec, origin=None)`, symmetric gauge `A(r) = ½ B × (r − origin)`, shape `(3, *shape)`. - D1: optional `vector_potential` (+ forward `external_potential`) on `TickScheduler` → `hop.step`. - Electric-sector wiring: pin the `A_0` sign convention (§0.5); document. - Tests #3 (B→0 quadratic), #4 (magnetic-only reproduces Paper I `Q`-tensor `{4,4,16}`, axis `(1,1,-1)`).

**Phase 3 — The physics verdict (R4 readout, exp_03)** - `experiments/exp_03_*.py`: E+B susceptibility estimator (experiment-side per D2); orientation sweep vs `(1,1,-1)`. - Test #5 (birefringence-order verdict: `O(1)` dim-4 vs `(ka)^2`- suppressed vs null E/B cancellation), `N`-limited error bar. **Load-bearing physics test.**

**Phase 4 — Scale path (R5, GPU) + performance — gated on hardware provisioning** - Peierls phase in the GPU RawKernel (`gpu.py`); `test_gpu_matches_cpu`. - Orientation sweep safe to fan out across processes/devices (determinism NFR: seed token dynamics reproducibly).
  - **PERF — cache static link phases (the hot-loop win).** `_hop_average`
    recomputes the per-link phase `exp(i·A_dot_v_mid)` every tick: a
    full-lattice dot + shift + `exp` + complex multiply, ×3 vectors, per
    session per tick. For a **static** background `A` (exp_03's entire case)
    those phases are identical every tick → precompute them **once** per
    `(A, parity)` into 3 complex arrays and reuse, dropping the `exp`
    from the steady state to a single cached multiply. **Reentrancy
    constraint (lock-free):** build these **eagerly at construction,
    read-only thereafter** — NOT a lazily-filled mutable cache (the
    first-write race would force a lock; see `05_interaction_engine.md`
    §2 concurrency). A *dynamic* session-sourced field recomputes its
    phases each tick anyway, inside the single-threaded Jacobi recompute
    phase, so there is no shared-cache race there either. Deferred from
    Phase 1 deliberately (adds state; pairs naturally with the RawKernel).
    Bigger lever than any per-call micro-opt. *(The
    once-per-`step()` shape-guard tuple was already tidied to
    `(3, *lattice.shape)`; negligible but free.)*

Phases 1–3 land CPU-only and are independently valuable (they pin the *order* at whatever `N` CPU allows). Phase 4 raises `N` to tighten the test-#5 error bar; it carries the external GPU dependency.

------------------------------------------------------------------------

## 3. API / release impact

- New surface: `step(vector_potential=…)` kwarg (additive); `uniform_B_potential` re-export; optional `TickScheduler` field — all default off ⇒ **MINOR** bump (`0.2.x` → `0.3.0`), no break.
- Run `api-stability-reviewer` (confirms additive-only) and `physics-naming-reviewer` (`gauge.py` + new hop code) on the diff.
- Release: bump `_version.py` + `CITATION.cff`; `release_notes/v0.3.0.md`; Paper IV pins `dcl_core >= 0.3.0`.

## 3.5 Data (.npy) impact

Two claims, both **additive** — they keep the artefact contract in
`docs/data_deposit_and_provenance.md` intact.

**Frozen-density rule (the gauge sector touches phase, not the carried
observable).** The deposited volume's observable is
`(N_RGB + N_CMY) / n_units` (total token density `|psi|^2`). By §0.4 the
Peierls phase rides on `psi_new` *before* the `|psi|^2` reduction, so:
- a **pure-gauge** `A = ∇Λ` leaves the density invariant tick-for-tick
  (Ward identity, test #2) — same bytes, same schema;
- a **physical** magnetic `A` changes the density's *evolution* but not
  its *form*: still a real `(T,X,Y,Z)` array, still the frozen
  `N_RGB`/`N_CMY` contract, still `float32`-downcast-OK.

So `arrays[*].axis_order` / `dtype` / `observable` for the density volume
are **unchanged**. A gauge run produces the same `.npy` schema as a v0.2.x
run; it is the field *inputs* that are new, not the readout shape. This is
the array-contract half of the MINOR (`0.3.0`) bump.

New E/B / susceptibility readouts (exp_03, R4) are **new entries in the
`arrays` list** — each self-describing via its own `observable` string —
never a redefinition of the density array. Same bundle, additional files.

**Manifest extension (record the field background → bump
`schema_version`).** A gauge run is only reproducible if the manifest says
*which field was applied and how*. Add an optional, additive `run.fields`
block capturing the provider + convention (so D1/D3/D4 choices are pinned
in the provenance, not just the code):

```json
"run": {
  "...": "...existing seed/params/wall_clock...",
  "fields": [
    {
      "key": "U(1)",
      "class": "GaugeFieldU1",
      "mode": "static-background",   // degenerate Field, no members; vs "session-sourced" (later)
      "A0": {"source": "external_potential", "convention": "Paper I App. B"},
      "A":  {"helper": "uniform_B_potential", "B_vec": [0, 0, 1],
             "gauge": "symmetric", "origin": null, "A_mid": "half-sum"}
    }
  ]
}
```

- `class`/`mode` distinguish the degenerate static exp_03 background (a
  `GaugeFieldU1` with fixed `A0`/`A` and **no members**) from a future
  session-sourced (Coulomb) field with depositing members — the D1d scope
  line, made legible in the data, in the **inverted-seam vocabulary** of
  `notes/figures/core3d-classes.md` (not the superseded `*Provider` names).
- `A0.convention` pins the §0.5 sign trap; `A.gauge`/`A_mid` pin D4 — a
  reader can tell an off-by-convention rerun from a real one.
- **`None`/absent `run.fields` ⇒ a v0.2.x-style field-free run** — old
  manifests stay valid, so the block is back-compatible.

Because the block is additive and optional, bump the manifest
`schema_version` **`"1.0"` → `"1.1"`** (readers that ignore `run.fields`
keep working). Note this is the *manifest* schema_version in
`data_deposit_and_provenance.md`, independent of the `dcl_core` package
MINOR bump — but introduce them together so v0.3.0 is the first engine
whose manifests can be `1.1`.

------------------------------------------------------------------------

## 4. Touched files

| File | Change | Req |
|------------------------|------------------------|------------------------|
| `core3d/hop.py` | `vector_potential` kwarg; Peierls phase in `_hop_average` | R1, R2 |
| `core3d/gauge.py` *(new)* | `uniform_B_potential`, E/B helpers | R3 |
| `core3d/scheduler.py` | forward `vector_potential`/`external_potential` | R1, D1 |
| `core3d/backends/cpu.py` | (maybe) complex-exp helper | R2 |
| `core3d/backends/gpu.py` | Peierls in RawKernel | R2, R5 |
| `core3d/__init__.py` | re-export `uniform_B_potential` | R3 |
| `tests/test_peierls.py` *(new)* | acceptance #1–#6 | §3 |
| `experiments/exp_03_*.py` *(new)* | E+B sweep + verdict | R4, R5 |
| `_version.py`, `CITATION.cff`, `release_notes/v0.3.0.md` | release | §3 |