# 04 — Gauge field, Peierls coupling, and the induced photon-response order (core3d)

> Filename retains `..._vacuum_response` for stable cross-references; the
> "vacuum-averaged → isotropy" framing it originally carried was **retracted
> 2026-06-16** (see §1.1) — the target is the induced birefringence *order*,
> not an isotropy test.

**Status:** REQUIREMENTS (proposed 2026-06-16, revised 2026-06-16).
Target release: **v0.3.0**.
**Engine:** `core3d` (canonical). `core` is **legacy-only** — kept to
reproduce existing experiments, not extended for new physics (see §1.1).
**Driver:** Paper IV (optical-axis birefringence), gauge-sector channel —
`exp_03`. **Upstream physics:** Paper I App. B (induced gauge action, the
`Q`-tensor with eigenvalues `{4,4,16}`); Paper II (gauge coupling ratio).

---

## 1. Why this is needed

Paper IV must determine the **fate of the gauge-sector** birefringence (the
induced photon kinetic anisotropy `Q`). The anisotropy is genuinely
uniaxial about `(1,1,-1)` (the bipartite structure breaks `O_h`; §1.1), so
the open question is its **order**: an `O(1)` dimension-4 effect (excluded
by astrophysical polarimetry → a framework tension), a `(ka)^2`-suppressed
dim-6-like effect (viable, like the matter sector), or null (electric /
magnetic cancellation). The decisive test is the gauge analogue of
`exp_01`: impose a background gauge field (**both** electric and magnetic)
and measure the lattice's induced response vs orientation relative to
`(1,1,-1)`, reading off the leading order in `(ka)`.

This requires a capability core3d does **not** currently have: a **U(1)
gauge field coupled to the hop via the Peierls substitution**. The hop
docstring (`core3d/hop.py`) already names this as the intended mechanism
(`exp(i A·v)` per hop) but `_hop_average` does plain periodic shifts — the
coupling is documented, not implemented. `BipartiteLattice` carries no
gauge field.

### 1.1 Why core3d, not core

> **REVISED 2026-06-16.** An earlier version of this section claimed the
> token vacuum-average yields the `O_h`-symmetric (isotropic) response and
> that `exp_03` would show "vacuum isotropic vs single-probe anisotropic."
> **That premise is wrong:** the bipartite RGB/CMY structure breaks `O_h`
> to a uniaxial subgroup fixing `(1,1,-1)` (`Q` fixed by 12/48 of `O_h`).
> The lattice has *one* fixed bipartite orientation, so the token ensemble
> is uniaxial too — averaging does **not** restore `O_h`. The target below
> is the **order/magnitude** of the (uniaxial) induced birefringence, not
> an isotropy test.

core3d is the canonical engine and the right place for this:

- **core (continuous amplitude):** a single complex spinor field — a single
  configuration. Legacy-only; not extended.
- **core3d (integer tokens):** exact integer-token A=1 accounting; the
  token ensemble is the proper statistical A=1 vacuum on the (fixed,
  uniaxial) lattice. This is the canonical engine for new physics.

What `exp_03` must produce is the **order of the induced photon
birefringence** — is it `O(1)` at dimension-4 (the regime excluded by
astrophysical polarimetry → a framework tension), or `(ka)^2`-suppressed
(a viable dim-6-like effect, like the matter sector), or null? That
verdict — not an isotropy claim — is the result.

### 1.2 A=1 grounding (constraints on any observable)

All quantities are **A=1 accounting** quantities. There is no fundamental
time, energy, or mass; with `c = ħ = 1` the only dimensionful input is the
minimum distance `a`. Therefore:

- The induced-response observable **must be probability-conserving**
  (A=1-exact in the integer-token sense) and **dimensionless** — a property
  of the token bookkeeping, not an energy. Calibration to physical units
  (via `a`, `c`) happens downstream, outside the engine.
- The headline result is a **dimensionless anisotropy vs `(k a)`**, not an
  energy scale.

---

## 2. Functional requirements

### R1 — Gauge field as a `step()` parameter
Thread a vector potential through `HopOperator.step`, mirroring the
existing `external_potential` argument (keeps `BipartiteLattice` a frozen
geometry object — no state added to the lattice).

```python
def step(self, session, parity,
         external_potential: np.ndarray | None = None,
         vector_potential:   np.ndarray | None = None,   # NEW, shape (3, *lattice.shape)
         ) -> tuple[np.ndarray, np.ndarray]: ...
```

- `external_potential` continues to feed the **on-site** `delta_phi`
  (temporal/mass phase) only.
- `vector_potential` feeds the **spatial** hop phases (R2). `None` ⇒
  current behaviour bit-for-bit.

### R2 — Peierls phase in the hop
In `_hop_average`, the shift along basis vector `v` acquires the link phase
`exp(i · A_mid · v)`, with the **mid-point convention**
`A_mid = ½ (A(x) + A(x+v))` to match Paper I's symmetrised Peierls form
(cancels the order-`a^3` corner artifact, Paper I App. B).

- Must hold for both `cpu` and `gpu` backends (one phase multiply per shift
  in the RawKernel).
- `A = 0` ⇒ identical output to the current hop (regression-tested).

### R3 — Uniform-B parameterization
A helper that builds a constant magnetic field `B` of arbitrary
orientation in the **symmetric gauge**:

```python
def uniform_B_potential(shape, B_vec, origin=None) -> np.ndarray:
    """A(r) = ½ B × (r - origin); returns array of shape (3, *shape)."""
```

Needed to orient `B` along `(1,1,-1)`, in the perpendicular plane, and at
oblique angles. (core's `set_em_twist` is localized and Cartesian-axis-only
— insufficient.)

### R4 — Induced-response readout (BOTH electric and magnetic)
A way to extract the **second-order induced response** of the token vacuum
to a background field, at orientation `n̂` — the A=1 susceptibility whose
directional dependence carries the birefringence. Concretely, expose enough
state that an experiment can compute, against the zero-field baseline,
`chi(n̂) = [ induced A=1 token redistribution ] / |field|^2` in the
small-field limit, **token-exact** (the integer-residual machinery stays
A=1-conserving with the Peierls phase present).

- **CRITICAL (2026-06-16): both sectors.** The magnetic response (the
  `Q`-tensor, from spatial `A`) alone is *not* sufficient — the photon
  dispersion needs the **electric** sector too, and the leading
  birefringence may cancel between them. The engine must support a
  background that produces an **electric** field (a temporal/tick-direction
  gauge phase) as well as the magnetic (spatial `A`). `exp_03`'s verdict
  (`O(1)` dim-4 vs `(ka)^2`-suppressed) lives in the **E+B interplay**;
  magnetic-only would re-derive Paper I's incomplete result.
- The susceptibility estimator itself lives in the experiment; the engine
  guarantees A=1-exact evolution under the (spatial **and** temporal) gauge
  field. *If a canonical in-engine readout is preferred, specify it here —
  open definitional choice.*

### R5 — Orientation sweep at scale (parallelism / GPU)
`exp_03` sweeps dozens of field orientations (relative to `(1,1,-1)`) ×
large `N` (to resolve the birefringence order) × lattices up to `129^3`.

- Orientations are **independent** ⇒ embarrassingly parallel; the runner
  must be safe to fan out across processes/devices.
- The Peierls hop must be implemented in the **GPU RawKernel** path, not
  only CPU, so large-`N`/large-lattice runs are tractable. The precision to
  which the birefringence **order** (`O(1)` vs `(ka)^2`-suppressed) can be
  pinned is `N`-limited, so GPU directly sets the precision of the central
  result.

---

## 3. Acceptance criteria (tests)

`tests/test_peierls.py` (new):

1. **Zero-field regression.** `vector_potential = None` (and all-zeros)
   reproduces the current hop output bit-for-bit.
2. **Gauge covariance.** A pure-gauge potential `A = ∇Λ` (lattice gradient
   of an arbitrary scalar `Λ`) leaves all **token densities** `|ψ|^2`
   invariant tick-for-tick (only phases change). This is the U(1) Ward
   identity at the lattice level and is the core correctness check.
3. **Continuum B→0.** Induced response → 0 as `|B| → 0`, quadratically.
4. **Magnetic-sector cross-check reproduces `Q`.** The magnetic-only
   induced response at oriented angles reproduces the Paper I `Q`-tensor
   eigenstructure `{4,4,16}`, axis `(1,1,-1)` (vs `induced_gauge_action.py`).
   Confirms the Peierls coupling is wired correctly. (Note: this is uniaxial
   — `O_h` is *not* restored by the token ensemble; see §1.1.)
5. **Birefringence-order verdict (the load-bearing physics test).** With the
   **full E+B** response, classify the leading photon birefringence on the
   fixed bipartite lattice as a function of `(ka)`: `O(1)` at dimension-4
   (→ excluded → framework tension), `(ka)^2`-suppressed (→ viable
   dim-6-like prediction), or null (E/B cancellation). Report the order with
   an `N`-limited error bar. *Not an isotropy test — the lattice is
   uniaxial; the question is the magnitude/scaling.*
6. **A=1 exactness under gauge field.** Total token count is conserved
   exactly every tick with the Peierls phase present (integer-residual
   machinery unaffected).

---

## 4. Non-functional requirements

- **A=1 exactness** preserved with the gauge field present (R2 must not
  break the `TokenResidual` accounting; note the existing complex-carry
  `TODO` in `hop.py` — the Peierls phase is carried on `ψ_new` *before*
  the `|ψ|^2` reduction, so it is compatible with the current real-target
  residual; document the interaction).
- **Backend parity:** CPU and GPU produce identical results to floating
  tolerance.
- **No API break:** all new arguments default to `None`/off; existing
  experiments and `core` are untouched.
- **Determinism:** seeded token dynamics reproducible across runs (needed
  for the orientation sweep to be comparable).

## 5. Out of scope (explicitly deferred)

- **Dynamical gauge propagation.** The gauge field here is a *background*
  (the induced action is the matter-vacuum response to it). A dynamical
  photon field (link variables with their own evolution) is not required
  for `exp_03` and is not in scope.
- **The `1/g^2` prefactor.** Paper I/II defer the explicit
  `-Tr ln D_lat[U]` one-loop coupling; `exp_03` measures the *tensor
  structure* (anisotropy vs isotropy), not the absolute coupling. Out of
  scope.
- **Non-abelian gauge fields** (`SU(2)`, `SU(3)`). U(1) only.

## 6. Dependencies and version

- Target: **core3d v0.3.0** (new feature, additive, no break).
- Bumps `_version.py`, `CITATION.cff`, `release_notes/v0.3.0.md`.
- Consumers: Paper IV `exp_03`; pins `dcl_core >= 0.3.0`.
- **External (infrastructure) dependency — GPU + parallelism.** R5 assumes
  a GPU runtime (CuPy/RawKernel path) and multi-process/-device fan-out are
  **provisioned** (offered 2026-06-16). This is a hardware/runtime
  dependency, distinct from the R2/R5 code requirement to implement the
  GPU Peierls path: the code can land CPU-first, but the **precision of the
  central birefringence-order result (acceptance test #5) scales with the
  available `N`**, which is GPU-bound. Track provisioning alongside this doc.

## 7. References

- `core3d/hop.py` — `_hop_average` (Peierls documented, not implemented);
  `step()` (the `external_potential` threading pattern to mirror).
- `core3d/lattice.py` — `BipartiteLattice` (frozen geometry; keep stateless).
- core `OctahedralLattice.set_em_twist` / `vector_potential` — prior art
  for the gauge field and Peierls coupling (legacy engine).
- Paper I `paper/sections/induced_gauge_action.tex` — the `Q`-tensor
  `{4,4,16}`, `O_h` averaging to Maxwell, optical axis `(1,1,-1)`.
- Paper IV `notes/gauge_sector_structural_conclusion.md` — the conclusion
  this test is designed to confirm/refute in A=1 dynamics.
