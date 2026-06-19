# core3d class design (Mermaid)

**Purpose:** a faithful class diagram of the *current* `dcl_core.core3d`
code (read 2026-06-18), plus a clearly-separated overlay of the proposed
D1 `FieldProvider` seam. Working artifact for the v0.3.0 gauge-field design
(`notes/gauge_field_v030_plan.md`). **Status:** DRAFT.

> Renders inline in VSCode (Markdown preview + a Mermaid extension) and on
> GitHub. Text-based on purpose — edit the source, diff in git.

---

## 1. Current code (faithful)

```mermaid
classDiagram
    direction LR

    class BipartiteLattice {
        <<frozen dataclass>>
        +tuple shape
        +str backend
        +n_sites() int
        +coordination() int
        +parity_field() ndarray
        +neighbour_offsets(parity) tuple
    }

    class DiscreteCausalSession {
        <<dataclass>>
        +BipartiteLattice lattice
        +int n_units
        +float omega
        +tuple momentum
        +ndarray N_RGB
        +ndarray N_CMY
        +ndarray phi_RGB
        +ndarray phi_CMY
        +epsilon_P() float
        +dp_min() float
        +delta_at(lattice, n_units, omega, position, momentum)$ DiscreteCausalSession
        +wavepacket(lattice, n_units, omega, center, sigma, momentum)$ DiscreteCausalSession
        +from_arrays(lattice, n_units, omega, N_RGB, N_CMY, phi_RGB, phi_CMY)$ DiscreteCausalSession
        +total_tokens() int
        +assert_unity() void
        +amplitude(component) ndarray
        +N_R / N_L / phi_R / phi_L
    }
    note for DiscreteCausalSession "N_R/N_L/phi_R/phi_L are deprecated\nphysics-frame aliases of N_RGB/N_CMY/phi_*"

    class HopOperator {
        <<dataclass>>
        +BipartiteLattice lattice
        +step(session, parity, external_potential) tuple
        -_hop_average(psi, vectors) ndarray
        +fourier_kernel(k) ndarray
    }

    class TokenResidual {
        <<dataclass>>
        +BipartiteLattice lattice
        +ndarray carry
        +quantise(N_target_RGB, N_target_CMY, n_units_total) tuple
        +drift_magnitude() float
    }
    note for TokenResidual "BresenhamResidual is a deprecated\nclass alias of TokenResidual"

    class TickScheduler {
        <<dataclass>>
        +BipartiteLattice lattice
        +HopOperator hop
        +list~DiscreteCausalSession~ sessions
        +int tick
        +dict~int,TokenResidual~ residuals
        +on_tick_complete callback
        +parity_now() parity
        +register(session) int
        +step() void
        +run(n_ticks) void
    }

    DiscreteCausalSession --> "1" BipartiteLattice : lattice
    HopOperator --> "1" BipartiteLattice : lattice
    TokenResidual --> "1" BipartiteLattice : lattice
    TickScheduler --> "1" BipartiteLattice : lattice
    TickScheduler --> "1" HopOperator : hop
    TickScheduler o-- "0..*" DiscreteCausalSession : sessions
    TickScheduler o-- "0..*" TokenResidual : residuals (1 per session)
    HopOperator ..> DiscreteCausalSession : reads amplitude() in step()
    TokenResidual ..> DiscreteCausalSession : quantises N_target (via scheduler)
```

**Not classes (module-level, shown for context):**
- `lattice.py` constants: `RGB_VECTORS`, `CMY_VECTORS`, `ALL_VECTORS`;
  type aliases `TickParity = "even"|"odd"`, `Component = "RGB"|"CMY"|"R"|"L"`.
- `backends/` — `get_backend(lattice.backend)` dispatches to the `cpu` or
  `gpu` module (functions: `shift`, `cos`, `sin`, `exp`, `sqrt`, `floor`,
  `zeros`, `indices`, `sum_all`, …). Every class reaches arrays through
  this, never directly.

**Per-tick flow** (`TickScheduler.step`): for each session →
`hop.step(session, parity)` → renormalise ψ → `residual.quantise(...)` →
write back `N_*`, `phi_*` → `assert_unity()`. *(Note: the scheduler does
not currently pass `external_potential` to `hop.step` — see D1.)*

---

## 2. PROPOSED overlay — abstract session + per-kind field singletons (NOT yet in code)

> REVISED 2026-06-18: inverted the seam. The engine (`TickScheduler`,
> `HopOperator`) depends on `AbstractSession` + `Field` only and branches
> on no concrete particle type. Type knowledge lives in the session
> hierarchy via two virtuals. Particle taxonomy is **OPEN** (not flat).

```mermaid
classDiagram
    direction LR

    class AbstractSession {
        <<abstract · abc.ABC>>
        +field_specs() list  «virtual: WHICH fields»
        +deposit_source(field) void  «virtual: HOW MUCH»
        +gauge_potential(parity) tuple
        +amplitude(component) ndarray
        +assert_unity() void
    }
    class DiscreteCausalSession {
        <<current mechanics>>
    }
    class ChargedSession {
        <<intermediate · coupling-group>>
        +field_specs()  «-> U(1) key»
    }
    class NeutralSession {
        <<intermediate>>
    }
    class ParticleSubtree {
        <<TBD — Electron / Photon / Proton …, NOT flat>>
        +deposit_source(field)  «charge / form factor»
    }

    class Field {
        <<abstract · one shared instance per kind, like the lattice>>
        +BipartiteLattice lattice
        +ndarray A0
        +ndarray A
        +add_member(session) void
        +recompute() void
        +potential_for(session, parity) tuple
    }
    class GaugeFieldU1

    class TickScheduler {
        +list~AbstractSession~ sessions
        +list~Field~ fields
        +register(session) int
        +step() void
    }
    class HopOperator {
        +step(session, parity, external_potential, vector_potential) tuple
    }

    AbstractSession <|.. DiscreteCausalSession
    DiscreteCausalSession <|-- ChargedSession
    DiscreteCausalSession <|-- NeutralSession
    ChargedSession <|-- ParticleSubtree
    Field <|-- GaugeFieldU1
    Field --> "1" BipartiteLattice : lattice (shared, like sessions)
    TickScheduler o-- "0..*" AbstractSession : sessions
    TickScheduler --> "0..*" Field : fields (shared instances, passed in)
    Field o-- "1..*" AbstractSession : members (like kind)
    AbstractSession ..> Field : field_specs() names which, at register()
    HopOperator ..> AbstractSession : reads amplitude(), gauge_potential()
```

**Ownership — "global like the lattice".** Fields are NOT a process/module
global: like `BipartiteLattice`, each is one shared instance the
experiment constructs and **passes by reference** (identity-shared per
simulation → no cross-test state bleed). The scheduler holds them
(`fields: list[Field]`) exactly as it holds its `lattice` ref. Unlike the
frozen lattice, a `Field` is **mutable** (recomputes `A0`/`A` each tick).

**Per-tick flow (scheduler branches on nothing):**
`register(session)` matches `session.field_specs()` to the passed-in
fields and calls `field.add_member(session)`. Then each `step()`:
`field.recompute()` for all fields (start-of-tick read ⇒ Jacobi, = D1b) →
per session `A0, A = session.gauge_potential(parity)` (pulls
`field.potential_for(self)`, self-source excluded = D1c) →
`hop.step(session, parity, A0, A)` → quantise → write back.

**Two virtuals at two depths** (the only structural commitment;
everything else in the taxonomy is open):
- `field_specs()` — *which* field — overridden HIGH (coupling-group, e.g.
  `ChargedSession`).
- `deposit_source(field)` — *how much* — overridden LOW (species).

The static `exp_03` background is then just a degenerate `Field` with a
fixed `A0`/`A` and no members. 71 candidate kinds (`dcl-generator-zoo`) ⇒
prefer a small hierarchy + parameter registry over 71 hand-written
classes. Open sub-decisions + rationale live in
`notes/gauge_field_v030_plan.md` (D1 refinement).
