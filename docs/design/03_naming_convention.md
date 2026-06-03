# Design: naming convention (two-frame, lattice-intrinsic first)

## Scope

This convention governs **`dcl_core.core3d`** (the integer-token engine)
**going forward**. `dcl_core.core` is Paper~I's verbatim port and is
**frozen for backward-compatibility** -- do not rename its symbols to
match this convention.

## The rule

Every non-trivial variable, function, and class lives in **two frames**,
named with a clear priority:

1. **The name says what the object IS in the lattice's own mathematics**
   -- the grid's intrinsic algebra, geometry, and equations, explored
   independently of existing physics. This is the *primary* frame.
2. **A `# physics:` comment says what it corresponds to in existing
   physics** -- *when* a correspondence exists. This is the *secondary*
   frame. Use "IS" for an exact correspondence, "approximates" for a
   continuum limit.

The grid is the thing we are studying; physics is the thing we compare
*to*. So the code reads as lattice mathematics first, with the physics
analogy annotated alongside -- never the other way round.

```python
# Lattice frame in the name; physics frame in the comment.
N_RGB: np.ndarray   # token count on the RGB sublattice component.
                    # physics: |psi_R|^2 * n_units (right-chiral density).
```

## Why two frames

`core3d` is where we test the *lattice's own* capabilities. Naming
purely after physics (`psi`, `gamma`, `momentum`) hides which structures
are intrinsic to the grid and which are borrowed analogies. Naming
purely after the lattice and dropping the physics loses the bridge that
makes the work interpretable. Keeping **both, with the lattice frame
load-bearing in the name**, lets a reader see the grid mathematics *and*
its physical reading at once.

## Alignment with `dcl-mathematics`

The lattice's intrinsic vocabulary is the one established (in LaTeX) by
the `dcl-mathematics` repo (Paper~IV, the diamond-progression
foundations). Mirror those formal symbols in code as closely as Python
allows. The canonical mapping:

| Lattice frame (the name) | Formal symbol | Physics frame (the `# physics:` comment) |
|---|---|---|
| diamond progression / `BipartiteLattice` | `Lambda_d` (here `Lambda_3`) | (3+1) Minkowski substrate; the Dirac structure |
| parity classes (even/odd) / `parity_field` | `V_d^+` / `V_d^-` | `gamma_5` grading -> chirality |
| `RGB_VECTORS` / `CMY_VECTORS`, `neighbour_offsets` | `E_d` simplex-edge generators | gamma-matrix directions |
| coordination | `coord(d) = 2d` | neighbour count / degree |
| `HopOperator`, hop along the active sublattice | (the lattice hop) | bipartite Dirac kinetic term |
| `TickScheduler`, parity alternation | joint-tick rule | discrete time-step; Zitterbewegung |
| token fields `N_RGB`, `N_CMY` (`int64`) | `N` on `V_d^+/V_d^-` | `|psi_R|^2`, `|psi_L|^2` (Born-rule density) |
| per-site phases `phi_RGB`, `phi_CMY` | (lattice U(1) phase) | phase of the Dirac spinor `psi_{R,L}` |
| A=1 token sum `sum N == n_units` | `A = 1` unity constraint | probability conservation |
| `epsilon_P = 1 / n_units` (alias `dp_min`) | `delta p_min` | "Planck of probability" |
| `TokenResidual.carry` | sub-token fractional accumulator | virtual / off-shell amplitude |
| coordination / automorphism algebra / hyperoctahedral group | `a_d` (71-dim at d=3) / `B_d` | gauge symmetry SU(3)xSU(2)xU(1) at d=3 |

`d` is the lattice dimension; `core3d` is the `d = 3` instance. Reserve
italic `d` for the dimension; if a discrete exterior derivative ever
lands, write it `dd`/`d_op` to avoid the collision (matches
`dcl-mathematics`'s upright-`d` convention).

## The physics frame is *derived*, and named so

The integer token field `N_RGB`/`N_CMY` and phase `phi_RGB`/`phi_CMY`
are the **lattice state** (primary). The complex amplitude
`psi = sqrt(N / n_units) * exp(i phi)` is a **derived physics object**;
keep it named in the physics frame (`psi_R`, `psi_L`) precisely because
it IS the physics reading of the lattice state. The two coexisting --
`N_RGB` (lattice) producing `psi_R` (physics) -- is this convention
working as intended, not a violation of it.

## Protected names (do not rename)

Some lattice-vocabulary names are load-bearing and fixed by CLAUDE.md's
"What NOT to Change":

- **`RGB` / `CMY`** sublattice geometry (and `RGB_VECTORS` etc.). These
  ARE the framework's bipartite/Dirac structure. They read like colour
  (physics-ish) but are the established *lattice* primitives -- keep
  them, and annotate the physics (gamma directions / chirality) in a
  comment.
- The **A=1** name and the **math-analog naming convention itself**.

A name sounding physics-flavoured is not, by itself, grounds to rename
it: ask whether it is the framework's established *lattice* vocabulary.

## Concrete checks

- **Variables** are lattice objects: `N_RGB`, `phi_CMY`, `epsilon_P`,
  `parity_field`, `coordination` -- not `count`, `phase_arr`, `eps`,
  `state_a`, `degree`.
- **Functions** are lattice operators / named procedures: `hop_average`,
  `quantise`, `neighbour_offsets`, `enforce_unity` -- not `update_state`,
  `do_step`, `apply`.
- **Classes** are lattice entities: `BipartiteLattice`, `HopOperator`,
  `TokenResidual`, `TickScheduler` -- not `LatticeManager`, `Engine`,
  `Helper`. (`BresenhamResidual` was renamed to `TokenResidual`:
  "Bresenham" named the *algorithm*, not the lattice object; the old
  name survives as a deprecated alias.)
- **Every non-trivial physics line carries a `# physics:` annotation**
  naming the correspondence (IS / approximates). Cross-reference the
  design doc, notes file, or `dcl-mathematics` symbol where one exists.

```python
delta_phi = omega + V   # on-site phase mismatch (mass + potential).
                        # physics: the Dirac mass+potential term.

# Structure factor of the bipartite hop.
# physics: approximates i k . gamma in the continuum (small-k) limit.
S_k = (np.exp(1j * k_dot_v).sum(axis=0) / 3)
```

## Tooling

`.claude/agents/physics-naming-reviewer.md` reads diffs and flags (a)
names that have slipped into operational-role naming, (b) physics-frame
names where a lattice-intrinsic name is available, and (c) non-trivial
physics lines missing their `# physics:` annotation. Invoke it before
committing changes to `src/dcl_core/core3d/`.

## Exemptions

- **`dcl_core.core`** -- frozen Paper~I port; not subject to this
  convention.
- **Test helpers and fixtures** (`small_shape`, `rng_seed`) -- not
  lattice code.
- **Backend primitives** (`zeros`, `shift`, `sum_all`) -- array
  operations, not lattice objects; follow array-library convention.
- **Loop indices** (`i`, `j`, `k`) in short scopes.
- **Deprecated aliases** kept for backward-compat (`N_R` -> `N_RGB`) are
  exempt by definition; mark them `# deprecated alias`.

## Anti-patterns

- Generic state holders: `data`, `state`, `result`, `value`, `obj`.
- Operational verbs hiding the object: `current_position` -- is it `r`?
  `r_peak`? Pick the lattice/math name.
- Comments that restate the code (`i += 1  # increment i`). Either the
  line is self-explanatory, or the comment names the lattice object /
  physics correspondence.
- A lattice-frame name with **no** physics annotation where a clear
  correspondence exists -- the bridge is half the point.
