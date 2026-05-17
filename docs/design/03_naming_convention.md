# Design: naming convention (mathematical-analog naming)

## Rule

Every non-trivial variable, function, and class is named after the
**mathematical object it IS**, not the operational role it plays in
the program.

## Why

The framework's logic is dense. Code that says "what it does in the
program" forces the reader to mentally translate every line back to
the physics:

```python
# Bad: operational naming
mat0, mat1 = self._update_state(arr, val)
```

Code that says "what it is in the theory" lets the reader read it as
the theory:

```python
# Good: mathematical-analog naming
psi_R_new, psi_L_new = self.hop.step(session, parity="even")
```

The second form is shorter, more accurate, and easier to verify
against the paper / design docs.

## Concrete checks

When adding or reviewing code:

- **Variable names** are mathematical objects, not their roles.
  `gamma_0`, `delta_phi`, `N_units`, `psi_R`, `epsilon_P`,
  `R1`, `omega_e` -- not `phase_shift`, `count`, `eps`, `state_a`,
  `radius`, `freq`.
- **Function names** are mathematical operators or named procedures.
  `hop_RGB`, `enforce_unity_spinor`, `quantise_via_bresenham` --
  not `update_state`, `do_step`, `apply_constraint`.
- **Class names** are mathematical entities. `BipartiteLattice`,
  `DiscreteCausalSession`, `HopOperator` -- not `LatticeManager`,
  `SimulationContext`, `Engine`.

## Comment / docstring convention

Every non-trivial physics line carries a one-line annotation naming
the object it IS:

```python
# delta_phi = omega + V(x):  on-site phase mismatch (mass + potential).
delta_phi = omega + V

# This IS the structure factor of the bipartite hop.
S_k = (1 - np.exp(1j * k_dot_v_RGB).sum(axis=0) / 3)
```

Cross-reference the design doc, reference doc, or notes file where
one exists. Use "IS" for exact correspondences; "approximates" for
continuum limits.

## Tooling

The `.claude/agents/physics-naming-reviewer.md` agent reads diffs
and flags variables that have slipped into operational-role naming.
Invoke it before committing changes to `src/dcl_core/`.

## Exemptions

- **Test helpers and fixtures** can use operational names
  (`small_shape`, `rng_seed`). They're not physics code.
- **Backend primitives** (`zeros`, `shift`, `sum_all`) describe array
  operations, not physical objects, and follow array-library
  convention.
- **Loop indices** (`i`, `j`, `k`) are fine for short scopes.

## Anti-patterns

- Generic state holders: `data`, `state`, `result`, `value`, `obj`.
  If a variable's name doesn't tell you what *kind* of state it
  holds, rename it.
- Verb-prefixed nouns that hide the object: `current_position` -- is
  it `r`? `x_CoM`? `r_peak`? Pick the math name.
- Comments that re-state the code: `i += 1  # increment i`. Either
  the line is self-explanatory (no comment) or the comment names
  the physics ("# i indexes the tick number; advance to next tick").
