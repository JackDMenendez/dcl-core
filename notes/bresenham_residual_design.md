# Bresenham Residual: Real vs. Complex Carry

**Status:** DRAFT
**Purpose:** Capture the open hypothesis that `BresenhamResidual.carry`
should be a complex (virtual-particle) amplitude rather than a real
fractional accumulator, and the experiment that would force the choice.
**Cited by:** `src/dcl_core/core3d/remainder.py` (parked-hypothesis
docstring, lines 27-32), `CLAUDE.md` (CURRENT STATUS, step 3).

---

## Setup

The integer-token framework computes one tick as:

1. `HopOperator.step` produces a fractional target
   `N_target(x) = n_units * |psi_new(x)|^2` per site, per chirality.
2. `BresenhamResidual.quantise` converts that fractional field to
   integer counts `N_new(x)` while preserving `sum(N) == n_units`
   exactly.

The current (v0.1.0-dev) design point is **real carry**:
`carry(x) in [0, 1)` per site per chirality, deposited as `+1` when
the carry crosses 1. This is the analogue of Bresenham line-drawing's
fractional accumulator on an integer pixel grid.

Real carry has a known information-theoretic limitation: `|psi|^2`
squaring discards the phase of `psi`, so the carry cannot represent
interference between two amplitudes that happen to round-floor
identically.

## Argument

A **complex carry** -- `carry(x) in C`, with `|carry(x)|^2 < 1`
representing the fractional amplitude not yet expressed as a discrete
token -- preserves phase between ticks. The physical interpretation,
if it holds, is much stronger than "preserves interference fidelity":

> The residual IS a virtual-particle excitation. While the carry sits
> below the integer threshold, momentum and charge are carried by an
> off-shell mode that does not yet manifest as a discrete token. When
> the carry crosses threshold the virtual mode "decoheres" -- it
> materialises as a real (on-shell) integer token.

This is testable against Paper II's generator zoo: the 71-dim per-site
automorphism algebra splits into 20 generators identifiable with
Standard-Model degrees of freedom and 51 unaccounted-for "extras."
The hypothesis under test is that the 51 extras **act on the carry
field but not on `N(x)`** -- they generate virtual-particle exchanges
that do not change manifest token counts.

If true, this gives the framework a *natural* physical origin for
the 51 extra generators (rather than them being a
representation-theoretic accident) and makes the integer-token model
continuum-equivalent at every order: real tokens carry the on-shell
content, complex carries carry the off-shell content, their sum
reproduces v1.0-style continuum dynamics exactly.

## Design consequences (must commit before implementing)

1. **Deposit rule.**
   - Real: `if carry >= 1: deposit +1, carry -= 1`.
   - Complex: choose between `|carry|^2 >= 1`, `Re(carry) >= 1`, or a
     phase-coherence-with-`N(x)` rule.
   - The deposit rule IS the virtual-particle decay law -- different
     choices give different virtual-mode lifetimes.

2. **Conservation invariant.**
   - Real: `sum N(x) == n_units` is the whole story.
   - Complex: the natural companion invariant is
     `sum N(x) + sum |carry(x)|^2 == n_units`, a two-pool ledger
     (manifest + virtual). The manifest pool alone is no longer
     conserved tick-to-tick; only the total is.

3. **Algebra link to the 51 extras.** Needs a derivation showing that
   the 51-dim subspace of the per-site automorphism algebra acts on
   `carry(x)` and leaves `N(x)` invariant. This is the bridge between
   "the carry is virtual" and "the math forces complex carry." Until
   the derivation exists, the connection is a working analogy, not a
   theorem.

## Where the determination gets made (and by whom)

**Not here.**  The decision will be made by a future maintainer
working in a downstream consumer, not inside `dcl_core` itself.

**Paper III's broader requirement** is a **comparison of discrete
versus continuous probability**: `dcl_core.core3d` (integer tokens)
run against `dcl_core.core` (Paper~I's continuous-amplitude engine,
ported verbatim) on matched initial conditions.  The
**minimum-momentum-uncertainty (min-Δp) experiments** are one slice
of that comparison: the regime where the carry's sub-token phase
information matters most.

Other comparison axes expected during the same study (some live in
`tests/test_cross_validation.py`, others downstream):

- **Arnold tongues** -- frequency-locking regions in a 2-D scan of
  `(omega_driver / omega_intrinsic, coupling_strength)`.  The
  continuous engine produces a known tongue structure; whether the
  discrete engine reproduces the same widths and rational-ratio
  locations is a structural test of continuum-limit equivalence,
  and *phase-resolution sensitive* -- a real-carry residual may
  shift / narrow tongues that a complex-carry residual would
  preserve.
- Free-particle wavepacket convergence as `n_units -> infinity`
  (already stubbed: `tests/test_cross_validation.py::test_free_propagation_matches_in_large_N_limit`).
- Two-body Coulomb orbit settling to the Bohr radius in both engines
  (already stubbed: `test_two_body_orbit_locks_in_both_cores`).
- (Add others as Paper III's design crystallises.)

The two engines coexist in one repo and one runtime, so the
comparison runs side-by-side with no external dependency.  This is
expected to live in **Paper III** (`external/dcl-paper-03-tidal-ionization`)
or in its own series.

When that work reaches a verdict, it triggers a `dcl_core` minor /
major bump: the residual implementation here switches from real to
complex carry, the deposit rule is updated, and the two-pool
conservation invariant becomes load-bearing.

## Predictions to discriminate

Whatever the specific experiment shape ends up being, the predictions
are:

- **Real-carry:** taking `|psi|^2` discards phase; resolution-limit
  observables (interference fringes, momentum correlations, fine
  momentum spread) wash out at the edge.
- **Complex-carry:** phase preserved in the carry until threshold;
  the resolution-limit signal survives.

Paper I's continuum engine remains the ground truth -- whichever
residual model reproduces its behaviour at the edge wins.

## Decision status

- **v0.1.0:** ships with **real carry** as a known-incomplete
  placeholder. The interfaces in `remainder.py` are written so the
  residual implementation is replaceable; calling code never inspects
  the carry's dtype directly.
- **v0.2.0+ (target):** when min-Δp work (or equivalent) shows phase
  loss, replace with complex carry and the corresponding deposit
  rule. The choice is then *forced* by downstream data, not chosen
  here by aesthetics.

## Open questions

1. What is the right deposit rule for a complex carry?
   `|carry|^2 >= 1` is the simplest; does it preserve unitarity at
   finite `N`?
2. Does the two-pool conservation invariant interact correctly with
   `TickScheduler.step`'s pairwise updates (multi-session
   interactions)?
3. Is there a closed-form derivation of the 51-dim subspace of the
   per-site automorphism algebra that commutes with the `N(x)`
   update? If so, that subspace IS what `carry(x)` carries.
4. Does complex carry require complex `N_target` too, or does it
   only matter at the rounding step? The hop already produces a
   complex `psi`; `quantise` is what currently destroys the phase.

## Pointers

- Source: `src/dcl_core/core3d/remainder.py` (parked-hypothesis
  docstring, lines 27-32)
- Roadmap: `CLAUDE.md` (CURRENT STATUS, step 3 -- "carry is real or
  complex" parking lot)
- Related design doc (planned): `docs/design/02_remainder_strategy.md`
- External: `external/dcl-paper-03-tidal-ionization` -- Paper III,
  the expected locus of the min-Δp experiments that force the
  carry-dtype decision
- External: `external/dcl-generator-zoo` -- the 71-dim algebra
  catalogue and the 20+51 split
- External: `external/dcl` -- Paper I's v1.0 continuum engine, used
  as ground truth for whatever resolution-limit experiment runs
