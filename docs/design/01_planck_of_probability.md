# Design: Planck of probability (integer-token A=1)

## Premise

In the v1.0 continuous-amplitude framework, A=1 means
`integral of |psi|^2 dV = 1`. The lattice realisation enforces this
by per-tick renormalisation (`enforce_unity_spinor`), which is a
band-aid: the bare hop operator is not exactly norm-preserving, and
the renormalisation step cancels any intended dissipation.

This package adopts a structurally different position: A=1 is an
**integer-counting law**.

## The invariant

`epsilon_P` is the smallest meaningful probability. A session has

    n_units = 1 / epsilon_P              (an integer)

probability tokens to distribute over the lattice. A=1 is

    sum_x (N_R(x) + N_L(x)) = n_units    (exact, integer equality)

with no floating-point tolerance.

## Consequences

- **A=1 is automatic.** No renormalisation step; the remainder rule
  (Bresenham residual) ensures the integer total is preserved by
  construction.
- **Session-to-session amplitude transfer is well-defined.** A photon
  emission moves one token from electron to photon -- the integer
  count of each session changes by exactly one (or more), with the
  total over all sessions preserved. The exp_15 / exp_19 v4 "drain
  cancelled by renormaliser" blocker dissolves.
- **`epsilon_P` is a UV regulator.** Modes that would contribute
  probability below `epsilon_P` round to zero. Vacuum-energy-style
  sums that diverge in continuum QFT are finite here, automatically.
- **Continuum recovery is N -> infinity.** The continuous-amplitude
  v1.0 framework is recovered as `n_units -> infinity` with
  `epsilon_P -> 0`. Existing v1.0 results are reproduced at large N.

## What this does NOT commit to

The implementation commits to integer-token A=1. It does **not** at
this stage commit to:

- **The residual being real vs. complex.** Real residual is the
  default starting point (Bresenham over `|psi|^2`); complex residual
  may be needed if interference experiments wash out under real-
  residual rounding. Decision deferred to `exp_03`-style runs.
- **Uncertainty being epsilon_P.** The hypothesis `Delta N * Delta phi
  >= epsilon_P` (number-phase uncertainty as structural feature) is
  attractive but parked. The platform is built to make it testable,
  not to assume it.
- **`epsilon_P` being universal.** Per-session `n_units` lets different
  particles in the same simulation have different budgets. Universal
  `epsilon_P` is the cleanest version; experiments will tell whether
  it suffices.
- **Phase discretisation.** Phase `phi(x)` is continuous in the
  current design. Whether `phi` also discretises to `Z_N` is a
  future question.

## Practical implication for code review

When reviewing a change:

- A=1 violations should appear as **integer inequality**, not as
  "drift away from 1 by epsilon". If you see a float-tolerance check
  on conservation, that's a regression toward v1.0 thinking.
- Renormalisation of `N_R`, `N_L` to enforce a sum target is a code
  smell. The right place to absorb fractional bits is the
  `BresenhamResidual`, not a post-hoc rescale.
- New code that introduces continuous-amplitude reasoning ("the
  amplitude at site x") without grounding in either tokens or phase
  has slipped back into v1.0 mode. Push it back to the integer-token
  side.
