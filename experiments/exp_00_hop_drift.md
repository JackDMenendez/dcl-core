# exp_00_hop_drift -- Pre-rounding drift diagnostic

**Status:** STUB
**Wall-clock target:** under 1 minute
**Hardware:** any (no GPU required)

## What this measures

For each combination of `lattice.shape` and `n_units`, compute the
discrepancy between the analytical hop output's summed `|psi|^2` and
the session's `n_units` (the conserved integer total). This is the
amount of probability the `BresenhamResidual` step has to absorb each
tick.

## Why it matters

The integer-token framework relies on a clean separation:

- **Hop produces a fractional target.** Small drift = the lattice
  hop is approximately unitary in the way the derivation needs.
- **Residual rounds back to integer.** As long as drift sits between
  `epsilon_P` and `n_units`, the residual step has something to do
  but nothing pathological.
- **A=1 holds by construction.** No `enforce_unity_spinor` band-aid
  required; conservation is integer arithmetic, not a renormalisation.

If the measured drift is comparable to `n_units` itself, the hop is
structurally non-unitary in a way the framework should not pretend to
ignore. If drift is below `epsilon_P`, then `epsilon_P` is too small
to matter at this configuration and the discrete and continuous
frameworks are observationally identical.

## How to read the output

The script writes `data/exp_00_hop_drift.npy` (raw drift values) and
`data/exp_00_hop_drift.log` (human-readable summary). The summary
table reports, for each (shape, n_units):

| Column | What it is |
|---|---|
| `shape` | Lattice shape |
| `n_units` | Session's probability budget |
| `epsilon_P` | `1 / n_units` |
| `drift_per_tick` | Mean `|sum_analytical - n_units|` across 100 ticks |
| `drift / epsilon_P` | Ratio (interesting when between 1 and 10) |
| `drift / n_units` | Ratio (must be << 1 for the framework to be sensible) |

## Expected ranges

For a numerically unitary lattice hop, `drift_per_tick` should be at
floating-point precision (~ `1e-13 * n_units`). For the v1.0-style
hop with the cos/sin/i mixing as currently coded, drift is likely
much larger -- that's the point of measuring.

A specific result to look for: **is drift bounded above the float-
epsilon level by a structural amount?** If yes, that amount IS the
empirical `epsilon_P` the framework should target. If no, the hop is
unitary to machine precision and any `n_units >> 10^16` is wasted.

## Acceptance criteria

This is a diagnostic, not a falsifier -- it does not have a PASS /
FAIL contract. The script exits 0 if the experiment ran cleanly and
the log file is non-empty; 1 otherwise.

## Follow-ons

Once drift is characterised, candidate follow-on experiments:

- `exp_01_continuum_recovery` -- show that for `n_units` above the
  measured drift floor, the discrete and continuous cores produce
  indistinguishable hydrogen ground states.
- `exp_02_interference_fringes` -- the make-or-break test for the
  remainder rule (real vs. complex residual).
