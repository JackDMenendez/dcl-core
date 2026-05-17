# Design: remainder / residual strategy

## Problem

The hop operator analytically produces a fractional new amplitude
target `psi_new(x)`. The squared modulus times `n_units` gives a
fractional new token count `N_target(x)`. The fractional parts must
land somewhere -- A=1 must hold exactly.

## Options considered

### (1) Bresenham-style accumulating residual (CHOSEN)

Maintain `carry(x)` alongside `N(x)`. Each tick:

```
n_provisional = floor(N_target + carry)
carry         = (N_target + carry) - n_provisional   # in [0, 1)
```

After the per-site pass, correct global sum by ±1 on highest-residual
sites until `sum(N) == n_units`. Exact A=1, deterministic, the carry
field has clear physical interpretation: it IS the "pending
probability" that hasn't yet promoted to a token.

### (2) Top-k redistribution

Round each site to nearest integer; adjust ±1 on the |delta| highest-
error sites until sum matches. Simpler than (1) but **discards the
temporal averaging**: fractional bits that should accumulate across
ticks and eventually deposit a token are lost each tick.

### (3) Stochastic rounding

Round up with probability equal to fractional part. Conserves
expected value, not exact integer total. Requires a global
correction step and adds noise that has to be characterised.

### (4) Random-walk tokens (per-token simulation)

Each of `n_units` tokens hops independently with probability set by
cos^2 / sin^2 factors. Exact A=1 by construction. Physically clean,
but classical walkers do not interfere -- you would have to give
each walker a signed phase contribution and handle cancellations as
redirection to interference bright spots. Complex; deferred.

## Why (1)

- Conservation is exact and integer-typed.
- Carry field is interpretable (sub-token probability).
- Long-run dynamics get correct temporal averaging "for free"
  (Bresenham's deposit-on-overflow does the right thing).
- GPU-friendly: carry is one extra float field per chirality.
- Easy to test (`tests/test_remainder.py`).

## Open question: real or complex residual?

The default carry field is **real**: it's a residual on `|psi|^2`.

But `|psi|^2 = N(x) / n_units` has already squared the amplitude,
discarding the phase that interference needs. A residual carrying
only `|psi|^2` fragments may suppress interference fringes that the
continuous core resolves.

If the planned `exp_03_interference` experiment (port of v1.0's
exp_03) shows fringes washing out below the v1.0 contrast, the
residual is genuinely an amplitude, not a probability, and the
carry field has to be complex: `carry_amp_R(x)`, `carry_amp_L(x)`,
with the integer count derived as

    N(x) = round(|carry_amp + psi_token_contribution|^2 * n_units)

Whether real or complex residual is correct is **an empirical
question the platform must be able to answer**, not a foundational
commitment.

## Implementation notes

- `BresenhamResidual.carry` is part of the session's persistable state.
  Checkpointing must save/restore it alongside `N` and `phi`, or
  long-run trajectories will diverge from the original.
- The "highest-residual" global correction step should be
  deterministic (e.g. sort by residual, then by lattice index as
  tiebreaker) so two runs from the same seed produce bit-identical
  output.
- Performance: for an N^3 lattice, the global rebalance is O(N^3
  log N^3) if implemented naively with sort. For large lattices,
  partial sort / top-k (e.g. `np.argpartition`) is enough.
