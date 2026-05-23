# Experiments index

One row per experiment. Edit when adding / renaming / removing.

| ID | Title | Status | Wall-clock (target hw) | Notes |
|---|---|---|---|---|
| `exp_00_hop_drift` | Pre-rounding drift diagnostic | STUB | < 1 min | Measures per-tick `|sum_pre - sum_post|` of the analytical hop, on the continuous-amplitude reference. Output guides choice of `n_units` (epsilon_P). |
| `exp_01_throughput` | Tick-throughput sweep across lattice shapes | PASS | ~5-15 min (Ryzen 2700) | Wall-clock per `TickScheduler.step` for shapes `8³` through `256³`. Dumps `.npy` + `.json` baseline; re-run on upgraded hardware to diff. Bump `N_MEASURE` by ~10× for overnight-quality runs. |
| `exp_02_phase_profile` | cProfile breakdown of one tick | PASS | < 1 min | Where the time goes WITHIN a tick (function-level, via cProfile). Companion to `exp_01`. Dumps raw `.prof` for `snakeviz` plus a text summary. Identifies memory-bandwidth vs per-element-compute vs Python-overhead bottlenecks. |

Add new rows below as experiments accumulate. Keep the table sorted
by ID.

## Status semantics

- **STUB** -- planned; `.py` may be empty.
- **PART** -- runs and demonstrates the mechanism; quantitative gap.
- **PASS** -- runs cleanly, confirms the claim.
- **FAIL** -- runs cleanly, disconfirms the claim (important; do
  not delete).

## Cross-references

If an experiment is downstream of a design doc or a test invariant,
add the cross-reference in the experiment's own `.md` file (not in
this table). The table is an index, not a knowledge graph.
