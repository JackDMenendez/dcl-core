# Experiments index

One row per experiment. Edit when adding / renaming / removing.

| ID | Title | Status | Wall-clock (target hw) | Notes |
|---|---|---|---|---|
| `exp_00_hop_drift` | Pre-rounding drift diagnostic | STUB | < 1 min | Measures per-tick `|sum_pre - sum_post|` of the analytical hop, on the continuous-amplitude reference. Output guides choice of `n_units` (epsilon_P). |

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
