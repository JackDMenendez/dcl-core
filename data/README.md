# data/

Data files produced and consumed by experiments.

## What goes here

- `*.npy` -- NumPy arrays produced by experiment scripts.
- `*.log` -- stdout captures from long-running experiments.
- `*.csv` -- tabular results that downstream figures read.

## What does NOT go here

- Source code (lives in `src/`).
- Figures (live downstream, e.g. in a paper repo that depends on this
  library).
- Build artefacts (live in `build/`, `dist/`).

## Tracking in git

`.npy` files are tracked when small enough that the repo stays
manageable. For multi-gigabyte outputs, store them in Zenodo or a
data-archive service and refer to them by DOI from the experiment's
companion `.md` doc.

`.log` files are tracked (the `.gitignore` has `!data/*.log`) so the
exact stdout of each run is part of the project history. Failed runs
go in `.err` (gitignored) so they don't clutter history.

## Naming

`<exp_id>_<descriptor>.{npy,log}` -- e.g. `exp_00_hop_drift.npy`,
`exp_00_hop_drift_shape16.log`. The `<exp_id>` prefix lets you
`ls data/exp_00*` and see everything that experiment touched.
