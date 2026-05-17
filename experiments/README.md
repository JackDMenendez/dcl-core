# experiments/

Numerical experiments that **exercise the platform**. Distinct from
the `tests/` directory: tests are millisecond-scale unit tests for
correctness of operators; experiments are seconds-to-hours runs that
demonstrate the platform reproduces known physics (or fails to, with
diagnostics).

Experiments here are not paper deliverables -- they live in this
software repo because the software is the deliverable. A paper that
cites this library will have its **own** experiments living in its
own repo (use `dcl-paper-experiment-template`), pinning to a specific
release of this core.

## Naming

Every experiment has three things:

| File | Role |
|---|---|
| `exp_NN_short_name.py` | Runnable script. Prints PASS/FAIL/STUB to stdout, drops data into `data/`. |
| `exp_NN_short_name.md` | Companion doc: what the experiment claims, parameters, expected output. |
| (entry in `EXPERIMENTS.md`) | Index row pointing at both. |

`NN` is a two-digit zero-padded sequence number. Sub-experiments use
a letter suffix (`exp_00b`, `exp_12d_tight`).

## Lifecycle of an experiment

1. **STUB**: `.md` exists, `.py` may be empty. Used to lock in the
   *question* before the answer.
2. **In progress**: `.py` runs but result is not yet clean PASS.
   The `.md` records what's still missing.
3. **PART**: experiment runs and demonstrates the mechanism, but
   quantitative match is below the stated precision.
4. **PASS**: experiment runs cleanly and confirms the claim.
5. **FAIL**: experiment runs cleanly and disconfirms. Important -- a
   confirmed FAIL is a first-class result. Do not delete the row.

## Conventions

- Experiments write to `data/<exp_id>_<descriptor>.npy` and
  `data/<exp_id>_<descriptor>.log`. Both are tracked by git when
  reasonably small.
- Long experiments print incremental progress so `tail -f` is useful.
- A failing experiment exits non-zero; every script ends with
  `if __name__ == "__main__": sys.exit(0 if success else 1)`.
- Experiments import from `dcl_core` (the public API only); they do
  not reach into private modules.

## Running

A single experiment:
```sh
python -u experiments/exp_00_hop_drift.py
```

The whole suite via the makefile:
```sh
make -C experiments all
```

## Categories

Suggested folder-flat categories (use prefixes, not subfolders):

| Prefix | What it's for |
|---|---|
| `exp_0N` | Sanity / diagnostic (cheap; should always PASS) |
| `exp_1N` | Hello-world physics (causal cone, inertia, ...) |
| `exp_2N` | Quantum-mechanical reproductions (interference, ...) |
| `exp_3N` | Quantitative reproductions (Bohr radius, ...) |
| `exp_4N` | Calibration sweeps (N value, lattice spacing, ...) |
| `exp_9N` | Stress / performance benchmarks |

Numbers are not promises; reorder freely until v1.0. Once v1.0 ships,
keep numbers stable so downstream papers can cite them.
