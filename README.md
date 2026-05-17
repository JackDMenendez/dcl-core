# Core-Software Template (`dcl_core`-style)

A starting scaffold for a **standalone research-software package**: a
versioned, pip-installable Python library that papers depend on rather
than vendor. Derived from the *Geometry First* (A=1 Discrete Causal
Lattice) project's infrastructure.

Use this template when:

- The software is the deliverable. Papers cite specific releases.
- Multiple papers / experiments will depend on the same core.
- Code lifetime is measured in years, not in one paper's review cycle.

For a paper-only project, use `dcl-paper-template`. For a paper that
ships with one-off companion experiments, use
`dcl-paper-experiment-template`. **This template is for the software
itself**, with papers as separate downstream projects.

## Click "Use this template" on GitHub

This repo is configured as a GitHub Template Repository. From the
GitHub web UI on the template's page, click **Use this template ->
Create a new repository**, then `git clone` the new repo locally.

## What you get

```
.
+-- src/dcl_core/                 -- the installable Python package
|   +-- __init__.py              -- top-level: exposes __version__ + submodules
|   +-- _version.py              -- single-source version string
|   +-- core/                    -- continuous-amplitude engine (Paper I port)
|   |   +-- __init__.py          -- Provenance + Paper I re-exports
|   |   +-- OctahedralLattice.py
|   |   +-- CausalSession.py
|   |   +-- CompositeCausalSession.py
|   |   +-- TickScheduler.py
|   |   +-- PhaseOscillator.py
|   |   +-- UnityConstraint.py
|   +-- core3d/                  -- integer-token engine (new design)
|       +-- __init__.py          -- core3d public API
|       +-- lattice.py           -- BipartiteLattice (frozen dataclass)
|       +-- session.py           -- DiscreteCausalSession (integer tokens)
|       +-- hop.py               -- HopOperator (bipartite Dirac)
|       +-- remainder.py         -- BresenhamResidual (fractional-bit carry)
|       +-- scheduler.py         -- TickScheduler (multi-session)
|       +-- backends/            -- CPU / GPU implementations
+-- tests/                        -- pytest; conservation laws, continuum limits
+-- experiments/                  -- platform exercise (NOT paper deliverables)
|   +-- exp_00_hop_drift.py      -- diagnostic; measures pre-rounding drift
+-- docs/                         -- tutorials, reference, design rationale
+-- data/                         -- experiment outputs
+-- notes/                        -- working theoretical notes
+-- release_notes/                -- per-version change log + Release body
+-- .claude/agents/               -- API-stability + naming reviewers
+-- .github/workflows/tests.yml   -- CI
+-- pyproject.toml                -- package metadata, dependencies, tool config
+-- virtual-env-requirements.txt  -- dev/test requirements
+-- CLAUDE.md                     -- project memory for Claude Code
+-- CITATION.cff                  -- machine-readable citation
+-- LICENSE                       -- MIT
+-- makefile common.mak           -- root build (env -> tests -> docs)
+-- build.sh build.cmd            -- platform wrappers around make
+-- setup.sh setup.cmd            -- create venv + install requirements
+-- .gitignore .gitattributes .gitmessage
```

## First steps after creating your repo

1. **Search-and-replace the placeholders** in:
   - `pyproject.toml` -- `name`, `description`, `authors`, repo URL
   - `src/dcl_core/_version.py` -- starting version string
   - `CITATION.cff` -- title, author, ORCID, repo URL
   - `LICENSE` -- year and copyright holder
   - `CLAUDE.md` -- short title, current status block
   - `README.md` (this file) -- replace with your project's own README
   - If renaming the package: rename `src/dcl_core/` to `src/<your_pkg>/`
     and update `pyproject.toml`'s `name` / `[tool.hatch.build]`
     fields, plus all `from dcl_core import ...` statements in tests
     and experiments.
2. **Set up the environment** (creates `.venv`, installs the package
   in editable mode plus dev requirements):
   ```sh
   ./setup.sh                  # POSIX / MSYS2 UCRT64 on Windows
   setup.cmd                   # Windows cmd / PowerShell
   ```
3. **Sanity-check the toolchain**:
   ```sh
   ./build.sh tests            # pytest against tests/
   python -u experiments/exp_00_hop_drift.py
   python -c "import dcl_core; print(dcl_core.__version__)"
   ```
4. **Replace the stubs** with real implementations:
   - `src/dcl_core/lattice.py`, `session.py`, `hop.py`, `remainder.py`,
     `scheduler.py`
   - `tests/test_*.py` -- conservation, continuum-limit, gauge-invariance
   - `experiments/exp_00_hop_drift.{py,md}` -- the first real diagnostic
5. **Each new core capability** gets three things in lock-step:
   - implementation in `src/dcl_core/`
   - unit tests in `tests/` (named after the invariant being verified)
   - one tutorial or reference page in `docs/`

## Why a separate software repo

The pattern this template assumes:

- The **core library** lives here and is the primary deliverable.
  Each release is tagged, Zenodo-deposited, and DOI-stamped.
- **Papers** that build on this core live in *separate* repos (use
  `dcl-paper-template` or `dcl-paper-experiment-template` for those).
- A paper repo depends on a specific released version of the core,
  named in its `pyproject.toml` or `virtual-env-requirements.txt`:
  ```text
  dcl_core==0.1.0
  ```
  or, for a pre-release / private dependency, by Git URL:
  ```text
  dcl_core @ git+https://github.com/JackDMenendez/dcl-core@v0.1.0
  ```
- This decoupling means the core can evolve at its own pace, and a
  paper's reproducibility is anchored to a specific software release
  (not a moving target).

## Build requirements

- GNU Make >= 4.3 (the stock Windows port is too old; on Windows use
  MSYS2 UCRT64 with `pacman -S make`).
- Python 3.11+ with `venv` (created by `setup.sh` / `setup.cmd`).
- Optional: a CUDA-capable GPU with CuPy installed for GPU backends
  (see `pyproject.toml`'s `[project.optional-dependencies]` for the
  `gpu` extra).

## Running tests

```sh
./build.sh tests                       # full suite
python -m pytest tests/test_hop.py -v  # one file
python -m pytest -k "conservation"     # by keyword
```

## Running experiments

Experiments here are **platform diagnostics and exercise**, not paper
deliverables. They confirm the library reproduces known behaviour at
small scale, measure drift / cost / accuracy as parameters vary, and
serve as runnable tutorials for new users.

A single experiment:
```sh
python -u experiments/exp_00_hop_drift.py
```

The whole suite:
```sh
make -C experiments all
```

Long experiments are listed in `experiments/EXPERIMENTS.md` with
expected wall-clock cost. Prefer named targets to the suite-level
`all`.

## Release flow

See `release_notes/README.md`. Short version:

1. CI green on `main`.
2. Bump version in `src/dcl_core/_version.py` and `CITATION.cff`.
3. Draft `release_notes/vX.Y.md` and `release_notes/vX.Y-release-message.md`.
4. **Deposit on Zenodo first** to get the DOI; do not commit the
   version bump until the DOI is in hand.
5. Commit version bump (DOI included).
6. Tag `vX.Y`, push the tag.
7. Create the GitHub Release using the release-message body.
8. (Optional) Publish to PyPI: `python -m build && python -m twine upload dist/*`.

Downstream papers pin to the released version. A future paper repo
that depends on `dcl_core==2.0.1` will keep working even after the
core has moved on to v2.1.

## License

Source: MIT (see `LICENSE`).
Documentation: CC BY 4.0.
