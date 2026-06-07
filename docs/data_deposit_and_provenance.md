# Data deposit + provenance convention

How experiment data artefacts (`.npy` volumes, frame stacks, movies,
meshes, figures) are organised, version-controlled, and archived to
Zenodo across the A=1 Discrete Causal Lattice series.

`dcl-core` owns this convention because it owns the array contract (axis
order, dtype) that the artefacts inherit. The **deposits themselves live
in the paper / experiment repos** (`dcl-delta-p-min`, Paper~III, ...) --
`dcl-core` is software-only. Those repos reference this file.

---

## Granularity (decided 2026-06-03)

- **Provenance is per experiment.** Each `exp_NN_<name>` produces a
  self-describing bundle (its data + one manifest).
- **Deposit is one record per paper.** All of a paper's experiment
  bundles go into a *single* Zenodo **dataset** record (one DOI per
  paper), with per-experiment subfolders preserved inside.

```
<paper-repo>/data/
  exp_09_<name>/
    exp_09_<name>.manifest.json    # provenance -- COMMITTED to git
    summary.npz                    # small reduced data -- COMMITTED
    field_TXYZ.npy                 # heavy volume/stack -- gitignored -> Zenodo
    orbit.mp4  mesh.glb            # heavy -- gitignored -> Zenodo
  exp_12_<name>/
    ...
```

At paper release, the heavy artefacts under all `data/exp_*/` are
bundled into the paper's one dataset record. Manifests live in **both**
git and the deposit.

---

## The git / Zenodo / regenerate triage

Data is **regeneratable** from the pinned code + Docker image
(`HOWTO_REPRODUCE.md`), so the deposit question is *archival + citation*,
not reproducibility. Three tiers:

| Tier | What | Where |
|---|---|---|
| **Commit to git** | manifests, final figures (PNG/PDF), small reduced data (`*.summary.npz`, `*.manifest.json`), anything < ~5 MB and human-meaningful | repo `data/`, `figures/` |
| **Zenodo dataset DOI** | volumes `(X,Y,Z)`, frame stacks `(T,X,Y,Z)`, movies (`.mp4`), meshes (`.glb`/`.ply`/`.obj`/`.vti`), expensive scans (e.g. multi-hour runs) | one dataset record **per paper** |
| **Regenerate-only** | cheap intermediates | gitignored; rebuilt via the documented `make data` / Docker recipe |

**Heavy data is its own `dataset` record -- never bundled into the
*software* deposit.** Mixing volumetric data into the software DOI is the
wrong upload type, fights Zenodo's per-record size limit (50 GB default),
and couples two things that version independently. (This supersedes the
old release-flow step that uploaded "lattice data" alongside the wheel.)

Size cutoff for commit-vs-deposit: **~5 MB** per file (tune per repo).

---

## The provenance manifest

One JSON manifest per experiment. It does **double duty**: it is the
visualiser's sidecar (axis order / dtype / observable) *and* the Zenodo
dataset's self-description (what produced these bytes, from which engine
version). This is the use case that overrides v0.1.0's "no serialization
sidecar" deferral -- design it once, here.

`data/exp_NN_<name>/exp_NN_<name>.manifest.json`:

```json
{
  "schema_version": "1.0",
  "experiment": {
    "id": "exp_12_dp_min_sweep",
    "repo": "dcl-delta-p-min",
    "paper": "Paper III (tidal ionization)"
  },
  "producer": {
    "dcl_core_version": "1.0.0",
    "dcl_core_commit": "83813fc",
    "dcl_core_doi": "10.5281/zenodo.XXXXXXX",
    "python": "3.14.5",
    "platform": "win32"
  },
  "run": {
    "timestamp": "2026-06-03T00:00:00Z",
    "seed": 20260509,
    "params": {"strength": 30.0, "softening": 0.5, "omega": 0.1019,
               "grid": 33, "ticks": 400, "n_units": 100000},
    "wall_clock_s": 39.0
  },
  "arrays": [
    {
      "file": "field_TXYZ.npy",
      "shape": [400, 33, 33, 33],
      "dtype": "float32",
      "axis_order": "t,x,y,z",
      "observable": "(N_RGB + N_CMY) / n_units  (total token density |psi|^2)",
      "n_units": 100000,
      "tick_to_time": "frame index = tick number",
      "tier": "zenodo"
    },
    {
      "file": "summary.npz",
      "shape": null,
      "dtype": "mixed",
      "observable": "r_peak(N), radial PDFs",
      "tier": "git"
    }
  ],
  "regeneration": {
    "command": "python src/experiments/exp_12_dp_min_sweep.py",
    "docker_image": "paper-experiments:latest"
  },
  "deposit": {
    "part_of_paper_doi": "10.5281/zenodo.YYYYYYY",
    "zenodo_record": null
  }
}
```

Rules the manifest enforces:

- **`producer.dcl_core_doi` / `dcl_core_version` pins the released
  engine** the data was generated against -- the same `@vX.Y.Z` discipline
  as the requirements pin. A dataset whose manifest says `@main` (or a
  dirty commit) is non-reproducible by construction; fix before deposit.
- **`arrays[*].axis_order` + `dtype` are the array contract** (lattice
  fields are `x,y,z`; stacks prepend `t`). Store `float64` phase / `int64`
  tokens at source; frames may be downcast to `float32` for size, never
  upcast. CuPy arrays go through `cupy.asnumpy()` before save.
- **Every deposited file is listed** with its tier, so the deposit step
  is mechanical (collect `tier == "zenodo"`).

---

## Zenodo `dataset` record metadata (one per paper)

No `.zenodo.json` exists in the series today (metadata is typed in the
web UI). Adopt a versioned `.zenodo.json` in each paper repo so the
deposit is reproducible. Template:

```json
{
  "upload_type": "dataset",
  "title": "Paper III (tidal ionization) -- experiment data (dcl-core v1.0.0)",
  "version": "v1.0-data",
  "creators": [{"name": "Menendez, Jack D."}],
  "communities": [{"identifier": "a1-discrete-causal-lattice"}],
  "related_identifiers": [
    {"relation": "isSupplementTo", "scheme": "doi", "identifier": "<paper DOI>"},
    {"relation": "isDerivedFrom",  "scheme": "doi", "identifier": "<dcl-core software DOI @ released version>"},
    {"relation": "isPartOf",       "scheme": "doi", "identifier": "<series community / concept DOI>"}
  ],
  "description": "Per-experiment data bundles (volumes, frame stacks, movies, meshes) behind <Paper>. Each exp_NN/ subfolder carries a manifest.json with full provenance. Regenerate via the repo's Docker image."
}
```

The paper's own (publication) record gets the reciprocal
`isSupplementedBy` -> dataset DOI.

---

## Release-flow integration

Slots into `release_notes/README.md`'s co-released ordering. Because the
dataset's `isDerivedFrom` needs the software DOI to exist, the existing
"dcl-core deposit FIRST" rule already orders this correctly:

1. `dcl-core` software deposit (DOI minted).
2. Paper-side pin bump to the released `@vX.Y.Z`; rerun experiments so
   manifests record the released `dcl_core_doi` (not `@main`).
3. Paper publication record deposited (paper DOI minted).
4. **Paper dataset record deposited** -- one per paper, aggregating all
   `data/exp_*/` heavy artefacts; `.zenodo.json` relations filled with
   the software + paper DOIs.
5. Cross-link: paper record `isSupplementedBy` the dataset DOI.

Do not deposit a dataset whose manifests pin `@main`; that breaks the
reproducibility chain the same way a `@main` requirements pin does.

---

## `.gitignore` pattern (for paper repos)

Heavy generated types must not leak into git history. Mirror the
existing `data/*.err` / `!data/*.log` allowlist style:

```gitignore
# Heavy experiment artefacts -> Zenodo dataset, never git.
data/**/*.npy
data/**/*.npz
*.mp4
*.glb
*.ply
*.obj
*.vti
# Allowlist the small, committed, human-meaningful artefacts:
!data/**/*.manifest.json
!data/**/*.summary.npy
!data/**/*.summary.npz
!**/figures/**
```

(`dcl-core` carries a defensive copy of this block even though it runs no
experiments, so a stray volume dropped here cannot be committed.)
