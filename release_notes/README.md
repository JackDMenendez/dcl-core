# release_notes/

One folder per released version of `dcl_core`, with these artefacts
per release:

- `vX.Y.md` -- **change log** for the release.  Long-form, internal:
  what changed, why, what's deferred, dependency / API changes,
  performance notes.
- `vX.Y-release-message.md` -- **GitHub Release body**.  The
  outward-facing version: headline change, API delta, what's out
  of scope.  Posted as the body of the GitHub Release alongside
  the tag.
- `vX.Y-zenodo-description.txt` -- text pasted into Zenodo's
  **Description** field at deposit time.  Plain text; format
  modelled on
  `C:\dev\dcl-sm-derivation\release_notes\zenodo_description.txt`.
- `zenodo_references.txt` -- text pasted into Zenodo's
  **References** field (bibliographic citations, one per line).
  Durable: reuse and edit each release.
- `zenodo_related_works.txt` -- entries for Zenodo's
  **Related / alternate identifiers** section (Relation,
  Identifier, Scheme, Resource Type per row).  Durable: reuse
  and edit each release.

Templates:

- `TEMPLATE.md` for the change log
- `TEMPLATE-release-message.md` for the Release body

## Release flow (v2 -- 2026-05-26)

This protocol applies uniformly to every subproject in the
A=1 Discrete Causal Lattice series (dcl-core, dcl-delta-p-min,
dcl-paper-03, future ones).  Each step is owned by either
**Claude** (the agent running inside this repo) or **User**;
do not skip an owner-marked step.  Steps marked **conditional**
apply only if the named artefact exists in this repo
(e.g. step 8 only if `paper/main.tex` is present -- dcl-core has
no paper, so steps 8 and 14 are skipped here).

| # | Step | Owner |
|---|---|---|
| 1 | CI green on `main`. | Claude |
| 2 | Bump `src/dcl_core/_version.py` and `CITATION.cff` (`version`, `date-released`). | Claude |
| 3 | Draft `release_notes/vX.Y.md` and `release_notes/vX.Y-release-message.md`. | Claude |
| 4 | Draft `release_notes/vX.Y-zenodo-description.txt` (model: `dcl-sm-derivation/release_notes/zenodo_description.txt`). | Claude |
| 5 | Draft or update `release_notes/zenodo_references.txt`. | Claude |
| 6 | Draft or update `release_notes/zenodo_related_works.txt`. | Claude |
| 7 | Run unit tests (`pytest tests -v`); CI matrix on `main` must also be green. | Claude |
| 8 | **Conditional (`paper/main.tex` exists -- N/A for dcl-core):** | |
| 8a |   Add version to title in `main.tex`. | Claude |
| 8b |   Review abstract, introduction, conclusion, `References.bib`; make necessary changes. | Claude |
| 8c |   Build `main.tex` to `build/`. | Claude |
| 8d |   Review PDF in `build/`. | User |
| 9 | Run `export-vscode-extensions.{cmd,sh}` -> tracked `extensions.txt` at repo root. | Claude |
| 10 | Run `generate-dockerfile.{cmd,sh}` -> tracked `Dockerfile`. | Claude |
| 11 | Reserve a Zenodo DOI (Zenodo "New upload" -> *Reserve DOI*) and supply the DOI string to Claude. | User |
| 12 | DOI lands in `release_notes/vX.Y.md`. | Claude |
| 13 | DOI lands in `CITATION.cff` (`doi:` field). | Claude |
| 14 | **Conditional (`paper/main.tex` exists -- N/A for dcl-core):** | |
| 14a |   DOI lands in `main.tex` (`\thanks{}` block). | Claude |
| 14b |   Rebuild PDF. | Claude |
| 14c |   Final document check. | User |
| 14d |   Rename PDF to `stage/<doc-title>-vX.Y.pdf` (durable per-version snapshot). | User |
| 14e |   Upload the snapshotted PDF to Zenodo. | User |
| 15 | Upload software files (wheel, sdist, lattice data, etc.) to Zenodo. | User |
| 16 | Commit generated files + version bump (DOI included). | Claude |
| 17 | Tag `vX.Y` and push the tag.  (Tags are immutable once pushed; Claude must surface what it is about to do before running this.) | Claude |
| 18 | Create the GitHub Release draft using the `vX.Y-release-message.md` body. | User |
| 19 | (Optional) Publish to PyPI -- **skip by default**; opt-in only when explicitly requested. | User |
| 20 | Publish the GitHub Release (click *Publish* in the GitHub UI). | User |
| 21 | Supply Claude with the project-plan delta needed for the release. | User |
| 22 | Update project plan with release info. | Claude |
| 23 | Update GitHub project board. | User |

After step 23, walk downstream consumers per the *Downstream paper
coordination* section below.

## Helper scripts required by steps 9 and 10

The two helper scripts referenced by steps 9 and 10 **do not yet
exist** in this repo (or in any other DCL subproject as of
2026-05-26).  Each release that runs this protocol is blocked at
those steps until the scripts are created.

- `export-vscode-extensions.cmd` / `.sh` should produce
  `extensions.txt` at the repo root, containing one VS Code
  extension ID per line (the output of `code --list-extensions`).
- `generate-dockerfile.cmd` / `.sh` should produce a tracked
  `Dockerfile` that reproduces the development environment
  sufficiently to run this repo's experiments / tests.

When the canonical implementations of these scripts land -- likely
in the user's `wcde` repo (`C:\dev\wcde`) or in
`dcl-sm-derivation`'s `release_notes/` -- copy them into each
subproject.  Until then, Claude must stop at steps 9-10 and ask
the User how to proceed.

## Semver impact summary at release time

The change log MUST classify the release's semver impact:

- **MAJOR** -- breaking changes to anything re-exported from
  `src/dcl_core/__init__.py`. Includes signature changes and
  removals.
- **MINOR** -- new public-API additions. Backwards compatible.
- **PATCH** -- internal refactoring, bug fixes, no API surface
  change.

Pre-1.0 (versions `0.X.Y`), MINOR may also break callers.  Document
breaking changes in the change log either way.

## Immutability

Once a tag is pushed and the GitHub Release is published, **the
released version is immutable**.  Do not amend the tagged commit.
Do not re-deposit on Zenodo.  A typo in the release notes gets a
follow-up PATCH release; do not rewrite history.

Downstream paper repos that pin `dcl_core==X.Y.Z` are guaranteed
by this immutability that their reproducibility claims hold.

## Downstream paper coordination (post-release)

Every dcl-core release triggers a *bump-and-rebuild* workflow in
every paper repo that pins `dcl_core`.  After step 20 (GitHub
Release published), **walk each downstream consumer and update its
pinned version** before the consumer's next release.  A paper that
is still pinned to `@main` or to a stale tag is non-reproducible
relative to the latest engine; the bump is what restores its
reproducibility guarantee.

### Known downstream consumers

Update this list as new papers adopt `dcl_core`.

**`dcl-paper-03-tidal-ionization` (Paper~III).**  Pins `dcl_core`
in `virtual-env-requirements.txt` via a git URL.  After a
`dcl-core vX.Y.Z` release: bump the pin to `@vX.Y.Z`, add a
`references:` entry to `CITATION.cff` citing this release's
Zenodo DOI, run `./refresh-deps.sh` to re-install, re-run the
experiments, then commit "Pre-release: pin dcl_core to vX.Y.Z".
See Paper~III's `release_notes/README.md` *Pre-release: bump
pinned dcl_core* section for the full checklist.

**`dcl-delta-p-min`.**  The dp_min cross-engine investigation;
also pins `dcl_core` via git URL.  Currently pinned at v0.1.0;
will need to bump to v0.2.0 (or equivalent) once the `prob_floor`
parameter lands on `dcl_core.core` for Phase 2 numerics.

(As more papers adopt `dcl_core`, add another paragraph per
consumer in the same shape.)

Papers that do **not** depend on `dcl_core` and require no action:

- `dcl` (Paper~I) -- vendored engine, immutable v1.0 deposit.
- `dcl-sm-derivation` (Paper~II) -- pure sympy.
- `dcl-generator-zoo` -- pure sympy.

### Co-released window

When a paper is planned for release in the same window as
`dcl-core` (the "co-released" case -- e.g. `dcl-core v0.1.0` +
Paper~III v0.1), the order is fixed:

1. Finalise and deposit `dcl-core` first; get the Zenodo DOI.
2. In the paper repo, on `main`: bump the pin in
   `virtual-env-requirements.txt`, add the `references:` entry in
   `CITATION.cff`, run `./refresh-deps.sh` to reinstall, re-run
   the experiment to confirm nothing broke under the pin change,
   commit "Pre-release: pin dcl_core to vX.Y.Z".
3. Proceed with the paper's own release flow from there.

If the order is reversed (paper deposited first, then dcl-core),
the paper's deposit will cite an unreleased engine version, which
defeats the purpose of pinning.  Do not reverse the order.
