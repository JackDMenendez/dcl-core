# release_notes/

One folder per released version, two files per release:

- `vX.Y.md` -- the **change log**. Long-form, internal: what changed,
  why, what's deferred, dependency / API changes, performance notes.
- `vX.Y-release-message.md` -- the **GitHub Release body**. The
  outward-facing version: headline change, API delta, what's out of
  scope. Posted as the body of the GitHub Release alongside the tag.

Templates:

- `TEMPLATE.md` for the change log
- `TEMPLATE-release-message.md` for the Release body

## Release flow

Pre-conditions: final commits landed on `main`; CI green across all
the matrix (Linux/macOS/Windows x Python 3.11/3.12).

1. **Final tests + sanity check.**  Run the full local test suite
   (`pytest tests -v`) and confirm the GitHub Actions CI is green
   on the head of `main`.  Open `tests/test_cross_validation.py`
   and confirm any newly-flipped (skipped -> active) tests pass at
   the current state of `dcl_core.core3d` implementations.

2. **[User] Reserve a Zenodo DOI.**  In the Zenodo "New upload"
   form, click *Reserve DOI* to mint a DOI without publishing the
   deposit yet.  Copy the DOI string (form
   `10.5281/zenodo.NNNNNNNN`).  The deposit stays in *Draft* status
   until step 6.

3. **Insert the DOI and bump version metadata.**  Update in one
   commit-able pass:

   - `src/dcl_core/_version.py`: bump `__version__ = "X.Y.Z"`.
   - `CITATION.cff`: bump `version:` and `date-released:`, fill
     in `doi: 10.5281/zenodo.NNNNNNNN`.
   - `release_notes/vX.Y.md` and
     `release_notes/vX.Y-release-message.md`: fill in the DOI
     in the header blocks.

4. **Build the distribution artefacts.**

   ```text
   python -m build
   ```

   Output lands in `dist/` as the wheel + sdist.  These are the
   files uploaded to Zenodo (as software assets) and, optionally,
   to PyPI.

5. **Commit the version bump + DOI fill-in.**  Suggested message:

   ```text
   vX.Y.Z release: fill DOI placeholders, build dist artefacts

   - DOI 10.5281/zenodo.NNNNNNNN added to CITATION.cff,
     release_notes/vX.Y*.md; __version__ bumped to X.Y.Z
   - Wheel + sdist built into dist/
   ```

6. **[User] Upload the dist artefacts to Zenodo and publish.**
   Drag `dist/dcl_core-X.Y.Z-py3-none-any.whl` and
   `dist/dcl_core-X.Y.Z.tar.gz` into the reserved-DOI draft
   deposit; fill in the metadata (title, authors, abstract,
   keywords, related identifiers pointing at Paper~I); click
   *Publish*.  This locks in the DOI and makes the deposit
   immutable.

7. **Tag and push.**

   ```text
   git tag vX.Y.Z
   git push origin vX.Y.Z
   ```

   Tags are immutable once pushed; confirm before pushing.

8. **Create the GitHub Release.**

   ```text
   gh release create vX.Y.Z \
       --title "vX.Y.Z -- <one-line headline>" \
       --notes-file release_notes/vX.Y.Z-release-message.md
   ```

9. **(Optional) Publish to PyPI.**

   ```text
   python -m twine upload dist/*
   ```

   Skip until adoption justifies the maintenance overhead;
   downstream papers can pin via `git+https` URLs without PyPI.

10. **Walk downstream consumers** per the *Downstream paper
    coordination* section below.  Bump each paper's pin in
    `virtual-env-requirements.txt` to the new `@vX.Y.Z` and add
    a `references:` entry to its `CITATION.cff`.

## Semver impact summary at release time

The change log MUST classify the release's semver impact:

- **MAJOR** -- breaking changes to anything re-exported from
  `src/dcl_core/__init__.py`. Includes signature changes and
  removals.
- **MINOR** -- new public-API additions. Backwards compatible.
- **PATCH** -- internal refactoring, bug fixes, no API surface
  change.

Pre-1.0 (versions `0.X.Y`), MINOR may also break callers. Document
breaking changes in the change log either way.

## Immutability

Once a tag is pushed and the GitHub Release is created, **the
released version is immutable**. Do not amend the tagged commit. Do
not re-deposit on Zenodo. A typo in the release notes gets a follow-up
PATCH release; do not rewrite history.

Downstream paper repos that depend on `dcl_core==X.Y.Z` are
guaranteed by this immutability that their reproducibility claims
hold.

## Downstream paper coordination (post-release)

Every dcl-core release triggers a *bump-and-rebuild* workflow in
every paper repo that pins `dcl_core`.  After tagging and pushing a
dcl-core release, **walk each downstream consumer and update its
pinned version** before the consumer's next release.  A paper that
is still pinned to `@main` or to a stale tag is non-reproducible
relative to the latest engine; the bump is what restores its
reproducibility guarantee.

### Known downstream consumers

Update this list as new papers adopt `dcl_core`.

**`dcl-paper-03-tidal-ionization` (Paper~III).**  Pins `dcl_core`
in `virtual-env-requirements.txt` via the line
`dcl_core @ git+https://github.com/JackDMenendez/dcl-core@...`.
After a `dcl-core vX.Y.Z` release: bump `@main` (or `@vX.Y.Z-prior`)
to `@vX.Y.Z`; add a `references:` entry to `CITATION.cff` citing
this release's Zenodo DOI; see Paper~III's
`release_notes/README.md` *Pre-release: bump pinned dcl_core*
section for the full checklist.

(As more papers adopt `dcl_core` -- proton internals, plasma /
gradient ionization, recombination, etc. -- add another paragraph
per consumer in the same shape.)

Papers that do **not** depend on `dcl_core` and require no action:

- `dcl` (Paper~I) -- vendored engine, immutable v1.0 deposit.
- `dcl-sm-derivation` (Paper~II) -- pure sympy.
- `dcl-generator-zoo` -- pure sympy.

### Co-released window

When a paper is planned for release in the same window as
`dcl-core` (the "co-released" case -- e.g. `dcl-core` v0.1.0 +
Paper~III v0.1), the order is fixed:

1. Finalise and deposit `dcl-core` first; get the Zenodo DOI.
2. In the paper repo, on `main`: bump the pin in
   `virtual-env-requirements.txt`, add the `references:` entry in
   `CITATION.cff`, reinstall the venv, re-run the experiment to
   confirm nothing broke under the pin change, commit
   "Pre-release: pin dcl_core to vX.Y.Z" (this is the paper's
   pre-release checkpoint).
3. Proceed with the paper's own release flow from there.

If the order is reversed (paper deposited first, then dcl-core),
the paper's deposit will cite an unreleased engine version, which
defeats the purpose of pinning.  Do not reverse the order.
