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

1. Final commits land on `main`; CI green.
2. Update `src/dcl_core/_version.py` and `CITATION.cff`
   (`version`, `date-released`).
3. Draft `release_notes/vX.Y.md` and `release_notes/vX.Y-release-message.md`.
4. **Deposit on Zenodo first to get the DOI.** Do not commit the
   version bump until the DOI is in hand -- the DOI lands in
   `CITATION.cff` before the release commit.
5. Commit the version bump (DOI included) with the change log.
6. Build the wheel + sdist: `make build`. The output in `dist/` is
   the artefact to upload to PyPI (optional) or to Zenodo as a
   software asset.
7. Tag `vX.Y` and push the tag.
8. Create the GitHub Release using the `vX.Y-release-message.md` body.

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
