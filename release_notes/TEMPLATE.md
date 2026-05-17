<!-- markdownlint-disable MD022 MD024 MD032 MD047 MD060 -->
# vX.Y Release Notes

**Released:** YYYY-MM-DD
**Prior version:** vX.Y-1 (released YYYY-MM-DD)
**Tag:** vX.Y
**Semver impact:** MAJOR | MINOR | PATCH
**DOI:** [10.5281/zenodo.XXXXXXXX](https://doi.org/10.5281/zenodo.XXXXXXXX)

## Why vX.Y

Two or three sentences on the *purpose* of this release. Not a list of
changes -- those come below. Examples:

- vX.Y closes a release-candidate cycle: the API surface in `__init__.py`
  is now stable; downstream papers can depend on it.
- vX.Y is a maintenance release: bug fixes and clarifications, no API
  changes.
- vX.Y is a major revision: the public API of one or more modules has
  changed; downstream consumers must update pins.

## Public-API changes (since vX.Y-1)

### Added

- ...

### Changed

- ...

### Removed

- ...

### Deprecated

- ...

(Anything in the four sections above implies at least a MINOR bump,
typically MAJOR for "Changed" / "Removed".)

## Internal changes

- ...

## New experiments

1. **`exp_NN` (one-line title).** Result, audit-table status (if any).
2. ...

## New design docs / notes

- `docs/design/<file>.md` -- what it captures.
- ...

## Performance notes

- ...

## Dependency changes

- ...

## Reproducibility

One paragraph on how the headline result (if any) can be reproduced
from a fresh clone of the tag, including approximate wall-clock time
on a named hardware class.

---

For the detailed commit list, see `git log vX.Y-1..vX.Y`.
