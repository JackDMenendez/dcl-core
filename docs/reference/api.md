# API reference

The public API of `dcl_core` is whatever `src/dcl_core/__init__.py`
re-exports. The set is small on purpose: everything else is subject
to change without a semver bump.

## Stability contract

| Change | Semver impact |
|---|---|
| Add a new re-export to `__init__.py` | MINOR |
| Remove a re-export | MAJOR (with prior deprecation cycle) |
| Change a re-exported symbol's signature | MAJOR |
| Refactor internals without changing `__init__.py` | PATCH |
| Bug fix in an internal | PATCH |

Pre-1.0 (versions `0.X.Y`), MINOR bumps may also break callers. From
1.0 onward, MINOR is strictly additive.

## Currently re-exported

| Symbol | Module | Brief |
|---|---|---|
| `__version__` | `_version` | Semver string |
| `BipartiteLattice` | `lattice` | RGB/CMY lattice geometry |
| `DiscreteCausalSession` | `session` | Integer-token session state |
| `HopOperator` | `hop` | Bipartite Dirac evolution |
| `BresenhamResidual` | `remainder` | Error-diffusion accumulator |
| `TickScheduler` | `scheduler` | Multi-session orchestration |

For per-symbol details, see the corresponding `docs/reference/<module>.md`.

## Constants exposed

The following are part of the public API even though they're not in
`__all__` (they live in `dcl_core.lattice` and are imported by name
from there):

| Constant | Value | Meaning |
|---|---|---|
| `RGB_VECTORS` | `((1,1,1), (1,-1,-1), (-1,1,-1))` | Even-tick basis |
| `CMY_VECTORS` | negations of RGB | Odd-tick basis |
| `ALL_VECTORS` | RGB + CMY | All six |

These are tuples of integer tuples, deliberately not numpy arrays;
they are intended for use as keys / specifications, not as buffers.

## Versions of this document

This file lives in `docs/reference/api.md` in the source tree. On
each release, the file at the tagged commit IS the API contract for
that release. Diffs across tags show API evolution.
