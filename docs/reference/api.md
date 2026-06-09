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

## `dcl_core.core` engine: `prob_floor`

`dcl_core.core.CausalSession` accepts an optional keyword-only
`prob_floor` parameter (added after v0.2.0):

```python
CausalSession(lattice, initial_node, instruction_frequency,
              momentum=(0, 0, 0), is_massless=False, *, prob_floor=None)
```

| Value | Behaviour |
|---|---|
| `None` (default) | No clamp. Bit-for-bit identical to the pre-`prob_floor` engine. |
| `float` in `(0, 1)` | Per-site Born-density floor (see below). |

**Semantics.** Per-site probability is `p(x) = |psi_R(x)|^2 +
|psi_L(x)|^2`. At end-of-tick, any node with `0 < p(x) < prob_floor`
has both spinor components scaled by the real factor
`sqrt(prob_floor / p(x))`, so `p(x) == prob_floor` exactly with phase
and the `psi_R/psi_L` ratio preserved; the joint state is then
renormalised so `sum(p) == 1` (A=1 preserved). Nodes with `p(x) == 0`
carry no phase and are left at zero — the floor is the minimum
*non-zero* probability quantum, the continuous-engine analogue of
`core3d`'s `1/N = delta_p_min` granularity.

This is the `delta_p_min` knob for the continuous engine, required by
`dcl-delta-p-min`'s 4-cell grid (`core` cells map `delta_p_min ->
prob_floor`; `core3d` cells map `delta_p_min -> N = 1/delta_p_min`).
Out of scope: per-edge/per-plaquette floors, a complementary
`prob_ceiling` (the unity renorm restores the high-side symmetry),
and a `prob_floor` on `core3d` (its quantum is `N` alone).

## Versions of this document

This file lives in `docs/reference/api.md` in the source tree. On
each release, the file at the tagged commit IS the API contract for
that release. Diffs across tags show API evolution.
