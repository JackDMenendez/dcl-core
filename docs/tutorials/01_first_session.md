# 01 -- Your first DiscreteCausalSession

This tutorial walks through the smallest meaningful program against
`dcl_core`: instantiating a lattice, a session, and a hop, then
running one tick and inspecting the state.

## Prerequisites

```sh
./setup.sh          # creates .venv, installs dcl_core in editable mode
```

Confirm the install:

```sh
python -c "import dcl_core; print(dcl_core.__version__)"
```

## A minimal program

```python
from dcl_core import (
    BipartiteLattice,
    DiscreteCausalSession,
    HopOperator,
    TickScheduler,
)

# 1. Geometry.
lattice = BipartiteLattice(shape=(16, 16, 16))

# 2. One session: 1,000,000 probability tokens, omega = 0.1.
session = DiscreteCausalSession(lattice=lattice, n_units=1_000_000, omega=0.1)

# 3. Hop operator + scheduler.
hop = HopOperator(lattice=lattice)
scheduler = TickScheduler(lattice=lattice, hop=hop)
scheduler.register(session)

# 4. Sanity: token total before any evolution.
print(f"Pre-tick total: {session.total_tokens()} (expected {session.n_units})")

# 5. Run 10 ticks.
scheduler.run(n_ticks=10)

# 6. Sanity: A=1 still holds exactly.
session.assert_unity()
print(f"Post-tick total: {session.total_tokens()} (exact integer equality)")
```

## What you should observe

- `session.total_tokens()` is exactly `n_units` both before and after
  the run -- no floating-point drift, no renormalisation step.
- The token distribution has spread according to the bipartite Dirac
  hop. Visualise with:

  ```python
  import matplotlib.pyplot as plt
  rho = session.N_R + session.N_L            # token density
  plt.imshow(rho.sum(axis=2))                # 2D projection
  plt.show()
  ```

## Next

- Tutorial 02 -- choosing `n_units` for a target physical accuracy.
- Reference -- the public API summary in `docs/reference/api.md`.
- Design -- why integer tokens? `docs/design/01_planck_of_probability.md`.
