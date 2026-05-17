"""exp_00_hop_drift -- Pre-rounding drift diagnostic.

What this experiment measures
-----------------------------
The analytical hop step (`HopOperator.step`) produces fractional new
amplitudes whose squared modulus, summed across the lattice, may not
equal `n_units` exactly. The discrepancy is what the
`BresenhamResidual` step has to absorb.

This experiment measures the **size** of that discrepancy across a
range of session configurations. The output guides the choice of
`n_units` for downstream physics experiments: drift should sit safely
above `epsilon_P = 1 / n_units`, otherwise the rounding step erases
real dynamics; safely below `n_units`, otherwise it's a structural
unitarity bug.

Why it's exp_00
---------------
Cheap, fast, runs on any host (no GPU required). Should be the first
thing executed against any new core implementation -- if it doesn't
produce sensible output, nothing downstream will either.

Status: STUB
------------
This script is a skeleton. Implementation deferred until the core
modules (`hop`, `session`) are concrete. It exists at template-time so
the experiment lifecycle is documented; remove the `STUB` exit once
the implementation is in place.

See `exp_00_hop_drift.md` for the design rationale and expected
output ranges.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
EXP_ID = "exp_00_hop_drift"


def main() -> int:
    """Run the drift diagnostic. Returns 0 on PASS, 1 on FAIL."""
    print(f"[{EXP_ID}] STARTING")

    # ------------------------------------------------------------------
    # STUB: replace with the real diagnostic once HopOperator is
    # implemented. Suggested shape:
    #
    #     from dcl_core import (
    #         BipartiteLattice, DiscreteCausalSession, HopOperator,
    #     )
    #
    #     for shape in [(8,)*3, (16,)*3, (32,)*3]:
    #         lat = BipartiteLattice(shape=shape)
    #         hop = HopOperator(lattice=lat)
    #         for n_units in [10**3, 10**6, 10**9]:
    #             session = DiscreteCausalSession(lat, n_units=n_units, omega=0.1)
    #             # initialise to a known Gaussian profile
    #             ...
    #             total_pre = session.total_tokens()
    #             psi_R_new, psi_L_new = hop.step(session, parity="even")
    #             total_post_analytical = (
    #                 np.abs(psi_R_new)**2 + np.abs(psi_L_new)**2
    #             ).sum() * n_units
    #             drift = float(total_post_analytical - total_pre)
    #             record(shape, n_units, drift)
    #
    #     log results -> data/exp_00_hop_drift.log
    #     save raw  -> data/exp_00_hop_drift.npy
    # ------------------------------------------------------------------

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    log_path = DATA_DIR / f"{EXP_ID}.log"
    log_path.write_text(
        f"[{EXP_ID}] STUB -- replace main() with the real diagnostic.\n"
        f"Numpy version: {np.__version__}\n",
        encoding="utf-8",
    )

    print(f"[{EXP_ID}] STUB -- exiting non-zero so audit can see it.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
