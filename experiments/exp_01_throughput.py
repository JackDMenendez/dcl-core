"""exp_01_throughput -- Tick-throughput baseline across lattice shapes.

What this experiment measures
-----------------------------
Wall-clock time per ``TickScheduler.step`` call as a function of
lattice shape, for a single registered session populated by
``delta_at`` at the lattice centre.  Output is a structured ``.npy``
plus a ``.json`` sidecar tagging the run with host / CPU / Python /
NumPy info, so future re-runs (on upgraded hardware) can be diffed
against the baseline by reading both files.

Why this experiment exists
--------------------------
The first thing you want to know before upgrading hardware is *where*
the bottleneck is.  The throughput curve here -- ticks/sec vs lattice
shape -- tells you two things:

1. **Absolute speed** at each shape.  Is 64^3 fast enough for the
   target experiment, or do we need a hardware change?
2. **Scaling slope.**  If wall time grows linearly with site count,
   we're compute- or bandwidth-bound (a CPU or memory upgrade
   helps).  If it grows super-linearly, there's a structural issue
   (cache thrashing, ``argpartition`` blow-up, etc.) that no
   hardware fix will solve.

Companion to ``exp_02_phase_profile.py``, which drills into *where
within* a tick the time goes.

Output
------
``data/exp_01_throughput.npy``
    Structured array with columns ``(shape, n_sites, tick_mean_s,
    tick_std_s, n_measure)``.  See ``benchmark_shape`` for the
    measurement protocol.
``data/exp_01_throughput.json``
    Host / CPU / Python / NumPy versions and the configuration used
    for the run, so the ``.npy`` is interpretable in isolation.

Status: ACTIVE
"""

from __future__ import annotations

import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
EXP_ID = "exp_01_throughput"


# Measurement budget.  Edit to taste; defaults aim at ~5-15 minutes
# total on a 2018-era Ryzen 7 2700 + DDR4-2400.  For the "overnight"
# baseline, multiply N_MEASURE values by ~10.
SHAPES: list[tuple[int, int, int]] = [
    (8, 8, 8),
    (16, 16, 16),
    (32, 32, 32),
    (64, 64, 64),
    (128, 128, 128),
    (256, 256, 256),
]
N_UNITS = 1_000_000
N_WARMUP = 5
N_MEASURE: dict[tuple[int, int, int], int] = {
    (8, 8, 8): 5000,
    (16, 16, 16): 2000,
    (32, 32, 32): 500,
    (64, 64, 64): 100,
    (128, 128, 128): 30,
    (256, 256, 256): 10,
}


def benchmark_shape(shape: tuple[int, int, int]) -> tuple[float, float, int]:
    """Measure mean / std tick time at ``shape``; return ``(mean_s, std_s, n)``.

    Builds a fresh ``BipartiteLattice``, populates one session via
    ``delta_at`` at the centre, runs ``N_WARMUP`` discarded ticks, then
    times the next ``N_MEASURE[shape]`` ticks individually.
    """
    from dcl_core.core3d import (
        BipartiteLattice,
        DiscreteCausalSession,
        HopOperator,
        TickScheduler,
    )

    lattice = BipartiteLattice(shape=shape)
    centre = tuple(s // 2 for s in shape)
    session = DiscreteCausalSession.delta_at(
        lattice, n_units=N_UNITS, omega=0.1, position=centre
    )
    scheduler = TickScheduler(lattice=lattice, hop=HopOperator(lattice=lattice))
    scheduler.register(session)

    # Warm up: discard first ticks to let allocators / caches settle.
    for _ in range(N_WARMUP):
        scheduler.step()

    # Measure individual tick durations.
    n_measure = N_MEASURE[shape]
    durations = np.zeros(n_measure, dtype=np.float64)
    for i in range(n_measure):
        t0 = time.perf_counter()
        scheduler.step()
        durations[i] = time.perf_counter() - t0
    return float(durations.mean()), float(durations.std()), n_measure


def format_seconds(seconds: float) -> str:
    """Pretty-print a duration in seconds, picking a sensible unit."""
    if seconds < 1e-3:
        return f"{seconds * 1e6:.2f} us"
    if seconds < 1.0:
        return f"{seconds * 1e3:.3f} ms"
    return f"{seconds:.3f} s"


def main() -> int:
    print(f"[{EXP_ID}] STARTING")
    print(f"  Host:    {platform.node()}")
    print(f"  System:  {platform.system()} {platform.release()}")
    print(f"  CPU:     {platform.processor() or platform.machine()}")
    print(f"  Python:  {platform.python_version()}")
    print(f"  NumPy:   {np.__version__}")
    print(f"  n_units: {N_UNITS}")
    print(f"  warmup:  {N_WARMUP} ticks")
    print()

    records: list[tuple[tuple[int, int, int], float, float, int]] = []
    for shape in SHAPES:
        n_sites = shape[0] * shape[1] * shape[2]
        n_measure_here = N_MEASURE[shape]
        print(
            f"  shape={shape!s:<18} sites={n_sites:>10}  "
            f"measuring {n_measure_here} ticks ...",
            end=" ",
            flush=True,
        )
        try:
            mean_s, std_s, n_measured = benchmark_shape(shape)
        except MemoryError:
            print("MemoryError; skipping this shape and all larger")
            break
        records.append((shape, mean_s, std_s, n_measured))
        ticks_per_sec = 1.0 / mean_s if mean_s > 0 else float("inf")
        sites_per_sec = n_sites / mean_s if mean_s > 0 else float("inf")
        print(
            f"{format_seconds(mean_s)}/tick "
            f"(sd {format_seconds(std_s)}, "
            f"{ticks_per_sec:.2f} ticks/s, "
            f"{sites_per_sec:.2e} sites/s)"
        )

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------
    # Save structured .npy.
    # ---------------------------------------------------------------
    dtype = np.dtype(
        [
            ("shape", "i8", (3,)),
            ("n_sites", "i8"),
            ("tick_mean_s", "f8"),
            ("tick_std_s", "f8"),
            ("n_measure", "i8"),
        ]
    )
    arr = np.array(
        [
            (np.array(shape, dtype=np.int64), shape[0] * shape[1] * shape[2], m, s, n)
            for (shape, m, s, n) in records
        ],
        dtype=dtype,
    )
    npy_path = DATA_DIR / f"{EXP_ID}.npy"
    np.save(npy_path, arr)
    print(f"\n[{EXP_ID}] saved: {npy_path}")

    # ---------------------------------------------------------------
    # JSON sidecar -- host / config / timestamp.
    # ---------------------------------------------------------------
    sidecar = {
        "experiment": EXP_ID,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "host": {
            "node": platform.node(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "python": platform.python_version(),
        "numpy": np.__version__,
        "config": {
            "n_units": N_UNITS,
            "n_warmup": N_WARMUP,
            "shapes": [list(s) for s in SHAPES],
            "n_measure": {str(list(k)): v for k, v in N_MEASURE.items()},
        },
        "results": [
            {
                "shape": list(shape),
                "n_sites": shape[0] * shape[1] * shape[2],
                "tick_mean_s": m,
                "tick_std_s": s,
                "n_measure": n,
            }
            for (shape, m, s, n) in records
        ],
    }
    json_path = DATA_DIR / f"{EXP_ID}.json"
    json_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    print(f"[{EXP_ID}] saved: {json_path}")

    # ---------------------------------------------------------------
    # Summary table to stdout.
    # ---------------------------------------------------------------
    print()
    print("Summary")
    print("-------")
    header = (
        f"{'shape':<18} {'sites':>10}  {'time / tick':>14}  "
        f"{'ticks/s':>10}  {'sites/s':>13}"
    )
    print(header)
    print("-" * len(header))
    for shape, mean_s, _std_s, _n in records:
        n_sites = shape[0] * shape[1] * shape[2]
        ticks_per_sec = 1.0 / mean_s if mean_s > 0 else float("inf")
        sites_per_sec = n_sites / mean_s if mean_s > 0 else float("inf")
        print(
            f"{str(shape):<18} {n_sites:>10}  "
            f"{format_seconds(mean_s):>14}  "
            f"{ticks_per_sec:>10.2f}  "
            f"{sites_per_sec:>13.2e}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
