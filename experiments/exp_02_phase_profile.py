"""exp_02_phase_profile -- Where the time goes inside one tick.

What this experiment measures
-----------------------------
A single fixed-shape session is stepped under ``cProfile``; the
function-level breakdown identifies which engine phases dominate the
wall-clock cost.  Used together with ``exp_01_throughput`` (macro
picture) to decide whether the next-hardware bottleneck is going to
be:

- **memory bandwidth** -- ``np.roll`` / ``np.abs`` / ``np.angle``
  dominate -> DDR5 RAM + faster GPU VRAM matters.
- **per-element compute** -- ``np.sqrt`` / ``np.exp`` /
  ``argpartition`` dominate -> a faster CPU (or GPU port) matters.
- **Python overhead** -- many tiny calls, total dominated by
  function-call cost -> consider vectorising further, no hardware
  fix needed.

Output
------
``data/exp_02_phase_profile.prof``
    Raw cProfile dump.  Inspect interactively with
    ``snakeviz data/exp_02_phase_profile.prof`` (install snakeviz
    separately: ``pip install snakeviz``).
``data/exp_02_phase_profile.log``
    Top-30 functions by cumulative time, suitable for a quick read.

Configuration
-------------
``SHAPE`` (default ``(64, 64, 64)``) and ``N_PROFILE`` (default 100)
sit at the top of the script -- edit before running for larger
lattices or longer profiles.

Status: ACTIVE
"""

from __future__ import annotations

import cProfile
import io
import platform
import pstats
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
EXP_ID = "exp_02_phase_profile"

SHAPE: tuple[int, int, int] = (64, 64, 64)
N_UNITS = 1_000_000
N_WARMUP = 5
N_PROFILE = 100
TOP_N = 30


def main() -> int:
    print(f"[{EXP_ID}] STARTING")
    print(f"  Host:    {platform.node()}")
    print(f"  CPU:     {platform.processor() or platform.machine()}")
    print(f"  Python:  {platform.python_version()}")
    print(f"  NumPy:   {np.__version__}")
    print(f"  Shape:   {SHAPE}  (sites = {SHAPE[0] * SHAPE[1] * SHAPE[2]})")
    print(f"  Profile: {N_PROFILE} ticks (after {N_WARMUP} warmup)")
    print()

    from dcl_core.core3d import (
        BipartiteLattice,
        DiscreteCausalSession,
        HopOperator,
        TickScheduler,
    )

    lattice = BipartiteLattice(shape=SHAPE)
    centre = tuple(s // 2 for s in SHAPE)
    session = DiscreteCausalSession.delta_at(
        lattice, n_units=N_UNITS, omega=0.1, position=centre
    )
    scheduler = TickScheduler(lattice=lattice, hop=HopOperator(lattice=lattice))
    scheduler.register(session)

    # Warm up outside the profile.
    for _ in range(N_WARMUP):
        scheduler.step()

    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(N_PROFILE):
        scheduler.step()
    profiler.disable()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Raw dump for interactive exploration.
    prof_path = DATA_DIR / f"{EXP_ID}.prof"
    profiler.dump_stats(str(prof_path))
    print(f"[{EXP_ID}] saved raw profile: {prof_path}")
    print(f"  (open interactively with: snakeviz {prof_path})")

    # Text summary: top N by cumulative time, then top N by total time.
    log_lines: list[str] = []

    def emit(line: str = "") -> None:
        print(line)
        log_lines.append(line)

    emit()
    emit("Top by CUMULATIVE time (includes time in callees)")
    emit("-" * 50)
    buf_cum = io.StringIO()
    pstats.Stats(profiler, stream=buf_cum).sort_stats("cumulative").print_stats(TOP_N)
    emit(buf_cum.getvalue())

    emit("Top by TOTAL time (excludes time in callees)")
    emit("-" * 50)
    buf_tot = io.StringIO()
    pstats.Stats(profiler, stream=buf_tot).sort_stats("tottime").print_stats(TOP_N)
    emit(buf_tot.getvalue())

    log_path = DATA_DIR / f"{EXP_ID}.log"
    header = (
        f"# {EXP_ID}\n"
        f"# timestamp_utc: {datetime.now(timezone.utc).isoformat()}\n"
        f"# host: {platform.node()}\n"
        f"# system: {platform.system()} {platform.release()}\n"
        f"# cpu: {platform.processor()}\n"
        f"# python: {platform.python_version()}\n"
        f"# numpy: {np.__version__}\n"
        f"# shape: {SHAPE} (sites={SHAPE[0] * SHAPE[1] * SHAPE[2]})\n"
        f"# n_units: {N_UNITS}\n"
        f"# n_warmup: {N_WARMUP}\n"
        f"# n_profile: {N_PROFILE}\n"
        f"\n"
    )
    log_path.write_text(header + "\n".join(log_lines), encoding="utf-8")
    print(f"[{EXP_ID}] saved text summary: {log_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
