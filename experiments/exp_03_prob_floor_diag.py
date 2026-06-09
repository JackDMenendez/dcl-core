"""exp_03_prob_floor_diag -- isolate the `prob_floor` knob on `core`.

What this experiment measures
-----------------------------
Sweeps ``CausalSession(prob_floor=...)`` across decades and, for each
value, evolves one fixed wavepacket and records:

- ``max_unity_residual`` -- the worst |sum(p) - 1| over the run.  The
  whole point of the floor's end-of-tick + joint-renormalise ordering
  is that this stays ~1e-16 regardless of the floor; a non-zero value
  here means the floor broke A=1.
- ``engaged`` -- whether the floor actually bit (the floored run's
  probability density diverged from the floor-free run).
- ``support`` -- number of nodes with non-zero probability (the floor
  lifts tail nodes rather than filling vacuum, so this tracks how far
  the packet has spread, not how many nodes the floor created).
- ``rms_radius`` -- sqrt(<r^2>) of the probability density about the
  lattice centre; the observable the downstream Arnold-tongue / orbit
  experiments care about.

This is the ``core`` column of ``dcl-delta-p-min``'s ``exp_09`` in
miniature: it confirms the parameter behaves before the 4-cell grid
consumes it.

Output
------
``data/exp_03_prob_floor_diag.npy``
    Structured array, one row per floor value:
    ``(prob_floor, engaged, max_unity_residual, support, rms_radius)``.
``data/exp_03_prob_floor_diag.log``
    Human-readable table + run metadata.

Configuration
-------------
``SIZE``, ``OMEGA``, ``MOMENTUM``, ``N_TICKS``, ``FLOORS`` sit at the
top of the script.

Status: ACTIVE
"""

from __future__ import annotations

import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from dcl_core.core.CausalSession import CausalSession
from dcl_core.core.OctahedralLattice import OctahedralLattice
from dcl_core.core.UnityConstraint import unity_residual_spinor

SIZE = 15
OMEGA = 0.3
MOMENTUM = (0.3, 0.1, 0.0)
N_TICKS = 30
# None = the floor-free baseline; the rest sweep ~one decade per step.
FLOORS: list[float | None] = [None, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 0.05, 0.25]

_DATA = Path(__file__).resolve().parent.parent / "data"


def _make(prob_floor: float | None) -> CausalSession:
    lat = OctahedralLattice(SIZE, SIZE, SIZE)
    centre = (SIZE // 2, SIZE // 2, SIZE // 2)
    return CausalSession(lat, centre, OMEGA, momentum=MOMENTUM,
                         prob_floor=prob_floor)


def _rms_radius(p: np.ndarray) -> float:
    """sqrt(<r^2>) of the probability density about the lattice centre."""
    nx, ny, nz = p.shape
    cx, cy, cz = (nx - 1) / 2, (ny - 1) / 2, (nz - 1) / 2
    gx, gy, gz = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz),
                             indexing="ij")
    r2 = (gx - cx) ** 2 + (gy - cy) ** 2 + (gz - cz) ** 2
    return float(np.sqrt((p * r2).sum() / p.sum()))


def run() -> np.ndarray:
    baseline_density: np.ndarray | None = None
    rows = []
    dtype = [
        ("prob_floor", "f8"), ("engaged", "?"),
        ("max_unity_residual", "f8"), ("support", "i8"), ("rms_radius", "f8"),
    ]

    for floor in FLOORS:
        s = _make(floor)
        max_res = 0.0
        for _ in range(N_TICKS):
            s.tick()
            max_res = max(max_res, unity_residual_spinor(s.psi_R, s.psi_L))
            s.advance_tick_counter()

        p = s.probability_density()
        if floor is None:
            baseline_density = p
            engaged = False
        else:
            engaged = not np.allclose(p, baseline_density)

        rows.append((
            np.nan if floor is None else floor,
            engaged, max_res, int((p > 0).sum()), _rms_radius(p),
        ))
        tag = "baseline" if floor is None else f"{floor:.0e}"
        print(f"  prob_floor={tag:>9}  engaged={engaged!s:>5}  "
              f"max|A-1|={max_res:.2e}  support={int((p > 0).sum()):>5}  "
              f"rms_r={_rms_radius(p):.4f}")

    return np.array(rows, dtype=dtype)


def main() -> None:
    print(f"exp_03_prob_floor_diag  --  SIZE={SIZE} OMEGA={OMEGA} "
          f"MOMENTUM={MOMENTUM} N_TICKS={N_TICKS}")
    result = run()

    _DATA.mkdir(parents=True, exist_ok=True)
    np.save(_DATA / "exp_03_prob_floor_diag.npy", result)

    stamp = datetime.now(timezone.utc).isoformat()
    lines = [
        f"exp_03_prob_floor_diag  {stamp}",
        f"platform: {platform.platform()}  python: {platform.python_version()}",
        f"SIZE={SIZE} OMEGA={OMEGA} MOMENTUM={MOMENTUM} N_TICKS={N_TICKS}",
        "",
        f"{'prob_floor':>12} {'engaged':>8} {'max|A-1|':>12} "
        f"{'support':>8} {'rms_radius':>11}",
    ]
    for r in result:
        floor_s = "baseline" if np.isnan(r["prob_floor"]) else f"{r['prob_floor']:.0e}"
        lines.append(
            f"{floor_s:>12} {str(bool(r['engaged'])):>8} "
            f"{r['max_unity_residual']:>12.2e} {int(r['support']):>8} "
            f"{r['rms_radius']:>11.4f}"
        )
    # final status line for any audit harness
    worst = float(result["max_unity_residual"].max())
    status = "PASS" if worst < 1e-10 else "FAIL"
    lines.append("")
    lines.append(f"A=1 preserved across all floors: {status} (worst |A-1|={worst:.2e})")
    (_DATA / "exp_03_prob_floor_diag.log").write_text("\n".join(lines) + "\n",
                                                      encoding="utf-8")
    print(f"\nwrote data/exp_03_prob_floor_diag.npy + .log  ({status})")


if __name__ == "__main__":
    main()
