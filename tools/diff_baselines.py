"""tools/diff_baselines.py -- Compare two `exp_01_throughput.npy` baselines.

Usage
-----
    python tools/diff_baselines.py BEFORE.npy AFTER.npy [--save OUT.npy]

Loads both throughput baselines, joins on lattice shape, prints a
per-shape speedup table.  Shapes present in only one file are listed
separately so partial OOM runs don't silently drop from the
comparison.

Output
------
Always prints a stdout table.  With ``--save OUT.npy``, also writes a
structured `.npy` with the comparison data so the diff is itself
versionable.

Companion to ``experiments/exp_01_throughput.py``; expects the
structured array dtype that script writes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

DTYPE_COMPARE = np.dtype(
    [
        ("shape", "i8", (3,)),
        ("n_sites", "i8"),
        ("before_mean_s", "f8"),
        ("after_mean_s", "f8"),
        ("speedup_ratio", "f8"),
    ]
)


def format_seconds(seconds: float) -> str:
    if seconds < 1e-3:
        return f"{seconds * 1e6:.2f} us"
    if seconds < 1.0:
        return f"{seconds * 1e3:.3f} ms"
    return f"{seconds:.3f} s"


def load_baseline(path: str) -> tuple[np.ndarray, dict | None]:
    """Load a baseline `.npy` and its `.json` sidecar (if present)."""
    p = Path(path)
    arr = np.load(p)
    sidecar_path = p.with_suffix(".json")
    sidecar = None
    if sidecar_path.exists():
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    return arr, sidecar


def diff_baselines(
    before_arr: np.ndarray, after_arr: np.ndarray
) -> tuple[list[dict], list[tuple[int, int, int]], list[tuple[int, int, int]]]:
    """Match records by lattice shape; return (matches, only_before, only_after)."""
    before_by_shape = {tuple(int(v) for v in rec["shape"]): rec for rec in before_arr}
    after_by_shape = {tuple(int(v) for v in rec["shape"]): rec for rec in after_arr}

    matched = sorted(set(before_by_shape) & set(after_by_shape))
    only_before = sorted(set(before_by_shape) - set(after_by_shape))
    only_after = sorted(set(after_by_shape) - set(before_by_shape))

    matches = []
    for shape in matched:
        b = before_by_shape[shape]
        a = after_by_shape[shape]
        before_t = float(b["tick_mean_s"])
        after_t = float(a["tick_mean_s"])
        ratio = before_t / after_t if after_t > 0 else float("inf")
        matches.append(
            {
                "shape": shape,
                "n_sites": int(b["n_sites"]),
                "before_mean_s": before_t,
                "after_mean_s": after_t,
                "speedup_ratio": ratio,
            }
        )
    return matches, only_before, only_after


def _host_line(meta: dict | None, label: str) -> str:
    if meta is None:
        return f"{label}: <no JSON sidecar>"
    host = meta.get("host", {})
    return (
        f"{label}: {host.get('node', '?')} "
        f"| {host.get('processor') or host.get('machine', '?')} "
        f"| numpy {meta.get('numpy', '?')}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Diff two exp_01_throughput.npy baselines."
    )
    parser.add_argument("before", help="Path to BEFORE.npy")
    parser.add_argument("after", help="Path to AFTER.npy")
    parser.add_argument(
        "--save",
        default=None,
        help="Optional path to write the structured comparison .npy",
    )
    args = parser.parse_args(argv)

    before_arr, before_meta = load_baseline(args.before)
    after_arr, after_meta = load_baseline(args.after)

    print(_host_line(before_meta, "BEFORE"))
    print(_host_line(after_meta, "AFTER "))
    print()

    matches, only_before, only_after = diff_baselines(before_arr, after_arr)

    header = (
        f"{'shape':<18} {'sites':>10}  {'before':>14}  {'after':>14}  "
        f"{'speedup':>10}"
    )
    print(header)
    print("-" * len(header))
    for m in matches:
        shape_str = str(m["shape"])
        print(
            f"{shape_str:<18} {m['n_sites']:>10}  "
            f"{format_seconds(m['before_mean_s']):>14}  "
            f"{format_seconds(m['after_mean_s']):>14}  "
            f"{m['speedup_ratio']:>9.2f}x"
        )

    if only_before:
        print()
        print(f"Shapes only in BEFORE: {[str(s) for s in only_before]}")
    if only_after:
        print()
        print(f"Shapes only in AFTER:  {[str(s) for s in only_after]}")

    if args.save:
        out_arr = np.array(
            [
                (
                    np.array(m["shape"], dtype=np.int64),
                    m["n_sites"],
                    m["before_mean_s"],
                    m["after_mean_s"],
                    m["speedup_ratio"],
                )
                for m in matches
            ],
            dtype=DTYPE_COMPARE,
        )
        out_path = Path(args.save)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, out_arr)
        print(f"\nSaved comparison: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
