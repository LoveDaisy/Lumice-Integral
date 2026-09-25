"""Cost of extracting certified level sets ``{D_P = delta}`` for a strip's worth of deviations (``lumice_integral.contour``).

Builds the ``DPField`` and an in-memory event store of one path, then times
:func:`lumice_integral.contour.extract_level_sets` on ``--deltas`` equally
spaced deviations strictly inside ``(min D_P, max D_P)`` (endpoints and
critical values excluded), twice: the first call includes the XLA
compilations, the second is the steady state.  The number of deviations is a
stand-in for what a strip needs (its row count: 161 for the band-sum layout,
801 for ch06); the exact pixel-to-``delta`` map belongs to the line-integral
stage.  Reports wall times, node counts and the process peak RSS.

Usage (Mac laptop):

    uv run python benchmarks/benchmark_contour_extraction.py --deltas 161
    uv run python benchmarks/benchmark_contour_extraction.py --deltas 801 --store-n 1000000 --output /tmp/contour.json
"""

from __future__ import annotations

import argparse
import json
import platform
import time

import numpy as np

from lumice_integral import contour
from lumice_integral.canonical_scene import canonical_crystal
from lumice_integral.dp_field import DPField
from lumice_integral.s2_store import build_event_store, max_rss_mb


def _deltas(field: DPField, count: int) -> np.ndarray:
    partition = field.interval_partition()
    lo, hi = partition[0].lower, partition[-1].upper
    deltas = np.linspace(lo, hi, count + 2)[1:-1]
    breaks = np.array([lo] + [interval.upper for interval in partition])
    # keep clear of the critical values (extraction refuses a delta at one)
    near = np.min(np.abs(deltas[:, None] - breaks[None, :]), axis=1) <= 1e-6
    deltas[near] += 2e-6
    return deltas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--faces", type=int, nargs="+", default=[3, 5])
    parser.add_argument("--deltas", type=int, default=161)
    parser.add_argument("--store-n", type=int, default=1_000_000)
    parser.add_argument("--grid", type=int, default=contour.DEFAULT_GRID)
    parser.add_argument("--refractive-index", type=float, default=1.31)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    faces = tuple(args.faces)
    crystal = canonical_crystal()

    t0 = time.perf_counter()
    field = DPField.build(crystal, faces, args.refractive_index)
    field.interval_partition()
    t_field = time.perf_counter() - t0
    t0 = time.perf_counter()
    store = build_event_store(crystal, args.refractive_index, [faces], args.store_n, run_checks=False)
    t_store = time.perf_counter() - t0
    deltas = _deltas(field, args.deltas)

    timings = []
    for _ in range(2):
        t0 = time.perf_counter()
        level_sets = contour.extract_level_sets(field, deltas, store, grid=args.grid)
        timings.append(time.perf_counter() - t0)
    components = [c for ls in level_sets for c in ls.components]
    nodes = np.array([len(c.points) for c in components])
    report = {
        "faces": list(faces),
        "deltas": int(len(deltas)),
        "store_n": args.store_n,
        "store_events": int(len(store.events.D)),
        "grid": args.grid,
        "field_and_partition_s": round(t_field, 2),
        "store_build_s": round(t_store, 2),
        "extract_first_call_s": round(timings[0], 2),
        "extract_steady_s": round(timings[1], 2),
        "extract_steady_ms_per_delta": round(1e3 * timings[1] / len(deltas), 2),
        "components": int(len(components)),
        "closed": int(sum(c.closed for c in components)),
        "open": int(sum(not c.closed for c in components)),
        "nodes_total": int(nodes.sum()),
        "nodes_per_component_median": float(np.median(nodes)),
        "nodes_per_component_max": int(nodes.max()),
        "max_rss_mb": round(max_rss_mb(), 1),
        "machine": f"{platform.system()} {platform.machine()} {platform.processor()}",
    }
    print(json.dumps(report, indent=2))
    if args.output:
        with open(args.output, "w") as handle:
            json.dump(report, handle, indent=2)


if __name__ == "__main__":
    main()
