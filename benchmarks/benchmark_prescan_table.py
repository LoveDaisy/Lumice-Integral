"""Benchmark the scene-level prescan table: build, query, pickle transfer.

For each ``N`` in ``--sample-counts`` it reports the build time of
``build_prescan_table`` (sampling + ``optics.path_3_5_domain_batch`` +
kd-tree), the size of the valid subset, the per-query latency of
``candidates(d, 2 deg)`` over a row band of pixel directions (the acceptance
target is <= 5 ms per pixel) and the pickle/unpickle round trip, which is
what every spawned worker pays once when ``strip_driver.render_window``
hands it the table through ``initargs``.

    uv run python benchmarks/benchmark_prescan_table.py
    uv run python benchmarks/benchmark_prescan_table.py --sample-counts 4000000,16000000
"""

from __future__ import annotations

import argparse
import json
import pickle
import platform
import statistics
import time

import jax

from lumice_integral.canonical_scene import CANONICAL_REFRACTIVE_INDEX, CANONICAL_RENDER, canonical_incident_direction
from lumice_integral.prescan import DEFAULT_RNG_SEED, build_prescan_table
from lumice_integral.strip_pixel import pixel_target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-counts", default="400000,1000000,4000000,8000000,16000000")
    parser.add_argument("--batch-size", type=int, default=200_000)
    parser.add_argument("--query-column", type=int, default=150)
    parser.add_argument("--angle-tolerance-deg", type=float, default=2.0)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    incident = canonical_incident_direction()
    targets = [pixel_target(CANONICAL_RENDER, row, args.query_column) for row in range(CANONICAL_RENDER["height"])]
    results = []
    for sample_count in (int(value) for value in args.sample_counts.split(",")):
        build_samples = []
        for _ in range(args.repeats):
            started = time.perf_counter()
            table = build_prescan_table(
                incident, CANONICAL_REFRACTIVE_INDEX, sample_count=sample_count, rng_seed=DEFAULT_RNG_SEED, batch_size=args.batch_size
            )
            build_samples.append(time.perf_counter() - started)

        started = time.perf_counter()
        pool_sizes = [table.candidates(target, args.angle_tolerance_deg).size for target in targets]
        query_seconds = (time.perf_counter() - started) / len(targets)

        started = time.perf_counter()
        blob = pickle.dumps(table)
        pickle_seconds = time.perf_counter() - started
        started = time.perf_counter()
        pickle.loads(blob)
        unpickle_seconds = time.perf_counter() - started

        record = {
            "sample_count": sample_count,
            "valid_count": table.valid_count,
            "build_s_median": statistics.median(build_samples),
            "build_s_min": min(build_samples),
            "query_ms_mean": 1e3 * query_seconds,
            "pool_size_mean": statistics.mean(pool_sizes),
            "pool_size_max": max(pool_sizes),
            "pickle_mb": len(blob) / 1e6,
            "pickle_s": pickle_seconds,
            "unpickle_s": unpickle_seconds,
        }
        results.append(record)
        print(
            f"N={sample_count:>9d} valid={table.valid_count:>8d} build={record['build_s_median']:.2f}s "
            f"query={record['query_ms_mean']:.3f}ms pool(mean/max)={record['pool_size_mean']:.0f}/{record['pool_size_max']} "
            f"pickle={record['pickle_mb']:.0f}MB dump={pickle_seconds:.2f}s load={unpickle_seconds:.2f}s",
            flush=True,
        )
    print(
        json.dumps(
            {
                "hostname": platform.node(),
                "platform": platform.platform(),
                "jax": jax.__version__,
                "jax_backend": jax.default_backend(),
                "query_column": args.query_column,
                "angle_tolerance_deg": args.angle_tolerance_deg,
                "results": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
