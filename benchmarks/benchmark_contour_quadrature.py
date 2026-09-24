"""Cost structure of contour quadrature: curve finding per ``delta`` vs line integration per pixel (``lumice_integral.contour_quadrature``).

The cost model (2026-09-24, design estimate): finding the curves scales with
the number of distinct deviations (``delta`` rings), integrating along them
with the number of pixels.  This measures both on the canonical 3-5 scene:

- a strip column's ``--deltas`` deviations (rows of column 150 from row 100): the wall
  time of :func:`.contour.extract_level_sets` (curve finding) and of
  :meth:`.contour_quadrature.LevelSetGeometry.build` (the pixel-independent
  geometry: points, weights, ``|grad D|``, arclength speed), per ``delta``;
- :meth:`.contour_quadrature.LevelSetGeometry.integrate` with ``J`` pixels
  per ``delta`` (``--pixels-per-delta``: the centre azimuths spread over
  ``+-3 deg`` of the column's, as pixels of one ring are), per pixel, for
  the canonical column density and the random family.

Each stage runs once to compile and is then timed.  Every number is the
steady state of one process.

Usage (Mac laptop)::

    uv run python benchmarks/benchmark_contour_quadrature.py
    uv run python benchmarks/benchmark_contour_quadrature.py --deltas 801 --output /tmp/contour-quadrature.json
"""

from __future__ import annotations

import argparse
import json
import platform
import time

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402

from lumice_integral.band_sum import pixel_band  # noqa: E402
from lumice_integral.camera import incident_direction_from_sun  # noqa: E402
from lumice_integral.canonical_scene import (  # noqa: E402
    CANONICAL_REFRACTIVE_INDEX,
    canonical_crystal,
    canonical_pose_density,
    canonical_sun_direction,
)
from lumice_integral.contour import extract_level_sets  # noqa: E402
from lumice_integral.contour_quadrature import GEOMETRY_CHUNK, LevelSetGeometry, QuadratureOptions  # noqa: E402
from lumice_integral.dp_field import DPField  # noqa: E402
from lumice_integral.pose_density import build_pose_density  # noqa: E402
from lumice_integral.s2_store import build_event_store, max_rss_mb  # noqa: E402


def _rotated(centre: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    """``centre`` turned by ``angle`` about ``axis`` (Rodrigues): same deviation from ``axis``, another azimuth."""
    return centre * np.cos(angle) + np.cross(axis, centre) * np.sin(angle) + axis * (axis @ centre) * (1.0 - np.cos(angle))


def _timed(function):
    start = time.perf_counter()
    out = function()
    return out, time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--column", type=int, default=150)
    parser.add_argument("--first-row", type=int, default=100, help="first row of the column (100: inside the lit band)")
    parser.add_argument("--deltas", type=int, default=256, help="consecutive rows of the column from --first-row")
    parser.add_argument("--pixels-per-delta", type=int, nargs="+", default=[1, 4, 16])
    parser.add_argument("--store-n", type=int, default=1_000_000)
    parser.add_argument("--relative-tolerance", type=float, default=QuadratureOptions().relative_tolerance)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    crystal = canonical_crystal()
    sun = canonical_sun_direction()
    s = incident_direction_from_sun(sun)
    options = QuadratureOptions(relative_tolerance=args.relative_tolerance)
    field = DPField.build(crystal, (3, 5), CANONICAL_REFRACTIVE_INDEX)
    field.interval_partition()
    store = build_event_store(crystal, CANONICAL_REFRACTIVE_INDEX, [(3, 5)], args.store_n, run_checks=False)
    bands = [pixel_band(row, args.column, sun) for row in range(args.first_row, args.first_row + args.deltas)]
    deltas = np.array([b[1] for b in bands])
    centres = np.array([b[0] for b in bands])

    extract_level_sets(field, deltas, store)  # compile
    level_sets, extract_s = _timed(lambda: extract_level_sets(field, deltas, store))
    lit = [k for k, ls in enumerate(level_sets) if ls.components]
    level_sets = [level_sets[k] for k in lit]
    centres = centres[lit]

    def build_all():
        return [LevelSetGeometry.build(field, level_sets[i:i + GEOMETRY_CHUNK], options) for i in range(0, len(level_sets), GEOMETRY_CHUNK)]

    build_all()  # compile
    geometries, geometry_s = _timed(build_all)
    points = int(sum(int(g.evaluations.sum()) for g in geometries))

    densities = {"column (canonical)": canonical_pose_density(), "random": build_pose_density("random")}
    integration = {}
    for name, density in densities.items():
        per_j = {}
        for j in args.pixels_per_delta:
            angles = np.linspace(-np.radians(3.0), np.radians(3.0), j) if j > 1 else np.zeros(1)

            def integrate_all():
                out = []
                for chunk, geometry in enumerate(geometries):
                    base = chunk * GEOMETRY_CHUNK
                    mine = centres[base:base + len(geometry.level_sets)]
                    jobs = [(k, _rotated(c, s, a)) for k, c in enumerate(mine) for a in angles]
                    out += geometry.integrate(sun, [c for _, c in jobs], density, [k for k, _ in jobs])
                return out

            integrate_all()  # compile
            results, seconds = _timed(integrate_all)
            per_j[j] = {
                "pixels": len(results),
                "integrate_s": round(seconds, 3),
                "ms_per_pixel": round(1e3 * seconds / len(results), 3),
                "added_points_per_pixel_mean": round(float(np.mean([r.evaluations for r in results])), 1),
            }
        integration[name] = per_j

    report = {
        "scene": "canonical ch06, path 3-5",
        "column": args.column,
        "rows": [args.first_row, args.first_row + args.deltas],
        "deltas_lit": len(level_sets),
        "relative_tolerance": args.relative_tolerance,
        "curve_finding": {
            "extract_s": round(extract_s, 3),
            "ms_per_delta": round(1e3 * extract_s / args.deltas, 3),
            "nodes_total": int(sum(len(c.points) for ls in level_sets for c in ls.components)),
        },
        "geometry": {
            "build_s": round(geometry_s, 3),
            "ms_per_delta": round(1e3 * geometry_s / len(level_sets), 3),
            "points_per_delta_mean": round(points / len(level_sets), 1),
        },
        "integration": integration,
        "max_rss_mb": round(max_rss_mb(), 1),
        "machine": f"{platform.system()} {platform.machine()} {platform.processor()}",
    }
    print(json.dumps(report, indent=2))
    if args.output:
        with open(args.output, "w") as handle:
            json.dump(report, handle, indent=2)


if __name__ == "__main__":
    main()
