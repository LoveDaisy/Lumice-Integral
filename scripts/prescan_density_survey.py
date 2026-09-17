"""Density evidence for the scene-level prescan table: cluster counts versus ``N``.

Builds one table of ``--max-samples`` Haar poses and, for every ``N`` in the
halving ladder ``max, max/2, ..., --min-samples``, takes its exact prefix
(``PrescanTable.prefix``; the prefix invariant is pinned by
``tests/test_prescan.py``) so all densities come from the same sampling
stream.  For every survey pixel it runs :func:`discover_components` on each
prefix and records the funnel (``pool_count``, ``raw_cluster_count``,
``admissible_count``), the component count, the sorted component arclengths
and the wall-clock time of the discovery.

The stability question the survey answers is: at which ``N`` does doubling
the density stop changing the *result* -- same component count and same
arclength multiset (within ``--arclength-rtol``) on every pixel?  The
smallest such ``N`` is the production default candidate.  Raw cluster counts
are reported too, but they are not the criterion: the geodesic clustering
may split one loop into more clusters as the pool densifies without changing
what is found.

Outputs ``<output-dir>/density_survey.csv`` (one row per pixel and ``N``) and
``<output-dir>/density_survey.md`` (the per-``N`` summary table and the
per-pixel stability matrix, ready to paste into
``docs/ch06-reference-fixture.md``).  Not a pytest case (minutes); run::

    uv run python scripts/prescan_density_survey.py --output-dir artifacts/prescan-density
"""

from __future__ import annotations

import argparse
import csv
import platform
import time
from pathlib import Path

import numpy as np

from lumice_integral.canonical_scene import CANONICAL_RENDER, canonical_crystal
from lumice_integral.discovery import discover_components
from lumice_integral.prescan import DEFAULT_RNG_SEED, build_prescan_table
from lumice_integral.strip_pixel import PixelOptions, canonical_strip_scene, pixel_target

# 32 pixels: the canonical neighbourhood (rows 148-152), the 225/226 pair, the
# lower band (300-650), the boundary-hugging caustic rows (700-800), the dark
# row 40, the former slow closers (49,0)/(50,9), and the same bands on the
# left / centre-left / right columns.
SURVEY_PIXELS: tuple[tuple[int, int], ...] = (
    *((row, 150) for row in (40, 100, 148, 150, 152, 200, 225, 226, 250, 300, 400, 500, 600, 650, 700, 750, 780, 800)),
    (49, 0), (150, 0), (400, 0), (700, 0),
    (50, 9),
    (150, 50), (300, 50), (700, 50),
    (100, 200), (500, 200), (780, 200),
    (150, 250), (400, 250), (700, 250),
)


def ladder(max_samples: int, min_samples: int) -> list[int]:
    counts = []
    n = max_samples
    while n >= min_samples:
        counts.append(n)
        n //= 2
    return sorted(counts)


def same_result(left: dict, right: dict, rtol: float) -> bool:
    if left["component_count"] != right["component_count"]:
        return False
    return all(
        np.isclose(a, b, rtol=rtol, atol=1e-6) for a, b in zip(left["arclengths"], right["arclengths"])
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=16_000_000)
    parser.add_argument("--min-samples", type=int, default=500_000)
    parser.add_argument("--rng-seed", type=int, default=DEFAULT_RNG_SEED)
    parser.add_argument("--arclength-rtol", type=float, default=1e-3)
    args = parser.parse_args(argv)
    counts = ladder(args.max_samples, args.min_samples)
    options = PixelOptions()

    start = time.perf_counter()
    scene = canonical_strip_scene(prescan_sample_count=args.max_samples, prescan_rng_seed=args.rng_seed)
    build_seconds = time.perf_counter() - start
    full = scene.prescan_table
    print(f"table: {full.valid_count} valid of {full.sample_count} in {build_seconds:.1f}s on {platform.node()}", flush=True)
    tables = {n: full.prefix(n) for n in counts}
    crystal = canonical_crystal()

    rows: list[dict] = []
    for row, column in SURVEY_PIXELS:
        target = pixel_target(CANONICAL_RENDER, row, column)
        for n in counts:
            tick = time.perf_counter()
            result = discover_components(
                target, crystal, tables[n], template=scene.discovery_template, **options.discovery_kwargs()
            )
            elapsed = time.perf_counter() - tick
            record = {
                "row": row,
                "column": column,
                "sample_count": n,
                "valid_count": tables[n].valid_count,
                "pool_count": result.pool_count,
                "raw_cluster_count": result.raw_cluster_count,
                "admissible_count": result.admissible_count,
                "component_count": result.component_count,
                "incomplete_count": result.incomplete_count,
                "arclengths": sorted(round(c.arclength, 6) for c in result.components),
                "discovery_s": round(elapsed, 3),
            }
            rows.append(record)
            print(
                f"({row:3d},{column:3d}) N={n:>9d} pool={result.pool_count:5d} clusters={result.raw_cluster_count:3d} "
                f"adm={result.admissible_count:3d} comp={result.component_count} inc={result.incomplete_count} "
                f"arcs={record['arclengths']} {elapsed:.1f}s",
                flush=True,
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "density_survey.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for record in rows:
            writer.writerow({**record, "arclengths": " ".join(f"{a:.6f}" for a in record["arclengths"])})

    by_pixel: dict[tuple[int, int], dict[int, dict]] = {}
    for record in rows:
        by_pixel.setdefault((record["row"], record["column"]), {})[record["sample_count"]] = record
    lines = [
        f"Host `{platform.node()}`; table of {full.sample_count} samples built in {build_seconds:.1f} s "
        f"({full.valid_count} valid, {100.0 * full.valid_count / full.sample_count:.1f} %); "
        f"{len(SURVEY_PIXELS)} pixels; seed {args.rng_seed}.",
        "",
        "| N | valid | pixels changed vs N/2 | clusters (sum) | admissible (sum) | components (sum) | incomplete (sum) | discovery s/pixel (mean) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for index, n in enumerate(counts):
        subset = [by_pixel[pixel][n] for pixel in SURVEY_PIXELS]
        changed = "-"
        if index:
            previous = counts[index - 1]
            changed_pixels = [
                f"({p[0]},{p[1]})" for p in SURVEY_PIXELS if not same_result(by_pixel[p][previous], by_pixel[p][n], args.arclength_rtol)
            ]
            changed = f"{len(changed_pixels)}" + (f" {' '.join(changed_pixels)}" if changed_pixels else "")
        lines.append(
            f"| {n} | {tables[n].valid_count} | {changed} | {sum(r['raw_cluster_count'] for r in subset)} | "
            f"{sum(r['admissible_count'] for r in subset)} | {sum(r['component_count'] for r in subset)} | "
            f"{sum(r['incomplete_count'] for r in subset)} | {np.mean([r['discovery_s'] for r in subset]):.2f} |"
        )
    lines += ["", "Per pixel: `components / clusters` at each N (arclengths in the CSV).", ""]
    lines.append("| pixel | " + " | ".join(str(n) for n in counts) + " |")
    lines.append("|---|" + "---|" * len(counts))
    for pixel in SURVEY_PIXELS:
        cells = [f"{by_pixel[pixel][n]['component_count']}/{by_pixel[pixel][n]['raw_cluster_count']}" for n in counts]
        lines.append(f"| ({pixel[0]},{pixel[1]}) | " + " | ".join(cells) + " |")
    (args.output_dir / "density_survey.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
