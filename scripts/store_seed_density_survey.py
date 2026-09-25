"""Density evidence for the Phase I seed store: discovery results versus store ``N`` and band half-width.

Successor of the retired ``prescan_density_survey.py`` (same 32 strip pixels,
:data:`SURVEY_PIXELS`).  For every pixel and every ``(N, half-width)`` of the
ladder it runs :func:`discover_components` on the ``3-5`` seed store of ``N``
points (built or loaded with ``s2_store.build_or_load`` under
``--store-cache-dir``) and records the funnel (``pool_count``,
``raw_cluster_count``, ``admissible_count``), the component count, the
component kinds and sorted arclengths, and the wall-clock time.  A
configuration *agrees* on a pixel when count, kinds and arclengths (within
``--arclength-rtol``) equal those of the reference configuration (the largest
``N`` at the production half-width, :data:`PRODUCTION_HALF_WIDTH_DEG`).

On the production configuration it also runs the completeness cross-check
:func:`discovery.check_band_coverage` (every band event revisited; suspects are
admissible fiber poses far from every traced curve) and reports the suspects
and the miss-probability bound ``exp(-k_min)`` per pixel.

Recorded in task ``phase1-seeds-from-store`` against the 4M Haar prescan table
it replaced: on these 32 pixels every store configuration from ``N = 1e5`` /
0.02 deg to ``N = 1e8`` / 2 deg found the prescan's components (count and
kinds everywhere; arclengths within 1.5e-3 on the two 0.17-0.19 rad caustic
loops, 1.8e-4 elsewhere).  Outputs ``<output-dir>/store_seed_density_survey.csv``
and ``.md``.  Not a pytest case (minutes; the ``N = 1e8`` store is ~1 GB)::

    uv run python scripts/store_seed_density_survey.py --output-dir /tmp/store-seed-density
"""

from __future__ import annotations

import argparse
import csv
import platform
import time
from pathlib import Path

import numpy as np

from lumice_integral.canonical_scene import CANONICAL_REFRACTIVE_INDEX, CANONICAL_RENDER, canonical_crystal, canonical_sun_direction
from lumice_integral.discovery import check_band_coverage, discover_components
from lumice_integral.optics import PATH_3_5_FACES
from lumice_integral.s2_store import DEFAULT_CACHE_DIR, DEFAULT_SEED_STORE_N, StoreSeeds, build_or_load
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
PRODUCTION_HALF_WIDTH_DEG = PixelOptions().band_half_width_deg


def summary(result) -> dict:
    ordered = sorted(result.components, key=lambda component: component.arclength)
    return {
        "pool_count": result.pool_count,
        "raw_cluster_count": result.raw_cluster_count,
        "admissible_count": result.admissible_count,
        "component_count": result.component_count,
        "incomplete_count": result.incomplete_count,
        "kinds": [component.kind for component in ordered],
        "arclengths": [round(component.arclength, 6) for component in ordered],
    }


def agrees(left: dict, right: dict, rtol: float) -> bool:
    return (
        left["component_count"] == right["component_count"]
        and left["kinds"] == right["kinds"]
        and all(np.isclose(a, b, rtol=rtol, atol=1e-6) for a, b in zip(left["arclengths"], right["arclengths"]))
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--store-n", type=int, nargs="+", default=[100_000, DEFAULT_SEED_STORE_N, 10_000_000, 100_000_000])
    parser.add_argument("--half-width-deg", type=float, nargs="+", default=[2.0, PRODUCTION_HALF_WIDTH_DEG, 0.02])
    parser.add_argument("--store-cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--arclength-rtol", type=float, default=2e-3)
    args = parser.parse_args(argv)
    crystal, sun, options = canonical_crystal(), canonical_sun_direction(), PixelOptions()
    template = canonical_strip_scene(seed_store_n=1_000).discovery_template
    seeds = {
        n: StoreSeeds(
            build_or_load(crystal, CANONICAL_REFRACTIVE_INDEX, [PATH_3_5_FACES], n, base_dir=args.store_cache_dir, mmap_mode="r"),
            PATH_3_5_FACES,
            sun,
        )
        for n in args.store_n
    }
    reference = (max(args.store_n), PRODUCTION_HALF_WIDTH_DEG)
    configs = [reference] + [(n, h) for n in args.store_n for h in args.half_width_deg if (n, h) != reference]
    production = (DEFAULT_SEED_STORE_N, PRODUCTION_HALF_WIDTH_DEG)

    rows: list[dict] = []
    for row, column in SURVEY_PIXELS:
        target = pixel_target(CANONICAL_RENDER, row, column)
        baseline = None
        for n, half_width in configs:
            kwargs = {**options.discovery_kwargs(), "band_half_width_deg": half_width}
            tick = time.perf_counter()
            result = discover_components(target, seeds[n], template=template, **kwargs)
            record = {"row": row, "column": column, "N": n, "half_width_deg": half_width, **summary(result)}
            record["discovery_s"] = round(time.perf_counter() - tick, 3)
            baseline = baseline or record
            record["agrees"] = agrees(baseline, record, args.arclength_rtol)
            if (n, half_width) == production:
                coverage = check_band_coverage(target, seeds[n], result, band_half_width_deg=half_width, template=template)
                record["coverage_suspects"] = coverage.suspect_count
                record["coverage_corrected"] = coverage.corrected_count
                record["coverage_k_min"] = min(coverage.component_event_counts, default=None)
            rows.append(record)
            print(
                f"({row:3d},{column:3d}) N={n:>9d} h={half_width:5.2f} pool={record['pool_count']:7d} "
                f"clusters={record['raw_cluster_count']:3d} comp={record['component_count']} inc={record['incomplete_count']} "
                f"{record['kinds']} {record['arclengths']} {'ok' if record['agrees'] else 'DIFF'} {record['discovery_s']}s",
                flush=True,
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for record in rows for key in record))
    with (args.output_dir / "store_seed_density_survey.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in rows:
            writer.writerow({**record, "kinds": " ".join(record["kinds"]), "arclengths": " ".join(f"{a:.6f}" for a in record["arclengths"])})
    lines = [
        f"Host `{platform.node()}`; {len(SURVEY_PIXELS)} pixels; reference N={reference[0]}, {reference[1]} deg; "
        f"agreement = component count, kinds, arclengths rtol {args.arclength_rtol}.",
        "",
        "| N | half-width deg | pixels differing | pool (median) | pool (max) | components (sum) | incomplete (sum) | s/pixel (mean) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for n, half_width in configs:
        subset = [r for r in rows if (r["N"], r["half_width_deg"]) == (n, half_width)]
        differing = [f"({r['row']},{r['column']})" for r in subset if not r["agrees"]]
        lines.append(
            f"| {n} | {half_width} | {len(differing)} {' '.join(differing)} | {int(np.median([r['pool_count'] for r in subset]))} | "
            f"{max(r['pool_count'] for r in subset)} | {sum(r['component_count'] for r in subset)} | "
            f"{sum(r['incomplete_count'] for r in subset)} | {np.mean([r['discovery_s'] for r in subset]):.2f} |"
        )
    checked = [r for r in rows if "coverage_suspects" in r]
    k_min = [r["coverage_k_min"] for r in checked if r["coverage_k_min"] is not None]
    lines += [
        "",
        f"Completeness cross-check on N={production[0]}, {production[1]} deg: {sum(r['coverage_suspects'] for r in checked)} suspects "
        f"on {len(checked)} pixels ({sum(r['coverage_corrected'] for r in checked)} events Newton-corrected); smallest per-component "
        f"event count k_min = {min(k_min) if k_min else None} (miss-probability bound exp(-k_min)).",
    ]
    (args.output_dir / "store_seed_density_survey.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
