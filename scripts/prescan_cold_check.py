"""Independent cold check of one pixel's component discovery with a denser prescan.

Builds a throw-away table of ``--samples`` Haar poses (default four times the
production ``prescan.DEFAULT_SAMPLE_COUNT``, never cached) and runs
:func:`discover_components` on pixel ``(--row, --column)`` with the
production discovery options, printing the funnel, the component count and
the arclengths.  Meant for spot checks against a rendered strip's
``pixels.csv`` (``task-strip-rerender-and-compare``): a component the
production table missed but this denser one finds is a density problem, a
disagreement in arclength is not.  Off the hot path by construction: it
reuses the same ``build_prescan_table`` / ``discover_components`` the
production path uses, only with a different ``N``.

    uv run python scripts/prescan_cold_check.py --row 150 --column 150
    uv run python scripts/prescan_cold_check.py --row 700 --column 150 --samples 32000000
"""

from __future__ import annotations

import argparse
import time

from lumice_integral.canonical_scene import CANONICAL_RENDER, canonical_crystal
from lumice_integral.discovery import discover_components
from lumice_integral.prescan import DEFAULT_RNG_SEED, DEFAULT_SAMPLE_COUNT
from lumice_integral.strip_pixel import PixelOptions, canonical_strip_scene, pixel_target


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--row", type=int, required=True)
    parser.add_argument("--column", type=int, required=True)
    parser.add_argument("--samples", type=int, default=4 * DEFAULT_SAMPLE_COUNT)
    parser.add_argument("--rng-seed", type=int, default=DEFAULT_RNG_SEED + 1, help="deliberately not the production seed")
    parser.add_argument("--angle-tolerance-deg", type=float, default=PixelOptions.angle_tolerance_deg)
    args = parser.parse_args(argv)
    if not (0 <= args.row < CANONICAL_RENDER["height"] and 0 <= args.column < CANONICAL_RENDER["width"]):
        parser.error(f"pixel must lie inside {CANONICAL_RENDER['height']} x {CANONICAL_RENDER['width']}")

    start = time.perf_counter()
    scene = canonical_strip_scene(prescan_sample_count=args.samples, prescan_rng_seed=args.rng_seed)
    table = scene.prescan_table
    print(f"table: {table.valid_count} valid of {table.sample_count} (seed {table.rng_seed}) in {time.perf_counter() - start:.1f}s")
    options = PixelOptions(angle_tolerance_deg=args.angle_tolerance_deg)
    start = time.perf_counter()
    result = discover_components(
        pixel_target(CANONICAL_RENDER, args.row, args.column),
        canonical_crystal(),
        table,
        template=scene.discovery_template,
        **options.discovery_kwargs(),
    )
    elapsed = time.perf_counter() - start
    print(
        f"pixel ({args.row},{args.column}): pool={result.pool_count} clusters={result.raw_cluster_count} "
        f"admissible={result.admissible_count} components={result.component_count} "
        f"incomplete={result.incomplete_count} completeness={result.completeness} ({elapsed:.1f}s)"
    )
    for index, component in enumerate(result.components):
        print(f"  component {index}: arclength={component.arclength:.6f} poses={len(component.result.poses)} reason={component.reason.value}")
    for index, candidate in enumerate(result.incomplete):
        print(f"  incomplete {index}: status={candidate.status.value} reason={candidate.reason.value} poses={len(candidate.result.poses)}")


if __name__ == "__main__":
    main()
