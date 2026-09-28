"""Export the LI -> Lumice analytic parity fixtures (``docs/analytic-parity-fixtures.md``).

One command writes every fixture of the matrix below and ``manifest.json``
into ``--output-dir``; the same rev re-exports byte for byte.  ``--verify``
reads every fixture back and recomputes it with this checkout (non-zero exit
on any failure).  The matrix is path topology (no internal reflection /
internal reflections with TIR-cut arcs / pyramid faces off the prism family)
x point category (``lumice_integral.parity_export.CATEGORIES``); the
selection method of each category is ``parity_export.choose_point``.

    uv run python scripts/export_analytic_parity.py --output-dir artifacts/analytic-parity --verify
    uv run python scripts/export_analytic_parity.py --output-dir /tmp/parity-smoke --cells 3-5__random --verify
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from lumice_integral.canonical_scene import CANONICAL_REFRACTIVE_INDEX, LUMICE_HEIGHT_OVER_DIAMETER, canonical_sun_direction
from lumice_integral.parity_export import (
    CATEGORIES,
    Cell,
    Scene,
    export_matrix,
    miller_wedge_deg,
    prism_crystal,
    pyramid_crystal,
    verify_directory,
)

SUN = tuple(float(x) for x in canonical_sun_direction())
# The ch06 canonical column (canonical_scene.canonical_crystal() is HexPrism.from_lumice(1.0)).
PRISM = prism_crystal(LUMICE_HEIGHT_OVER_DIAMETER)
# tests/test_optics_crystal_native.py::ASYMMETRIC_PYRAMID, its Miller indices given as the wedge angles the
# LUMICE_ANALYTIC_Crystal carries.
PYRAMID = pyramid_crystal(
    0.5, 0.25, 0.6, miller_wedge_deg((1, 0, 1)), miller_wedge_deg((2, 0, 3)), face_distance=(1, 1.1, 0.9, 1, 1.2, 0.95)
)
PYRAMID_SEED_SEARCH_SKIP = (
    "the reference discovery refuses pyramids before discovery: its sample store has no pyramid "
    "(docs/phase1-math-contract.md section 9.5.10, s2_store.crystal_description)"
)

TOPOLOGIES = {
    "3-5": (
        Scene(PRISM, (3, 5), CANONICAL_REFRACTIVE_INDEX, SUN),
        "no internal reflection: the 22 deg halo path on the canonical column (tests/test_discovery.py's production scene)",
        None,
    ),
    "3-5-6-7": (
        Scene(PRISM, (3, 5, 6, 7), CANONICAL_REFRACTIVE_INDEX, SUN),
        "two internal reflections (partial or total, Fresnel-weighted) on the canonical column; its arcs end on exit TIR "
        "or path infeasibility",
        None,
    ),
    "13-15-26-28": (
        Scene(PYRAMID, (13, 15, 26, 28), CANONICAL_REFRACTIVE_INDEX, SUN),
        "pyramid faces off the prism family on the asymmetric pyramid of tests/test_optics_crystal_native.py "
        "(focusing.classify reaches it, roadmap 2026-09-27)",
        PYRAMID_SEED_SEARCH_SKIP,
    ),
}

CATEGORY_RATIONALE = {
    "random": "a generic admissible point: first uniform draw of a fixed-seed generator that passes discovery's gates",
    "critical": "next to an interior extremum of D_P (a caustic): a short loop around a Jacobian-degenerate point",
    "near_boundary": "a pose 1e-3 inside the nearest validity gate (Snell/incidence), plus its mirror image outside",
}


def matrix() -> list[Cell]:
    cells = []
    for scene, topology_rationale, skip in TOPOLOGIES.values():
        for category in CATEGORIES:
            cells.append(
                Cell(
                    scene,
                    category,
                    f"{topology_rationale}; {CATEGORY_RATIONALE[category]}",
                    store_n=None if skip else 100_000,
                    seed_search_skip=skip,
                )
            )
    return cells


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/analytic-parity"))
    parser.add_argument("--cells", nargs="*", help="cell names (<path>__<category>) to export; default: all")
    parser.add_argument("--verify", action="store_true", help="read every fixture back and recompute it")
    args = parser.parse_args(argv)

    cells = matrix()
    if args.cells:
        unknown = set(args.cells) - {cell.name for cell in cells}
        if unknown:
            parser.error(f"unknown cells {sorted(unknown)}; known: {[cell.name for cell in cells]}")
        cells = [cell for cell in cells if cell.name in args.cells]
    manifest = export_matrix(cells, args.output_dir)
    for entry in manifest["cells"]:
        skipped = "; ".join(f"{item['fixture']} skipped: {item['reason']}" for item in entry["skipped"])
        print(f"{entry['name']}: {len(entry['files'])} files" + (f" ({skipped})" if skipped else ""))
    if not args.verify:
        return 0
    checks = verify_directory(args.output_dir)
    failed = [check for check in checks if check.failures]
    for check in failed:
        print(f"FAIL {check.fixture}: " + "; ".join(check.failures))
    print(f"verify: {len(checks) - len(failed)}/{len(checks)} fixtures pass ({sum(c.compared for c in checks)} comparisons)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
