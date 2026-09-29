"""Export the LI -> Lumice analytic parity fixtures (``docs/analytic-parity-fixtures.md``).

One command writes every fixture of the matrix below and ``manifest.json``
into ``--output-dir``; the same rev re-exports byte for byte.  ``--verify``
reads every fixture back and recomputes it with this checkout (non-zero exit
on any failure).  The matrix is path topology (no internal reflection /
internal reflections with TIR-cut arcs / pyramid faces off the prism family)
x point category (``lumice_integral.parity_export.CATEGORIES``); the
selection method of each category is ``parity_export.choose_point``.  The
edge cells (``EDGE_CELLS``, wave 2) follow the matrix: one named case each,
chosen for the contract section 11 rows it serves (``docs/phase1-math-contract.md``
section 11.1).

    uv run python scripts/export_analytic_parity.py --output-dir artifacts/analytic-parity --verify
    uv run python scripts/export_analytic_parity.py --output-dir /tmp/parity-smoke --cells 3-5__random --verify
    uv run python scripts/export_analytic_parity.py --output-dir /tmp/parity-edge --cells 3-5__limits --verify
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

from lumice_integral.camera import linear_pixel_outgoing_direction
from lumice_integral.canonical_scene import (
    CANONICAL_REFRACTIVE_INDEX,
    CANONICAL_RENDER,
    LUMICE_HEIGHT_OVER_DIAMETER,
    canonical_incident_direction,
    canonical_sun_direction,
)
from lumice_integral.geometry.pyramid import pyramid_face_angle
from lumice_integral.parity_export import (
    CATEGORIES,
    Cell,
    EdgeCell,
    OptionVariant,
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
        None,
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


# ------------------------------------------------------------------ edge cells (wave 2)
# tests/test_discovery.py::REFERENCE_PYRAMID: regular, prism band 0.5, both caps 0.5 at the ice face angle.
REFERENCE_PYRAMID = pyramid_crystal(0.5, 0.5, 0.5, 90.0 - pyramid_face_angle(), 90.0 - pyramid_face_angle())
SCENE_3_5 = Scene(PRISM, (3, 5), CANONICAL_REFRACTIVE_INDEX, SUN)


def pixel_target(row: int, column: int) -> tuple[float, float, float]:
    """The outgoing direction of a ch06 strip pixel (tests/test_discovery.py::pixel_target)."""
    return tuple(float(x) for x in linear_pixel_outgoing_direction(row, column, **CANONICAL_RENDER))


def target_at_deviation(delta_deg: float) -> tuple[float, float, float]:
    """``delta`` from ``s`` in the vertical plane through it, upward (tests/test_discovery.py::target_at_deviation)."""
    s = canonical_incident_direction()
    across = np.cross(s, np.array([0.0, 0.0, 1.0]))
    up = np.cross(across / np.linalg.norm(across), s)
    delta = math.radians(delta_deg)
    return tuple(float(x) for x in math.cos(delta) * s + math.sin(delta) * up)


# Controller settings around the reference defaults (contract section 10.1: initial steps 0.03-0.08; the
# threshold set of tests/test_reference_core_conformance.py::optical_controller_sweep).
PERTURBATIONS = (
    OptionVariant("initial_step_0.03", "perturbation", (("initial_step", 0.03),)),
    OptionVariant("initial_step_0.08", "perturbation", (("initial_step", 0.08),)),
    OptionVariant(
        "controller_thresholds",
        "perturbation",
        (("minimum_step", 2e-5), ("maximum_step", 0.10), ("shrink_factor", 0.4), ("growth_factor", 1.15), ("maximum_retries", 10)),
    ),
)
# Options that end the 3-5 random trace early, one per terminal reason (probe of task output-level-conformance).
LIMITS = (
    OptionVariant("step_budget", "limit", (("maximum_accepted_steps", 5),)),
    OptionVariant("arclength_budget", "limit", (("maximum_arclength", 0.3),)),
    OptionVariant("evaluation_budget", "limit", (("maximum_evaluations", 15),)),
    OptionVariant("corrector_failure", "limit", (("maximum_advance", 0.01), ("maximum_retries", 0))),
    OptionVariant(
        "step_underflow", "limit", (("initial_step", 0.04), ("minimum_step", 0.03), ("maximum_step", 0.04), ("maximum_advance", 0.01))
    ),
)

EDGE_CELLS = (
    EdgeCell(
        "short_loop",
        SCENE_3_5,
        "critical_offset",
        ("C05", "C06", "C14"),
        "a 0.165 rad loop 0.01 deg above the 3-5 minimum deviation, about four initial steps long: the step-aware "
        "closure must close it on its first traversal (a loop below the 2 x initial_step closure extent is "
        "traversed twice by design, contract section 6.4), and J_perp is smallest here (caustic)",
        critical_offset_deg=0.01,
        variants=PERTURBATIONS,
    ),
    EdgeCell(
        "strip_short_loop_r100_c126",
        SCENE_3_5,
        "target",
        ("C06", "C15"),
        "ch06 strip pixel (100, 126): a 1.645 rad loop shorter than pi that the retired absolute closure gate "
        "traversed twice (test_strip_short_loops_close_at_their_single_traversal_length)",
        target=pixel_target(100, 126),
        seed_search=True,
    ),
    EdgeCell(
        "caustic_loop_r49_c0",
        SCENE_3_5,
        "target",
        ("C06", "C15"),
        "ch06 strip pixel (49, 0): a 0.19 rad loop at the caustic edge whose extra seeds used to end on a spurious "
        "event (test_caustic_edge_pixels_are_single_short_closed_loops)",
        target=pixel_target(49, 0),
        seed_search=True,
    ),
    EdgeCell(
        "boundary_hugging_r700_c150",
        SCENE_3_5,
        "target",
        ("C06", "C08", "C16"),
        "ch06 strip pixel (700, 150): a loop running along the exit TIR boundary (exit Snell discriminant below "
        "event_slowdown_margin) that exhausted the step budget before the rate-based event slowdown",
        target=pixel_target(700, 150),
        seed_search=True,
    ),
    EdgeCell(
        "boundary_hugging_r780_c150",
        SCENE_3_5,
        "target",
        ("C06", "C08", "C16"),
        "ch06 strip pixel (780, 150): the second boundary-hugging loop of test_boundary_hugging_pixels_fold_every_candidate_into_one_closed_loop",
        target=pixel_target(780, 150),
        seed_search=True,
    ),
    EdgeCell(
        "two_arcs_60deg",
        Scene(PRISM, (1, 3), CANONICAL_REFRACTIVE_INDEX, SUN),
        "target",
        ("C06", "C08", "C17", "C18"),
        "path 1-3 at delta = 60 deg: two distinct components, each an arc cut by exit TIR at one end and by the "
        "entry ray leaving face 1 (path_infeasible) at the other (test_path_1_3_at_60_deg_is_two_distinct_components)",
        target=target_at_deviation(60.0),
        seed_search=True,
        variants=PERTURBATIONS[:2],
    ),
    EdgeCell(
        "rank_loss_extremum",
        SCENE_3_5,
        "extremum",
        ("C07",),
        "the seed is the interior minimum of D_P itself, where the fiber degenerates to a point: sigma_2 ~ 1e-16, "
        "far below any rank gate, so the trace must end rank_loss with no accepted pose",
    ),
    EdgeCell(
        "limits",
        SCENE_3_5,
        "random",
        ("C09", "C11"),
        "the 3-5 random seed under options that end the trace early: each budget, a corrector that cannot meet "
        "maximum_advance, and a step that falls below minimum_step",
        variants=LIMITS,
    ),
    EdgeCell(
        "antipodal_target",
        SCENE_3_5,
        "antipodal_target",
        ("C04",),
        "the 3-5 random pose aimed at the antipode -d of its own outgoing direction: an algebraic zero of the "
        "projected residual that the target-neighbourhood gate must reject",
    ),
    EdgeCell(
        "boundary_arc_90deg",
        Scene(REFERENCE_PYRAMID, (13, 24, 26), CANONICAL_REFRACTIVE_INDEX, SUN),
        "target",
        ("C18", "C19"),
        "pyramid 13-24-26 on the reference pyramid at 90 deg: one arc cut by the path domain at both ends, no "
        "interior critical point (test_pyramid_boundary_only_path_is_one_arc_up_to_its_boundary_extremum)",
        target=target_at_deviation(90.0),
        seed_search=True,
    ),
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/analytic-parity"))
    parser.add_argument("--cells", nargs="*", help="cell names (<path>__<category> or <path>__<edge label>) to export; default: all")
    parser.add_argument("--verify", action="store_true", help="read every fixture back and recompute it")
    args = parser.parse_args(argv)

    cells, edge_cells = matrix(), list(EDGE_CELLS)
    if args.cells:
        known = [cell.name for cell in (*cells, *edge_cells)]
        unknown = set(args.cells) - set(known)
        if unknown:
            parser.error(f"unknown cells {sorted(unknown)}; known: {known}")
        cells = [cell for cell in cells if cell.name in args.cells]
        edge_cells = [cell for cell in edge_cells if cell.name in args.cells]
    manifest = export_matrix(cells, args.output_dir, edge_cells)
    for entry in [*manifest["cells"], *manifest.get("edge_cells", [])]:
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
