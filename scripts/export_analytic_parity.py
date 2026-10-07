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
section 11.1).  The module B cells (``BAND_SUM_CELLS``, ``docs/band-sum-contract.md``) come last: one
``band_sum`` fixture each, a path, a pose density and a small pixel table.

    uv run python scripts/export_analytic_parity.py --output-dir artifacts/analytic-parity --verify
    uv run python scripts/export_analytic_parity.py --output-dir /tmp/parity-smoke --cells 3-5__random --verify
    uv run python scripts/export_analytic_parity.py --output-dir /tmp/parity-edge --cells 3-5__limits --verify
    uv run python scripts/export_analytic_parity.py --output-dir /tmp/parity-band --cells 3-5__band_sum_plate --verify
    uv run python scripts/export_analytic_parity.py --output-dir /tmp/parity-pinned --cells 3-6-4-8__band_sum_plate --verify
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

from lumice_integral.camera import linear_pixel_outgoing_direction, sun_direction
from lumice_integral.canonical_scene import (
    CANONICAL_REFRACTIVE_INDEX,
    CANONICAL_RENDER,
    CANONICAL_SUN_ALTITUDE_DEG,
    CANONICAL_SUN_AZIMUTH_DEG,
    LUMICE_HEIGHT_OVER_DIAMETER,
    canonical_incident_direction,
    canonical_sun_direction,
)
from lumice_integral.geometry.pyramid import pyramid_face_angle
from lumice_integral.parity_export import (
    CATEGORIES,
    BandSumCell,
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


# ------------------------------------------------------------------ band-sum cells (module B, docs/band-sum-contract.md)
# A linear view with the sun at pixel (20, 20) (about 1.5 deg per pixel), and Lambert views (the Analyze
# projection, docs/band-sum-contract.md section 3.3) centred on the sun and on the antisun; each cell's pixels
# were picked from a scan of that view: lit pixels with K of tens to hundreds, a dark one, and the pixel that
# contains s or -s (singular) or, for a rank-0 path, the sun (point mass).
LINEAR_VIEW = {"kind": "linear", "render": {"width": 41, "height": 41, "fov_deg": 60.0, "view": {"azimuth": 0.0, "elevation": 15.0}}}


def lambert(centre_sky, field_radius_deg: float, size: int) -> dict:
    return {"kind": "lambert_azimuthal_equal_area", "centre_sky": [float(x) for x in centre_sky], "field_radius_deg": field_radius_deg, "size": size}


SUN_VIEW = lambert(SUN, 30.0, 65)  # about 0.93 deg per pixel, the sun at the centre of pixel (32, 32)
ANTISUN_VIEW = lambert([-x for x in SUN], 60.0, 65)  # about 1.8 deg per pixel, the antisun at the centre of (32, 32)
SUN_CLOSE_VIEW = lambert(SUN, 5.0, 9)  # about 1.1 deg per pixel, the sun at the centre of pixel (4, 4)
SCENE_3_6 = Scene(PRISM, (3, 6), CANONICAL_REFRACTIVE_INDEX, SUN)
# The family-pinned cells: Lambert views like SUN_CLOSE_VIEW centred on the spot's sky point.  1-2-1 and
# 1-2-3-4-1 need a thin plate: on the column the ray leaves through a prism face after the basal reflection, so
# the plate family (c axis vertical) has no valid pose there (w = 0 on the whole sigma = 0 ring).
PLATE_CRYSTAL = prism_crystal(0.2)
SCENE_3_6_4_8 = Scene(PRISM, (3, 6, 4, 8), CANONICAL_REFRACTIVE_INDEX, SUN)
PARHELION_120_VIEW = lambert(sun_direction(CANONICAL_SUN_ALTITUDE_DEG, CANONICAL_SUN_AZIMUTH_DEG + 120.0), 5.0, 9)
SUBSUN_VIEW = lambert(sun_direction(-CANONICAL_SUN_ALTITUDE_DEG, CANONICAL_SUN_AZIMUTH_DEG), 5.0, 9)
SUBPARHELION_120_VIEW = lambert(sun_direction(-CANONICAL_SUN_ALTITUDE_DEG, CANONICAL_SUN_AZIMUTH_DEG + 120.0), 5.0, 9)
PINNED_SPOT_PIXELS = ((4, 4), (3, 4), (5, 4), (4, 3), (4, 5), (0, 4), (8, 4), (4, 1), (4, 7), (4, 0))
# A refractive index other than the canonical 1.31 (task entry-measure-exit-gate-index): the grazing slab path
# 1-3-4-2 (in through one basal face, two prism reflections, out through the parallel basal face) on a plate
# with two long faces, the sun on the horizon.  The internal ray sits just inside the critical cone of n = 1.307,
# outside that of 1.31; an exit gate at the package index 1.31 zeroes every event.  The spot (azimuth 120 deg,
# elevation -0.5 to 2.6 deg) does not depend on n: parallel entry and exit faces.
SLAB_CRYSTAL = prism_crystal(0.3, (1.5, 1.0, 1.0, 1.5, 1.0, 1.0))
HORIZON_SUN = tuple(float(x) for x in sun_direction(0.0, CANONICAL_SUN_AZIMUTH_DEG))
HORIZON_PARHELION_120_VIEW = lambert(sun_direction(0.0, CANONICAL_SUN_AZIMUTH_DEG + 120.0), 5.0, 9)

BAND_SUM_CELLS = (
    BandSumCell(
        "random",
        SCENE_3_5,
        {"family": "random"},
        LINEAR_VIEW,
        ((20, 20), (20, 12), (7, 20), (6, 20), (5, 20), (4, 20), (3, 33)),
        20_000,
        "no internal reflection, Haar-uniform poses, linear lens: the sun pixel (singular), a pixel inside the "
        "22 deg halo (empty band) and the halo's inner edge and tail",
    ),
    BandSumCell(
        "plate",
        SCENE_3_5,
        {"family": "plate", "zenith_std_deg": 1.0},
        SUN_VIEW,
        ((32, 32), (30, 8), (31, 8), (31, 6), (29, 7), (33, 8), (26, 9), (26, 10), (32, 57), (32, 20)),
        20_000,
        "a narrow zenith family (plate, 1 deg) on a Lambert view: the parhelion of the single path 3-5 (one side "
        "only), its tails down to 1e-88, the mirror side where every band event has rho = 0 (K_rho_pos = 0 < K), an "
        "empty band inside the halo, and the sun pixel",
    ),
    BandSumCell(
        "parry",
        SCENE_3_5,
        {"family": "parry", "zenith_std_deg": 1.0, "roll_std_deg": 1.0},
        LINEAR_VIEW,
        ((20, 20), (0, 12), (0, 20), (0, 28), (1, 12), (1, 26), (2, 20), (5, 20), (8, 13), (9, 20)),
        20_000,
        "the roll-locked family (Parry, zenith and roll 1 deg): the density reads all three body-axis zenith "
        "components (the roll atan2(-e2, e1)); the upper Parry arc, tails to 1e-238 and a dark pixel",
    ),
    BandSumCell(
        "random",
        Scene(PRISM, (3, 5, 6, 7), CANONICAL_REFRACTIVE_INDEX, SUN),
        {"family": "random"},
        ANTISUN_VIEW,
        ((32, 32), (32, 14), (20, 20), (14, 26), (26, 20), (2, 32), (8, 8), (26, 26)),
        50_000,
        "two internal reflections weighted by Fresnel R on a Lambert view about the antisun: the antisolar pixel "
        "(delta = pi, singular), the lit ring and a dark pixel beyond the path's largest deviation",
    ),
    BandSumCell(
        "random",
        Scene(PYRAMID, (13, 15, 26, 28), CANONICAL_REFRACTIVE_INDEX, SUN),
        {"family": "random"},
        ANTISUN_VIEW,
        ((32, 32), (32, 14), (14, 20), (8, 26), (20, 20), (14, 14), (2, 2), (26, 26)),
        50_000,
        "pyramid faces off the prism family on the asymmetric pyramid (the matrix's crystal): the lit ring "
        "(D 121-149 deg), a pixel on each side of it and the antisolar pixel",
    ),
    BandSumCell(
        "rank0",
        SCENE_3_6,
        {"family": "random"},
        SUN_CLOSE_VIEW,
        ((4, 4), (4, 5), (3, 3), (0, 0)),
        20_000,
        "a rank-0 path (3-6, parallel faces): a point mass m on the pixel containing the sun, 0 elsewhere; under "
        "the random density m is the lattice mean of w, deterministic",
    ),
    BandSumCell(
        "rank0_plate",
        SCENE_3_6,
        {"family": "plate", "zenith_std_deg": 1.0},
        SUN_CLOSE_VIEW,
        ((4, 4), (4, 5)),
        20_000,
        "the same point mass under the plate family: m from LI's Haar stream, compared statistically",
        rank0_sample_count=4_000_000,
    ),
    # Family-pinned paths (focusing.family_pinned: rank 2, wedge 0, fold matrix commuting with the rotations
    # about the family's sigma -> 0 support axis; a plate or Lowitz density pins body z, a Parry density body x --
    # task family-pinned-parry-axis; the cells below are all plate/Lowitz): the sigma -> 0 plate family lies in
    # one level set of D_P, so it lands on one sky point and
    # the spot narrows with sigma.  3-5__band_sum_plate above is the control: the same family, not pinned.
    BandSumCell(
        "plate",
        SCENE_3_6_4_8,
        {"family": "plate", "zenith_std_deg": 1.0},
        PARHELION_120_VIEW,
        PINNED_SPOT_PIXELS,
        20_000,
        "family pinned, reflection group element 6 (fold matrix R_z(120 deg)): the 120 deg parhelion at D = "
        "113.548 deg, its spot centre, the four neighbours, the vertical and horizontal tails down to 1e-26",
    ),
    BandSumCell(
        "plate_sigma_0.5",
        SCENE_3_6_4_8,
        {"family": "plate", "zenith_std_deg": 0.5},
        PARHELION_120_VIEW,
        PINNED_SPOT_PIXELS,
        20_000,
        "the same view and pixels at half the zenith spread: the centre brightens, the tails fall to 1e-92 "
        "(the spot's width scales with sigma; the 3-5 parhelion's does not)",
    ),
    BandSumCell(
        "plate",
        Scene(PLATE_CRYSTAL, (1, 2, 1), CANONICAL_REFRACTIVE_INDEX, SUN),
        {"family": "plate", "zenith_std_deg": 1.0},
        SUBSUN_VIEW,
        ((4, 4), (3, 4), (0, 4), (8, 4), (4, 3), (4, 2), (4, 1), (4, 0)),
        20_000,
        "family pinned, element 11 (a basal reflection) on a thin plate: the subsun at D = 30 deg, lit along the "
        "vertical, falling to 1e-15 across it (the column crystal has no valid pose on this family)",
    ),
    BandSumCell(
        "plate",
        Scene(PLATE_CRYSTAL, (1, 2, 3, 4, 1), CANONICAL_REFRACTIVE_INDEX, SUN),
        {"family": "plate", "zenith_std_deg": 1.0},
        SUBPARHELION_120_VIEW,
        ((4, 4), (3, 4), (5, 4), (4, 3), (4, 5), (4, 6), (4, 7), (0, 4), (4, 1)),
        20_000,
        "family pinned, element 5 on a thin plate: the 120 deg subparhelion at D = 122.242 deg, a pixel on the "
        "path's smallest deviation (120 deg) and an empty band beyond it",
    ),
    BandSumCell(
        "plate_n1.307",
        Scene(SLAB_CRYSTAL, (1, 3, 4, 2), 1.307, HORIZON_SUN),
        {"family": "plate", "zenith_std_deg": 1.0},
        HORIZON_PARHELION_120_VIEW,
        ((2, 4), (1, 4), (3, 4), (0, 4), (4, 4), (6, 4), (8, 4), (2, 5), (2, 6), (2, 3)),
        20_000,
        "a refractive index other than 1.31 (n = 1.307): the grazing slab 1-3-4-2 on a plate with the sun on the "
        "horizon, where the exit gate's critical angle must follow n (an exit gate at 1.31 zeroes every event); "
        "the spot's vertical streak at azimuth 120 deg, its horizontal tails and an empty band beside it",
    ),
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/analytic-parity"))
    parser.add_argument("--cells", nargs="*", help="cell names (<path>__<category> or <path>__<edge label>) to export; default: all")
    parser.add_argument("--verify", action="store_true", help="read every fixture back and recompute it")
    args = parser.parse_args(argv)

    cells, edge_cells, band_sum_cells = matrix(), list(EDGE_CELLS), list(BAND_SUM_CELLS)
    if args.cells:
        known = [cell.name for cell in (*cells, *edge_cells, *band_sum_cells)]
        unknown = set(args.cells) - set(known)
        if unknown:
            parser.error(f"unknown cells {sorted(unknown)}; known: {known}")
        cells = [cell for cell in cells if cell.name in args.cells]
        edge_cells = [cell for cell in edge_cells if cell.name in args.cells]
        band_sum_cells = [cell for cell in band_sum_cells if cell.name in args.cells]
    manifest = export_matrix(cells, args.output_dir, edge_cells, band_sum_cells)
    for entry in [*manifest["cells"], *manifest.get("edge_cells", []), *manifest.get("band_sum_cells", [])]:
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
