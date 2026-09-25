"""Phase II contour quadrature against Phase I, pixel by pixel (task ``s2-contour-quadrature``, AC3).

For each pixel (default: the canonical ``(150, 150)`` and rows 150/300/450/600
of column 126) this reports

- the Phase II value :meth:`.contour_quadrature.LevelSetGeometry.integrate`
  with its error estimate and diagnostics;
- the production Phase I value: every component of
  :func:`.discovery.discover_components` integrated by
  :func:`.quadrature.integrate_fiber_resampled` with ``eps = 1e-12`` (the
  ``eps -> 0`` limit Phase II computes) and ``rtol = 1e-9``;
- the same Phase I grid with ``nu'`` taken by a central difference of the
  spline instead of analytically.  The derivative of the phase condition
  ``nu . delta = 0`` is ``nu . delta' = -nu' . delta``; until task
  phase1-quadrature-start-and-speed production dropped the right-hand side,
  an ``O(delta)`` speed bias no grid removes (``delta`` ~6e-6 on the canonical
  loop is set by the fixed predictor spline), and this column was the
  diagnostic that exposed it.  Production now uses the analytic ``nu'``
  (:attr:`.resample.ResampledPredictors.phase_tangent_rates`); this column
  keeps a second, finite-difference source of ``nu'`` feeding the *same*
  speed kernel (``quadrature._parametric_speed``, one implementation of the
  linear system), evaluated at 4097, 16385 and 65537 nodes with the last two
  Richardson-extrapolated at order 2 (the slope jumps of ``entry_measure``).
  With the fix, ``relative_difference_production`` and
  ``relative_difference_corrected`` agree to the quadrature error.

Usage::

    uv run python scripts/compare_contour_quadrature_phase1.py --output /tmp/contour-phase1.json
    uv run python scripts/compare_contour_quadrature_phase1.py --pose-density-family random --output /tmp/contour-phase1-random.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from lumice_integral import quadrature as Q  # noqa: E402
from lumice_integral.band_sum import pixel_band  # noqa: E402
from lumice_integral.canonical_scene import CANONICAL_REFRACTIVE_INDEX, canonical_crystal, canonical_pose_density, canonical_sun_direction  # noqa: E402
from lumice_integral.continuation import retract_to_fiber_batch  # noqa: E402
from lumice_integral.contour import extract_level_sets  # noqa: E402
from lumice_integral.contour_quadrature import LevelSetGeometry, QuadratureOptions  # noqa: E402
from lumice_integral.discovery import discover_components, retarget_problem  # noqa: E402
from lumice_integral.dp_field import DPField  # noqa: E402
from lumice_integral.resample import fiber_spline, resample_spline, uniform_parameters  # noqa: E402
from lumice_integral.pose_density import build_pose_density  # noqa: E402
from lumice_integral.s2_store import build_event_store  # noqa: E402
from lumice_integral.strip_pixel import PixelOptions, canonical_strip_scene, pixel_target  # noqa: E402

DEFAULT_PIXELS = ((150, 150), (150, 126), (300, 126), (450, 126), (600, 126))
PHASE2_OPTIONS = QuadratureOptions(relative_tolerance=1e-11)
PHASE1_OPTIONS = Q.ResampleOptions(epsilon=1e-12, relative_tolerance=1e-9, maximum_node_count=262145)
CORRECTED_NODE_COUNTS = (4097, 16385, 65537)
NU_DIFFERENCE_STEP = 1e-5


def corrected_phase1(problem, result) -> dict:
    """Phase I integral with a finite-difference ``nu'`` on uniform grids (module docstring)."""
    spline = fiber_spline(result)
    values = []
    for n in CORRECTED_NODE_COUNTS:
        parameters = uniform_parameters(spline, n)
        grid = Q._evaluate_grid_nodes(problem, spline, parameters, PHASE1_OPTIONS)
        predictors = resample_spline(spline, parameters)
        retraction = retract_to_fiber_batch(problem, predictors.rotations, predictors.phase_tangents,
                                            iterations=PHASE1_OPTIONS.retraction_iterations)
        lo = np.clip(parameters - NU_DIFFERENCE_STEP, 0.0, spline.total)
        hi = np.clip(parameters + NU_DIFFERENCE_STEP, 0.0, spline.total)
        nu_prime = (resample_spline(spline, hi).phase_tangents - resample_spline(spline, lo).phase_tangents) / (hi - lo)[:, None]
        speed = np.asarray(Q._parametric_speed_batch_kernel(
            jnp.asarray(retraction.tangents), jnp.asarray(predictors.body_velocities),
            jnp.asarray(predictors.phase_tangents), jnp.asarray(nu_prime), jnp.asarray(retraction.deltas)))
        spacing = spline.total / (n - 1)
        values.append(Q._composite_simpson(np.where(grid.finite, grid.integrand * speed, 0.0), spacing) * Q.HAAR_TO_DVOL_G_FACTOR)
    return {
        "node_counts": list(CORRECTED_NODE_COUNTS),
        "values": values,
        "extrapolated": values[-1] + (values[-1] - values[-2]) / 3.0,
        "last_difference": values[-1] - values[-2],
    }


def compare_pixel(scene, field, store, row: int, column: int, density) -> dict:
    sun = canonical_sun_direction()
    centre, delta, _, _ = pixel_band(row, column, sun)
    start = time.perf_counter()
    (level_set,) = extract_level_sets(field, [delta], store)
    geometry = LevelSetGeometry.build(field, [level_set], PHASE2_OPTIONS)
    (phase2,) = geometry.integrate(sun, [centre], density)
    phase2_s = time.perf_counter() - start

    start = time.perf_counter()
    target = pixel_target(scene.render, row, column)
    discovered = discover_components(target, scene.seeds, template=scene.discovery_template,
                                     **PixelOptions().discovery_kwargs())
    production, corrected, kinds = 0.0, [], []
    production_error = 0.0
    for component in discovered.components:
        problem = retarget_problem(scene.production_template, target, component.seed)
        quadrature = Q.integrate_fiber_resampled(problem, component.result, PHASE1_OPTIONS)
        production += quadrature.value
        production_error += quadrature.error_estimate
        corrected.append(corrected_phase1(problem, component.result))
        kinds.append(component.kind)
    phase1_s = time.perf_counter() - start
    corrected_value = float(sum(c["extrapolated"] for c in corrected))
    reference = phase2.value if phase2.value != 0.0 else 1.0
    return {
        "row": row,
        "column": column,
        "delta_deg": float(np.degrees(delta)),
        "phase2": {
            "value": phase2.value, "error_estimate": phase2.error_estimate, "n_closed": phase2.n_closed,
            "n_open": phase2.n_open, "panels": phase2.panels, "geometry_evaluations": phase2.geometry_evaluations,
            "pixel_evaluations": phase2.evaluations, "low_order_splits": phase2.low_order_splits,
            "exhausted_panels": phase2.exhausted_panels, "seconds": phase2_s,
        },
        "phase1_production": {
            "value": production, "error_estimate": production_error, "components": kinds,
            "completeness": "complete" if discovered.incomplete_count == 0 else "unknown", "seconds": phase1_s,
        },
        "phase1_corrected_speed": {"value": corrected_value, "components": corrected},
        "relative_difference_production": production / reference - 1.0,
        "relative_difference_corrected": corrected_value / reference - 1.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pixel", type=int, nargs=2, action="append", metavar=("ROW", "COLUMN"))
    parser.add_argument("--store-n", type=int, default=200_000, help="event store of the level-set seed check")
    parser.add_argument("--pose-density-family", choices=("column", "random"), default="column",
                        help="column: the canonical density (0.5 deg); random: Haar")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pixels = [tuple(p) for p in args.pixel] if args.pixel else list(DEFAULT_PIXELS)
    density = canonical_pose_density() if args.pose_density_family == "column" else build_pose_density("random")
    scene = canonical_strip_scene(pose_density=density)
    field = DPField.build(canonical_crystal(), (3, 5), CANONICAL_REFRACTIVE_INDEX)
    store = build_event_store(canonical_crystal(), CANONICAL_REFRACTIVE_INDEX, [(3, 5)], args.store_n, run_checks=False)
    report = {"pose_density_family": args.pose_density_family, "phase1_options": {"epsilon": PHASE1_OPTIONS.epsilon, "relative_tolerance": PHASE1_OPTIONS.relative_tolerance,
                                 "maximum_node_count": PHASE1_OPTIONS.maximum_node_count},
              "pixels": []}
    for row, column in pixels:
        entry = compare_pixel(scene, field, store, row, column, density)
        report["pixels"].append(entry)
        print(f"({row}, {column}) delta {entry['delta_deg']:.4f} deg: phase2 {entry['phase2']['value']:.12g} "
              f"(err {entry['phase2']['error_estimate']:.1e}); production phase1 rel {entry['relative_difference_production']:+.2e}; "
              f"corrected-speed phase1 rel {entry['relative_difference_corrected']:+.2e}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
