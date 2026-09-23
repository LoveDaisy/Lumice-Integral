"""Absolute radiometric scale probe: Lumice Integral strip value vs Lumice ``raw / emitted_energy``.

The strip value of one pixel (``strip_io`` ``value_semantics``, contract
section 7) is the pose-space coarea integral

    V(w) = (1 / 8 pi^2) int rho A_P T / J_perp dH^1   [crystal length^2 per steradian]

i.e. the cross-section-weighted expectation ``E_R[A_P(R) T(R) delta(w - Phi(R))]``
over the Haar probability measure: the power per steradian that one crystal of
random pose sends along path ``P`` into direction ``w``, per unit incident
irradiance.  ``A_P`` is the *absolute* entry cross-section (``entry_measure``).

Lumice (Ice Halo repository ``src/core/simulator.cpp``, read as evidence,
never linked) samples every ray's crystal pose from ``rho`` alone
(``InitRay_rot``), gives it the emission weight of its wavelength
(``InitRay_d_w_previdx``: weight 1 here, no pose-dependent factor), and only
then picks the entry point with probability proportional to the projected area
of each front-facing sub-triangle (``InitRay_p_fid``:
``max(-d . n, 0) * area``).  One emitted ray therefore enters path ``P`` with
probability ``A_P(R) / A_tot(R)``, ``A_tot`` being the crystal's whole projected
silhouette; the orientation is *not* weighted by ``A_tot``.  With Fresnel
factors identical on both sides (per-interface s/p average,
``optics_shared.h::GetReflectRatio`` vs ``optics.fresnel_transmission_path``)
the expected Lumice pixel is

    raw[p] / E = N_sym * ybar(550) * Omega_p * V~(w_p),
    V~(w) = (1 / 8 pi^2) int rho (A_P / A_tot) T / J_perp dH^1

with ``N_sym`` the number of symmetry-equivalent raypaths the ``[3, 5]``
filter admits (``PBD``: 12; under the canonical density -- uniform azimuth
and roll, zenith symmetric about 90 deg -- every equivalent raypath has the
same sky image in expectation, so the fold is one scalar), ``ybar(550)`` the
CIE 1931 Y matching function Lumice multiplies into the Y channel, and
``Omega_p`` the pixel's solid angle (linear lens: ``cos^3(theta) / scale^2``).
Hence the conversion

    raw[p] / E = K_p * V(w_p),  K_p = N_sym * ybar * Omega_p / A_eff(w_p),

with ``A_eff = V / V~`` the fiber-weighted harmonic mean of ``A_tot`` (crystal
length^2).  ``V~`` is computed here on the very components ``V`` is (one
discovery, two quadratures), with ``rho / A_tot`` as the pose weight; nothing
is fitted.  The remaining differences are Monte Carlo noise and the point
(pixel-centre) versus pixel-area model.

Nothing here imports or calls Lumice; ``src/`` is not modified.  Usage::

    uv run python scripts/probe_absolute_scale.py --lumice-run <run1_dir> --lumice-run <run2_dir> --output-dir <dir>
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from lumice_integral.camera import camera_rotation, linear_pixel_sky_direction, linear_scale
from lumice_integral.canonical_scene import (
    CANONICAL_REFRACTIVE_INDEX,
    CANONICAL_RENDER,
    canonical_crystal,
    canonical_incident_direction,
    canonical_pose_density,
)
from lumice_integral.discovery import discover_components, retarget_problem
from lumice_integral.geometry import HexPrism
from lumice_integral.pose_density import PoseDensity
from lumice_integral.quadrature import integrate_fiber_resampled
from lumice_integral.optics import PATH_3_5_FACES
from lumice_integral.strip_pixel import PixelOptions, StripScene, build_strip_scene, pixel_target, subpixel_targets

# Lumice's ice index at 550 nm: IceRefractiveIndex::Get (src/core/optics.cpp), Sellmeier with kCoefAvr
# {0.701777, 1.091144, 0.884400, 0.796950} (src/core/optics.hpp); the canonical scene keeps the historical 1.31
LUMICE_REFRACTIVE_INDEX_550 = 1.3110129
# CIE 1931 2-degree ybar at 550 nm, the entry Lumice's kCmfY table holds (src/util/color_data.hpp)
YBAR_550 = 0.9949501
LUMICE_SYMMETRY_FOLD = {"P": 6, "PBD": 12}
DEFAULT_COLUMNS = (106, 126, 146)


def silhouette_area(crystal: HexPrism, s_body: np.ndarray) -> np.ndarray:
    """Projected area ``sum_f max(-s . n_f, 0) area_f`` of a convex crystal for body-frame directions ``(N, 3)``.

    The same sum Lumice's ``InitRay_p_fid`` normalises its entry-point sampling by.
    """
    normals = []
    areas = []
    for face in crystal.faces:
        v = crystal.vertices[list(face.vertex_ids)]
        newell = np.sum(np.cross(v, np.roll(v, -1, axis=0)), axis=0)  # 2 x vector area of the planar polygon
        normals.append(crystal.normal(face))
        areas.append(0.5 * float(np.linalg.norm(newell)))
    n = np.asarray(normals)
    a = np.asarray(areas)
    return np.maximum(-(np.asarray(s_body) @ n.T), 0.0) @ a


@dataclass(frozen=True)
class PerSilhouetteDensity:
    """``rho(R) / A_tot(R)``: Lumice's per-ray pose weight (duck-typed :data:`PoseDensity`)."""

    base: PoseDensity
    crystal: HexPrism
    incident_direction: np.ndarray
    unit: str = "1 / crystal length^2"
    normalization: str = "rho_pose divided by the crystal's projected silhouette area along the incident direction"

    def evaluate_batch(self, rotations: np.ndarray) -> np.ndarray:
        rotations = np.asarray(rotations, dtype=np.float64)
        s_body = np.einsum("nji,j->ni", rotations, self.incident_direction)  # R^T s
        return np.asarray(self.base.evaluate_batch(rotations), dtype=np.float64) / silhouette_area(self.crystal, s_body)

    def __call__(self, rotation: np.ndarray) -> float:
        return float(self.evaluate_batch(np.asarray(rotation)[None])[0])


def pixel_solid_angle(row: int, column: int, render: dict[str, Any]) -> float:
    """Solid angle of one linear-lens pixel: ``cos^3(theta) / scale^2`` (``scale`` pixels per unit tangent)."""
    scale = linear_scale(render["fov_deg"], render["width"], render["height"])
    axis = camera_rotation(render["view"])[:, 2]
    sky = linear_pixel_sky_direction(row, column, width=render["width"], height=render["height"], fov_deg=render["fov_deg"], view=render["view"])
    cos_theta = float(np.dot(sky, axis) / np.linalg.norm(sky))
    return cos_theta**3 / scale**2


def load_run(run_dir: Path) -> tuple[np.ndarray, dict[str, Any], str]:
    arr = np.load(run_dir / "img_01.npy")
    meta = json.loads((run_dir / "img_01.json").read_text())
    symmetry = json.loads((run_dir / "config.json").read_text())["filter"][0]["symmetry"]
    return np.asarray(arr[:, :, 1], dtype=np.float64), meta, symmetry


def integrate(scene: StripScene, target: np.ndarray, component: Any, options: PixelOptions) -> float:
    problem = retarget_problem(scene.production_template, target, component.seed)
    q = integrate_fiber_resampled(problem, component.result, options.quadrature)
    return float(q.value) if q.status == "available" else float("nan")


def pixel_values(scene: StripScene, scene_lumice: StripScene, target: np.ndarray, options: PixelOptions, warm: tuple[np.ndarray, ...]) -> tuple[float, float, Any]:
    """``(V, V~, discovery)`` of one target: one discovery, the two quadratures on its components."""
    found = discover_components(
        target, scene.crystal, scene.prescan_table, template=scene.discovery_template, extra_seeds=warm, **options.discovery_kwargs()
    )
    value = sum(integrate(scene, target, c, options) for c in found.components)
    value_lumice = sum(integrate(scene_lumice, target, c, options) for c in found.components)
    return float(value), float(value_lumice), found


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lumice-run", type=Path, action="append", required=True, help="run directory with img_01.npy, img_01.json, config.json (repeatable)")
    parser.add_argument("--columns", type=int, nargs="+", default=list(DEFAULT_COLUMNS))
    parser.add_argument("--row-step", type=int, default=10)
    parser.add_argument("--lit-floor", type=float, default=1e-2, help="probe rows where the merged Lumice column is above this fraction of its maximum")
    parser.add_argument(
        "--refractive-index",
        type=float,
        default=CANONICAL_REFRACTIVE_INDEX,
        help=f"ice index of the Lumice Integral side (canonical {CANONICAL_REFRACTIVE_INDEX}; Lumice's own 550 nm value is {LUMICE_REFRACTIVE_INDEX_550})",
    )
    parser.add_argument("--bright-floor", type=float, default=0.1, help="the summary's bright band: probed pixels whose Lumice value is above this fraction of the column's probed maximum")
    parser.add_argument("--subpixel-rows", type=int, nargs="*", default=[150, 300, 450], help="rows (on every probed column) where the point value is compared with a sub-pixel mean")
    parser.add_argument("--subpixel-grid", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    runs = [load_run(d) for d in args.lumice_run]
    if len({s for _, _, s in runs}) != 1:
        raise SystemExit("runs disagree on symmetry")
    symmetry = runs[0][2]
    n_sym = LUMICE_SYMMETRY_FOLD[symmetry]
    merged = np.sum([y for y, _, _ in runs], axis=0)
    emitted = float(sum(m["emitted_energy"] for _, m, _ in runs))
    per_ray = merged / emitted

    crystal = canonical_crystal()
    incident = canonical_incident_direction()
    common = dict(incident_direction=incident, refractive_index=args.refractive_index, crystal=crystal, render=CANONICAL_RENDER)
    scene = build_strip_scene(PATH_3_5_FACES, pose_density=canonical_pose_density(), **common)
    scene_lumice = build_strip_scene(
        PATH_3_5_FACES,
        pose_density=PerSilhouetteDensity(scene.pose_density, crystal, incident),
        prescan_table=scene.prescan_table,
        **common,
    )
    options = PixelOptions()
    render = dict(CANONICAL_RENDER)
    axis_solid_angle = 1.0 / linear_scale(render["fov_deg"], render["width"], render["height"]) ** 2

    records: list[dict[str, Any]] = []
    start = time.perf_counter()
    for column in args.columns:
        col = per_ray[:, column]
        rows = [r for r in range(0, render["height"], args.row_step) if col[r] >= args.lit_floor * col.max()]
        warm: tuple[np.ndarray, ...] = ()
        for row in rows:
            value, value_lumice, found = pixel_values(scene, scene_lumice, pixel_target(render, row, column), options, warm)
            warm = tuple(c.seed for c in found.components)
            omega = pixel_solid_angle(row, column, render)
            a_eff = value / value_lumice if value_lumice > 0 else float("nan")
            k_pixel = n_sym * YBAR_550 * omega / a_eff if value_lumice > 0 else float("nan")
            predicted = n_sym * YBAR_550 * omega * value_lumice
            run_values = [float(y[row, column] / m["emitted_energy"]) for y, m, _ in runs]
            records.append(
                {
                    "row": row,
                    "column": column,
                    "components": len(found.components),
                    "complete": found.incomplete_count == 0,
                    "li_value": value,
                    "li_value_per_silhouette": value_lumice,
                    "a_eff": a_eff,
                    "pixel_solid_angle": omega,
                    "k_pixel": k_pixel,
                    "lumice_per_ray": float(per_ray[row, column]),
                    "lumice_per_ray_runs": run_values,
                    "predicted_lumice_per_ray": predicted,
                    "measured_over_predicted": float(per_ray[row, column] / predicted) if predicted > 0 else float("nan"),
                }
            )
            r = records[-1]
            print(f"col {column} row {row}: V {value:.4e} A_eff {a_eff:.4f} lumice/pred {r['measured_over_predicted']:.4f}")
    # point (pixel-centre) versus pixel-area model away from the caustic: the sub-pixel mean of V~
    subpixel: list[dict[str, Any]] = []
    for column in args.columns:
        for row in args.subpixel_rows:
            point = next((r for r in records if r["row"] == row and r["column"] == column), None)
            if point is None or not point["li_value_per_silhouette"] > 0:
                continue
            warm = ()
            values = []
            for target in subpixel_targets(render, row, column, args.subpixel_grid):
                _, v_lumice, found = pixel_values(scene, scene_lumice, target, options, warm)
                warm = tuple(c.seed for c in found.components)
                values.append(v_lumice)
            mean = float(np.mean(values))
            subpixel.append(
                {
                    "row": row,
                    "column": column,
                    "grid": args.subpixel_grid,
                    "point_over_subpixel_mean": point["li_value_per_silhouette"] / mean,
                    "measured_over_predicted_subpixel": point["lumice_per_ray"] / (n_sym * YBAR_550 * point["pixel_solid_angle"] * mean),
                }
            )
            print(f"subpixel col {column} row {row}: point/mean {subpixel[-1]['point_over_subpixel_mean']:.4f}")
    elapsed = time.perf_counter() - start

    with (args.output_dir / "absolute_scale_pixels.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=[k for k in records[0] if k != "lumice_per_ray_runs"] + ["lumice_per_ray_run1", "lumice_per_ray_run2"])
        writer.writeheader()
        for r in records:
            row = {k: v for k, v in r.items() if k != "lumice_per_ray_runs"}
            row.update({f"lumice_per_ray_run{i + 1}": v for i, v in enumerate(r["lumice_per_ray_runs"][:2])})
            writer.writerow(row)

    summary: dict[str, Any] = {
        "generated": dt.datetime.now().astimezone().isoformat(),
        "lumice_runs": [str(d) for d in args.lumice_run],
        "emitted_energy": emitted,
        "symmetry": symmetry,
        "symmetry_fold": n_sym,
        "ybar_550": YBAR_550,
        "refractive_index": args.refractive_index,
        "lumice_refractive_index_550": LUMICE_REFRACTIVE_INDEX_550,
        "axis_solid_angle": axis_solid_angle,
        "axis_solid_angle_sidecar": runs[0][1]["axis_solid_angle"],
        "crystal": {"a": crystal.a, "h": crystal.h},
        "elapsed_s": elapsed,
        "subpixel_check": subpixel,
        "columns": {},
    }
    for column in args.columns:
        sel = [r for r in records if r["column"] == column and np.isfinite(r["measured_over_predicted"])]
        ratio = np.array([r["measured_over_predicted"] for r in sel])
        a_eff = np.array([r["a_eff"] for r in sel])
        runs_arr = np.array([r["lumice_per_ray_runs"] for r in sel])
        noise = None
        if runs_arr.shape[1] >= 2:
            rel = (runs_arr[:, 0] - runs_arr[:, 1]) / (0.5 * (runs_arr[:, 0] + runs_arr[:, 1]))
            noise = float(np.std(rel) / 2.0)  # merged relative noise, see compare_strip_v2.merge_lumice_float
        lumice = np.array([r["lumice_per_ray"] for r in sel])
        bright = lumice >= args.bright_floor * lumice.max()
        summary["columns"][str(column)] = {
            "pixels": len(sel),
            "bright_band": {
                "floor": args.bright_floor,
                "rows": [int(min(r["row"] for r, b in zip(sel, bright) if b)), int(max(r["row"] for r, b in zip(sel, bright) if b))],
                "pixels": int(bright.sum()),
                "measured_over_predicted_median": float(np.median(ratio[bright])),
                "measured_over_predicted_standard_error": float(np.std(ratio[bright]) / np.sqrt(bright.sum())),
                "k_pixel_median": float(np.median([r["k_pixel"] for r, b in zip(sel, bright) if b])),
                "a_eff_min_max": [float(a_eff[bright].min()), float(a_eff[bright].max())],
            },
            "measured_over_predicted_median": float(np.median(ratio)),
            "measured_over_predicted_p10_p90": [float(np.percentile(ratio, 10)), float(np.percentile(ratio, 90))],
            "a_eff_median": float(np.median(a_eff)),
            "a_eff_min_max": [float(a_eff.min()), float(a_eff.max())],
            "merged_relative_noise_per_pixel": noise,
        }
        print(f"column {column}: measured/predicted median {summary['columns'][str(column)]['measured_over_predicted_median']:.4f}, A_eff {a_eff.min():.3f}..{a_eff.max():.3f}, noise {noise}")
    (args.output_dir / "absolute_scale_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {args.output_dir}")


if __name__ == "__main__":
    main()
