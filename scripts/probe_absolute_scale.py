"""Absolute radiometric scale probe: Lumice Integral strip value vs Lumice ``raw / emitted_energy``.

The strip value of one pixel (``strip_io`` ``value_semantics``, contract
section 7) is the pose-space coarea integral

    V(w) = (1 / 8 pi^2) int rho A_P T / J_perp dH^1   [crystal length^2 per steradian]

i.e. the cross-section-weighted expectation ``E_R[A_P(R) T(R) delta(w - Phi(R))]``
over the Haar probability measure: the power per steradian that one crystal of
random pose sends along path ``P`` into direction ``w``, per unit incident
irradiance.  ``A_P`` is the *absolute* entry cross-section (``entry_measure``).

Lumice (Ice Halo repository at ``2056f699``, ``src/core/simulator.cpp``, read
as evidence, never linked) samples every ray's crystal pose from ``rho`` alone
(``InitRay_rot``), gives it the emission weight of its wavelength
(``InitRay_d_w_previdx``: weight 1 here), and at entry (``InitRay_p_fid``)
multiplies that weight by ``A_tot(R) / (S / 2)`` (``lm_pcg::entry_weight``,
``src/core/shared/pcg_shared.h``; ``A_tot`` the crystal's projected silhouette
along the ray, ``S`` its total surface area) before picking the entry
sub-triangle with probability ``max(-d . n, 0) * area / A_tot``.  The weight a
ray loses at entry still counts toward ``emitted_energy`` (``doc/configuration.md``).
One emitted ray therefore lands on path ``P`` with expected weight
``(A_tot / (S/2)) (A_P / A_tot) = A_P / (S/2)``: ``A_tot`` cancels.  With
Fresnel factors identical on both sides (per-interface s/p average,
``optics_shared.h::GetReflectRatio`` vs ``optics.fresnel_transmission_path``)
the expected Lumice pixel is

    raw[p] / E = K_p * V(w_p),  K_p = N_sym * ybar(550) * Omega_p / (S / 2)

with ``N_sym`` the number of symmetry-equivalent raypaths the ``[3, 5]``
filter admits (``PBD``: 12; under the canonical density -- uniform azimuth
and roll, zenith symmetric about 90 deg -- every equivalent raypath has the
same sky image in expectation, so the fold is one scalar), ``ybar(550)`` the
CIE 1931 Y matching function Lumice multiplies into the Y channel, ``Omega_p``
the pixel's solid angle (linear lens: ``cos^3(theta) / scale^2``) and ``S``
in the strip's length unit (hexagon edge ``a = 1``; the ratio ``A_tot / (S/2)``
is scale-free, so Lumice's equal-surface-area crystal size convention only
matters between crystals).  ``K_p`` depends on the pixel through ``Omega_p``
alone; nothing is fitted.  The remaining differences are Monte Carlo noise and
the point (pixel-centre) versus pixel-area model.

Lumice before ``6fc48bb4`` did not apply the entry weight; its conversion
carried the fiber-weighted harmonic mean ``A_eff`` of ``A_tot`` instead of
``S / 2`` (``docs/ch06-reference-fixture.md`` section 7, stage 4, history).

Nothing here imports or calls Lumice; ``src/`` is not modified.  Usage::

    uv run python scripts/probe_absolute_scale.py --lumice-run <run1_dir> --lumice-run <run2_dir> --output-dir <dir>
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import time
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


def total_surface_area(crystal: HexPrism) -> float:
    """Total surface area ``S`` of the crystal: the sum of its face areas (Newell), ``6 a h + 3 sqrt(3) a^2`` for a prism.

    ``S / 2`` is the entry-weight denominator of Lumice's ``lm_pcg::entry_weight`` (``s_total`` summed
    over the entry sub-triangles, i.e. over every face).
    """
    total = 0.0
    for face in crystal.faces:
        v = crystal.vertices[list(face.vertex_ids)]
        newell = np.sum(np.cross(v, np.roll(v, -1, axis=0)), axis=0)  # 2 x vector area of the planar polygon
        total += 0.5 * float(np.linalg.norm(newell))
    return total


def pixel_constant(row: int, column: int, render: dict[str, Any], *, n_sym: int, surface_area: float) -> float:
    """``K_p = N_sym * ybar(550) * Omega_p / (S / 2)`` of one pixel (module docstring)."""
    return n_sym * YBAR_550 * pixel_solid_angle(row, column, render) / (0.5 * surface_area)


def pixel_solid_angle(row: int, column: int, render: dict[str, Any]) -> float:
    """Solid angle of one linear-lens pixel: ``cos^3(theta) / scale^2`` (``scale`` pixels per unit tangent)."""
    scale = linear_scale(render["fov_deg"], render["width"], render["height"])
    axis = camera_rotation(render["view"])[:, 2]
    sky = linear_pixel_sky_direction(row, column, width=render["width"], height=render["height"], fov_deg=render["fov_deg"], view=render["view"])
    cos_theta = float(np.dot(sky, axis) / np.linalg.norm(sky))
    return cos_theta**3 / scale**2


def pixel_solid_angles(render: dict[str, Any]) -> np.ndarray:
    """``(H, W)`` linear-lens pixel solid angles, vectorised (:func:`pixel_solid_angle` for every pixel).

    Uses the axis-aligned tangent-plane shortcut (``cos_theta = 1 / sqrt(1 + x^2 + y^2)``) instead of
    :func:`~lumice_integral.camera.linear_pixel_sky_direction`'s per-pixel rotation, which is equivalent for
    this camera model but avoids a Python loop over every pixel; ``tests/test_probe_absolute_scale.py``
    checks it against the scalar form pixel by pixel.
    """
    width, height = int(render["width"]), int(render["height"])
    scale = linear_scale(render["fov_deg"], width, height)
    x = ((np.arange(width) + 0.5) - width / 2.0) / scale
    y = ((np.arange(height) + 0.5) - height / 2.0) / scale
    cos_theta = 1.0 / np.sqrt(1.0 + x[None, :] ** 2 + y[:, None] ** 2)
    return cos_theta**3 / scale**2


def merged_relative_noise(a: np.ndarray, b: np.ndarray) -> float:
    """Relative noise of the *merged* pair of i.i.d. runs ``a``, ``b`` of the same quantity.

    ``rel = (a - b) / (0.5 * (a + b))``, ``std(rel) / 2`` (see
    ``scripts/compare_strip_v2.py::merge_lumice_float`` for the derivation: the run-to-run relative
    difference has standard deviation ``sqrt(2)`` times the single-run relative noise, and the merged
    (averaged) relative noise is half of that difference's std). The single shared implementation for
    this quantity (a56); do not re-derive it per caller.
    """
    rel = (a - b) / (0.5 * (a + b))
    return float(np.std(rel) / 2.0)


def load_run(run_dir: Path) -> tuple[np.ndarray, dict[str, Any], str]:
    arr = np.load(run_dir / "img_01.npy")
    meta = json.loads((run_dir / "img_01.json").read_text())
    symmetry = json.loads((run_dir / "config.json").read_text())["filter"][0]["symmetry"]
    return np.asarray(arr[:, :, 1], dtype=np.float64), meta, symmetry


def integrate(scene: StripScene, target: np.ndarray, component: Any, options: PixelOptions) -> float:
    problem = retarget_problem(scene.production_template, target, component.seed)
    q = integrate_fiber_resampled(problem, component.result, options.quadrature)
    return float(q.value) if q.status == "available" else float("nan")


def pixel_value(scene: StripScene, target: np.ndarray, options: PixelOptions, warm: tuple[np.ndarray, ...]) -> tuple[float, Any]:
    """``(V, discovery)`` of one target: one discovery, one quadrature per component."""
    found = discover_components(
        target, scene.crystal, scene.prescan_table, template=scene.discovery_template, extra_seeds=warm, **options.discovery_kwargs()
    )
    return float(sum(integrate(scene, target, c, options) for c in found.components)), found


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
    surface_area = total_surface_area(crystal)
    scene = build_strip_scene(
        PATH_3_5_FACES,
        pose_density=canonical_pose_density(),
        incident_direction=canonical_incident_direction(),
        refractive_index=args.refractive_index,
        crystal=crystal,
        render=CANONICAL_RENDER,
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
            value, found = pixel_value(scene, pixel_target(render, row, column), options, warm)
            warm = tuple(c.seed for c in found.components)
            omega = pixel_solid_angle(row, column, render)
            k_pixel = pixel_constant(row, column, render, n_sym=n_sym, surface_area=surface_area)
            predicted = k_pixel * value
            run_values = [float(y[row, column] / m["emitted_energy"]) for y, m, _ in runs]
            measured = float(per_ray[row, column])
            records.append(
                {
                    "row": row,
                    "column": column,
                    "components": len(found.components),
                    "complete": found.incomplete_count == 0,
                    "li_value": value,
                    "pixel_solid_angle": omega,
                    "k_pixel": k_pixel,
                    # the area that would make raw / E = N_sym ybar Omega_p V / area exact; S / 2 predicted
                    "implied_entry_area": n_sym * YBAR_550 * omega * value / measured if measured > 0 else float("nan"),
                    "lumice_per_ray": measured,
                    "lumice_per_ray_runs": run_values,
                    "predicted_lumice_per_ray": predicted,
                    "measured_over_predicted": measured / predicted if predicted > 0 else float("nan"),
                }
            )
            r = records[-1]
            print(f"col {column} row {row}: V {value:.4e} implied area {r['implied_entry_area']:.4f} lumice/pred {r['measured_over_predicted']:.4f}")
    # point (pixel-centre) versus pixel-area model away from the caustic: the sub-pixel mean of V
    subpixel: list[dict[str, Any]] = []
    for column in args.columns:
        for row in args.subpixel_rows:
            point = next((r for r in records if r["row"] == row and r["column"] == column), None)
            if point is None or not point["li_value"] > 0:
                continue
            warm = ()
            values = []
            for target in subpixel_targets(render, row, column, args.subpixel_grid):
                v, found = pixel_value(scene, target, options, warm)
                warm = tuple(c.seed for c in found.components)
                values.append(v)
            mean = float(np.mean(values))
            subpixel.append(
                {
                    "row": row,
                    "column": column,
                    "grid": args.subpixel_grid,
                    "point_over_subpixel_mean": point["li_value"] / mean,
                    "measured_over_predicted_subpixel": point["lumice_per_ray"] / (point["k_pixel"] * mean),
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
        "surface_area": surface_area,
        "entry_area": 0.5 * surface_area,
        "convention": "raw[p] / emitted_energy = K_p * V(p), K_p = N_sym * ybar(550) * Omega_p / (S / 2)",
        "elapsed_s": elapsed,
        "subpixel_check": subpixel,
        "columns": {},
    }
    for column in args.columns:
        sel = [r for r in records if r["column"] == column and np.isfinite(r["measured_over_predicted"])]
        ratio = np.array([r["measured_over_predicted"] for r in sel])
        implied = np.array([r["implied_entry_area"] for r in sel])
        runs_arr = np.array([r["lumice_per_ray_runs"] for r in sel])
        noise = None
        if runs_arr.shape[1] >= 2:
            noise = merged_relative_noise(runs_arr[:, 0], runs_arr[:, 1])
        lumice = np.array([r["lumice_per_ray"] for r in sel])
        bright = lumice >= args.bright_floor * lumice.max()
        k_bright = np.array([r["k_pixel"] for r, b in zip(sel, bright) if b])
        summary["columns"][str(column)] = {
            "pixels": len(sel),
            "bright_band": {
                "floor": args.bright_floor,
                "rows": [int(min(r["row"] for r, b in zip(sel, bright) if b)), int(max(r["row"] for r, b in zip(sel, bright) if b))],
                "pixels": int(bright.sum()),
                "measured_over_predicted_median": float(np.median(ratio[bright])),
                "measured_over_predicted_standard_error": float(np.std(ratio[bright]) / np.sqrt(bright.sum())),
                "k_pixel_median": float(np.median(k_bright)),
                # K_p varies with Omega_p only; the implied area carries every pose-dependent residual
                "k_pixel_relative_span": float(k_bright.max() / k_bright.min() - 1.0),
                "implied_entry_area_median": float(np.median(implied[bright])),
                "implied_entry_area_min_max": [float(implied[bright].min()), float(implied[bright].max())],
                "implied_entry_area_relative_std": float(np.std(implied[bright]) / np.mean(implied[bright])),
            },
            "measured_over_predicted_median": float(np.median(ratio)),
            "measured_over_predicted_p10_p90": [float(np.percentile(ratio, 10)), float(np.percentile(ratio, 90))],
            "implied_entry_area_median": float(np.median(implied)),
            "merged_relative_noise_per_pixel": noise,
        }
        c = summary["columns"][str(column)]
        print(
            f"column {column}: measured/predicted median {c['measured_over_predicted_median']:.4f}, "
            f"bright implied area {c['bright_band']['implied_entry_area_min_max'][0]:.3f}..{c['bright_band']['implied_entry_area_min_max'][1]:.3f} "
            f"(S/2 {0.5 * surface_area:.3f}), noise {noise}"
        )
    (args.output_dir / "absolute_scale_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {args.output_dir}")


if __name__ == "__main__":
    main()
