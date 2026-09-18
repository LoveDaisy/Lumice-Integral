"""Defect-2 localisation probe: factor-by-factor export and independent estimates.

Defect 2 (owner review of the v1 strip): between rows ``300`` and ``650`` the
rendered strip decays about ``3x`` faster than the historical raw and the
Lumice remake.  This probe does *not* fix anything; it asks which named factor
of the integrand

    rho_pose * entry_measure * fresnel_transmission * path_validity / (J_perp + eps)

can carry a factor of that size, on the diagnostic pixels of column ``126``
at rows ``150 / 300 / 450 / 600`` (shared with ``scripts/compare_strip_v2.py``,
see :data:`PROBE_COLUMN` / :data:`PROBE_ROWS`).

For each pixel it runs the production pipeline (``strip_pixel.render_pixel``),
re-traces every integrated component with the production weights
(``continuation.trace_fiber``) and exports the four factors and ``J_perp`` at
every accepted pose, together with the c-axis zenith of the pose.  Two
factors are then re-estimated independently of the production code:

- ``rho_pose``: from the c-axis zenith alone, with the Lumice definition of a
  ``gauss`` zenith distribution (``p_sphere(theta) ~ exp(-(theta-mu)^2 / 2 sigma^2)
  sin(theta)``, ``mu = 90 deg``, ``sigma = 0.5 deg``; Lumice
  ``doc/crystal-orientation-sampling.md`` section 4.1, and the ``band1e9``
  Lumice config of the Writing-Lab remake) normalised by ``scipy.integrate.quad``
  and expressed relative to the Haar (uniform-on-the-sphere) density.  The
  *legacy* Lumice variant without the ``sin(theta)`` weight is reported too.
- ``entry_measure``: by Monte Carlo ray casting on the crystal's own vertex
  list: uniform points on face ``3``, Snell refraction, straight internal
  line, first exit face by plane intersection; the fraction that leaves
  through face ``5`` times the face area times ``cos_i`` is the effective
  entry cross-section perpendicular to the incident direction (the contract
  of ``geometry.entry_measure``).

Both estimators are self-checked on a configuration with a known answer
before any ratio is reported (a uniform sphere density must give ``rho = 1``
everywhere; the ray-cast fraction of an all-faces path must be ``1``).

Output: ``defect2_probe.json`` (per pixel, per component, per pose arrays are
summarised; the ratios production / independent are the deliverable) and a
console summary.  Usage::

    uv run python scripts/probe_defect2_factors.py --output-dir <task>/artifacts
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.integrate import quad

from lumice_integral.canonical_scene import (
    CANONICAL_REFRACTIVE_INDEX,
    CANONICAL_ZENITH_MEAN_DEG,
    CANONICAL_ZENITH_STD_DEG,
)
from lumice_integral.continuation import trace_fiber
from lumice_integral.discovery import retarget_problem
from lumice_integral.geometry import HexPrism
from lumice_integral.quadrature import ResampleOptions
from lumice_integral.strip_pixel import PixelOptions, canonical_strip_scene, pixel_target, render_pixel

PROBE_COLUMN = 126
PROBE_ROWS = (150, 300, 450, 600)
PATH = (3, 5)
FACTOR_NAMES = ("rho_pose", "entry_measure", "fresnel_transmission", "path_validity")
WIDTH_SENSITIVITY_STD_DEG = (0.25, 0.5, 1.0, 2.0)


# ---------- independent estimate 1: rho_pose from the zenith alone ----------
def lumice_gauss_density_relative_to_haar(
    theta: np.ndarray, *, mean_deg: float, std_deg: float, area_weighted: bool = True
) -> np.ndarray:
    """``dP / d mu_Haar`` of a Lumice ``gauss`` zenith distribution at zenith ``theta``.

    Lumice samples ``theta`` from ``p(theta) ~ g(theta) sin(theta)`` (area
    weighted; ``area_weighted=False`` is the ``gauss_legacy`` path that omits
    ``sin``), with uniform azimuth and roll.  The direction ``n`` then has the
    sphere density ``p_n = g(theta) / Z`` per unit area (``Z = 2 pi int g sin``)
    -- or ``g(theta) / (sin(theta) Z')`` for the legacy path -- and relative
    to the uniform density ``1 / (4 pi)`` that is ``4 pi p_n``.
    """
    mu, sigma = np.radians(mean_deg), np.radians(std_deg)

    def g(t: float) -> float:
        return float(np.exp(-0.5 * ((t - mu) / sigma) ** 2))

    theta = np.asarray(theta, dtype=np.float64)
    if area_weighted:
        z, _ = quad(lambda t: g(t) * np.sin(t), 0.0, np.pi, points=[mu], limit=400)
        return 4.0 * np.pi * np.vectorize(g)(theta) / (2.0 * np.pi * z)
    z, _ = quad(g, 0.0, np.pi, points=[mu], limit=400)
    return 4.0 * np.pi * np.vectorize(g)(theta) / (2.0 * np.pi * z * np.sin(theta))


def self_check_rho() -> dict[str, Any]:
    """A very wide Gaussian is uniform on the sphere: the Haar ratio must be ~1 near the mean."""
    theta = np.radians(np.array([60.0, 90.0, 120.0]))
    wide = lumice_gauss_density_relative_to_haar(theta, mean_deg=90.0, std_deg=1e6)
    # and the canonical width integrates to one against the uniform sphere measure
    mu, sigma = np.radians(90.0), np.radians(0.5)
    total, _ = quad(
        lambda t: lumice_gauss_density_relative_to_haar(np.array([t]), mean_deg=90.0, std_deg=0.5)[0] * 0.5 * np.sin(t),
        mu - 12 * sigma,
        mu + 12 * sigma,
        limit=400,
    )
    return {"wide_gaussian_ratio_to_haar": wide.tolist(), "canonical_integral_against_haar": float(total)}


# ---------- independent estimate 2: entry_measure by ray casting ----------
def _face_planes(crystal: HexPrism) -> tuple[list[int], np.ndarray, np.ndarray]:
    numbers, normals, offsets = [], [], []
    for face in crystal.faces:
        pts = crystal.vertices[list(face.vertex_ids)]
        # Newell normal, outward by the face vertex order convention
        n = np.zeros(3)
        for i in range(len(pts)):
            p, q = pts[i], pts[(i + 1) % len(pts)]
            n += np.cross(p, q)
        n /= np.linalg.norm(n)
        numbers.append(face.number)
        normals.append(n)
        offsets.append(float(n @ pts.mean(axis=0)))
    return numbers, np.asarray(normals), np.asarray(offsets)


def _uniform_points_on_face(crystal: HexPrism, face_number: int, count: int, rng: np.random.Generator) -> tuple[np.ndarray, float]:
    """Uniform samples on a convex planar face by fan triangulation; returns points and the face area."""
    pts = crystal.vertices[list(crystal.face(face_number).vertex_ids)]
    tris = [(pts[0], pts[i], pts[i + 1]) for i in range(1, len(pts) - 1)]
    areas = np.array([0.5 * np.linalg.norm(np.cross(b - a, c - a)) for a, b, c in tris])
    choice = rng.choice(len(tris), size=count, p=areas / areas.sum())
    u, v = rng.random(count), rng.random(count)
    flip = u + v > 1.0
    u[flip], v[flip] = 1.0 - u[flip], 1.0 - v[flip]
    out = np.empty((count, 3))
    for k, (a, b, c) in enumerate(tris):
        sel = choice == k
        out[sel] = a + u[sel, None] * (b - a) + v[sel, None] * (c - a)
    return out, float(areas.sum())


def entry_measure_ray_cast(
    rotation: np.ndarray,
    incident_direction: np.ndarray,
    crystal: HexPrism,
    *,
    path: tuple[int, int] = PATH,
    n_ice: float = CANONICAL_REFRACTIVE_INDEX,
    samples: int = 20_000,
    rng: np.random.Generator | None = None,
    exit_faces: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    """Monte Carlo effective entry cross-section of the two-face path ``path`` (body frame).

    ``exit_faces`` overrides the accepted exit set (``(path[1],)`` by default;
    the self-check passes every face so the fraction must be one).
    """
    rng = np.random.default_rng(0) if rng is None else rng
    R = np.asarray(rotation, dtype=np.float64)
    s_body = R.T @ (np.asarray(incident_direction, dtype=np.float64) / np.linalg.norm(incident_direction))
    numbers, normals, offsets = _face_planes(crystal)
    entry_index = numbers.index(path[0])
    n_a = normals[entry_index]
    cos_i = -float(n_a @ s_body)
    if cos_i <= 0.0:
        return {"value": 0.0, "fraction": 0.0, "status": "entry_backface", "cos_i": cos_i}
    eta = 1.0 / n_ice
    d_in = eta * s_body + (eta * cos_i - np.sqrt(1.0 - eta * eta * (1.0 - cos_i * cos_i))) * n_a
    d_in /= np.linalg.norm(d_in)
    points, face_area = _uniform_points_on_face(crystal, path[0], samples, rng)
    # first plane crossed while leaving: t_f = (offset_f - n_f . p) / (n_f . d), over faces with n_f . d > 0
    along = normals @ d_in
    leaving = along > 1e-12
    t = np.full((samples, len(numbers)), np.inf)
    t[:, leaving] = (offsets[leaving][None, :] - points @ normals[leaving].T) / along[leaving][None, :]
    t[:, entry_index] = np.inf
    exit_index = np.argmin(t, axis=1)
    accepted = set(exit_faces if exit_faces is not None else (path[1],))
    hit = np.isin(np.asarray(numbers)[exit_index], list(accepted))
    # exit-face TIR gate: the whole footprint fails together (one internal direction)
    n_b = normals[numbers.index(path[1])]
    cos_exit_internal = float(n_b @ d_in)
    tir = (n_ice * n_ice) * (1.0 - cos_exit_internal * cos_exit_internal) >= 1.0 if cos_exit_internal > 0 else True
    fraction = float(hit.mean())
    value = 0.0 if (tir and exit_faces is None) else fraction * face_area * cos_i
    return {
        "value": value,
        "fraction": fraction,
        "fraction_stderr": float(np.sqrt(max(fraction * (1.0 - fraction), 0.0) / samples)),
        "status": "exit_critical_angle" if (tir and exit_faces is None) else "ok",
        "cos_i": cos_i,
        "face_area": face_area,
    }


def self_check_entry(crystal: HexPrism, incident_direction: np.ndarray) -> dict[str, Any]:
    """With every face accepted as exit the fraction is exactly one; the face area of a unit prism side is ``a * h``."""
    probe = entry_measure_ray_cast(np.eye(3), incident_direction, crystal, exit_faces=tuple(f.number for f in crystal.faces), samples=5_000)
    return {"all_faces_fraction": probe["fraction"], "face_area": probe["face_area"], "expected_face_area": crystal.a * crystal.h}


# ---------- production factors along the fiber ----------
def c_axis_zenith(poses: np.ndarray) -> np.ndarray:
    return np.degrees(np.arccos(np.clip(poses[:, 2, 2], -1.0, 1.0)))


def summarise(values: np.ndarray) -> dict[str, float]:
    v = np.asarray(values, dtype=np.float64)
    return {"min": float(v.min()), "median": float(np.median(v)), "max": float(v.max()), "mean": float(v.mean())}


def probe_pixel(scene, row: int, column: int, options: PixelOptions, *, mc_samples: int, mc_poses: int) -> dict[str, Any]:
    result = render_pixel(scene, row, column, options)
    target = pixel_target(scene.render, row, column)
    out: dict[str, Any] = {
        "row": row,
        "column": column,
        "value": result.value,
        "completeness": result.completeness,
        "component_count": result.component_count,
        "components": [],
    }
    rng = np.random.default_rng(20260917 + row)
    for record in result.components:
        if not record.integrated:
            out["components"].append({"kind": record.kind, "integrated": False, "reason": record.reason})
            continue
        problem = retarget_problem(scene.production_template, target, record.seed)
        fiber = trace_fiber(problem)
        poses = np.asarray(fiber.poses)
        zenith = c_axis_zenith(poses)
        factors = {name: np.asarray(fiber.weight_observables[name].values, dtype=np.float64) for name in FACTOR_NAMES}
        j_perp = np.asarray([d.normal_jacobian for d in fiber.jacobian_diagnostics], dtype=np.float64)
        arclength = np.concatenate([[0.0], np.cumsum(np.asarray(fiber.arclength_increments))])[: len(poses)]
        integrand = factors["rho_pose"] * factors["entry_measure"] * factors["fresnel_transmission"] * factors["path_validity"] / (j_perp + options.quadrature.epsilon)

        rho_indep = lumice_gauss_density_relative_to_haar(np.radians(zenith), mean_deg=CANONICAL_ZENITH_MEAN_DEG, std_deg=CANONICAL_ZENITH_STD_DEG)
        rho_legacy = lumice_gauss_density_relative_to_haar(np.radians(zenith), mean_deg=CANONICAL_ZENITH_MEAN_DEG, std_deg=CANONICAL_ZENITH_STD_DEG, area_weighted=False)
        rho_ratio = factors["rho_pose"] / rho_indep
        rho_ratio_legacy = factors["rho_pose"] / rho_legacy

        # entry_measure by ray casting on a subset of poses (the MC is the cost)
        index = np.unique(np.linspace(0, len(poses) - 1, mc_poses).astype(int))
        mc = [entry_measure_ray_cast(poses[i], scene.incident_direction, scene.crystal, samples=mc_samples, rng=rng) for i in index]
        mc_values = np.array([m["value"] for m in mc])
        prod_values = factors["entry_measure"][index]
        with np.errstate(divide="ignore", invalid="ignore"):
            entry_ratio = np.where(mc_values > 0, prod_values / mc_values, np.nan)

        # where along the fiber does the integrand mass sit, and what zenith is that
        weights = integrand * np.gradient(arclength) if len(poses) > 1 else integrand
        mass_zenith = float(np.sum(weights * zenith) / np.sum(weights)) if np.sum(weights) > 0 else float("nan")
        # zenith-width sensitivity: reweight the traced integrand by rho(std') / rho(0.5 deg) and
        # integrate along the trace (trapezoid); only the ratio to the 0.5 deg value is reported
        base = np.trapezoid(integrand, arclength) if len(poses) > 1 else float("nan")
        width_sensitivity = {}
        for std in WIDTH_SENSITIVITY_STD_DEG:
            rho_alt = lumice_gauss_density_relative_to_haar(np.radians(zenith), mean_deg=CANONICAL_ZENITH_MEAN_DEG, std_deg=std)
            with np.errstate(divide="ignore", invalid="ignore"):
                alt = np.trapezoid(integrand * np.where(rho_indep > 0, rho_alt / rho_indep, 0.0), arclength) if len(poses) > 1 else float("nan")
            width_sensitivity[f"{std:g}"] = float(alt / base) if base > 0 else float("nan")
        out["components"].append(
            {
                "kind": record.kind,
                "integrated": True,
                "quadrature_value": record.value,
                "pose_count": int(len(poses)),
                "arclength": float(arclength[-1]),
                "zenith_deg": summarise(zenith),
                "integrand_weighted_mean_zenith_deg": mass_zenith,
                "value_ratio_if_zenith_std_deg": width_sensitivity,
                "zenith_offset_from_mean_deg_at_integrand_peak": float(zenith[np.argmax(integrand)] - CANONICAL_ZENITH_MEAN_DEG),
                "factors": {name: summarise(values) for name, values in factors.items()},
                "j_perp": summarise(j_perp),
                "integrand": summarise(integrand),
                "rho_pose_over_independent": summarise(rho_ratio),
                "rho_pose_over_independent_legacy_no_sin": summarise(rho_ratio_legacy),
                "entry_measure_over_ray_cast": {
                    "poses_checked": int(len(index)),
                    "mc_samples_per_pose": mc_samples,
                    "ratio": summarise(entry_ratio[np.isfinite(entry_ratio)]) if np.isfinite(entry_ratio).any() else None,
                    "production_zero_where_mc_positive": int(np.sum((prod_values == 0) & (mc_values > 0))),
                    "mc_zero_where_production_positive": int(np.sum((mc_values == 0) & (prod_values > 0))),
                    "mc_status": sorted({m["status"] for m in mc}),
                },
                # per-pose export (decimated to keep the JSON readable)
                "profile": {
                    "arclength": arclength[index].tolist(),
                    "zenith_deg": zenith[index].tolist(),
                    **{name: factors[name][index].tolist() for name in FACTOR_NAMES},
                    "j_perp": j_perp[index].tolist(),
                    "integrand": integrand[index].tolist(),
                    "entry_measure_ray_cast": mc_values.tolist(),
                },
            }
        )
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--column", type=int, default=PROBE_COLUMN)
    parser.add_argument("--rows", type=int, nargs="+", default=list(PROBE_ROWS))
    parser.add_argument("--mc-samples", type=int, default=20_000, help="ray-cast samples per checked pose")
    parser.add_argument("--mc-poses", type=int, default=24, help="poses per component checked by ray casting")
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    scene = canonical_strip_scene()
    options = PixelOptions(quadrature=ResampleOptions())
    checks = {"rho_pose": self_check_rho(), "entry_measure": self_check_entry(scene.crystal, scene.incident_direction)}
    print("self-checks:", json.dumps(checks))
    if abs(checks["rho_pose"]["canonical_integral_against_haar"] - 1.0) > 1e-6:
        raise SystemExit("rho_pose independent estimator does not integrate to one against Haar")
    if max(abs(x - 1.0) for x in checks["rho_pose"]["wide_gaussian_ratio_to_haar"]) > 1e-6:
        raise SystemExit("rho_pose independent estimator is not uniform for a very wide Gaussian")
    if checks["entry_measure"]["all_faces_fraction"] != 1.0:
        raise SystemExit("ray-cast estimator loses rays even when every exit face is accepted")

    pixels = [probe_pixel(scene, row, args.column, options, mc_samples=args.mc_samples, mc_poses=args.mc_poses) for row in args.rows]
    report = {
        "generated": dt.datetime.now().astimezone().isoformat(),
        "scene": {
            "zenith_mean_deg": CANONICAL_ZENITH_MEAN_DEG,
            "zenith_std_deg": CANONICAL_ZENITH_STD_DEG,
            "lumice_config_source": "Writing-Lab 06-monte-carlo-vs-integration/data/compare_two_methods/band1e9/config.json: crystal[0].axis.zenith = {type: gauss, mean: 90, std: 0.5}",
            "lumice_sampling_doc": "Lumice doc/crystal-orientation-sampling.md section 4.1: p_sphere(theta) ~ p(theta) sin(theta)",
        },
        "self_checks": checks,
        "pixels": pixels,
    }
    out = args.output_dir / "defect2_probe.json"
    out.write_text(json.dumps(report, indent=2) + "\n")

    for px in pixels:
        print(f"pixel ({px['row']},{px['column']}): value {px['value']:.6g} {px['completeness']} {px['component_count']} component(s)")
        for c in px["components"]:
            if not c["integrated"]:
                print(f"  {c['kind']}: not integrated ({c['reason']})")
                continue
            z = c["zenith_deg"]
            print(
                f"  {c['kind']}: L={c['arclength']:.3f} poses={c['pose_count']} zenith [{z['min']:.3f}, {z['max']:.3f}] deg, "
                f"integrand-peak zenith offset {c['zenith_offset_from_mean_deg_at_integrand_peak']:+.3f} deg"
            )
            for name in FACTOR_NAMES:
                f = c["factors"][name]
                print(f"    {name:22s} [{f['min']:.4g}, {f['max']:.4g}] median {f['median']:.4g}")
            print(f"    {'J_perp':22s} [{c['j_perp']['min']:.4g}, {c['j_perp']['max']:.4g}] median {c['j_perp']['median']:.4g}")
            r = c["rho_pose_over_independent"]
            print(f"    rho_pose / independent: [{r['min']:.6f}, {r['max']:.6f}]   (legacy no-sin: [{c['rho_pose_over_independent_legacy_no_sin']['min']:.6f}, {c['rho_pose_over_independent_legacy_no_sin']['max']:.6f}])")
            ws = ", ".join(f"{k}deg: {v:.3f}" for k, v in c["value_ratio_if_zenith_std_deg"].items())
            print(f"    value ratio if zenith std were ({ws})")
            e = c["entry_measure_over_ray_cast"]
            if e["ratio"]:
                print(f"    entry_measure / ray-cast: [{e['ratio']['min']:.4f}, {e['ratio']['max']:.4f}] median {e['ratio']['median']:.4f} over {e['poses_checked']} poses; zero mismatches {e['production_zero_where_mc_positive']}/{e['mc_zero_where_production_positive']}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
