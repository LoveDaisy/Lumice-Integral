"""ch11 pose-density families on the canonical 3-5 scene: profiles, landing map, width sweep.

Every family shares the fibers (discovery and continuation never read
``rho_pose``); only the weight changes.  Three diagnostics, one figure and one
JSON (``pose_density_families.json``):

1. **Column profile.**  ``render_pixel`` on column :data:`PROBE_COLUMN` at
   ``--rows`` (default every ``--row-step`` rows from 150 to 600) for each of
   the five families, plotted on a log axis.  Families whose value is exactly
   zero at every rendered pixel are reported as such in the legend rather
   than silently dropped by the log scale: on the labelled path ``3 -> 5``
   the column-126 fibers have c-axis zenith ``88-92 deg`` and roll
   ``90-155 deg`` (face 3 on the side), so plate (zenith about ``0 deg``),
   parry and lowitz (roll locked at ``0 deg``, face 3 on top) underflow to
   zero there.
2. **Landing map.**  Where each family's ``3 -> 5`` light goes: the Haar
   prescan samples (``prescan.build_prescan_table``) weighted by each family's
   ``rho_pose`` and binned by outgoing sky direction, with the strip's field
   of view outlined.  This is the picture that explains 1: plate's ``3 -> 5``
   light is the parhelion, parry's the upper Parry arc, lowitz's the Lowitz
   arcs, all outside the strip; column's lower tangent arc is inside.
3. **Zenith-width sweep** (column family, ``defect2_findings.md`` section 4):
   on the four probe pixels the traced integrand is reweighted by
   ``rho(std') / rho(0.5 deg)`` and trapezoid-integrated along arclength; the
   ratio to the ``0.5 deg`` value is reported (:func:`zenith_width_sensitivity`,
   also the authority of ``tests/test_pose_density_zenith_sensitivity.py``).

Plotting needs matplotlib, which is not a project dependency::

    uv run --with matplotlib python scripts/compare_pose_density_families.py \\
        --output-dir <task>/artifacts

The probe positions (column ``126``, rows ``150 / 300 / 450 / 600``) are shared
with ``scripts/probe_defect2_factors.py``; keep :data:`PROBE_COLUMN` /
:data:`PROBE_ROWS` in sync with it.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from lumice_integral.camera import project_linear
from lumice_integral.canonical_scene import (
    CANONICAL_REFRACTIVE_INDEX,
    CANONICAL_RENDER,
    CANONICAL_ZENITH_MEAN_DEG,
    CANONICAL_ZENITH_STD_DEG,
    canonical_incident_direction,
)
from lumice_integral.continuation import FiberResult, trace_fiber
from lumice_integral.discovery import retarget_problem
from lumice_integral.pose_density import PoseDensity, build_pose_density
from lumice_integral.pose_density_provenance import pose_density_provenance
from lumice_integral.prescan import PrescanTable, build_prescan_table
from lumice_integral.quadrature import pointwise_integrand
from lumice_integral.strip_pixel import PixelOptions, StripScene, canonical_strip_scene, pixel_target, render_pixel

PROBE_COLUMN = 126
PROBE_ROWS = (150, 300, 450, 600)
WIDTH_SENSITIVITY_STD_DEG = (0.25, 0.5, 1.0, 2.0)

# The five ch11 families at the Lumice ``kAxisPresets`` widths (column at the
# canonical 0.5 deg, plate at the same width for comparability; parry/lowitz
# at the preset 1 deg zenith / 1 deg roll and 40 deg / 1 deg).
FAMILY_PARAMETERS: dict[str, dict[str, float]] = {
    "random": {},
    "plate": {"zenith_std_deg": CANONICAL_ZENITH_STD_DEG},
    "column": {"zenith_std_deg": CANONICAL_ZENITH_STD_DEG},
    "parry": {"zenith_std_deg": 1.0, "roll_std_deg": 1.0},
    "lowitz": {"zenith_std_deg": 40.0, "roll_std_deg": 1.0},
}


def family_densities(parameters: Mapping[str, Mapping[str, float]] = FAMILY_PARAMETERS) -> dict[str, PoseDensity]:
    return {family: build_pose_density(family, **kwargs) for family, kwargs in parameters.items()}


def fiber_arclength(fiber: FiberResult) -> np.ndarray:
    """Cumulative arclength aligned with ``fiber.poses`` (``0`` at the first pose)."""
    increments = np.asarray(fiber.arclength_increments, dtype=np.float64)
    return np.concatenate([[0.0], np.cumsum(increments)])[: len(fiber.poses)]


def zenith_width_sensitivity(
    fiber: FiberResult,
    *,
    base_density: PoseDensity,
    epsilon: float,
    zenith_mean_deg: float = CANONICAL_ZENITH_MEAN_DEG,
    std_deg: tuple[float, ...] = WIDTH_SENSITIVITY_STD_DEG,
) -> dict[str, float]:
    """``I(std') / I(base)`` with ``I(std) = int rho_std W_P / (J_perp + eps) ds`` (trapezoid).

    ``fiber`` is a production trace whose ``rho_pose`` observable is
    ``base_density``; the integrand is reweighted pose by pose by
    ``rho_std'(R) / rho_base(R)`` (both from :class:`.ZenithGaussianPoseDensity`,
    so the ratio is the exact Gaussian-width ratio, no quadrature involved) and
    the ratio of the two trapezoid integrals along arclength is returned per
    width.  Poses where the base density is zero contribute nothing.
    """
    poses = np.asarray(fiber.poses, dtype=np.float64)
    arclength = fiber_arclength(fiber)
    integrand = pointwise_integrand(fiber, epsilon=epsilon)
    base = base_density.evaluate_batch(poses)
    base_value = float(np.trapezoid(integrand, arclength))
    ratios: dict[str, float] = {}
    for std in std_deg:
        density = build_pose_density("column", zenith_mean_deg=zenith_mean_deg, zenith_std_deg=std)
        with np.errstate(divide="ignore", invalid="ignore"):
            reweighted = np.where(base > 0.0, density.evaluate_batch(poses) / base, 0.0)
        ratios[f"{std:g}"] = float(np.trapezoid(integrand * reweighted, arclength) / base_value)
    return ratios


def probe_width_sensitivity(scene: StripScene, options: PixelOptions, rows: tuple[int, ...] = PROBE_ROWS, column: int = PROBE_COLUMN) -> dict[str, Any]:
    """Section 3 of the module docstring on the probe pixels of the column scene (one component each)."""
    out: dict[str, Any] = {}
    for row in rows:
        result = render_pixel(scene, row, column, options)
        target = pixel_target(scene.render, row, column)
        components = []
        for record in result.components:
            if not record.integrated:
                continue
            fiber = trace_fiber(retarget_problem(scene.production_template, target, record.seed))
            components.append(
                {
                    "kind": record.kind,
                    "pose_count": int(len(fiber.poses)),
                    "arclength": float(fiber_arclength(fiber)[-1]),
                    "value_ratio_if_zenith_std_deg": zenith_width_sensitivity(
                        fiber, base_density=scene.pose_density, epsilon=options.quadrature.epsilon
                    ),
                }
            )
        out[str(row)] = {"value": result.value, "completeness": result.completeness, "components": components}
    return out


def render_profiles(
    densities: Mapping[str, PoseDensity], rows: tuple[int, ...], column: int, prescan_table: PrescanTable, options: PixelOptions
) -> dict[str, dict[str, Any]]:
    """Section 1: ``render_pixel`` per family on the shared prescan table."""
    profiles: dict[str, dict[str, Any]] = {}
    for family, density in densities.items():
        scene = canonical_strip_scene(prescan_table=prescan_table, pose_density=density)
        values, errors, complete, seconds = [], [], [], []
        for row in rows:
            result = render_pixel(scene, row, column, options)
            values.append(result.value)
            errors.append(result.error_estimate)
            complete.append(result.completeness == "complete")
            seconds.append(result.timings["total_s"])
        profiles[family] = {
            "rows": list(rows),
            "value": values,
            "error_estimate": errors,
            "complete": complete,
            "total_s": float(sum(seconds)),
            "all_zero": bool(np.all(np.asarray(values) == 0.0)),
        }
    return profiles


def landing_map(densities: Mapping[str, PoseDensity], table: PrescanTable, *, bins_deg: float = 1.0) -> dict[str, Any]:
    """Section 2: family-weighted prescan samples binned by sky elevation / azimuth."""
    sky = -np.asarray(table.directions, dtype=np.float64)
    elevation = np.degrees(np.arcsin(np.clip(sky[:, 2], -1.0, 1.0)))
    azimuth = np.degrees(np.arctan2(sky[:, 1], sky[:, 0]))
    inside = np.zeros(len(sky), dtype=bool)
    for i, direction in enumerate(sky):
        try:
            u, v = project_linear(direction, **CANONICAL_RENDER)
        except ValueError:
            continue
        inside[i] = 0.0 <= u < CANONICAL_RENDER["width"] and 0.0 <= v < CANONICAL_RENDER["height"]
    edges_elevation = np.arange(-30.0, 90.0 + bins_deg, bins_deg)
    edges_azimuth = np.arange(-90.0, 90.0 + bins_deg, bins_deg)
    out: dict[str, Any] = {
        "elevation_edges_deg": edges_elevation.tolist(),
        "azimuth_edges_deg": edges_azimuth.tolist(),
        "valid_samples": int(table.valid_count),
        "inside_strip_samples": int(inside.sum()),
        "families": {},
    }
    for family, density in densities.items():
        weight = density.evaluate_batch(table.rotations)
        total = float(weight.sum())
        histogram, _, _ = np.histogram2d(elevation, azimuth, bins=[edges_elevation, edges_azimuth], weights=weight)
        significant = weight > 1e-3 * weight.max() if total > 0 else np.zeros_like(inside)
        out["families"][family] = {
            "total_weight": total,
            "inside_strip_fraction": float(weight[inside].sum() / total) if total > 0 else float("nan"),
            "significant_samples": int(significant.sum()),
            "elevation_percentiles_deg": np.percentile(elevation[significant], [1, 50, 99]).tolist() if significant.any() else None,
            "azimuth_percentiles_deg": np.percentile(azimuth[significant], [1, 50, 99]).tolist() if significant.any() else None,
            "histogram": histogram,
        }
    return out


def strip_outline() -> tuple[np.ndarray, np.ndarray]:
    """Sky elevation / azimuth (deg) of the strip's border, for the landing-map panel."""
    height, width = CANONICAL_RENDER["height"], CANONICAL_RENDER["width"]
    border = [(0, c) for c in range(0, width + 1, 10)] + [(r, width) for r in range(0, height + 1, 10)]
    border += [(height, c) for c in range(width, -1, -10)] + [(r, 0) for r in range(height, -1, -10)]
    sky = np.array([-pixel_target(CANONICAL_RENDER, r - 0.5, c - 0.5) for r, c in border])
    return np.degrees(np.arcsin(sky[:, 2])), np.degrees(np.arctan2(sky[:, 1], sky[:, 0]))


def plot(profiles: Mapping[str, Mapping[str, Any]], landing: Mapping[str, Any], column: int, output: Path, label: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    families = list(profiles)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), gridspec_kw={"height_ratios": [1.0, 1.0]})
    ax = axes[0, 0]
    for family in families:
        profile = profiles[family]
        rows = np.asarray(profile["rows"])
        values = np.asarray(profile["value"])
        if profile["all_zero"]:
            ax.plot([], [], label=f"{family}: 0 at every rendered pixel")
            continue
        positive = values > 0
        ax.plot(rows[positive], values[positive], marker="o", markersize=3, label=family)
        if (~positive).any():
            ax.plot(rows[~positive], np.full((~positive).sum(), np.nan), "x")
    ax.set_yscale("log")
    ax.set_xlabel("row")
    ax.set_ylabel("pixel value (Haar-converted, path 3-5)")
    ax.set_title(f"column {column} profile by pose-density family {label}".strip())
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    outline_elevation, outline_azimuth = strip_outline()
    edges_elevation = np.asarray(landing["elevation_edges_deg"])
    edges_azimuth = np.asarray(landing["azimuth_edges_deg"])
    for ax, family in zip(axes.flat[1:], families):
        histogram = np.asarray(landing["families"][family]["histogram"])
        positive = histogram[histogram > 0]
        if positive.size:
            ax.pcolormesh(
                edges_azimuth, edges_elevation, histogram,
                norm=LogNorm(vmin=max(positive.min(), positive.max() * 1e-4), vmax=positive.max()),
                cmap="viridis", shading="flat",
            )
        ax.plot(outline_azimuth, outline_elevation, "r-", linewidth=1.0, label="strip field of view")
        fraction = landing["families"][family]["inside_strip_fraction"]
        ax.set_title(f"{family}: 3-5 light on the sky, {100 * fraction:.1f}% inside the strip")
        ax.set_xlabel("sky azimuth (deg)")
        ax.set_ylabel("sky elevation (deg)")
        ax.set_xlim(-60, 60)
        ax.set_ylim(-30, 65)
        ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(output, dpi=130)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--column", type=int, default=PROBE_COLUMN)
    parser.add_argument("--rows", type=int, nargs="+", default=None, help="profile rows (default: 150..600 every --row-step)")
    parser.add_argument("--row-step", type=int, default=25)
    parser.add_argument("--prescan-samples", type=int, default=400_000, help="Haar samples of the shared prescan table")
    parser.add_argument("--prescan-seed", type=int, default=20260916)
    parser.add_argument("--label", default="")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = tuple(args.rows) if args.rows else tuple(range(150, 601, args.row_step))

    incident = canonical_incident_direction()
    table = build_prescan_table(incident, CANONICAL_REFRACTIVE_INDEX, sample_count=args.prescan_samples, rng_seed=args.prescan_seed)
    densities = family_densities()
    options = PixelOptions()

    print("landing map ...", flush=True)
    landing = landing_map(densities, table)
    for family, entry in landing["families"].items():
        print(f"  {family:7s} inside-strip fraction {entry['inside_strip_fraction']:.3f}  elevation p1/50/99 {entry['elevation_percentiles_deg']}  azimuth {entry['azimuth_percentiles_deg']}")

    print(f"profiles on column {args.column}, {len(rows)} rows x {len(densities)} families ...", flush=True)
    profiles = render_profiles(densities, rows, args.column, table, options)
    for family, profile in profiles.items():
        print(f"  {family:7s} {profile['total_s']:.1f} s  all_zero={profile['all_zero']}  max={max(profile['value']):.4g}")

    print("zenith-width sweep (column family) ...", flush=True)
    column_scene = canonical_strip_scene(prescan_table=table, pose_density=densities["column"])
    sweep = probe_width_sensitivity(column_scene, options, column=args.column)
    for row, entry in sweep.items():
        print(f"  row {row}: {entry['components'][0]['value_ratio_if_zenith_std_deg'] if entry['components'] else 'no component'}")

    report = {
        "generated": dt.datetime.now().astimezone().isoformat(),
        "label": args.label,
        "column": args.column,
        "families": {family: pose_density_provenance(family, **kwargs) for family, kwargs in FAMILY_PARAMETERS.items()},
        "prescan": {"sample_count": args.prescan_samples, "rng_seed": args.prescan_seed, "valid_count": table.valid_count},
        "profiles": profiles,
        "landing": {
            key: value for key, value in landing.items() if key != "families"
        } | {"families": {f: {k: v for k, v in e.items() if k != "histogram"} for f, e in landing["families"].items()}},
        "zenith_width_sensitivity": sweep,
    }
    (args.output_dir / "pose_density_families.json").write_text(json.dumps(report, indent=2))
    if not args.no_figures:
        plot(profiles, landing, args.column, args.output_dir / "pose_density_families.png", args.label)
    print("wrote", args.output_dir / "pose_density_families.json")


if __name__ == "__main__":
    main()
