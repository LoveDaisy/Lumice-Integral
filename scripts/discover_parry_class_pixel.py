"""Locate a pixel where a tilted Parry density separates ``3-5`` from ``3-7`` (AC2, second half).

Why a *tilted* Parry: the roll-locked family with ``roll_mean = 0`` (face 3
up) is invariant under the crystal's ``C2'`` rotation about the face-3 normal
(zenith ``theta -> pi - theta``, roll ``psi -> -psi``), and that rotation maps
the labelled path ``3-5`` onto ``3-7``; so under the stock Parry density the
two members have identical images, exactly like the column density (the
first half of AC2).  A roll mean away from ``0`` and ``180`` deg breaks that
symmetry; ``build_pose_density("parry", roll_mean_deg=...)`` exposes it.

Procedure (parameters are the ``--`` options, defaults recorded in
``tests/test_path_class_ac2_symmetry.py``):

1. Build the ``3-5`` and ``3-7`` prescan tables of the canonical incident
   direction (``--prescan-samples`` Haar poses, ``--prescan-seed``).
2. Weight each table's poses by the tilted Parry density *widened by
   ``--selection-widening``* (a 1 deg x 1 deg lock leaves only a handful of
   significant Haar samples out of 400k, too few for a map) and bin the
   outgoing *sky* directions on a ``--bin-deg`` elevation/azimuth grid (the
   landing map of ``scripts/compare_pose_density_families.py``).
3. Among the bins where ``3-5`` carries at least ``--min-fraction`` of its
   weight and ``3-7`` carries none (``--zero-fraction`` of its own maximum),
   take the bin of largest ``3-5`` weight; fall back to the bin of largest
   ``|log(w_35 / w_37)|`` among bins where both are significant if no such
   bin exists (and say so).
4. Render the whole ``[3, 5]`` class at that sky direction: a ``--window``
   pixel wide linear camera of ``--fov-deg`` centred on the bin centre, the
   centre pixel through :func:`lumice_integral.path_class.render_class_pixel`,
   under the tilted density *and* under the stock Parry density (roll mean
   ``0``) at the same pixel.  The stock render is the symmetry evidence
   (``3-5`` and ``3-7`` agree to quadrature tolerance), the tilted render the
   separation.  Print every member's value, the class value / ``12 x 3-5``
   ratio, and the constants to freeze.

Run::

    uv run python scripts/discover_parry_class_pixel.py --roll-mean-deg 20
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np

from lumice_integral.canonical_scene import CANONICAL_REFRACTIVE_INDEX, canonical_incident_direction
from lumice_integral.path_class import canonical_class_scene, render_class_pixel
from lumice_integral.pose_density import PoseDensity, build_pose_density
from lumice_integral.prescan import PrescanTable, build_prescan_table
from lumice_integral.strip_pixel import PixelOptions


def landing_histogram(table: PrescanTable, density: PoseDensity, edges_elevation: np.ndarray, edges_azimuth: np.ndarray) -> np.ndarray:
    sky = -np.asarray(table.directions, dtype=np.float64)
    elevation = np.degrees(np.arcsin(np.clip(sky[:, 2], -1.0, 1.0)))
    azimuth = np.degrees(np.arctan2(sky[:, 1], sky[:, 0]))
    weight = density.evaluate_batch(table.rotations)
    histogram, _, _ = np.histogram2d(elevation, azimuth, bins=[edges_elevation, edges_azimuth], weights=weight)
    return histogram


def choose_bin(w35: np.ndarray, w37: np.ndarray, *, min_fraction: float, zero_fraction: float) -> tuple[tuple[int, int], str]:
    total = w35.sum()
    strong = (w35 >= min_fraction * total) & (w37 <= zero_fraction * w37.max())
    if strong.any():
        candidates = np.where(strong, w35, -np.inf)
        index = np.unravel_index(int(np.argmax(candidates)), w35.shape)
        return (int(index[0]), int(index[1])), "strong: 3-7 carries no weight where 3-5 does"
    both = (w35 >= min_fraction * total) & (w37 >= min_fraction * w37.sum())
    if not both.any():
        raise SystemExit("no bin separates the two members; widen --bin-deg or lower --min-fraction")
    ratio = np.where(both, np.abs(np.log(w35 / np.where(w37 > 0, w37, np.nan))), -np.inf)
    index = np.unravel_index(int(np.nanargmax(ratio)), w35.shape)
    return (int(index[0]), int(index[1])), "weak: both members carry weight, ratio farthest from 1"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--roll-mean-deg", type=float, default=20.0)
    parser.add_argument("--roll-std-deg", type=float, default=1.0)
    parser.add_argument("--zenith-std-deg", type=float, default=1.0)
    parser.add_argument("--prescan-samples", type=int, default=400_000)
    parser.add_argument("--prescan-seed", type=int, default=20260916)
    parser.add_argument("--bin-deg", type=float, default=1.0)
    parser.add_argument("--min-fraction", type=float, default=0.01, help="bin weight as a fraction of the member's total")
    parser.add_argument("--zero-fraction", type=float, default=1e-9, help="'no weight' threshold relative to the member's maximum bin")
    parser.add_argument("--selection-widening", type=float, default=5.0, help="zenith/roll width multiplier of the landing map density")
    parser.add_argument("--window", type=int, default=21)
    parser.add_argument("--fov-deg", type=float, default=6.0)
    args = parser.parse_args(argv)

    incident = canonical_incident_direction()
    tables = {
        path_id: build_prescan_table(
            incident, CANONICAL_REFRACTIVE_INDEX, sample_count=args.prescan_samples, rng_seed=args.prescan_seed, path_id=path_id
        )
        for path_id in ("3-5", "3-7")
    }
    edges_elevation = np.arange(-30.0, 90.0 + args.bin_deg, args.bin_deg)
    edges_azimuth = np.arange(-90.0, 90.0 + args.bin_deg, args.bin_deg)

    stock = build_pose_density("parry", zenith_std_deg=args.zenith_std_deg, roll_std_deg=args.roll_std_deg)
    tilted = build_pose_density(
        "parry", zenith_std_deg=args.zenith_std_deg, roll_mean_deg=args.roll_mean_deg, roll_std_deg=args.roll_std_deg
    )
    selector = build_pose_density(
        "parry",
        zenith_std_deg=args.zenith_std_deg * args.selection_widening,
        roll_mean_deg=args.roll_mean_deg,
        roll_std_deg=args.roll_std_deg * args.selection_widening,
    )
    maps = {path_id: landing_histogram(table, selector, edges_elevation, edges_azimuth) for path_id, table in tables.items()}
    (i, j), rationale = choose_bin(maps["3-5"], maps["3-7"], min_fraction=args.min_fraction, zero_fraction=args.zero_fraction)
    elevation = float(0.5 * (edges_elevation[i] + edges_elevation[i + 1]))
    azimuth = float(0.5 * (edges_azimuth[j] + edges_azimuth[j + 1]))
    render = {"width": args.window, "height": args.window, "fov_deg": args.fov_deg, "view": {"azimuth": azimuth, "elevation": elevation}}
    centre = (args.window // 2, args.window // 2)

    def render_members(density: PoseDensity) -> tuple[dict[str, Any], float, str]:
        scene = canonical_class_scene(
            (3, 5), prescan_sample_count=args.prescan_samples, prescan_rng_seed=args.prescan_seed, pose_density=density, render=render
        )
        result = render_class_pixel(scene, centre[0], centre[1], PixelOptions())
        members = {
            "-".join(map(str, member)): {
                "value": r.value, "error_estimate": r.error_estimate, "completeness": r.completeness, "components": r.component_count,
            }
            for member, r in result.members.items()
        }
        return members, result.value, result.completeness

    tilted_members, tilted_value, tilted_completeness = render_members(tilted)
    stock_members, stock_value, stock_completeness = render_members(stock)
    value_3_5 = tilted_members["3-5"]["value"]
    out: dict[str, Any] = {
        "script": "scripts/discover_parry_class_pixel.py",
        "command": f"uv run python scripts/discover_parry_class_pixel.py --roll-mean-deg {args.roll_mean_deg:g}",
        "pose_density": {"family": "parry", "zenith_std_deg": args.zenith_std_deg, "roll_mean_deg": args.roll_mean_deg, "roll_std_deg": args.roll_std_deg},
        "prescan": {"sample_count": args.prescan_samples, "rng_seed": args.prescan_seed},
        "landing_bin_deg": args.bin_deg,
        "selection_widening": args.selection_widening,
        "bin_weight_3_5_fraction": float(maps["3-5"][i, j] / maps["3-5"].sum()),
        "bin_weight_3_7_fraction": float(maps["3-7"][i, j] / maps["3-7"].sum()),
        "selection": rationale,
        "render": render,
        "pixel": {"row": centre[0], "column": centre[1], "sky_elevation_deg": elevation, "sky_azimuth_deg": azimuth},
        "tilted": {
            "members": tilted_members,
            "class_value": tilted_value,
            "class_over_12x_3_5": tilted_value / (12.0 * value_3_5) if value_3_5 else None,
            "completeness": tilted_completeness,
        },
        "stock_parry": {
            "members": stock_members,
            "class_value": stock_value,
            "class_over_12x_3_5": stock_value / (12.0 * stock_members["3-5"]["value"]) if stock_members["3-5"]["value"] else None,
            "completeness": stock_completeness,
        },
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
