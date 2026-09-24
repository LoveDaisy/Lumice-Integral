"""Pose-density family check: a band-sum path-class render against a Lumice float export, absolute and shape.

The band-sum renderer (``scripts/render_band_sum.py --path-class``) writes the
class value ``V(w)`` of every pixel (the sum over the PBD class members,
crystal length^2 per steradian, hexagon edge ``a = 1``).  Lumice's raypath
filter with ``symmetry: PBD`` admits the same 12 raypaths, so the conversion of
``scripts/probe_absolute_scale.py`` holds with the class fold already inside
``V``:

    raw[p] / E = K_p * V(w_p),  K_p = ybar(550) * Omega_p / (S / 2)

(Lumice >= ``6fc48bb4``: every ray's weight is multiplied by ``A_tot / (S/2)``
at entry; ``docs/ch06-reference-fixture.md`` section 7, stage 4).  Nothing is
fitted.  Reported, per family render:

- ``total``: the flux ratio ``sum(raw / E) / sum(K_p V)`` over every pixel lit on
  either side, with the two Lumice seeds' own ratios as its noise;
- ``bright``: pixels whose merged Lumice value is above ``--bright-floor`` of the
  maximum: the per-pixel ratio (median, mean, standard error) and the entry
  area it implies, ``ybar Omega_p V / (raw / E)`` (predicted: ``S / 2`` on
  every pixel, the ratio's reciprocal times ``S / 2``), next to the expected
  per-pixel noise (Lumice: ``|x1 - x2| / (x1 + x2)``, the merged relative
  noise of two i.i.d. seeds; Lumice Integral: ``1 / sqrt(K_eff)`` from the
  band-sum ``pixels.csv``);
- ``regions``: the same flux ratio per image half (top / bottom, left /
  right), so a pose-dependent factor between two halo features shows up as
  two different ratios;
- ``profiles``: the max-normalised row and column through the brightest
  merged Lumice pixel, the RMS of their difference on the lit part (either
  side above ``--profile-floor``) and the same RMS between the two Lumice
  seeds (the noise floor of the profile, ``sqrt(2)`` times the merged one).

The band-sum value is a deviation-band average and Lumice's a pixel-area
average: next to sharp edges they differ by the averaging itself, which the
flux ratio does not see.  Nothing here imports or calls Lumice.  Usage::

    uv run python scripts/compare_lumice_family.py --li-dir <band-sum-dir> \\
        --lumice-run <run1_dir> --lumice-run <run2_dir> --output <metrics.json>
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from lumice_integral.camera import linear_scale
from lumice_integral.canonical_scene import canonical_crystal
from lumice_integral.strip_io import read_strip

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_absolute_scale import YBAR_550, load_run, total_surface_area  # noqa: E402


def pixel_solid_angles(render: dict[str, Any]) -> np.ndarray:
    """``(H, W)`` linear-lens pixel solid angles ``cos^3(theta) / scale^2`` (``probe_absolute_scale.pixel_solid_angle``)."""
    width, height = int(render["width"]), int(render["height"])
    scale = linear_scale(render["fov_deg"], width, height)
    x = ((np.arange(width) + 0.5) - width / 2.0) / scale
    y = ((np.arange(height) + 0.5) - height / 2.0) / scale
    cos_theta = 1.0 / np.sqrt(1.0 + x[None, :] ** 2 + y[:, None] ** 2)
    return cos_theta**3 / scale**2


def check_camera(render: dict[str, Any], config: dict[str, Any]) -> None:
    lumice = config["render"][0]
    expected = ("linear", [render["width"], render["height"]], float(render["fov_deg"]), [float(render["view"]["azimuth"]), float(render["view"]["elevation"])])
    actual = (lumice["lens"]["type"], list(lumice["resolution"]), float(lumice["lens"]["fov"]), [float(lumice["view"]["azimuth"]), float(lumice["view"]["elevation"])])
    if actual != expected:
        raise SystemExit(f"Lumice camera {actual} != band-sum camera {expected}")


def read_k_eff(li_dir: Path, shape: tuple[int, int]) -> np.ndarray:
    k_eff = np.zeros(shape)
    with (li_dir / "pixels.csv").open() as fh:
        for rec in csv.DictReader(fh):
            k_eff[int(rec["row"]), int(rec["column"])] = float(rec["K_eff"])
    return k_eff


def flux_ratio(measured: np.ndarray, predicted: np.ndarray, mask: np.ndarray) -> float:
    return float(measured[mask].sum() / predicted[mask].sum()) if predicted[mask].sum() > 0 else float("nan")


def profile_rms(a: np.ndarray, b: np.ndarray, floor: float) -> tuple[float, int]:
    na, nb = a / a.max(), b / b.max()
    lit = (na > floor) | (nb > floor)
    return float(np.sqrt(np.mean((na[lit] - nb[lit]) ** 2))), int(lit.sum())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--li-dir", type=Path, required=True, help="render_band_sum.py --path-class output directory")
    parser.add_argument("--lumice-run", type=Path, action="append", required=True, help="Lumice run directory (img_01.npy, img_01.json, config.json); two for the noise floor")
    parser.add_argument("--bright-floor", type=float, default=0.1)
    parser.add_argument("--profile-floor", type=float, default=0.05)
    parser.add_argument("--output", type=Path, required=True, help="metrics JSON")
    args = parser.parse_args(argv)

    arrays, provenance = read_strip(args.li_dir)
    value = np.nan_to_num(arrays.values, nan=0.0)
    render = dict(provenance["scene"]["camera"]["value"])
    path = provenance["scene"]["path"]["value"]
    runs = [load_run(d) for d in args.lumice_run]
    for d, (y, _, symmetry) in zip(args.lumice_run, runs):
        if symmetry != "PBD" or y.shape != value.shape:
            raise SystemExit(f"{d}: symmetry {symmetry} / shape {y.shape} (need PBD, {value.shape})")
        config = json.loads((d / "config.json").read_text())
        if config["filter"][0]["raypath"] != path:
            raise SystemExit(f"{d}: raypath {config['filter'][0]['raypath']} != band-sum class representative {path}")
        check_camera(render, config)

    surface_area = total_surface_area(canonical_crystal())
    omega = pixel_solid_angles(render)
    k_pixel = YBAR_550 * omega / (0.5 * surface_area)
    predicted = k_pixel * value
    per_run = [y / m["emitted_energy"] for y, m, _ in runs]
    measured = np.sum([y for y, _, _ in runs], axis=0) / sum(m["emitted_energy"] for _, m, _ in runs)
    k_eff = read_k_eff(args.li_dir, value.shape)

    lit = (measured > 0) | (predicted > 0)
    bright = (measured >= args.bright_floor * measured.max()) & (predicted > 0)
    ratio = measured[bright] / predicted[bright]
    implied = YBAR_550 * omega[bright] * value[bright] / measured[bright]
    out: dict[str, Any] = {
        "generated": dt.datetime.now().astimezone().isoformat(),
        "li_dir": str(args.li_dir),
        "lumice_runs": [str(d) for d in args.lumice_run],
        "pose_density": provenance["scene"].get("pose_density", {}).get("value"),
        "path_class_representative": path,
        "camera": render,
        "emitted_energy": [m["emitted_energy"] for _, m, _ in runs],
        "surface_area": surface_area,
        "entry_area": 0.5 * surface_area,
        "convention": "raw[p] / emitted_energy = K_p * V(p), K_p = ybar(550) * Omega_p / (S / 2); V = band-sum PBD class value",
        "total": {
            "lit_pixels": int(lit.sum()),
            "measured_over_predicted": flux_ratio(measured, predicted, lit),
            "per_run": [flux_ratio(r, predicted, lit) for r in per_run],
            "predicted_flux_share_outside_lumice_lit": float(predicted[lit & (measured == 0)].sum() / predicted[lit].sum()),
            "measured_flux_share_outside_li_lit": float(measured[lit & (predicted == 0)].sum() / measured[lit].sum()),
        },
        "bright": {
            "floor": args.bright_floor,
            "pixels": int(bright.sum()),
            "measured_over_predicted_median": float(np.median(ratio)),
            "measured_over_predicted_mean": float(np.mean(ratio)),
            "measured_over_predicted_standard_error": float(np.std(ratio) / np.sqrt(ratio.size)),
            "measured_over_predicted_relative_std": float(np.std(ratio) / np.mean(ratio)),
            "implied_entry_area_median": float(np.median(implied)),
            "implied_entry_area_p10_p90": [float(np.percentile(implied, 10)), float(np.percentile(implied, 90))],
            "k_pixel_relative_span": float(k_pixel[bright].max() / k_pixel[bright].min() - 1.0),
            "li_relative_noise_rms": float(np.sqrt(np.mean(1.0 / np.maximum(k_eff[bright], 1e-300)))),
        },
        "regions": {},
        "profiles": {},
    }
    if len(per_run) >= 2:
        x1, x2 = per_run[0][bright], per_run[1][bright]
        lumice_noise = np.abs(x1 - x2) / (x1 + x2)
        out["bright"]["lumice_merged_relative_noise_rms"] = float(np.sqrt(np.mean(lumice_noise**2)))
        out["bright"]["expected_ratio_relative_std"] = float(np.sqrt(out["bright"]["lumice_merged_relative_noise_rms"] ** 2 + out["bright"]["li_relative_noise_rms"] ** 2))
    h, w = value.shape
    halves = {
        "top": (slice(0, h // 2), slice(None)),
        "bottom": (slice(h // 2, None), slice(None)),
        "left": (slice(None), slice(0, w // 2)),
        "right": (slice(None), slice(w // 2, None)),
    }
    for name, (rs, cs) in halves.items():
        mask = np.zeros_like(lit)
        mask[rs, cs] = lit[rs, cs]
        if predicted[mask].sum() > 0:
            out["regions"][name] = {
                "measured_over_predicted": flux_ratio(measured, predicted, mask),
                "per_run": [flux_ratio(r, predicted, mask) for r in per_run],
                "predicted_flux_share": float(predicted[mask].sum() / predicted[lit].sum()),
            }
    peak_row, peak_column = np.unravel_index(np.argmax(measured), measured.shape)
    for name, index in (("row", (int(peak_row), slice(None))), ("column", (slice(None), int(peak_column)))):
        rms, count = profile_rms(measured[index], predicted[index], args.profile_floor)
        entry: dict[str, Any] = {"index": int(peak_row if name == "row" else peak_column), "lit": count, "rms_max_normalised": rms}
        if len(per_run) >= 2:
            entry["rms_lumice_run1_vs_run2"] = profile_rms(per_run[0][index], per_run[1][index], args.profile_floor)[0]
        out["profiles"][name] = entry

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    t, b = out["total"], out["bright"]
    print(f"total flux measured/predicted {t['measured_over_predicted']:.4f} (runs {', '.join(f'{r:.4f}' for r in t['per_run'])})")
    print(
        f"bright ({b['pixels']} px): ratio median {b['measured_over_predicted_median']:.4f}, mean {b['measured_over_predicted_mean']:.4f} "
        f"+- {b['measured_over_predicted_standard_error']:.4f}, rel std {b['measured_over_predicted_relative_std']:.4f} "
        f"(expected {b.get('expected_ratio_relative_std', float('nan')):.4f}); implied area {b['implied_entry_area_median']:.3f} (S/2 {0.5 * surface_area:.3f})"
    )
    for name, r in out["regions"].items():
        print(f"region {name}: {r['measured_over_predicted']:.4f} (flux share {r['predicted_flux_share']:.3f})")
    for name, p in out["profiles"].items():
        print(f"profile {name} {p['index']}: rms {p['rms_max_normalised']:.4f} (lumice seeds {p.get('rms_lumice_run1_vs_run2', float('nan')):.4f}, n={p['lit']})")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
