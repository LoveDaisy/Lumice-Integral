"""Render check of the weight-kink colour criterion (``lumice_integral.chromatic``) with the S^2 band sum.

Random orientation makes the image a function of ``delta`` alone, so one
camera column in the anti-sun vertical plane is the whole profile: sun at the
canonical 15 deg, view azimuth 180, elevation 40, a ``3 x 301`` window whose
301 rows span ``delta ~ 100 .. 150`` deg (``~0.17`` deg per pixel; the linear
lens's ``--fov-deg`` is the width's, so it is ``0.534`` deg for 50 deg of
height).  Per path class, three renders (``scripts/render_band_sum.py``, the
class under ``G_true``): monochrome at ``N_RED`` and at ``N_BLUE``, and D65
colour (Lumice's 5-slot pool, linear CIE XYZ).  Checks:

- ``3-1-6``: the dark hole's rim.  The profile at each index drops at
  ``delta = 2 arcsin sqrt(n^2 - 1)`` (the steepest drop, compared in pixels
  with the prediction; the hole radius is ``180 - delta``), the blue/red
  ratio exceeds ``TINT_RATIO_MIN`` between the two rims, and the colour
  render's ``z`` chromaticity is higher there than on the total side.
- ``3-1-5``: the red-edge candidate (the exit Snell gate, ``color = red``,
  ``visible = False`` by the criterion: its image spreads over ~110 deg of
  ``delta``) and the internal kink (``color = blue``, a band between the red
  and the blue kink images).  Sharp colour steps of the smoothed blue/red
  ratio are listed (:func:`_colour_steps`); a red edge is a step into a red
  side (ratio ``<= 1 / TINT_RATIO_MIN``) outside the blue kink's own
  ``delta`` interval.

Renders go to ``--output-dir/<case>-{red,blue,d65}`` (reused when present);
``summary.json`` there holds every number.  Stores are cached under
``artifacts/s2-store`` (``render_band_sum.py``'s default).

    uv run python scripts/verify_chromatic_kink.py --output-dir /tmp/chromatic-kink
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from lumice_integral import chromatic
from lumice_integral.dp_field import DPField
from lumice_integral.geometry import HexPrism

ROOT = Path(__file__).resolve().parent.parent
CAMERA = ["--width", "3", "--height", "301", "--fov-deg", "0.534", "--view-azimuth", "180", "--view-elevation", "40"]
CASES = {"316": (3, 1, 6), "315": (3, 1, 5)}
# pixels below this fraction of the profile's maximum are too dark for a colour to be seen or measured
LIT_FRACTION = 0.01
# a colour step is measured across +-1 deg (twice the solar disc, which smears anything narrower) ...
STEP_HALF_WIDTH_DEG = 1.0
# ... on the blue / red ratio after a running median over this many pixels (~0.85 deg)
SMOOTH_PIXELS = 5


def _render(case: str, label: str, faces: tuple[int, ...], args: argparse.Namespace) -> Path:
    out = args.output_dir / f"{case}-{label}"
    if (out / "pixels.csv").exists():
        return out
    colour = ["--illuminant", "D65", "--wavelength-count", "panel_wide_fov"] if label == "d65" else [
        "--refractive-index", repr(chromatic.N_RED if label == "red" else chromatic.N_BLUE)
    ]
    command = [
        sys.executable, str(ROOT / "scripts" / "render_band_sum.py"), "--store-n", str(args.store_n),
        "--path", *map(str, faces), "--path-class", "--pose-density-family", "random", *CAMERA, *colour,
        "--workers", str(args.workers), "--output-dir", str(out), "--quiet",
    ]  # fmt: skip
    env = {**os.environ, "JAX_PLATFORMS": "cpu", "OMP_NUM_THREADS": "1"}
    subprocess.run(command, check=True, cwd=ROOT, env=env)
    return out


def _profile(directory: Path, columns: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    """``delta`` (deg) and the requested columns of the middle camera column, sorted by ``delta``."""
    with open(directory / "pixels.csv") as handle:
        rows = [row for row in csv.DictReader(handle) if row["column"] == "1"]
    delta = np.array([float(row["delta_deg"]) for row in rows])
    values = np.array([[float(row[c]) for c in columns] for row in rows])
    order = np.argsort(delta)
    return delta[order], values[order]


def _steepest_drop(delta: np.ndarray, value: np.ndarray, lo: float, hi: float) -> float:
    """``delta`` of the most negative ``dI / d delta`` inside ``[lo, hi]`` (midpoint of that pixel pair)."""
    slope = np.diff(value) / np.diff(delta)
    middle = 0.5 * (delta[1:] + delta[:-1])
    inside = (middle >= lo) & (middle <= hi)
    return float(middle[inside][np.argmin(slope[inside])])


def _chromaticity_z(xyz: np.ndarray) -> np.ndarray:
    total = xyz.sum(axis=1)
    return np.where(total > 0.0, xyz[:, 2] / np.where(total > 0.0, total, 1.0), np.nan)


def check_316(args: argparse.Namespace) -> dict:
    faces = CASES["316"]
    delta, red = _profile(_render("316", "red", faces, args), ("value",))
    _, blue = _profile(_render("316", "blue", faces, args), ("value",))
    delta_xyz, xyz = _profile(_render("316", "d65", faces, args), ("X", "Y", "Z"))
    red, blue = red[:, 0], blue[:, 0]
    pixel_deg = float(np.median(np.diff(delta)))
    predicted = {label: float(np.degrees(2.0 * np.arcsin(np.sqrt(n * n - 1.0)))) for label, n in (("red", chromatic.N_RED), ("blue", chromatic.N_BLUE))}
    lo, hi = predicted["red"] - 5.0, predicted["blue"] + 5.0
    measured = {"red": _steepest_drop(delta, red, lo, hi), "blue": _steepest_drop(delta, blue, lo, hi)}
    fringe = (delta > predicted["red"] + pixel_deg) & (delta < predicted["blue"] - pixel_deg)
    total_side = (delta > predicted["red"] - 5.0) & (delta < predicted["red"] - pixel_deg)
    ratio = blue / np.where(red > 0.0, red, np.nan)
    z = _chromaticity_z(xyz)
    fringe_xyz = (delta_xyz > predicted["red"] + pixel_deg) & (delta_xyz < predicted["blue"] - pixel_deg)
    total_xyz = (delta_xyz > predicted["red"] - 5.0) & (delta_xyz < predicted["red"] - pixel_deg)
    verdict = chromatic.diagnose(HexPrism.from_ratio(2.0), faces)
    return {
        "pixel_deg": pixel_deg,
        "rim_predicted_deg": predicted,
        "rim_measured_deg": measured,
        "rim_error_pixels": {k: abs(measured[k] - predicted[k]) / pixel_deg for k in predicted},
        "hole_radius_predicted_deg": {k: 180.0 - v for k, v in predicted.items()},
        "hole_radius_measured_deg": {k: 180.0 - v for k, v in measured.items()},
        "colour_steps": _colour_steps(delta, red, blue, (red > LIT_FRACTION * red.max()) & (blue > LIT_FRACTION * blue.max())),
        "blue_over_red_in_fringe_median": float(np.nanmedian(ratio[fringe])),
        "blue_over_red_on_total_side_median": float(np.nanmedian(ratio[total_side])),
        "z_chromaticity_fringe_median": float(np.nanmedian(z[fringe_xyz])),
        "z_chromaticity_total_side_median": float(np.nanmedian(z[total_xyz])),
        "criterion": {"kind": verdict.kind, "color": verdict.color, "visible": verdict.visible, "position_deg": float(np.degrees(verdict.position))},
    }


def _colour_steps(delta: np.ndarray, red: np.ndarray, blue: np.ndarray, lit: np.ndarray) -> list[dict]:
    """Sharp colour steps: ``|log(ratio(delta + w) / ratio(delta - w))| >= log TINT_RATIO_MIN`` with ``w = STEP_HALF_WIDTH_DEG``.

    ``ratio = I_blue / I_red`` after a ``SMOOTH_PIXELS`` running median (Monte Carlo noise of the stores is a few
    per cent per pixel).  Adjacent pixels of one step are merged; each step reports its centre, the two side
    ratios and ``into_red`` (the lower side is at or below ``1 / TINT_RATIO_MIN``).
    """
    ratio = np.full(len(delta), np.nan)
    ratio[lit] = blue[lit] / red[lit]
    half = SMOOTH_PIXELS // 2
    windows = [ratio[max(0, i - half) : i + half + 1] for i in range(len(ratio))]
    smooth = np.array([np.nanmedian(w) if np.isfinite(w).any() else np.nan for w in windows])
    threshold = np.log(chromatic.TINT_RATIO_MIN)
    steps: list[dict] = []
    for i, centre in enumerate(delta):
        below = np.flatnonzero(np.abs(delta - (centre - STEP_HALF_WIDTH_DEG)) < 0.5 * np.median(np.diff(delta)) + 1e-9)
        above = np.flatnonzero(np.abs(delta - (centre + STEP_HALF_WIDTH_DEG)) < 0.5 * np.median(np.diff(delta)) + 1e-9)
        if not len(below) or not len(above):
            continue
        lo, hi = smooth[below[0]], smooth[above[0]]
        if not (np.isfinite(lo) and np.isfinite(hi)) or abs(np.log(hi / lo)) < threshold:
            continue
        if steps and centre - steps[-1]["last_deg"] <= 2.0 * STEP_HALF_WIDTH_DEG:
            step = steps[-1]
            if abs(np.log(hi / lo)) > abs(np.log(step["ratio_above"] / step["ratio_below"])):
                step.update(delta_deg=float(centre), ratio_below=float(lo), ratio_above=float(hi))
            step["last_deg"] = float(centre)
            continue
        steps.append({"delta_deg": float(centre), "ratio_below": float(lo), "ratio_above": float(hi), "last_deg": float(centre)})
    for step in steps:
        step["into_red"] = bool(min(step["ratio_below"], step["ratio_above"]) <= 1.0 / chromatic.TINT_RATIO_MIN)
        del step["last_deg"]
    return steps


def check_315(args: argparse.Namespace) -> dict:
    faces = CASES["315"]
    delta, red = _profile(_render("315", "red", faces, args), ("value",))
    _, blue = _profile(_render("315", "blue", faces, args), ("value",))
    delta_xyz, xyz = _profile(_render("315", "d65", faces, args), ("X", "Y", "Z"))
    red, blue = red[:, 0], blue[:, 0]
    lit = (red > LIT_FRACTION * red.max()) & (blue > LIT_FRACTION * blue.max())
    verdict = chromatic.diagnose(HexPrism.from_ratio(2.0), faces)
    features = [
        {
            "kind": f.kind, "source": f.source, "color": f.color, "visible": f.visible,
            "delta_red_deg": float(np.degrees(f.delta_red)), "delta_blue_deg": float(np.degrees(f.delta_blue)),
            "shift_deg": float(np.degrees(f.shift)), "spread_deg": float(np.degrees(f.spread)),
        }
        for f in verdict.features
    ]  # fmt: skip
    # the blue kink's own interval (from the red kink's smallest D to the blue kink's largest, one step width wider)
    field_red, field_blue = (DPField.build(HexPrism.from_ratio(2.0), faces, n) for n in (chromatic.N_RED, chromatic.N_BLUE))
    kink_values = np.degrees(np.concatenate([k.values for f in (field_red, field_blue) for k in f.weight_kinks]))
    kink_interval = [float(kink_values.min() - 2.0 * STEP_HALF_WIDTH_DEG), float(kink_values.max() + 2.0 * STEP_HALF_WIDTH_DEG)]
    steps = _colour_steps(delta, red, blue, lit)
    outside = [s for s in steps if not kink_interval[0] <= s["delta_deg"] <= kink_interval[1]]
    ratio = np.where(lit, blue / np.where(red > 0.0, red, np.nan), np.nan)
    z = _chromaticity_z(xyz)
    band = (delta_xyz >= kink_interval[0]) & (delta_xyz <= kink_interval[1]) & lit
    return {
        "lit_delta_range_deg": [float(delta[lit].min()), float(delta[lit].max())],
        "blue_over_red_range_lit": [float(np.nanmin(ratio)), float(np.nanmax(ratio))],
        "blue_over_red_max_at_deg": float(delta[int(np.nanargmax(ratio))]),
        "kink_interval_deg": kink_interval,
        "colour_steps": steps,
        "red_edge_seen": any(s["into_red"] for s in outside),
        "steps_outside_kink_interval": outside,
        "blue_band_seen": any(max(s["ratio_below"], s["ratio_above"]) >= chromatic.TINT_RATIO_MIN for s in steps),
        "z_chromaticity_in_kink_interval_median": float(np.nanmedian(z[band])),
        "z_chromaticity_below_kink_interval_median": float(np.nanmedian(z[(delta_xyz < kink_interval[0]) & lit])),
        "criterion": {"kind": verdict.kind, "color": verdict.color, "visible": verdict.visible, "features": features},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--store-n", type=int, default=10_000_000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cases", nargs="+", choices=sorted(CASES), default=sorted(CASES, reverse=True))
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checks = {"316": check_316, "315": check_315}
    summary = {"n_red": chromatic.N_RED, "n_blue": chromatic.N_BLUE, "store_n": args.store_n, "camera": CAMERA}
    for case in args.cases:
        summary[case] = checks[case](args)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
