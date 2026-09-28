"""Pose-density family check: a band-sum path-class render against a Lumice float export, absolute and shape.

The band-sum renderer (``scripts/render_band_sum.py --path-class``) writes the
class value ``V(w)`` of every pixel (the sum over the class members under the
crystal's own group ``G_true``, crystal length^2 per steradian, hexagon edge
``a = 1``).  The Lumice run must admit exactly the same raypaths, checked
against the render's ``options.path_class.members`` (:func:`check_filter`):

- on the regular prism (``G_true = D6h``) a raypath filter with ``symmetry:
  PBD`` on the class representative (Lumice's fold is the ``D6h`` orbit there);
- on any prism, exact raypath filters (no ``symmetry``) for the members, one
  alone or ORed by a flat ``complex`` composition.  Lumice's P/B/D fold
  permutes face numbers as on the regular hexagon whatever the
  ``face_distance``, so below ``D6h`` it merges inequivalent paths and is
  refused (explore ``panel-row-symmetry-convention``).

The conversion of ``scripts/probe_absolute_scale.py`` then holds with the class
fold already inside ``V``, ``S`` the surface area of the crystal the render's
provenance records (``strip_io.scene_crystal``):

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
  per-pixel noise (Lumice: ``probe_absolute_scale.merged_relative_noise`` of
  the two i.i.d. seeds; Lumice Integral: ``1 / sqrt(K_eff)`` from the
  band-sum ``pixels.csv``);
- ``regions``: the same flux ratio per image half (top / bottom, left /
  right), so a pose-dependent factor between two halo features shows up as
  two different ratios;
- ``profiles``: the max-normalised row and column through the brightest
  merged Lumice pixel, the RMS of their difference on the lit part (either
  side above ``--profile-floor``) and the same RMS between the two Lumice
  seeds (the noise floor of the profile, ``sqrt(2)`` times the merged one).

Colour (the render's ``provenance.json`` ``format`` decides; anything else is
refused): a ``render_band_sum.py --illuminant`` / ``--discrete-wavelength-nm``
render (``spectrum.xyz_band_sum``) is compared channel by channel in linear
XYZ against all three channels of ``img_01.npy`` (never the tone-mapped 8-bit
PNG), with the Lumice light source's spectrum checked against the render's:

    raw[p, c] / E = XYZ[p, c] * Omega_p / (S / 2),  c in X, Y, Z

where ``XYZ`` already carries the CMF and Lumice's slot and emitted-weight
normalisation (no ``ybar(550)`` here).  The four blocks above are reported per
channel under ``channels``, with the flux-weighted chromaticity ``(x, y)`` of
both sides in ``chromaticity``.  The monochrome form is the same expression
with ``XYZ[p, Y] = ybar(550) V``: a discrete 550 nm colour render gives the
monochrome ``Y`` metrics bit for bit.  Lumice's pool has ``M = 64`` slots
unless ``LUMICE_WL_POOL_SIZE`` says otherwise (it is not in ``config.json``);
set it to the render's ``--wavelength-count`` for a like-for-like pool, and
run the Metal backend (``--backend metal --seed N``: the CPU backend samples
the wavelength per batch instead of the pool, and unseeded Metal runs repeat
one random stream, which leaves no noise floor).

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

from lumice_integral.band_sum import FORMAT_VERSION as MONOCHROME_FORMAT
from lumice_integral.geometry import Polyhedron
from lumice_integral.path_class import g_true_orbit
from lumice_integral.spectrum.xyz_band_sum import CHANNELS, read_xyz_band_sum_strip
from lumice_integral.spectrum.xyz_band_sum import FORMAT_VERSION as XYZ_FORMAT
from lumice_integral.strip_io import read_strip, scene_crystal
from lumice_integral.symmetry.crystal_group import true_symmetry_group

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_absolute_scale import YBAR_550, load_run_xyz, merged_relative_noise, pixel_solid_angles, total_surface_area  # noqa: E402


def check_camera(render: dict[str, Any], config: dict[str, Any]) -> None:
    lumice = config["render"][0]
    expected = ("linear", [render["width"], render["height"]], float(render["fov_deg"]), [float(render["view"]["azimuth"]), float(render["view"]["elevation"])])
    actual = (lumice["lens"]["type"], list(lumice["resolution"]), float(lumice["lens"]["fov"]), [float(lumice["view"]["azimuth"]), float(lumice["view"]["elevation"])])
    if actual != expected:
        raise SystemExit(f"Lumice camera {actual} != band-sum camera {expected}")


def check_filter(config: dict[str, Any], representative: list[int], members: list[list[int]], crystal: Polyhedron) -> str:
    """``SystemExit`` unless the Lumice run admits exactly the raypaths ``members``; returns the form used.

    The scattering entry's filter is either a ``PBD`` raypath filter on ``representative`` (accepted only when
    ``G_true`` is all of ``D6h``: the candidate group is ``D6h``, so order 24 is ``D6h`` itself), or exact
    raypath filters (no ``symmetry``, ``filter_in``) whose sequences, as a set, are ``members``: one filter,
    or a ``complex`` filter whose ``composition`` is a flat list of their ids (an OR).  ``PBD`` is Lumice's
    label meaning (L1); requiring ``|G_true| == 24`` is exactly the condition under which L1 coincides with
    the physical meaning (L2) this comparison needs (``docs/conventions.md`` #21).
    """
    entries = [e for layer in config["scene"]["scattering"] for e in layer["entries"]]
    if len(entries) != 1 or "filter" not in entries[0]:
        raise SystemExit(f"Lumice config: need one scattering entry with a filter, got {entries}")
    filters = {f["id"]: f for f in config["filter"]}
    top = filters[entries[0]["filter"]]
    if top.get("symmetry"):
        if top["symmetry"] != "PBD" or top["type"] != "raypath" or top["raypath"] != representative:
            raise SystemExit(f"Lumice filter {top}: a folded filter must be a PBD raypath filter on {representative}")
        if len(true_symmetry_group(crystal)) != 24:
            raise SystemExit("Lumice filter uses PBD, but the crystal's G_true is smaller than D6h: its fold merges inequivalent paths")
        if sorted(map(tuple, members)) != sorted(g_true_orbit(representative)):
            raise SystemExit(f"Lumice PBD admits {sorted(g_true_orbit(representative))}, the band-sum class has {sorted(map(tuple, members))}")
        return "PBD"
    if top["type"] == "raypath":
        parts = [top]
    elif top["type"] == "complex" and top.get("composition"):
        parts = [filters.get(i) if isinstance(i, int) else None for i in top["composition"]]  # a nested list is an AND clause
    else:
        raise SystemExit(f"Lumice filter {top}: need a raypath or a complex filter")
    for part in parts:
        if part is None or part["type"] != "raypath" or part.get("symmetry") or part.get("action", "filter_in") != "filter_in":
            raise SystemExit(f"Lumice filter {top}: every part must be an exact filter_in raypath filter without symmetry, got {part}")
    admitted = sorted(tuple(part["raypath"]) for part in parts)
    if top.get("action", "filter_in") != "filter_in" or len(set(admitted)) != len(admitted) or admitted != sorted(map(tuple, members)):
        raise SystemExit(f"Lumice filter admits {admitted}, the band-sum class has {sorted(map(tuple, members))}")
    return "exact" if top["type"] == "raypath" else "complex"


def read_k_eff(li_dir: Path, shape: tuple[int, int], column: str = "K_eff") -> np.ndarray:
    k_eff = np.zeros(shape)
    with (li_dir / "pixels.csv").open() as fh:
        for rec in csv.DictReader(fh):
            k_eff[int(rec["row"]), int(rec["column"])] = float(rec[column])
    return k_eff


def flux_ratio(measured: np.ndarray, predicted: np.ndarray, mask: np.ndarray) -> float:
    return float(measured[mask].sum() / predicted[mask].sum()) if predicted[mask].sum() > 0 else float("nan")


def profile_rms(a: np.ndarray, b: np.ndarray, floor: float) -> tuple[float, int]:
    if a.max() <= 0 or b.max() <= 0:
        raise ValueError("profile_rms: a profile's peak is <= 0, nothing to max-normalise")
    na, nb = a / a.max(), b / b.max()
    lit = (na > floor) | (nb > floor)
    return float(np.sqrt(np.mean((na[lit] - nb[lit]) ** 2))), int(lit.sum())


def load_li(li_dir: Path) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, Any]]:
    """``({channel: image}, K_eff image, provenance)`` of a band-sum render, monochrome or colour (by ``format``).

    Monochrome (``band_sum.FORMAT_VERSION``): one channel ``Y = ybar(550) V``.  Colour
    (``xyz_band_sum.FORMAT_VERSION``): ``X``, ``Y``, ``Z`` as written, which already carry the CMF and
    Lumice's slot / emitted-weight normalisation, and ``K_eff_min`` as the noise.  Either way the
    prediction is ``raw[p, c] / E = channel[p] Omega_p / (S / 2)``, the same expression, so a
    discrete 550 nm colour render and the monochrome render at ``n(550)`` give bit-identical ``Y``
    metrics.  Any other or missing ``format`` is refused, never read as monochrome.
    """
    try:
        provenance = json.loads((li_dir / "provenance.json").read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"{li_dir}: no readable provenance.json ({exc})") from None
    kind = provenance.get("format")
    if kind == MONOCHROME_FORMAT:
        arrays, provenance = read_strip(li_dir)
        channels = {"Y": YBAR_550 * np.nan_to_num(arrays.values, nan=0.0)}
        k_eff = read_k_eff(li_dir, arrays.values.shape)
    elif kind == XYZ_FORMAT:
        xyz, _, provenance = read_xyz_band_sum_strip(li_dir)
        channels = {c: xyz[:, :, i] for i, c in enumerate(CHANNELS)}
        k_eff = read_k_eff(li_dir, xyz.shape[:2], column="K_eff_min")
    else:
        raise SystemExit(f"{li_dir}: provenance format {kind!r} is neither {MONOCHROME_FORMAT!r} nor {XYZ_FORMAT!r}")
    return channels, k_eff, provenance


def check_spectrum(config: dict[str, Any], spectrum: dict[str, Any]) -> None:
    """``SystemExit`` unless the Lumice light source has the colour render's spectrum (illuminant name, or the one wavelength)."""
    lumice = config["scene"]["light_source"].get("spectrum")
    if "illuminant" in spectrum:
        if lumice != spectrum["illuminant"]:
            raise SystemExit(f"Lumice spectrum {lumice!r} != band-sum illuminant {spectrum['illuminant']!r}")
    elif not (isinstance(lumice, list) and len(lumice) == 1 and float(lumice[0]["wavelength"]) == spectrum["discrete_wavelength_nm"]):
        raise SystemExit(f"Lumice spectrum {lumice!r} != band-sum wavelength {spectrum['discrete_wavelength_nm']} nm")


def channel_metrics(
    measured: np.ndarray,
    predicted: np.ndarray,
    per_run: list[np.ndarray],
    k_eff: np.ndarray,
    surface_area: float,
    omega: np.ndarray,
    *,
    bright_floor: float,
    profile_floor: float,
) -> dict[str, Any]:
    """``total`` / ``bright`` / ``regions`` / ``profiles`` of one channel (module docstring)."""
    k_pixel = omega / (0.5 * surface_area)
    lit = (measured > 0) | (predicted > 0)
    bright = (measured >= bright_floor * measured.max()) & (predicted > 0)
    ratio = measured[bright] / predicted[bright]
    implied = predicted[bright] * (0.5 * surface_area) / measured[bright]
    out: dict[str, Any] = {
        "total": {
            "lit_pixels": int(lit.sum()),
            "measured_over_predicted": flux_ratio(measured, predicted, lit),
            "per_run": [flux_ratio(r, predicted, lit) for r in per_run],
            "predicted_flux_share_outside_lumice_lit": float(predicted[lit & (measured == 0)].sum() / predicted[lit].sum()),
            "measured_flux_share_outside_li_lit": float(measured[lit & (predicted == 0)].sum() / measured[lit].sum()),
        },
        "bright": {
            "floor": bright_floor,
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
        out["bright"]["lumice_merged_relative_noise_rms"] = merged_relative_noise(x1, x2)
        out["bright"]["expected_ratio_relative_std"] = float(np.sqrt(out["bright"]["lumice_merged_relative_noise_rms"] ** 2 + out["bright"]["li_relative_noise_rms"] ** 2))
    h, w = measured.shape
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
        rms, count = profile_rms(measured[index], predicted[index], profile_floor)
        entry: dict[str, Any] = {"index": int(peak_row if name == "row" else peak_column), "lit": count, "rms_max_normalised": rms}
        if len(per_run) >= 2:
            entry["rms_lumice_run1_vs_run2"] = profile_rms(per_run[0][index], per_run[1][index], profile_floor)[0]
        out["profiles"][name] = entry
    return out


def chromaticity(measured: dict[str, np.ndarray], predicted: dict[str, np.ndarray]) -> dict[str, Any]:
    """Flux-weighted CIE ``(x, y)`` of the whole image on either side (lit in any channel) and their difference."""
    lit = np.any([(measured[c] > 0) | (predicted[c] > 0) for c in CHANNELS], axis=0)

    def xy(images: dict[str, np.ndarray]) -> list[float]:
        flux = [float(images[c][lit].sum()) for c in CHANNELS]
        total = sum(flux)
        if total == 0.0:
            raise SystemExit("chromaticity: no positive flux in either image over the lit mask (window has no light)")
        return [flux[0] / total, flux[1] / total]

    m, p = xy(measured), xy(predicted)
    return {"measured_xy": m, "predicted_xy": p, "difference_xy": [m[0] - p[0], m[1] - p[1]]}


def print_channel(label: str, metrics: dict[str, Any], entry_area: float) -> None:
    t, b = metrics["total"], metrics["bright"]
    print(f"{label}total flux measured/predicted {t['measured_over_predicted']:.4f} (runs {', '.join(f'{r:.4f}' for r in t['per_run'])})")
    print(
        f"{label}bright ({b['pixels']} px): ratio median {b['measured_over_predicted_median']:.4f}, mean {b['measured_over_predicted_mean']:.4f} "
        f"+- {b['measured_over_predicted_standard_error']:.4f}, rel std {b['measured_over_predicted_relative_std']:.4f} "
        f"(expected {b.get('expected_ratio_relative_std', float('nan')):.4f}); implied area {b['implied_entry_area_median']:.3f} (S/2 {entry_area:.3f})"
    )
    for name, r in metrics["regions"].items():
        print(f"{label}region {name}: {r['measured_over_predicted']:.4f} (flux share {r['predicted_flux_share']:.3f})")
    for name, p in metrics["profiles"].items():
        print(f"{label}profile {name} {p['index']}: rms {p['rms_max_normalised']:.4f} (lumice seeds {p.get('rms_lumice_run1_vs_run2', float('nan')):.4f}, n={p['lit']})")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--li-dir", type=Path, required=True, help="render_band_sum.py --path-class output directory (monochrome or colour)")
    parser.add_argument("--lumice-run", type=Path, action="append", required=True, help="Lumice run directory (img_01.npy, img_01.json, config.json); two for the noise floor")
    parser.add_argument("--bright-floor", type=float, default=0.1)
    parser.add_argument("--profile-floor", type=float, default=0.05)
    parser.add_argument("--output", type=Path, required=True, help="metrics JSON")
    args = parser.parse_args(argv)

    channels, k_eff, provenance = load_li(args.li_dir)
    colour = provenance["format"] == XYZ_FORMAT
    shape = k_eff.shape
    render = dict(provenance["scene"]["camera"]["value"])
    path = provenance["scene"]["path"]["value"]
    members = provenance["options"]["path_class"]["members"]
    crystal = scene_crystal(provenance["scene"])
    runs = [load_run_xyz(d) for d in args.lumice_run]
    filter_forms = set()
    for d, (xyz, _, _) in zip(args.lumice_run, runs):
        if xyz.shape[:2] != shape:
            raise SystemExit(f"{d}: shape {xyz.shape[:2]} (need {shape})")
        config = json.loads((d / "config.json").read_text())
        filter_forms.add(check_filter(config, path, members, crystal))
        check_camera(render, config)
        if colour:
            check_spectrum(config, provenance["options"]["spectrum"])

    surface_area = total_surface_area(crystal)
    omega = pixel_solid_angles(render)
    energy = [m["emitted_energy"] for _, m, _ in runs]
    metrics, measured_all, predicted_all = {}, {}, {}
    for name, image in channels.items():
        c = CHANNELS.index(name)
        predicted = omega / (0.5 * surface_area) * image
        per_run = [xyz[:, :, c] / e for (xyz, _, _), e in zip(runs, energy)]
        measured = np.sum([xyz[:, :, c] for xyz, _, _ in runs], axis=0) / sum(energy)
        metrics[name] = channel_metrics(
            measured, predicted, per_run, k_eff, surface_area, omega, bright_floor=args.bright_floor, profile_floor=args.profile_floor
        )
        measured_all[name], predicted_all[name] = measured, predicted
    out: dict[str, Any] = {
        "generated": dt.datetime.now().astimezone().isoformat(),
        "li_dir": str(args.li_dir),
        "li_format": provenance["format"],
        "lumice_runs": [str(d) for d in args.lumice_run],
        "pose_density": provenance["scene"].get("pose_density", {}).get("value"),
        "path_class_representative": path,
        "path_class_members": members,
        "crystal": provenance["scene"]["crystal"]["value"],
        "lumice_filter": sorted(filter_forms),
        "camera": render,
        "emitted_energy": energy,
        "surface_area": surface_area,
        "entry_area": 0.5 * surface_area,
    }
    if colour:
        out["spectrum"] = provenance["options"]["spectrum"]
        out["convention"] = (
            "raw[p, c] / emitted_energy = XYZ[p, c] Omega_p / (S / 2), c in X, Y, Z; XYZ = the colour band-sum render "
            "(xyz_band_sum: (1/M) sum_i spd_i CMF_i V_{n_i} / emitted_weight), linear, no tone mapping"
        )
        out["channels"] = metrics
        out["chromaticity"] = chromaticity(measured_all, predicted_all)
    else:
        out["convention"] = "raw[p] / emitted_energy = K_p * V(p), K_p = ybar(550) * Omega_p / (S / 2); V = band-sum class value (G_true class)"
        out.update(metrics["Y"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    for name, m in metrics.items():
        print_channel(f"[{name}] " if colour else "", m, 0.5 * surface_area)
    if colour:
        ch = out["chromaticity"]
        print(f"chromaticity xy measured {ch['measured_xy'][0]:.5f}, {ch['measured_xy'][1]:.5f}; predicted {ch['predicted_xy'][0]:.5f}, {ch['predicted_xy'][1]:.5f}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
