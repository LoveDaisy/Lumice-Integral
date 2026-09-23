"""Regressions and example figures of the band-sum renderer (task ``band-sum-renderer``).

Stages (``--stage``):

- ``full``: a full-image render of ``scripts/render_band_sum.py`` (``--band-dir``,
  optionally a coarser ``--coarse-dir`` of the same scene) against the Phase I
  strip (``--reference-dir``, the point-pixel ``artifacts/strip-full``).  Lit =
  reference above ``1e-2`` of its column maximum.  Reports the lit relative
  error (median / p95 / max / RMS), log RMS, the lit and whole-image sum ratios,
  the distribution of the per-column medians, and the worst pixels, each
  attributed: ``pixel-model`` (the reference changes by more than ``10 %`` to a
  neighbouring pixel: the band average and the point value differ there),
  ``sampling`` (``|rel| sqrt(K_eff) <= 4``), else ``unexplained``; each worst
  pixel also carries the fraction of its band outside the store's deviation
  range (dark by construction: the inner-edge caustic).  With ``--coarse-dir``
  the worst pixels carry their error at the coarse ``N`` (sampling error
  shrinks as ``N^-1/2``, a pixel-model bias does not), the pixels beyond the
  noise are checked for that persistence, and the low-``K_eff`` pixels get the
  mean / median / skewness of their error (a right-skewed sampling
  distribution puts the median below an unbiased mean).
- ``class``: task 14's three profiles (plate / Parry / Lowitz, class ``[3, 5]``)
  through ``band_sum.class_band_sum_pixel`` on task 14's own twelve member stores
  (``transport=False``, the same points) against ``metrics_<family>_N<n>.csv``
  (``--tiers``; all twelve stores of a tier are in memory at once, ~0.36 GB each
  at ``5e7``); and the production ``transport=True`` plan (one ``3-5`` store,
  ``--store-n`` from ``--store-cache-dir``, twelve pose factors) against the
  same CSV at that ``N`` (different points: a sampling-level comparison).
- ``figure``: log-scale images of render directories (``uv run --with matplotlib``).

Usage::

    uv run python scripts/regress_band_sum.py --stage full --band-dir artifacts/band-sum-full \\
        --coarse-dir artifacts/band-sum-full-N1e7 --output <report.json>
    uv run python scripts/regress_band_sum.py --stage class --tiers 1000000 10000000 50000000 \\
        --store-n 100000000 --output <report.json>
    uv run --with matplotlib python scripts/regress_band_sum.py --stage figure --band-dir <dir> --output <png>
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from lumice_integral.band_sum import class_band_sum_pixel, pixel_band, store_plan
from lumice_integral.canonical_scene import (
    CANONICAL_REFRACTIVE_INDEX,
    canonical_crystal,
    canonical_incident_direction,
)
from lumice_integral.optics import path_id_of
from lumice_integral.path_class import build_path_class
from lumice_integral.pose_density import build_pose_density
from lumice_integral.s2_store import DEFAULT_CACHE_DIR, build_or_load
from lumice_integral.strip_io import read_strip

# Machine-specific default (the main checkout's gitignored strip-full artifact); other environments
# must pass --reference-dir explicitly, or read_strip will fail to find this path.
DEFAULT_REFERENCE_DIR = Path("/Users/zhangjiajie/Codes/Lumice Integral/artifacts/strip-full")
TASK14_ARTIFACTS = Path(__file__).resolve().parents[1] / "scratchpad/task-narrow-density-band-sum-probe/artifacts"
LIT_FRACTION = 1e-2
STEEP_NEIGHBOUR_CHANGE = 0.1  # task 14's criterion for "the two pixel models differ visibly"
NOISE_Z = 4.0
WORST = 25


# ------------------------------------------------------------------ full
def read_k_eff(band_dir: Path, shape: tuple[int, int]) -> np.ndarray:
    k_eff = np.full(shape, np.nan)
    with (band_dir / "pixels.csv").open() as handle:
        for row in csv.DictReader(handle):
            k_eff[int(row["row"]), int(row["column"])] = float(row["K_eff"])
    return k_eff


def neighbour_change(reference: np.ndarray) -> np.ndarray:
    """Largest ``|ref(neighbour) / ref - 1|`` over the four neighbours (``inf`` where ``ref = 0``)."""
    padded = np.pad(reference, 1, mode="edge")
    with np.errstate(divide="ignore", invalid="ignore"):
        changes = [
            np.abs(padded[1 + dr : 1 + dr + reference.shape[0], 1 + dc : 1 + dc + reference.shape[1]] / reference - 1.0)
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
        ]
    return np.nan_to_num(np.max(changes, axis=0), nan=np.inf)


def lit_statistics(estimate: np.ndarray, reference: np.ndarray, lit: np.ndarray) -> dict[str, Any]:
    rel = (estimate[lit] - reference[lit]) / reference[lit]
    positive = estimate[lit] > 0.0
    log_ratio = np.log(estimate[lit][positive] / reference[lit][positive])
    return {
        "lit_pixels": int(lit.sum()),
        "lit_zero_estimate": int((~positive).sum()),
        "median_abs_rel": float(np.median(np.abs(rel))),
        "p95_abs_rel": float(np.percentile(np.abs(rel), 95)),
        "max_abs_rel": float(np.max(np.abs(rel))),
        "rms_rel": float(np.sqrt(np.mean(rel**2))),
        "log_rms": float(np.sqrt(np.mean(log_ratio**2))),
        "median_ratio": float(np.median(estimate[lit] / reference[lit])),
        "lit_sum_ratio": float(estimate[lit].sum() / reference[lit].sum()),
    }


def stage_full(band_dir: Path, reference_dir: Path, coarse_dir: Path | None) -> dict[str, Any]:
    band, band_provenance = read_strip(band_dir)
    reference, _ = read_strip(reference_dir)
    if band.values.shape != reference.values.shape:
        raise ValueError(f"shapes differ: {band.values.shape} vs {reference.values.shape}")
    both = band.rendered & reference.rendered
    if not both.all():
        raise ValueError("the regression compares full images; some pixels are not rendered")
    ref = reference.values
    column_max = ref.max(axis=0, keepdims=True)
    lit = ref > LIT_FRACTION * column_max
    est = band.values
    k_eff = read_k_eff(band_dir, est.shape)
    steep = neighbour_change(ref)
    report: dict[str, Any] = {
        "band_dir": str(band_dir),
        "reference_dir": str(reference_dir),
        "N": band_provenance["options"]["N"],
        "lit_definition": f"reference > {LIT_FRACTION} x its column maximum",
        "lit": lit_statistics(est, ref, lit),
        "whole_image_sum_ratio": float(est.sum() / ref.sum()),
        "execution": {
            key: band_provenance["execution"].get(key)
            for key in (
                "workers",
                "wall_clock_s",
                "stores_wall_clock_s",
                "render_wall_clock_s",
                "per_pixel_mean_s",
                "max_rss_mb_parent",
                "max_rss_mb_worker",
            )
        },
    }
    rel = np.where(lit, (est - ref) / np.where(lit, ref, 1.0), np.nan)
    column_median = np.nanmedian(np.abs(rel), axis=0)
    report["per_column_median_abs_rel"] = {
        "percentiles_0_25_50_75_100": [float(np.percentile(column_median, q)) for q in (0, 25, 50, 75, 100)],
        "columns_above_5e-3": int((column_median > 5e-3).sum()),
        "column_126": float(column_median[126]),
        "values": [float(v) for v in column_median],
    }
    not_steep = lit & (steep <= STEEP_NEIGHBOUR_CHANGE)
    report["lit_without_pixel_model_pixels"] = lit_statistics(est, ref, not_steep)
    report["lit_pixel_model_pixels"] = int((lit & ~not_steep).sum())
    z = np.abs(rel) * np.sqrt(k_eff)
    label = np.where(steep > STEEP_NEIGHBOUR_CHANGE, "pixel-model", np.where(z <= NOISE_Z, "sampling", "unexplained"))
    report["attribution_counts_lit"] = {name: int(((label == name) & lit).sum()) for name in ("pixel-model", "sampling", "unexplained")}
    coarse = None
    if coarse_dir is not None:
        coarse_arrays, coarse_provenance = read_strip(coarse_dir)
        coarse = coarse_arrays.values
        report["coarse"] = {"dir": str(coarse_dir), "N": coarse_provenance["options"]["N"], "lit": lit_statistics(coarse, ref, lit)}
    beyond_noise = lit & (z > NOISE_Z)
    if coarse is not None:
        rel_coarse = np.where(lit, (coarse - ref) / np.where(lit, ref, 1.0), np.nan)
        persistent = beyond_noise & (np.abs(rel_coarse - rel) < 0.3 * np.abs(rel))
        # A pixel-model bias is the same at both N; sampling error shrinks by sqrt(10).
        report["beyond_noise_persistence"] = {
            "lit_beyond_noise": int(beyond_noise.sum()),
            "of_which_pixel_model": int((beyond_noise & (label == "pixel-model")).sum()),
            "persistent_between_N": int(persistent.sum()),
            "criterion": f"|rel| sqrt(K_eff) > {NOISE_Z} at the fine N; persistent: |rel_coarse - rel| < 0.3 |rel|",
        }
        low = lit & (k_eff < 1000.0)
        block = {"lit_pixels_K_eff_below_1000": int(low.sum())}
        for tag, values in (("fine", est), ("coarse", coarse)):
            r = values[low] / ref[low] - 1.0
            block[tag] = {
                "median_rel": float(np.median(r)),
                "mean_rel": float(np.mean(r)),
                "sum_ratio": float(values[low].sum() / ref[low].sum()),
                "skewness": float(np.mean((r - r.mean()) ** 3) / np.std(r) ** 3),
            }
        report["low_K_eff_sampling_skew"] = block
    store_d_range = _store_d_range(band_provenance)
    order = np.argsort(np.where(lit, -np.abs(np.nan_to_num(rel)), 0.0), axis=None)[:WORST]
    worst = []
    for flat in order:
        r, c = np.unravel_index(flat, est.shape)
        entry = {
            "row": int(r),
            "column": int(c),
            "estimate": float(est[r, c]),
            "reference": float(ref[r, c]),
            "rel": float(rel[r, c]),
            "K_eff": float(k_eff[r, c]),
            "noise_z": float(z[r, c]),
            "reference_neighbour_change": float(steep[r, c]),
            "attribution": str(label[r, c]),
            **_band_outside_store(int(r), int(c), band_provenance, store_d_range),
        }
        if coarse is not None:
            entry["rel_coarse"] = float((coarse[r, c] - ref[r, c]) / ref[r, c])
        worst.append(entry)
    report["worst_pixels"] = worst
    unexplained = lit & (label == "unexplained")
    if unexplained.any():
        rows, cols = np.nonzero(unexplained)
        report["unexplained_pixels"] = [
            {"row": int(r), "column": int(c), "rel": float(rel[r, c]), "K_eff": float(k_eff[r, c]), "noise_z": float(z[r, c])}
            for r, c in zip(rows, cols)
        ]
    return report


def _store_d_range(provenance: dict[str, Any]) -> tuple[float, float]:
    """``[min D, max D]`` (degrees) of the render's store, from its cache provenance."""
    directory = Path(provenance["options"]["stores"][0]["directory"])
    d_range = json.loads((directory / "provenance.json").read_text())["diagnostics"]["D_range_deg"]
    return float(d_range[0]), float(d_range[1])


def _band_outside_store(row: int, column: int, provenance: dict[str, Any], d_range: tuple[float, float]) -> dict[str, float]:
    """The pixel's band (degrees) and the fraction of it outside the store's deviation range (dark by construction)."""
    render = provenance["scene"]["camera"]["value"]
    render = {k: render[k] for k in ("width", "height", "fov_deg", "view")}
    _, _, lo, hi = pixel_band(row, column, canonical_incident_direction(), render)
    lo, hi = np.degrees(lo), np.degrees(hi)
    inside = max(0.0, min(hi, d_range[1]) - max(lo, d_range[0]))
    return {"band_deg": [float(lo), float(hi)], "band_fraction_outside_store_D_range": float(1.0 - inside / (hi - lo))}


# ----------------------------------------------------------------- class
def load_task14_member(member, n: int) -> dict[str, np.ndarray]:
    with np.load(TASK14_ARTIFACTS / "members" / path_id_of(member) / f"events_N{n}.npz") as data:
        return {key: data[key] for key in data.files}


def read_task14_metrics(family: str, n: int) -> dict[tuple[int, int], dict[str, float]]:
    with (TASK14_ARTIFACTS / f"metrics_{family}_N{n}.csv").open() as handle:
        return {
            (int(r["row"]), int(r["column"])): {k: float(r[k]) for k in ("estimate", "K", "K_rho_pos", "K_eff")}
            for r in csv.DictReader(handle)
        }


def compare_profile(stores, family: str, spec: dict[str, Any], n: int, probe: dict) -> dict[str, Any]:
    s = canonical_incident_direction()
    density = build_pose_density(family, **spec["density"])
    worst_rel, worst_counts, count, rows = 0.0, 0, 0, []
    for row, column in spec["pixels"]:
        result = class_band_sum_pixel(stores, s, density, row, column, n, spec["render"])
        ref = probe[(row, column)]
        rel = abs(result.value - ref["estimate"]) / abs(ref["estimate"]) if ref["estimate"] else abs(result.value)
        worst_rel = max(worst_rel, rel)
        worst_counts += (result.K, result.K_rho_pos) != (int(ref["K"]), int(ref["K_rho_pos"]))
        count += 1
        rows.append((result.value, ref["estimate"], result.K_eff, ref["K_eff"]))
    return {"pixels": count, "max_rel": worst_rel, "K_or_K_rho_pos_mismatches": worst_counts, "rows": rows}


def stage_class(tiers: list[int], store_n: int | None, store_cache_dir: Path) -> dict[str, Any]:
    crystal = canonical_crystal()
    path_class = build_path_class(crystal, (3, 5))
    windows = json.loads((TASK14_ARTIFACTS / "profile_windows.json").read_text())["families"]
    report: dict[str, Any] = {"task14_artifacts": str(TASK14_ARTIFACTS), "same_points": {}, "transport": None}
    for n in tiers:
        start = time.perf_counter()
        per_member = store_plan(path_class, crystal, transport=False)
        stores = [(load_task14_member(group.members[0], n), group) for group in per_member]
        block = {}
        for family, spec in windows.items():
            result = compare_profile(stores, family, spec, n, read_task14_metrics(family, n))
            result.pop("rows")
            block[family] = result
            print(n, family, json.dumps(result))
        del stores
        block["wall_clock_s"] = time.perf_counter() - start
        report["same_points"][str(n)] = block
    if store_n is not None:
        store = build_or_load(
            crystal, CANONICAL_REFRACTIVE_INDEX, [(3, 5)], store_n, base_dir=store_cache_dir, incident_direction=canonical_incident_direction()
        )
        (plan,) = store_plan(path_class, crystal)
        stores = [(store.events.arrays(), plan)]
        block = {"N": store_n, "cache_key": store.spec.cache_key(), "transports": len(plan.transports)}
        for family, spec in windows.items():
            probe = read_task14_metrics(family, store_n)
            result = compare_profile(stores, family, spec, store_n, probe)
            value, ref, k_eff, _ = (np.array(v) for v in zip(*result.pop("rows")))
            peak = ref > 0.1 * ref.max()
            ratio = value[peak] / ref[peak]
            # Two independent samplings of one integral: the difference over the combined noise.
            z = (value[peak] - ref[peak]) / (ref[peak] * np.sqrt(2.0 / k_eff[peak]))
            block[family] = {
                "peak_pixels": int(peak.sum()),
                "ratio_median": float(np.median(ratio)),
                "ratio_min": float(ratio.min()),
                "ratio_max": float(ratio.max()),
                "sum_ratio": float(value[peak].sum() / ref[peak].sum()),
                "z_rms": float(np.sqrt(np.mean(z**2))),
                "z_max_abs": float(np.max(np.abs(z))),
            }
            print("transport", family, json.dumps(block[family]))
        report["transport"] = block
    return report


# ---------------------------------------------------------------- figure
def stage_figure(band_dirs: list[Path], output: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    figure, axes = plt.subplots(1, len(band_dirs), figsize=(5.5 * len(band_dirs), 6), squeeze=False)
    for axis, band_dir in zip(axes[0], band_dirs):
        arrays, provenance = read_strip(band_dir)
        values = np.where(arrays.values > 0.0, arrays.values, np.nan)
        top = np.nanmax(values)
        cmap = matplotlib.colormaps["inferno"].with_extremes(bad="black", under="black")  # 0 = no event, black
        image = axis.imshow(values, norm=LogNorm(vmin=top * 1e-4, vmax=top), cmap=cmap, interpolation="nearest")
        camera = provenance["scene"]["camera"]["value"]
        density = provenance["scene"]["pose_density"]["value"]
        path_class = provenance["options"]["path_class"]
        what = f"class [{path_class['path_ids'][0]}] ({path_class['size']})" if path_class["size"] > 1 else path_class["path_ids"][0]
        axis.set_title(
            f"{what}, {density['family']}, N={provenance['options']['N']:.0e}\n"
            f"{camera['width']}x{camera['height']} px, fov {camera['fov_deg']:.3g} deg, "
            f"view az {camera['view']['azimuth']:g} el {camera['view']['elevation']:g}",
            fontsize=8,
        )
        axis.set_xticks([])
        axis.set_yticks([])
        figure.colorbar(image, ax=axis, fraction=0.04)
    figure.suptitle(title, fontsize=9)
    figure.tight_layout()
    figure.savefig(output, dpi=130)
    print("wrote", output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=("full", "class", "figure"), required=True)
    parser.add_argument("--band-dir", type=Path, nargs="+")
    parser.add_argument("--coarse-dir", type=Path, default=None)
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=DEFAULT_REFERENCE_DIR,
        help="strip-full read_strip directory; default is a machine-specific path, pass explicitly on other machines",
    )
    parser.add_argument("--tiers", type=int, nargs="*", default=[1_000_000, 10_000_000, 50_000_000])
    parser.add_argument("--store-n", type=int, default=None)
    parser.add_argument("--store-cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--title", default="band-sum renderer (log scale, 4 decades)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.stage == "figure":
        stage_figure(args.band_dir, args.output, args.title)
        return
    if args.stage == "full":
        report = stage_full(args.band_dir[0], args.reference_dir, args.coarse_dir)
        print(json.dumps({k: v for k, v in report.items() if k not in ("worst_pixels", "per_column_median_abs_rel", "unexplained_pixels")}, indent=1))
    else:
        report = stage_class(args.tiers, args.store_n, args.store_cache_dir)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("wrote", args.output)


if __name__ == "__main__":
    main()
