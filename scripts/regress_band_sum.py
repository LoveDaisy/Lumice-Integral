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
  same CSV at that ``N`` (different points: a sampling-level comparison; the
  difference over ``sqrt(1/K_eff + 1/K_eff_task14)``).  Per family it also
  tabulates the median ``K_eff`` of the production render (distinct events,
  ``band_sum.K_EFF_SEMANTICS``), of the task ``band-sum-renderer`` pooling of
  every ``(event, transport)`` pair (``per_transport_sample``, recomputed
  literally), of the representative's store alone (one member) and of task
  14's twelve independent stores.
- ``k-eff``: the ruler of the noise diagnostic (a16).  Two i.i.d. uniform
  stores (``s2_store.RandomSphereSampler``, seeds 1 and 2, ``--random-n``) of
  class ``[3, 5]`` render task 14's three profiles; with i.i.d. points the
  Kish size is the Monte Carlo noise predictor, so
  ``z = (a - b) / sqrt(a^2 / K_eff_a + b^2 / K_eff_b)`` is standard normal
  when ``K_eff`` counts what is independent.  Reported for the per-event
  ``K_eff`` and for the pooled ``per_transport_sample`` one.  (The Fibonacci
  lattice is not i.i.d.: its error is below ``1 / sqrt(K_eff)``, so the
  ``class`` stage's ``z`` is below 1 by the lattice gain.)
- ``scatter``: the scatter renderer (task ``band-sum-scatter-renderer``) against the
  gather it replaced.  With ``--band-dir`` and ``--baseline-dir``: two renders of
  one scene pixel by pixel (``pixels.csv``: ``K`` / ``K_rho_pos`` exactly, value and
  ``K_eff`` relative, lit pixels; the inner-edge rows 48-56 separately; every pixel
  beyond ``1e-12`` listed with its value relative to the image maximum).  Without
  them: ``render_band_sum_window`` against ``class_band_sum_pixel`` pixel by pixel on
  class ``[1,3,5]`` (24 members, 12 reached by mirrors only; random density,
  ``--random-n`` Fibonacci points, a 150 deg camera) and on task 14's three
  profiles (class ``[3,5]``, one store, ``--random-n`` points), value and ``K_eff``.

- ``contour``: the band sum (``--band-dir``, optionally ``--coarse-dir``) against a
  band-average contour-quadrature render (``--contour-dir``,
  ``scripts/render_contour_quadrature.py --band-nodes k``): the same pixel model,
  so ``rel`` is the band sum's own error and ``z = rel sqrt(K_eff)`` its noise
  scale (task ``s2-contour-quadrature``).  ``--contour-point-dir`` (a point
  render) adds the pixel-model difference and, with ``--reference-dir``, Phase I
  ``strip-full`` against the point render.
- ``figure``: log-scale images of render directories (``uv run --with matplotlib``).

Usage::

    uv run python scripts/regress_band_sum.py --stage full --band-dir artifacts/band-sum-full \\
        --coarse-dir artifacts/band-sum-full-N1e7 --output <report.json>
    uv run python scripts/regress_band_sum.py --stage class --tiers 1000000 10000000 50000000 \\
        --store-n 100000000 --output <report.json>
    uv run python scripts/regress_band_sum.py --stage k-eff --random-n 10000000 --output <report.json>
    uv run python scripts/regress_band_sum.py --stage scatter --band-dir <new> --baseline-dir artifacts/band-sum-full \
        --output <report.json>
    uv run python scripts/regress_band_sum.py --stage scatter --random-n 10000000 --output <report.json>

    uv run python scripts/regress_band_sum.py --stage contour --band-dir artifacts/band-sum-full \\
        --coarse-dir artifacts/band-sum-full-N1e7 --contour-dir artifacts/contour-quadrature-band \\
        --contour-point-dir artifacts/contour-quadrature-full --output <report.json>
    uv run --with matplotlib python scripts/regress_band_sum.py --stage figure --band-dir <dir> --output <png>
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from lumice_integral.band_sum import (
    K_EFF_SEMANTICS,
    BandSumScene,
    ScatterSums,
    StoreGroup,
    Transport,
    band_poses,
    class_band_sum_pixel,
    kish_k_eff,
    pixel_band,
    pixel_bands,
    render_band_sum_window,
    scatter_results,
    scatter_store,
    store_plan,
)
from lumice_integral.canonical_scene import (
    CANONICAL_REFRACTIVE_INDEX,
    canonical_crystal,
    canonical_sun_direction,
)
from lumice_integral.optics import path_id_of
from lumice_integral.path_class import build_path_class
from lumice_integral.pose_density import build_pose_density
from lumice_integral.s2_store import (
    DEFAULT_CACHE_DIR,
    RandomSphereSampler,
    S2EventStore,
    build_event_store,
    build_or_load,
    events_from_schema1,
)
from lumice_integral.strip_io import Window, read_strip

# Machine-specific default (the main checkout's gitignored strip-full artifact); other environments
# must pass --reference-dir explicitly, or read_strip will fail to find this path.
DEFAULT_REFERENCE_DIR = Path("/Users/zhangjiajie/Codes/Lumice Integral/artifacts/strip-full")
TASK14_ARTIFACTS = Path(__file__).resolve().parents[1] / "scratchpad/task-narrow-density-band-sum-probe/artifacts"
LIT_FRACTION = 1e-2
STEEP_NEIGHBOUR_CHANGE = 0.1  # task 14's criterion for "the two pixel models differ visibly"
NOISE_Z = 4.0
WORST = 25


def k_eff_semantics_of(provenance: dict[str, Any]) -> str:
    """What a render's ``K_eff`` counts; renders before ``options.k_eff_semantics`` pooled every transport."""
    return provenance["options"].get("k_eff_semantics", "per_transport_sample")


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
        "k_eff_semantics": k_eff_semantics_of(band_provenance),
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
    _, _, lo, hi = pixel_band(row, column, canonical_sun_direction(), render)
    lo, hi = np.degrees(lo), np.degrees(hi)
    inside = max(0.0, min(hi, d_range[1]) - max(lo, d_range[0]))
    return {"band_deg": [float(lo), float(hi)], "band_fraction_outside_store_D_range": float(1.0 - inside / (hi - lo))}


# ----------------------------------------------------------------- class
def load_task14_member(member, n: int) -> dict[str, np.ndarray]:
    """A task 14 member store (schema 1, ``u = R^-1 s``) in the current schema (``u = R^-1 s_hat``)."""
    with np.load(TASK14_ARTIFACTS / "members" / path_id_of(member) / f"events_N{n}.npz") as data:
        return events_from_schema1({key: data[key] for key in data.files})


def read_task14_metrics(family: str, n: int) -> dict[tuple[int, int], dict[str, float]]:
    with (TASK14_ARTIFACTS / f"metrics_{family}_N{n}.csv").open() as handle:
        return {
            (int(r["row"]), int(r["column"])): {k: float(r[k]) for k in ("estimate", "K", "K_rho_pos", "K_eff")}
            for r in csv.DictReader(handle)
        }


def pooled_k_eff(stores, sun, density, row: int, column: int, render) -> float:
    """Task ``band-sum-renderer``'s ``K_eff``: Kish size of every ``(event, transport)`` pair (``per_transport_sample``).

    Recomputed literally for the comparison table only; the renderer counts distinct events.
    """
    centre, _, lo, hi = pixel_band(row, column, sun, render)
    total, square = 0.0, 0.0
    for events, group in stores:
        rotations, weight = band_poses(events, sun, centre, lo, hi)
        if len(weight) == 0:
            continue
        for t in group.transports:
            contribution = weight * density.evaluate_batch(rotations if t.g is None else rotations @ t.g.T)
            total += float(np.sum(contribution))
            square += float(np.sum(contribution**2))
    return kish_k_eff(total, square)


def single_member_k_eff(stores, sun, density, row: int, column: int, n: int, render) -> float:
    """``K_eff`` of one member's events: the single transport with the largest contribution (0 if none is lit).

    Not the representative's own: under a Lowitz density the representative is dark where another member is lit.
    """
    ((events, group),) = stores
    best = max(
        (class_band_sum_pixel([(events, StoreGroup(group.members, (t,)))], sun, density, row, column, n, render) for t in group.transports),
        key=lambda r: r.total,
    )
    return best.K_eff


def compare_profile(stores, family: str, spec: dict[str, Any], n: int, probe: dict, *, k_eff_table: bool = False) -> dict[str, Any]:
    sun = canonical_sun_direction()
    density = build_pose_density(family, **spec["density"])
    worst_rel, worst_counts, count, rows, table = 0.0, 0, 0, [], []
    for row, column in spec["pixels"]:
        result = class_band_sum_pixel(stores, sun, density, row, column, n, spec["render"])
        ref = probe[(row, column)]
        rel = abs(result.value - ref["estimate"]) / abs(ref["estimate"]) if ref["estimate"] else abs(result.value)
        worst_rel = max(worst_rel, rel)
        worst_counts += (result.K, result.K_rho_pos) != (int(ref["K"]), int(ref["K_rho_pos"]))
        count += 1
        rows.append((result.value, ref["estimate"], result.K_eff, ref["K_eff"]))
        if k_eff_table:
            one = single_member_k_eff(stores, sun, density, row, column, n, spec["render"])
            table.append((result.K_eff, pooled_k_eff(stores, sun, density, row, column, spec["render"]), one, ref["K_eff"]))
    out = {"pixels": count, "max_rel": worst_rel, "K_or_K_rho_pos_mismatches": worst_counts, "rows": rows}
    if k_eff_table:
        out["k_eff_rows"] = table
    return out


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
            crystal, CANONICAL_REFRACTIVE_INDEX, [(3, 5)], store_n, base_dir=store_cache_dir
        )
        (plan,) = store_plan(path_class, crystal)
        stores = [(store.events.arrays(), plan)]
        block = {
            "N": store_n,
            "cache_key": store.spec.cache_key(),
            "transports": len(plan.transports),
            "k_eff_semantics": {
                "K_eff": K_EFF_SEMANTICS,
                "K_eff_pooled": "per_transport_sample (task band-sum-renderer, recomputed)",
                "K_eff_single_member": "one member's events: the single transport with the largest contribution",
                "K_eff_task14": "task 14's twelve independent member stores, pooled (distinct events)",
            },
        }
        for family, spec in windows.items():
            probe = read_task14_metrics(family, store_n)
            result = compare_profile(stores, family, spec, store_n, probe, k_eff_table=True)
            value, ref, k_eff, ref_k_eff = (np.array(v) for v in zip(*result.pop("rows")))
            new, pooled, single, task14 = (np.array(v) for v in zip(*result.pop("k_eff_rows")))
            peak = ref > 0.1 * ref.max()
            ratio = value[peak] / ref[peak]
            # Two independent samplings of one integral: the difference over the combined noise.
            z = (value[peak] - ref[peak]) / (ref[peak] * np.sqrt(1.0 / k_eff[peak] + 1.0 / ref_k_eff[peak]))
            block[family] = {
                "peak_pixels": int(peak.sum()),
                "ratio_median": float(np.median(ratio)),
                "ratio_min": float(ratio.min()),
                "ratio_max": float(ratio.max()),
                "sum_ratio": float(value[peak].sum() / ref[peak].sum()),
                "z_rms": float(np.sqrt(np.mean(z**2))),
                "z_max_abs": float(np.max(np.abs(z))),
                "K_eff_median_peak": {
                    "K_eff": float(np.median(new[peak])),
                    "K_eff_pooled": float(np.median(pooled[peak])),
                    "K_eff_single_member": float(np.median(single[peak])),
                    "K_eff_task14": float(np.median(task14[peak])),
                },
                "K_eff_ratio_median_peak": {
                    "K_eff / single_member": float(np.median(new[peak] / single[peak])),
                    "pooled / single_member": float(np.median(pooled[peak] / single[peak])),
                    "task14 / K_eff": float(np.median(task14[peak] / new[peak])),
                },
            }
            print("transport", family, json.dumps(block[family]))
        report["transport"] = block
    return report


def stage_k_eff(random_n: int) -> dict[str, Any]:
    """Two i.i.d. stores of class ``[3, 5]`` on task 14's profiles: ``z`` of their difference under each ``K_eff``."""
    crystal = canonical_crystal()
    sun = canonical_sun_direction()
    path_class = build_path_class(crystal, (3, 5))
    (plan,) = store_plan(path_class, crystal)
    windows = json.loads((TASK14_ARTIFACTS / "profile_windows.json").read_text())["families"]
    bands = [pixel_band(r, c, sun, spec["render"])[2:] for spec in windows.values() for r, c in spec["pixels"]]
    window = (min(lo for lo, _ in bands), max(hi for _, hi in bands))
    stores = []
    for seed in (1, 2):
        start = time.perf_counter()
        sampler = RandomSphereSampler(seed)
        store = build_event_store(
            crystal, CANONICAL_REFRACTIVE_INDEX, [(3, 5)], random_n, sampler=sampler,
            sampling=f"{sampler.description}, seed {seed}", deviation_window=window, run_checks=False,
        )
        stores.append([(store.events.arrays(), plan)])
        print(f"random store seed {seed}: {len(store.events)} events in the window, {time.perf_counter() - start:.1f} s")
    report: dict[str, Any] = {
        "N": random_n, "sampling": RandomSphereSampler.description, "seeds": [1, 2],
        "deviation_window_rad": list(window), "k_eff_semantics": K_EFF_SEMANTICS,
        "z": "(a - b) / sqrt(a^2 / K_eff_a + b^2 / K_eff_b) over pixels with both values > 0 and both K_eff >= 30",
    }
    for family, spec in windows.items():
        density = build_pose_density(family, **spec["density"])
        rows = []
        for row, column in spec["pixels"]:
            a, b = (class_band_sum_pixel(st, sun, density, row, column, random_n, spec["render"]) for st in stores)
            pa, pb = (pooled_k_eff(st, sun, density, row, column, spec["render"]) for st in stores)
            one = single_member_k_eff(stores[0], sun, density, row, column, random_n, spec["render"])
            rows.append((a.value, b.value, a.K_eff, b.K_eff, pa, pb, one))
        va, vb, ka, kb, pa, pb, one = (np.array(v) for v in zip(*rows))
        used = (va > 0.0) & (vb > 0.0) & (np.minimum(ka, kb) >= 30.0)
        block: dict[str, Any] = {"pixels": len(rows), "used": int(used.sum())}
        for tag, (k1, k2) in (("per_event", (ka, kb)), ("per_transport_sample", (pa, pb))):
            z = (va[used] - vb[used]) / np.sqrt(va[used] ** 2 / k1[used] + vb[used] ** 2 / k2[used])
            block[tag] = {
                "z_rms": float(np.sqrt(np.mean(z**2))),
                "z_mean": float(np.mean(z)),
                "frac_abs_z_gt_2": float(np.mean(np.abs(z) > 2.0)),
                "z_max_abs": float(np.max(np.abs(z))),
                "K_eff_median": float(np.median(k1[used])),
            }
        block["K_eff_single_member_median"] = float(np.median(one[used]))
        report[family] = block
        print("k-eff", family, json.dumps(block))
    return report


# --------------------------------------------------------------- scatter
SCATTER_TOLERANCE = 1e-12
INNER_EDGE_ROWS = (48, 57)  # canonical camera rows about the 22 deg inner-edge caustic


def _csv_float(text: str) -> float:
    """A ``pixels.csv`` float; renders before chore ``band-sum-small-fixes`` wrote some as ``np.float64(x)``."""
    return float(text.removeprefix("np.float64(").removesuffix(")"))


def read_pixels_csv(band_dir: Path) -> dict[tuple[int, int], dict[str, float]]:
    with (band_dir / "pixels.csv").open() as handle:
        return {
            (int(r["row"]), int(r["column"])): {
                k: _csv_float(r[k]) for k in ("value", "K", "K_rho_pos", "K_eff", "delta_deg", "band_width_rad")
            }
            for r in csv.DictReader(handle)
        }


def relative_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.where(b != 0.0, np.abs(a - b) / np.where(b != 0.0, np.abs(b), 1.0), np.abs(a))


def compare_results(new: list, old: list, rows: tuple[int, int] | None = None) -> dict[str, Any]:
    """Two pixel lists of one scene (``BandSumPixelResult`` or ``pixels.csv`` dicts), pixel by pixel."""
    if not new:
        raise ValueError("compare_results: 'new' is empty, nothing to compare")
    get = (lambda r, k: r[k]) if isinstance(new[0], dict) else (lambda r, k: getattr(r, k))  # noqa: E731
    fields = {k: (np.array([get(r, k) for r in new]), np.array([get(r, k) for r in old])) for k in ("value", "K", "K_rho_pos", "K_eff")}
    value_new, value_old = fields["value"]
    lit = value_old > 0.0
    top = float(value_old.max()) if len(value_old) else 0.0
    rel = relative_difference(value_new, value_old)
    rel_k_eff = relative_difference(*fields["K_eff"])
    out: dict[str, Any] = {
        "pixels": len(new),
        "lit": int(lit.sum()),
        "lit_mismatch": int(np.count_nonzero(lit != (value_new > 0.0))),
        "K_mismatch": int(np.count_nonzero(fields["K"][0] != fields["K"][1])),
        "K_rho_pos_mismatch": int(np.count_nonzero(fields["K_rho_pos"][0] != fields["K_rho_pos"][1])),
        "value_max_rel_lit": float(rel[lit].max(initial=0.0)),
        "K_eff_max_rel_lit": float(rel_k_eff[lit].max(initial=0.0)),
        "value_max_abs_over_image_max": float(np.max(np.abs(value_new - value_old), initial=0.0) / top) if top else 0.0,
    }
    beyond = np.nonzero(lit & ((rel > SCATTER_TOLERANCE) | (rel_k_eff > SCATTER_TOLERANCE)))[0]
    out["beyond_tolerance"] = [
        {
            "row": int(get(new[i], "row")), "column": int(get(new[i], "column")),
            "value_rel": float(rel[i]), "K_eff_rel": float(rel_k_eff[i]),
            "value_over_image_max": float(value_old[i] / top), "K_eff": float(fields["K_eff"][1][i]),
        }
        for i in beyond[np.argsort(-rel[beyond])][:WORST]
    ]
    out["beyond_tolerance_count"] = int(len(beyond))
    if rows is not None:
        row_of = np.array([get(r, "row") for r in new])
        band = lit & (row_of >= rows[0]) & (row_of < rows[1])
        out["inner_edge_rows"] = {
            "rows": list(rows), "lit": int(band.sum()),
            "value_max_rel": float(rel[band].max(initial=0.0)), "K_eff_max_rel": float(rel_k_eff[band].max(initial=0.0)),
        }
    return out


def stage_scatter_dirs(band_dir: Path, baseline_dir: Path) -> dict[str, Any]:
    new, old = read_pixels_csv(band_dir), read_pixels_csv(baseline_dir)
    if new.keys() != old.keys():
        raise SystemExit(f"{band_dir} and {baseline_dir} render different pixels")
    keys = sorted(new)
    bands = [k for k in keys if (new[k]["delta_deg"], new[k]["band_width_rad"]) != (old[k]["delta_deg"], old[k]["band_width_rad"])]
    listed = lambda table: [{"row": k[0], "column": k[1], **table[k]} for k in keys]  # noqa: E731
    report = compare_results(listed(new), listed(old), INNER_EDGE_ROWS)
    report["band_mismatch"] = len(bands)
    report["execution"] = {
        name: {k: json.loads((d / "provenance.json").read_text())["execution"].get(k) for k in (
            "workers", "wall_clock_s", "render_wall_clock_s", "pixel_bands_wall_clock_s", "pixel_cpu_seconds",
            "max_rss_mb_parent", "max_rss_mb_worker", "renderer",
        )}
        for name, d in (("new", band_dir), ("baseline", baseline_dir))
    }
    return report


def stage_scatter_windows(random_n: int, store_cache_dir: Path, workers: int) -> dict[str, Any]:
    crystal = canonical_crystal()
    sun = canonical_sun_direction()
    report: dict[str, Any] = {"N": random_n, "tolerance": SCATTER_TOLERANCE}
    # Class [1,3,5]: 24 members, 12 of them through mirrors; random density; the window renderer itself.
    render = {"width": 61, "height": 61, "fov_deg": 150.0, "view": {"azimuth": 0.0, "elevation": 15.0}}
    scene = BandSumScene(
        build_path_class(crystal, (1, 3, 5)), crystal, CANONICAL_REFRACTIVE_INDEX, sun, build_pose_density("random"), render
    )
    start = time.perf_counter()
    new, execution = render_band_sum_window(scene, Window((0, 61), (0, 61)), random_n, workers=workers, base_dir=store_cache_dir)
    scatter_s = time.perf_counter() - start
    (group,) = scene.plan
    events = S2EventStore.load(Path(execution["stores"][0]["directory"])).events.arrays()
    start = time.perf_counter()
    old = [class_band_sum_pixel([(events, group)], sun, scene.pose_density, r.row, r.column, random_n, render) for r in new]
    block = compare_results(new, old)
    block.update(
        transports=len(group.transports), improper=int(sum(t.g is not None and np.linalg.det(t.g) < 0.0 for t in group.transports)),
        scatter_wall_clock_s=scatter_s, gather_wall_clock_s=time.perf_counter() - start, render=render, workers=workers,
    )
    report["class_1-3-5_random"] = block
    print("class 1-3-5", json.dumps({k: v for k, v in block.items() if k != "beyond_tolerance"}))
    del events
    # Task 14's three profiles on class [3,5]: one store, twelve transports; scatter_store against the gather.
    path_class = build_path_class(crystal, (3, 5))
    (group,) = store_plan(path_class, crystal)
    store = build_or_load(crystal, CANONICAL_REFRACTIVE_INDEX, [(3, 5)], random_n, base_dir=store_cache_dir, mmap_mode="r")
    arrays = S2EventStore.load(Path(store_cache_dir) / store.spec.cache_key()).events.arrays()
    windows = json.loads((TASK14_ARTIFACTS / "profile_windows.json").read_text())["families"]
    for family, spec in windows.items():
        density = build_pose_density(family, **spec["density"])
        start = time.perf_counter()
        bands = pixel_bands([tuple(p) for p in spec["pixels"]], sun, spec["render"])
        sums = ScatterSums.zeros(len(bands))
        scatter_store(store.events, group, density, bands, sums)
        new = scatter_results(bands, sums, random_n)
        scatter_s = time.perf_counter() - start
        start = time.perf_counter()
        old = [class_band_sum_pixel([(arrays, group)], sun, density, r, c, random_n, spec["render"]) for r, c in spec["pixels"]]
        block = compare_results(new, old)
        block.update(scatter_wall_clock_s=scatter_s, gather_wall_clock_s=time.perf_counter() - start)
        report[f"task14_{family}"] = block
        print("task 14", family, json.dumps({k: v for k, v in block.items() if k != "beyond_tolerance"}))

# --------------------------------------------------------------- contour
def _z_block(estimate: np.ndarray, reference: np.ndarray, k_eff: np.ndarray, lit: np.ndarray) -> dict[str, Any]:
    """``rel = estimate / reference - 1`` and ``z = rel sqrt(K_eff)`` on the lit pixels: bias and spread."""
    rel = estimate[lit] / reference[lit] - 1.0
    z = rel * np.sqrt(k_eff[lit])
    return {
        **lit_statistics(estimate, reference, lit),
        "mean_rel": float(np.mean(rel)),
        "mean_rel_standard_error": float(np.std(rel) / np.sqrt(rel.size)),
        "z_mean": float(np.mean(z)),
        "z_std": float(np.std(z)),
        "abs_z_percentiles_50_90_99_max": [float(np.percentile(np.abs(z), q)) for q in (50, 90, 99, 100)],
        "fraction_abs_z_above_4": float(np.mean(np.abs(z) > NOISE_Z)),
    }


def _max_relative_error_estimate(render_dir: Path, lit: np.ndarray) -> float:
    """Largest ``error_estimate / value`` of a contour render's ``pixels.csv`` over ``lit``."""
    worst = 0.0
    with (render_dir / "pixels.csv").open() as handle:
        for row in csv.DictReader(handle):
            if lit[int(row["row"]), int(row["column"])]:
                worst = max(worst, float(row["error_estimate"]) / float(row["value"]))
    return worst


def stage_contour(
    band_dir: Path, contour_dir: Path, coarse_dir: Path | None, point_dir: Path | None, reference_dir: Path | None
) -> dict[str, Any]:
    """The band sum against the deterministic contour quadrature on the band-sum pixel model (task s2-contour-quadrature, AC4).

    ``contour_dir`` is a ``render_contour_quadrature.py --band-nodes k`` render
    (band average over the same corners): the difference is the band sum's
    own error, ``z = rel sqrt(K_eff)``.  ``coarse_dir`` (the band sum at a
    smaller ``N``) shows whether it shrinks as ``N^-1/2``.  ``point_dir`` (a
    ``--band-nodes 0`` render) gives the size of the pixel-model difference
    and what the band sum looks like against the wrong model;
    ``reference_dir`` (Phase I ``strip-full``) is compared with the point render.
    """
    band, band_provenance = read_strip(band_dir)
    contour, contour_provenance = read_strip(contour_dir)
    if contour_provenance["options"].get("band_nodes", 0) < 1:
        raise ValueError(f"{contour_dir} is not a band-average render (band_nodes = 0)")
    if band.values.shape != contour.values.shape or not (band.rendered & contour.rendered).all():
        raise ValueError("the regression compares full images of one shape")
    ref = contour.values
    lit = ref > LIT_FRACTION * ref.max(axis=0, keepdims=True)
    k_eff = read_k_eff(band_dir, ref.shape)
    report: dict[str, Any] = {
        "band_dir": str(band_dir),
        "contour_dir": str(contour_dir),
        "N": band_provenance["options"]["N"],
        "k_eff_semantics": k_eff_semantics_of(band_provenance),
        "contour_pixel_model": contour_provenance["options"]["pixel_model"],
        "contour_relative_error_estimate_max_lit": _max_relative_error_estimate(contour_dir, lit),
        "lit_definition": f"contour band average > {LIT_FRACTION} x its column maximum",
        "band_sum_vs_contour_band": _z_block(band.values, ref, k_eff, lit),
        "whole_image_sum_ratio": float(band.values.sum() / ref.sum()),
    }
    rel = np.where(lit, band.values / np.where(lit, ref, 1.0) - 1.0, np.nan)
    if coarse_dir is not None:
        coarse, coarse_provenance = read_strip(coarse_dir)
        coarse_k_eff = read_k_eff(coarse_dir, ref.shape)
        block = _z_block(coarse.values, ref, coarse_k_eff, lit)
        n_ratio = band_provenance["options"]["N"] / coarse_provenance["options"]["N"]
        block["rms_rel_ratio_coarse_over_fine"] = block["rms_rel"] / report["band_sum_vs_contour_band"]["rms_rel"]
        block["sampling_expectation_sqrt_N_ratio"] = float(np.sqrt(n_ratio))
        block["N"] = coarse_provenance["options"]["N"]
        report["coarse_band_sum_vs_contour_band"] = block
    if point_dir is not None:
        point, _ = read_strip(point_dir)
        model = point.values[lit] / ref[lit] - 1.0
        steep = neighbour_change(point.values)
        report["pixel_model_point_vs_band"] = {
            "median_abs": float(np.median(np.abs(model))),
            "p99_abs": float(np.percentile(np.abs(model), 99)),
            "max_abs": float(np.max(np.abs(model))),
            "lit_pixels_above_1e-2": int(np.sum(np.abs(model) > 1e-2)),
            "lit_pixels_steep": int(np.sum(steep[lit] > STEEP_NEIGHBOUR_CHANGE)),
        }
        # an edge pixel whose band reaches the lit range while its centre does not has a point value of 0
        point_lit = lit & (point.values > 0.0)
        report["pixel_model_point_vs_band"]["lit_pixels_point_zero"] = int((lit & ~point_lit).sum())
        report["band_sum_vs_contour_point_wrong_model"] = _z_block(band.values, point.values, k_eff, point_lit)
        if reference_dir is not None:
            reference, _ = read_strip(reference_dir)
            both = lit & (reference.values > 0.0)
            phase1 = reference.values[both] / point.values[both] - 1.0
            report["phase1_strip_vs_contour_point"] = {
                "reference_dir": str(reference_dir),
                "lit_pixels": int(both.sum()),
                "median_abs_rel": float(np.median(np.abs(phase1))),
                "p99_abs_rel": float(np.percentile(np.abs(phase1), 99)),
                "max_abs_rel": float(np.max(np.abs(phase1))),
                "mean_rel": float(np.mean(phase1)),
            }
    order = np.argsort(np.where(lit, -np.abs(np.nan_to_num(rel * np.sqrt(k_eff))), 0.0), axis=None)[:WORST]
    report["worst_pixels_by_abs_z"] = [
        {"row": int(r), "column": int(c), "band_sum": float(band.values[r, c]), "contour_band": float(ref[r, c]),
         "rel": float(rel[r, c]), "K_eff": float(k_eff[r, c]), "z": float(rel[r, c] * np.sqrt(k_eff[r, c]))}
        for r, c in (np.unravel_index(flat, ref.shape) for flat in order)
    ]
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
    parser.add_argument("--stage", choices=("full", "class", "k-eff", "scatter", "contour", "figure"), required=True)
    parser.add_argument("--band-dir", type=Path, nargs="+")
    parser.add_argument("--coarse-dir", type=Path, default=None)
    parser.add_argument("--baseline-dir", type=Path, default=None, help="--stage scatter: the render --band-dir is compared with")
    parser.add_argument("--workers", type=int, default=1, help="--stage scatter: workers of the class [1,3,5] window render")
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=None,
        help="strip-full read_strip directory; default is a machine-specific path, pass explicitly on other machines",
    )
    parser.add_argument("--contour-dir", type=Path, default=None, help="--stage contour: a band-average contour render")
    parser.add_argument("--contour-point-dir", type=Path, default=None, help="--stage contour: a point-pixel contour render")
    parser.add_argument("--tiers", type=int, nargs="*", default=[1_000_000, 10_000_000, 50_000_000])
    parser.add_argument("--store-n", type=int, default=None)
    parser.add_argument("--store-cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--random-n", type=int, default=10_000_000, help="points of each i.i.d. store (--stage k-eff); store points (--stage scatter)"
    )
    parser.add_argument("--title", default="band-sum renderer (log scale, 4 decades)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference_dir_explicit = args.reference_dir is not None
    if args.reference_dir is None:
        args.reference_dir = DEFAULT_REFERENCE_DIR
    if args.stage == "figure":
        stage_figure(args.band_dir, args.output, args.title)
        return
    if args.stage == "full":
        report = stage_full(args.band_dir[0], args.reference_dir, args.coarse_dir)
        print(json.dumps({k: v for k, v in report.items() if k not in ("worst_pixels", "per_column_median_abs_rel", "unexplained_pixels")}, indent=1))
    elif args.stage == "contour":
        if args.contour_dir is None:
            parser.error("--stage contour needs --contour-dir")
        reference = None
        if args.contour_point_dir is not None:
            if args.reference_dir.exists():
                reference = args.reference_dir
            elif reference_dir_explicit:
                print(f"warning: --reference-dir {args.reference_dir} does not exist; skipping phase1_strip_vs_contour_point", file=sys.stderr)
        report = stage_contour(args.band_dir[0], args.contour_dir, args.coarse_dir, args.contour_point_dir, reference)
        print(json.dumps({k: v for k, v in report.items() if k != "worst_pixels_by_abs_z"}, indent=1))
    elif args.stage == "k-eff":
        report = stage_k_eff(args.random_n)
    elif args.stage == "scatter":
        if (args.band_dir is None) != (args.baseline_dir is None):
            parser.error("--stage scatter compares --band-dir with --baseline-dir (both), or renders windows (neither)")
        if args.band_dir is not None:
            report = stage_scatter_dirs(args.band_dir[0], args.baseline_dir)
            print(json.dumps({k: v for k, v in report.items() if k != "beyond_tolerance"}, indent=1))
        else:
            report = stage_scatter_windows(args.random_n, args.store_cache_dir, args.workers)
    else:
        report = stage_class(args.tiers, args.store_n, args.store_cache_dir)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("wrote", args.output)


if __name__ == "__main__":
    main()
