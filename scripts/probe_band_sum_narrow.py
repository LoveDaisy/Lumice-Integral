"""Band-sum probe on the narrow pose-density families (roadmap section 4.2): plate, Parry, Lowitz.

Task ``narrow-density-band-sum-probe``.  The band sum of ``scripts/probe_band_sum.py``
(one event store per labelled path, section 4.2) is applied to the ray-path *class*
``[3, 5]`` (``path_class.build_path_class``, its 12 PBD members): one event store per
member, the class value is the sum of the members' band sums.  Every member shares
the pixel's band ``[delta_lo, delta_hi]`` and its constant ``2 pi N (delta_hi -
delta_lo) sin(delta)``, so the class estimate is the band-sum estimate of the pooled
member contributions (``band_sum_estimate`` of the pooled total, ``kish_k_eff`` of the
pooled total and square) -- the estimator formulas stay in ``probe_band_sum.py``.

Stages (``--stage``):

- ``precompute``: per member, ``probe_band_sum.precompute`` into
  ``members/<path_id>/``.  Without ``--windowed`` the full store (used by
  ``locate``); with it only events whose deviation lies in the union of the
  profile pixels' bands (``profile_windows.json``), which gives the profile
  pixels bit-identical estimates at a fraction of the memory and disk.
  ``--sampling random`` builds an i.i.d. uniform store in ``members_random/``
  instead of the Fibonacci lattice (the ``1 / sqrt(K_eff)`` control).
- ``locate``: class band sum of the three families on a wide window around
  the sun (``--locate-n``, default ``1e7``), or on a ``--zoom`` window;
  ``locate*_<family>.npz`` / ``locate*.json``.
- ``profiles``: one pixel line per family through its lit region at the ch06
  pixel scale (``--profile family:el:az:row|column:length[:rationale]``,
  ``profile_windows.json``).
- ``reference``: Phase I ``path_class.render_class_pixel`` on every profile
  pixel, plus a refined-quadrature recomputation of six lit pixels
  (``reference_<family>.csv``).
- ``band-reference``: on the steep pixels (``STEEP_NEIGHBOUR_CHANGE``), the
  Phase I band average, the like-for-like reference of a band-sum pixel
  (``reference_band_<family>.json``).
- ``render``: the class band sum on the profiles at every precomputed tier
  (``metrics_<family>[_random]_N<n>.csv``, ``summary_<family>[_random].json``).
- ``verdict``: backend / needs_rho_aware_store / unusable per family
  (``verdict_<family>.json``).
- ``plot``: log-scale figures from the files above (``uv run --with matplotlib``).

The rho-aware store of the plan is only tried for a family whose
extrapolated ``N`` exceeds ``1e9``; none did (roadmap section 4.2), so it is
not implemented.  Nothing here imports or calls Lumice; ``src/`` is not modified.
Usage (``A`` = the artifacts directory)::

    uv run python scripts/probe_band_sum_narrow.py --stage precompute --tiers 10000000 --output-dir $A
    uv run python scripts/probe_band_sum_narrow.py --stage locate --output-dir $A --workers 3
    uv run python scripts/probe_band_sum_narrow.py --stage locate --output-dir $A --zoom 15.4 23.4 8.1 81
    uv run python scripts/probe_band_sum_narrow.py --stage profiles --output-dir $A --profile plate:15.0:24.4:row:121 ...
    uv run python scripts/probe_band_sum_narrow.py --stage reference --output-dir $A
    uv run python scripts/probe_band_sum_narrow.py --stage precompute --tiers 1000000 50000000 100000000 --windowed --output-dir $A --workers 3
    uv run python scripts/probe_band_sum_narrow.py --stage band-reference --output-dir $A
    uv run python scripts/probe_band_sum_narrow.py --stage render --output-dir $A --workers 3
    uv run python scripts/probe_band_sum_narrow.py --stage verdict --output-dir $A
    uv run --with matplotlib python scripts/probe_band_sum_narrow.py --stage plot --output-dir $A
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import dataclasses
import datetime as dt
import json
import multiprocessing
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from probe_band_sum import (
    LIT_FRACTION,
    TARGET_ERROR,
    band_contributions,
    band_sum_estimate,
    kish_k_eff,
    load_events,
    max_rss_mb,
    pixel_band,
    power_law,
    precompute,
)

from lumice_integral.canonical_scene import canonical_crystal, canonical_incident_direction
from lumice_integral.optics import path_id_of
from lumice_integral.path_class import PathClass, build_path_class
from lumice_integral.pose_density import build_pose_density

REPRESENTATIVE = (3, 5)
FAMILIES = ("plate", "parry", "lowitz")
# Lumice preset reference widths (docs/ch11-pose-density-families.md section 1).
FAMILY_PARAMETERS: dict[str, dict[str, float]] = {
    "plate": {"zenith_std_deg": 1.0},
    "parry": {"zenith_std_deg": 1.0, "roll_std_deg": 1.0},
    "lowitz": {"zenith_std_deg": 40.0, "roll_std_deg": 1.0},
}
LOCATE_RENDER: dict[str, Any] = {"width": 181, "height": 181, "fov_deg": 110.0, "view": {"azimuth": 0.0, "elevation": 30.0}}
MAX_WORKERS = 4


def family_density(family: str):
    return build_pose_density(family, **FAMILY_PARAMETERS[family])


def class_of_scene() -> PathClass:
    path_class = build_path_class(canonical_crystal(), REPRESENTATIVE)
    if path_class.halo_map_rank != 2 or path_class.size != 12:  # white-box premise of the plan (section 2)
        raise RuntimeError(f"class [3,5]: rank {path_class.halo_map_rank}, {path_class.size} members")
    return path_class


def member_dir(output_dir: Path, member: Sequence[int], store: str = "members") -> Path:
    return output_dir / store / path_id_of(member)


def pool(workers: int) -> concurrent.futures.ProcessPoolExecutor:
    return concurrent.futures.ProcessPoolExecutor(
        max_workers=min(workers, MAX_WORKERS), mp_context=multiprocessing.get_context("spawn")
    )


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


# ------------------------------------------------------------- precompute
RANDOM_SEED = 20260923


@dataclasses.dataclass(frozen=True)
class RandomSphereSampler:
    """i.i.d. uniform points on ``S^2`` (``q = 1 / 4 pi``, so no inverse weights); chunk-seeded, reproducible."""

    seed: int = RANDOM_SEED
    description = "i.i.d. uniform on S^2 (normalised Gaussian triples, numpy default_rng([seed, first]) per chunk)"

    def __call__(self, first: int, stop: int) -> tuple[np.ndarray, None]:
        points = np.random.default_rng([self.seed, first]).normal(size=(stop - first, 3))
        return points / np.linalg.norm(points, axis=1, keepdims=True), None


SAMPLERS = {"fibonacci": None, "random": RandomSphereSampler()}


def _precompute_member(args) -> dict[str, Any]:
    directory, member, n, run_checks, window, sampling = args
    directory.mkdir(parents=True, exist_ok=True)
    sampler = SAMPLERS[sampling]
    if sampler is None:
        return precompute(n, directory, run_checks=run_checks, faces=member, deviation_window=window)
    return precompute(
        n, directory, run_checks=run_checks, faces=member, deviation_window=window, sampler=sampler, sampling=sampler.description
    )


def profile_deviation_window(output_dir: Path) -> tuple[float, float]:
    """Union of the bands of every profile pixel (``profile_windows.json``), in radians."""
    s = canonical_incident_direction()
    lo, hi = np.inf, -np.inf
    for spec in json.loads((output_dir / "profile_windows.json").read_text())["families"].values():
        for row, column in spec["pixels"]:
            _, _, lo_d, hi_d = pixel_band(row, column, s, spec["render"])
            lo, hi = min(lo, lo_d), max(hi, hi_d)
    return float(lo), float(hi)


def stage_precompute(
    output_dir: Path, tiers: Sequence[int], *, windowed: bool, workers: int, skip_checks: bool, sampling: str = "fibonacci"
) -> None:
    path_class = class_of_scene()
    window = profile_deviation_window(output_dir) if windowed else None
    store = "members" if sampling == "fibonacci" else f"members_{sampling}"
    jobs = [
        (member_dir(output_dir, member, store), member, n, not skip_checks and i == 0 and j == 0, window, sampling)
        for j, n in enumerate(tiers)
        for i, member in enumerate(path_class.members)
    ]
    start = time.perf_counter()
    with pool(workers) as executor:
        metas = list(executor.map(_precompute_member, jobs))
    wall = time.perf_counter() - start
    summary_path = output_dir / "precompute_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {"runs": []}
    summary["class"] = path_class.provenance()
    summary["runs"].append(
        {
            "tiers": list(tiers),
            "windowed": windowed,
            "sampling": sampling,
            "store": store,
            "deviation_window_rad": window,
            "workers": min(workers, MAX_WORKERS),
            "wall_clock_s": wall,
            "members": {
                f"{path_id_of(member)}/N{meta['N']}": {
                    "N": meta["N"],
                    "kept_events": meta["kept_events"],
                    "kept_fraction": meta["kept_fraction"],
                    "fibonacci_mean_w": meta["fibonacci_mean_w"],
                    "wall_clock_s": meta["wall_clock_s"],
                    "max_rss_mb_process": meta["max_rss_mb_process"],
                    "self_checks": meta["self_checks"],
                }
                for (_, member, *_), meta in zip(jobs, metas)
            },
            "created": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        }
    )
    write_json(summary_path, summary)
    print(f"precompute {list(tiers)} windowed={windowed}: {wall:.1f} s wall")


# ----------------------------------------------------------------- pixels
@dataclasses.dataclass
class Accumulator:
    """Per pixel and density: pooled ``sum c``, ``sum c^2``, ``K``, ``K_rho_pos``, and the smallest member ``K_eff``."""

    total: np.ndarray
    square: np.ndarray
    k: np.ndarray
    k_pos: np.ndarray
    k_eff_min_member: np.ndarray
    max_rss_mb_worker: float = 0.0

    @classmethod
    def zeros(cls, densities: int, pixels: int) -> Accumulator:
        return cls(
            np.zeros((densities, pixels)),
            np.zeros((densities, pixels)),
            np.zeros((densities, pixels), dtype=np.int64),
            np.zeros((densities, pixels), dtype=np.int64),
            np.full((densities, pixels), np.inf),
        )

    def add(self, other: Accumulator) -> None:
        self.total += other.total
        self.square += other.square
        self.k += other.k
        self.k_pos += other.k_pos
        self.k_eff_min_member = np.minimum(self.k_eff_min_member, other.k_eff_min_member)
        self.max_rss_mb_worker = max(self.max_rss_mb_worker, other.max_rss_mb_worker)


def member_band_sums(
    events: dict[str, np.ndarray], densities: Sequence, pixels: Sequence[tuple[int, int]], render: Mapping[str, Any]
) -> Accumulator:
    """One member's band contributions at every pixel, per density (member ``K_eff`` only where it contributes)."""
    s = canonical_incident_direction()
    acc = Accumulator.zeros(len(densities), len(pixels))
    for p, (row, column) in enumerate(pixels):
        centre, _, lo_d, hi_d = pixel_band(row, column, s, render)
        for d, contribution in enumerate(band_contributions(events, s, densities, centre, lo_d, hi_d)):
            total, square = float(np.sum(contribution)), float(np.sum(contribution**2))
            acc.total[d, p], acc.square[d, p] = total, square
            acc.k[d, p], acc.k_pos[d, p] = len(contribution), np.count_nonzero(contribution > 0.0)
            if total > 0.0:
                acc.k_eff_min_member[d, p] = kish_k_eff(total, square)
    return acc


def _member_band_sums_job(args) -> Accumulator:
    directory, n, families, pixels, render = args
    acc = member_band_sums(load_events(directory, n), [family_density(f) for f in families], pixels, render)
    acc.max_rss_mb_worker = max_rss_mb()
    return acc


def class_band_sums(
    output_dir: Path,
    n: int,
    families: Sequence[str],
    pixels: Sequence[tuple[int, int]],
    render: Mapping[str, Any],
    workers: int,
    store: str = "members",
) -> Accumulator:
    """Pooled class accumulators: each member's store is loaded, summed at every pixel, and dropped."""
    path_class = class_of_scene()
    jobs = [(output_dir / store / path_id_of(m), n, tuple(families), list(pixels), dict(render)) for m in path_class.members]
    acc = Accumulator.zeros(len(families), len(pixels))
    with pool(workers) as executor:
        for member_acc in executor.map(_member_band_sums_job, jobs):
            acc.add(member_acc)
    return acc


def pixel_geometry(pixels: Sequence[tuple[int, int]], render: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    s = canonical_incident_direction()
    bands = [pixel_band(row, column, s, render) for row, column in pixels]
    return np.array([b[1] for b in bands]), np.array([b[3] - b[2] for b in bands])


def estimates(acc: Accumulator, n: int, delta: np.ndarray, width: np.ndarray) -> np.ndarray:
    return np.array(
        [[band_sum_estimate(t, n, w, d) for t, w, d in zip(row, width, delta)] for row in acc.total]
    )


# ----------------------------------------------------------------- locate
def sky_angles(row: int, column: int, render: Mapping[str, Any]) -> tuple[float, float]:
    """Elevation and azimuth (degrees) of the pixel centre's sky direction."""
    from lumice_integral.camera import linear_pixel_sky_direction

    sky = linear_pixel_sky_direction(row, column, **render)
    return float(np.degrees(np.arcsin(sky[2]))), float(np.degrees(np.arctan2(sky[1], sky[0])))


def stage_locate(output_dir: Path, n: int, workers: int, render: Mapping[str, Any], tag: str = "") -> None:
    pixels = [(r, c) for r in range(int(render["height"])) for c in range(int(render["width"]))]
    start = time.perf_counter()
    acc = class_band_sums(output_dir, n, FAMILIES, pixels, render, workers)
    wall = time.perf_counter() - start
    delta, width = pixel_geometry(pixels, render)
    values = estimates(acc, n, delta, width)
    shape = (int(render["height"]), int(render["width"]))
    report: dict[str, Any] = {
        "N": n,
        "render": dict(render),
        "wall_clock_s": wall,
        "max_rss_mb_process": max_rss_mb(),
        "max_rss_mb_worker": acc.max_rss_mb_worker,
        "families": {},
    }
    for f, family in enumerate(FAMILIES):
        image = values[f].reshape(shape)
        k_eff = np.array([kish_k_eff(t, q) for t, q in zip(acc.total[f], acc.square[f])]).reshape(shape)
        np.savez(output_dir / f"locate{tag}_{family}.npz", value=image, K=acc.k[f].reshape(shape), K_eff=k_eff)
        if not image.max() > 0.0:
            report["families"][family] = {"density": FAMILY_PARAMETERS[family], "peak_value": 0.0}
            print(family, "no light in this window")
            continue
        peak = np.unravel_index(int(np.argmax(image)), shape)
        lit = image > LIT_FRACTION * image.max()
        rows, cols = np.nonzero(lit)
        report["families"][family] = {
            "density": FAMILY_PARAMETERS[family],
            "peak_pixel": [int(peak[0]), int(peak[1])],
            "peak_value": float(image[peak]),
            "peak_sky_elevation_azimuth_deg": sky_angles(int(peak[0]), int(peak[1]), render),
            "peak_on_window_edge": bool(peak[0] in (0, shape[0] - 1) or peak[1] in (0, shape[1] - 1)),
            "peak_K_eff": float(k_eff[peak]),
            "lit_pixels": int(lit.sum()),
            "lit_row_range": [int(rows.min()), int(rows.max())],
            "lit_column_range": [int(cols.min()), int(cols.max())],
        }
        print(family, json.dumps(report["families"][family]))
    write_json(output_dir / f"locate{tag}.json", report)


# --------------------------------------------------------------- profiles
PROFILE_PIXEL_DEG = 0.024  # the ch06 strip's pixel (fov 6 deg over 251 px, 0.0239 deg)


def profile_render(elevation_deg: float, azimuth_deg: float, length: int) -> dict[str, Any]:
    """``length x length`` linear window centred on ``(elevation, azimuth)`` with ``PROFILE_PIXEL_DEG`` at the centre."""
    half_tangent = length * np.tan(np.radians(PROFILE_PIXEL_DEG)) / 2.0
    return {
        "width": length,
        "height": length,
        "fov_deg": float(2.0 * np.degrees(np.arctan(half_tangent))),
        "view": {"azimuth": float(azimuth_deg), "elevation": float(elevation_deg)},
    }


def stage_profiles(output_dir: Path, specs: Sequence[str]) -> None:
    """``family:elevation:azimuth:orientation:length[:rationale]`` -> the middle row (``row``) or column of the window."""
    families: dict[str, Any] = {}
    for spec in specs:
        family, elevation, azimuth, orientation, length, *rationale = spec.split(":", 5)
        length_i = int(length)
        if length_i % 2 == 0 or length_i < 60 or orientation not in ("row", "column"):
            raise ValueError(f"profile {spec!r}: odd length >= 60 and orientation row|column required")
        render = profile_render(float(elevation), float(azimuth), length_i)
        mid = length_i // 2
        pixels = [(mid, c) for c in range(length_i)] if orientation == "row" else [(r, mid) for r in range(length_i)]
        first, last = sky_angles(*pixels[0], render), sky_angles(*pixels[-1], render)
        families[family] = {
            "density": FAMILY_PARAMETERS[family],
            "centre_elevation_azimuth_deg": [float(elevation), float(azimuth)],
            "orientation": orientation,
            "length": length_i,
            "render": render,
            "pixels": [list(p) for p in pixels],
            "first_last_pixel_sky_elevation_azimuth_deg": [first, last],
            "rationale": rationale[0] if rationale else "",
        }
    path = output_dir / "profile_windows.json"
    payload = json.loads(path.read_text()) if path.exists() else {"pixel_deg": PROFILE_PIXEL_DEG, "families": {}}
    payload["families"].update(families)
    write_json(path, payload)
    print(json.dumps({f: {k: v for k, v in b.items() if k != "pixels"} for f, b in families.items()}, indent=1))


def load_profile(output_dir: Path, family: str) -> tuple[list[tuple[int, int]], dict[str, Any]]:
    spec = json.loads((output_dir / "profile_windows.json").read_text())["families"][family]
    return [tuple(p) for p in spec["pixels"]], spec["render"]


# -------------------------------------------------------------- reference
REFINED_QUADRATURE = {"relative_tolerance": 1e-6, "initial_node_count": 513, "maximum_node_count": 4097}
REFINED_PIXELS = 6  # >= 5 lit pixels per family get the refined quadrature (issue)
REFINEMENT_THRESHOLD = 1e-3
_SCENES: dict[tuple[str, str], Any] = {}


def _class_scene(family: str, render: Mapping[str, Any]):
    from lumice_integral.path_class import canonical_class_scene

    key = (family, json.dumps(render, sort_keys=True))
    if key not in _SCENES:
        _SCENES[key] = canonical_class_scene(REPRESENTATIVE, pose_density=family_density(family), render=render)
        if _SCENES[key].path_class.halo_map_rank != 2:
            raise RuntimeError("class [3,5] must be rank 2")
    return _SCENES[key]


def _reference_job(args) -> list[dict[str, Any]]:
    from lumice_integral.quadrature import ResampleOptions
    from lumice_integral.path_class import render_class_pixel
    from lumice_integral.strip_pixel import PixelOptions

    family, render, pixels, refined = args
    scene = _class_scene(family, render)
    options = PixelOptions(quadrature=ResampleOptions(**REFINED_QUADRATURE)) if refined else PixelOptions()
    rows = []
    for row, column in pixels:
        start = time.perf_counter()
        result = render_class_pixel(scene, row, column, options)
        rows.append(
            {
                "row": row,
                "column": column,
                "value": result.value,
                "error_estimate": result.error_estimate,
                "completeness": result.completeness,
                "node_count_exhausted": int(result.events["quadrature_node_count_exhausted"]),
                "components": result.component_count,
                "seconds": time.perf_counter() - start,
            }
        )
    return rows


def _phase1_pixels(family: str, render, pixels, workers: int, refined: bool) -> list[dict[str, Any]]:
    workers = min(workers, MAX_WORKERS)
    chunks = [pixels[i::workers] for i in range(workers)]
    with pool(workers) as executor:
        results = [r for chunk in executor.map(_reference_job, [(family, dict(render), c, refined) for c in chunks if c]) for r in chunk]
    return sorted(results, key=lambda r: pixels.index((r["row"], r["column"])))


REFERENCE_FIELDS = (
    "row", "column", "value_default", "error_default", "completeness_default", "node_exhausted_default",
    "components", "seconds_default", "value_refined", "error_refined", "node_exhausted_refined",
    "rel_diff", "refined_used", "reference",
)


def stage_reference(output_dir: Path, families: Sequence[str], workers: int) -> None:
    for family in families:
        pixels, render = load_profile(output_dir, family)
        start = time.perf_counter()
        default = _phase1_pixels(family, render, pixels, workers, refined=False)
        default_wall = time.perf_counter() - start
        values = np.array([r["value"] for r in default])
        lit = np.flatnonzero(values > LIT_FRACTION * values.max())
        # the peak and five more lit pixels spread evenly over the lit set
        chosen = sorted({int(np.argmax(values)), *(int(lit[i]) for i in np.linspace(0, len(lit) - 1, REFINED_PIXELS - 1).round().astype(int))})
        start = time.perf_counter()
        refined = _phase1_pixels(family, render, [pixels[i] for i in chosen], workers, refined=True)
        refined_wall = time.perf_counter() - start
        by_pixel = {(r["row"], r["column"]): r for r in refined}
        records = []
        for r in default:
            fine = by_pixel.get((r["row"], r["column"]))
            rel = (fine["value"] - r["value"]) / fine["value"] if fine and fine["value"] > 0 else float("nan")
            use = bool(fine is not None and abs(rel) > REFINEMENT_THRESHOLD)
            records.append(
                {
                    "row": r["row"],
                    "column": r["column"],
                    "value_default": r["value"],
                    "error_default": r["error_estimate"],
                    "completeness_default": r["completeness"],
                    "node_exhausted_default": r["node_count_exhausted"],
                    "components": r["components"],
                    "seconds_default": r["seconds"],
                    "value_refined": fine["value"] if fine else "",
                    "error_refined": fine["error_estimate"] if fine else "",
                    "node_exhausted_refined": fine["node_count_exhausted"] if fine else "",
                    "rel_diff": rel if fine else "",
                    "refined_used": use,
                    "reference": fine["value"] if use else r["value"],
                }
            )
        with (output_dir / f"reference_{family}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=REFERENCE_FIELDS)
            writer.writeheader()
            writer.writerows(records)
        rels = [abs(r["rel_diff"]) for r in records if r["rel_diff"] != ""]
        meta = {
            "pixels": len(records),
            "lit_pixels": int(len(lit)),
            "refined_pixels": len(refined),
            "max_abs_rel_diff_refined_vs_default": float(np.nanmax(rels)) if rels else None,
            "refined_used": int(sum(r["refined_used"] for r in records)),
            "incomplete_default": int(sum(r["completeness_default"] != "complete" for r in records)),
            "node_exhausted_default_pixels": int(sum(r["node_exhausted_default"] > 0 for r in records)),
            "node_exhausted_refined_pixels": int(sum(r["node_count_exhausted"] > 0 for r in refined)),
            "default_wall_s": default_wall,
            "refined_wall_s": refined_wall,
            "pixel_s_mean_default": float(np.mean([r["seconds"] for r in default])),
            "refined_quadrature": REFINED_QUADRATURE,
            "workers": min(workers, MAX_WORKERS),
            "created": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        write_json(output_dir / f"reference_{family}.json", meta)
        print(family, json.dumps(meta))


# The band sum's pixel is the band average of I(delta') sin(delta') / sin(delta) over the pixel's
# deviation band at the centre azimuth; the Phase I reference is the pixel-centre value.  Where the
# reference changes by more than STEEP_NEIGHBOUR_CHANGE from a neighbouring pixel the two pixel models
# differ visibly, and the Phase I band average (BAND_NODES midpoint nodes, render_class_pixel with a
# target override) is the like-for-like reference.  The criterion reads the reference only.
STEEP_NEIGHBOUR_CHANGE = 0.1
BAND_NODES = 8


def steep_pixels(values: np.ndarray) -> list[int]:
    lit = values > LIT_FRACTION * values.max()
    steep = []
    for i in np.flatnonzero(lit):
        change = [abs(values[j] / values[i] - 1.0) for j in (i - 1, i + 1) if 0 <= j < len(values)]
        if max(change) > STEEP_NEIGHBOUR_CHANGE:
            steep.append(int(i))
    return steep


def _band_reference_job(args) -> list[dict[str, Any]]:
    from lumice_integral.path_class import render_class_pixel
    from lumice_integral.strip_pixel import PixelOptions

    family, render, pixels = args
    scene = _class_scene(family, render)
    s = canonical_incident_direction()
    out = []
    for row, column in pixels:
        centre, delta, lo_d, hi_d = pixel_band(row, column, s, render)
        e = centre - (centre @ s) * s
        e /= np.linalg.norm(e)
        nodes = lo_d + (np.arange(BAND_NODES) + 0.5) * (hi_d - lo_d) / BAND_NODES
        values = [
            render_class_pixel(scene, row, column, PixelOptions(), target=np.cos(d) * s + np.sin(d) * e).value
            for d in nodes
        ]
        band = float(np.mean(np.array(values) * np.sin(nodes)) / np.sin(delta))
        out.append({"row": row, "column": column, "reference_band": band, "node_values": values})
    return out


def stage_band_reference(output_dir: Path, families: Sequence[str], workers: int) -> None:
    for family in families:
        pixels, render = load_profile(output_dir, family)
        with (output_dir / f"reference_{family}.csv").open() as handle:
            point = [float(r["reference"]) for r in csv.DictReader(handle)]
        chosen = [pixels[i] for i in steep_pixels(np.array(point))]
        workers_n = min(workers, MAX_WORKERS)
        start = time.perf_counter()
        with pool(workers_n) as executor:
            jobs = [(family, dict(render), chosen[i::workers_n]) for i in range(workers_n) if chosen[i::workers_n]]
            rows = [r for chunk in executor.map(_band_reference_job, jobs) for r in chunk]
        wall = time.perf_counter() - start
        index = {p: i for i, p in enumerate(pixels)}
        rows.sort(key=lambda r: index[(r["row"], r["column"])])
        for r in rows:
            r["pixel_index"] = index[(r["row"], r["column"])]
            r["reference_point"] = point[r["pixel_index"]]
            r["band_over_point"] = r["reference_band"] / r["reference_point"]
        write_json(
            output_dir / f"reference_band_{family}.json",
            {
                "criterion": f"lit and |I(neighbour) / I - 1| > {STEEP_NEIGHBOUR_CHANGE} (reference only)",
                "band_nodes": BAND_NODES,
                "rule": "midpoint nodes in [delta_lo, delta_hi] at the centre azimuth, mean of I sin(delta') / sin(delta)",
                "wall_clock_s": wall,
                "pixels": rows,
            },
        )
        print(family, len(rows), "steep pixels, band/point:", [round(r["band_over_point"], 4) for r in rows])


def load_band_reference(output_dir: Path, family: str) -> dict[tuple[int, int], float]:
    path = output_dir / f"reference_band_{family}.json"
    if not path.exists():
        return {}
    return {(r["row"], r["column"]): r["reference_band"] for r in json.loads(path.read_text())["pixels"]}


def load_reference(output_dir: Path, family: str) -> dict[tuple[int, int], float]:
    with (output_dir / f"reference_{family}.csv").open() as handle:
        return {(int(r["row"]), int(r["column"])): float(r["reference"]) for r in csv.DictReader(handle)}


# ----------------------------------------------------------------- render
@dataclasses.dataclass(frozen=True)
class ProfileMetrics:
    """One row of ``metrics_<family>_N<n>.csv``."""

    row: int
    column: int
    delta_deg: float
    band_width_rad: float
    K: int
    K_rho_pos: int
    K_eff: float
    K_eff_min_member: float
    estimate: float
    reference: float
    lit: bool
    rel_error: float
    reference_band: float  # band-averaged Phase I on the steep pixels, the point reference elsewhere
    rel_error_band: float


def available_class_tiers(output_dir: Path, store: str = "members") -> list[int]:
    first = output_dir / store / path_id_of(class_of_scene().members[0])
    return sorted(int(p.stem.removeprefix("events_N")) for p in first.glob("events_N*.npz"))


def profile_metrics(
    output_dir: Path, family: str, n: int, workers: int, store: str
) -> tuple[list[ProfileMetrics], float, float]:
    pixels, render = load_profile(output_dir, family)
    reference = load_reference(output_dir, family)
    band_reference = load_band_reference(output_dir, family)
    start = time.perf_counter()
    acc = class_band_sums(output_dir, n, [family], pixels, render, workers, store)
    wall = time.perf_counter() - start
    delta, width = pixel_geometry(pixels, render)
    estimate = estimates(acc, n, delta, width)[0]
    peak = max(reference.values())
    rows = []
    for p, (row, column) in enumerate(pixels):
        ref = reference[(row, column)]
        lit = ref > LIT_FRACTION * peak
        k_min = float(acc.k_eff_min_member[0, p])
        rows.append(
            ProfileMetrics(
                row, column, float(np.degrees(delta[p])), float(width[p]), int(acc.k[0, p]), int(acc.k_pos[0, p]),
                kish_k_eff(acc.total[0, p], acc.square[0, p]), k_min if np.isfinite(k_min) else 0.0,
                float(estimate[p]), ref, bool(lit), (estimate[p] - ref) / ref if lit else float("nan"),
                band_reference.get((row, column), ref),
                (estimate[p] - band_reference.get((row, column), ref)) / band_reference.get((row, column), ref)
                if lit else float("nan"),
            )
        )
    return rows, wall, acc.max_rss_mb_worker


def tier_summary(rows: Sequence[ProfileMetrics], n: int, wall: float, worker_rss: float) -> dict[str, Any]:
    lit = [r for r in rows if r.lit]
    rel = np.array([r.rel_error for r in lit])
    positive = [r for r in lit if r.estimate > 0]
    log_ratio = np.log([r.estimate / r.reference for r in positive]) if positive else np.array([np.nan])
    k_eff = np.array([r.K_eff for r in lit])
    rel_band = np.array([r.rel_error_band for r in lit])
    return {
        "pixels": len(rows),
        "lit_pixels": len(lit),
        # against the like-for-like (band-averaged on steep pixels) reference: the sampling error
        "band_ref_rms_rel_error": float(np.sqrt(np.mean(rel_band**2))),
        "band_ref_median_abs_rel_error": float(np.median(np.abs(rel_band))),
        "band_ref_p95_abs_rel_error": float(np.percentile(np.abs(rel_band), 95)),
        "band_ref_max_abs_rel_error": float(np.max(np.abs(rel_band))),
        "band_ref_pixels": int(sum(r.reference_band != r.reference for r in lit)),
        "lit_zero_estimate": len(lit) - len(positive),
        "rms_rel_error": float(np.sqrt(np.mean(rel**2))),
        "median_abs_rel_error": float(np.median(np.abs(rel))),
        "p95_abs_rel_error": float(np.percentile(np.abs(rel), 95)),
        "max_abs_rel_error": float(np.max(np.abs(rel))),
        "log_rms": float(np.sqrt(np.mean(log_ratio**2))),
        # ~0.67 if the error is sampling noise of size 1/sqrt(K_eff); much larger = systematic
        "median_abs_rel_times_sqrt_K_eff": float(np.median(np.abs(rel) * np.sqrt(k_eff))),
        "median_ratio": float(np.median([r.estimate / r.reference for r in lit])),
        "sum_ratio": float(sum(r.estimate for r in lit) / sum(r.reference for r in lit)),
        "K_median_lit": float(np.median([r.K for r in lit])),
        "K_eff_median_lit": float(np.median(k_eff)),
        "K_eff_min_lit": float(np.min(k_eff)),
        "K_eff_over_K_median_lit": float(np.median([r.K_eff / r.K for r in lit if r.K > 0])),
        "K_eff_per_N_median_lit": float(np.median(k_eff) / n),
        "K_eff_min_member_median_lit": float(np.median([r.K_eff_min_member for r in lit])),
        "render_wall_s": wall,
        "render_s_per_pixel": wall / len(rows),
        "max_rss_mb_process": max_rss_mb(),
        "max_rss_mb_worker": worker_rss,
    }


def n_for_target(fit: Mapping[str, float] | None) -> float | None:
    if fit is None or fit["slope"] >= 0:
        return None
    return float(10 ** ((np.log10(TARGET_ERROR) - fit["log10_C"]) / fit["slope"]))


def extrapolate(per_tier: Mapping[str, Mapping[str, float]]) -> dict[str, Any]:
    """Power laws in ``N`` of the lit RMS errors and median ``K_eff``; ``N`` for a ``1e-2`` lit RMS.

    ``N_for_target`` uses the like-for-like reference (sampling error); ``N_for_target_point_reference``
    the pixel-centre reference, which also carries the pixel-model difference on steep pixels.
    """
    ns = sorted(int(n) for n in per_tier)
    fit = power_law(ns, [per_tier[str(n)]["band_ref_rms_rel_error"] for n in ns])
    point_fit = power_law(ns, [per_tier[str(n)]["rms_rel_error"] for n in ns])
    out: dict[str, Any] = {
        "tiers": ns,
        "band_ref_rms_rel_error_fit": fit,
        "N_for_target": n_for_target(fit),
        "rms_rel_error_fit": point_fit,
        "N_for_target_point_reference": n_for_target(point_fit),
    }
    out["K_eff_median_fit"] = power_law(ns, [per_tier[str(n)]["K_eff_median_lit"] for n in ns])
    # Conservative bound: the error of scattered points is 1/sqrt(K_eff) (the i.i.d. random store
    # measures |rel| sqrt(K_eff) ~ 0.6); the lattice usually does better but not reliably (aliasing).
    # K_eff is linear in N, so N = TARGET^-2 / (K_eff / N) at the largest tier.
    last = per_tier[str(ns[-1])]
    out["N_for_target_mc_bound_median_pixel"] = float(TARGET_ERROR**-2 / last["K_eff_per_N_median_lit"])
    out["N_for_target_mc_bound_worst_pixel"] = float(TARGET_ERROR**-2 * ns[-1] / last["K_eff_min_lit"])
    return out


def stage_render(output_dir: Path, families: Sequence[str], workers: int, store: str = "members", tag: str = "") -> None:
    tiers = available_class_tiers(output_dir, store)
    for family in families:
        per_tier: dict[str, Any] = {}
        for n in tiers:
            rows, wall, worker_rss = profile_metrics(output_dir, family, n, workers, store)
            with (output_dir / f"metrics_{family}{tag}_N{n}.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=[f.name for f in dataclasses.fields(ProfileMetrics)])
                writer.writeheader()
                writer.writerows(dataclasses.asdict(r) for r in rows)
            per_tier[str(n)] = tier_summary(rows, n, wall, worker_rss)
            print(family + tag, n, json.dumps(per_tier[str(n)]))
        summary = {
            "family": family,
            "density": FAMILY_PARAMETERS[family],
            "store": store,
            "tiers": per_tier,
            "extrapolation": extrapolate(per_tier),
            "created": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        write_json(output_dir / f"summary_{family}{tag}.json", summary)
        print(family + tag, "extrapolation", json.dumps(summary["extrapolation"]))


# ---------------------------------------------------------------- verdict
BACKEND_N = 1e9  # N_star above this: collapsed (issue: "extrapolated N > 1e9")
STOP_LOSS_N = 1e10


def stage_verdict(output_dir: Path, families: Sequence[str]) -> None:
    """Three-way verdict per family from the uniform-store summary (and the rho-aware one, if run)."""
    for family in families:
        summary = json.loads((output_dir / f"summary_{family}.json").read_text())
        n_star = summary["extrapolation"]["N_for_target"]
        tiers = summary["tiers"]
        reached = sorted(int(n) for n in tiers if tiers[n]["band_ref_rms_rel_error"] <= TARGET_ERROR)
        verdict: dict[str, Any] = {
            "family": family,
            "N_star_uniform": n_star,
            "N_star_uniform_point_reference": summary["extrapolation"]["N_for_target_point_reference"],
            "N_mc_bound_median_pixel": summary["extrapolation"]["N_for_target_mc_bound_median_pixel"],
            "N_mc_bound_worst_pixel": summary["extrapolation"]["N_for_target_mc_bound_worst_pixel"],
            "tiers": sorted(int(n) for n in tiers),
            "smallest_tier_reaching_target": reached[0] if reached else None,
        }
        if n_star is not None and n_star <= BACKEND_N:
            verdict["verdict"] = "backend"
            verdict["confirmed"] = bool(reached and reached[0] <= BACKEND_N)  # a measured tier <= 1e9 reaches 1e-2
            verdict["robust_to_lost_lattice_gain"] = bool(verdict["N_mc_bound_worst_pixel"] <= BACKEND_N)
        else:
            importance = output_dir / f"summary_{family}_rho.json"
            if importance.exists():
                rho = json.loads(importance.read_text())
                verdict["N_star_rho_aware"] = rho["extrapolation"]["N_for_target"]
                good = verdict["N_star_rho_aware"] is not None and verdict["N_star_rho_aware"] <= BACKEND_N
                verdict["verdict"] = "needs_rho_aware_store" if good else "unusable"
            else:
                verdict["verdict"] = "collapsed (rho-aware store not run)"
        verdict["created"] = dt.datetime.now().astimezone().isoformat(timespec="seconds")
        write_json(output_dir / f"verdict_{family}.json", verdict)
        print(family, json.dumps(verdict))


# ------------------------------------------------------------------- plot
def plot_locate(output_dir: Path, plt) -> None:
    from matplotlib.colors import LogNorm

    report = json.loads((output_dir / "locate.json").read_text())
    render = report["render"]
    profiles_path = output_dir / "profile_windows.json"
    profiles = json.loads(profiles_path.read_text())["families"] if profiles_path.exists() else {}
    figure, axes = plt.subplots(2, 3, figsize=(16, 10))
    for f, family in enumerate(FAMILIES):
        with np.load(output_dir / f"locate_{family}.npz") as data:
            image, k_eff = data["value"], data["K_eff"]
        top = image.max()
        shown = axes[0, f].imshow(np.where(image > 0, image, np.nan), norm=LogNorm(top * 1e-5, top), cmap="magma")
        figure.colorbar(shown, ax=axes[0, f], shrink=0.8)
        axes[0, f].set_title(f"{family}: class [3,5] band sum, N={report['N']:.0e} (log)")
        shown = axes[1, f].imshow(np.where(k_eff > 0, k_eff, np.nan), norm=LogNorm(1, max(k_eff.max(), 10)), cmap="viridis")
        figure.colorbar(shown, ax=axes[1, f], shrink=0.8)
        axes[1, f].set_title(f"{family}: K_eff (Kish, pooled members)")
        if family in profiles:  # the profile's end points projected into the wide window
            from lumice_integral.camera import project_linear

            ends = []
            for elevation, azimuth in profiles[family]["first_last_pixel_sky_elevation_azimuth_deg"]:
                e, a = np.radians(elevation), np.radians(azimuth)
                ends.append(project_linear(np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)]), **render))
            for axis in axes[:, f]:
                axis.plot([ends[0][0] - 0.5, ends[1][0] - 0.5], [ends[0][1] - 0.5, ends[1][1] - 0.5], "c-", lw=2)
        for axis in axes[:, f]:
            axis.set_xlabel("column")
            axis.set_ylabel("row")
    figure.suptitle(
        f"wide window: view az {render['view']['azimuth']} el {render['view']['elevation']}, fov {render['fov_deg']} deg, "
        f"{render['width']}x{render['height']} px (sun at el 15, az 0)"
    )
    figure.tight_layout()
    figure.savefig(output_dir / "locate.png", dpi=110)
    plt.close(figure)


def read_metrics(path: Path) -> dict[str, np.ndarray]:
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    return {key: np.array([float(r[key] == "True") if key == "lit" else float(r[key]) for r in rows]) for key in rows[0]}


def plot_profiles(output_dir: Path, plt) -> None:
    windows = json.loads((output_dir / "profile_windows.json").read_text())["families"]
    for family, spec in windows.items():
        summary_path = output_dir / f"summary_{family}.json"
        if not summary_path.exists():
            continue
        variants = [("", json.loads(summary_path.read_text()))]
        if (output_dir / f"summary_{family}_rho.json").exists():
            variants.append(("_rho", json.loads((output_dir / f"summary_{family}_rho.json").read_text())))
        figure, axes = plt.subplots(3, 1, figsize=(9, 11), sharex=True)
        x = None
        for tag, summary in variants:
            for n in summary["tiers"]:
                m = read_metrics(output_dir / f"metrics_{family}{tag}_N{n}.csv")
                x = np.arange(len(m["estimate"]))
                label = f"{'rho-aware' if tag else 'uniform'} N={int(n):.0e}"
                style = "x" if tag else "."
                axes[0].semilogy(x, np.where(m["estimate"] > 0, m["estimate"], np.nan), style, ms=3, label=label)
                lit = m["lit"] > 0
                axes[1].semilogy(x[lit], np.abs(m["rel_error_band"][lit]), style, ms=3, label=label)
                axes[2].semilogy(x, np.maximum(m["K_eff"], 1e-1), style, ms=3, label=label)
        axes[0].semilogy(x, np.where(m["reference"] > 0, m["reference"], np.nan), "k-", lw=0.8, label="Phase I reference")
        steep = m["reference_band"] != m["reference"]
        axes[0].semilogy(x[steep], m["reference_band"][steep], "k+", ms=6, label="Phase I band average (steep px)")
        axes[1].semilogy(x[steep], np.abs(m["rel_error"][steep]), "+", color="0.5", ms=6, label="vs point reference (steep px)")
        axes[0].set_ylabel("I (pixel value)")
        axes[1].axhline(TARGET_ERROR, color="k", ls="--", lw=0.8)
        axes[1].set_ylabel("|relative error| (lit, like-for-like ref.)")
        axes[2].set_ylabel("K_eff (Kish, pooled members)")
        (e0, a0), (e1, a1) = spec["first_last_pixel_sky_elevation_azimuth_deg"]
        axes[2].set_xlabel(
            f"profile pixel ({spec['orientation']}, {PROFILE_PIXEL_DEG} deg/px; el/az {e0:.2f}/{a0:.2f} -> {e1:.2f}/{a1:.2f} deg)"
        )
        for axis in axes:
            axis.legend(fontsize=7)
            axis.grid(True, which="both", alpha=0.3)
        figure.suptitle(f"class [3,5] band sum vs Phase I, {family} {FAMILY_PARAMETERS[family]}")
        figure.tight_layout()
        figure.savefig(output_dir / f"profile_{family}.png", dpi=130)
        plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for family in windows:
        colour = {"plate": "C0", "parry": "C1", "lowitz": "C2"}[family]
        for tag, style in (("", "o-"), ("_random", "^:"), ("_rho", "s--")):
            path = output_dir / f"summary_{family}{tag}.json"
            if not path.exists():
                continue
            block = json.loads(path.read_text())["tiers"]
            ns = sorted(int(n) for n in block)
            name = family + {"": " Fibonacci", "_random": " i.i.d. random", "_rho": " rho-aware"}[tag]
            axes[0].loglog(ns, [block[str(n)]["band_ref_rms_rel_error"] for n in ns], style, color=colour, label=f"{name} RMS")
            axes[1].loglog(ns, [block[str(n)]["K_eff_median_lit"] for n in ns], style, color=colour, label=name)
    axes[0].axhline(TARGET_ERROR, color="k", ls="--", lw=0.8)
    for axis, label in zip(axes, ("lit RMS relative error (like-for-like ref.)", "median K_eff (lit)")):
        axis.set_xlabel("N (points on S^2 per member store)")
        axis.set_ylabel(label)
        axis.legend(fontsize=7)
        axis.grid(True, which="both", alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_dir / "convergence.png", dpi=130)
    plt.close(figure)


def stage_plot(output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if (output_dir / "locate.json").exists():
        plot_locate(output_dir, plt)
    if (output_dir / "profile_windows.json").exists():
        plot_profiles(output_dir, plt)
    print("figures written to", output_dir)


# ------------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=("precompute", "locate", "profiles", "reference", "band-reference", "render", "verdict", "plot"), required=True)
    parser.add_argument("--families", nargs="+", default=list(FAMILIES))
    parser.add_argument("--profile", nargs="+", default=[], help="family:elevation:azimuth:row|column:length[:rationale]")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tiers", type=int, nargs="+", default=[10_000_000])
    parser.add_argument("--windowed", action="store_true", help="precompute only the profiles' deviation window")
    parser.add_argument("--sampling", choices=tuple(SAMPLERS), default="fibonacci", help="point set of the store (precompute/render)")
    parser.add_argument("--skip-self-checks", action="store_true")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--locate-n", type=int, default=10_000_000)
    parser.add_argument(
        "--zoom", nargs=4, type=float, metavar=("ELEVATION", "AZIMUTH", "FOV_DEG", "SIZE"),
        help="locate on a zoomed window instead of the wide one (files tagged _zoom<el>_<az>)",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == "precompute":
        stage_precompute(
            args.output_dir, args.tiers, windowed=args.windowed, workers=args.workers, skip_checks=args.skip_self_checks,
            sampling=args.sampling,
        )
    elif args.stage == "locate":
        if args.zoom is None:
            stage_locate(args.output_dir, args.locate_n, args.workers, LOCATE_RENDER)
        else:
            elevation, azimuth, fov, size = args.zoom
            zoom = {"width": int(size), "height": int(size), "fov_deg": fov, "view": {"azimuth": azimuth, "elevation": elevation}}
            stage_locate(args.output_dir, args.locate_n, args.workers, zoom, tag=f"_zoom{elevation:g}_{azimuth:g}")
    elif args.stage == "profiles":
        stage_profiles(args.output_dir, args.profile)
    elif args.stage == "reference":
        stage_reference(args.output_dir, args.families, args.workers)
    elif args.stage == "band-reference":
        stage_band_reference(args.output_dir, args.families, args.workers)
    elif args.stage == "render":
        if args.sampling == "fibonacci":
            stage_render(args.output_dir, args.families, args.workers)
        else:
            stage_render(args.output_dir, args.families, args.workers, store=f"members_{args.sampling}", tag=f"_{args.sampling}")
    elif args.stage == "verdict":
        stage_verdict(args.output_dir, args.families)
    elif args.stage == "plot":
        stage_plot(args.output_dir)


if __name__ == "__main__":
    main()
