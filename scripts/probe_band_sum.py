"""Band-sum quadrature probe (roadmap section 4.2): precomputed S^2 events summed per deviation band.

The band sum is the second discretisation of the Phase II level-set integral
(roadmap section 4.1(b)).  With ``u = R^-1 s`` and the Haar split
``d mu_Haar = dA(u) / 4 pi * d psi / 2 pi``, the pixel value of the Phase I
strip (contract section 7, ``I = (1 / 8 pi^2) int rho A T / J_perp dH^1``)
satisfies

    I(delta, alpha) sin(delta) = (1 / 8 pi^2) int_{D_P = delta} rho A T / |grad D_P| dl,

and integrating over a deviation band ``[delta_lo, delta_hi]`` at the pixel's
azimuth ``alpha`` turns the line integral into an area integral on ``S^2``.
With ``N`` equal-area points (``4 pi / N`` each):

    I_hat = sum_{D_i in band} w_i rho(R_i) / (2 pi N (delta_hi - delta_lo) sin(delta))

where ``w_i = A_P(u_i) T_P(u_i)`` and ``R_i`` is the unique pose with
``R_i u_i = s`` and ``R_i Phi_P(u_i)`` at deviation ``D_i`` and azimuth
``alpha`` (Gislen eq. 19 at ``omega = D_i``; built here from two orthonormal
frames).  The derived constant is used as is; nothing is fitted.

Stages (``--stage``):

- ``precompute``: the event store of :mod:`lumice_integral.s2_store`
  (Fibonacci points on ``S^2``, one rotation per point, the production batch
  evaluators, ``w > 0`` events sorted by ``D``; psi-invariance, gate-coverage
  and Haar-mean self-checks), written in this probe's flat layout: one
  ``events_N<n>.npz`` + ``precompute_N<n>.json`` per ``N``; an existing file
  is never overwritten.
- ``reference``: the random-orientation Phase I reference of scene 2
  (``strip_pixel.render_pixel`` on ``canonical_strip_scene(pose_density=random)``),
  cached in ``reference_random_c<column>.npz``.
- ``render``: per pixel of each scene and each available ``N``: band count
  ``K``, ``K_rho_pos`` (``rho > 0``), ``K_eff`` (Kish effective sample size of
  ``w rho``), the estimate, and the comparison with the reference.  Writes
  ``metrics_<scene>_N<n>.csv`` and ``summary.json`` (includes the power-law
  extrapolation to a lit-band error of ``1e-2``).
- ``plot``: log-scale figures from ``summary.json`` and the CSVs only
  (needs ``uv run --with matplotlib``).

Nothing here imports or calls Lumice.  The store itself is
``lumice_integral.s2_store`` (task ``s2-event-store``; this script built it
before, bit-identically).  Usage::

    uv run python scripts/probe_band_sum.py --stage precompute --precompute-n 1000000 10000000 --output-dir <dir>
    uv run python scripts/probe_band_sum.py --stage reference --output-dir <dir>
    uv run python scripts/probe_band_sum.py --stage render --output-dir <dir>
    uv run --with matplotlib python scripts/probe_band_sum.py --stage plot --output-dir <dir>
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as dt
import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from lumice_integral.band_sum import (  # the estimator, migrated to src/ (task band-sum-renderer)
    band_rotations,
    band_sum_pixel,
    pixel_band,
)
from lumice_integral.canonical_scene import (
    CANONICAL_REFRACTIVE_INDEX,
    CANONICAL_RENDER,
    canonical_crystal,
    canonical_incident_direction,
    canonical_pose_density,
)
from lumice_integral.optics import PATH_3_5_FACES
from lumice_integral.pose_density import build_pose_density
from lumice_integral.s2_store import (
    CHUNK,
    FIBONACCI_SAMPLING,
    ROTATION_PER_POINT,
    PointSampler,
    build_event_store,
    max_rss_mb,
)

DEFAULT_REFERENCE_DIR = Path("/Users/zhangjiajie/Codes/Lumice Integral/artifacts/strip-full")
DEFAULT_COLUMN = 126
RANDOM_ROW_STEP = 13  # 801 rows -> 62 pixels (plan default assumption 3)
SWEEP_COLUMNS = (26, 76, 176, 226)  # generality check of the narrow-rho finding (column 126 is the sun vertical)
SWEEP_ROW_STEP = 8
LIT_FRACTION = 1e-2  # lit band: reference > LIT_FRACTION * column maximum
TARGET_ERROR = 1e-2


# ------------------------------------------------------------- precompute
def precompute(
    n: int,
    output_dir: Path,
    *,
    run_checks: bool,
    faces: Sequence[int] = PATH_3_5_FACES,
    deviation_window: tuple[float, float] | None = None,
    sampler: PointSampler | None = None,
    sampling: str = FIBONACCI_SAMPLING,
) -> dict[str, Any]:
    """Event store of path ``faces`` (:func:`lumice_integral.s2_store.build_event_store`) in the probe's flat layout.

    ``events_N<n>.npz`` + ``precompute_N<n>.json`` in ``output_dir``, never
    overwritten.  This layout (one flat directory per store family, tiers by
    ``N``) predates the library's parameter-hashed cache directories and is
    kept on purpose so the task 13/14 artifacts stay readable; it is not a
    second store implementation.
    """
    events_path = output_dir / f"events_N{n}.npz"
    meta_path = output_dir / f"precompute_N{n}.json"
    for path in (events_path, meta_path):
        if path.exists():
            raise FileExistsError(f"{path} exists; refusing to overwrite a precomputed tier")
    s = canonical_incident_direction()
    store = build_event_store(
        canonical_crystal(),
        CANONICAL_REFRACTIVE_INDEX,
        [faces],
        n,
        incident_direction=s,
        sampler=sampler,
        sampling=sampling,
        deviation_window=deviation_window,
        run_checks=run_checks,
        log=print,
    )
    np.savez(events_path, **store.events.arrays())
    diagnostics = store.diagnostics
    meta = {
        "N": n,
        "sampling": sampling,
        "rotation_per_point": ROTATION_PER_POINT,
        "path": list(faces),
        "deviation_window_rad": None if deviation_window is None else [float(v) for v in deviation_window],
        "crystal": {"type": "hexagonal_column", "height_ratio": 2.0},
        "refractive_index": CANONICAL_REFRACTIVE_INDEX,
        "incident_direction": s.tolist(),
        "kept_events": diagnostics["kept_events"],
        "valid_domain_points": diagnostics["valid_domain_points"],
        "kept_fraction": diagnostics["kept_fraction"],
        "fibonacci_mean_w": diagnostics["fibonacci_mean_w"],
        "D_range_deg": diagnostics["D_range_deg"],
        "dtype": "float64",
        "wall_clock_s": diagnostics["wall_clock_s"],
        "chunk": CHUNK,
        "max_rss_mb_process": max_rss_mb(),
        "self_checks": diagnostics["self_checks"],
        "created": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"N={n}: kept {meta['kept_events']} ({meta['kept_fraction']:.4f}), {meta['wall_clock_s']:.1f} s, rss {meta['max_rss_mb_process']:.0f} MB")
    return meta


def load_events(output_dir: Path, n: int) -> dict[str, np.ndarray]:
    with np.load(output_dir / f"events_N{n}.npz") as data:
        return {key: data[key] for key in data.files}


# ----------------------------------------------------------------- render
@dataclasses.dataclass(frozen=True)
class PixelMetrics:
    """One row of ``metrics_<scene>_N<n>.csv`` (the schema shared by render, summary and plot)."""

    row: int
    column: int
    delta_deg: float
    band_width_rad: float
    K: int
    K_rho_pos: int
    K_eff: float
    estimate: float
    reference: float
    lit: bool
    rel_error: float


def gislen_eq19(a: np.ndarray, b: np.ndarray, a0: np.ndarray, b0: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """Gislen eq. 19 (inverse-rendering note), batched: ``(a0, b0) -> (a, b)`` at scattering angle ``omega``."""
    c = np.cos(omega)[:, None, None]
    outer = lambda x, y: np.einsum("ni,nj->nij", x, y)  # noqa: E731
    total = outer(a, a0) + outer(b, b0) - c * (outer(a, b0) + outer(b, a0)) + outer(np.cross(a, b), np.cross(a0, b0))
    return total / (np.sin(omega) ** 2)[:, None, None]


def self_check_rotation(events: dict[str, np.ndarray], s: np.ndarray, rows: list[int], column: int) -> dict[str, Any]:
    """The frame construction equals eq. 19 at ``omega = D_i``; eq. 19 at the pixel's ``delta`` is not a rotation."""
    worst_frame_vs_eq19, orthogonality_frame, orthogonality_pixel_omega, band_widths = 0.0, 0.0, [], []
    for row in rows:
        centre, delta, lo_d, hi_d = pixel_band(row, column, s)
        lo, hi = np.searchsorted(events["D"], [lo_d, hi_d])
        if hi <= lo:
            continue
        rotations = band_rotations(events, lo, hi, s, centre)
        deviation = events["D"][lo:hi]
        e = centre - (centre @ s) * s
        e /= np.linalg.norm(e)
        b_i = np.cos(deviation)[:, None] * s + np.sin(deviation)[:, None] * e
        s_rows = np.broadcast_to(s, b_i.shape)
        eq19 = gislen_eq19(s_rows, b_i, events["u"][lo:hi], events["phi"][lo:hi], deviation)
        worst_frame_vs_eq19 = max(worst_frame_vs_eq19, float(np.max(np.abs(eq19 - rotations))))
        eye = np.eye(3)
        orthogonality_frame = max(
            orthogonality_frame,
            float(np.max(np.abs(np.einsum("nji,njk->nik", rotations, rotations) - eye))),
        )
        pixel = gislen_eq19(
            s_rows, np.broadcast_to(centre, b_i.shape), events["u"][lo:hi], events["phi"][lo:hi], np.full(hi - lo, delta)
        )
        orthogonality_pixel_omega.append(
            np.linalg.norm(np.einsum("nji,njk->nik", pixel, pixel) - eye, axis=(1, 2))
        )
        band_widths.append(hi_d - lo_d)
    residual = np.concatenate(orthogonality_pixel_omega)
    return {
        "max_abs_frame_minus_eq19_at_D_i": worst_frame_vs_eq19,
        "max_abs_RtR_minus_I_frame": orthogonality_frame,
        "eq19_at_pixel_delta_RtR_minus_I_fro": {
            "median": float(np.median(residual)),
            "max": float(np.max(residual)),
            "band_width_rad_median": float(np.median(band_widths)),
        },
        "passed": bool(worst_frame_vs_eq19 < 1e-10 and orthogonality_frame < 1e-12),
    }


def scene_pixels(scene: str, column: int) -> list[tuple[int, int]]:
    height = int(CANONICAL_RENDER["height"])
    if scene == "canonical":
        return [(row, column) for row in range(height)]
    if scene == "random":
        return [(row, column) for row in range(0, height, RANDOM_ROW_STEP)]
    if scene == "canonical-sweep":
        return [(row, c) for c in SWEEP_COLUMNS for row in range(0, height, SWEEP_ROW_STEP)]
    raise ValueError(scene)


def scene_density(scene: str):
    return build_pose_density("random") if scene == "random" else canonical_pose_density()


def compute_reference_random(output_dir: Path, column: int) -> dict[str, Any]:
    """Phase I production pixels of scene 2 (random orientation), cached."""
    from lumice_integral.strip_pixel import PixelOptions, canonical_strip_scene, render_pixel

    path = output_dir / f"reference_random_c{column}.npz"
    if path.exists():
        raise FileExistsError(f"{path} exists; refusing to overwrite the Phase I reference")
    start = time.perf_counter()
    scene = canonical_strip_scene(pose_density=build_pose_density("random"))
    scene_s = time.perf_counter() - start
    options = PixelOptions()
    rows, values, errors, complete, seconds = [], [], [], [], []
    for row, col in scene_pixels("random", column):
        t0 = time.perf_counter()
        result = render_pixel(scene, row, col, options)
        seconds.append(time.perf_counter() - t0)
        rows.append(row)
        values.append(result.value)
        errors.append(result.error_estimate)
        complete.append(result.completeness == "complete")
    np.savez(
        path,
        rows=np.array(rows),
        values=np.array(values),
        error_estimates=np.array(errors),
        complete=np.array(complete),
        seconds=np.array(seconds),
    )
    meta = {
        "pixels": len(rows),
        "scene_build_s": scene_s,
        "pixel_s_total": float(np.sum(seconds)),
        "pixel_s_mean": float(np.mean(seconds)),
        "incomplete_pixels": int(len(rows) - np.count_nonzero(complete)),
        "max_rss_mb_process": max_rss_mb(),
    }
    (output_dir / f"reference_random_c{column}.json").write_text(json.dumps(meta, indent=2) + "\n")
    print("reference:", json.dumps(meta))
    return meta


def reference_values(scene: str, output_dir: Path, reference_dir: Path, column: int) -> dict[tuple[int, int], float]:
    if scene == "random":
        with np.load(output_dir / f"reference_random_c{column}.npz") as data:
            if not np.all(data["complete"]):
                print(f"warning: {np.count_nonzero(~data['complete'])} random reference pixels are not 'complete'")
            return {(int(r), column): float(v) for r, v in zip(data["rows"], data["values"])}
    from lumice_integral.strip_io import read_strip  # verifies every array's recorded SHA-256

    arrays, _ = read_strip(reference_dir)
    return {(row, col): float(arrays.values[row, col]) for row, col in scene_pixels(scene, column)}


def available_tiers(output_dir: Path) -> list[int]:
    return sorted(int(p.stem.removeprefix("events_N")) for p in output_dir.glob("events_N*.npz"))


def power_law(ns: list[int], values: list[float]) -> dict[str, float] | None:
    """Least-squares ``value = C N^slope`` in log-log (``None`` with fewer than two positive points)."""
    points = [(n, v) for n, v in zip(ns, values) if v > 0 and np.isfinite(v)]
    if len(points) < 2:
        return None
    x = np.log10([p[0] for p in points])
    y = np.log10([p[1] for p in points])
    slope, intercept = np.polyfit(x, y, 1)
    return {"slope": float(slope), "log10_C": float(intercept)}


def render(output_dir: Path, reference_dir: Path, column: int, scenes: list[str]) -> dict[str, Any]:
    s = canonical_incident_direction()
    tiers = available_tiers(output_dir)
    if not tiers:
        raise FileNotFoundError(f"no events_N*.npz in {output_dir}; run --stage precompute first")
    summary: dict[str, Any] = {"tiers": tiers, "column": column, "lit_fraction": LIT_FRACTION, "scenes": {}}
    for scene in scenes:
        reference = reference_values(scene, output_dir, reference_dir, column)
        pixels = scene_pixels(scene, column)
        column_max = {c: max(v for (r, cc), v in reference.items() if cc == c) for c in {c for _, c in pixels}}
        density = scene_density(scene)
        per_tier: dict[str, Any] = {}
        for n in tiers:
            events = load_events(output_dir, n)
            start = time.perf_counter()
            rows: list[PixelMetrics] = []
            for row, col in pixels:
                estimate, delta, k, k_pos, k_eff, width = band_sum_pixel(events, s, density, row, col, n)
                ref = reference[(row, col)]
                lit = ref > LIT_FRACTION * column_max[col]
                rel = (estimate - ref) / ref if lit else float("nan")
                rows.append(PixelMetrics(row, col, float(np.degrees(delta)), width, k, k_pos, k_eff, estimate, ref, bool(lit), rel))
            wall = time.perf_counter() - start
            with (output_dir / f"metrics_{scene}_N{n}.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=[f.name for f in dataclasses.fields(PixelMetrics)])
                writer.writeheader()
                writer.writerows(dataclasses.asdict(r) for r in rows)
            lit_rows = [r for r in rows if r.lit]
            rel = np.array([r.rel_error for r in lit_rows])
            positive = [r for r in lit_rows if r.estimate > 0]
            log_ratio = np.log([r.estimate / r.reference for r in positive]) if positive else np.array([np.nan])
            ratio = np.array([r.estimate / r.reference for r in lit_rows])
            per_tier[str(n)] = {
                "pixels": len(rows),
                "lit_pixels": len(lit_rows),
                "lit_zero_estimate": len(lit_rows) - len(positive),
                "rms_rel_error": float(np.sqrt(np.mean(rel**2))),
                "median_abs_rel_error": float(np.median(np.abs(rel))),
                "log_rms": float(np.sqrt(np.mean(log_ratio**2))),
                "median_ratio": float(np.median(ratio)),
                "sum_ratio": float(sum(r.estimate for r in lit_rows) / sum(r.reference for r in lit_rows)),
                "K_median_lit": float(np.median([r.K for r in lit_rows])),
                "K_eff_median_lit": float(np.median([r.K_eff for r in lit_rows])),
                "K_eff_min_lit": float(np.min([r.K_eff for r in lit_rows])),
                "K_eff_over_K_median_lit": float(np.median([r.K_eff / r.K for r in lit_rows if r.K > 0])),
                "K_eff_per_N_median_lit": float(np.median([r.K_eff for r in lit_rows]) / n),
                "render_wall_s": wall,
                "render_s_per_pixel": wall / len(rows),
                "max_rss_mb_process": max_rss_mb(),
            }
            print(scene, n, json.dumps(per_tier[str(n)]))
        ns = [int(n) for n in per_tier]
        fit = power_law(ns, [per_tier[str(n)]["rms_rel_error"] for n in ns])
        extrapolation: dict[str, Any] = {"rms_rel_error_fit": fit}
        if fit is not None and fit["slope"] < 0:
            extrapolation["N_for_target"] = float(10 ** ((np.log10(TARGET_ERROR) - fit["log10_C"]) / fit["slope"]))
        keff = power_law(ns, [per_tier[str(n)]["K_eff_median_lit"] for n in ns])
        extrapolation["K_eff_median_fit"] = keff
        summary["scenes"][scene] = {"tiers": per_tier, "extrapolation": extrapolation}
        print(scene, "extrapolation", json.dumps(extrapolation))
    largest = load_events(output_dir, tiers[-1])
    summary["rotation_self_check"] = self_check_rotation(largest, s, list(range(20, 801, 60)), column)
    print("rotation self-check:", json.dumps(summary["rotation_self_check"]))
    summary["created"] = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


# ------------------------------------------------------------------- plot
def read_metrics(path: Path) -> dict[str, np.ndarray]:
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    return {key: np.array([float(r[key] == "True") if key == "lit" else float(r[key]) for r in rows]) for key in rows[0]}


def plot(output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = json.loads((output_dir / "summary.json").read_text())
    tiers = summary["tiers"]
    for scene in summary["scenes"]:
        figure, axes = plt.subplots(3, 1, figsize=(9, 11), sharex=True)
        for n in tiers:
            m = read_metrics(output_dir / f"metrics_{scene}_N{n}.csv")
            order = np.argsort(m["delta_deg"])
            x = m["delta_deg"][order]
            axes[0].semilogy(x, np.where(m["estimate"][order] > 0, m["estimate"][order], np.nan), ".", ms=2, label=f"band sum N={n:.0e}")
            lit = m["lit"][order] > 0
            axes[1].semilogy(x[lit], np.abs(m["rel_error"][order][lit]), ".", ms=2, label=f"N={n:.0e}")
            axes[2].semilogy(x, np.maximum(m["K_eff"][order], 1e-1), ".", ms=2, label=f"K_eff N={n:.0e}")
        axes[0].semilogy(x, np.where(m["reference"][order] > 0, m["reference"][order], np.nan), "k-", lw=0.8, label="reference")
        axes[0].set_ylabel("I (pixel value)")
        axes[1].axhline(TARGET_ERROR, color="k", ls="--", lw=0.8)
        axes[1].set_ylabel("|relative error| (lit band)")
        axes[2].set_ylabel("K_eff (Kish)")
        axes[2].set_xlabel("deviation delta (deg)")
        for axis in axes:
            axis.legend(fontsize=7)
            axis.grid(True, which="both", alpha=0.3)
        figure.suptitle(f"band sum vs reference, scene {scene}")
        figure.tight_layout()
        figure.savefig(output_dir / f"profile_{scene}.png", dpi=130)
        plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for scene, block in summary["scenes"].items():
        ns = [int(n) for n in block["tiers"]]
        axes[0].loglog(ns, [block["tiers"][str(n)]["rms_rel_error"] for n in ns], "o-", label=f"{scene} RMS rel")
        axes[0].loglog(ns, [block["tiers"][str(n)]["median_abs_rel_error"] for n in ns], "s--", label=f"{scene} median |rel|")
        axes[1].loglog(ns, [block["tiers"][str(n)]["K_eff_median_lit"] for n in ns], "o-", label=f"{scene} median K_eff")
    axes[0].axhline(TARGET_ERROR, color="k", ls="--", lw=0.8)
    for axis, label in zip(axes, ("lit-band relative error", "median K_eff (lit)")):
        axis.set_xlabel("N (Fibonacci points on S^2)")
        axis.set_ylabel(label)
        axis.legend(fontsize=7)
        axis.grid(True, which="both", alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_dir / "convergence.png", dpi=130)
    plt.close(figure)
    print("figures written to", output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=("precompute", "reference", "render", "plot"), required=True)
    parser.add_argument("--precompute-n", type=int, nargs="+", default=[1_000_000, 10_000_000])
    parser.add_argument("--skip-self-checks", action="store_true", help="precompute without the section 4.1(a) checks")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR)
    parser.add_argument("--column", type=int, default=DEFAULT_COLUMN)
    parser.add_argument("--scenes", nargs="+", default=["canonical", "random", "canonical-sweep"])
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == "precompute":
        for index, n in enumerate(args.precompute_n):
            precompute(n, args.output_dir, run_checks=not args.skip_self_checks and index == 0)
    elif args.stage == "reference":
        compute_reference_random(args.output_dir, args.column)
    elif args.stage == "render":
        if not args.reference_dir.is_dir():
            raise FileNotFoundError(f"reference directory {args.reference_dir} does not exist")
        render(args.output_dir, args.reference_dir, args.column, args.scenes)
    else:
        plot(args.output_dir)


if __name__ == "__main__":
    main()
