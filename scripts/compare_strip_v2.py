"""Compare a rendered ch06 strip with the historical raw and the Lumice remake.

The comparison is designed around the v1 lesson that a rank correlation of
``0.99`` hid a whole-band ``x2`` and that linear-scale panels hid a ``~3x``
too-steep fall-off: every metric here is sensitive to *multiplicative* bias
and every panel is on a log scale.

1. Row-band ratio: for each 50-row band, ``median(ours / historical)`` over
   pixels that are rendered, ``complete`` and positive in both, on the
   centre columns ``101-151`` (the two images differ in width, so an
   all-column median mixes the wings into the vertical decay; the
   all-column curve is kept as a secondary series); the curve is reported
   raw and relative to its whole-image median, so a uniform scale factor
   (the two arrays have unrelated units) sits at ``1`` and any band that
   decays differently stands out.
2. Log-domain profiles: the vertical profile of column ``126`` and the
   horizontal profiles of rows ``150 / 300 / 450``; every profile is
   max-normalised, taken to ``log10`` and the RMS of the difference against
   the historical profile is reported, over the lit band (both above
   ``1e-3``) and over every positive point.
3. Inner-edge row offset: the first row of the column-``126`` profile whose
   normalised value exceeds a threshold, for ours and the historical raw,
   reported at several thresholds because the choice is a calibration.
4. Spearman rank correlation, kept as an auxiliary number only.
5. Optional three-way radiometric check against a Lumice *float* export
   (``--lumice-float``, the ``img_01.npy`` of ``Lumice render --format npy``):
   the Lumice PNG is tone-mapped 8-bit and can only arbitrate display space,
   the float Y accumulator is proportional to the energy that landed in each
   pixel, so the same log-profile RMS is reported for Lumice-float vs
   historical, ours vs Lumice-float and ours vs historical on the profile
   column and its ``+-20`` neighbours and on rows ``150 / 300 / 450 / 600``,
   together with the inner-edge rows and the per-band normalised ratio along
   the column.  A second independent run (``--lumice-float-run2``) is summed
   into the arbitrating profile and the run-to-run difference is reported as
   the Monte Carlo noise floor of those numbers.

Inputs are read only; the historical raw and the Lumice PNG live in the
Writing-Lab project (``--writing-lab-dir``).  The display mapping is the
one documented in ``docs/ch06-reference-fixture.md`` section 3.2 and is
implemented here (there is no other implementation in this repository).
Plotting needs matplotlib, which is not a project dependency::

    uv run --with matplotlib python scripts/compare_strip_v2.py \\
        --strip-dir artifacts/strip-full --output-dir <task>/artifacts

The diagnostic positions (column ``126``, rows ``150 / 300 / 450``) are shared
with ``scripts/probe_defect2_factors.py``; keep :data:`PROBE_COLUMN` /
:data:`PROBE_ROWS` in sync with it.  The float comparison adds row ``600``
and the ``+-20`` columns locally (:data:`LUMICE_FLOAT_EXTRA_ROWS`,
:data:`LUMICE_FLOAT_COLUMN_OFFSETS`) so that the shared constants stay put.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from lumice_integral.strip_io import STATUS_BITS, read_strip

DEFAULT_WRITING_LAB_DIR = Path(
    "/Users/zhangjiajie/Codes/Writing-Lab/现代冰晕研究漫谈/06-monte-carlo-vs-integration/data"
)
HISTORICAL_RAW = "data_251x801.bin"
LUMICE_REMAKE = "compare_two_methods/band1e9/img_01.png"
HEIGHT, WIDTH = 801, 251
PROBE_COLUMN = 126
PROBE_ROWS = (150, 300, 450)
BAND_ROWS = 50
CENTRE_COLUMNS = (101, 152)  # half-open; +-25 columns around the profile column
PROFILE_LIT_FLOOR = 1e-3  # normalised level above which a profile point counts as inside the lit band
EDGE_THRESHOLDS = (1e-3, 1e-2, 1e-1)
PROFILE_FLOOR = 1e-4
# --lumice-float only: the tail row and the +-20 columns of task lumice-raw-profile-oracle;
# kept local so PROBE_ROWS / PROBE_COLUMN stay shared with probe_defect2_factors.py unchanged
LUMICE_FLOAT_EXTRA_ROWS = (600,)
LUMICE_FLOAT_COLUMN_OFFSETS = (-20, 0, 20)
# rays admitted per emitted ray relative to Lumice Integral's single 3-5 path: a raypath filter with
# symmetry P admits the 6 prism-rotation images, PBD the 12 P/B/D-equivalent raypaths, all on the
# same emitted budget (doc/raypath-symmetry.zh.md of the Ice Halo repository); a denominator
# bookkeeping, never a per-pixel factor, and only recorded here for the reader to fold by hand
LUMICE_SYMMETRY_FOLD = {"P": 6, "PBD": 12}


# ---------- ch06 display mapping (docs/ch06-reference-fixture.md section 3.2) ----------
def white_of(x: np.ndarray) -> float:
    """``white_lim``: the 99th percentile of the finite positive values."""
    v = x[np.isfinite(x) & (x > 0)]
    return float(np.percentile(v, 99)) if v.size else 1.0


def display(x: np.ndarray, white: float) -> np.ndarray:
    """Log tone mapping with ``black = 2^-7 white``, normalised to ``[0, 1]`` (no gamma)."""
    black = 2.0**-7 * white
    y = (np.nan_to_num(x, nan=0.0) + black) * white / (np.nan_to_num(x, nan=0.0) + white)
    y = np.log10(y)
    y = (y - np.log10(black)) / (np.log10(white) - np.log10(black))
    return np.clip(y, 0.0, 1.0)


# ---------- inputs ----------
def load_historical(path: Path) -> np.ndarray:
    return np.fromfile(path, dtype="<f4").reshape(HEIGHT, WIDTH).astype(np.float64)


def load_lumice(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    from PIL import Image

    image = Image.open(path).convert("L")
    note = {"path": str(path), "stored_size": list(image.size), "resized": False, "resample": None}
    if image.size != (WIDTH, HEIGHT):
        image = image.resize((WIDTH, HEIGHT), resample=Image.Resampling.BILINEAR)
        note.update(resized=True, resample="bilinear")
    return np.asarray(image, dtype=np.float64), note


def load_lumice_float(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Y channel of a Lumice ``--format npy`` export plus its sidecar, as ``(H, W)`` float64.

    Unlike :func:`load_lumice` (a PNG, no sidecar) this loader *requires* the
    ``img_0N.json`` sidecar next to the ``.npy``: the exposure scalars in it
    (``emitted_energy`` above all) are what make two runs summable and the
    symmetry folding checkable, so a bare array is refused rather than
    silently compared.  ``symmetry`` is not in the sidecar; it is read from a
    ``config.json`` in the same directory when one is there.
    """
    sidecar = path.with_suffix(".json")
    if not sidecar.is_file():
        raise SystemExit(
            f"{sidecar} missing: load_lumice_float requires the img_0N.json sidecar that "
            "`Lumice render --format npy` writes next to the .npy (emitted_energy, sim_ray_num, seed); "
            "this is stricter than load_lumice, which reads a PNG with no sidecar"
        )
    arr = np.load(path)
    if arr.shape != (HEIGHT, WIDTH, 3):
        raise SystemExit(f"{path}: shape {arr.shape} != {(HEIGHT, WIDTH, 3)} (H, W, XYZ)")
    meta = json.loads(sidecar.read_text())
    note: dict[str, Any] = {
        "path": str(path),
        "channel": "Y",
        "dtype": str(arr.dtype),
        **{k: meta.get(k) for k in ("normalization", "emitted_energy", "sim_ray_num", "seed", "axis_solid_angle", "intensity_factor", "lumice_api_version")},
        "symmetry": None,
        "symmetry_fold_to_single_3_5": None,
    }
    config = path.with_name("config.json")
    if config.is_file():
        filters = json.loads(config.read_text()).get("filter", [])
        symmetry = filters[0].get("symmetry") if filters else None
        note.update(symmetry=symmetry, symmetry_fold_to_single_3_5=LUMICE_SYMMETRY_FOLD.get(symmetry))
    return np.asarray(arr[:, :, 1], dtype=np.float64), note


def merge_lumice_float(runs: list[tuple[np.ndarray, dict[str, Any]]]) -> tuple[np.ndarray, dict[str, Any]]:
    """Sum independent runs into one accumulator; ``emitted_energy`` / ``sim_ray_num`` add up alongside.

    Noise bookkeeping for the report: with ``r = sigma / mu`` the relative
    noise of one pixel in a single run, the difference of two i.i.d. runs
    ``(X1 - X2) / mu`` has standard deviation ``sqrt(2) r`` and the summed
    profile ``X1 + X2`` has relative noise ``r / sqrt(2)``; so the *merged*
    relative noise is half the measured relative run-to-run std, and the
    single-run noise is that std over ``sqrt(2)``.  The three are reported
    under their own keys and must not be mixed up.
    """
    total = np.sum([y for y, _ in runs], axis=0)
    note: dict[str, Any] = {
        "runs": [n for _, n in runs],
        "emitted_energy": float(sum(n["emitted_energy"] for _, n in runs)),
        "sim_ray_num": int(sum(n["sim_ray_num"] for _, n in runs)),
        "symmetry": runs[0][1]["symmetry"],
        "symmetry_fold_to_single_3_5": runs[0][1]["symmetry_fold_to_single_3_5"],
    }
    if len({n["symmetry"] for _, n in runs}) > 1 or len({n["normalization"] for _, n in runs}) > 1:
        raise SystemExit(f"--lumice-float runs disagree on symmetry / normalization: {note['runs']}")
    return total, note


def lumice_float_noise(runs: list[np.ndarray], positions: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict[str, Any]:
    """Run-to-run relative difference on the lit band of each profile (see :func:`merge_lumice_float`)."""
    if len(runs) < 2:
        return {"available": False, "reason": "single run, no independent repeat to difference"}
    out: dict[str, Any] = {"available": True, "runs_differenced": 2}
    for key, (pa, pb) in positions.items():
        na, nb = norm(pa), norm(pb)
        lit = (na >= PROFILE_LIT_FLOOR) & (nb >= PROFILE_LIT_FLOOR)
        if not lit.any():
            out[key] = {"count": 0}
            continue
        mu = 0.5 * (pa[lit] + pb[lit])
        rel = (pa[lit] - pb[lit]) / mu
        std = float(np.std(rel))
        out[key] = {
            "count": int(lit.sum()),
            "relative_difference_std": std,
            "single_run_relative_noise": std / np.sqrt(2.0),
            "merged_relative_noise": std / 2.0,
            "log_rms_run1_vs_run2": log_profile_rms(pa, pb)["lit_band"]["rms_log10"],
        }
    return out


def load_baseline(path: Path | None) -> np.ndarray | None:
    """v1 58-column baseline (``strip58_v1_results.npz``, NaN where not rendered)."""
    if path is None:
        return None
    payload = np.load(path)
    values = np.asarray(payload["value"], dtype=np.float64)
    if "unknown" in payload:  # unknown-completeness pixels carry a forced 0, not a value
        values = np.where(np.asarray(payload["unknown"], dtype=bool), np.nan, values)
    return values


# ---------- metrics ----------
def norm(v: np.ndarray) -> np.ndarray:
    v = np.nan_to_num(np.asarray(v, dtype=np.float64), nan=0.0)
    m = v.max()
    return v / m if m > 0 else v


def band_ratios(ours: np.ndarray, hist: np.ndarray, usable: np.ndarray) -> list[dict[str, Any]]:
    both = usable & np.isfinite(ours) & (ours > 0) & (hist > 0)
    global_median = float(np.median(ours[both] / hist[both])) if both.any() else float("nan")
    rows_out = []
    for start in range(0, HEIGHT, BAND_ROWS):
        stop = min(start + BAND_ROWS, HEIGHT)
        mask = both[start:stop]
        ratio = ours[start:stop][mask] / hist[start:stop][mask]
        entry: dict[str, Any] = {"rows": [start, stop], "count": int(mask.sum())}
        if ratio.size:
            median = float(np.median(ratio))
            entry.update(
                median=median,
                median_relative=median / global_median,
                p25_relative=float(np.percentile(ratio, 25)) / global_median,
                p75_relative=float(np.percentile(ratio, 75)) / global_median,
            )
        rows_out.append(entry)
    return [{"global_median": global_median, "count": int(both.sum())}, *rows_out]


def log_profile_rms(a: np.ndarray, b: np.ndarray, valid: np.ndarray | None = None) -> dict[str, Any]:
    """RMS of ``log10(norm(a)) - log10(norm(b))``: over every positive point, and over the lit band only.

    The far wings of the historical raw hold values down to ``1e-8`` where the
    strip is ``1e-12``; their log difference says nothing about the band, so the
    ``lit_band`` numbers (both profiles above :data:`PROFILE_LIT_FLOOR`) are the
    ones to read, the ``all`` numbers show how much the wings add.
    """
    na, nb = norm(a), norm(b)
    mask = np.isfinite(na) & np.isfinite(nb) & (na > 0) & (nb > 0)
    if valid is not None:
        mask &= valid
    out: dict[str, Any] = {}
    for key, m in (("all", mask), ("lit_band", mask & (na >= PROFILE_LIT_FLOOR) & (nb >= PROFILE_LIT_FLOOR))):
        if not m.any():
            out[key] = {"rms_log10": float("nan"), "count": 0}
            continue
        diff = np.log10(na[m]) - np.log10(nb[m])
        out[key] = {
            "rms_log10": float(np.sqrt(np.mean(diff**2))),
            "mean_log10": float(np.mean(diff)),
            "count": int(m.sum()),
            "spearman": float(spearmanr(na[m], nb[m]).correlation),
        }
    return out


def first_index_above(profile: np.ndarray, threshold: float) -> int | None:
    hits = np.where(norm(profile) >= threshold)[0]
    return int(hits[0]) if hits.size else None


def edge_offsets(hist_col: np.ndarray, ours_col: np.ndarray, lum_col: np.ndarray) -> list[dict[str, Any]]:
    out = []
    for threshold in EDGE_THRESHOLDS:
        h, o, l = (first_index_above(v, threshold) for v in (hist_col, ours_col, lum_col))
        out.append(
            {
                "threshold": threshold,
                "historical_row": h,
                "ours_row": o,
                "lumice_row": l,
                "ours_minus_historical": None if h is None or o is None else o - h,
                "lumice_minus_historical": None if h is None or l is None else l - h,
            }
        )
    return out


def spearman_block(ours: np.ndarray, hist: np.ndarray, lum: np.ndarray, rendered: np.ndarray, complete: np.ndarray) -> dict[str, Any]:
    o = np.nan_to_num(ours, nan=0.0)
    lit_both = rendered & complete & (o > 0) & (hist > 0)
    out = {
        "lit_in_both": {"count": int(lit_both.sum()), "rho": float(spearmanr(o[lit_both], hist[lit_both]).correlation)},
        "complete": {"count": int(complete.sum()), "rho": float(spearmanr(o[complete], hist[complete]).correlation)},
        "rendered": {"count": int(rendered.sum()), "rho": float(spearmanr(o[rendered], hist[rendered]).correlation)},
        "rendered_vs_lumice": {"count": int(rendered.sum()), "rho": float(spearmanr(o[rendered], lum[rendered]).correlation)},
    }
    return out


def decay_ratio_along_column(ours: np.ndarray, hist: np.ndarray, usable: np.ndarray, column: int) -> list[dict[str, Any]]:
    """``(ours / hist)`` along one column, each profile max-normalised, per 50-row band (the defect-2 signature)."""
    o, h = norm(ours[:, column]), norm(hist[:, column])
    out = []
    for start in range(0, HEIGHT, BAND_ROWS):
        stop = min(start + BAND_ROWS, HEIGHT)
        mask = usable[start:stop, column] & (o[start:stop] > 0) & (h[start:stop] > 0)
        entry: dict[str, Any] = {"rows": [start, stop], "count": int(mask.sum())}
        if mask.any():
            ratio = o[start:stop][mask] / h[start:stop][mask]
            entry["median_normalised_ratio"] = float(np.median(ratio))
        out.append(entry)
    return out


def lit_width(profile: np.ndarray, threshold: float) -> int:
    """Number of points of a max-normalised profile at or above ``threshold`` (the band's lateral extent)."""
    return int((norm(profile) >= threshold).sum())


def lumice_float_block(
    runs: list[tuple[np.ndarray, dict[str, Any]]],
    *,
    hist: np.ndarray,
    ours: np.ndarray,
    rendered: np.ndarray,
    complete: np.ndarray,
    column: int,
) -> dict[str, Any]:
    """The three-way radiometric check: merged Lumice float vs historical vs ours (module docstring, item 5)."""
    lumf, source = merge_lumice_float(runs)
    ours0 = np.where(rendered, np.nan_to_num(ours, nan=0.0), 0.0)
    columns = [c for c in (column + d for d in LUMICE_FLOAT_COLUMN_OFFSETS) if 0 <= c < WIDTH and rendered[:, c].any()]
    rows = (*PROBE_ROWS, *LUMICE_FLOAT_EXTRA_ROWS)
    profiles: dict[str, Any] = {}
    noise_positions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    widths: dict[str, Any] = {}
    for c in columns:
        profiles[f"column_{c}_lumice_float_vs_historical"] = log_profile_rms(lumf[:, c], hist[:, c])
        profiles[f"column_{c}_ours_vs_lumice_float"] = log_profile_rms(ours0[:, c], lumf[:, c], complete[:, c])
        profiles[f"column_{c}_ours_vs_historical"] = log_profile_rms(ours0[:, c], hist[:, c], complete[:, c])
        if len(runs) > 1:
            noise_positions[f"column_{c}"] = (runs[0][0][:, c], runs[1][0][:, c])
    for r in rows:
        profiles[f"row_{r}_lumice_float_vs_historical"] = log_profile_rms(lumf[r], hist[r])
        profiles[f"row_{r}_ours_vs_lumice_float"] = log_profile_rms(ours0[r], lumf[r], complete[r])
        profiles[f"row_{r}_ours_vs_historical"] = log_profile_rms(ours0[r], hist[r], complete[r])
        if len(runs) > 1:
            noise_positions[f"row_{r}"] = (runs[0][0][r], runs[1][0][r])
        widths[f"row_{r}"] = {
            f"{t:g}": {"historical": lit_width(hist[r], t), "ours": lit_width(ours0[r], t), "lumice_float": lit_width(lumf[r], t)}
            for t in EDGE_THRESHOLDS[1:]
        }
    centre = np.zeros(WIDTH, dtype=bool)
    centre[CENTRE_COLUMNS[0] : CENTRE_COLUMNS[1]] = True
    everywhere = np.ones((HEIGHT, WIDTH), dtype=bool)
    return {
        "source": source,
        "caveat": "Y accumulator of `Lumice render --format npy`, proportional to the energy landing in each pixel; "
        "max-normalised like the other two, so a uniform scale (and the symmetry fold) drops out",
        "columns": columns,
        "rows": list(rows),
        "log_profiles": profiles,
        "inner_edge_offsets": {f"column_{c}": edge_offsets(hist[:, c], ours0[:, c], lumf[:, c]) for c in columns},
        "column_decay_ratio_lumice_float_vs_historical": {f"column_{c}": decay_ratio_along_column(lumf, hist, everywhere, c) for c in columns},
        "column_decay_ratio_ours_vs_lumice_float": {f"column_{c}": decay_ratio_along_column(ours0, lumf, complete, c) for c in columns},
        "column_decay_ratio_ours_vs_historical": {f"column_{c}": decay_ratio_along_column(ours0, hist, complete, c) for c in columns},
        "row_lit_width": widths,
        "band_ratio_columns": list(CENTRE_COLUMNS),
        "band_ratio_lumice_float_vs_historical": band_ratios(lumf, hist, centre[None, :] & everywhere),
        "band_ratio_ours_vs_lumice_float": band_ratios(ours, lumf, complete & centre[None, :]),
        "noise": lumice_float_noise([y for y, _ in runs], noise_positions),
        "_merged": lumf,
    }


# ---------- figures ----------
def make_figures(
    out_dir: Path,
    *,
    hist: np.ndarray,
    lum: np.ndarray,
    ours: np.ndarray,
    status: np.ndarray,
    rendered: np.ndarray,
    unknown: np.ndarray,
    bands: list[dict[str, Any]],
    bands_all_columns: list[dict[str, Any]],
    baseline_bands: list[dict[str, Any]] | None,
    bands_on_baseline_columns: list[dict[str, Any]] | None,
    column: int,
    label: str,
) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    w_h, w_o = white_of(hist), white_of(np.where(rendered, ours, np.nan))
    disp_h, disp_o, disp_l = display(hist, w_h), display(np.where(rendered, ours, 0.0), w_o), lum / 255.0
    has_arc = (status & STATUS_BITS["has_arc"]) != 0
    lit = np.nan_to_num(ours, nan=0.0) > 0
    smap = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)  # 0 not rendered
    smap[rendered & ~unknown & lit] = 1
    smap[rendered & ~unknown & ~lit] = 2
    smap[rendered & unknown] = 3
    smap[rendered & has_arc] = 4

    fig, axes = plt.subplots(1, 4, figsize=(17, 9), constrained_layout=True)
    axes[0].imshow(disp_h, cmap="gray", vmin=0, vmax=1, aspect="auto")
    axes[0].set_title(f"historical direct raw\n(ch06 log display, white=p99={w_h:.3g})")
    axes[1].imshow(disp_l, cmap="gray", vmin=0, vmax=1, aspect="auto")
    axes[1].set_title("Lumice remake 1e9 rays\n(tone-mapped 8-bit as stored)")
    canvas = np.stack([disp_o] * 3, axis=-1)
    canvas[~rendered] = 0.35
    axes[2].imshow(canvas, aspect="auto")
    axes[2].set_title(f"Lumice Integral v2 (ours)\n{int(rendered.any(axis=0).sum())} of 251 columns; gray = not rendered\n(same log display, white=p99={w_o:.3g})")
    cmap = ListedColormap(["#595959", "#2a7de1", "#d9d9d9", "#d1232a", "#f2a900"])
    axes[3].imshow(smap, cmap=cmap, vmin=0, vmax=4, aspect="auto", interpolation="nearest")
    axes[3].set_title("status layer (ours)\nblue = complete & lit · light = complete & zero\nred = unknown completeness · orange = has open arc")
    for ax in axes:
        ax.set_xlabel("column")
    axes[0].set_ylabel("row")
    fig.suptitle(f"ch06 251x801 strip — {label} (each image on its own display scale; radiometry NOT aligned)", fontsize=12)
    images_path = out_dir / "strip_images_v2.png"
    fig.savefig(images_path, dpi=110)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    ax = axes[0, 0]
    centres = [0.5 * (b["rows"][0] + b["rows"][1]) for b in bands[1:] if "median_relative" in b]
    rel = [b["median_relative"] for b in bands[1:] if "median_relative" in b]
    lo = [b["p25_relative"] for b in bands[1:] if "median_relative" in b]
    hi = [b["p75_relative"] for b in bands[1:] if "median_relative" in b]
    ax.fill_between(centres, lo, hi, color="#2a7de1", alpha=0.15, lw=0, label="ours: p25-p75 (centre columns)")
    ax.semilogy(centres, rel, "o-", color="#2a7de1", lw=2, label=f"ours v2, columns {CENTRE_COLUMNS[0]}-{CENTRE_COLUMNS[1] - 1} (median per 50-row band)")
    ac = [0.5 * (b["rows"][0] + b["rows"][1]) for b in bands_all_columns[1:] if "median_relative" in b]
    ar = [b["median_relative"] for b in bands_all_columns[1:] if "median_relative" in b]
    ax.semilogy(ac, ar, "x-", color="#7fb3e6", lw=1, ms=5, label="ours v2, all columns (wing shapes differ)")
    if baseline_bands:
        bc = [0.5 * (b["rows"][0] + b["rows"][1]) for b in baseline_bands[1:] if "median_relative" in b]
        br = [b["median_relative"] for b in baseline_bands[1:] if "median_relative" in b]
        ax.semilogy(bc, br, "s--", color="#999999", lw=1.5, ms=4, label="v1 baseline (its centre columns)")
    if bands_on_baseline_columns:
        bc = [0.5 * (b["rows"][0] + b["rows"][1]) for b in bands_on_baseline_columns[1:] if "median_relative" in b]
        br = [b["median_relative"] for b in bands_on_baseline_columns[1:] if "median_relative" in b]
        ax.semilogy(bc, br, "^:", color="#2a7de1", lw=1.2, ms=4, alpha=0.7, label="ours v2 on the same columns")
    ax.axhline(1.0, color="#e07b00", lw=1.2, ls="--", label="uniform scale")
    ax.set_ylim(0.05, 20)
    ax.set_xlabel("row (band centre)")
    ax.set_ylabel("median(ours / historical) / whole-image median")
    ax.set_title("row-band ratio against the historical raw (log)")
    ax.legend(loc="upper left", frameon=False, fontsize=8)

    rows = np.arange(HEIGHT)
    ax = axes[0, 1]
    ax.semilogy(rows, np.maximum(norm(hist[:, column]), PROFILE_FLOOR), color="#e07b00", lw=2, label="historical")
    ax.semilogy(rows, np.maximum(norm(lum[:, column]), PROFILE_FLOOR), color="#5aa02c", lw=2, label="Lumice grey")
    ax.semilogy(rows, np.maximum(norm(np.where(rendered[:, column], ours[:, column], 0.0)), PROFILE_FLOOR), color="#2a7de1", lw=2, label="ours v2")
    unk_rows = rows[unknown[:, column]]
    if unk_rows.size:
        ax.plot(unk_rows, np.full(unk_rows.size, PROFILE_FLOOR * 1.5), "|", color="#d1232a", ms=8, label="ours: unknown rows")
    ax.set_title(f"vertical profile, column {column} (max-normalised, log)")
    ax.set_xlabel("row")
    ax.set_ylim(PROFILE_FLOOR, 1.5)
    ax.legend(loc="upper right", frameon=False, fontsize=8)

    ax = axes[0, 2]
    o, h = norm(np.where(rendered[:, column], ours[:, column], 0.0)), norm(hist[:, column])
    ok = (o > 0) & (h > 0) & ~unknown[:, column]
    ax.semilogy(rows[ok], (o / np.where(h > 0, h, np.nan))[ok], ".", color="#2a7de1", ms=3, label="ours / historical")
    ol = norm(lum[:, column])
    okl = (ol > 0) & (h > 0)
    ax.semilogy(rows[okl], (ol / np.where(h > 0, h, np.nan))[okl], ".", color="#5aa02c", ms=2, alpha=0.6, label="Lumice grey / historical")
    ax.axhline(1.0, color="#e07b00", lw=1.2, ls="--")
    ax.set_ylim(0.01, 100)
    ax.set_title(f"normalised ratio along column {column} (log)")
    ax.set_xlabel("row")
    ax.legend(loc="upper left", frameon=False, fontsize=8)

    cols = np.arange(WIDTH)
    for ax, r in zip(axes[1], PROBE_ROWS):
        ax.semilogy(cols, np.maximum(norm(hist[r]), PROFILE_FLOOR), color="#e07b00", lw=2, label="historical")
        ax.semilogy(cols, np.maximum(norm(lum[r]), PROFILE_FLOOR), color="#5aa02c", lw=2, label="Lumice grey")
        ax.semilogy(cols, np.maximum(norm(np.where(rendered[r], ours[r], 0.0)), PROFILE_FLOOR), color="#2a7de1", lw=2, label="ours v2")
        ax.set_title(f"horizontal profile, row {r} (max-normalised, log)")
        ax.set_xlabel("column")
        ax.set_ylim(PROFILE_FLOOR, 1.5)
    axes[1, 0].legend(loc="upper right", frameon=False, fontsize=8)
    for ax in axes.ravel():
        ax.grid(True, which="major", color="#dddddd", lw=0.6)
    fig.suptitle(f"ch06 strip profiles — {label}", fontsize=12)
    profiles_path = out_dir / "strip_profiles_v2.png"
    fig.savefig(profiles_path, dpi=110)
    plt.close(fig)
    return {"images": str(images_path), "profiles": str(profiles_path)}


def make_float_figure(
    out_dir: Path,
    *,
    hist: np.ndarray,
    ours: np.ndarray,
    rendered: np.ndarray,
    lumf: np.ndarray,
    column: int,
    label: str,
) -> str:
    """Three-way log profiles with the Lumice float export on the +-20 columns and the four rows."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ours0 = np.where(rendered, np.nan_to_num(ours, nan=0.0), 0.0)
    columns = [c for c in (column + d for d in LUMICE_FLOAT_COLUMN_OFFSETS) if 0 <= c < WIDTH and rendered[:, c].any()]
    rows_list = (*PROBE_ROWS, *LUMICE_FLOAT_EXTRA_ROWS)
    series = (("historical", hist, "#e07b00"), ("Lumice float", lumf, "#5aa02c"), ("ours v2", ours0, "#2a7de1"))
    fig, axes = plt.subplots(3, 4, figsize=(22, 14), constrained_layout=True)
    rows = np.arange(HEIGHT)
    for ax, c in zip(axes[0], columns):
        for name, arr, colour in series:
            ax.semilogy(rows, np.maximum(norm(arr[:, c]), PROFILE_FLOOR), color=colour, lw=1.6, label=name)
        ax.set_title(f"vertical profile, column {c} (max-normalised, log)")
        ax.set_xlabel("row")
        ax.set_ylim(PROFILE_FLOOR, 1.5)
    axes[0, 0].legend(loc="upper right", frameon=False, fontsize=8)
    for ax, c in zip(axes[1], columns):
        h, f, o = norm(hist[:, c]), norm(lumf[:, c]), norm(ours0[:, c])
        ok = (h > 0) & (f > 0)
        ax.semilogy(rows[ok], f[ok] / h[ok], ".", color="#5aa02c", ms=3, label="Lumice float / historical")
        ok = (f > 0) & (o > 0)
        ax.semilogy(rows[ok], o[ok] / f[ok], ".", color="#2a7de1", ms=3, label="ours / Lumice float")
        ax.axhline(1.0, color="#e07b00", lw=1.2, ls="--")
        ax.set_ylim(0.01, 100)
        ax.set_title(f"normalised ratio along column {c} (log)")
        ax.set_xlabel("row")
    axes[1, 0].legend(loc="upper left", frameon=False, fontsize=8)
    axes[0, 3].axis("off")
    axes[1, 3].axis("off")
    cols = np.arange(WIDTH)
    for ax, r in zip(axes[2], rows_list):
        for name, arr, colour in series:
            ax.semilogy(cols, np.maximum(norm(arr[r]), PROFILE_FLOOR), color=colour, lw=1.6, label=name)
        ax.set_title(f"horizontal profile, row {r} (max-normalised, log)")
        ax.set_xlabel("column")
        ax.set_ylim(PROFILE_FLOOR, 1.5)
    for ax in axes.ravel():
        ax.grid(True, which="major", color="#dddddd", lw=0.6)
    fig.suptitle(f"ch06 strip, three-way radiometric profiles with the Lumice float export — {label}", fontsize=12)
    path = out_dir / "strip_profiles_lumice_float.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return str(path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--strip-dir", type=Path, required=True, help="render output directory (provenance.json + raw arrays)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--writing-lab-dir", type=Path, default=DEFAULT_WRITING_LAB_DIR)
    parser.add_argument("--baseline-npz", type=Path, default=None, help="v1 strip58_v1_results.npz for the band-ratio regression check")
    parser.add_argument("--column", type=int, default=PROBE_COLUMN, help="vertical profile column (nearest rendered column is used)")
    parser.add_argument("--label", default="", help="free text for figure titles and the JSON")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--lumice-float", type=Path, default=None, help="Lumice `--format npy` img_0N.npy (sidecar img_0N.json required) for the radiometric three-way check")
    parser.add_argument("--lumice-float-run2", type=Path, default=None, help="second independent run of the same config; summed into the arbitrating profile, differenced for the noise floor")
    args = parser.parse_args(argv)
    if args.lumice_float_run2 is not None and args.lumice_float is None:
        parser.error("--lumice-float-run2 needs --lumice-float")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    arrays, provenance = read_strip(args.strip_dir)
    if provenance["arrays"]["shape"] != [HEIGHT, WIDTH]:
        raise SystemExit(f"strip shape {provenance['arrays']['shape']} != {[HEIGHT, WIDTH]}")
    rendered = arrays.rendered
    unknown = (arrays.status & STATUS_BITS["unknown_completeness"]) != 0
    complete = rendered & ~unknown
    ours = np.where(rendered, arrays.values, np.nan)
    hist = load_historical(args.writing_lab_dir / HISTORICAL_RAW)
    lum, lumice_note = load_lumice(args.writing_lab_dir / LUMICE_REMAKE)
    baseline = load_baseline(args.baseline_npz)
    float_runs = [load_lumice_float(p) for p in (args.lumice_float, args.lumice_float_run2) if p is not None]

    rendered_columns = np.where(rendered.any(axis=0))[0]
    column = int(rendered_columns[np.argmin(np.abs(rendered_columns - args.column))])
    label = args.label or provenance.get("execution", {}).get("label", "") or str(args.strip_dir)

    centre = np.zeros(WIDTH, dtype=bool)
    centre[CENTRE_COLUMNS[0] : CENTRE_COLUMNS[1]] = True
    bands = band_ratios(ours, hist, complete & centre[None, :])
    bands_all_columns = band_ratios(ours, hist, complete)
    baseline_bands = None
    bands_on_baseline_columns = None
    if baseline is not None:
        # like-for-like on the centre columns the v1 baseline rendered (108 .. 144, every ninth)
        baseline_columns = np.isfinite(baseline).any(axis=0) & centre
        baseline_bands = band_ratios(baseline, hist, np.isfinite(baseline) & baseline_columns[None, :])
        bands_on_baseline_columns = band_ratios(ours, hist, complete & baseline_columns[None, :])
    profiles: dict[str, Any] = {
        f"column_{column}_vs_historical": log_profile_rms(np.where(rendered[:, column], ours[:, column], 0.0), hist[:, column], complete[:, column]),
        f"column_{column}_vs_lumice": log_profile_rms(np.where(rendered[:, column], ours[:, column], 0.0), lum[:, column], complete[:, column]),
        f"column_{column}_lumice_vs_historical": log_profile_rms(lum[:, column], hist[:, column]),
    }
    for r in PROBE_ROWS:
        profiles[f"row_{r}_vs_historical"] = log_profile_rms(np.where(rendered[r], ours[r], 0.0), hist[r], complete[r])
        profiles[f"row_{r}_vs_lumice"] = log_profile_rms(np.where(rendered[r], ours[r], 0.0), lum[r], complete[r])
        profiles[f"row_{r}_lumice_vs_historical"] = log_profile_rms(lum[r], hist[r])
    edges = {
        f"column_{c}": edge_offsets(hist[:, c], np.where(rendered[:, c], ours[:, c], 0.0), lum[:, c])
        for c in sorted({column, 100, 150})
        if rendered[:, c].any()
    }
    status_summary = {
        "rendered": int(rendered.sum()),
        "complete": int(complete.sum()),
        "unknown": int(unknown.sum()),
        "lit": int((complete & (np.nan_to_num(ours, nan=0.0) > 0)).sum()),
        "has_arc": int(((arrays.status & STATUS_BITS["has_arc"]) != 0).sum()),
        "quadrature_unavailable": int(((arrays.status & STATUS_BITS["quadrature_unavailable"]) != 0).sum()),
        "node_count_exhausted": int(((arrays.status & STATUS_BITS["node_count_exhausted"]) != 0).sum()),
        "unknown_fraction_by_band": [
            {"rows": [a, min(a + BAND_ROWS, HEIGHT)], "fraction": float(unknown[a : a + BAND_ROWS][rendered[a : a + BAND_ROWS]].mean()) if rendered[a : a + BAND_ROWS].any() else None}
            for a in range(0, HEIGHT, BAND_ROWS)
        ],
    }
    metrics: dict[str, Any] = {
        "generated": dt.datetime.now().astimezone().isoformat(),
        "strip_dir": str(args.strip_dir),
        "strip_format": provenance.get("format"),
        "strip_execution": {k: provenance.get("execution", {}).get(k) for k in ("hostname", "started", "finished", "wall_clock_s", "workers", "label")},
        "historical_raw": str(args.writing_lab_dir / HISTORICAL_RAW),
        "lumice_remake": lumice_note,
        "lumice_caveat": "tone-mapped 8-bit grey; log-domain numbers against it measure display space, not radiometry",
        "profile_column": column,
        "status_summary": status_summary,
        "band_ratio_columns": list(CENTRE_COLUMNS),
        "band_ratio_vs_historical": bands,
        "band_ratio_vs_historical_all_columns": bands_all_columns,
        "band_ratio_v1_baseline": baseline_bands,
        "band_ratio_ours_on_v1_columns": bands_on_baseline_columns,
        "column_decay_ratio": decay_ratio_along_column(ours, hist, complete, column),
        "log_profiles": profiles,
        "inner_edge_offsets": edges,
        "spearman_auxiliary": spearman_block(ours, hist, lum, rendered, complete),
    }
    if float_runs:
        metrics["lumice_float"] = lumice_float_block(float_runs, hist=hist, ours=ours, rendered=rendered, complete=complete, column=column)
    if not args.no_figures:
        metrics["figures"] = make_figures(
            args.output_dir,
            hist=hist,
            lum=lum,
            ours=ours,
            status=arrays.status,
            rendered=rendered,
            unknown=unknown,
            bands=bands,
            bands_all_columns=bands_all_columns,
            baseline_bands=baseline_bands,
            bands_on_baseline_columns=bands_on_baseline_columns,
            column=column,
            label=label,
        )
        if float_runs:
            metrics["figures"]["lumice_float_profiles"] = make_float_figure(
                args.output_dir, hist=hist, ours=ours, rendered=rendered, lumf=metrics["lumice_float"]["_merged"], column=column, label=label
            )
    if float_runs:
        del metrics["lumice_float"]["_merged"]
    out = args.output_dir / "compare_metrics.json"
    out.write_text(json.dumps(metrics, indent=2) + "\n")

    print(f"strip: {status_summary['rendered']} rendered, {status_summary['complete']} complete, {status_summary['unknown']} unknown, {status_summary['has_arc']} with arcs")
    print(f"band ratio, columns {CENTRE_COLUMNS[0]}-{CENTRE_COLUMNS[1] - 1} (median relative to whole-image median):")
    for b in bands[1:]:
        if "median_relative" in b:
            print(f"  rows {b['rows'][0]:3d}-{b['rows'][1]:3d}: {b['median_relative']:.3f}  (n={b['count']})")
    for key, value in profiles.items():
        print(f"log profile {key}: lit-band rms {value['lit_band']['rms_log10']:.3f} (n={value['lit_band']['count']}), all rms {value['all']['rms_log10']:.3f} (n={value['all']['count']})")
    for key, value in edges.items():
        for e in value:
            print(f"edge {key} @{e['threshold']:g}: hist {e['historical_row']} ours {e['ours_row']} lumice {e['lumice_row']} -> ours-hist {e['ours_minus_historical']}")
    if float_runs:
        lf = metrics["lumice_float"]
        print(f"lumice float: {lf['source']['sim_ray_num']:.3g} rays over {len(lf['source']['runs'])} run(s), symmetry {lf['source']['symmetry']} (fold x{lf['source']['symmetry_fold_to_single_3_5']} to single 3-5)")
        for key, value in lf["log_profiles"].items():
            print(f"log profile {key}: lit-band rms {value['lit_band']['rms_log10']:.3f} (n={value['lit_band']['count']})")
        for key, value in lf["noise"].items():
            if isinstance(value, dict) and "relative_difference_std" in value:
                print(f"noise {key}: run1-run2 rel std {value['relative_difference_std']:.4f}, merged rel noise {value['merged_relative_noise']:.4f}, log rms {value['log_rms_run1_vs_run2']:.4f}")
        for key, value in lf["inner_edge_offsets"].items():
            for e in value:
                print(f"edge {key} @{e['threshold']:g}: hist {e['historical_row']} ours {e['ours_row']} lumice-float {e['lumice_row']}")
    print(f"spearman lit-in-both {metrics['spearman_auxiliary']['lit_in_both']['rho']:.4f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
