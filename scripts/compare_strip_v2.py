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

Inputs are read only; the historical raw and the Lumice PNG live in the
Writing-Lab project (``--writing-lab-dir``).  The display mapping is the
one documented in ``docs/ch06-reference-fixture.md`` section 3.2 and is
implemented here (there is no other implementation in this repository).
Plotting needs matplotlib, which is not a project dependency::

    uv run --with matplotlib python scripts/compare_strip_v2.py \\
        --strip-dir artifacts/strip-full --output-dir <task>/artifacts

The diagnostic positions (column ``126``, rows ``150 / 300 / 450``) are shared
with ``scripts/probe_defect2_factors.py``; keep :data:`PROBE_COLUMN` /
:data:`PROBE_ROWS` in sync with it.
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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--strip-dir", type=Path, required=True, help="render output directory (provenance.json + raw arrays)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--writing-lab-dir", type=Path, default=DEFAULT_WRITING_LAB_DIR)
    parser.add_argument("--baseline-npz", type=Path, default=None, help="v1 strip58_v1_results.npz for the band-ratio regression check")
    parser.add_argument("--column", type=int, default=PROBE_COLUMN, help="vertical profile column (nearest rendered column is used)")
    parser.add_argument("--label", default="", help="free text for figure titles and the JSON")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)
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
    print(f"spearman lit-in-both {metrics['spearman_auxiliary']['lit_in_both']['rho']:.4f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
