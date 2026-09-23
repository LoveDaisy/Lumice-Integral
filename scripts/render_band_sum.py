"""Render a strip (or any linear-camera window) with the S^2 band sum (``lumice_integral.band_sum``).

Per pixel: the events of the path's (or path class's) S^2 event stores whose
deviation lies in the pixel's band, each with its pose rebuilt from its own
deviation, weighted by the pose density and summed with the derived constant
``1 / (2 pi N (delta_hi - delta_lo) sin(delta))`` (roadmap section 4.2).  The
output directory has the ``strip_io`` layout (``strip_io.read_strip`` and
``scripts/compare_strip_v2.py`` read it) with a band-sum ``pixels.csv``
(``K``, ``K_rho_pos``, ``K_eff`` per pixel) and ``provenance.json``.

Examples::

    uv run python scripts/render_band_sum.py --store-n 1000000 --rows 140:160 --columns 140:160 \\
        --workers 2 --output-dir /tmp/band-sum-smoke
    JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 \\
        uv run python scripts/render_band_sum.py --store-n 100000000 --workers 4 --output-dir artifacts/band-sum-full
    uv run python scripts/render_band_sum.py --store-n 100000000 --path 3 5 --path-class --workers 4 \\
        --pose-density-family plate --pose-density-zenith-std-deg 1 \\
        --width 301 --height 201 --fov-deg 40 --view-elevation 15 --output-dir /tmp/band-sum-plate

Scene: the canonical ch06 constants (hexagonal column ``h/a = 2``, ``n = 1.31``,
sun at 15 deg) and, by default, the canonical 251 x 801 camera; ``--width`` /
``--height`` / ``--fov-deg`` / ``--view-azimuth`` / ``--view-elevation`` replace
the camera (linear lens).  ``--pose-density-*`` are ``render_ch06_strip.py``'s
flags with the same validation (``pose_density.resolve_pose_density_parameters``).

Stores: ``--path`` (face sequence, default ``3 5``) alone renders that one
path; ``--path-class`` expands it to its PBD class, one store for all its
``Phi`` groups, each served through a ``D6h`` element, mirrors included
(``band_sum.store_plan``; ``--no-symmetry-transport`` gives every ``Phi``
group its own store, a verification mode).  Stores are built once in the parent (``--store-n``
Fibonacci points, cached under ``--store-cache-dir`` by parameter hash,
``s2_store.build_or_load``) before the workers start; a cached store is
loaded, never rebuilt.  A rank-0 class is the task 9 point mass on the sun
pixel.

``--workers``: the default 1 is for smokes; a full image wants 4 (the macOS
cap, :data:`MAC_MAX_WORKERS`; ~1 GB of store per worker at ``N = 1e8``).
Set ``JAX_PLATFORMS=cpu OMP_NUM_THREADS=1`` for multi-worker runs (the
setting ``render_ch06_strip.py`` runs with) so the workers do not oversubscribe
threads.  An existing non-empty ``--output-dir`` is refused unless
``--overwrite`` is given.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import platform
import sys
from pathlib import Path

import numpy as np

from lumice_integral.band_sum import BandSumScene, render_band_sum_window, single_path_class, write_band_sum_strip
from lumice_integral.canonical_scene import (
    CANONICAL_POSE_DENSITY_FAMILY,
    CANONICAL_REFRACTIVE_INDEX,
    CANONICAL_RENDER,
    CANONICAL_ZENITH_STD_DEG,
    canonical_crystal,
    canonical_incident_direction,
)
from lumice_integral.optics import path_id_of
from lumice_integral.path_class import build_path_class
from lumice_integral.pose_density import POSE_DENSITY_FAMILIES, build_pose_density
from lumice_integral.pose_density_provenance import pose_density_provenance
from lumice_integral.s2_store import DEFAULT_CACHE_DIR
from lumice_integral.strip_io import Window

MAC_MAX_WORKERS = 4


def parse_range(text: str, upper: int) -> tuple[int, int]:
    start, _, stop = text.partition(":")
    lower_bound = int(start) if start else 0
    upper_bound = int(stop) if stop else upper
    if not (0 <= lower_bound < upper_bound <= upper):
        raise argparse.ArgumentTypeError(f"range {text!r} must satisfy 0 <= a < b <= {upper}")
    return lower_bound, upper_bound


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true", help="write into an existing non-empty --output-dir")
    parser.add_argument("--store-n", type=int, required=True, help="Fibonacci points of each S^2 event store (e.g. 100000000)")
    parser.add_argument("--path", type=int, nargs="+", default=[3, 5], help="face sequence (default 3 5)")
    parser.add_argument("--path-class", action="store_true", help="render the PBD class of --path, not the path alone")
    parser.add_argument(
        "--no-symmetry-transport",
        action="store_true",
        help="with --path-class: one store per Phi group instead of one per class (verification mode)",
    )
    parser.add_argument("--store-cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--skip-store-self-checks", action="store_true", help="build new stores without the section 4.1(a) checks")
    parser.add_argument("--width", type=int, default=CANONICAL_RENDER["width"])
    parser.add_argument("--height", type=int, default=CANONICAL_RENDER["height"])
    parser.add_argument("--fov-deg", type=float, default=CANONICAL_RENDER["fov_deg"])
    parser.add_argument("--view-azimuth", type=float, default=CANONICAL_RENDER["view"]["azimuth"])
    parser.add_argument("--view-elevation", type=float, default=CANONICAL_RENDER["view"]["elevation"])
    parser.add_argument("--rows", default=None, help="half-open row range a:b (default: every row)")
    parser.add_argument("--columns", default=None, help="half-open column range a:b (default: every column)")
    parser.add_argument("--column-step", type=int, default=1, help="render every k-th column only (coarse preview)")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=f"worker processes (default 1 is for smokes; full images: {MAC_MAX_WORKERS}, the macOS cap)",
    )
    parser.add_argument(
        "--pose-density-family",
        choices=POSE_DENSITY_FAMILIES,
        default=CANONICAL_POSE_DENSITY_FAMILY,
        help=f"pose-density family weighting every pixel (default {CANONICAL_POSE_DENSITY_FAMILY!r}, the canonical density)",
    )
    parser.add_argument(
        "--pose-density-zenith-mean-deg", type=float, default=None, help="c-axis zenith mean (family default when omitted)"
    )
    parser.add_argument(
        "--pose-density-zenith-std-deg",
        type=float,
        default=None,
        help=f"c-axis zenith width: {CANONICAL_ZENITH_STD_DEG:g} when omitted for {CANONICAL_POSE_DENSITY_FAMILY!r}, "
        "required for plate / parry / lowitz, not accepted for random",
    )
    parser.add_argument("--pose-density-roll-mean-deg", type=float, default=None, help="roll mean (parry / lowitz; family default when omitted)")
    parser.add_argument("--pose-density-roll-std-deg", type=float, default=None, help="roll width (required for parry / lowitz)")
    parser.add_argument("--label", default="", help="free-text note stored in provenance.execution")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    if args.workers < 1:
        parser.error("--workers must be positive")
    if platform.system() == "Darwin" and args.workers > MAC_MAX_WORKERS:
        parser.error(f"--workers {args.workers} exceeds the macOS limit of {MAC_MAX_WORKERS}")
    if args.store_n < 1:
        parser.error("--store-n must be positive")
    if args.no_symmetry_transport and not args.path_class:
        parser.error("--no-symmetry-transport needs --path-class")
    if min(args.width, args.height) < 1 or not 0.0 < args.fov_deg < 180.0:
        parser.error("--width / --height must be positive and --fov-deg in (0, 180)")
    try:
        rows = parse_range(args.rows, args.height) if args.rows else (0, args.height)
        columns = parse_range(args.columns, args.width) if args.columns else (0, args.width)
        window = Window(rows, columns, args.column_step)
    except (argparse.ArgumentTypeError, ValueError) as exc:
        parser.error(str(exc))
    zenith_std_deg = args.pose_density_zenith_std_deg
    if zenith_std_deg is None and args.pose_density_family == CANONICAL_POSE_DENSITY_FAMILY:
        zenith_std_deg = CANONICAL_ZENITH_STD_DEG
    density_arguments = {
        "zenith_mean_deg": args.pose_density_zenith_mean_deg,
        "zenith_std_deg": zenith_std_deg,
        "roll_mean_deg": args.pose_density_roll_mean_deg,
        "roll_std_deg": args.pose_density_roll_std_deg,
    }
    try:
        pose_density = build_pose_density(args.pose_density_family, **density_arguments)
        pose_density_block = pose_density_provenance(args.pose_density_family, **density_arguments)
    except ValueError as exc:
        parser.error(str(exc))
    crystal = canonical_crystal()
    try:
        path_class = build_path_class(crystal, args.path) if args.path_class else single_path_class(crystal, args.path)
    except (ValueError, KeyError, IndexError) as exc:
        parser.error(f"--path {' '.join(map(str, args.path))}: {exc}")
    output_dir: Path = args.output_dir
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        parser.error(f"{output_dir} exists and is not empty; pass --overwrite to replace its files")

    render = {
        "width": args.width,
        "height": args.height,
        "fov_deg": args.fov_deg,
        "view": {"azimuth": args.view_azimuth, "elevation": args.view_elevation},
    }
    scene = BandSumScene(
        path_class=path_class,
        crystal=crystal,
        refractive_index=CANONICAL_REFRACTIVE_INDEX,
        incident_direction=canonical_incident_direction(),
        pose_density=pose_density,
        render=render,
        transport=not args.no_symmetry_transport,
    )

    def log(message: str) -> None:
        if not args.quiet:
            print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)

    started = dt.datetime.now().astimezone()
    what = f"class of {path_id_of(path_class.representative)} ({path_class.size} members)" if args.path_class else path_id_of(path_class.representative)
    log(f"band sum of {what}, N = {args.store_n}: {window.pixel_count} pixels with {args.workers} worker(s)")
    results, execution = render_band_sum_window(
        scene,
        window,
        args.store_n,
        workers=args.workers,
        base_dir=args.store_cache_dir,
        run_checks=not args.skip_store_self_checks,
        log=None if args.quiet else log,
    )
    finished = dt.datetime.now().astimezone()
    execution.update(
        {
            "started": started.isoformat(),
            "finished": finished.isoformat(),
            "command": [sys.argv[0], *(argv if argv is not None else sys.argv[1:])],
            "cwd": os.getcwd(),
            "hostname": platform.node(),
            "label": args.label,
        }
    )
    files = write_band_sum_strip(
        output_dir,
        results,
        scene=scene,
        n=args.store_n,
        window=window,
        pose_density_block=pose_density_block,
        execution=execution,
        repo=Path(__file__).resolve().parent.parent,
    )
    lit = int(np.count_nonzero([r.value > 0.0 for r in results]))
    log(
        f"done: {len(results)} pixels ({lit} lit) in {execution['wall_clock_s']:.1f} s wall clock "
        f"(stores {execution.get('stores_wall_clock_s', 0.0):.1f} s); provenance {files['provenance']}"
    )


if __name__ == "__main__":
    main()
