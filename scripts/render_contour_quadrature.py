"""Render a strip (or any linear-camera window) by contour quadrature (``lumice_integral.contour_quadrature``).

Per pixel: every component of the level set ``{D_P = delta} ∩ U_P`` of the
pixel's deviation (certified complete, ``contour.extract_level_sets``), the
line integral of ``rho w / |grad D_P|`` along each by adaptive Simpson on
points solved onto the level set, times ``1 / (8 pi^2 sin delta)``
(``docs/phase2.md`` section 4).  Deterministic: no sampling noise, an error
estimate per pixel.  The output directory has the ``strip_io`` layout
(``strip_io.read_strip`` and ``scripts/compare_strip_v2.py`` read it) with a
contour-quadrature ``pixels.csv`` and ``provenance.json``.

Examples::

    uv run python scripts/render_contour_quadrature.py --rows 140:160 --columns 145:155 --output-dir /tmp/contour-quad-smoke
    JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 \\
        uv run python scripts/render_contour_quadrature.py --workers 4 --output-dir artifacts/contour-quadrature-full
    # the band-sum pixel model (band average, 4-point Gauss-Legendre per piece), for a like-for-like comparison
    JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 \\
        uv run python scripts/render_contour_quadrature.py --band-nodes 4 --workers 4 --output-dir artifacts/contour-quadrature-band

Scene and camera flags are ``scripts/render_band_sum.py``'s, copied rather than
shared (they describe the pixel geometry and the density, not either
algorithm; whoever changes the common set checks the other script):
canonical ch06 constants, ``--width`` / ``--height`` / ``--fov-deg`` /
``--view-*`` for another linear camera, ``--pose-density-*`` with the same
validation.  ``--path`` is one face sequence (no path classes here).

``--store-n``: the event store (``s2_store.build_or_load``, cached under
``--store-cache-dir``) is not a sample of the image here; it only seeds the
extraction's independent check (store seeds must lie on the extracted
components), so a small ``N`` suffices.  ``--band-nodes 0`` (default) is the
point pixel of Phase I, ``k > 0`` the band average of the band sum.
``--workers``: a full image wants 4 on a Mac with ``JAX_PLATFORMS=cpu
OMP_NUM_THREADS=1``.  An existing non-empty ``--output-dir`` is refused unless
``--overwrite`` is given.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import platform
import sys
from pathlib import Path

from lumice_integral.band_sum import single_path_class
from lumice_integral.canonical_scene import (
    CANONICAL_POSE_DENSITY_FAMILY,
    CANONICAL_REFRACTIVE_INDEX,
    CANONICAL_RENDER,
    CANONICAL_ZENITH_STD_DEG,
    canonical_crystal,
    canonical_sun_direction,
)
from lumice_integral.contour_quadrature import (
    ContourQuadratureScene,
    QuadratureOptions,
    render_contour_quadrature_window,
    write_contour_quadrature_strip,
)
from lumice_integral.optics import path_id_of
from lumice_integral.pose_density import POSE_DENSITY_FAMILIES, build_pose_density
from lumice_integral.pose_density_provenance import pose_density_provenance
from lumice_integral.s2_store import DEFAULT_CACHE_DIR
from lumice_integral.strip_io import Window

MAC_MAX_WORKERS = 4
DEFAULT_STORE_N = 1_000_000


def parse_range(text: str, upper: int) -> tuple[int, int]:
    start, _, stop = text.partition(":")
    lower_bound = int(start) if start else 0
    upper_bound = int(stop) if stop else upper
    if not (0 <= lower_bound < upper_bound <= upper):
        raise argparse.ArgumentTypeError(f"range {text!r} must satisfy 0 <= a < b <= {upper}")
    return lower_bound, upper_bound


def main(argv: list[str] | None = None) -> None:
    defaults = QuadratureOptions()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true", help="write into an existing non-empty --output-dir")
    parser.add_argument("--path", type=int, nargs="+", default=[3, 5], help="face sequence (default 3 5)")
    parser.add_argument("--store-n", type=int, default=DEFAULT_STORE_N, help="Fibonacci points of the seed-check event store")
    parser.add_argument("--store-cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--skip-store-self-checks", action="store_true", help="build a new store without the section 4.1(a) checks")
    parser.add_argument("--band-nodes", type=int, default=0, help="0: point pixel (Phase I model); k > 0: band average (band-sum model)")
    parser.add_argument("--relative-tolerance", type=float, default=defaults.relative_tolerance)
    parser.add_argument("--max-depth", type=int, default=defaults.max_depth, help="halvings of a node-to-node panel at most")
    parser.add_argument("--width", type=int, default=CANONICAL_RENDER["width"])
    parser.add_argument("--height", type=int, default=CANONICAL_RENDER["height"])
    parser.add_argument("--fov-deg", type=float, default=CANONICAL_RENDER["fov_deg"])
    parser.add_argument("--view-azimuth", type=float, default=CANONICAL_RENDER["view"]["azimuth"])
    parser.add_argument("--view-elevation", type=float, default=CANONICAL_RENDER["view"]["elevation"])
    parser.add_argument("--rows", default=None, help="half-open row range a:b (default: every row)")
    parser.add_argument("--columns", default=None, help="half-open column range a:b (default: every column)")
    parser.add_argument("--column-step", type=int, default=1, help="render every k-th column only (coarse preview)")
    parser.add_argument("--workers", type=int, default=1, help=f"worker processes (full images: {MAC_MAX_WORKERS}, the macOS cap)")
    parser.add_argument(
        "--pose-density-family",
        choices=POSE_DENSITY_FAMILIES,
        default=CANONICAL_POSE_DENSITY_FAMILY,
        help=f"pose-density family weighting every pixel (default {CANONICAL_POSE_DENSITY_FAMILY!r}, the canonical density)",
    )
    parser.add_argument("--pose-density-zenith-mean-deg", type=float, default=None, help="c-axis zenith mean (family default when omitted)")
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
    if args.store_n < 1 or args.band_nodes < 0:
        parser.error("--store-n must be positive and --band-nodes >= 0")
    if min(args.width, args.height) < 1 or not 0.0 < args.fov_deg < 180.0:
        parser.error("--width / --height must be positive and --fov-deg in (0, 180)")
    try:
        options = QuadratureOptions(relative_tolerance=args.relative_tolerance, max_depth=args.max_depth)
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
        path = single_path_class(crystal, args.path)  # render_band_sum.py's validation of --path
    except (ValueError, KeyError, IndexError) as exc:
        parser.error(f"--path {' '.join(map(str, args.path))}: {exc}")
    if path.halo_map_rank == 0:
        parser.error(f"--path {path_id_of(path.representative)} has halo-map rank 0: a point mass, not a level-set integral")
    output_dir: Path = args.output_dir
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        parser.error(f"{output_dir} exists and is not empty; pass --overwrite to replace its files")

    render = {
        "width": args.width,
        "height": args.height,
        "fov_deg": args.fov_deg,
        "view": {"azimuth": args.view_azimuth, "elevation": args.view_elevation},
    }
    scene = ContourQuadratureScene(
        faces=path.representative,
        crystal=crystal,
        refractive_index=CANONICAL_REFRACTIVE_INDEX,
        sun_direction=canonical_sun_direction(),
        pose_density=pose_density,
        render=render,
        options=options,
        band_nodes=args.band_nodes,
    )

    def log(message: str) -> None:
        if not args.quiet:
            print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)

    started = dt.datetime.now().astimezone()
    log(f"contour quadrature of {path_id_of(scene.faces)} ({scene.pixel_model}): {window.pixel_count} pixels with {args.workers} worker(s)")
    results, execution = render_contour_quadrature_window(
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
    files = write_contour_quadrature_strip(
        output_dir,
        results,
        scene=scene,
        window=window,
        pose_density_block=pose_density_block,
        execution=execution,
        repo=Path(__file__).resolve().parent.parent,
    )
    lit = sum(r.value > 0.0 for r in results)
    cpu = execution["cpu_seconds"]
    log(
        f"done: {len(results)} pixels ({lit} lit) in {execution['wall_clock_s']:.1f} s wall clock; CPU s: extract "
        f"{cpu['extract_s']:.1f}, geometry {cpu['geometry_s']:.1f}, integrate {cpu['integrate_s']:.1f}; provenance {files['provenance']}"
    )


if __name__ == "__main__":
    main()
