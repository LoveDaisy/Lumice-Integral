"""Render the canonical ch06 251 x 801 direct 3-5 strip (or a window of it).

Per pixel: component discovery with neighbour hot start -> production trace
with the four named weights -> adaptive line quadrature -> component sum
(``lumice_integral.strip_pixel``).  Columns are rendered in parallel by
spawned worker processes (``lumice_integral.strip_driver``) and written as
headerless float64/float32 raw arrays plus a status layer, a per-pixel CSV
and ``provenance.json`` (``lumice_integral.strip_io``).

Examples::

    uv run python scripts/render_ch06_strip.py --rows 140:160 --columns 140:160 \\
        --workers 4 --output-dir /tmp/strip-smoke
    XLA_FLAGS="--xla_cpu_multi_thread_eigen=false --xla_cpu_intra_op_parallelism_threads=1" OMP_NUM_THREADS=1 \\
        uv run python scripts/render_ch06_strip.py --workers 30 --output-dir artifacts/strip-full --resume

The scene-level prescan table (``--prescan-samples`` Haar poses, ``--rng-seed``)
is built once in the parent before the workers start and shared with them;
``--prescan-cache`` keeps it on disk with a provenance sidecar for reuse.

Spawned workers get glibc malloc trimming (``strip_driver.WORKER_MALLOC_ENV``)
unless the variables are already set; without it a Linux worker's RSS grows by
~2 GB per column.  ``--workers 1`` renders in-process, so export them yourself
there if the run is long.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import platform
import sys
from pathlib import Path

from lumice_integral.canonical_scene import CANONICAL_RENDER
from lumice_integral.continuation import ContinuationOptions
from lumice_integral.quadrature import QuadratureOptions
from lumice_integral.prescan import DEFAULT_RNG_SEED, DEFAULT_SAMPLE_COUNT
from lumice_integral.strip_driver import PIXEL_MODELS, DriverOptions, PrescanBuildOptions, render_window
from lumice_integral.strip_io import Window, write_strip
from lumice_integral.strip_pixel import PixelOptions


def parse_range(text: str, upper: int) -> tuple[int, int]:
    start, _, stop = text.partition(":")
    lower_bound = int(start) if start else 0
    upper_bound = int(stop) if stop else upper
    if not (0 <= lower_bound < upper_bound <= upper):
        raise argparse.ArgumentTypeError(f"range {text!r} must satisfy 0 <= a < b <= {upper}")
    return lower_bound, upper_bound


def main(argv: list[str] | None = None) -> None:
    height, width = CANONICAL_RENDER["height"], CANONICAL_RENDER["width"]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rows", default=f"0:{height}", help=f"half-open row range a:b (default 0:{height})")
    parser.add_argument("--columns", default=f"0:{width}", help=f"half-open column range a:b (default 0:{width})")
    parser.add_argument("--column-step", type=int, default=1, help="render every k-th column only (coarse preview)")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true", help="reuse finished column checkpoints in <output-dir>/columns")
    parser.add_argument("--pixel-model", choices=PIXEL_MODELS, default="point")
    parser.add_argument("--subpixel-grid", type=int, default=3)
    parser.add_argument("--subpixel-rows", default=None, help="row band a:b where the subpixel model applies")
    parser.add_argument("--cold-check-interval", type=int, default=8)
    parser.add_argument("--quadrature-rtol", type=float, default=1e-6)
    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument("--order-estimate", action="store_true", help="also run the 2-level order-estimate pass per fiber")
    parser.add_argument("--discovery-step-budget", type=int, default=250)
    parser.add_argument("--retry-step-budget", type=int, default=None, help="default: production maximum_accepted_steps")
    parser.add_argument(
        "--stall-floor-window",
        type=int,
        default=PixelOptions.stall_floor_window,
        help="skip the retrace of a step_budget candidate whose last N accepted discovery steps sat at "
        "minimum_step; a value above --discovery-step-budget disables the skip",
    )
    parser.add_argument(
        "--prescan-samples",
        type=int,
        default=DEFAULT_SAMPLE_COUNT,
        help="Haar samples of the scene-level prescan table, built once per run and shared by every worker",
    )
    parser.add_argument("--rng-seed", type=int, default=DEFAULT_RNG_SEED, help="seed of the prescan sampling stream")
    parser.add_argument(
        "--prescan-cache",
        type=Path,
        default=None,
        help="optional .npz path to cache the prescan table (reused on --resume when its provenance matches)",
    )
    parser.add_argument("--label", default="", help="free-text note stored in provenance.execution")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    window = Window(parse_range(args.rows, height), parse_range(args.columns, width), args.column_step)
    pixel_options = PixelOptions(
        discovery_step_budget=args.discovery_step_budget,
        retry_step_budget=args.retry_step_budget,
        stall_floor_window=args.stall_floor_window,
        continuation=ContinuationOptions(),
        quadrature=QuadratureOptions(
            epsilon=args.epsilon,
            relative_tolerance=args.quadrature_rtol,
            convergence_order_levels=2 if args.order_estimate else 0,
        ),
    )
    options = DriverOptions(
        pixel=pixel_options,
        prescan=PrescanBuildOptions(
            sample_count=args.prescan_samples, rng_seed=args.rng_seed, cache_path=args.prescan_cache
        ),
        cold_check_interval=args.cold_check_interval,
        pixel_model=args.pixel_model,
        subpixel_grid=args.subpixel_grid,
        subpixel_rows=parse_range(args.subpixel_rows, height) if args.subpixel_rows else None,
    )
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    def log(message: str) -> None:
        if not args.quiet:
            print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)

    started = dt.datetime.now().astimezone()
    log(f"rendering rows {window.rows} x columns {window.columns} ({window.pixel_count} pixels) with {args.workers} worker(s)")
    results, execution = render_window(
        window,
        options,
        workers=args.workers,
        checkpoint_dir=output_dir / "columns",
        resume=args.resume,
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
    files = write_strip(
        output_dir,
        results,
        options=pixel_options,
        window=window,
        height=height,
        width=width,
        pixel_model=options.pixel_model_block(),
        execution=execution,
        repo=Path(__file__).resolve().parent.parent,
        prescan=options.prescan.as_json(),
    )
    unknown = sum(r.completeness != "complete" for r in results)
    lit = sum(r.component_count > 0 for r in results)
    log(
        f"done: {len(results)} pixels ({lit} with components, {unknown} unknown completeness) "
        f"in {execution['wall_clock_s']:.1f}s wall clock; provenance {files['provenance']}"
    )


if __name__ == "__main__":
    main()
