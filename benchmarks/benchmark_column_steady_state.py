"""Steady-state per-pixel cost of one strip column, with XLA compile counting.

This is the ruler for the strip pipeline's *real* single-process cost: one
column of the ch06 scene rendered top-down through the production
:func:`lumice_integral.strip_driver.render_column` (warm seeds from the pixel
above, default :class:`DriverOptions`), timing every pixel and counting every
XLA compilation JAX reports through ``jax.log_compiles``.  The steady-state
figure drops the first ``--warmup`` pixels (cold kernels, first seed-store
query) and takes the median of the rest.

It is *not* the same quantity as the single-pixel hot benchmark of
scrum strip-pipeline-v2 5.4 (``probe_step7_timing.py``: the same pixel run
repeatedly, every shape already cached).  That number is a lower bound on
what one pixel costs; this one is what a column actually costs.

Usage (Mac laptop, a few minutes with the default seed store built in memory):

    uv run python benchmarks/benchmark_column_steady_state.py --column 126 --rows 100:160
    uv run python benchmarks/benchmark_column_steady_state.py --column 126 --rows 0:801 \\
        --seed-store-cache-dir artifacts/s2-store --output /tmp/column-126.json
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import re
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

# ``Compiling jit(rotation_distance) with global shapes and types (...)``
# (``jax/_src/interpreters/pxla.py``); the module name is the token after
# ``Compiling``.  A message without that shape is not counted.
_COMPILE_MESSAGE = re.compile(r"^Compiling (\S+) with global shapes")
JAX_LOGGER_NAME = "jax"


def compile_name(message: str) -> str | None:
    """The compiled module name of a JAX ``Compiling ...`` log line, else ``None``."""
    match = _COMPILE_MESSAGE.match(message)
    return match.group(1) if match else None


class CompileCounter(logging.Handler):
    """Logging handler that counts JAX compile messages by module name."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.names: Counter[str] = Counter()

    @property
    def total(self) -> int:
        return sum(self.names.values())

    def emit(self, record: logging.LogRecord) -> None:
        name = compile_name(record.getMessage())
        if name is not None:
            self.names[name] += 1


def summarize(
    pixel_seconds: list[float], pixel_compiles: list[int], warmup: int
) -> dict[str, Any]:
    """Aggregate per-pixel timings and compile counts into the reported figures.

    ``warmup`` pixels are dropped from the steady-state statistics but kept
    in the totals; a window shorter than ``warmup`` reports the steady-state
    fields as ``None`` rather than measuring the warm-up itself.
    """
    if len(pixel_seconds) != len(pixel_compiles):
        raise ValueError("pixel_seconds and pixel_compiles must have the same length")
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    steady_seconds = pixel_seconds[warmup:]
    steady_compiles = pixel_compiles[warmup:]
    steady = bool(steady_seconds)
    return {
        "pixel_count": len(pixel_seconds),
        "warmup_pixels": min(warmup, len(pixel_seconds)),
        "total_seconds": float(sum(pixel_seconds)),
        "total_compiles": int(sum(pixel_compiles)),
        "steady_pixel_count": len(steady_seconds),
        "steady_median_seconds_per_pixel": float(statistics.median(steady_seconds)) if steady else None,
        "steady_mean_seconds_per_pixel": float(statistics.fmean(steady_seconds)) if steady else None,
        "steady_max_seconds_per_pixel": float(max(steady_seconds)) if steady else None,
        "steady_compiles": int(sum(steady_compiles)) if steady else None,
        "steady_pixels_with_compiles": int(sum(count > 0 for count in steady_compiles)) if steady else None,
    }


def parse_range(text: str) -> tuple[int, int]:
    start, stop = text.split(":")
    return int(start), int(stop)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--column", type=int, default=126)
    parser.add_argument("--rows", default="0:801", help="half-open row range a:b")
    parser.add_argument("--warmup", type=int, default=10, help="leading pixels excluded from the steady state")
    parser.add_argument(
        "--seed-store-cache-dir", type=Path, default=None, help="optional s2_store cache base directory of the seed store"
    )
    parser.add_argument("--output", type=Path, default=None, help="write the JSON report here as well as to stdout")
    parser.add_argument("--quiet", action="store_true", help="no per-pixel progress lines")
    args = parser.parse_args(argv)

    # Imported here so ``compile_name``/``summarize`` stay importable without JAX.
    import jax

    from lumice_integral.strip_driver import DriverOptions, SeedStoreOptions, build_scene_seeds, render_column
    from lumice_integral.strip_pixel import canonical_strip_scene

    def log(message: str) -> None:
        if not args.quiet:
            print(message, flush=True)

    options = DriverOptions(seed_store=SeedStoreOptions(cache_dir=args.seed_store_cache_dir))
    rows = range(*parse_range(args.rows))

    counter = CompileCounter()
    jax_logger = logging.getLogger(JAX_LOGGER_NAME)
    jax_logger.addHandler(counter)
    # ``jax.log_compiles`` raises the compile messages to WARNING so a
    # default-configured logger passes them to ``counter``; nothing else is
    # printed because the handler swallows the records it counts.
    with jax.log_compiles(True):
        seeds = build_scene_seeds(options, log=log)
        scene = canonical_strip_scene(seeds=seeds)
        setup_names = Counter(counter.names)

        marks: list[tuple[float, int]] = [(time.perf_counter(), counter.total)]

        def on_pixel(message: str) -> None:
            marks.append((time.perf_counter(), counter.total))
            elapsed = marks[-1][0] - marks[-2][0]
            compiles = marks[-1][1] - marks[-2][1]
            log(f"{message} | {elapsed:.3f} s, {compiles} compiles")

        results = render_column(scene, args.column, rows, options, log=on_pixel)
    jax_logger.removeHandler(counter)

    pixel_seconds = [later[0] - earlier[0] for earlier, later in zip(marks, marks[1:])]
    pixel_compiles = [later[1] - earlier[1] for earlier, later in zip(marks, marks[1:])]
    report = {
        "column": args.column,
        "rows": [rows.start, rows.stop],
        "platform": platform.platform(),
        "machine": platform.node(),
        "jax_version": jax.__version__,
        "jax_backend": jax.default_backend(),
        "setup_compiles": sum(setup_names.values()),
        "pixel_compiles_by_name": dict((counter.names - setup_names).most_common()),
        "pixels": [
            {
                "row": result.row,
                "seconds": seconds,
                "compiles": compiles,
                "component_count": result.component_count,
                "pool_count": result.pool_count,
            }
            for result, seconds, compiles in zip(results, pixel_seconds, pixel_compiles)
        ],
        **summarize(pixel_seconds, pixel_compiles, args.warmup),
    }
    text = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
