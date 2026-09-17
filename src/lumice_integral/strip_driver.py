"""Batch orchestration of :func:`.strip_pixel.render_pixel` over a pixel window.

Scan order and parallel axis: one task per *column*; inside a column the rows
are scanned top-down and every pixel hot-starts from the integrated
components of the pixel above (the only hot-start direction with regression
evidence: ``tests/test_discovery.py`` six-pixel baselines and the
``explore-component-discovery`` row-drift survey).  Columns never share
seeds, so column tasks are independent and embarrassingly parallel.

Every ``cold_check_interval``-th row of a column also runs the cold prescan
and compares component fingerprints with the hot-start chain (the "spot
check" of the acceptance criteria); a mismatch keeps the cold result and is
counted in the status layer.

Pixel models: ``"point"`` evaluates the pixel-centre direction;
``"subpixel"`` averages the full pipeline over a ``subpixel_grid`` x
``subpixel_grid`` grid of sub-pixel centres, either for every row or only for
``subpixel_rows`` (a row band such as the caustic neighbourhood).  Each
sub-pixel keeps its own hot-start chain down the column.

Workers are ``multiprocessing`` *spawn* processes (never fork: JAX runtime
state must not be inherited); each builds its own :class:`.strip_pixel.StripScene`
once, so per-process JIT compilation happens once.  Finished columns are
checkpointed as pickles so a long run can be resumed.

Prescan table: the scene-level :class:`.prescan.PrescanTable` is built (or
loaded from ``DriverOptions.prescan.cache_path``) exactly once in the parent
process before any column is scheduled and handed to every worker through the
pool's ``initargs``; workers only unpickle it (rebuilding the kd-tree) and
never sample.  ``prescan.cache_path`` is optional and outside the checkpoint
fingerprint (it changes where the table is read from, not what it holds).

Worker memory: before task-scene-prescan-table every cold discovery evaluated
a 400k-sample prescan eagerly, and on glibc the freed intermediates stayed in
the malloc arenas, so a worker's RSS climbed by ~60-100 MB per cold check
(measured 2026-09-16 on ``home-wsl``: 2.4 GB after 400 rows of one column,
flat at 0.37 GB with trimming enabled).  :data:`WORKER_MALLOC_ENV` is still
applied with ``setdefault`` before the pool is spawned (the per-pixel
clustering and tracing allocate too); it is a no-op on macOS and is recorded
in ``provenance.environment``.
"""

from __future__ import annotations

import multiprocessing
import os
import pickle
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from .canonical_scene import CANONICAL_REFRACTIVE_INDEX, canonical_incident_direction
from .prescan import DEFAULT_RNG_SEED, DEFAULT_SAMPLE_COUNT, PrescanTable, build_or_load_prescan_table
from .strip_io import Window
from .strip_pixel import (
    HotSeed,
    PixelOptions,
    PixelResult,
    StripScene,
    canonical_strip_scene,
    render_pixel,
    subpixel_targets,
)

PIXEL_MODELS = ("point", "subpixel")
LogCallback = Callable[[str], None]

# glibc malloc tuning for spawned workers (see the module docstring); values
# are strings because they go straight into ``os.environ``.
WORKER_MALLOC_ENV: dict[str, str] = {
    "MALLOC_ARENA_MAX": "2",
    "MALLOC_TRIM_THRESHOLD_": "131072",
    "MALLOC_MMAP_THRESHOLD_": "131072",
}


@dataclass(frozen=True)
class PrescanBuildOptions:
    """How the scene-level prescan table is built (or where it is cached).

    ``sample_count``/``rng_seed`` determine the table's content and are part
    of the checkpoint fingerprint; ``cache_path`` only says where to keep it
    and is excluded from equality, so a resumed run may point at another
    cache file without recomputing columns.
    """

    sample_count: int = DEFAULT_SAMPLE_COUNT
    rng_seed: int = DEFAULT_RNG_SEED
    cache_path: Path | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if self.sample_count < 1:
            raise ValueError("prescan sample_count must be positive")

    def as_json(self) -> dict[str, Any]:
        return {
            "sample_count": int(self.sample_count),
            "rng_seed": int(self.rng_seed),
            "cache_path": str(self.cache_path) if self.cache_path is not None else None,
        }


@dataclass(frozen=True)
class DriverOptions:
    pixel: PixelOptions = field(default_factory=PixelOptions)
    # ``default_factory`` on purpose: a checkpoint written before the scene-
    # level prescan existed was rendered under a different discovery policy
    # (per-pixel 400k prescan) and must be recomputed, which the
    # ``AttributeError`` branch of :func:`load_checkpoints` does.
    prescan: PrescanBuildOptions = field(default_factory=PrescanBuildOptions)
    cold_check_interval: int = 8  # rows; 0 disables the spot checks
    pixel_model: str = "point"
    subpixel_grid: int = 3
    subpixel_rows: tuple[int, int] | None = None  # half-open row band; None = every row

    def __post_init__(self) -> None:
        if self.pixel_model not in PIXEL_MODELS:
            raise ValueError(f"pixel_model must be one of {PIXEL_MODELS}")
        if self.cold_check_interval < 0:
            raise ValueError("cold_check_interval must be non-negative")
        if self.subpixel_grid < 1:
            raise ValueError("subpixel_grid must be positive")

    def uses_subpixel(self, row: int) -> bool:
        if self.pixel_model != "subpixel":
            return False
        if self.subpixel_rows is None:
            return True
        return self.subpixel_rows[0] <= row < self.subpixel_rows[1]

    def pixel_model_block(self) -> dict[str, Any]:
        return {
            "model": self.pixel_model,
            "epsilon": self.pixel.quadrature.epsilon,
            "subpixel_grid": self.subpixel_grid if self.pixel_model == "subpixel" else None,
            "subpixel_rows": list(self.subpixel_rows) if self.subpixel_rows else None,
            "description": {
                "point": "pixel-centre direction, J_perp regularised by epsilon",
                "subpixel": (
                    "arithmetic mean of the full pipeline over subpixel_grid^2 sub-pixel "
                    "centre directions (rows in subpixel_rows, or all rows when null); "
                    "other rows use the point model"
                ),
            }[self.pixel_model],
        }


def _merge_subpixels(row: int, column: int, parts: Sequence[PixelResult]) -> PixelResult:
    """One :class:`PixelResult` for the mean over sub-pixels.

    ``components`` are the centre-most sub-pixel's (so downstream hot-start
    seeds are well defined); counts, events and timings are summed;
    completeness is ``"complete"`` iff every sub-pixel is.

    ``error_estimate`` is the arithmetic mean of the sub-pixels' own
    quadrature error estimates only (code-review round 1 Suggestion).  Near a
    caustic, sub-pixel values within one pixel can themselves differ by
    O(10-40%) (the pixel-model probe, ``docs/ch06-reference-fixture.md``
    section 7 stage 4); that model-internal spread is not folded in here, so
    this is not an upper bound on the pixel's total uncertainty when
    sub-pixel averaging is enabled in a caustic neighbourhood.
    """
    centre = parts[len(parts) // 2]
    events: Counter = Counter()
    timings: Counter = Counter()
    for part in parts:
        events.update(part.events)
        timings.update(part.timings)
    return PixelResult(
        row=row,
        column=column,
        value=sum(part.value for part in parts) / len(parts),
        error_estimate=sum(part.error_estimate for part in parts) / len(parts),
        components=centre.components,
        completeness="complete" if all(part.completeness == "complete" for part in parts) else "unknown",
        discovery_completeness=centre.discovery_completeness,
        seed_source=centre.seed_source,
        incomplete_count=sum(part.incomplete_count for part in parts),
        pool_count=sum(part.pool_count for part in parts),
        raw_cluster_count=sum(part.raw_cluster_count for part in parts),
        admissible_count=sum(part.admissible_count for part in parts),
        events={name: int(count) for name, count in events.items()},
        timings={name: float(value) for name, value in timings.items()},
    )


def render_column(
    scene: StripScene,
    column: int,
    rows: range,
    options: DriverOptions,
    log: LogCallback | None = None,
) -> list[PixelResult]:
    """Scan one column top-down with the hot-start chain (module docstring)."""
    results: list[PixelResult] = []
    previous: tuple[HotSeed, ...] | None = None
    previous_parts: list[tuple[HotSeed, ...] | None] | None = None
    for index, row in enumerate(rows):
        cold_check = bool(options.cold_check_interval) and index > 0 and index % options.cold_check_interval == 0
        if options.uses_subpixel(row):
            targets = subpixel_targets(scene.render, row, column, options.subpixel_grid)
            if previous_parts is None or len(previous_parts) != len(targets):
                previous_parts = [previous] * len(targets)
            parts = [
                render_pixel(scene, row, column, options.pixel, target=target, previous=seeds, cold_check=cold_check)
                for target, seeds in zip(targets, previous_parts)
            ]
            previous_parts = [part.hot_seeds or None for part in parts]
            result = _merge_subpixels(row, column, parts)
        else:
            result = render_pixel(scene, row, column, options.pixel, previous=previous, cold_check=cold_check)
            previous_parts = None
        previous = result.hot_seeds or None
        results.append(result)
        if log is not None:
            log(
                f"column {column} row {row}: value={result.value:.6g} components={result.component_count} "
                f"{result.completeness} {result.seed_source} {result.timings.get('total_s', 0.0):.2f}s"
            )
    return results


# --- multiprocessing -------------------------------------------------------------

_WORKER_SCENE: StripScene | None = None
_WORKER_OPTIONS: DriverOptions | None = None


def _init_worker(options: DriverOptions, table: PrescanTable) -> None:
    global _WORKER_SCENE, _WORKER_OPTIONS
    _WORKER_SCENE = canonical_strip_scene(prescan_table=table)
    _WORKER_OPTIONS = options


def build_scene_prescan_table(
    options: DriverOptions, *, log: LogCallback | None = None, repo: Path | None = None
) -> PrescanTable:
    """The canonical scene's table for ``options.prescan`` (built or loaded once per run)."""
    return build_or_load_prescan_table(
        options.prescan.cache_path,
        canonical_incident_direction(),
        CANONICAL_REFRACTIVE_INDEX,
        sample_count=options.prescan.sample_count,
        rng_seed=options.prescan.rng_seed,
        repo=repo,
        log=log,
    )


def _render_column_task(task: tuple[int, tuple[int, int]]) -> tuple[int, list[PixelResult], float]:
    column, rows = task
    assert _WORKER_SCENE is not None and _WORKER_OPTIONS is not None
    start = time.perf_counter()
    results = render_column(_WORKER_SCENE, column, range(*rows), _WORKER_OPTIONS)
    return column, results, time.perf_counter() - start


def _checkpoint_path(checkpoint_dir: Path, column: int) -> Path:
    return checkpoint_dir / f"column_{column:04d}.pkl"


def load_checkpoints(
    checkpoint_dir: Path,
    window: Window,
    options: DriverOptions,
    log: LogCallback | None = None,
) -> dict[int, list[PixelResult]]:
    """Columns of ``window`` already rendered for exactly ``window.rows`` and ``options``.

    A checkpoint is only reused when its recorded :class:`DriverOptions`
    (pixel model, ``quadrature``/``continuation``/discovery numerics, ...)
    equals ``options`` exactly; a mismatch or a legacy payload written before
    this fingerprint existed is discarded and the column is recomputed, so a
    resumed run never silently mixes columns produced under different
    numerical policies into one ``provenance.json`` (code-review round 1
    Major 1: options drift across runs was previously unchecked).

    An options field added after a checkpoint was written is absent from the
    unpickled instance's ``__dict__``; the dataclass ``__eq__`` then reads the
    class-level literal default, so such a checkpoint still compares equal to
    the current defaults and is reused.  A field without a literal default
    (``default_factory``) has no class attribute to fall back on and the
    comparison raises ``AttributeError``; that is treated like a payload
    that predates the fingerprint rather than left to crash the resume.
    """
    found: dict[int, list[PixelResult]] = {}
    for column in window.column_range:
        path = _checkpoint_path(checkpoint_dir, column)
        if not path.exists():
            continue
        payload = pickle.loads(path.read_bytes())
        if payload.get("rows") != list(window.rows):
            continue
        if "options" not in payload:
            if log is not None:
                log(f"column {column} checkpoint predates the options fingerprint; recomputing")
            continue
        try:
            same_options = payload["options"] == options
        except AttributeError:
            if log is not None:
                log(f"column {column} checkpoint options predate a current option field; recomputing")
            continue
        if not same_options:
            if log is not None:
                log(f"column {column} checkpoint options differ from this run's options; recomputing")
            continue
        found[column] = payload["results"]
    return found


def render_window(
    window: Window,
    options: DriverOptions,
    *,
    workers: int = 1,
    checkpoint_dir: Path | None = None,
    resume: bool = False,
    scene: StripScene | None = None,
    log: LogCallback | None = None,
) -> tuple[list[PixelResult], dict[str, Any]]:
    """Render every pixel of ``window``; returns the results and an execution record.

    ``workers <= 1`` renders in-process (``scene`` may be supplied, in which
    case its own ``prescan_table`` is used and ``options.prescan`` is not
    consulted); otherwise ``workers`` spawn processes each take whole
    columns and share the table built here (module docstring).  The table's
    size and build time are reported in the execution record.
    """
    if workers < 1:
        raise ValueError("workers must be positive")
    start = time.perf_counter()
    done: dict[int, list[PixelResult]] = {}
    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if resume:
            done = load_checkpoints(checkpoint_dir, window, options, log=log)
            if log is not None and done:
                log(f"resumed {len(done)} column(s) from {checkpoint_dir}")
    pending = [column for column in window.column_range if column not in done]
    column_seconds: dict[int, float] = {}

    def finish(column: int, results: list[PixelResult], seconds: float) -> None:
        done[column] = results
        column_seconds[column] = seconds
        if checkpoint_dir is not None:
            _checkpoint_path(checkpoint_dir, column).write_bytes(
                pickle.dumps({"rows": list(window.rows), "results": results, "options": options})
            )
        if log is not None:
            unknown = sum(r.completeness != "complete" for r in results)
            log(
                f"column {column} done: {len(results)} pixels, {unknown} unknown, "
                f"{seconds:.1f}s ({len(done)}/{len(window.column_range)} columns)"
            )

    # The table is obtained once, before any column: built or loaded per
    # ``options.prescan`` unless an in-process ``scene`` already carries one.
    prescan: dict[str, Any] = {"source": "not-needed", "seconds": 0.0}
    if pending and (workers > 1 or scene is None):
        prescan_start = time.perf_counter()
        table = build_scene_prescan_table(options, log=log)
        prescan = {
            "source": str(options.prescan.cache_path) if options.prescan.cache_path is not None else "in-memory",
            "seconds": time.perf_counter() - prescan_start,
            "sample_count": table.sample_count,
            "valid_count": table.valid_count,
            "rng_seed": table.rng_seed,
        }
        if workers == 1:
            scene = canonical_strip_scene(prescan_table=table)
    elif pending:
        prescan = {
            "source": "scene",
            "seconds": 0.0,
            "sample_count": scene.prescan_table.sample_count,
            "valid_count": scene.prescan_table.valid_count,
            "rng_seed": scene.prescan_table.rng_seed,
        }

    if workers == 1:
        for column in pending:
            column_start = time.perf_counter()
            results = render_column(scene, column, window.row_range, options, log)
            finish(column, results, time.perf_counter() - column_start)
    elif pending:
        for name, value in WORKER_MALLOC_ENV.items():
            os.environ.setdefault(name, value)
        context = multiprocessing.get_context("spawn")
        tasks = [(column, window.rows) for column in pending]
        with context.Pool(workers, initializer=_init_worker, initargs=(options, table)) as pool:
            for column, results, seconds in pool.imap_unordered(_render_column_task, tasks):
                finish(column, results, seconds)

    results = [result for column in sorted(done) for result in done[column]]
    execution = {
        "wall_clock_s": time.perf_counter() - start,
        "workers": workers,
        "columns_rendered_now": len(pending),
        "columns_resumed": len(window.column_range) - len(pending),
        "column_seconds": {str(column): seconds for column, seconds in sorted(column_seconds.items())},
        "checkpoint_dir": str(checkpoint_dir) if checkpoint_dir is not None else None,
        "prescan": prescan,
    }
    return results, execution


__all__ = [
    "DriverOptions",
    "PIXEL_MODELS",
    "PrescanBuildOptions",
    "WORKER_MALLOC_ENV",
    "build_scene_prescan_table",
    "load_checkpoints",
    "render_column",
    "render_window",
]
