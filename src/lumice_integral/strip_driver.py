"""Batch orchestration of :func:`.strip_pixel.render_pixel` over a pixel window.

Scan order and parallel axis: one task per *column*; inside a column the rows
are scanned top-down and every pixel receives the integrated components of
the pixel above as ``warm_seeds`` (Gauss-Newton starts added to its own
seed-store candidate pool -- never a substitute for the pool, so there is no
hot-start blind spot and no periodic cold check; task-pixel-pipeline-v2).
Columns never share seeds, so column tasks are independent and
embarrassingly parallel.

Pixel models: ``"point"`` evaluates the pixel-centre direction;
``"subpixel"`` averages the full pipeline over a ``subpixel_grid`` x
``subpixel_grid`` grid of sub-pixel centres, either for every row or only for
``subpixel_rows`` (a row band such as the caustic neighbourhood).  Each
sub-pixel keeps its own warm-seed chain down the column.

Workers are ``multiprocessing`` *spawn* processes (never fork: JAX runtime
state must not be inherited); each builds its own :class:`.strip_pixel.StripScene`
once, so per-process JIT compilation happens once.  Finished columns are
checkpointed as pickles so a long run can be resumed; a checkpoint records
the :data:`.strip_io.FORMAT_VERSION` it was written under and is only reused
by the same version (the pickled result types change with the format).

Seed store: the scene-level :class:`.s2_store.StoreSeeds` (the ``3-5``
event store of ``DriverOptions.seed_store.n`` points on the canonical crystal)
is built (or loaded from ``DriverOptions.seed_store.cache_dir``) exactly once
in the parent process before any column is scheduled and handed to every
worker through the pool's ``initargs``; workers only unpickle it and never
sample.  ``seed_store.cache_dir`` is optional and outside the checkpoint
fingerprint (it changes where the store is read from, not what it holds).

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

import numpy as np

from .canonical_scene import (
    CANONICAL_POSE_DENSITY_FAMILY,
    CANONICAL_REFRACTIVE_INDEX,
    CANONICAL_ZENITH_MEAN_DEG,
    CANONICAL_ZENITH_STD_DEG,
    canonical_crystal,
    canonical_sun_direction,
)
from .optics import PATH_3_5_FACES
from .pose_density import PoseDensity, build_pose_density, resolve_pose_density_parameters
from .pose_density_provenance import pose_density_provenance
from .s2_store import DEFAULT_SEED_STORE_N, StoreSeeds
from .strip_io import FORMAT_VERSION, Window
from .strip_pixel import (
    PixelOptions,
    PixelResult,
    StripScene,
    canonical_strip_scene,
    render_pixel,
    seed_store,
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
class SeedStoreOptions:
    """How the scene-level seed store is built (or where it is cached).

    ``n`` (the store's point count) determines its content and is part of the
    checkpoint fingerprint; ``cache_dir`` (an :func:`.s2_store.build_or_load`
    base directory; ``None`` builds in memory) only says where to keep it and
    is excluded from equality, so a resumed run may point at another cache
    without recomputing columns.
    """

    n: int = DEFAULT_SEED_STORE_N
    cache_dir: Path | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if self.n < 1:
            raise ValueError("seed store n must be positive")

    def as_json(self) -> dict[str, Any]:
        return {"N": int(self.n), "cache_dir": str(self.cache_dir) if self.cache_dir is not None else None}


@dataclass(frozen=True)
class DriverOptions:
    pixel: PixelOptions = field(default_factory=PixelOptions)
    # ``default_factory`` on purpose: a checkpoint written before the seed
    # store existed was rendered under a different discovery policy (the Haar
    # prescan table) and must be recomputed, which the ``AttributeError``
    # branch of :func:`load_checkpoints` does.
    seed_store: SeedStoreOptions = field(default_factory=SeedStoreOptions)
    pixel_model: str = "point"
    subpixel_grid: int = 3
    subpixel_rows: tuple[int, int] | None = None  # half-open row band; None = every row
    # The pose-density family and its degree-valued parameters
    # (:func:`.pose_density.build_pose_density`); the five fields are one
    # parameter set and must be extended together.  Literal defaults on
    # purpose, equal to the canonical column density: a checkpoint pickled
    # before these fields existed was rendered with that density and compares
    # equal to the defaults (see :func:`load_checkpoints`).  ``__post_init__``
    # validates the set through ``resolve_pose_density_parameters`` (its
    # ``ValueError`` propagates unchanged) and stores the resolved values, so
    # an explicit family default (``zenith_mean_deg=90`` for column) and an
    # omitted one fingerprint alike; parameters a family does not take must be
    # ``None`` (``random`` therefore needs ``pose_density_zenith_mean_deg=None,
    # pose_density_zenith_std_deg=None``).
    pose_density_family: str = CANONICAL_POSE_DENSITY_FAMILY
    pose_density_zenith_mean_deg: float | None = CANONICAL_ZENITH_MEAN_DEG
    pose_density_zenith_std_deg: float | None = CANONICAL_ZENITH_STD_DEG
    pose_density_roll_mean_deg: float | None = None
    pose_density_roll_std_deg: float | None = None

    def __post_init__(self) -> None:
        if self.pixel_model not in PIXEL_MODELS:
            raise ValueError(f"pixel_model must be one of {PIXEL_MODELS}")
        if self.subpixel_grid < 1:
            raise ValueError("subpixel_grid must be positive")
        resolved = resolve_pose_density_parameters(
            self.pose_density_family,  # type: ignore[arg-type]
            zenith_mean_deg=self.pose_density_zenith_mean_deg,
            zenith_std_deg=self.pose_density_zenith_std_deg,
            roll_mean_deg=self.pose_density_roll_mean_deg,
            roll_std_deg=self.pose_density_roll_std_deg,
        )
        for name in ("zenith_mean_deg", "zenith_std_deg", "roll_mean_deg", "roll_std_deg"):
            object.__setattr__(self, f"pose_density_{name}", resolved.get(name))

    def pose_density_arguments(self) -> dict[str, float | None]:
        """Keyword arguments of ``build_pose_density`` / ``pose_density_provenance`` (besides the family)."""
        return {
            "zenith_mean_deg": self.pose_density_zenith_mean_deg,
            "zenith_std_deg": self.pose_density_zenith_std_deg,
            "roll_mean_deg": self.pose_density_roll_mean_deg,
            "roll_std_deg": self.pose_density_roll_std_deg,
        }

    def pose_density(self) -> PoseDensity:
        """The density every pixel of the run is weighted with."""
        return build_pose_density(self.pose_density_family, **self.pose_density_arguments())  # type: ignore[arg-type]

    def pose_density_block(self) -> dict[str, Any]:
        """The ``scene.pose_density`` value of ``provenance.json`` for this run's density."""
        return pose_density_provenance(self.pose_density_family, **self.pose_density_arguments())  # type: ignore[arg-type]

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

    ``components`` are the centre-most sub-pixel's (so downstream warm seeds
    are well defined); counts, events and timings are summed;
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
        incomplete_count=sum(part.incomplete_count for part in parts),
        pool_count=sum(part.pool_count for part in parts),
        extra_seed_count=sum(part.extra_seed_count for part in parts),
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
    """Scan one column top-down, warming every pixel with the one above (module docstring)."""
    results: list[PixelResult] = []
    previous: tuple[np.ndarray, ...] = ()
    previous_parts: list[tuple[np.ndarray, ...]] | None = None
    for row in rows:
        if options.uses_subpixel(row):
            targets = subpixel_targets(scene.render, row, column, options.subpixel_grid)
            if previous_parts is None or len(previous_parts) != len(targets):
                previous_parts = [previous] * len(targets)
            parts = [
                render_pixel(scene, row, column, options.pixel, target=target, warm_seeds=seeds)
                for target, seeds in zip(targets, previous_parts)
            ]
            previous_parts = [part.warm_seeds for part in parts]
            result = _merge_subpixels(row, column, parts)
        else:
            result = render_pixel(scene, row, column, options.pixel, warm_seeds=previous)
            previous_parts = None
        previous = result.warm_seeds
        results.append(result)
        if log is not None:
            log(
                f"column {column} row {row}: value={result.value:.6g} components={result.component_count} "
                f"(arcs {result.arc_count}) {result.completeness} {result.timings.get('total_s', 0.0):.2f}s"
            )
    return results


# --- multiprocessing -------------------------------------------------------------

_WORKER_SCENE: StripScene | None = None
_WORKER_OPTIONS: DriverOptions | None = None


def _init_worker(options: DriverOptions, seeds: StoreSeeds) -> None:
    global _WORKER_SCENE, _WORKER_OPTIONS
    _WORKER_SCENE = canonical_strip_scene(seeds=seeds, pose_density=options.pose_density())
    _WORKER_OPTIONS = options


def build_scene_seeds(options: DriverOptions, *, log: LogCallback | None = None) -> StoreSeeds:
    """The canonical scene's seeds for ``options.seed_store`` (the ``3-5`` store, built or loaded once per run)."""
    store = seed_store(
        canonical_crystal(),
        CANONICAL_REFRACTIVE_INDEX,
        (PATH_3_5_FACES,),
        options.seed_store.n,
        cache_dir=options.seed_store.cache_dir,
        log=log,
    )
    return StoreSeeds(store, PATH_3_5_FACES, canonical_sun_direction())


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

    A checkpoint is only reused when it was written under the current
    :data:`.strip_io.FORMAT_VERSION` and its recorded :class:`DriverOptions`
    (pixel model, ``quadrature``/``continuation``/discovery numerics, ...)
    equals ``options`` exactly; a mismatch, another format version, or a
    legacy payload written before this fingerprint existed is discarded and
    the column is recomputed, so a resumed run never silently mixes columns
    produced under different numerical policies or pipelines into one
    ``provenance.json`` (code-review round 1 Major 1: options drift across
    runs was previously unchecked).

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
        if payload.get("format") != FORMAT_VERSION:
            if log is not None:
                log(f"column {column} checkpoint format {payload.get('format')!r} != {FORMAT_VERSION!r}; recomputing")
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
    case its own ``seeds`` are used and ``options.seed_store`` is not
    consulted; its ``pose_density`` must equal ``options.pose_density()``,
    which is what provenance records); otherwise ``workers`` spawn processes each take whole
    columns and share the seeds built here (module docstring).  The store's
    size and build time are reported in the execution record.
    """
    if workers < 1:
        raise ValueError("workers must be positive")
    if scene is not None and scene.pose_density != options.pose_density():
        raise ValueError(f"scene pose density {scene.pose_density!r} does not match the options' {options.pose_density()!r}")
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
                pickle.dumps(
                    {"format": FORMAT_VERSION, "rows": list(window.rows), "results": results, "options": options}
                )
            )
        if log is not None:
            unknown = sum(r.completeness != "complete" for r in results)
            log(
                f"column {column} done: {len(results)} pixels, {unknown} unknown, "
                f"{seconds:.1f}s ({len(done)}/{len(window.column_range)} columns)"
            )

    def store_report(seeds: StoreSeeds, source: str, seconds: float) -> dict[str, Any]:
        return {
            "source": source,
            "seconds": seconds,
            "N": seeds.n,
            "kept_events": len(seeds.store.events),
            "cache_key": seeds.store.spec.cache_key(),
        }

    def build_seeds() -> tuple[StoreSeeds, dict[str, Any]]:
        store_start = time.perf_counter()
        seeds = build_scene_seeds(options, log=log)
        cache_dir = options.seed_store.cache_dir
        return seeds, store_report(seeds, str(cache_dir) if cache_dir is not None else "in-memory", time.perf_counter() - store_start)

    # The seeds are obtained once, before any column: built or loaded per
    # ``options.seed_store`` unless an in-process ``scene`` already carries them.
    # Whether to build them and whether a ``Pool`` will read them are decided by
    # the same branch (rather than two separately-shaped conditions) so that
    # "seeds are bound before ``Pool(...)`` uses them" holds by construction
    # instead of requiring two independent conditions to be kept in sync
    # (round 2 code review: a split if/elif pair made that invariant provable
    # only by cross-referencing two conditions, and was twice misread as a
    # possible ``UnboundLocalError``).
    store: dict[str, Any] = {"source": "not-needed", "seconds": 0.0}
    if pending and workers > 1:
        seeds, store = build_seeds()
        for name, value in WORKER_MALLOC_ENV.items():
            os.environ.setdefault(name, value)
        context = multiprocessing.get_context("spawn")
        tasks = [(column, window.rows) for column in pending]
        with context.Pool(workers, initializer=_init_worker, initargs=(options, seeds)) as pool:
            for column, results, seconds in pool.imap_unordered(_render_column_task, tasks):
                finish(column, results, seconds)
    elif pending:
        if scene is None:
            seeds, store = build_seeds()
            scene = canonical_strip_scene(seeds=seeds, pose_density=options.pose_density())
        else:
            store = store_report(scene.seeds, "scene", 0.0)
        for column in pending:
            column_start = time.perf_counter()
            results = render_column(scene, column, window.row_range, options, log)
            finish(column, results, time.perf_counter() - column_start)

    results = [result for column in sorted(done) for result in done[column]]
    execution = {
        "wall_clock_s": time.perf_counter() - start,
        "workers": workers,
        "columns_rendered_now": len(pending),
        "columns_resumed": len(window.column_range) - len(pending),
        "column_seconds": {str(column): seconds for column, seconds in sorted(column_seconds.items())},
        "checkpoint_dir": str(checkpoint_dir) if checkpoint_dir is not None else None,
        "seed_store": store,
    }
    return results, execution


__all__ = [
    "DriverOptions",
    "PIXEL_MODELS",
    "SeedStoreOptions",
    "WORKER_MALLOC_ENV",
    "build_scene_seeds",
    "load_checkpoints",
    "render_column",
    "render_window",
]
