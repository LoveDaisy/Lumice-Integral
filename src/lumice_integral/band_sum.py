"""The S^2 band-sum renderer (roadmap section 4.2): event-store band sums per pixel, path or path class.

Estimator.  With ``N`` equal-area points on ``S^2`` (``4 pi / N`` each) the
Phase I pixel value ``I`` at deviation ``delta`` satisfies, integrated over
the pixel's deviation band ``[delta_lo, delta_hi]`` (the extremes of its four
corners' deviations),

    I_hat = sum_{D_i in band} w_i rho(R_i) / (2 pi N (delta_hi - delta_lo) sin(delta))

with ``w_i = A_P(u_i) T_P(u_i)`` from the event store (:mod:`.s2_store`,
``u = R^-1 s_hat`` with ``s_hat`` toward the sun, ``docs/conventions.md``) and
``R_i`` the pose with ``R_i u_i = s_hat`` and ``R_i phi_i`` at the event's *own*
deviation ``D_i`` and the pixel's azimuth (:func:`.s2_store.event_rotations`;
the pixel's ``delta`` in place of ``D_i`` does not give a rotation).  The
constant is derived, not fitted.  A pixel value is a *band average*, not a
point value: next to a steep edge or an inner caustic it differs from the
point-pixel Phase I value by the band's own averaging (roadmap section 4.2).
Per pixel the renderer reports ``K`` (band events), ``K_rho_pos`` (events
with ``rho > 0``) and ``K_eff`` (Kish effective sample size of ``w rho``) next
to the value, the sampling-noise diagnostic ``~ 1 / sqrt(K_eff)``.  All three
count distinct precomputed events (``K_EFF_SEMANTICS``, class paragraph
below).

The estimator functions (:func:`pixel_band`, :func:`band_rotations`,
:func:`band_contributions`, :func:`band_sum_estimate`, :func:`kish_k_eff`,
:func:`band_sum_pixel`) are migrated verbatim from ``scripts/probe_band_sum.py``
(tasks 13/14), which now imports them from here.

Path classes (:func:`store_plan`).  The members of a PBD class are grouped by
:func:`.path_class.phi_key` (members sharing ``Phi`` share one store with
summed weights, :mod:`.s2_store`), and the ``Phi`` groups by ``D6h``
elements, proper and improper (:func:`.path_class.path_class_symmetry`):
a class is one ``D6h`` orbit, so one store serves every group, each through
its element ``g`` (:func:`.s2_store.transported_rotations`: the poses of
the events moved to ``(g u, g phi)``, ``L_g R g^T``; ``D`` and ``w`` are
invariant).  The class value is the sum of the member contributions.  The
precomputation view (roadmap section 4.2): the store on all of ``S^2`` is
the same information as every member's store on a fundamental domain, so
symmetry saves repeated evaluation and makes no new samples.  ``K``,
``K_rho_pos`` and ``K_eff`` therefore count distinct events: per store
event the contributions of all its transports are summed first,
``c_i = w_i sum_g rho(R_i^(g))``, and the Kish size is of ``{c_i}``
(``K_EFF_SEMANTICS = "per_event"``; task ``band-sum-renderer`` pooled every
``(event, transport)`` pair, ``"per_transport_sample"``, which overstates
``K_eff`` by up to the number of transports whose ``rho`` agree, 6 for a
plate density on ``[3,5]``).  ``transport=False`` gives every ``Phi`` group
its own store (a verification mode: on the same points it is the task 14
layout, whose member stores are independent samples).  The Fibonacci
lattice is not closed under ``D6h``, so the two modes sample different
points and agree to the discretisation level, not bit for bit.

A rank-0 class (``halo_map_rank == 0``) is not a band sum: its contribution
is the point mass of task 9 (:func:`.path_class.render_class_pixel` on the
one pixel containing the sun, ``m / pixel_solid_angle``, ``0`` elsewhere; the
Haar-stream estimate of :func:`.path_class.estimate_rank0_contribution`).

Rendering (:func:`render_band_sum_window`).  The parent builds or loads every
store of the plan once (:func:`.s2_store.build_or_load`) before any worker
starts; spawned workers load the saved stores (SHA-256 checked, no rebuild)
and render whole columns.  Output (:func:`write_band_sum_strip`) is the
:mod:`.strip_io` directory layout (``read_strip`` reads it) with a band-sum
``pixels.csv`` and ``provenance.json``.

The sun enters as ``s_hat`` (``sun`` / :attr:`BandSumScene.sun_direction`);
the deviation of a pixel is measured between propagation directions
(``s = -s_hat`` from :func:`.camera.incident_direction_from_sun`), and the
Phase I hand-offs (the rank-0 point mass) take that ``s``.

Nothing here imports or calls Lumice.
"""

from __future__ import annotations

import concurrent.futures
import csv
import dataclasses
import json
import multiprocessing
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .camera import incident_direction_from_sun, linear_pixel_outgoing_direction
from .canonical_scene import CANONICAL_RENDER
from .geometry import HexPrism, halo_map_rank, wedge_angle_deg
from .optics import normalize_faces, path_id_of
from .path_class import PathClass, build_class_scene, path_class_symmetry, phi_key, render_class_pixel, sun_pixel
from .pose_density import PoseDensity
from .prescan import DEFAULT_RNG_SEED, DEFAULT_SAMPLE_COUNT
from .provenance import git_commit, sha256_of
from .s2_store import (
    DEFAULT_CACHE_DIR,
    S2EventStore,
    build_or_load,
    event_rotations,
    max_rss_mb,
    transported_rotations,
)
from .strip_io import FILE_NAMES, Window, environment_block, scene_block, write_binary_arrays
from .strip_pixel import STATUS_HAS_COMPONENT, STATUS_RENDERED, PixelOptions

Faces = tuple[int, ...]
FORMAT_VERSION = "lumice-integral.band-sum/v1"
ESTIMATOR = (
    "band sum (roadmap section 4.2): I = sum_{D_i in [delta_lo, delta_hi]} w_i rho(R_i) / "
    "(2 pi N (delta_hi - delta_lo) sin(delta)); band from the pixel's four corner deviations; "
    "R_i from the event's own deviation D_i (s2_store.event_rotations); derived constant, not fitted"
)
PIXEL_CSV_COLUMNS = ("row", "column", "value", "delta_deg", "band_width_rad", "K", "K_rho_pos", "K_eff")
# What K, K_rho_pos and K_eff count (provenance ``options.k_eff_semantics``): distinct store events, each
# with its transports' contributions summed.  Renders without the field (task band-sum-renderer) pooled
# every (event, transport) pair: "per_transport_sample".
K_EFF_SEMANTICS = "per_event"


# ------------------------------------------------------------- estimator
# Migrated verbatim from scripts/probe_band_sum.py (task band-sum-quadrature-probe).
def pixel_band(
    row: int, column: int, sun: np.ndarray, render: Mapping[str, Any] = CANONICAL_RENDER
) -> tuple[np.ndarray, float, float, float]:
    """Pixel-centre direction, its deviation, and ``[delta_lo, delta_hi]`` from the four corners.

    The deviation is the angle between the incoming ``s = -s_hat`` and the
    outgoing propagation directions (equal to the sky point's angle from ``s_hat``).
    """
    s = incident_direction_from_sun(sun)
    centre = linear_pixel_outgoing_direction(row, column, **render)
    corners = [
        linear_pixel_outgoing_direction(row + dr, column + dc, **render)
        for dr in (-0.5, 0.5)
        for dc in (-0.5, 0.5)
    ]
    deviations = [float(np.arccos(np.clip(c @ s, -1.0, 1.0))) for c in corners]
    return centre, float(np.arccos(np.clip(centre @ s, -1.0, 1.0))), min(deviations), max(deviations)


def band_rotations(events: Mapping[str, np.ndarray], lo: int, hi: int, sun: np.ndarray, centre: np.ndarray) -> np.ndarray:
    """``R_i`` with ``R_i u_i = s_hat`` and ``R_i phi_i`` at deviation ``D_i``, azimuth of ``centre``."""
    return event_rotations(events["u"][lo:hi], events["phi"][lo:hi], events["D"][lo:hi], sun, centre)


def band_poses(
    events: Mapping[str, np.ndarray], sun: np.ndarray, centre: np.ndarray, lo_d: float, hi_d: float
) -> tuple[np.ndarray, np.ndarray]:
    """The band's poses ``R_i`` and weights ``w_i`` (times ``iw_i`` for a non-uniform store); empty if no event."""
    lo, hi = np.searchsorted(events["D"], [lo_d, hi_d])
    if hi <= lo:
        return np.zeros((0, 3, 3)), np.zeros(0)
    rotations = band_rotations(events, lo, hi, sun, centre)
    weight = events["w"][lo:hi] if "iw" not in events else events["w"][lo:hi] * events["iw"][lo:hi]
    return rotations, weight


def band_contributions(
    events: Mapping[str, np.ndarray], sun: np.ndarray, densities: Sequence, centre: np.ndarray, lo_d: float, hi_d: float
) -> list[np.ndarray]:
    """Per density, the band's contributions ``w_i rho(R_i)`` (times ``iw_i`` for a non-uniform store).

    The rotations are built once and shared by every density.
    """
    rotations, weight = band_poses(events, sun, centre, lo_d, hi_d)
    if len(weight) == 0:
        return [np.zeros(0) for _ in densities]
    return [weight * density.evaluate_batch(rotations) for density in densities]


def band_sum_estimate(total: float, n: int, width: float, delta: float) -> float:
    """The section 4.2 band sum ``sum / (2 pi N (delta_hi - delta_lo) sin(delta))`` (derived, not fitted)."""
    return total / (2.0 * np.pi * n * width * np.sin(delta))


def kish_k_eff(total: float, square: float) -> float:
    """Kish effective sample size ``(sum c)^2 / sum c^2`` of the contributions (0 for an empty band)."""
    return total * total / square if square > 0.0 else 0.0


def band_sum_pixel(
    events, sun, density, row: int, column: int, n: int, render: Mapping[str, Any] = CANONICAL_RENDER
) -> tuple[float, float, int, int, float, float]:
    """``(estimate, delta, K, K_rho_pos, K_eff, band_width)`` of one pixel (``sun`` = ``s_hat``)."""
    centre, delta, lo_d, hi_d = pixel_band(row, column, sun, render)
    width = hi_d - lo_d
    (contribution,) = band_contributions(events, sun, [density], centre, lo_d, hi_d)
    total = float(np.sum(contribution))
    square = float(np.sum(contribution**2))
    return (
        band_sum_estimate(total, n, width, delta),
        delta,
        int(len(contribution)),
        int(np.count_nonzero(contribution > 0.0)),
        kish_k_eff(total, square),
        width,
    )


# ------------------------------------------------------------ store plan
@dataclasses.dataclass(frozen=True, eq=False)
class Transport:
    """A ``Phi`` group served by a store through the ``D6h`` element ``g`` (:func:`.s2_store.transported_rotations`).

    ``g`` may be proper or improper.  ``g is None`` is the identity (the
    store's own group; no multiplication, so the poses are bit-identical to
    the untransported ones).

    ``eq=False`` (identity comparison): ``g`` is an ``np.ndarray``, which
    breaks the dataclass-generated ``__eq__``/``__hash__`` (ambiguous truth
    value / unhashable) if ever compared or hashed.
    """

    members: tuple[Faces, ...]
    g: np.ndarray | None

    def as_json(self) -> dict[str, Any]:
        return {
            "members": [path_id_of(m) for m in self.members],
            "g": None if self.g is None else np.round(self.g, 12).tolist(),
        }


@dataclasses.dataclass(frozen=True, eq=False)
class StoreGroup:
    """One event store (of the ``Phi`` group ``members``) and the groups it serves.

    ``eq=False`` for the same reason as :class:`Transport`: it holds
    ``Transport`` instances (which carry an ``np.ndarray`` field), so
    identity comparison avoids the same ambiguous-truth-value/unhashable trap.
    """

    members: tuple[Faces, ...]
    transports: tuple[Transport, ...]

    @property
    def served_members(self) -> tuple[Faces, ...]:
        return tuple(m for t in self.transports for m in t.members)

    def as_json(self) -> dict[str, Any]:
        return {"store_members": [path_id_of(m) for m in self.members], "transports": [t.as_json() for t in self.transports]}


def single_path_class(crystal: HexPrism, faces: Sequence[int]) -> PathClass:
    """A one-member :class:`.path_class.PathClass` of ``faces`` (the single-path renderer's unit)."""
    faces = normalize_faces(faces)
    return PathClass(faces, (faces,), wedge_angle_deg(crystal, faces), halo_map_rank(crystal, faces))


def store_plan(path_class: PathClass, crystal: HexPrism, *, transport: bool = True) -> tuple[StoreGroup, ...]:
    """Stores of a rank-2 class: ``Phi`` groups, then one store for the ``D6h`` orbit of groups (module docstring).

    Every member is served exactly once (checked).  A class is one ``D6h``
    orbit, so with ``transport`` the plan is a single store (the
    representative's group); a member outside the orbit is a class
    construction error (``RuntimeError`` from
    :func:`.path_class.path_class_symmetry`).  ``transport=False`` gives
    every group its own store.  A rank-0 class has no stores (empty plan).
    """
    if path_class.halo_map_rank == 0:
        return ()
    by_key: dict[Any, list[Faces]] = {}
    for member in path_class.members:
        by_key.setdefault(phi_key(crystal, member), []).append(member)
    groups = [tuple(sorted(g)) for g in by_key.values()]
    groups.sort(key=lambda g: (path_class.representative not in g, g))
    if not transport:
        plan = tuple(StoreGroup(group, (Transport(group, None),)) for group in groups)
    else:
        source = groups[0]
        rooted = PathClass(source[0], path_class.members, path_class.wedge_deg, path_class.halo_map_rank)
        symmetry = path_class_symmetry(rooted, crystal)
        transports = [Transport(source, None)]
        for group in groups[1:]:
            # g maps source[0] onto a member of ``group``; Phi_{g m}(u) = g Phi_m(g^-1 u) for every member m
            # of ``source`` (proper or improper g), so g maps the whole Phi group onto ``group`` (same size).
            # Any member's element serves; a proper one is preferred (no reflection factor, cheaper).
            g = next((symmetry[m] for m in group if np.linalg.det(symmetry[m]) > 0.0), symmetry[group[0]])
            transports.append(Transport(group, g))
        plan = (StoreGroup(source, tuple(transports)),)
    served = sorted(m for s in plan for m in s.served_members)
    if served != sorted(path_class.members):
        raise RuntimeError(f"store plan serves {served}, class has {sorted(path_class.members)}")
    return plan


# ---------------------------------------------------------------- pixels
@dataclasses.dataclass(frozen=True)
class BandSumPixelResult:
    """One pixel: the value, its deviation band and the diagnostics.

    ``K``, ``K_rho_pos`` and ``K_eff`` count distinct store events (``K_EFF_SEMANTICS``);
    ``total`` is the sum of the contributions and ``square`` the sum of the squared per-event ones.
    """

    row: int
    column: int
    value: float
    delta: float
    band_width_rad: float
    K: int
    K_rho_pos: int
    K_eff: float
    total: float
    square: float

    def csv_row(self) -> dict[str, Any]:
        return {
            "row": self.row,
            "column": self.column,
            "value": repr(self.value),
            "delta_deg": repr(float(np.degrees(self.delta))),
            "band_width_rad": repr(self.band_width_rad),
            "K": self.K,
            "K_rho_pos": self.K_rho_pos,
            "K_eff": repr(self.K_eff),
        }


def class_band_sum_pixel(
    stores: Sequence[tuple[Mapping[str, np.ndarray], StoreGroup]],
    sun: np.ndarray,
    density: PoseDensity,
    row: int,
    column: int,
    n: int,
    render: Mapping[str, Any] = CANONICAL_RENDER,
) -> BandSumPixelResult:
    """Band sum of one pixel over every ``(events, group)`` store of a plan; diagnostics per distinct event.

    Per store the band, its poses and weights are computed once; each
    transport evaluates ``rho`` at its transported poses
    (:func:`.s2_store.transported_rotations`).  The value is the sum over
    stores, transports and events.  ``K`` / ``K_rho_pos`` / ``K_eff`` are of
    the per-event contributions ``c_i = w_i sum_t rho(R_i^(t))`` (module
    docstring): transports reuse the same events and add no samples.  The
    value is accumulated per transport, in the order of task
    ``band-sum-renderer``, so it is unchanged bit for bit.  With one store
    and one identity transport this is :func:`band_sum_pixel`, value for
    value, except that an empty band is ``0`` also where ``sin(delta) = 0``.
    """
    centre, delta, lo_d, hi_d = pixel_band(row, column, sun, render)
    width = hi_d - lo_d
    total, square, k, k_pos = 0.0, 0.0, 0, 0
    for events, group in stores:
        rotations, weight = band_poses(events, sun, centre, lo_d, hi_d)
        if len(weight) == 0:
            continue
        per_event = np.zeros(len(weight))
        for t in group.transports:
            poses = rotations if t.g is None else transported_rotations(rotations, t.g, sun, centre)
            contribution = weight * density.evaluate_batch(poses)
            total += float(np.sum(contribution))
            per_event += contribution
        square += float(np.sum(per_event**2))
        k += int(len(per_event))
        k_pos += int(np.count_nonzero(per_event > 0.0))
    # An empty band is 0 even where the constant is singular (the pixel containing delta = 0, i.e. the sun).
    value = band_sum_estimate(total, n, width, delta) if total != 0.0 else 0.0
    return BandSumPixelResult(int(row), int(column), value, delta, width, k, k_pos, kish_k_eff(total, square), total, square)


# ---------------------------------------------------------------- scene
@dataclasses.dataclass(frozen=True)
class BandSumScene:
    """What a band-sum render depends on, besides the stores: optics, crystal, camera, density, class.

    ``sun_direction`` is ``s_hat`` (toward the sun); :attr:`incident_direction` the propagation ``-s_hat``.
    """

    path_class: PathClass
    crystal: HexPrism
    refractive_index: float
    sun_direction: np.ndarray
    pose_density: PoseDensity
    render: Mapping[str, Any]
    transport: bool = True
    rank0_sample_count: int = DEFAULT_SAMPLE_COUNT
    rank0_rng_seed: int = DEFAULT_RNG_SEED

    @property
    def incident_direction(self) -> np.ndarray:
        """The propagation direction ``s = -s_hat`` for the Phase I hand-offs."""
        return incident_direction_from_sun(self.sun_direction)

    @property
    def plan(self) -> tuple[StoreGroup, ...]:
        return store_plan(self.path_class, self.crystal, transport=self.transport)


def prepare_stores(
    scene: BandSumScene,
    n: int,
    *,
    base_dir: Path = DEFAULT_CACHE_DIR,
    run_checks: bool = True,
    log: Callable[[str], None] | None = None,
) -> list[tuple[Path, StoreGroup, dict[str, Any]]]:
    """Build or load every store of the plan (serially, in the parent) -> ``(directory, group, record)``."""
    out = []
    for group in scene.plan:
        start = time.perf_counter()
        store = build_or_load(
            scene.crystal,
            scene.refractive_index,
            group.members,
            n,
            base_dir=base_dir,
            sun_direction=scene.sun_direction,
            run_checks=run_checks,
            log=log,
        )
        key = store.spec.cache_key()
        directory = Path(base_dir) / key
        record = {
            "cache_key": key,
            "directory": str(directory),
            "store_members": [path_id_of(m) for m in group.members],
            "kept_events": int(len(store.events)),
            "build_parameters": store.spec.build_parameters(),
            "build_wall_clock_s": store.diagnostics.get("wall_clock_s"),
            "build_or_load_wall_clock_s": time.perf_counter() - start,
            "transports": [t.as_json() for t in group.transports],
        }
        if log is not None:
            log(f"store {key}: {len(store.events)} events, {record['build_or_load_wall_clock_s']:.1f} s")
        del store
        out.append((directory, group, record))
    return out


def _rank0_values(scene: BandSumScene, pixels: Sequence[tuple[int, int]]) -> tuple[dict[tuple[int, int], float], dict[str, Any]]:
    """Task 9 point mass of a rank-0 class: the sun pixel only (``path_class.render_class_pixel``)."""
    sun = sun_pixel(scene.render, scene.incident_direction)
    record: dict[str, Any] = {"sun_pixel": None if sun is None else list(sun)}
    if sun is None or tuple(sun) not in set(pixels):
        record["reason"] = "the sun pixel is outside the rendered pixels; the point mass contributes 0"
        return {}, record
    class_scene = build_class_scene(
        scene.path_class,
        incident_direction=scene.incident_direction,
        refractive_index=scene.refractive_index,
        crystal=scene.crystal,
        pose_density=scene.pose_density,
        render=scene.render,
        prescan_sample_count=scene.rank0_sample_count,
        prescan_rng_seed=scene.rank0_rng_seed,
    )
    result = render_class_pixel(class_scene, sun[0], sun[1], PixelOptions())
    record.update(result.provenance["rank0"])
    return {tuple(sun): float(result.value)}, record


def render_pixels(
    scene: BandSumScene, stores: Sequence[tuple[Mapping[str, np.ndarray], StoreGroup]], n: int, pixels: Sequence[tuple[int, int]]
) -> list[BandSumPixelResult]:
    """In-process band sums of ``pixels`` (rank-2 classes; the stores' events as ``S2Events.arrays()`` dicts)."""
    sun = np.asarray(scene.sun_direction, dtype=np.float64)
    return [class_band_sum_pixel(stores, sun, scene.pose_density, row, column, n, scene.render) for row, column in pixels]


_WORKER: dict[str, Any] = {}


def _worker_init(scene: BandSumScene, directories: Sequence[tuple[Path, StoreGroup]], n: int) -> None:
    _WORKER["scene"] = scene
    _WORKER["n"] = n
    _WORKER["stores"] = [(S2EventStore.load(d).events.arrays(), group) for d, group in directories]


def _worker_column(job: tuple[int, Sequence[int]]) -> tuple[list[BandSumPixelResult], float, float]:
    column, rows = job
    start = time.perf_counter()
    results = render_pixels(_WORKER["scene"], _WORKER["stores"], _WORKER["n"], [(row, column) for row in rows])
    return results, time.perf_counter() - start, max_rss_mb()


def render_band_sum_window(
    scene: BandSumScene,
    window: Window,
    n: int,
    *,
    workers: int = 1,
    base_dir: Path = DEFAULT_CACHE_DIR,
    run_checks: bool = True,
    log: Callable[[str], None] | None = None,
) -> tuple[list[BandSumPixelResult], dict[str, Any]]:
    """Render every pixel of ``window``: stores built or loaded first, then columns across ``workers``.

    Returns the pixel results and an execution record (store records, wall
    clocks, per-pixel time, peak RSS of the parent and of the workers).
    """
    if workers < 1:
        raise ValueError("workers must be positive")
    start = time.perf_counter()
    pixels = [(row, column) for column in window.column_range for row in window.row_range]
    execution: dict[str, Any] = {"workers": workers, "N": n, "store_base_dir": str(base_dir)}
    if scene.path_class.halo_map_rank == 0:
        values, record = _rank0_values(scene, pixels)
        results = [
            BandSumPixelResult(r, c, values.get((r, c), 0.0), float("nan"), float("nan"), 0, 0, 0.0, 0.0, 0.0)
            for r, c in pixels
        ]
        execution.update(rank0=record, stores=[], wall_clock_s=time.perf_counter() - start)
        return results, execution
    prepared = prepare_stores(scene, n, base_dir=base_dir, run_checks=run_checks, log=log)
    stores_s = time.perf_counter() - start
    execution["stores"] = [record for _, _, record in prepared]
    execution["stores_wall_clock_s"] = stores_s
    directories = [(d, group) for d, group, _ in prepared]
    jobs = [(column, list(window.row_range)) for column in window.column_range]
    render_start = time.perf_counter()
    results: list[BandSumPixelResult] = []
    column_seconds: list[float] = []
    worker_rss = 0.0
    if workers == 1:
        _worker_init(scene, directories, n)
        load_s = time.perf_counter() - render_start
        for job in jobs:
            chunk, seconds, rss = _worker_column(job)
            results.extend(chunk)
            column_seconds.append(seconds)
            worker_rss = max(worker_rss, rss)
        _WORKER.clear()
        execution["in_process_store_load_s"] = load_s
    else:
        context = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers, mp_context=context, initializer=_worker_init, initargs=(scene, directories, n)
        ) as executor:
            for index, (chunk, seconds, rss) in enumerate(executor.map(_worker_column, jobs)):
                results.extend(chunk)
                column_seconds.append(seconds)
                worker_rss = max(worker_rss, rss)
                if log is not None and (index + 1) % 25 == 0:
                    log(f"{index + 1}/{len(jobs)} columns, {time.perf_counter() - render_start:.1f} s")
    render_s = time.perf_counter() - render_start
    execution.update(
        render_wall_clock_s=render_s,
        wall_clock_s=time.perf_counter() - start,
        pixel_cpu_seconds=float(sum(column_seconds)),
        per_pixel_mean_s=float(sum(column_seconds)) / max(len(pixels), 1),
        max_rss_mb_parent=max_rss_mb(),
        max_rss_mb_worker=worker_rss,
    )
    return results, execution


# ---------------------------------------------------------------- output
def write_band_sum_strip(
    output_dir: Path,
    results: Sequence[BandSumPixelResult],
    *,
    scene: BandSumScene,
    n: int,
    window: Window,
    pose_density_block: Mapping[str, Any],
    execution: Mapping[str, Any],
    repo: Path | None = None,
) -> dict[str, Path]:
    """The :mod:`.strip_io` layout (``read_strip`` reads it) with a band-sum CSV and provenance."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    height, width = int(scene.render["height"]), int(scene.render["width"])
    values = np.zeros((height, width), dtype=np.float64)
    status = np.zeros((height, width), dtype=np.uint8)
    component_count = np.zeros((height, width), dtype=np.uint8)
    for r in results:
        lit = r.value != 0.0
        values[r.row, r.column] = r.value
        status[r.row, r.column] = STATUS_RENDERED | (STATUS_HAS_COMPONENT if lit else 0)
        component_count[r.row, r.column] = 1 if lit else 0
    files = write_binary_arrays(output_dir, values, status, component_count)
    files["pixels"] = output_dir / FILE_NAMES["pixels"]
    with files["pixels"].open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PIXEL_CSV_COLUMNS)
        writer.writeheader()
        for r in sorted(results, key=lambda r: (r.column, r.row)):
            writer.writerow(r.csv_row())
    rendered = (status & STATUS_RENDERED) != 0
    rendered_values = values[rendered]
    # A rank-0 class is a point mass (task 9), not a band sum: its K_eff is a hardcoded 0.0 placeholder,
    # not "zero effective samples, high noise" — excluded here so it does not pollute the noise diagnostic.
    is_rank0 = scene.path_class.halo_map_rank == 0
    k_eff = np.array([]) if is_rank0 else np.array([r.K_eff for r in results if r.value > 0.0])
    scene_json = scene_block(pose_density_block)
    scene_json["path"] = {"value": list(scene.path_class.representative), "provenance": "run-option"}
    scene_json["camera"] = {"value": {"lens": "linear", **dict(scene.render)}, "provenance": "run-option"}
    scene_json["image_shape"] = {"value": [height, width], "provenance": "run-option"}
    provenance = {
        "format": FORMAT_VERSION,
        "product": "S^2 band-sum render (band-averaged pixels, point light source)",
        "generator": {
            "package": "lumice_integral",
            "modules": ["band_sum", "s2_store", "strip_io"],
            "git_commit": git_commit(repo),
            "lumice_dependency": "none (independent implementation; Lumice is neither imported nor invoked)",
        },
        "scene": scene_json,
        "options": {
            "estimator": ESTIMATOR,
            "N": n,
            "path_class": scene.path_class.provenance(),
            "symmetry_transport": scene.transport,
            "store_plan": [g.as_json() for g in scene.plan],
            "k_eff_semantics": K_EFF_SEMANTICS,
            "stores": list(execution.get("stores", [])),
            "rank0": execution.get("rank0"),
            "class_value": (
                "sum of member contributions; K, K_rho_pos, K_eff count distinct store events, each event's "
                "contribution summed over the transports serving it (k_eff_semantics)"
            ),
        },
        "window": window.as_json(),
        "arrays": {
            "shape": [height, width],
            "order": "row-major, row 0 at the top of the image, column 0 at the left",
            "files": {
                "float64": {"name": FILE_NAMES["float64"], "dtype": "<f8", "sha256": sha256_of(files["float64"])},
                "float32": {"name": FILE_NAMES["float32"], "dtype": "<f4", "sha256": sha256_of(files["float32"])},
                "status": {"name": FILE_NAMES["status"], "dtype": "u1", "sha256": sha256_of(files["status"])},
                "component_count": {"name": FILE_NAMES["component_count"], "dtype": "u1", "sha256": sha256_of(files["component_count"])},
                "pixels": {"name": FILE_NAMES["pixels"], "sha256": sha256_of(files["pixels"]), "columns": list(PIXEL_CSV_COLUMNS)},
            },
            "status_bits": {"rendered": STATUS_RENDERED, "has_component": STATUS_HAS_COMPONENT},
            "status_bit_meanings": {
                "rendered": "pixel was computed (0 = outside the rendered window; its value is a placeholder 0)",
                "has_component": "the band sum is non-zero (component_count is 1 there, else 0)",
                "other bits": (
                    "always 0: a band sum has no fiber components, arcs, completeness or quadrature; "
                    "a 0 there is not a verified-completeness statement"
                ),
            },
            "value_semantics": (
                "band average over the pixel's deviation band [delta_lo, delta_hi] at the pixel-centre azimuth, "
                "sum of the class members; not a point value (differs from the Phase I point pixel at steep edges "
                "and inner caustics); sampling noise ~ 1/sqrt(K_eff), K_eff per pixel in pixels.csv; K, K_rho_pos "
                "and K_eff count distinct precomputed store events (options.k_eff_semantics = per_event): a "
                "symmetry transport reuses an event and adds no sample"
            ),
            "radiometric_normalization": "the Phase I pixel normalisation (roadmap section 4.2); not aligned with Lumice",
        },
        "summary": {
            "rendered_pixels": int(rendered.sum()),
            "pixels_with_light": int((values > 0.0).sum()),
            "value_min": float(rendered_values.min()) if rendered_values.size else None,
            "value_max": float(rendered_values.max()) if rendered_values.size else None,
            "value_mean": float(rendered_values.mean()) if rendered_values.size else None,
            "K_eff_median_lit": float(np.median(k_eff)) if k_eff.size else None,
            "K_eff_min_lit": float(np.min(k_eff)) if k_eff.size else None,
        },
        "execution": dict(execution),
        "environment": environment_block(),
    }
    files["provenance"] = output_dir / FILE_NAMES["provenance"]
    files["provenance"].write_text(json.dumps(provenance, indent=2) + "\n")
    return files


__all__ = [
    "ESTIMATOR",
    "FORMAT_VERSION",
    "K_EFF_SEMANTICS",
    "PIXEL_CSV_COLUMNS",
    "BandSumPixelResult",
    "BandSumScene",
    "StoreGroup",
    "Transport",
    "band_contributions",
    "band_poses",
    "band_rotations",
    "band_sum_estimate",
    "band_sum_pixel",
    "class_band_sum_pixel",
    "kish_k_eff",
    "pixel_band",
    "prepare_stores",
    "render_band_sum_window",
    "render_pixels",
    "single_path_class",
    "store_plan",
    "write_band_sum_strip",
]
