"""One pixel of the ch06 strip: candidates -> one trace each -> quadrature -> sum.

:func:`render_pixel` is the pure per-pixel pipeline behind the strip image
driver (:mod:`.strip_driver` schedules it, :mod:`.strip_io` writes it out).
For one target direction ``d`` it

1. runs :func:`.discovery.discover_components` once: the candidate pool is
   the band of the scene's :class:`.s2_store.StoreSeeds` plus ``warm_seeds`` (the
   converged poses of any neighbouring pixels the caller chooses; they only
   warm the Gauss-Newton start of their cluster and are neither traced on
   their own nor a source of completeness); every cluster representative is
   corrected, gated, deduplicated against the components already accepted by
   SO(3) distance, and traced *once* with the production
   :class:`.continuation.ContinuationOptions`; a trace ending on a named event
   is traced backward from the same seed and stitched into an open arc
   (task-pixel-pipeline-v2);
2. integrates every component (closed loop or arc) with
   :func:`.quadrature.integrate_fiber_resampled` on the production problem
   (the four named weights) retargeted to ``d``;
3. sums the component values and error estimates linearly (the error sum is a
   conservative bound, not a root-sum-square).

There is no separate hot-start path and no periodic cold check: every pixel
always queries the seed store, so the blind spot the old hot-start chain
had (a component with no seed in the neighbour) does not exist here, and the
cold check that bounded it has nothing left to bound.

``completeness`` is the pixel-level procedural signal: ``"complete"`` iff every
admissible candidate converged (closed or arc, no ``incomplete`` candidate)
and every component's quadrature was available.  It is *not* a certificate
that every connected component of ``X_(3-5, d)`` was found (that remains an
open item of ``docs/phase1-math-contract.md`` section 8); ``value`` is always
the partial sum of the components that were integrated, never ``NaN`` and
never silently substituted.  An arc is a legitimate partial contribution (the
integrand is continuous to zero at a TIR boundary, and truncation estimates
at both ends are reported per component); consumers must read the status
layer, not the raw value.

Nothing here imports or calls Lumice.
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import jax.numpy as jnp
import numpy as np

from .camera import incident_direction_from_sun, linear_pixel_outgoing_direction
from .canonical_scene import (
    CANONICAL_REFRACTIVE_INDEX,
    CANONICAL_RENDER,
    canonical_crystal,
    canonical_pose_density,
    canonical_sun_direction,
)
from .continuation import ContinuationOptions, FiberProblem
from .discovery import DISCOVERY_EVENT_NAMES, DiscoveredComponent, discover_components, retarget_problem
from .geometry import HexPrism
from .optics import PATH_3_5_FACES, normalize_faces, path_id_of, path_problem, problem_path_label
from .pose_density import PoseDensity
from .quadrature import ResampleOptions, integrate_fiber_resampled
from .s2_store import DEFAULT_SEED_STORE_N, S2EventStore, StoreSeeds, build_event_store, build_or_load, crystal_description
from .weights import build_path_weight_evaluators

Completeness = str  # "complete" | "unknown"

# Status-layer bits (uint8), one per pixel; see strip_io.STATUS_BITS for the
# self-describing export.  ``RENDERED`` is the only bit a consumer may rely on
# to tell "computed zero" from "outside the rendered window".
STATUS_RENDERED = 1
STATUS_UNKNOWN_COMPLETENESS = 2
STATUS_HAS_COMPONENT = 4
STATUS_HAS_ARC = 8
STATUS_QUADRATURE_UNAVAILABLE = 16
STATUS_NODE_COUNT_EXHAUSTED = 32

EVENT_NAMES = (
    *DISCOVERY_EVENT_NAMES,
    "incomplete_candidate",
    "quadrature_unavailable",
    "quadrature_node_count_exhausted",
    "quadrature_non_finite_nodes",
)
# Per-pixel wall-clock stages (seconds): ``discovery_s`` is the whole
# discover_components call, ``trace_s`` the part of it inside trace_fiber,
# ``quadrature_s`` the integrations; ``total_s`` wraps everything.
STAGE_NAMES = ("discovery_s", "trace_s", "quadrature_s", "total_s")


@dataclass(frozen=True)
class StripScene:
    """Scene constants, the seed store and the two shared problem templates of one process.

    ``discovery_template`` is weightless (the traces must not pay for weight
    evaluation at every accepted pose); ``production_template`` carries the
    four named weights the quadrature evaluates on its own grid.  Both share
    the same evaluator closures, so every per-pixel problem derived by
    :func:`.discovery.retarget_problem` hits the same ``jax.jit`` caches.
    ``seeds`` is the scene-level :class:`.s2_store.StoreSeeds` every pixel
    queries (a store built or loaded once per scene, read-only afterwards)
    and the single source of the scene's face sequence (:attr:`faces`); its
    incident direction, refractive index and crystal must be the scene's,
    and both templates must be problems of its path.
    """

    incident_direction: np.ndarray
    refractive_index: float
    crystal: HexPrism
    pose_density: PoseDensity
    render: Mapping[str, Any]
    discovery_template: FiberProblem
    production_template: FiberProblem
    seeds: StoreSeeds

    def __post_init__(self) -> None:
        seeds = self.seeds
        if not np.array_equal(seeds.incident_direction, np.asarray(self.incident_direction, dtype=np.float64)):
            raise ValueError("seeds incident direction does not match the scene")
        if seeds.refractive_index != float(self.refractive_index):
            raise ValueError("seeds refractive index does not match the scene")
        if seeds.store.spec.crystal != crystal_description(self.crystal):
            raise ValueError(f"seed store crystal {seeds.store.spec.crystal} is not the scene's {crystal_description(self.crystal)}")
        expected = problem_path_label(seeds.faces, self.refractive_index)
        for name in ("discovery_template", "production_template"):
            if getattr(self, name).path != expected:
                raise ValueError(f"{name} path {getattr(self, name).path!r} does not match the seeds' {expected!r}")

    @property
    def faces(self) -> tuple[int, ...]:
        return self.seeds.faces

    @property
    def path_id(self) -> str:
        return self.seeds.path_id

    @property
    def width(self) -> int:
        return int(self.render["width"])

    @property
    def height(self) -> int:
        return int(self.render["height"])


def build_strip_scene(
    faces: Sequence[int],
    *,
    sun_direction: np.ndarray,
    refractive_index: float,
    crystal: HexPrism,
    pose_density: PoseDensity,
    render: Mapping[str, Any],
    seeds: StoreSeeds | None = None,
    seed_store_n: int = DEFAULT_SEED_STORE_N,
    seed_store_cache_dir: Path | None = None,
) -> StripScene:
    """Assemble the scene of one face sequence (single authority; :func:`canonical_strip_scene` is its ch06 binding).

    ``seeds`` (a store the driver or a class scene built or loaded once) is
    used as is and must seed ``faces``; otherwise the store of ``faces`` alone
    is made here by :func:`seed_store`.  ``sun_direction`` is the public
    ``s_hat`` (toward the sun); the scene's :attr:`StripScene.incident_direction`
    is the propagation ``-s_hat`` of the optics
    (:func:`.camera.incident_direction_from_sun`).  The two templates are one
    :func:`.optics.path_problem` of ``faces`` (weightless for discovery, with
    :func:`.weights.build_path_weight_evaluators` for production).
    """
    faces = normalize_faces(faces)
    sun = np.asarray(sun_direction, dtype=np.float64)
    incident = incident_direction_from_sun(sun)
    index = float(refractive_index)
    if seeds is None:
        store = seed_store(crystal, index, (faces,), seed_store_n, cache_dir=seed_store_cache_dir)
        seeds = StoreSeeds(store, faces, sun)
    elif seeds.faces != faces:
        raise ValueError(f"seeds are of path {seeds.path_id!r}, not {path_id_of(faces)!r}")
    template = path_problem(
        jnp.asarray(np.eye(3)),
        faces,
        jnp.asarray(incident),
        target_direction=jnp.asarray(pixel_target(render, 0, 0)),
        refractive_index=jnp.asarray(index, dtype=jnp.float64),
    )
    evaluators = build_path_weight_evaluators(
        faces=faces,
        incident_direction=incident,
        refractive_index=index,
        crystal=crystal,
        pose_density=pose_density,
    )
    return StripScene(
        incident_direction=incident,
        refractive_index=index,
        crystal=crystal,
        pose_density=pose_density,
        render=dict(render),
        discovery_template=template,
        production_template=replace(template, weight_evaluators=evaluators),
        seeds=seeds,
    )


def seed_store(
    crystal: HexPrism,
    refractive_index: float,
    members: Sequence[Sequence[int]],
    n: int = DEFAULT_SEED_STORE_N,
    *,
    cache_dir: Path | None = None,
    log: Callable[[str], None] | None = None,
) -> S2EventStore:
    """The event store the seeds of ``members`` (one ``Phi`` group) come from.

    ``cache_dir=None`` builds it in memory without the self-checks (tests and
    in-process rendering; ~1 s at the default ``N``); otherwise
    :func:`.s2_store.build_or_load` builds it once into ``cache_dir`` with the
    self-checks and later runs load it (SHA-256 checked).
    """
    if cache_dir is None:
        return build_event_store(crystal, refractive_index, members, n, run_checks=False)
    return build_or_load(crystal, refractive_index, members, n, base_dir=cache_dir, log=log)


def canonical_strip_scene(
    *,
    seeds: StoreSeeds | None = None,
    seed_store_n: int = DEFAULT_SEED_STORE_N,
    seed_store_cache_dir: Path | None = None,
    pose_density: PoseDensity | None = None,
    crystal: HexPrism | None = None,
) -> StripScene:
    """The ch06 canonical scene (``docs/ch06-reference-fixture.md`` section 3.3), path 3-5.

    ``seeds`` (a store the driver built or loaded once) is used as is;
    otherwise :func:`seed_store` makes one of ``seed_store_n`` points (in
    memory unless ``seed_store_cache_dir``).  ``pose_density`` replaces the
    canonical column density (``canonical_pose_density()``) by another
    :mod:`.pose_density` family for diagnostics; discovery, tracing and the
    seed store do not depend on it (only the ``rho_pose`` weight does), so
    the same store serves every family.  ``crystal`` likewise replaces
    ``canonical_crystal()``: the ``entry_measure`` weight, the admissibility
    gate of discovery and the store's ``w > 0`` filter depend on it, so a
    supplied ``seeds`` must be of that crystal; tests use it to keep
    baselines recorded on the pre-2026-09-20 ``h/a = 1`` crystal.
    """
    return build_strip_scene(
        PATH_3_5_FACES,
        sun_direction=canonical_sun_direction(),
        refractive_index=CANONICAL_REFRACTIVE_INDEX,
        crystal=canonical_crystal() if crystal is None else crystal,
        pose_density=canonical_pose_density() if pose_density is None else pose_density,
        render=CANONICAL_RENDER,
        seeds=seeds,
        seed_store_n=seed_store_n,
        seed_store_cache_dir=seed_store_cache_dir,
    )


def pixel_target(render: Mapping[str, Any], row: float, column: float) -> np.ndarray:
    """Solver target ``d`` of the pixel centre; fractional ``row``/``column`` shift it."""
    return linear_pixel_outgoing_direction(row, column, **render)


def subpixel_targets(render: Mapping[str, Any], row: int, column: int, grid: int) -> list[np.ndarray]:
    """``grid x grid`` sub-pixel centre directions of pixel ``(row, column)``.

    Sub-pixel ``(i, j)`` is centred at ``(row + (i + 1/2) / grid, column + (j + 1/2) / grid)``
    in continuous pixel coordinates, which :func:`.camera.linear_pixel_outgoing_direction`
    expresses as the integer offset plus ``1/2``.
    """
    if grid < 1:
        raise ValueError("subpixel grid must be positive")
    return [
        pixel_target(render, row + (i + 0.5) / grid - 0.5, column + (j + 0.5) / grid - 0.5)
        for i in range(grid)
        for j in range(grid)
    ]


@dataclass(frozen=True)
class PixelOptions:
    """Every numerical policy of one pixel, in one place (exported to provenance).

    The seed store's size is a scene policy, not a pixel one: it lives in
    :class:`.strip_driver.SeedStoreOptions` and the resulting
    :attr:`StripScene.seeds`; the band half-width of its query is a pixel
    one.  There is one step budget:
    ``continuation.maximum_accepted_steps`` (every trace is a production
    trace).
    """

    # Half-width of the store band around the target's deviation (discovery.discover_components).
    band_half_width_deg: float = 0.2
    cluster_radius_rad: float = 0.3
    # SO(3) geodesic distance below which a corrected candidate seed is the
    # same component as an already accepted curve (``discovery.distance_to_curve``);
    # the continuation's ``closure_distance`` by default (same scale).
    distance_threshold: float = ContinuationOptions.closure_distance
    continuation: ContinuationOptions = field(default_factory=ContinuationOptions)
    # The resampled fixed-grid quadrature at its calibrated defaults
    # (``relative_tolerance=1e-4``: within 1e-4 of the retired adaptive
    # rtol=1e-8 integrator on the four Step 5 fixtures at 13-26 ms per fiber,
    # task-resample-and-integrate).  The float32 image copy resolves ~1e-7
    # relative, so the image is now quadrature-limited at 1e-4, visibly.
    quadrature: ResampleOptions = field(default_factory=ResampleOptions)

    def discovery_kwargs(self) -> dict[str, Any]:
        """Keyword arguments of :func:`.discovery.discover_components`."""
        return dict(
            continuation=self.continuation,
            band_half_width_deg=self.band_half_width_deg,
            cluster_radius_rad=self.cluster_radius_rad,
            distance_threshold=self.distance_threshold,
        )


@dataclass(frozen=True)
class ComponentRecord:
    """One integrated component (scalars only; picklable across worker processes).

    ``kind`` is ``"closed"`` or ``"arc"``; ``reason`` the terminal event of the
    curve and ``start_reason`` the backward end's for an arc (``""`` for a
    closed loop — the picklable/CSV-safe encoding of the upstream
    ``DiscoveredComponent.start_reason: TerminationReason | None``, not a
    semantic downgrade from ``Optional``).  ``start_truncation_estimate``/``end_truncation_estimate``
    are the Haar-converted open-arc truncation estimates of
    :class:`.quadrature.ResampledQuadratureResult` (``nan`` when not
    applicable; never added to ``value``).
    """

    seed: np.ndarray
    kind: str
    arclength: float
    pose_count: int
    status: str
    reason: str
    start_reason: str
    quadrature_status: str
    value: float
    error_estimate: float
    start_truncation_estimate: float
    end_truncation_estimate: float
    node_count: int
    refinement_rounds: int
    node_count_exhausted: bool
    non_finite_node_count: int

    @property
    def integrated(self) -> bool:
        return self.quadrature_status == "available"


@dataclass(frozen=True)
class PixelResult:
    """Everything the strip driver keeps for one pixel (see module docstring)."""

    row: int
    column: int
    value: float
    error_estimate: float
    components: tuple[ComponentRecord, ...]
    completeness: Completeness
    incomplete_count: int
    pool_count: int
    extra_seed_count: int
    raw_cluster_count: int
    admissible_count: int
    events: Mapping[str, int]
    timings: Mapping[str, float]

    @property
    def component_count(self) -> int:
        return len(self.components)

    @property
    def arc_count(self) -> int:
        return sum(component.kind == "arc" for component in self.components)

    @property
    def warm_seeds(self) -> tuple[np.ndarray, ...]:
        """Converged poses a neighbouring pixel can warm its candidates with (integrated components only)."""
        return tuple(component.seed for component in self.components if component.integrated)

    @property
    def status_bits(self) -> int:
        bits = STATUS_RENDERED
        if self.completeness != "complete":
            bits |= STATUS_UNKNOWN_COMPLETENESS
        if any(component.integrated for component in self.components):
            bits |= STATUS_HAS_COMPONENT
        if any(component.integrated and component.kind == "arc" for component in self.components):
            bits |= STATUS_HAS_ARC
        if self.events.get("quadrature_unavailable", 0):
            bits |= STATUS_QUADRATURE_UNAVAILABLE
        if self.events.get("quadrature_node_count_exhausted", 0):
            bits |= STATUS_NODE_COUNT_EXHAUSTED
        return bits


def _integrate_component(
    scene: StripScene,
    target: np.ndarray,
    component: DiscoveredComponent,
    options: PixelOptions,
    events: Counter,
    timings: Counter,
) -> ComponentRecord:
    problem = retarget_problem(scene.production_template, target, component.seed)
    start = time.perf_counter()
    quadrature = integrate_fiber_resampled(problem, component.result, options.quadrature)
    timings["quadrature_s"] += time.perf_counter() - start
    if quadrature.status != "available":
        events["quadrature_unavailable"] += 1
    if quadrature.node_count_exhausted:
        events["quadrature_node_count_exhausted"] += 1
    if quadrature.non_finite_node_count:
        events["quadrature_non_finite_nodes"] += 1
    return ComponentRecord(
        seed=np.asarray(component.seed, dtype=np.float64),
        kind=component.kind,
        arclength=float(component.arclength),
        pose_count=int(len(component.result.poses)),
        status=component.status.value,
        reason=component.reason.value,
        start_reason=component.start_reason.value if component.start_reason is not None else "",
        quadrature_status=quadrature.status,
        value=float(quadrature.value) if quadrature.status == "available" else 0.0,
        error_estimate=float(quadrature.error_estimate) if quadrature.status == "available" else 0.0,
        start_truncation_estimate=float(quadrature.start_endpoint_truncation_estimate),
        end_truncation_estimate=float(quadrature.endpoint_truncation_estimate),
        node_count=int(quadrature.node_count),
        refinement_rounds=int(quadrature.refinement_rounds),
        node_count_exhausted=bool(quadrature.node_count_exhausted),
        non_finite_node_count=int(quadrature.non_finite_node_count),
    )


def render_pixel(
    scene: StripScene,
    row: int,
    column: int,
    options: PixelOptions,
    *,
    target: np.ndarray | None = None,
    warm_seeds: Sequence[np.ndarray] | None = None,
) -> PixelResult:
    """Run the full pipeline for pixel ``(row, column)`` (module docstring).

    ``target`` overrides the pixel-centre direction (sub-pixel sampling);
    ``warm_seeds`` are converged poses of any neighbouring pixels (typically
    :attr:`PixelResult.warm_seeds` of the pixel above), added to the candidate
    pool as Gauss-Newton starts.
    """
    target = pixel_target(scene.render, row, column) if target is None else np.asarray(target, dtype=np.float64)
    events: Counter = Counter()
    timings: Counter = Counter({name: 0.0 for name in STAGE_NAMES})
    total_start = time.perf_counter()

    start = time.perf_counter()
    discovered = discover_components(
        target,
        scene.seeds,
        template=scene.discovery_template,
        extra_seeds=tuple(warm_seeds) if warm_seeds else (),
        **options.discovery_kwargs(),
    )
    timings["discovery_s"] += time.perf_counter() - start
    timings["trace_s"] += discovered.trace_seconds
    events.update(discovered.events)
    events["incomplete_candidate"] += discovered.incomplete_count

    components = tuple(
        _integrate_component(scene, target, component, options, events, timings)
        for component in discovered.components
    )
    timings["total_s"] = time.perf_counter() - total_start

    complete = discovered.incomplete_count == 0 and not events["quadrature_unavailable"]
    return PixelResult(
        row=int(row),
        column=int(column),
        value=float(sum(component.value for component in components)),
        error_estimate=float(sum(component.error_estimate for component in components)),
        components=components,
        completeness="complete" if complete else "unknown",
        incomplete_count=discovered.incomplete_count,
        pool_count=discovered.pool_count,
        extra_seed_count=discovered.extra_seed_count,
        raw_cluster_count=discovered.raw_cluster_count,
        admissible_count=discovered.admissible_count,
        events={name: int(events[name]) for name in EVENT_NAMES},
        timings={name: float(timings[name]) for name in STAGE_NAMES},
    )


__all__ = [
    "EVENT_NAMES",
    "ComponentRecord",
    "PixelOptions",
    "PixelResult",
    "STATUS_HAS_ARC",
    "STATUS_HAS_COMPONENT",
    "STATUS_NODE_COUNT_EXHAUSTED",
    "STATUS_QUADRATURE_UNAVAILABLE",
    "STATUS_RENDERED",
    "STATUS_UNKNOWN_COMPLETENESS",
    "STAGE_NAMES",
    "StripScene",
    "build_strip_scene",
    "canonical_strip_scene",
    "pixel_target",
    "render_pixel",
    "seed_store",
    "subpixel_targets",
]
