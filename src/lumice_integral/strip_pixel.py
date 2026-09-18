"""One pixel of the ch06 strip: candidates -> one trace each -> quadrature -> sum.

:func:`render_pixel` is the pure per-pixel pipeline behind the strip image
driver (:mod:`.strip_driver` schedules it, :mod:`.strip_io` writes it out).
For one target direction ``d`` it

1. runs :func:`.discovery.discover_components` once: the candidate pool is
   the scene's :class:`.prescan.PrescanTable` query plus ``warm_seeds`` (the
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
always queries the prescan table, so the blind spot the old hot-start chain
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
from typing import Any, Mapping, Sequence

import jax.numpy as jnp
import numpy as np

from .camera import linear_pixel_outgoing_direction
from .canonical_scene import (
    CANONICAL_REFRACTIVE_INDEX,
    CANONICAL_RENDER,
    canonical_crystal,
    canonical_incident_direction,
    canonical_pose_density,
)
from .continuation import ContinuationOptions, FiberProblem
from .discovery import DISCOVERY_EVENT_NAMES, DiscoveredComponent, discover_components, retarget_problem
from .geometry import HexPrism
from .optics import path_3_5_problem
from .pose_density import ZenithGaussianPoseDensity
from .prescan import DEFAULT_RNG_SEED, DEFAULT_SAMPLE_COUNT, PrescanTable, build_prescan_table
from .quadrature import ResampleOptions, integrate_fiber_resampled
from .weights import build_3_5_weight_evaluators

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
    """Scene constants, the prescan table and the two shared problem templates of one process.

    ``discovery_template`` is weightless (the traces must not pay for weight
    evaluation at every accepted pose); ``production_template`` carries the
    four named weights the quadrature evaluates on its own grid.  Both share
    the same evaluator closures, so every per-pixel problem derived by
    :func:`.discovery.retarget_problem` hits the same ``jax.jit`` caches.
    ``prescan_table`` is the scene-level :class:`.prescan.PrescanTable` every
    pixel queries (built once per scene, read-only afterwards).
    """

    incident_direction: np.ndarray
    refractive_index: float
    crystal: HexPrism
    pose_density: ZenithGaussianPoseDensity
    render: Mapping[str, Any]
    discovery_template: FiberProblem
    production_template: FiberProblem
    prescan_table: PrescanTable

    def __post_init__(self) -> None:
        table = self.prescan_table
        if not np.array_equal(table.incident_direction, np.asarray(self.incident_direction, dtype=np.float64)):
            raise ValueError("prescan_table incident direction does not match the scene")
        if table.refractive_index != float(self.refractive_index):
            raise ValueError("prescan_table refractive index does not match the scene")

    @property
    def width(self) -> int:
        return int(self.render["width"])

    @property
    def height(self) -> int:
        return int(self.render["height"])


def canonical_strip_scene(
    *,
    prescan_table: PrescanTable | None = None,
    prescan_sample_count: int = DEFAULT_SAMPLE_COUNT,
    prescan_rng_seed: int = DEFAULT_RNG_SEED,
) -> StripScene:
    """The ch06 canonical scene (``docs/ch06-reference-fixture.md`` section 3.3).

    ``prescan_table`` (a table the driver built or loaded once) is used as is;
    otherwise one is built here from ``prescan_sample_count`` /
    ``prescan_rng_seed`` (in-process rendering and tests).
    """
    incident = canonical_incident_direction()
    crystal = canonical_crystal()
    pose_density = canonical_pose_density()
    if prescan_table is None:
        prescan_table = build_prescan_table(
            incident, CANONICAL_REFRACTIVE_INDEX, sample_count=prescan_sample_count, rng_seed=prescan_rng_seed
        )
    template = path_3_5_problem(
        jnp.asarray(np.eye(3)),
        jnp.asarray(incident),
        target_direction=jnp.asarray(pixel_target(CANONICAL_RENDER, 0, 0)),
        refractive_index=jnp.asarray(CANONICAL_REFRACTIVE_INDEX, dtype=jnp.float64),
    )
    evaluators = build_3_5_weight_evaluators(
        incident_direction=incident,
        refractive_index=CANONICAL_REFRACTIVE_INDEX,
        crystal=crystal,
        pose_density=pose_density,
    )
    return StripScene(
        incident_direction=incident,
        refractive_index=CANONICAL_REFRACTIVE_INDEX,
        crystal=crystal,
        pose_density=pose_density,
        render=dict(CANONICAL_RENDER),
        discovery_template=template,
        production_template=replace(template, weight_evaluators=evaluators),
        prescan_table=prescan_table,
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

    The prescan sampling (sample count, seed) is a scene policy, not a pixel
    one: it lives in :class:`.strip_driver.PrescanBuildOptions` and the
    resulting :attr:`StripScene.prescan_table`.  There is one step budget:
    ``continuation.maximum_accepted_steps`` (every trace is a production
    trace).
    """

    angle_tolerance_deg: float = 2.0
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
            angle_tolerance_deg=self.angle_tolerance_deg,
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
        scene.crystal,
        scene.prescan_table,
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
    "canonical_strip_scene",
    "pixel_target",
    "render_pixel",
    "subpixel_targets",
]
