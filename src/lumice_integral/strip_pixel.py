"""One pixel of the ch06 strip: discovery -> production trace -> quadrature -> sum.

:func:`render_pixel` is the pure per-pixel pipeline behind the strip image
driver (:mod:`.strip_driver` schedules it, :mod:`.strip_io` writes it out).
For one target direction ``d`` it

1. finds the 3-5 fiber components reaching ``d`` -- either by hot-starting
   every component of a neighbouring pixel (:func:`.discovery.hot_start_component`
   with the production step budget, gated by
   :func:`.discovery.detect_arclength_jump`) or, when no neighbour is given, a
   hot start fails, a jump is detected, or a cold check is requested, by the
   full prescan :func:`.discovery.discover_components` with the small discovery
   budget, whose ``incomplete`` candidates are then retraced once with the
   production budget (rows just below the 22-degree inner-edge caustic close
   only after 300-1300 accepted steps; task-strip-image-driver Step 1);
2. retraces every closed component with the production
   :class:`.continuation.ContinuationOptions` and the four named weights
   (the discovery traces are weightless and budget-limited by design);
3. integrates each production trace with :func:`.quadrature.integrate_fiber`;
4. sums the component values and error estimates linearly (the error sum is a
   conservative bound, not a root-sum-square).

Completeness vocabulary (two different signals, deliberately named apart):

- ``discovery_completeness`` is :attr:`.discovery.ComponentDiscoveryResult.completeness`
  of the cold prescan that ran for this pixel (``"not-run"`` on a pure hot
  start): procedural, "no unclassified candidate in this pool".
- ``completeness`` is the pixel-level procedural signal: ``"complete"`` iff no
  step above produced evidence of a missing or unusable component (no
  incomplete candidate in the discovery that was finally used, every
  production trace closed with the discovery arclength, every quadrature
  available).  A hot-start failure or arclength jump that fell back to a
  successful cold prescan does not by itself make the pixel ``"unknown"``;
  it is reported through ``seed_source``/``events`` and the status bits.
  It is *not* a certificate that every connected component of ``X_(3-5, d)``
  was found; ``value`` is always the partial sum of the
  components that were integrated, never ``NaN`` and never silently
  substituted.  Consumers must read the status layer, not the raw value.

  Known bounded blind spot (code-review round 1 Major 2): :func:`_hot_start_all`
  only revisits the neighbour's own seeds, one hot start per entry of
  ``previous``, so a component that first becomes admissible between two rows
  (a caustic/topology branch) has no matching seed and produces no event; the
  hot-started pixel is reported ``"complete"`` even though a real component
  was missed.  A full supplementary discovery on every hot-started pixel
  would pay the same ~0.4 s JIT/prescan tax the ``template`` reuse in
  :mod:`.discovery` exists to avoid (task-strip-image-driver Step 0 DECISION,
  progress.md 20:40), so it is deliberately not attempted here.  The only
  mitigation is :data:`.strip_driver.DriverOptions.cold_check_interval`'s
  periodic cold prescan, which bounds the miss to at most
  ``cold_check_interval - 1`` rows before self-healing (see
  ``test_hot_start_chain_cannot_discover_a_component_absent_from_the_previous_seeds``
  in ``tests/test_strip_pixel.py``, which pins this exact bound rather than
  leaving it unexercised); it does not retroactively fix rows already
  emitted inside that window.  A caustic-heavy region that needs a tighter
  bound should lower ``cold_check_interval``.

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
from .continuation import ContinuationOptions, FiberProblem, FiberStatus, trace_fiber
from .discovery import (
    ComponentDiscoveryResult,
    DiscoveredComponent,
    IncompleteCandidate,
    dedup_components,
    detect_arclength_jump,
    discover_components,
    hot_start_component,
    retarget_problem,
)
from .geometry import HexPrism
from .optics import path_3_5_problem
from .pose_density import ZenithGaussianPoseDensity
from .quadrature import QuadratureOptions, integrate_fiber
from .weights import build_3_5_weight_evaluators

Completeness = str  # "complete" | "unknown"

# Status-layer bits (uint8), one per pixel; see strip_io.STATUS_BITS for the
# self-describing export.  ``RENDERED`` is the only bit a consumer may rely on
# to tell "computed zero" from "outside the rendered window".
STATUS_RENDERED = 1
STATUS_UNKNOWN_COMPLETENESS = 2
STATUS_HAS_COMPONENT = 4
STATUS_COLD_DISCOVERY = 8
STATUS_ARCLENGTH_JUMP = 16
STATUS_PRODUCTION_FAILURE = 32
STATUS_DEPTH_EXHAUSTED = 64
STATUS_COLD_CHECK_MISMATCH = 128

EVENT_NAMES = (
    "incomplete_retry",
    "incomplete_recovered",
    "incomplete_candidate",
    "hot_start_inadmissible",
    "hot_start_incomplete",
    "hot_start_merged",
    "arclength_jump",
    "cold_check",
    "cold_check_mismatch",
    "production_not_closed",
    "production_arclength_mismatch",
    "quadrature_unavailable",
    "quadrature_depth_exhausted",
)
# Per-pixel wall-clock stages (seconds); ``total_s`` wraps the other four.
STAGE_NAMES = ("hot_start_s", "cold_discovery_s", "production_s", "quadrature_s", "total_s")


@dataclass(frozen=True)
class StripScene:
    """Scene constants plus the two shared problem templates of one process.

    ``discovery_template`` is weightless (discovery traces must not pay for
    weight evaluation); ``production_template`` carries the four named weights.
    Both share the same evaluator closures, so every per-pixel problem derived
    by :func:`.discovery.retarget_problem` hits the same ``jax.jit`` caches.
    """

    incident_direction: np.ndarray
    refractive_index: float
    crystal: HexPrism
    pose_density: ZenithGaussianPoseDensity
    render: Mapping[str, Any]
    discovery_template: FiberProblem
    production_template: FiberProblem

    @property
    def width(self) -> int:
        return int(self.render["width"])

    @property
    def height(self) -> int:
        return int(self.render["height"])


def canonical_strip_scene() -> StripScene:
    """The ch06 canonical scene (``docs/ch06-reference-fixture.md`` section 3.3)."""
    incident = canonical_incident_direction()
    crystal = canonical_crystal()
    pose_density = canonical_pose_density()
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
    """Every numerical policy of one pixel, in one place (exported to provenance)."""

    rng_seed: int = 20260916
    prescan_samples: int = 400_000
    discovery_step_budget: int = 250
    # Budget of the one retrace of every cold-discovery ``incomplete`` candidate
    # and of every hot start; ``None`` means the production budget.
    retry_step_budget: int | None = None
    angle_tolerance_deg: float = 2.0
    cluster_radius_rad: float = 0.3
    arclength_rtol: float = 1e-3
    jump_relative_threshold: float = 0.2
    continuation: ContinuationOptions = field(default_factory=ContinuationOptions)
    # Image default: 1e-6 (the float32 copy resolves ~1e-7 relative) and no
    # order-estimate pass; the single-pixel fixture defaults (1e-8, levels 2)
    # cost ~4x more per fiber for a diagnostic the image does not consume.
    quadrature: QuadratureOptions = field(
        default_factory=lambda: QuadratureOptions(
            relative_tolerance=1e-6, convergence_order_levels=0
        )
    )

    @property
    def effective_retry_step_budget(self) -> int:
        if self.retry_step_budget is None:
            return self.continuation.maximum_accepted_steps
        return self.retry_step_budget

    def discovery_kwargs(self) -> dict[str, Any]:
        return dict(
            rng_seed=self.rng_seed,
            prescan_samples=self.prescan_samples,
            discovery_step_budget=self.discovery_step_budget,
            angle_tolerance_deg=self.angle_tolerance_deg,
            cluster_radius_rad=self.cluster_radius_rad,
            arclength_rtol=self.arclength_rtol,
        )


@dataclass(frozen=True)
class HotSeed:
    """A converged component of a neighbouring pixel: what a hot start needs."""

    seed: np.ndarray
    arclength: float


@dataclass(frozen=True)
class ComponentRecord:
    """One integrated component (scalars only; picklable across worker processes)."""

    seed: np.ndarray
    discovery_arclength: float
    production_status: str
    production_reason: str
    production_arclength: float
    production_pose_count: int
    quadrature_status: str
    value: float
    error_estimate: float
    refinements: int
    node_count: int
    maximum_depth_reached: int
    depth_exhausted_edge_count: int

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
    discovery_completeness: str
    seed_source: str
    incomplete_count: int
    pool_count: int
    raw_cluster_count: int
    admissible_count: int
    events: Mapping[str, int]
    timings: Mapping[str, float]

    @property
    def component_count(self) -> int:
        return len(self.components)

    @property
    def hot_seeds(self) -> tuple[HotSeed, ...]:
        """Seeds a neighbouring pixel can hot-start from (integrated components only)."""
        return tuple(
            HotSeed(component.seed, component.discovery_arclength)
            for component in self.components
            if component.integrated
        )

    @property
    def status_bits(self) -> int:
        bits = STATUS_RENDERED
        if self.completeness != "complete":
            bits |= STATUS_UNKNOWN_COMPLETENESS
        if any(component.integrated for component in self.components):
            bits |= STATUS_HAS_COMPONENT
        if self.seed_source != "hot" or self.events.get("cold_check", 0):
            bits |= STATUS_COLD_DISCOVERY
        if self.events.get("arclength_jump", 0):
            bits |= STATUS_ARCLENGTH_JUMP
        if self.events.get("production_not_closed", 0) or self.events.get("quadrature_unavailable", 0):
            bits |= STATUS_PRODUCTION_FAILURE
        if self.events.get("quadrature_depth_exhausted", 0):
            bits |= STATUS_DEPTH_EXHAUSTED
        if self.events.get("cold_check_mismatch", 0):
            bits |= STATUS_COLD_CHECK_MISMATCH
        return bits


def _hot_start_all(
    scene: StripScene,
    target: np.ndarray,
    previous: Sequence[HotSeed],
    options: PixelOptions,
    events: Counter,
) -> ComponentDiscoveryResult | None:
    """Hot-start every neighbour component; ``None`` means fall back to cold discovery.

    Result component count is always ``len(previous)``: a genuinely new
    component with no matching seed is never found here (module docstring
    "known bounded blind spot").
    """
    records: list[DiscoveredComponent] = []
    for seed in previous:
        record = hot_start_component(
            seed.seed,
            target,
            scene.incident_direction,
            scene.refractive_index,
            scene.crystal,
            discovery_step_budget=options.effective_retry_step_budget,
            template=scene.discovery_template,
        )
        if record is None:
            events["hot_start_inadmissible"] += 1
            return None
        if isinstance(record, IncompleteCandidate):
            events["hot_start_incomplete"] += 1
            return None
        if detect_arclength_jump(
            [seed.arclength, record.arclength],
            relative_threshold=options.jump_relative_threshold,
        ):
            events["arclength_jump"] += 1
            return None
        records.append(record)
    result = dedup_components(records, options.arclength_rtol, pool_count=0, raw_cluster_count=0)
    if result.component_count < len(records):
        events["hot_start_merged"] += len(records) - result.component_count
    return result


def _cold_discovery(
    scene: StripScene, target: np.ndarray, options: PixelOptions, events: Counter
) -> ComponentDiscoveryResult:
    """Cold prescan, then one production-budget retrace of each incomplete candidate.

    A recovered candidate is folded into the components by the same
    fingerprint dedup as the first pass; one that is still not closed stays
    ``incomplete`` (so the pixel stays ``"unknown"``).
    """
    first = discover_components(
        target,
        scene.incident_direction,
        scene.refractive_index,
        scene.crystal,
        template=scene.discovery_template,
        **options.discovery_kwargs(),
    )
    if not first.incomplete:
        return first
    records: list[DiscoveredComponent | IncompleteCandidate] = list(first.components)
    for candidate in first.incomplete:
        events["incomplete_retry"] += 1
        retraced = hot_start_component(
            candidate.seed,
            target,
            scene.incident_direction,
            scene.refractive_index,
            scene.crystal,
            discovery_step_budget=options.effective_retry_step_budget,
            template=scene.discovery_template,
        )
        if isinstance(retraced, DiscoveredComponent):
            events["incomplete_recovered"] += 1
            records.append(retraced)
        else:
            # ``None`` cannot happen for a seed that already passed the gates
            # with the same tolerances; keep the first-pass evidence if it does.
            records.append(retraced if retraced is not None else candidate)
    return dedup_components(
        records,
        options.arclength_rtol,
        pool_count=first.pool_count,
        raw_cluster_count=first.raw_cluster_count,
    )


def _same_components(
    left: ComponentDiscoveryResult, right: ComponentDiscoveryResult, rtol: float
) -> bool:
    """Same component multiset by the discovery fingerprint (arclength within ``rtol``)."""
    if left.component_count != right.component_count:
        return False
    remaining = [component.arclength for component in right.components]
    for component in left.components:
        match = next(
            (index for index, arclength in enumerate(remaining)
             if np.isclose(component.arclength, arclength, rtol=rtol, atol=1e-6)),
            None,
        )
        if match is None:
            return False
        remaining.pop(match)
    return True


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
    result = trace_fiber(problem, options.continuation)
    timings["production_s"] += time.perf_counter() - start
    production_arclength = float(np.sum(result.arclength_increments)) if len(result.poses) else 0.0
    if result.status != FiberStatus.CLOSED:
        events["production_not_closed"] += 1
    elif not np.isclose(production_arclength, component.arclength, rtol=options.arclength_rtol, atol=1e-6):
        events["production_arclength_mismatch"] += 1
    start = time.perf_counter()
    quadrature = integrate_fiber(problem, result, options.continuation, options.quadrature)
    timings["quadrature_s"] += time.perf_counter() - start
    if quadrature.status != "available":
        events["quadrature_unavailable"] += 1
    if quadrature.depth_exhausted_edges:
        events["quadrature_depth_exhausted"] += 1
    return ComponentRecord(
        seed=np.asarray(component.seed, dtype=np.float64),
        discovery_arclength=float(component.arclength),
        production_status=result.status.value,
        production_reason=result.reason.value,
        production_arclength=production_arclength,
        production_pose_count=int(len(result.poses)),
        quadrature_status=quadrature.status,
        value=float(quadrature.value) if quadrature.status == "available" else 0.0,
        error_estimate=float(quadrature.error_estimate) if quadrature.status == "available" else 0.0,
        refinements=int(quadrature.refinements),
        node_count=int(quadrature.node_count),
        maximum_depth_reached=int(quadrature.maximum_depth_reached),
        depth_exhausted_edge_count=len(quadrature.depth_exhausted_edges),
    )


def render_pixel(
    scene: StripScene,
    row: int,
    column: int,
    options: PixelOptions,
    *,
    target: np.ndarray | None = None,
    previous: Sequence[HotSeed] | None = None,
    cold_check: bool = False,
) -> PixelResult:
    """Run the full pipeline for pixel ``(row, column)`` (module docstring).

    ``target`` overrides the pixel-centre direction (sub-pixel sampling);
    ``previous`` are the neighbour's integrated components to hot-start from;
    ``cold_check`` additionally runs the cold prescan after a successful hot
    start and, if the component fingerprints differ, keeps the cold result and
    counts ``cold_check_mismatch``.
    """
    target = pixel_target(scene.render, row, column) if target is None else np.asarray(target, dtype=np.float64)
    events: Counter = Counter()
    timings: Counter = Counter({name: 0.0 for name in STAGE_NAMES})
    total_start = time.perf_counter()

    discovered: ComponentDiscoveryResult | None = None
    seed_source = "cold"
    start = time.perf_counter()
    if previous:
        discovered = _hot_start_all(scene, target, previous, options, events)
        seed_source = "hot" if discovered is not None else "cold-fallback"
    timings["hot_start_s"] += time.perf_counter() - start
    if discovered is not None and cold_check:
        events["cold_check"] += 1
        start = time.perf_counter()
        cold = _cold_discovery(scene, target, options, events)
        timings["cold_discovery_s"] += time.perf_counter() - start
        if not _same_components(discovered, cold, options.arclength_rtol):
            events["cold_check_mismatch"] += 1
            discovered = cold
            seed_source = "cold-check"
    if discovered is None:
        start = time.perf_counter()
        discovered = _cold_discovery(scene, target, options, events)
        timings["cold_discovery_s"] += time.perf_counter() - start
    discovery_completeness = discovered.completeness if seed_source != "hot" else "not-run"
    if discovered.incomplete_count:
        events["incomplete_candidate"] += discovered.incomplete_count

    components = tuple(
        _integrate_component(scene, target, component, options, events, timings)
        for component in discovered.components
    )
    timings["total_s"] = time.perf_counter() - total_start

    # A hot-start failure or jump that fell back to a successful cold prescan
    # is not missing evidence (the cold result's own incomplete count is); it
    # stays visible through ``seed_source`` and the status bits.
    failure_events = (
        "incomplete_candidate",
        "production_not_closed",
        "production_arclength_mismatch",
        "quadrature_unavailable",
    )
    completeness = "complete" if not any(events[name] for name in failure_events) else "unknown"
    return PixelResult(
        row=int(row),
        column=int(column),
        value=float(sum(component.value for component in components)),
        error_estimate=float(sum(component.error_estimate for component in components)),
        components=components,
        completeness=completeness,
        discovery_completeness=discovery_completeness,
        seed_source=seed_source,
        incomplete_count=discovered.incomplete_count,
        pool_count=discovered.pool_count,
        raw_cluster_count=discovered.raw_cluster_count,
        admissible_count=discovered.admissible_count,
        events={name: int(events[name]) for name in EVENT_NAMES},
        timings={name: float(timings[name]) for name in STAGE_NAMES},
    )


__all__ = [
    "EVENT_NAMES",
    "ComponentRecord",
    "HotSeed",
    "PixelOptions",
    "PixelResult",
    "STATUS_ARCLENGTH_JUMP",
    "STATUS_COLD_CHECK_MISMATCH",
    "STATUS_COLD_DISCOVERY",
    "STATUS_DEPTH_EXHAUSTED",
    "STATUS_HAS_COMPONENT",
    "STATUS_PRODUCTION_FAILURE",
    "STATUS_RENDERED",
    "STATUS_UNKNOWN_COMPLETENESS",
    "STAGE_NAMES",
    "StripScene",
    "canonical_strip_scene",
    "pixel_target",
    "render_pixel",
    "subpixel_targets",
]
