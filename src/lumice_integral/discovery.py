"""Single-pixel component discovery for one fixed-path fiber.

Given one pixel's target direction ``d``, this module finds the connected
pieces of the inverse-image fiber ``X_(P, d)`` in SO(3) that the scene's
S^2 event store of path ``P`` can reach, traces each once with the production continuation, and
returns them as distinct *components* -- closed loops or open arcs -- plus the
candidates that could not be classified.

The procedure (task-pixel-pipeline-v2, after the author's ruling of
2026-09-17 that an open arc is a first-class component, not a failure):

1. Candidate pool: ``extra_seeds`` (already converged poses of neighbouring
   pixels, any neighbour -- the caller decides which) followed by the band of
   the scene's :class:`.s2_store.StoreSeeds`: the store events (``w = A T >
   0`` on the crystal) whose deviation lies within ``band_half_width_deg`` of
   the deviation ``delta`` of ``d``, each posed in the azimuth of ``d``
   (:meth:`.s2_store.StoreSeeds.candidates`), so a candidate's outgoing
   direction is ``|D_i - delta|`` from ``d``.  The store is the SO(3)
   sampling quotiented by the twist about the sun (task
   ``phase1-seeds-from-store``, which retired the Haar prescan table after a
   32-pixel probe found the same components).  ``seeds.faces`` is the single
   source of the face sequence this call discovers, and the store's crystal
   and refractive index are the problem's.
2. Greedy geodesic clustering of the whole pool with radius
   ``cluster_radius_rad``.  A cluster's representative is its first extra
   seed if it contains one (that is all a warm seed does: it puts the
   Gauss-Newton start of its cluster on a neighbouring solution), else its
   store member with the smallest ``|D_i - delta|``.
3. Per cluster, in pool order: Gauss-Newton the representative onto the
   fiber, gate it with :func:`.optics.path_domain` and
   :func:`.geometry.entry_measure`, then *deduplicate before tracing*: a
   corrected seed whose SO(3) geodesic distance to any accepted pose of an
   already accepted component is below ``distance_threshold`` is the same
   component and is folded without a trace (``dedup_merged``).
4. Otherwise :func:`.continuation.trace_fiber` once with the caller's
   production :class:`.continuation.ContinuationOptions`:

   - ``closed``: a :class:`DiscoveredComponent` of ``kind == "closed"``;
   - ``event_terminated`` by one of the five named events
     (:data:`ARC_EVENTS`: tir, branch, path-infeasible, visibility, chart):
     trace the same seed once more with ``initial_tangent_sign = -1``; if
     that also ends on a named event the two traces are stitched
     (:func:`.resample.stitch_open_arc`) into a component of ``kind ==
     "arc"`` (``arc_stitched``).  A backward trace that does not
     (``arc_backward_failed``), or that closes -- which the forward trace of
     the same seed should have done first (``arc_backward_closed_anomaly``)
     -- leaves the candidate ``incomplete`` with both traces kept;
   - any other outcome (``numerical_failure``, ``budget_exhausted``, or an
     unnamed event such as ``rank_loss``, counted as
     ``incomplete_unnamed_event``): an :class:`IncompleteCandidate`.

``completeness`` is a *procedural* signal, not a mathematical certificate.
``"complete"`` means only that every admissible candidate of this pool
converged (closed or arc) and no ``incomplete`` evidence was observed; it does
not prove that every connected component of ``X_(P,d)`` was found (that
remains an open item of ``docs/phase1-math-contract.md``).  A dark pixel with
no admissible candidate is therefore ``"complete"`` with zero components.

Batch callers pass ``template`` (a :class:`FiberProblem` of the same path,
incident direction and refractive index); the per-pixel problem is then
:func:`retarget_problem` of that template, so the continuation kernels keyed
on the template's ``direction_evaluator`` identity are compiled once per
process instead of once per pixel (about 0.4 s per fresh problem on the M2 Max
CPU, task-strip-image-driver Step 0).  Without ``template`` every call builds
a fresh problem.
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, replace
from functools import partial
from typing import Literal, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from .analytic import tangent_basis
from .continuation import (
    ContinuationOptions,
    DirectionEvaluator,
    FiberProblem,
    FiberResult,
    FiberStatus,
    TargetChart,
    TerminationReason,
    target_residual,
    trace_fiber,
)
from .geometry import Polyhedron, entry_measure
from .optics import PATH_3_5_FACES, path_domain, path_problem, problem_path_label
from .resample import OpenArc, stitch_open_arc
from .s2_store import StoreSeeds
from .so3 import exp, rotation_distances

Completeness = Literal["complete", "unknown"]
ComponentKind = Literal["closed", "arc"]
# The named events an arc may end on (docs/phase1-math-contract.md section 8);
# ``rank_loss``/``topology_ambiguity`` are "open" degeneracies, not arc ends.
ARC_EVENTS = frozenset(
    {
        TerminationReason.TIR_BOUNDARY,
        TerminationReason.BRANCH_BOUNDARY,
        TerminationReason.PATH_INFEASIBLE,
        TerminationReason.VISIBILITY_BOUNDARY,
        TerminationReason.CHART_BOUNDARY,
    }
)
# Per-call counters of the classification funnel (module docstring step 3/4).
DISCOVERY_EVENT_NAMES = (
    "dedup_merged",
    "arc_stitched",
    "arc_backward_failed",
    "arc_backward_closed_anomaly",
    "incomplete_unnamed_event",
    "incomplete_not_converged",
)


@dataclass(frozen=True)
class DiscoveredComponent:
    """One distinct component: its corrected seed and the single production trace.

    ``kind == "closed"``: ``result`` is the closed :class:`FiberResult`.
    ``kind == "arc"``: ``result`` is the stitched :class:`.resample.OpenArc`
    (its ``forward``/``backward`` traces, ``reason``/``start_reason`` events
    and ``branch_diagnostics`` margins at both ends).  ``arclength`` is the
    sum of the accepted chords of the whole curve; ``status``/``reason`` are
    the curve's terminal ones (``closed``/``closed_loop`` or
    ``event_terminated``/the forward end's event).
    """

    seed: np.ndarray
    kind: ComponentKind
    result: FiberResult | OpenArc
    arclength: float
    status: FiberStatus
    reason: TerminationReason

    @property
    def start_reason(self) -> TerminationReason | None:
        """The backward end's event of an arc; ``None`` for a closed loop."""
        return self.result.start_reason if self.kind == "arc" else None


@dataclass(frozen=True)
class IncompleteCandidate:
    """An admissible candidate whose trace did not converge to a closed loop or an arc.

    It is evidence of nothing conclusive: neither a distinct component nor a
    duplicate of one.  ``result`` keeps the forward trace and ``backward`` the
    backward one when it was attempted (module docstring step 4); ``cause``
    is the :data:`DISCOVERY_EVENT_NAMES` entry that classified it.
    """

    seed: np.ndarray
    result: FiberResult
    status: FiberStatus
    reason: TerminationReason
    cause: str
    backward: FiberResult | None = None


@dataclass(frozen=True)
class ComponentDiscoveryResult:
    """Components and unclassified candidates found for one target direction.

    ``completeness`` is procedural (see the module docstring): ``"complete"``
    iff ``incomplete`` is empty.  The count fields record the funnel
    (``pool_count`` store-band poses plus ``extra_seed_count`` warm seeds ->
    clusters -> admissible) for regression and diagnosis;
    ``events`` the classification counters (:data:`DISCOVERY_EVENT_NAMES`);
    ``trace_seconds`` the wall clock spent inside :func:`.continuation.trace_fiber`.
    """

    components: tuple[DiscoveredComponent, ...]
    incomplete: tuple[IncompleteCandidate, ...]
    completeness: Completeness
    pool_count: int
    extra_seed_count: int
    raw_cluster_count: int
    admissible_count: int
    events: Mapping[str, int]
    trace_seconds: float

    @property
    def component_count(self) -> int:
        return len(self.components)

    @property
    def arc_count(self) -> int:
        return sum(component.kind == "arc" for component in self.components)

    @property
    def incomplete_count(self) -> int:
        return len(self.incomplete)


# Pool clustering and curve dedup compare one pose against a few thousand and
# never differentiate; they run on the host through ``so3.rotation_distances``
# because a ``jax.jit(vmap(rotation_distance))`` kernel is recompiled for
# every distinct pool size / curve length, i.e. for nearly every pixel
# (task-pixel-cost-shape-stable-kernels).


def _geodesic_cluster(rotations: np.ndarray, radius: float) -> list[list[int]]:
    """Greedy clustering by SO(3) geodesic distance to the first unassigned member."""
    count = rotations.shape[0]
    unassigned = set(range(count))
    clusters: list[list[int]] = []
    while unassigned:
        seed_index = next(iter(unassigned))
        distances = rotation_distances(rotations[seed_index], rotations)
        members = [i for i in unassigned if distances[i] < radius]
        clusters.append(members)
        unassigned -= set(members)
    return clusters


def distance_to_curve(rotation: np.ndarray, poses: np.ndarray) -> float:
    """Smallest SO(3) geodesic distance from ``rotation`` to the sampled ``poses``.

    The curve is represented by its accepted poses only (chords of at most
    ``ContinuationOptions.maximum_step``), so a pose *on* the curve can still
    be up to half a chord away from the nearest sample; the dedup threshold
    must absorb that.
    """
    return float(np.min(rotation_distances(rotation, poses)))


@partial(jax.jit, static_argnums=(0,))
def _newton_step_kernel(
    direction_evaluator: DirectionEvaluator, rotation: Array, chart_direction: Array, chart_basis: Array
) -> tuple[Array, Array]:
    """One Gauss-Newton update of ``rotation`` towards the target chart's fiber.

    Returns the updated pose and the residual norm *before* the update, so
    the caller can stop when the pose it already holds is on the fiber.
    """

    def residual(delta: Array) -> Array:
        direction = direction_evaluator(rotation @ exp(delta))
        return chart_basis.T @ (direction - chart_direction)

    zero = jnp.zeros(3, dtype=rotation.dtype)
    value, jacobian = residual(zero), jax.jacfwd(residual)(zero)
    delta = -jacobian.T @ jnp.linalg.solve(jacobian @ jacobian.T, value)
    return rotation @ exp(delta), jnp.linalg.norm(value)


def _newton_correct(
    problem: FiberProblem, rotation: Array, tolerance: float, iterations: int = 30
) -> tuple[Array, float]:
    """Unconstrained Gauss-Newton from an arbitrary pose onto the target fiber.

    Unlike :func:`.continuation.retract_to_fiber` this needs no on-fiber base
    pose or phase tangent; it only pulls a candidate pose onto the fiber.
    Each iteration is one compiled kernel keyed on the problem's
    ``direction_evaluator`` (shared through ``template``).
    """
    chart = problem.target_chart
    for _ in range(iterations):
        updated, norm = _newton_step_kernel(problem.direction_evaluator, rotation, chart.direction, chart.basis)
        if float(norm) <= tolerance:
            return rotation, float(norm)
        rotation = updated
    return rotation, float(jnp.linalg.norm(target_residual(problem, rotation)))


def _admissible_seed(
    template: FiberProblem,
    options: ContinuationOptions,
    crystal: Polyhedron,
    faces: tuple[int, ...],
    refractive_index: float,
    raw_rotation: Array,
) -> np.ndarray | None:
    """Correct one candidate onto the fiber and gate it; ``None`` if inadmissible.

    The incident direction is read from ``template`` so the gates cannot
    drift from the problem being traced; ``faces`` and ``refractive_index``
    are passed separately because :class:`FiberProblem` stores them only in
    its ``path`` label.  The gates are the residual, domain validity and a
    positive entry measure.
    """
    tolerance = options.residual_tolerance + options.relative_residual_tolerance
    corrected, residual_norm = _newton_correct(template, raw_rotation, tolerance * 1e-2)
    corrected_np = np.asarray(corrected)
    incident = np.asarray(template.incident_direction)
    domain = path_domain(corrected_np, faces, incident, refractive_index)
    measure = entry_measure(corrected_np, faces, incident, crystal, n_ice=refractive_index)
    if not (residual_norm <= tolerance and domain.valid and measure.value > 0):
        return None
    return corrected_np


def _trace_and_classify(
    template: FiberProblem,
    options: ContinuationOptions,
    seed: np.ndarray,
    events: Counter,
    timings: Counter,
) -> DiscoveredComponent | IncompleteCandidate:
    """One production trace of ``seed``, then the closed / arc / incomplete branch."""
    problem = replace(template, seed=jnp.asarray(seed))
    start = time.perf_counter()
    forward = trace_fiber(problem, options)
    timings["trace_s"] += time.perf_counter() - start
    if forward.status == FiberStatus.CLOSED:
        return DiscoveredComponent(
            seed, "closed", forward, float(forward.arclength_increments.sum()), forward.status, forward.reason
        )
    if forward.status != FiberStatus.EVENT_TERMINATED:
        events["incomplete_not_converged"] += 1
        return IncompleteCandidate(seed, forward, forward.status, forward.reason, "incomplete_not_converged")
    if forward.reason not in ARC_EVENTS:
        events["incomplete_unnamed_event"] += 1
        return IncompleteCandidate(seed, forward, forward.status, forward.reason, "incomplete_unnamed_event")
    start = time.perf_counter()
    backward = trace_fiber(problem, replace(options, initial_tangent_sign=-options.initial_tangent_sign))
    timings["trace_s"] += time.perf_counter() - start
    if backward.status == FiberStatus.CLOSED:
        # The forward trace of the same seed on the same one-dimensional
        # fiber should have closed first; the data contradict the model, so
        # the candidate is exposed as incomplete rather than accepted.
        events["arc_backward_closed_anomaly"] += 1
        return IncompleteCandidate(
            seed, forward, forward.status, forward.reason, "arc_backward_closed_anomaly", backward
        )
    if backward.status != FiberStatus.EVENT_TERMINATED or backward.reason not in ARC_EVENTS:
        events["arc_backward_failed"] += 1
        return IncompleteCandidate(seed, forward, forward.status, forward.reason, "arc_backward_failed", backward)
    arc = stitch_open_arc(forward, backward)
    events["arc_stitched"] += 1
    return DiscoveredComponent(seed, "arc", arc, arc.arclength, arc.status, arc.reason)


def retarget_problem(
    template: FiberProblem, target_direction: np.ndarray, seed: np.ndarray
) -> FiberProblem:
    """``template`` with a new target chart and seed, keeping its evaluator closures.

    The chart is rebuilt the way :func:`.optics.path_problem` builds it
    (``tangent_basis`` of the target, the template's ``minimum_dot``); the
    ``direction_evaluator`` / ``domain_and_event_evaluator`` objects and the
    weight evaluators are shared, so ``jax.jit`` caches keyed on them stay warm.
    """
    target = jnp.asarray(target_direction, dtype=jnp.float64)
    return replace(
        template,
        seed=jnp.asarray(seed, dtype=jnp.float64),
        target_chart=TargetChart(
            target, tangent_basis(target), minimum_dot=template.target_chart.minimum_dot
        ),
    )


def _problem_template(
    target_direction: np.ndarray,
    faces: tuple[int, ...],
    incident_direction: np.ndarray,
    refractive_index: float,
    seed: np.ndarray,
    template: FiberProblem | None,
) -> FiberProblem:
    if template is not None:
        incident = np.asarray(template.incident_direction)
        if not np.allclose(incident, np.asarray(incident_direction, dtype=np.float64)):
            raise ValueError("template incident direction does not match incident_direction")
        expected = problem_path_label(faces, refractive_index)
        if template.path != expected:
            raise ValueError(f"template path {template.path!r} does not match the {expected!r} problem")
        return retarget_problem(template, target_direction, seed)
    return path_problem(
        jnp.asarray(seed, dtype=jnp.float64),
        faces,
        jnp.asarray(incident_direction, dtype=jnp.float64),
        target_direction=jnp.asarray(target_direction, dtype=jnp.float64),
        refractive_index=jnp.asarray(refractive_index, dtype=jnp.float64),
    )


def discover_components(
    target_direction: np.ndarray,
    seeds: StoreSeeds,
    *,
    continuation: ContinuationOptions | None = None,
    extra_seeds: Sequence[np.ndarray] = (),
    band_half_width_deg: float = 0.2,
    cluster_radius_rad: float = 0.3,
    distance_threshold: float = ContinuationOptions.closure_distance,
    template: FiberProblem | None = None,
) -> ComponentDiscoveryResult:
    """Discover the fiber components of the seeds' path reaching ``target_direction``.

    ``seeds`` is the scene's store view (:class:`.s2_store.StoreSeeds`) and
    the single source of the face sequence (``seeds.faces``), the incident
    direction, the refractive index and the crystal of the problem (the
    crystal feeds the finite-crystal :func:`.geometry.entry_measure` gate
    applied to each corrected candidate; the store kept only its ``w > 0``
    events).  ``continuation`` is the production
    policy every trace runs under (default :class:`ContinuationOptions`);
    ``extra_seeds`` are converged poses of neighbouring pixels used as
    Gauss-Newton starts (module docstring step 2), never traced separately
    and never a source of completeness.  ``band_half_width_deg`` defaults to
    0.2 deg (with the production ``N = 1e6`` store a pool of the prescan's
    size; the task ``phase1-seeds-from-store`` probe found every component of
    its 32 pixels already at ``N = 1e5`` and 0.02 deg), the 0.3 rad cluster
    radius to the ``explore-component-discovery`` survey; ``distance_threshold`` defaults to the continuation's
    ``closure_distance`` (same scale: "is this pose on that curve").  The
    result's ``completeness`` is procedural; see the module docstring.
    ``template`` (optional) is a problem of the same path, incident direction
    and index whose evaluator closures are reused via :func:`retarget_problem`.
    """
    if distance_threshold <= 0.0:
        raise ValueError("distance_threshold must be positive")
    options = continuation or ContinuationOptions()
    faces = seeds.faces
    incident = seeds.incident_direction
    refractive_index = seeds.refractive_index
    crystal = seeds.crystal
    target = np.asarray(target_direction, dtype=np.float64)
    band_rotations, band_offsets = seeds.candidates(target, np.radians(band_half_width_deg))
    extra = np.asarray(extra_seeds, dtype=np.float64).reshape(-1, 3, 3)
    pool_rotations = np.concatenate((extra, band_rotations))
    extra_count = len(extra)
    # Alignment of the band members with the target (the smaller |D_i - delta|, the better); extra
    # seeds have no band offset and are chosen as representatives by position instead.
    pool_alignment = np.concatenate((np.full(extra_count, -np.inf), -band_offsets))

    clusters = _geodesic_cluster(pool_rotations, cluster_radius_rad)
    template = _problem_template(
        target,
        faces,
        incident,
        refractive_index,
        pool_rotations[0] if len(pool_rotations) else np.eye(3),
        template,
    )
    events: Counter = Counter({name: 0 for name in DISCOVERY_EVENT_NAMES})
    timings: Counter = Counter({"trace_s": 0.0})
    components: list[DiscoveredComponent] = []
    incomplete: list[IncompleteCandidate] = []
    admissible_count = 0
    for cluster in clusters:
        warm = [i for i in cluster if i < extra_count]
        representative = warm[0] if warm else max(cluster, key=lambda i: pool_alignment[i])
        seed = _admissible_seed(
            template, options, crystal, faces, refractive_index, jnp.asarray(pool_rotations[representative])
        )
        if seed is None:
            continue
        admissible_count += 1
        if any(distance_to_curve(seed, component.result.poses) < distance_threshold for component in components):
            events["dedup_merged"] += 1
            continue
        record = _trace_and_classify(template, options, seed, events, timings)
        if isinstance(record, DiscoveredComponent):
            components.append(record)
        else:
            incomplete.append(record)
    return ComponentDiscoveryResult(
        components=tuple(components),
        incomplete=tuple(incomplete),
        completeness="complete" if not incomplete else "unknown",
        pool_count=int(len(band_rotations)),
        extra_seed_count=extra_count,
        raw_cluster_count=len(clusters),
        admissible_count=admissible_count,
        events={name: int(events[name]) for name in DISCOVERY_EVENT_NAMES},
        trace_seconds=float(timings["trace_s"]),
    )


@dataclass(frozen=True)
class BandCoverage:
    """How the band events of one pixel sit relative to the components discovery traced (a diagnostic).

    Every event of the band (``event_count``) is either *near* a traced curve
    as posed (SO(3) distance below ``near_radius``), or Newton-corrected onto
    the fiber and then *on* a traced curve (distance below
    ``distance_threshold``), *inadmissible* (the discovery gates reject it),
    or a *suspect*: an admissible fiber pose far from every traced curve,
    i.e. evidence of a component discovery missed.  ``suspects`` are those
    corrected poses and ``suspect_clusters`` their count after the discovery
    clustering.  ``component_event_counts[k]`` is the number of events
    assigned to component ``k`` (nearest curve); the smallest of them,
    ``k_min``, gives ``miss_probability_bound = exp(-k_min)``: the chance
    that ``N`` independent uniform points put no event on a component whose
    band measure is that of the least-covered component found (for i.i.d.
    points ``(1 - mu / 4 pi)^N <= exp(-N mu / 4 pi)`` with ``N mu / 4 pi``
    estimated by ``k_min``; the store's Fibonacci lattice is deterministic
    and quasi-uniform, so this is the Monte Carlo reading of the same
    density, not a certificate).  ``None`` when nothing was traced.
    """

    event_count: int
    near_count: int
    corrected_count: int
    on_curve_count: int
    inadmissible_count: int
    suspects: tuple[np.ndarray, ...]
    suspect_clusters: int
    component_event_counts: tuple[int, ...]
    miss_probability_bound: float | None

    @property
    def suspect_count(self) -> int:
        return len(self.suspects)


def miss_probability(band_measure_sr: float, n: int) -> float:
    """``exp(-n mu / 4 pi)``: the chance that ``n`` i.i.d. uniform points on ``S^2`` miss a region of measure ``mu``."""
    return float(np.exp(-n * band_measure_sr / (4.0 * np.pi)))


def check_band_coverage(
    target_direction: np.ndarray,
    seeds: StoreSeeds,
    result: ComponentDiscoveryResult,
    *,
    band_half_width_deg: float = 0.2,
    near_radius: float = ContinuationOptions.closure_distance,
    distance_threshold: float = ContinuationOptions.closure_distance,
    cluster_radius_rad: float = 0.3,
    continuation: ContinuationOptions | None = None,
    template: FiberProblem | None = None,
) -> BandCoverage:
    """Completeness cross-check of one pixel's discovery against *every* event of its store band.

    Discovery traces one representative per cluster of the band; this check
    revisits all of them (:class:`BandCoverage`).  An event posed within
    ``near_radius`` of a traced curve is counted as covered without
    correction (the posed events lie ``|D_i - delta|`` off the fiber in
    direction: within 0.035 rad of the curve on the 32 survey pixels away
    from the caustic, up to 0.117 rad on its short loops, task
    ``phase1-seeds-from-store``); the default is discovery's own dedup
    distance, so an event is never taken as covered where discovery would
    not fold it.  Every other event is
    corrected onto the fiber and gated exactly as a discovery candidate
    (:func:`_admissible_seed`) and compared with the curves at
    ``distance_threshold``.  A diagnostic and a test tool, not part of the
    rendering path: it costs one Newton correction per far event.
    """
    options = continuation or ContinuationOptions()
    target = np.asarray(target_direction, dtype=np.float64)
    rotations, _ = seeds.candidates(target, np.radians(band_half_width_deg))
    curves = [np.asarray(component.result.poses) for component in result.components]
    counts = [0] * len(curves)
    near = corrected_count = on_curve = inadmissible = 0
    suspects: list[np.ndarray] = []
    problem = None
    for rotation in rotations:
        distances = [float(np.min(rotation_distances(rotation, poses))) for poses in curves]
        if distances and min(distances) < near_radius:
            near += 1
            counts[int(np.argmin(distances))] += 1
            continue
        if problem is None:
            problem = _problem_template(target, seeds.faces, seeds.incident_direction, seeds.refractive_index, rotation, template)
        corrected_count += 1
        seed = _admissible_seed(problem, options, seeds.crystal, seeds.faces, seeds.refractive_index, jnp.asarray(rotation))
        if seed is None:
            inadmissible += 1
            continue
        distances = [distance_to_curve(seed, poses) for poses in curves]
        if distances and min(distances) < distance_threshold:
            on_curve += 1
            counts[int(np.argmin(distances))] += 1
        else:
            suspects.append(seed)
    clusters = _geodesic_cluster(np.asarray(suspects).reshape(-1, 3, 3), cluster_radius_rad)
    return BandCoverage(
        event_count=int(len(rotations)),
        near_count=near,
        corrected_count=corrected_count,
        on_curve_count=on_curve,
        inadmissible_count=inadmissible,
        suspects=tuple(suspects),
        suspect_clusters=len(clusters),
        component_event_counts=tuple(counts),
        miss_probability_bound=float(np.exp(-min(counts))) if counts else None,
    )


__all__ = [
    "ARC_EVENTS",
    "DISCOVERY_EVENT_NAMES",
    "PATH_3_5_FACES",
    "BandCoverage",
    "ComponentDiscoveryResult",
    "DiscoveredComponent",
    "IncompleteCandidate",
    "check_band_coverage",
    "discover_components",
    "distance_to_curve",
    "miss_probability",
    "retarget_problem",
]
