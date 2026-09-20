"""Single-pixel component discovery for the 3-5 fiber.

Given one pixel's target direction ``d``, this module finds the connected
pieces of the inverse-image fiber ``X_(3-5, d)`` in SO(3) that the scene's
prescan can reach, traces each once with the production continuation, and
returns them as distinct *components* -- closed loops or open arcs -- plus the
candidates that could not be classified.

The procedure (task-pixel-pipeline-v2, after the author's ruling of
2026-09-17 that an open arc is a first-class component, not a failure):

1. Candidate pool: ``extra_seeds`` (already converged poses of neighbouring
   pixels, any neighbour -- the caller decides which) followed by the query of
   the scene's :class:`.prescan.PrescanTable` (Haar samples that pass all
   four refraction discriminants of the smooth 3-5 branch, indexed by
   outgoing direction) for the poses whose outgoing direction lies within
   ``angle_tolerance_deg`` of ``d``.
2. Greedy geodesic clustering of the whole pool with radius
   ``cluster_radius_rad``.  A cluster's representative is its first extra
   seed if it contains one (that is all a warm seed does: it puts the
   Gauss-Newton start of its cluster on a neighbouring solution), else its
   best-aligned prescan member.
3. Per cluster, in pool order: Gauss-Newton the representative onto the
   fiber, gate it with :func:`.optics.path_3_5_domain` and
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

Batch callers pass ``template`` (a 3-5 :class:`FiberProblem` for the same
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
from .optics import path_3_5_domain, path_3_5_problem
from .prescan import PrescanTable
from .resample import OpenArc, stitch_open_arc
from .so3 import exp, rotation_distances

PATH_3_5_FACES = (3, 5)

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
    (``pool_count`` prescan poses plus ``extra_seed_count`` warm seeds ->
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
    pose or phase tangent; it only pulls a prescan sample onto the fiber.
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
    refractive_index: float,
    raw_rotation: Array,
) -> np.ndarray | None:
    """Correct one candidate onto the fiber and gate it; ``None`` if inadmissible.

    The incident direction is read from ``template`` so the gates cannot
    drift from the problem being traced; ``refractive_index`` is passed
    separately because :class:`FiberProblem` does not store it as a number.
    The gates are the residual, domain validity and a positive entry measure.
    """
    tolerance = options.residual_tolerance + options.relative_residual_tolerance
    corrected, residual_norm = _newton_correct(template, raw_rotation, tolerance * 1e-2)
    corrected_np = np.asarray(corrected)
    incident = np.asarray(template.incident_direction)
    domain = path_3_5_domain(corrected_np, incident, refractive_index)
    measure = entry_measure(
        corrected_np, PATH_3_5_FACES, incident, crystal, n_ice=refractive_index
    )
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

    The chart is rebuilt the way :func:`.optics.path_3_5_problem` builds it
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
    incident_direction: np.ndarray,
    refractive_index: float,
    seed: np.ndarray,
    template: FiberProblem | None,
) -> FiberProblem:
    if template is not None:
        incident = np.asarray(template.incident_direction)
        if not np.allclose(incident, np.asarray(incident_direction, dtype=np.float64)):
            raise ValueError("template incident direction does not match incident_direction")
        if template.path != f"3-5:n={float(refractive_index):.8g}":
            raise ValueError(f"template path {template.path!r} does not match the 3-5 problem")
        return retarget_problem(template, target_direction, seed)
    return path_3_5_problem(
        jnp.asarray(seed, dtype=jnp.float64),
        jnp.asarray(incident_direction, dtype=jnp.float64),
        target_direction=jnp.asarray(target_direction, dtype=jnp.float64),
        refractive_index=jnp.asarray(refractive_index, dtype=jnp.float64),
    )


def discover_components(
    target_direction: np.ndarray,
    crystal: Polyhedron,
    table: PrescanTable,
    *,
    continuation: ContinuationOptions | None = None,
    extra_seeds: Sequence[np.ndarray] = (),
    angle_tolerance_deg: float = 2.0,
    cluster_radius_rad: float = 0.3,
    distance_threshold: float = ContinuationOptions.closure_distance,
    template: FiberProblem | None = None,
) -> ComponentDiscoveryResult:
    """Discover the 3-5 fiber components reaching ``target_direction``.

    ``table`` is the scene's prescan (:func:`.prescan.build_prescan_table`)
    and the single source of the incident direction and refractive index of
    the problem; ``crystal`` only feeds the finite-crystal
    :func:`.geometry.entry_measure` gate applied to each corrected candidate
    (the table does not depend on it).  ``continuation`` is the production
    policy every trace runs under (default :class:`ContinuationOptions`);
    ``extra_seeds`` are converged poses of neighbouring pixels used as
    Gauss-Newton starts (module docstring step 2), never traced separately
    and never a source of completeness.  Pool defaults come from the
    ``explore-component-discovery`` survey (2 deg tolerance, 0.3 rad cluster
    radius); ``distance_threshold`` defaults to the continuation's
    ``closure_distance`` (same scale: "is this pose on that curve").  The
    result's ``completeness`` is procedural; see the module docstring.
    ``template`` (optional) is a 3-5 problem for the same incident direction
    and index whose evaluator closures are reused via :func:`retarget_problem`.
    """
    if distance_threshold <= 0.0:
        raise ValueError("distance_threshold must be positive")
    options = continuation or ContinuationOptions()
    incident = table.incident_direction
    refractive_index = table.refractive_index
    target = np.asarray(target_direction, dtype=np.float64)
    pool_indices = table.candidates(target, angle_tolerance_deg)
    extra = np.asarray(extra_seeds, dtype=np.float64).reshape(-1, 3, 3)
    pool_rotations = np.concatenate((extra, table.rotations[pool_indices]))
    extra_count = len(extra)
    # Alignment of the prescan members with the target; extra seeds have no
    # table direction and are chosen as representatives by position instead.
    pool_alignment = np.concatenate(
        (np.full(extra_count, -np.inf), table.directions[pool_indices] @ target)
    )

    clusters = _geodesic_cluster(pool_rotations, cluster_radius_rad)
    template = _problem_template(
        target,
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
            template, options, crystal, refractive_index, jnp.asarray(pool_rotations[representative])
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
        pool_count=int(len(pool_indices)),
        extra_seed_count=extra_count,
        raw_cluster_count=len(clusters),
        admissible_count=admissible_count,
        events={name: int(events[name]) for name in DISCOVERY_EVENT_NAMES},
        trace_seconds=float(timings["trace_s"]),
    )


__all__ = [
    "ARC_EVENTS",
    "DISCOVERY_EVENT_NAMES",
    "ComponentDiscoveryResult",
    "DiscoveredComponent",
    "IncompleteCandidate",
    "discover_components",
    "distance_to_curve",
    "retarget_problem",
]
