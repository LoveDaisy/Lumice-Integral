"""Single-pixel component discovery for the 3-5 fiber.

Given one pixel's target direction ``d``, this module finds seeds on the
inverse-image fiber ``X_(3-5, d)`` in SO(3) and traces each candidate with a
small step budget, so that a caller such as a strip driver receives a list of
distinct closed components plus the candidates that could not be classified.

The procedure is the one validated by ``explore-component-discovery``:

1. Haar-uniform prescan of ``prescan_samples`` rotations through the smooth
   3-5 branch; keep the pool whose outgoing direction lies within
   ``angle_tolerance_deg`` of ``d`` and passes all four refraction
   discriminants.
2. Greedy geodesic clustering of the whole pool with radius
   ``cluster_radius_rad`` (the whole pool, not a top-K by alignment).
3. Per cluster: take the best-aligned member, Gauss-Newton it onto the fiber,
   gate it with :func:`.optics.path_3_5_domain` and
   :func:`.geometry.entry_measure`, then run :func:`.continuation.trace_fiber`
   with ``maximum_accepted_steps = discovery_step_budget``.
4. Deduplicate ``closed`` traces by the fingerprint
   ``(status, reason, arclength within arclength_rtol)``.  Accepted pose
   counts are *not* part of the fingerprint: adaptive stepping lands on a
   different number of poses for the same physical loop depending on where
   the corrector enters it.  Traces that did not close are never folded into
   a component; they are returned separately as ``incomplete``.

``completeness`` is a *procedural* signal, not a mathematical certificate.
``"complete"`` means only that every admissible candidate in this sampling
pool closed and no ``status != closed`` evidence was observed; it does not
prove that every connected component of ``X_(P,d)`` was found (that remains
an open item of ``docs/phase1-math-contract.md``).  A dark pixel with no
admissible candidate is therefore ``"complete"`` with zero components.

Discovery uses a deliberately small step budget (default 250) that is
independent of the production :class:`.continuation.ContinuationOptions`
default; a candidate that exhausts it is reported as ``incomplete`` and the
caller decides whether to retrace it with the production budget.

Batch callers may pass ``template`` (a 3-5 :class:`FiberProblem` for the same
incident direction and refractive index) to :func:`discover_components` and
:func:`hot_start_component`; the per-pixel problem is then
:func:`retarget_problem` of that template, so the continuation kernels keyed on
the template's ``direction_evaluator`` identity are compiled once per process
instead of once per pixel (about 0.4 s per fresh problem on the M2 Max CPU,
measured by ``task-strip-image-driver`` Step 0).  Without ``template`` every
call builds a fresh problem, as before.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from .analytic import tangent_basis
from .continuation import (
    ContinuationOptions,
    FiberProblem,
    FiberResult,
    FiberStatus,
    StepDiagnostic,
    TargetChart,
    TerminationReason,
    local_residual_jacobian,
    target_residual,
    trace_fiber,
)
from .geometry import Polyhedron, entry_measure
from .optics import path_3_5, path_3_5_domain, path_3_5_problem
from .so3 import exp, rotation_distance

PATH_3_5_FACES = (3, 5)

Completeness = Literal["complete", "unknown"]


@dataclass(frozen=True)
class DiscoveredComponent:
    """One distinct closed component: its corrected seed and the discovery trace."""

    seed: np.ndarray
    result: FiberResult
    arclength: float
    status: FiberStatus
    reason: TerminationReason


@dataclass(frozen=True)
class IncompleteCandidate:
    """An admissible candidate whose discovery trace did not close.

    It is evidence of nothing conclusive: neither a distinct component nor a
    duplicate of one.  ``result`` keeps the truncated trace for diagnosis.
    """

    seed: np.ndarray
    result: FiberResult
    status: FiberStatus
    reason: TerminationReason


@dataclass(frozen=True)
class ComponentDiscoveryResult:
    """Components and unclassified candidates found for one target direction.

    ``completeness`` is procedural (see the module docstring): ``"complete"``
    iff ``incomplete`` is empty.  The count fields record the funnel of the
    prescan (pool -> clusters -> admissible) for regression and diagnosis.
    """

    components: tuple[DiscoveredComponent, ...]
    incomplete: tuple[IncompleteCandidate, ...]
    completeness: Completeness
    pool_count: int
    raw_cluster_count: int
    admissible_count: int

    @property
    def component_count(self) -> int:
        return len(self.components)

    @property
    def incomplete_count(self) -> int:
        return len(self.incomplete)


def _haar_rotations(count: int, rng: np.random.Generator) -> np.ndarray:
    """Haar-uniform rotation matrices from normalised Gaussian quaternions."""
    quaternion = rng.standard_normal((count, 4))
    quaternion /= np.linalg.norm(quaternion, axis=1, keepdims=True)
    w, x, y, z = quaternion.T
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
            np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
            np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
        ],
        axis=1,
    )


def _geodesic_cluster(rotations: np.ndarray, radius: float) -> list[list[int]]:
    """Greedy clustering by SO(3) geodesic distance to the first unassigned member."""
    count = rotations.shape[0]
    unassigned = set(range(count))
    clusters: list[list[int]] = []
    rotation_array = jnp.asarray(rotations)
    while unassigned:
        seed_index = next(iter(unassigned))
        seed_rotation = rotation_array[seed_index]
        distances = np.asarray(
            jax.vmap(lambda r: rotation_distance(seed_rotation, r))(rotation_array)
        )
        members = [i for i in unassigned if distances[i] < radius]
        clusters.append(members)
        unassigned -= set(members)
    return clusters


def _newton_correct(
    problem: FiberProblem, rotation: Array, tolerance: float, iterations: int = 30
) -> tuple[Array, float]:
    """Unconstrained Gauss-Newton from an arbitrary pose onto the target fiber.

    Unlike :func:`.continuation.retract_to_fiber` this needs no on-fiber base
    pose or phase tangent; it only pulls a prescan sample onto the fiber.
    """
    for _ in range(iterations):
        residual = target_residual(problem, rotation)
        norm = float(jnp.linalg.norm(residual))
        if norm <= tolerance:
            return rotation, norm
        jacobian = local_residual_jacobian(problem, rotation)
        delta = -jacobian.T @ jnp.linalg.solve(jacobian @ jacobian.T, residual)
        rotation = rotation @ exp(delta)
    return rotation, float(jnp.linalg.norm(target_residual(problem, rotation)))


def _correct_and_trace(
    template: FiberProblem,
    options: ContinuationOptions,
    crystal: Polyhedron,
    refractive_index: float,
    raw_rotation: Array,
) -> DiscoveredComponent | IncompleteCandidate | None:
    """Correct one candidate, gate its admissibility, and trace it once.

    ``template`` fixes the incident direction and target chart; every
    per-candidate :class:`FiberProblem` is ``replace(template, seed=...)`` so
    the JIT caches keyed on the template's closures are shared across
    candidates.  The incident direction is read from ``template`` so the
    gates cannot drift from the problem being traced; ``refractive_index`` is
    passed separately because :class:`FiberProblem` does not store it as a
    number.  Returns ``None`` when the candidate fails the admissibility
    gates (residual, domain validity, positive entry measure).
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
    result = trace_fiber(replace(template, seed=jnp.asarray(corrected_np)), options)
    if result.status != FiberStatus.CLOSED:
        return IncompleteCandidate(corrected_np, result, result.status, result.reason)
    return DiscoveredComponent(
        corrected_np,
        result,
        float(result.arclength_increments.sum()),
        result.status,
        result.reason,
    )


def dedup_components(
    records: Sequence[DiscoveredComponent | IncompleteCandidate],
    arclength_rtol: float,
    *,
    pool_count: int,
    raw_cluster_count: int,
) -> ComponentDiscoveryResult:
    """Fold closed records with the same ``(status, reason, arclength)`` fingerprint.

    Public (like ``template``/:func:`retarget_problem`) so a batch caller such
    as :mod:`.strip_pixel` can fold its own hot-start and incomplete-retry
    records with the same fingerprint rule :func:`discover_components` uses
    internally, instead of reimplementing deduplication against a private
    symbol.
    """
    components: list[DiscoveredComponent] = []
    incomplete: list[IncompleteCandidate] = []
    for record in records:
        if isinstance(record, IncompleteCandidate):
            incomplete.append(record)
            continue
        duplicate = any(
            record.status == component.status
            and record.reason == component.reason
            and np.isclose(record.arclength, component.arclength, rtol=arclength_rtol, atol=1e-6)
            for component in components
        )
        if not duplicate:
            components.append(record)
    return ComponentDiscoveryResult(
        components=tuple(components),
        incomplete=tuple(incomplete),
        completeness="complete" if not incomplete else "unknown",
        pool_count=pool_count,
        raw_cluster_count=raw_cluster_count,
        admissible_count=len(records),
    )


def _discovery_options(discovery_step_budget: int) -> ContinuationOptions:
    return ContinuationOptions(maximum_accepted_steps=discovery_step_budget)


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
    incident_direction: np.ndarray,
    refractive_index: float,
    crystal: Polyhedron,
    *,
    rng_seed: int,
    prescan_samples: int = 400_000,
    discovery_step_budget: int = 250,
    angle_tolerance_deg: float = 2.0,
    cluster_radius_rad: float = 0.3,
    arclength_rtol: float = 1e-3,
    template: FiberProblem | None = None,
) -> ComponentDiscoveryResult:
    """Discover the 3-5 fiber components reaching ``target_direction``.

    ``rng_seed`` is required so that a batch caller decides explicitly whether
    pixels share a prescan or not.  Defaults come from the
    ``explore-component-discovery`` survey: 400k samples (count stable up to
    1.6M), 2 deg tolerance, 0.3 rad cluster radius, and a 250-step discovery
    budget independent of the production continuation default.  The result's
    ``completeness`` is procedural; see the module docstring.  ``template``
    (optional) is a 3-5 problem for the same incident direction and index whose
    evaluator closures are reused via :func:`retarget_problem`.
    """
    incident = np.asarray(incident_direction, dtype=np.float64)
    target = np.asarray(target_direction, dtype=np.float64)
    rng = np.random.default_rng(rng_seed)
    rotations = _haar_rotations(prescan_samples, rng)
    evaluation = jax.vmap(
        lambda r: path_3_5(r, jnp.asarray(incident), jnp.asarray(refractive_index))
    )(jnp.asarray(rotations))
    valid = (
        (np.asarray(evaluation.entry.incidence_cosine) > 0)
        & (np.asarray(evaluation.entry.discriminant) > 0)
        & (np.asarray(evaluation.exit.incidence_cosine) > 0)
        & (np.asarray(evaluation.exit.discriminant) > 0)
    )
    alignment = np.asarray(evaluation.direction) @ target
    within = valid & (alignment >= np.cos(np.radians(angle_tolerance_deg)))
    pool_indices = np.nonzero(within)[0]
    pool_rotations = rotations[pool_indices]
    pool_alignment = alignment[pool_indices]

    clusters = _geodesic_cluster(pool_rotations, cluster_radius_rad)
    template = _problem_template(
        target,
        incident,
        refractive_index,
        pool_rotations[0] if len(pool_rotations) else np.eye(3),
        template,
    )
    options = _discovery_options(discovery_step_budget)
    records: list[DiscoveredComponent | IncompleteCandidate] = []
    for cluster in clusters:
        representative = max(cluster, key=lambda i: pool_alignment[i])
        record = _correct_and_trace(
            template, options, crystal, refractive_index, jnp.asarray(pool_rotations[representative])
        )
        if record is not None:
            records.append(record)
    return dedup_components(
        records,
        arclength_rtol,
        pool_count=int(len(pool_indices)),
        raw_cluster_count=len(clusters),
    )


def hot_start_component(
    converged_seed: np.ndarray,
    target_direction: np.ndarray,
    incident_direction: np.ndarray,
    refractive_index: float,
    crystal: Polyhedron,
    *,
    discovery_step_budget: int = 250,
    template: FiberProblem | None = None,
) -> DiscoveredComponent | IncompleteCandidate | None:
    """Re-seed a neighbouring pixel from a seed that converged on another.

    Skips the prescan and clustering and runs the same correction, gates,
    trace, and classification as :func:`discover_components` on the single
    candidate ``converged_seed``.  Returns ``None`` if the candidate is not
    admissible for the new target.  Without ``template`` a new 3-5 problem
    (new target chart, fresh closures) is built per call; with it the problem
    is :func:`retarget_problem` of the template.
    """
    problem = _problem_template(
        target_direction, incident_direction, refractive_index, converged_seed, template
    )
    return _correct_and_trace(
        problem,
        _discovery_options(discovery_step_budget),
        crystal,
        refractive_index,
        jnp.asarray(converged_seed, dtype=jnp.float64),
    )


def detect_arclength_jump(
    arclengths: Sequence[float],
    *,
    relative_threshold: float = 0.2,
) -> list[int]:
    """Indices ``i`` where ``arclengths[i] -> arclengths[i+1]`` jumps by more than
    ``relative_threshold`` relative to ``arclengths[i]``.

    Meant for a component arclength sequence along a scan line: a jump marks
    a topology change between neighbouring pixels (branch switch), where
    hot-starting from the previous seed is unsafe.  The default ``0.2`` sits
    between the one observed real boundary (row 225 -> 226 of the canonical
    strip, about 50 %) and the per-row drift inside a clean run (well under
    1 %); it is calibrated on that single sample only.
    """
    values = np.asarray(arclengths, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("arclengths must be a one-dimensional sequence")
    if relative_threshold <= 0.0:
        raise ValueError("relative_threshold must be positive")
    if values.size < 2:
        return []
    change = np.abs(np.diff(values)) / np.abs(values[:-1])
    return [int(i) for i in np.nonzero(change > relative_threshold)[0]]


def is_floor_locked(
    step_diagnostics: Sequence[StepDiagnostic],
    *,
    minimum_step: float,
    window: int,
) -> bool:
    """``True`` iff the last ``window`` accepted steps all proposed ``minimum_step``.

    Read-only criterion for the "step floor never left" stall diagnosed by
    ``explore-continuation-degenerate-stall-diagnosis``: once some domain
    margin sits below ``ContinuationOptions.event_slowdown_margin`` without
    tending to zero, ``_adapt_accepted_step`` only ever shrinks the step, and
    after it clamps to ``minimum_step`` the trace crawls there until its
    budget runs out.  A caller that owns a budget-limited trace whose reason
    is :attr:`TerminationReason.STEP_BUDGET` can therefore decide, from the
    trace it already paid for, that a retrace with a larger budget of the
    same seed and the same numerics would crawl the same way (the first steps
    are deterministic), and skip it.

    Only ``accepted`` entries count; a rejected trial that shrank the step is
    not an accepted step at the floor.  The comparison is exact (``<=``):
    ``_adapt_accepted_step`` writes ``max(minimum_step, step)``, so a step at
    the floor *is* ``minimum_step``, not an approximation of it.  Fewer than
    ``window`` accepted steps, or any step above the floor inside the window,
    gives ``False``; the window must be positive.  ``window`` is a procedural
    calibration (task-discovery-stall-early-exit Step 0), not a proof: it is
    the smallest trailing run observed on stalled candidates with a margin
    over the longest run observed on candidates that a production-budget
    retrace did close.
    """
    if window < 1:
        raise ValueError("window must be positive")
    if minimum_step <= 0.0:
        raise ValueError("minimum_step must be positive")
    trailing = 0
    for diagnostic in reversed(step_diagnostics):
        if not diagnostic.accepted:
            continue
        if diagnostic.proposed_step > minimum_step:
            return False
        trailing += 1
        if trailing >= window:
            return True
    return False
