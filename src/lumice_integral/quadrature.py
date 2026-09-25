"""Line quadrature of the partial physical integrand on one fiber.

This module is the single authority for the scalar

    I_(P,C)(d) = (1 / 8 pi^2) * int_C rho_H(R) W_P(R) / (J_perp F_P(R) + eps) dH^1_g(R)

of ``docs/phase1-math-contract.md`` section 7 restricted to the factors Phase I
can evaluate today.  ``W_P`` is the product of the *explicitly named* factors
:data:`INTEGRAND_FACTOR_NAMES`; ``rho_H`` (``rho_pose``) is the density of the
pose law relative to Haar probability and is required, not optional.  The
``1 / (8 pi^2)`` Haar-to-``dVol_g`` conversion is applied exactly once, in
:func:`integrate_fiber_resampled`, and reported next to the raw value.

Method (task-resample-and-integrate; the earlier adaptive chord-parametrised
Simpson integrator with one host-side Newton retraction per refinement node
was removed in the same task, its alignment values are frozen in
``tests/test_resample_quadrature.py``).  The accepted samples of a
:class:`.continuation.FiberResult` define a C^1 predictor curve ``P(t)``
(:mod:`.resample`: a cubic Hermite quaternion spline with the trace's exact
tangents, ``t`` the cumulative chord).  On a uniform grid ``t_k`` every
predictor pose is retracted onto the fiber in one batch,

    gamma(t) = P(t) exp(delta(t)),   nu(t) . delta(t) = 0,

with ``nu`` the predictor's unit body velocity as phase tangent
(:func:`.continuation.retract_to_fiber_batch`), and the arclength speed
``ds/dt = |gamma^{-1} gamma'|`` follows from the implicit function theorem:

    lambda tau - dexp_delta(delta') = exp(delta)^T vee(P^T P'),   nu . delta' = -nu' . delta,

a 4x4 linear system in ``(lambda, delta')`` (:func:`_parametric_speed`) whose
``dexp`` is taken from ``jax.jacfwd`` of :func:`.so3.exp` and whose ``nu'`` is
analytic from the spline (:attr:`.resample.ResampledPredictors.phase_tangent_rates`).
The ``-nu' . delta`` term matters although ``delta`` is small: ``delta`` is set
by the fixed spline predictor, not by the grid, so dropping it biased ``ds/dt``
by ``O(delta)`` at every node count (canonical pixel ~5.6e-6 low against the
Phase II contour quadrature; task phase1-quadrature-start-and-speed).  Composite Simpson
is applied to ``g(t) = f(gamma(t)) lambda(t)`` on the grid, so the sum is an
exact parametrisation of the ``dH^1_g`` integral up to the quadrature error;
the error estimate compares the grid with its every-other-node subset and the
node count doubles until the estimate meets the tolerance or a declared
maximum.  ``t`` is never reported as arclength.

Everything here is host-side post-processing of a traced curve (a
:class:`FiberResult`, or an :class:`.resample.OpenArc` stitched from a forward
and a backward trace of one seed, task-pixel-pipeline-v2); nothing changes
``trace_fiber`` or its termination decisions.  An ``OpenArc`` has an event at
*both* ends, so it gets a start-side truncation estimate in addition to the
terminal one (:func:`_endpoint_truncation`).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from .continuation import (
    FiberProblem,
    FiberResult,
    FiberStatus,
    TerminationReason,
    arclength_to_event,
    retract_to_fiber_batch,
)
from .resample import FiberSpline, OpenArc, TraceLike, fiber_spline, resample_spline, uniform_parameters
from .so3 import exp, vee
from .weights import evaluate_weights_batch

HAAR_TO_DVOL_G_FACTOR = 1.0 / (8.0 * np.pi**2)
DENSITY_FACTOR_NAME = "rho_pose"
# Deliberately explicit: W_P changes only when this tuple changes.  Adding a
# newly available factor (visibility, source_factor, ...) is a code change here.
INTEGRAND_FACTOR_NAMES = ("entry_measure", "fresnel_transmission", "path_validity")
COVERAGE_NOTE = "partial: one component reached from one seed; completeness unknown"


def integrand_availability(problem: FiberProblem) -> str:
    """``available`` or ``unavailable_missing_<factor>`` for the first missing factor."""
    for name in (DENSITY_FACTOR_NAME, *INTEGRAND_FACTOR_NAMES):
        if name not in problem.weight_evaluators:
            return f"unavailable_missing_{name}"
    return "available"


def integrand_expression(
    density: np.ndarray | float,
    factor_product: np.ndarray | float,
    normal_jacobian: np.ndarray | float,
    epsilon: float,
) -> np.ndarray | float:
    """The one place that combines ``rho_H * W_P / (J_perp + eps)``."""
    return density * factor_product / (normal_jacobian + epsilon)


def pointwise_integrand(result: FiberResult, *, epsilon: float) -> np.ndarray:
    """``rho_H W_P / (J_perp + eps)`` at every accepted pose, aligned with ``result.poses``.

    Uses only the already evaluated ``result.weight_observables`` and
    ``result.jacobian_diagnostics``; no refinement nodes are inserted.  Raises
    ``ValueError("... unavailable_missing_<factor>")`` instead of substituting
    one for a missing factor.
    """
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    observables = result.weight_observables
    for name in (DENSITY_FACTOR_NAME, *INTEGRAND_FACTOR_NAMES):
        observable = observables.get(name)
        if observable is None or observable.status != "available":
            raise ValueError(f"pointwise integrand is unavailable_missing_{name}")
    sample_count = len(result.poses)
    if len(result.jacobian_diagnostics) != sample_count:
        raise ValueError("jacobian diagnostics do not align with poses")
    density = np.asarray(observables[DENSITY_FACTOR_NAME].values, dtype=np.float64)
    factor_product = np.ones(sample_count, dtype=np.float64)
    for name in INTEGRAND_FACTOR_NAMES:
        factor_product = factor_product * np.asarray(
            observables[name].values, dtype=np.float64
        )
    normal_jacobian = np.asarray(
        [diagnostic.normal_jacobian for diagnostic in result.jacobian_diagnostics],
        dtype=np.float64,
    )
    values = integrand_expression(density, factor_product, normal_jacobian, epsilon)
    return np.asarray(values, dtype=np.float64).reshape(sample_count)


def _parametric_speed(
    tangent: Array, predictor_velocity: Array, phase_tangent: Array, phase_tangent_rate: Array, delta: Array
) -> Array:
    """``ds/dt`` of ``gamma(t) = P(t) exp(delta(t))`` at one point.

    ``P`` is a predictor curve with body velocity ``predictor_velocity``
    (``vee(P^T dP/dt)``), ``delta`` the retraction offset kept orthogonal to
    ``phase_tangent`` (whose ``t`` derivative is ``phase_tangent_rate``), and
    ``tangent`` the unit fiber tangent at ``gamma`` (oriented along
    ``phase_tangent``).  Differentiating ``gamma^{-1} gamma' = lambda tangent``
    and ``phase_tangent . delta = 0`` gives the 4x4 linear system
    ``lambda tangent - dexp_delta(delta') = exp(delta)^T predictor_velocity``,
    ``phase_tangent . delta' = -phase_tangent_rate . delta`` in
    ``(lambda, delta')``; returns ``lambda``.
    On a geodesic fiber (the analytic circle) ``delta = 0`` and ``lambda`` is
    the predictor's own speed.
    """
    rotation = exp(delta)
    # dexp in the body frame: column k is vee(exp(delta)^T d exp(delta) / d delta_k).
    derivative = jax.jacfwd(exp)(delta)
    body_derivative = jnp.einsum("ji,jlk->ilk", rotation, derivative)
    dexp = jnp.stack([vee(body_derivative[:, :, k]) for k in range(3)], axis=1)
    system = jnp.zeros((4, 4), dtype=delta.dtype)
    system = system.at[:3, 0].set(tangent)
    system = system.at[:3, 1:].set(-dexp)
    system = system.at[3, 1:].set(phase_tangent)
    right_hand_side = jnp.concatenate(
        (rotation.T @ predictor_velocity, -jnp.dot(phase_tangent_rate, delta)[None])
    )
    return jnp.linalg.solve(system, right_hand_side)[0]


_parametric_speed_batch_kernel = jax.jit(jax.vmap(_parametric_speed))


# --- Resampled fixed-grid quadrature ----------------------------------------


RESAMPLED_QUADRATURE_METHOD = (
    "composite Simpson on a uniform grid of the cumulative-chord parameter of a "
    "C1 cubic Hermite quaternion spline through the accepted poses (exact fiber "
    "tangents at the knots); every grid node retracted onto the fiber by a fixed "
    "number of batched bordered Newton iterations; arclength speed ds/dt from "
    "the implicit function theorem at the retracted node; error estimate "
    "|I_N - I_(N+1)/2| with the node count doubled (N -> 2N-1) until it meets the "
    "relative tolerance or maximum_node_count"
)


def _is_simpson_doubling_count(node_count: int) -> bool:
    """``N = 4k + 1``: even panel count for Simpson at ``N`` and at ``(N + 1) / 2``."""
    return node_count >= 5 and (node_count - 1) % 4 == 0


@dataclass(frozen=True)
class ResampleOptions:
    """Observable numerical policy of :func:`integrate_fiber_resampled`.

    ``initial_node_count`` and every doubled count must be ``4k + 1`` so the
    Simpson rule applies at ``N`` and at the every-other-node subset
    ``(N + 1) / 2`` that provides the error estimate.  Defaults from the
    task-resample-and-integrate Step 5 evidence on the canonical pixel and
    rows 100/300/500 (col 126) of the ch06 strip: the slope jumps of
    ``entry_measure`` make the uniform grid converge at order ~2, so
    ``relative_tolerance = 1e-4`` (final grids 129-513 nodes, 13-26 ms) is
    what keeps every fixture within 1e-4 of the rtol=1e-8 adaptive reference;
    ``1e-3`` stops at 129 nodes and misses that on the longer loops (1e-4 and
    2e-4).  Two Newton iterations take the spline predictor's ~1e-6 residual
    to round-off (one leaves ~1e-11).
    """

    epsilon: float = 1e-6
    relative_tolerance: float = 1e-4
    initial_node_count: int = 129
    maximum_node_count: int = 1025
    retraction_iterations: int = 2

    def __post_init__(self) -> None:
        if not (self.epsilon > 0.0 and math.isfinite(self.epsilon)):
            raise ValueError("epsilon must be a positive finite number")
        if not (self.relative_tolerance > 0.0 and math.isfinite(self.relative_tolerance)):
            raise ValueError("relative_tolerance must be a positive finite number")
        if not _is_simpson_doubling_count(self.initial_node_count):
            raise ValueError("initial_node_count must be 4k + 1 with k >= 1")
        if self.maximum_node_count < self.initial_node_count:
            raise ValueError("maximum_node_count must be at least initial_node_count")
        if self.retraction_iterations < 1:
            raise ValueError("retraction_iterations must be positive")


@dataclass(frozen=True)
class ResampledQuadratureResult:
    """Scalar partial integral of one fiber by the resampled fixed-grid method.

    ``value``/``error_estimate``/``endpoint_truncation_estimate`` carry the
    ``1 / (8 pi^2)`` Haar conversion; the ``raw_*`` fields do not.
    ``node_count`` is the final grid (``4k + 1`` nodes; on a closed loop the
    last node repeats the first), ``node_count_history`` every
    ``(N, raw I_N)`` pair of the doubling sequence including the coarsest
    half grid.  ``node_count_exhausted`` is ``True`` when the last error
    estimate still exceeded the tolerance but the next doubling would pass
    ``maximum_node_count``: the value is reported as is, not as converged.
    ``residual_before_*`` are target-chart residual norms of the spline
    predictor (distance off the fiber), ``residual_after_*`` after the fixed
    Newton iterations; ``non_finite_node_count`` nodes whose retraction or
    factors were non-finite (their integrand is taken as ``0`` and counted,
    never hidden).  ``endpoint_truncation_estimate`` (open arcs only) is the
    terminal integrand value times the linear-rate arclength to the nearest
    event (:func:`.continuation.arclength_to_event`) at the curve's *terminal*
    end.  For a one-sided :class:`FiberResult` the start is the seed, not a
    boundary, and ``start_endpoint_truncation_estimate`` is ``nan``; for a
    stitched :class:`.resample.OpenArc` the start is the backward trace's
    event and gets the mirrored estimate.  Naming note: ``endpoint_truncation_*``
    keeps its task-resample-and-integrate name (it is read by ``figure_data``
    and frozen in ``tests/test_resample_quadrature.py``) and still means the
    terminal end; ``start_endpoint_truncation_*`` is the newer, prefixed name
    for the other end.  They are a historical name plus a new one, not a
    symmetric start/end pair.  ``factor_seconds`` is the wall clock of every
    batch factor evaluation (``entry_measure`` is a host loop).  Never claims
    component completeness.
    """

    status: str
    method: str
    fiber_status: str
    factor_names: tuple[str, ...]
    density_factor_name: str
    epsilon: float
    relative_tolerance: float
    initial_node_count: int
    maximum_node_count: int
    retraction_iterations: int
    node_count: int
    refinement_rounds: int
    node_count_exhausted: bool
    node_count_history: tuple[tuple[int, float], ...]
    value: float
    raw_value: float
    error_estimate: float
    raw_error_estimate: float
    haar_to_dvol_g_factor: float
    residual_before_max: float
    residual_before_median: float
    residual_after_max: float
    residual_after_median: float
    non_finite_node_count: int
    endpoint_truncation_estimate: float
    endpoint_truncation_note: str
    factor_seconds: Mapping[str, float]
    start_endpoint_truncation_estimate: float = float("nan")
    start_endpoint_truncation_note: str = "one-sided trace: the start is the seed, not a boundary"
    coverage: str = COVERAGE_NOTE
    component_completeness: str = "unknown"


@dataclass(frozen=True)
class _GridNodes:
    """Per-node quantities of one uniform grid (all arrays aligned, length ``N``)."""

    parameters: np.ndarray
    integrand: np.ndarray
    speed: np.ndarray
    residual_before: np.ndarray
    residual_after: np.ndarray
    finite: np.ndarray
    factor_seconds: dict[str, float]

    @property
    def weighted(self) -> np.ndarray:
        """``g = f * ds/dt``, the Simpson integrand on the parameter grid."""
        return self.integrand * self.speed

    def interleave(self, odd: "_GridNodes") -> "_GridNodes":
        """Merge this grid (even nodes of the doubled grid) with the new odd nodes."""
        assert len(odd.parameters) == len(self.parameters) - 1

        def merge(even: np.ndarray, new: np.ndarray) -> np.ndarray:
            merged = np.empty(len(even) + len(new), dtype=even.dtype)
            merged[0::2] = even
            merged[1::2] = new
            return merged

        seconds = {
            name: self.factor_seconds.get(name, 0.0) + odd.factor_seconds.get(name, 0.0)
            for name in set(self.factor_seconds) | set(odd.factor_seconds)
        }
        return _GridNodes(
            merge(self.parameters, odd.parameters),
            merge(self.integrand, odd.integrand),
            merge(self.speed, odd.speed),
            merge(self.residual_before, odd.residual_before),
            merge(self.residual_after, odd.residual_after),
            merge(self.finite, odd.finite),
            seconds,
        )


def _evaluate_grid_nodes(
    problem: FiberProblem,
    spline: FiberSpline,
    parameters: np.ndarray,
    options: ResampleOptions,
) -> _GridNodes:
    """Predict, retract, evaluate factors and ``ds/dt`` at ``parameters`` in batch."""
    predictors = resample_spline(spline, parameters)
    retraction = retract_to_fiber_batch(
        problem, predictors.rotations, predictors.phase_tangents,
        iterations=options.retraction_iterations,
    )
    rotations = retraction.rotations
    names = (DENSITY_FACTOR_NAME, *INTEGRAND_FACTOR_NAMES)
    weights = evaluate_weights_batch(problem.weight_evaluators, rotations, names)
    factor_product = np.ones(len(parameters), dtype=np.float64)
    for name in INTEGRAND_FACTOR_NAMES:
        factor_product = factor_product * weights.values[name]
    speed = np.asarray(
        _parametric_speed_batch_kernel(
            jnp.asarray(retraction.tangents),
            jnp.asarray(predictors.body_velocities),
            jnp.asarray(predictors.phase_tangents),
            jnp.asarray(predictors.phase_tangent_rates),
            jnp.asarray(retraction.deltas),
        ),
        dtype=np.float64,
    )
    integrand = np.asarray(
        integrand_expression(
            weights.values[DENSITY_FACTOR_NAME], factor_product,
            retraction.normal_jacobians, options.epsilon,
        ),
        dtype=np.float64,
    )
    finite = retraction.finite & np.isfinite(integrand) & np.isfinite(speed)
    integrand = np.where(finite, integrand, 0.0)
    speed = np.where(finite, speed, 0.0)
    return _GridNodes(
        np.asarray(parameters, dtype=np.float64), integrand, speed,
        retraction.residual_before, retraction.residual_after, finite, weights.seconds,
    )


def _composite_simpson(values: np.ndarray, spacing: float) -> float:
    """Composite Simpson sum of uniformly spaced ``values`` (odd length)."""
    assert len(values) % 2 == 1 and len(values) >= 3
    return float(
        spacing / 3.0 * (values[0] + values[-1] + 4.0 * np.sum(values[1:-1:2]) + 2.0 * np.sum(values[2:-1:2]))
    )


_EVENT_REASONS = {
    TerminationReason.TIR_BOUNDARY, TerminationReason.BRANCH_BOUNDARY,
    TerminationReason.PATH_INFEASIBLE, TerminationReason.VISIBILITY_BOUNDARY,
    TerminationReason.CHART_BOUNDARY, TerminationReason.RANK_LOSS,
    TerminationReason.TOPOLOGY_AMBIGUITY,
}


def _event_truncation(
    reason: TerminationReason,
    margins: Mapping[str, float],
    previous_margins: Mapping[str, float],
    advance: float,
    integrand: float,
    end: str,
) -> tuple[float, str]:
    """Raw truncation estimate beyond one event-terminated end of a curve.

    ``margins`` belong to the end pose, ``previous_margins`` to its neighbour
    along the curve and ``advance`` to the edge between them; ``end`` names
    the end in the note (``"terminal"`` or ``"start"``).
    """
    if reason not in _EVENT_REASONS:
        return float("nan"), (
            f"open arc ended by {reason.value}, not by an event: the missing "
            f"arclength beyond the {end} pose is unbounded by any margin"
        )
    distance = arclength_to_event(margins, previous_margins, advance)
    if not np.isfinite(distance):
        return float("nan"), (
            f"open arc ended by {reason.value} but no margin decreased over "
            f"the last accepted edge at the {end} end: no linear-rate distance to the event"
        )
    return float(integrand * distance), (
        f"{end} integrand value times the linear-rate arclength from the {end} "
        f"accepted pose to the {reason.value} event "
        f"(continuation.arclength_to_event: {distance:.3e})"
    )


def _endpoint_truncation(result: TraceLike, terminal_integrand: float) -> tuple[float, str]:
    """Raw open-arc truncation estimate at the terminal end, with its note.

    The terminal end is the forward trace's end for a one-sided
    :class:`FiberResult` and for a stitched :class:`.resample.OpenArc` alike;
    the one-sided note states that the seed end is not a boundary, the
    two-sided start is handled by :func:`_start_truncation`.
    """
    if result.status == FiberStatus.CLOSED:
        return float("nan"), "closed loop: no endpoints"
    margins = result.branch_diagnostics.accepted_margins
    if len(margins) < 2:
        return float("nan"), "open arc with a single accepted pose: no margin rate"
    estimate, note = _event_truncation(
        result.reason, margins[-1], margins[-2], float(result.arclength_increments[-1]),
        terminal_integrand, "terminal",
    )
    if not isinstance(result, OpenArc):
        note += "; the seed end is not a boundary"
    return estimate, note


def _start_truncation(result: TraceLike, start_integrand: float) -> tuple[float, str]:
    """Raw truncation estimate at the start end: only a stitched :class:`OpenArc` has one."""
    if not isinstance(result, OpenArc):
        return float("nan"), ResampledQuadratureResult.start_endpoint_truncation_note
    margins = result.branch_diagnostics.accepted_margins
    if result.seed_index < 1 or len(margins) < 2:
        # The backward trace met its event at the seed: the start *is* the
        # seed pose and the only edge is the forward one, whose margin rate
        # describes the wrong end.
        return float("nan"), (
            f"open arc whose start end is the seed itself ({result.start_reason.value} "
            "met before any backward step): no margin rate at the start"
        )
    return _event_truncation(
        result.start_reason, margins[0], margins[1], float(result.arclength_increments[0]),
        start_integrand, "start",
    )


def _resampled_unavailable(
    status: str, result: TraceLike, options: ResampleOptions
) -> ResampledQuadratureResult:
    return ResampledQuadratureResult(
        status=status,
        method=RESAMPLED_QUADRATURE_METHOD,
        fiber_status=result.status.value,
        factor_names=INTEGRAND_FACTOR_NAMES,
        density_factor_name=DENSITY_FACTOR_NAME,
        epsilon=options.epsilon,
        relative_tolerance=options.relative_tolerance,
        initial_node_count=options.initial_node_count,
        maximum_node_count=options.maximum_node_count,
        retraction_iterations=options.retraction_iterations,
        node_count=0,
        refinement_rounds=0,
        node_count_exhausted=False,
        node_count_history=(),
        value=float("nan"),
        raw_value=float("nan"),
        error_estimate=float("nan"),
        raw_error_estimate=float("nan"),
        haar_to_dvol_g_factor=HAAR_TO_DVOL_G_FACTOR,
        residual_before_max=float("nan"),
        residual_before_median=float("nan"),
        residual_after_max=float("nan"),
        residual_after_median=float("nan"),
        non_finite_node_count=0,
        endpoint_truncation_estimate=float("nan"),
        endpoint_truncation_note=f"not computed: {status}",
        factor_seconds={},
        start_endpoint_truncation_note=f"not computed: {status}",
    )


def integrate_fiber_resampled(
    problem: FiberProblem,
    result: TraceLike,
    options: ResampleOptions | None = None,
) -> ResampledQuadratureResult:
    """Integrate the partial physical integrand along ``result`` on a resampled grid.

    ``result`` is a :class:`FiberResult` or a stitched :class:`.resample.OpenArc`
    (any :class:`.resample.TraceLike`).
    See :data:`RESAMPLED_QUADRATURE_METHOD`.  The accepted samples define the
    predictor spline (:mod:`.resample`); every uniform grid node is retracted
    onto the fiber in one batch (:func:`.continuation.retract_to_fiber_batch`),
    the four named factors and ``J_perp`` are evaluated in batch, and the
    composite Simpson sum of ``f * ds/dt`` over the parameter grid is compared
    with the same sum over every other node.  Doubling reuses the previous
    grid as the even nodes of the next.  The only correctness evidence is the
    external alignment recorded in ``tests/test_resample_quadrature.py``; the
    internal ``|I_N - I_(N+1)/2|`` estimate is self-consistency, not proof.
    """
    options = options or ResampleOptions()
    status = integrand_availability(problem)
    if status != "available":
        return _resampled_unavailable(status, result, options)
    minimum_samples = 3 if result.status == FiberStatus.CLOSED else 2
    if len(result.poses) < minimum_samples:
        return _resampled_unavailable("unavailable_no_edges", result, options)

    spline = fiber_spline(result)
    node_count = options.initial_node_count
    grid = _evaluate_grid_nodes(problem, spline, uniform_parameters(spline, node_count), options)
    history: list[tuple[int, float]] = []
    rounds = 0
    while True:
        spacing = spline.total / (node_count - 1)
        fine = _composite_simpson(grid.weighted, spacing)
        coarse = _composite_simpson(grid.weighted[0::2], 2.0 * spacing)
        if not history:
            history.append(((node_count + 1) // 2, coarse))
        history.append((node_count, fine))
        error = abs(fine - coarse)
        if error <= options.relative_tolerance * abs(fine):
            exhausted = False
            break
        if 2 * node_count - 1 > options.maximum_node_count:
            exhausted = True
            break
        rounds += 1
        odd_parameters = uniform_parameters(spline, 2 * node_count - 1)[1::2]
        grid = grid.interleave(_evaluate_grid_nodes(problem, spline, odd_parameters, options))
        node_count = 2 * node_count - 1

    truncation, truncation_note = _endpoint_truncation(result, float(grid.integrand[-1]))
    start_truncation, start_truncation_note = _start_truncation(result, float(grid.integrand[0]))
    return ResampledQuadratureResult(
        status="available",
        method=RESAMPLED_QUADRATURE_METHOD,
        fiber_status=result.status.value,
        factor_names=INTEGRAND_FACTOR_NAMES,
        density_factor_name=DENSITY_FACTOR_NAME,
        epsilon=options.epsilon,
        relative_tolerance=options.relative_tolerance,
        initial_node_count=options.initial_node_count,
        maximum_node_count=options.maximum_node_count,
        retraction_iterations=options.retraction_iterations,
        node_count=node_count,
        refinement_rounds=rounds,
        node_count_exhausted=exhausted,
        node_count_history=tuple(history),
        value=fine * HAAR_TO_DVOL_G_FACTOR,
        raw_value=fine,
        error_estimate=error * HAAR_TO_DVOL_G_FACTOR,
        raw_error_estimate=error,
        haar_to_dvol_g_factor=HAAR_TO_DVOL_G_FACTOR,
        residual_before_max=float(np.nanmax(grid.residual_before)),
        residual_before_median=float(np.nanmedian(grid.residual_before)),
        residual_after_max=float(np.nanmax(grid.residual_after)),
        residual_after_median=float(np.nanmedian(grid.residual_after)),
        non_finite_node_count=int(np.count_nonzero(~grid.finite)),
        endpoint_truncation_estimate=truncation * HAAR_TO_DVOL_G_FACTOR,
        endpoint_truncation_note=truncation_note,
        factor_seconds=dict(grid.factor_seconds),
        start_endpoint_truncation_estimate=start_truncation * HAAR_TO_DVOL_G_FACTOR,
        start_endpoint_truncation_note=start_truncation_note,
    )


__all__ = [
    "COVERAGE_NOTE",
    "DENSITY_FACTOR_NAME",
    "HAAR_TO_DVOL_G_FACTOR",
    "INTEGRAND_FACTOR_NAMES",
    "RESAMPLED_QUADRATURE_METHOD",
    "ResampleOptions",
    "ResampledQuadratureResult",
    "integrand_availability",
    "integrand_expression",
    "integrate_fiber_resampled",
    "pointwise_integrand",
]
