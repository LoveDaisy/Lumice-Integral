"""Adaptive line quadrature of the partial physical integrand on one fiber.

This module is the single authority for the scalar

    I_(P,C)(d) = (1 / 8 pi^2) * int_C rho_H(R) W_P(R) / (J_perp F_P(R) + eps) dH^1_g(R)

of ``docs/phase1-math-contract.md`` section 7 restricted to the factors Phase I
can evaluate today.  ``W_P`` is the product of the *explicitly named* factors
:data:`INTEGRAND_FACTOR_NAMES`; ``rho_H`` (``rho_pose``) is the density of the
pose law relative to Haar probability and is required, not optional.  The
``1 / (8 pi^2)`` Haar-to-``dVol_g`` conversion is applied exactly once, in
:func:`integrate_fiber`, and reported next to the raw value.

Geometry of one panel.  Every edge between consecutive fiber nodes ``R_l`` and
``R_r`` is parametrised by its chord ``xi = log(R_l^T R_r)``:

    gamma(u) = R_l exp(u xi) exp(delta(u)),   u in [0, 1],   xi_hat . delta(u) = 0,

where ``exp(delta(u))`` is the continuation corrector's retraction onto the
fiber (:func:`.continuation.retract_to_fiber` with phase tangent ``xi_hat``).
The endpoints are exact (``delta(0) = delta(1) = 0``) and the arclength speed
``ds/du = |gamma^{-1} gamma'|`` follows from the implicit function theorem:

    lambda t(gamma(u)) - dexp_delta(delta') = exp(delta)^T xi,   xi_hat . delta' = 0,

a 4x4 linear system in ``(lambda, delta')`` whose ``dexp`` is taken from
``jax.jacfwd`` of :func:`.so3.exp`.  Simpson's rule is then applied to
``g(u) = f(gamma(u)) lambda(u)`` on ``u in [0, 1]``, so each (sub)panel is an
exact parametrisation of the same ``dH^1_g`` integral and the classical
Richardson error estimate ``|S_whole - S_half| / 15`` applies.  On a geodesic
fiber (the analytic circle) ``delta = 0`` and ``lambda = |xi|`` identically.

Everything here is host-side numpy post-processing of a :class:`FiberResult`;
nothing changes ``trace_fiber`` or its termination decisions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from .continuation import (
    ContinuationOptions,
    FiberProblem,
    FiberResult,
    retract_to_fiber,
)
from .so3 import exp, log

HAAR_TO_DVOL_G_FACTOR = 1.0 / (8.0 * np.pi**2)
DENSITY_FACTOR_NAME = "rho_pose"
# Deliberately explicit: W_P changes only when this tuple changes.  Adding a
# newly available factor (visibility, source_factor, ...) is a code change here.
INTEGRAND_FACTOR_NAMES = ("entry_measure", "fresnel_transmission", "path_validity")
QUADRATURE_METHOD = (
    "adaptive composite Simpson with recursive bisection over chord-parametrised "
    "SO(3) fiber edges; refinement nodes corrector-retracted onto the fiber; "
    "arclength speed from the implicit function theorem; Richardson error "
    "estimate |S_whole - S_half| / 15 with Richardson-extrapolated panel values"
)
COVERAGE_NOTE = "partial: one component reached from one seed; completeness unknown"
_ORDER_ESTIMATE_LEVELS = 2
_ORDER_ESTIMATE_FLOOR = 1e-14


@dataclass(frozen=True)
class QuadratureOptions:
    """Observable numerical policy of the line quadrature."""

    # Defaults calibrated on the canonical pixel fiber (see
    # docs/ch06-reference-fixture.md section 4.1): relative_tolerance=1e-8 needs
    # depth 16 and 1e-9 depth 20 at the slope jumps of entry_measure, so 24
    # leaves headroom; the cost of depth is linear (only kink panels go deep).
    epsilon: float = 1e-6
    relative_tolerance: float = 1e-8
    maximum_refinement_depth: int = 24
    # Uniform bisection levels of the separate order-estimate pass (diagnostic
    # only; it never changes ``value``).  ``0`` skips the pass and reports the
    # order as not estimated: an image driver that integrates ~10^5 fibers
    # cannot afford the ~50 % of quadrature wall clock it costs per fiber
    # (task-strip-image-driver Step 0: 714 of 1377 nodes on the canonical pixel).
    convergence_order_levels: int = _ORDER_ESTIMATE_LEVELS

    def __post_init__(self) -> None:
        if not (self.epsilon > 0.0 and math.isfinite(self.epsilon)):
            raise ValueError("epsilon must be a positive finite number")
        if not (self.relative_tolerance > 0.0 and math.isfinite(self.relative_tolerance)):
            raise ValueError("relative_tolerance must be a positive finite number")
        if self.maximum_refinement_depth < 1:
            raise ValueError("maximum_refinement_depth must be positive")
        if self.convergence_order_levels not in (0, _ORDER_ESTIMATE_LEVELS):
            raise ValueError(
                f"convergence_order_levels must be 0 or {_ORDER_ESTIMATE_LEVELS}"
            )


@dataclass(frozen=True)
class ConvergenceOrderEstimate:
    """Empirical Richardson orders from three uniform bisection levels.

    ``order`` is the global estimate from the fiber totals; ``edge_orders`` the
    same estimate per edge (``None`` where the differences sit at the
    floating-point floor).  A piecewise-smooth integrand (e.g. a slope jump of
    ``entry_measure`` inside an edge) drags the global order towards 2 while
    the smooth edges still show Simpson's 4; ``median_edge_order`` and
    ``low_order_edges`` (empirical order below 3) make that visible.
    """

    order: float | None
    levels: tuple[float, ...]
    note: str
    edge_orders: tuple[float | None, ...] = ()
    median_edge_order: float | None = None
    low_order_edges: tuple[int, ...] = ()


@dataclass(frozen=True)
class QuadratureResult:
    """Scalar partial integral of one fiber with explicit convergence evidence.

    ``value``/``error_estimate`` carry the ``1 / (8 pi^2)`` Haar conversion;
    ``raw_value``/``raw_error_estimate`` do not.  ``node_count`` counts the
    distinct fiber poses used by the adaptive pass (accepted continuation
    samples plus retracted midpoints); the order estimate's extra nodes are
    reported separately.  Never claims component completeness.
    """

    status: str
    method: str
    fiber_status: str
    factor_names: tuple[str, ...]
    density_factor_name: str
    epsilon: float
    relative_tolerance: float
    maximum_refinement_depth: int
    refinements: int
    node_count: int
    value: float
    raw_value: float
    error_estimate: float
    raw_error_estimate: float
    haar_to_dvol_g_factor: float
    convergence_order_estimate: float | None
    convergence_order_note: str
    raw_convergence_order_levels: tuple[float, ...]
    convergence_order_node_count: int
    median_edge_convergence_order: float | None
    low_order_edges: tuple[int, ...]
    maximum_depth_reached: int
    refinement_failures: tuple[str, ...]
    depth_exhausted_edges: tuple[int, ...]
    coverage: str = COVERAGE_NOTE
    component_completeness: str = "unknown"


def integrand_availability(problem: FiberProblem) -> str:
    """``available`` or ``unavailable_missing_<factor>`` for the first missing factor."""
    for name in (DENSITY_FACTOR_NAME, *INTEGRAND_FACTOR_NAMES):
        if name not in problem.weight_evaluators:
            return f"unavailable_missing_{name}"
    return "available"


def _integrand_expression(
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
    values = _integrand_expression(density, factor_product, normal_jacobian, epsilon)
    return np.asarray(values, dtype=np.float64).reshape(sample_count)


def _evaluate_integrand(
    problem: FiberProblem, rotation: np.ndarray, normal_jacobian: float, epsilon: float
) -> float:
    """Same expression as :func:`pointwise_integrand` for one retracted pose."""
    evaluators = problem.weight_evaluators
    density = float(evaluators[DENSITY_FACTOR_NAME].evaluate(rotation))
    factor_product = 1.0
    for name in INTEGRAND_FACTOR_NAMES:
        factor_product *= float(evaluators[name].evaluate(rotation))
    return float(_integrand_expression(density, factor_product, normal_jacobian, epsilon))


def _vee(matrix: Array) -> Array:
    return jnp.array([matrix[2, 1], matrix[0, 2], matrix[1, 0]])


@jax.jit
def _chord_speed_kernel(tangent: Array, chord: Array, delta: Array) -> Array:
    """``ds/du`` of ``gamma(u) = R_l exp(u chord) exp(delta(u))`` at one point.

    ``tangent`` is the unit fiber tangent there (oriented along ``chord``) and
    ``delta`` the current retraction offset, orthogonal to ``chord``.  Solves
    ``lambda tangent - dexp_delta(delta') = exp(delta)^T chord`` together with
    ``chord_hat . delta' = 0`` for ``(lambda, delta')`` and returns ``lambda``.
    """
    rotation = exp(delta)
    # dexp in the body frame: column k is vee(exp(delta)^T d exp(delta) / d delta_k).
    derivative = jax.jacfwd(exp)(delta)
    body_derivative = jnp.einsum("ji,jlk->ilk", rotation, derivative)
    dexp = jnp.stack([_vee(body_derivative[:, :, k]) for k in range(3)], axis=1)
    chord_hat = chord / jnp.linalg.norm(chord)
    system = jnp.zeros((4, 4), dtype=chord.dtype)
    system = system.at[:3, 0].set(tangent)
    system = system.at[:3, 1:].set(-dexp)
    system = system.at[3, 1:].set(chord_hat)
    right_hand_side = jnp.concatenate((rotation.T @ chord, jnp.zeros(1, dtype=chord.dtype)))
    return jnp.linalg.solve(system, right_hand_side)[0]


def _chord_speed(tangent: np.ndarray, chord: np.ndarray, delta: np.ndarray) -> float:
    chord_hat = chord / np.linalg.norm(chord)
    oriented = tangent if float(np.dot(tangent, chord_hat)) >= 0.0 else -tangent
    return float(
        _chord_speed_kernel(
            jnp.asarray(oriented, dtype=jnp.float64),
            jnp.asarray(chord, dtype=jnp.float64),
            jnp.asarray(delta, dtype=jnp.float64),
        )
    )


@dataclass(frozen=True)
class _Node:
    rotation: np.ndarray
    tangent: np.ndarray
    integrand: float


@dataclass(frozen=True)
class _PanelEstimate:
    """One Simpson estimate of a panel; ``midpoint`` is ``None`` on failure."""

    value: float
    chord: float
    midpoint: _Node | None
    failure: str | None = None


def _simpson_panel_estimate(
    problem: FiberProblem,
    options: ContinuationOptions,
    left: _Node,
    right: _Node,
    *,
    epsilon: float,
) -> _PanelEstimate:
    """Single Simpson estimate of ``int f dH^1_g`` over one chord-parametrised panel.

    Shared by the adaptive pass and the uniform order-estimate pass; contains
    no refinement decision.  A failed midpoint retraction falls back to the
    trapezoid of the endpoint values and reports the corrector's reason.
    """
    chord = np.asarray(log(jnp.asarray(left.rotation.T @ right.rotation)), dtype=np.float64)
    chord_length = float(np.linalg.norm(chord))
    if chord_length == 0.0:
        return _PanelEstimate(0.0, 0.0, None)
    chord_hat = chord / chord_length
    zero = np.zeros(3, dtype=np.float64)
    g_left = left.integrand * _chord_speed(left.tangent, chord, zero)
    g_right = right.integrand * _chord_speed(right.tangent, chord, zero)
    predicted = np.asarray(left.rotation @ exp(jnp.asarray(0.5 * chord)), dtype=np.float64)
    retracted = retract_to_fiber(problem, options, left.rotation, predicted, chord_hat)
    if not retracted.accepted:
        reason = retracted.reason.value if retracted.reason is not None else "unknown"
        return _PanelEstimate(
            0.5 * (g_left + g_right),
            chord_length,
            None,
            f"midpoint retraction rejected ({reason}): {retracted.message}",
        )
    assert retracted.rotation is not None
    assert retracted.tangent is not None
    assert retracted.jacobian_diagnostic is not None
    midpoint = _Node(
        retracted.rotation,
        retracted.tangent,
        _evaluate_integrand(
            problem,
            retracted.rotation,
            retracted.jacobian_diagnostic.normal_jacobian,
            epsilon,
        ),
    )
    delta = np.asarray(log(jnp.asarray(predicted.T @ midpoint.rotation)), dtype=np.float64)
    g_middle = midpoint.integrand * _chord_speed(midpoint.tangent, chord, delta)
    return _PanelEstimate((g_left + 4.0 * g_middle + g_right) / 6.0, chord_length, midpoint)


@dataclass
class _PassAccounting:
    """Mutable bookkeeping of one quadrature pass (never part of the result)."""

    midpoints: int = 0
    refinements: int = 0
    failures: list[str] = field(default_factory=list)
    depth_exhausted: set[int] = field(default_factory=set)
    maximum_depth: int = 0

    def record(self, estimate: _PanelEstimate, label: str) -> _PanelEstimate:
        if estimate.midpoint is not None:
            self.midpoints += 1
        if estimate.failure is not None:
            self.failures.append(f"{label}: {estimate.failure}")
        return estimate


def _simpson_panel(
    problem: FiberProblem,
    options: ContinuationOptions,
    quadrature_options: QuadratureOptions,
    left: _Node,
    right: _Node,
    whole: _PanelEstimate,
    *,
    tolerance: float,
    depth: int,
    edge_index: int,
    accounting: _PassAccounting,
) -> tuple[float, float]:
    """Adaptive recursion over one panel: returns ``(value, error_estimate)``.

    ``whole`` is the panel's own single Simpson estimate (already computed so
    the midpoint is reused).  Accepted panels return the Richardson
    extrapolation ``S_half + (S_half - S_whole) / 15``.
    """
    if whole.midpoint is None:
        return whole.value, 0.0
    accounting.maximum_depth = max(accounting.maximum_depth, depth)
    label = f"edge {edge_index} depth {depth + 1}"
    left_half = accounting.record(
        _simpson_panel_estimate(
            problem, options, left, whole.midpoint, epsilon=quadrature_options.epsilon
        ),
        label,
    )
    right_half = accounting.record(
        _simpson_panel_estimate(
            problem, options, whole.midpoint, right, epsilon=quadrature_options.epsilon
        ),
        label,
    )
    half = left_half.value + right_half.value
    error = abs(whole.value - half) / 15.0
    if left_half.midpoint is None or right_half.midpoint is None:
        # A rejected refinement node: keep the unrefined estimate as evidence
        # instead of aborting the whole integral; the failure is reported.
        return whole.value, error
    if error <= tolerance:
        return half + (half - whole.value) / 15.0, error
    if depth >= quadrature_options.maximum_refinement_depth:
        accounting.depth_exhausted.add(edge_index)
        return half + (half - whole.value) / 15.0, error
    accounting.refinements += 1
    left_value, left_error = _simpson_panel(
        problem,
        options,
        quadrature_options,
        left,
        whole.midpoint,
        left_half,
        tolerance=0.5 * tolerance,
        depth=depth + 1,
        edge_index=edge_index,
        accounting=accounting,
    )
    right_value, right_error = _simpson_panel(
        problem,
        options,
        quadrature_options,
        whole.midpoint,
        right,
        right_half,
        tolerance=0.5 * tolerance,
        depth=depth + 1,
        edge_index=edge_index,
        accounting=accounting,
    )
    return left_value + right_value, left_error + right_error


def _uniform_levels(
    problem: FiberProblem,
    options: ContinuationOptions,
    left: _Node,
    right: _Node,
    whole: _PanelEstimate,
    *,
    levels: int,
    epsilon: float,
    accounting: _PassAccounting,
    label: str,
) -> list[float]:
    """Composite Simpson sums of one panel after ``0 .. levels`` uniform bisections."""
    if levels == 0 or whole.midpoint is None:
        return [whole.value] * (levels + 1)
    left_half = accounting.record(
        _simpson_panel_estimate(problem, options, left, whole.midpoint, epsilon=epsilon),
        label,
    )
    right_half = accounting.record(
        _simpson_panel_estimate(problem, options, whole.midpoint, right, epsilon=epsilon),
        label,
    )
    left_levels = _uniform_levels(
        problem, options, left, whole.midpoint, left_half,
        levels=levels - 1, epsilon=epsilon, accounting=accounting, label=label,
    )
    right_levels = _uniform_levels(
        problem, options, whole.midpoint, right, right_half,
        levels=levels - 1, epsilon=epsilon, accounting=accounting, label=label,
    )
    return [whole.value] + [a + b for a, b in zip(left_levels, right_levels)]


def _uniform_bisection_estimate(
    problem: FiberProblem,
    options: ContinuationOptions,
    nodes: list[_Node],
    wholes: list[_PanelEstimate],
    *,
    levels: int,
    epsilon: float,
    accounting: _PassAccounting,
) -> tuple[tuple[float, ...], list[tuple[float, ...]]]:
    """Raw composite Simpson values at bisection levels ``0..levels``: fiber totals and per edge."""
    totals = np.zeros(levels + 1, dtype=np.float64)
    per_edge: list[tuple[float, ...]] = []
    for index, (left, right, whole) in enumerate(zip(nodes[:-1], nodes[1:], wholes)):
        edge_levels = _uniform_levels(
            problem, options, left, right, whole,
            levels=levels, epsilon=epsilon, accounting=accounting,
            label=f"order-estimate edge {index}",
        )
        totals += edge_levels
        per_edge.append(tuple(float(level) for level in edge_levels))
    return tuple(float(total) for total in totals), per_edge


def _richardson_order(levels: tuple[float, ...]) -> float | None:
    """``log2(|I1-I0| / |I2-I1|)`` or ``None`` at the floating-point floor."""
    first, second, third = levels[-3:]
    coarse_difference = abs(second - first)
    fine_difference = abs(third - second)
    floor = _ORDER_ESTIMATE_FLOOR * max(abs(second), 1.0)
    if fine_difference <= floor or coarse_difference <= floor:
        return None
    return math.log2(coarse_difference / fine_difference)


def _order_from_levels(
    levels: tuple[float, ...], per_edge: list[tuple[float, ...]]
) -> ConvergenceOrderEstimate:
    edge_orders = tuple(_richardson_order(edge_levels) for edge_levels in per_edge)
    finite_orders = [order for order in edge_orders if order is not None]
    median = float(np.median(finite_orders)) if finite_orders else None
    low = tuple(
        index for index, order in enumerate(edge_orders) if order is not None and order < 3.0
    )
    order = _richardson_order(levels)
    if order is None:
        first, second, third = levels[-3:]
        note = (
            "global order not estimable: successive uniform refinements differ at the "
            f"floating-point floor (|I1-I0|={abs(second - first):.3e}, "
            f"|I2-I1|={abs(third - second):.3e})"
        )
    else:
        note = (
            "empirical Richardson order log2(|I1-I0| / |I2-I1|) from uniform bisection "
            "levels 0, 1, 2 of every edge (Simpson theory: 4); the global value is "
            "dominated by edges whose integrand is only piecewise smooth, see "
            "median_edge_order and low_order_edges"
        )
    return ConvergenceOrderEstimate(order, levels, note, edge_orders, median, low)


def estimate_convergence_order(
    problem: FiberProblem,
    options: ContinuationOptions,
    result: FiberResult,
    *,
    epsilon: float,
) -> ConvergenceOrderEstimate:
    """Standalone order estimate; :func:`integrate_fiber` reuses its first level."""
    nodes = _fiber_nodes(result, epsilon=epsilon)
    accounting = _PassAccounting()
    wholes = [
        accounting.record(
            _simpson_panel_estimate(problem, options, left, right, epsilon=epsilon),
            f"order-estimate edge {index}",
        )
        for index, (left, right) in enumerate(zip(nodes[:-1], nodes[1:]))
    ]
    levels, per_edge = _uniform_bisection_estimate(
        problem, options, nodes, wholes,
        levels=_ORDER_ESTIMATE_LEVELS, epsilon=epsilon, accounting=accounting,
    )
    return _order_from_levels(levels, per_edge)


def _fiber_nodes(result: FiberResult, *, epsilon: float) -> list[_Node]:
    integrand = pointwise_integrand(result, epsilon=epsilon)
    poses = np.asarray(result.poses, dtype=np.float64)
    tangents = np.asarray(result.tangents, dtype=np.float64)
    return [
        _Node(poses[index], tangents[index], float(integrand[index]))
        for index in range(len(poses))
    ]


def _unavailable(
    status: str, result: FiberResult, quadrature_options: QuadratureOptions
) -> QuadratureResult:
    return QuadratureResult(
        status=status,
        method=QUADRATURE_METHOD,
        fiber_status=result.status.value,
        factor_names=INTEGRAND_FACTOR_NAMES,
        density_factor_name=DENSITY_FACTOR_NAME,
        epsilon=quadrature_options.epsilon,
        relative_tolerance=quadrature_options.relative_tolerance,
        maximum_refinement_depth=quadrature_options.maximum_refinement_depth,
        refinements=0,
        node_count=0,
        value=float("nan"),
        raw_value=float("nan"),
        error_estimate=float("nan"),
        raw_error_estimate=float("nan"),
        haar_to_dvol_g_factor=HAAR_TO_DVOL_G_FACTOR,
        convergence_order_estimate=None,
        convergence_order_note=f"not computed: {status}",
        raw_convergence_order_levels=(),
        convergence_order_node_count=0,
        median_edge_convergence_order=None,
        low_order_edges=(),
        maximum_depth_reached=0,
        refinement_failures=(),
        depth_exhausted_edges=(),
    )


def integrate_fiber(
    problem: FiberProblem,
    result: FiberResult,
    options: ContinuationOptions | None = None,
    quadrature_options: QuadratureOptions | None = None,
) -> QuadratureResult:
    """Integrate the partial physical integrand along ``result.poses``.

    The accepted continuation samples are the initial nodes; every edge is
    refined adaptively with corrector-retracted midpoints until its Richardson
    error estimate meets its arclength-proportional share of
    ``relative_tolerance * |first-pass value|`` or ``maximum_refinement_depth``
    is reached (reported in ``depth_exhausted_edges``).  A separate uniform
    bisection pass yields the empirical convergence order.
    """
    options = options or ContinuationOptions()
    quadrature_options = quadrature_options or QuadratureOptions()
    status = integrand_availability(problem)
    if status != "available":
        return _unavailable(status, result, quadrature_options)
    if len(result.poses) < 2:
        return _unavailable("unavailable_no_edges", result, quadrature_options)
    epsilon = quadrature_options.epsilon
    nodes = _fiber_nodes(result, epsilon=epsilon)

    accounting = _PassAccounting()
    wholes = [
        accounting.record(
            _simpson_panel_estimate(problem, options, left, right, epsilon=epsilon),
            f"edge {index} depth 0",
        )
        for index, (left, right) in enumerate(zip(nodes[:-1], nodes[1:]))
    ]
    first_pass = sum(whole.value for whole in wholes)
    total_chord = sum(whole.chord for whole in wholes)
    absolute_tolerance = quadrature_options.relative_tolerance * abs(first_pass)
    raw_value = 0.0
    raw_error = 0.0
    for index, (left, right, whole) in enumerate(zip(nodes[:-1], nodes[1:], wholes)):
        share = whole.chord / total_chord if total_chord > 0.0 else 0.0
        value, error = _simpson_panel(
            problem,
            options,
            quadrature_options,
            left,
            right,
            whole,
            tolerance=absolute_tolerance * share,
            depth=0,
            edge_index=index,
            accounting=accounting,
        )
        raw_value += value
        raw_error += error

    order_accounting = _PassAccounting()
    if quadrature_options.convergence_order_levels == 0:
        order = ConvergenceOrderEstimate(
            None, (first_pass,), "order estimate skipped (convergence_order_levels=0)"
        )
    else:
        levels, per_edge = _uniform_bisection_estimate(
            problem, options, nodes, wholes,
            levels=quadrature_options.convergence_order_levels,
            epsilon=epsilon, accounting=order_accounting,
        )
        order = _order_from_levels(levels, per_edge)

    return QuadratureResult(
        status="available",
        method=QUADRATURE_METHOD,
        fiber_status=result.status.value,
        factor_names=INTEGRAND_FACTOR_NAMES,
        density_factor_name=DENSITY_FACTOR_NAME,
        epsilon=epsilon,
        relative_tolerance=quadrature_options.relative_tolerance,
        maximum_refinement_depth=quadrature_options.maximum_refinement_depth,
        refinements=accounting.refinements,
        node_count=len(nodes) + accounting.midpoints,
        value=raw_value * HAAR_TO_DVOL_G_FACTOR,
        raw_value=raw_value,
        error_estimate=raw_error * HAAR_TO_DVOL_G_FACTOR,
        raw_error_estimate=raw_error,
        haar_to_dvol_g_factor=HAAR_TO_DVOL_G_FACTOR,
        convergence_order_estimate=order.order,
        convergence_order_note=order.note,
        raw_convergence_order_levels=order.levels,
        convergence_order_node_count=order_accounting.midpoints,
        median_edge_convergence_order=order.median_edge_order,
        low_order_edges=order.low_order_edges,
        maximum_depth_reached=accounting.maximum_depth,
        refinement_failures=tuple(accounting.failures + order_accounting.failures),
        depth_exhausted_edges=tuple(sorted(accounting.depth_exhausted)),
    )


__all__ = [
    "COVERAGE_NOTE",
    "ConvergenceOrderEstimate",
    "DENSITY_FACTOR_NAME",
    "HAAR_TO_DVOL_G_FACTOR",
    "INTEGRAND_FACTOR_NAMES",
    "QUADRATURE_METHOD",
    "QuadratureOptions",
    "QuadratureResult",
    "estimate_convergence_order",
    "integrand_availability",
    "integrate_fiber",
    "pointwise_integrand",
]
