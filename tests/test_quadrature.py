"""Line quadrature of the partial physical integrand: analytic and canonical evidence."""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lumice_integral.analytic import BODY_AXIS, direction_map, tangent_basis
from lumice_integral.canonical_scene import canonical_pixel_problem
from lumice_integral.continuation import (
    ContinuationOptions,
    FiberProblem,
    FiberStatus,
    TargetChart,
    trace_fiber,
)
from lumice_integral.quadrature import (
    HAAR_TO_DVOL_G_FACTOR,
    INTEGRAND_FACTOR_NAMES,
    QuadratureOptions,
    _chord_speed_kernel,
    _fiber_nodes,
    _PassAccounting,
    _simpson_panel,
    _simpson_panel_estimate,
    estimate_convergence_order,
    integrand_availability,
    integrate_fiber,
    pointwise_integrand,
)
from lumice_integral.so3 import exp
from lumice_integral.weights import WeightEvaluator

EPSILON = 1e-6


def _constant(value: float) -> WeightEvaluator:
    return WeightEvaluator(lambda _: value, "dimensionless", "test constant")


def _circle_angle(rotation: np.ndarray) -> float:
    """``theta`` of ``R = exp(theta e3)`` on the analytic circle seeded at the identity."""
    return float(np.arctan2(rotation[1, 0], rotation[0, 0]))


def _circle_problem(density) -> FiberProblem:
    """Analytic circle ``F(R) = R e3`` with ``rho_pose = density`` and unit other factors."""
    return FiberProblem(
        path="analytic-body-axis",
        incident_direction=jnp.array([1.0, 0.0, 0.0], dtype=jnp.float64),
        target_chart=TargetChart(BODY_AXIS, tangent_basis(BODY_AXIS)),
        direction_evaluator=direction_map,
        seed=jnp.eye(3, dtype=jnp.float64),
        weight_evaluators={
            "rho_pose": WeightEvaluator(density, "dimensionless", "relative to Haar"),
            **{name: _constant(1.0) for name in INTEGRAND_FACTOR_NAMES},
        },
    )


@pytest.fixture(scope="module")
def canonical():
    problem = canonical_pixel_problem()
    result = trace_fiber(problem)
    return problem, result, integrate_fiber(problem, result)


# --- Step 1: options and pointwise integrand -------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"epsilon": 0.0},
        {"epsilon": float("inf")},
        {"relative_tolerance": -1e-8},
        {"maximum_refinement_depth": 0},
    ],
)
def test_quadrature_options_reject_nonpositive_policy(kwargs):
    with pytest.raises(ValueError):
        QuadratureOptions(**kwargs)


def test_pointwise_integrand_combines_named_factors_over_regularised_j_perp(canonical):
    problem, result, _ = canonical
    values = pointwise_integrand(result, epsilon=EPSILON)

    assert values.shape == (len(result.poses),)
    assert values.dtype == np.float64
    assert np.all(np.isfinite(values)) and np.all(values > 0.0)
    observables = result.weight_observables
    for index in (0, len(values) // 3, len(values) - 1):
        expected = observables["rho_pose"].values[index]
        for name in INTEGRAND_FACTOR_NAMES:
            expected *= observables[name].values[index]
        expected /= result.jacobian_diagnostics[index].normal_jacobian + EPSILON
        assert values[index] == pytest.approx(expected, rel=0.0, abs=1e-12 * expected)
    # The gate factor is one everywhere on the canonical fiber; the raw
    # J_perp array in the result is untouched by epsilon.
    np.testing.assert_array_equal(observables["path_validity"].values, 1.0)
    assert min(d.normal_jacobian for d in result.jacobian_diagnostics) > 0.08


def test_missing_density_factor_is_unavailable_not_silently_one(canonical):
    problem, result, _ = canonical
    without_density = replace(
        problem,
        weight_evaluators={
            name: evaluator
            for name, evaluator in problem.weight_evaluators.items()
            if name != "rho_pose"
        },
    )
    assert integrand_availability(without_density) == "unavailable_missing_rho_pose"
    with pytest.raises(ValueError, match="unavailable_missing_rho_pose"):
        pointwise_integrand(
            trace_fiber(without_density, ContinuationOptions(maximum_accepted_steps=2)),
            epsilon=EPSILON,
        )

    quadrature = integrate_fiber(without_density, result)
    assert quadrature.status == "unavailable_missing_rho_pose"
    assert np.isnan(quadrature.value) and quadrature.node_count == 0

    geometry_only = canonical_pixel_problem(with_weights=False)
    assert integrand_availability(geometry_only) == "unavailable_missing_rho_pose"


# --- Step 2: panel primitive and adaptive recursion -------------------------


def test_chord_speed_matches_the_ad_velocity_of_a_synthetic_retracted_curve():
    rng = np.random.default_rng(1)
    base = jnp.asarray(exp(jnp.asarray(rng.normal(size=3))))
    chord = jnp.array([0.03, -0.02, 0.01], dtype=jnp.float64)
    chord_hat = chord / jnp.linalg.norm(chord)
    first = jnp.cross(chord_hat, jnp.array([0.0, 0.0, 1.0])) * 0.3
    second = jnp.cross(chord_hat, first) * 0.2

    def delta(u):
        return u * (1.0 - u) * first + u * u * (1.0 - u) * second

    def gamma(u):
        return base @ exp(u * chord) @ exp(delta(u))

    for u in (0.0, 0.25, 0.5, 0.9, 1.0):
        u = jnp.asarray(u, dtype=jnp.float64)
        body_velocity = gamma(u).T @ jax.jacfwd(gamma)(u)
        omega = jnp.array(
            [body_velocity[2, 1], body_velocity[0, 2], body_velocity[1, 0]]
        )
        speed = float(jnp.linalg.norm(omega))
        estimated = float(_chord_speed_kernel(omega / speed, chord, delta(u)))
        assert estimated == pytest.approx(speed, rel=0.0, abs=1e-14)


def test_simpson_panel_is_exact_for_a_quadratic_integrand_on_the_circle():
    coefficients = (0.7, 0.3, 0.1)

    def quadratic(rotation):
        theta = _circle_angle(np.asarray(rotation))
        return coefficients[0] + coefficients[1] * theta + coefficients[2] * theta**2

    problem = _circle_problem(quadratic)
    options = ContinuationOptions()
    result = trace_fiber(problem, options)
    nodes = _fiber_nodes(result, epsilon=EPSILON)
    # A panel away from the theta = +-pi wrap of the test weight.
    left, right = nodes[2], nodes[3]
    theta_left, theta_right = _circle_angle(left.rotation), _circle_angle(right.rotation)
    antiderivative = lambda t: (
        coefficients[0] * t + coefficients[1] * t**2 / 2.0 + coefficients[2] * t**3 / 3.0
    )
    exact = (antiderivative(theta_right) - antiderivative(theta_left)) / (1.0 + EPSILON)

    estimate = _simpson_panel_estimate(problem, options, left, right, epsilon=EPSILON)
    assert estimate.midpoint is not None and estimate.failure is None
    assert estimate.chord == pytest.approx(theta_right - theta_left, abs=1e-14)
    assert estimate.value == pytest.approx(exact, rel=0.0, abs=1e-14)

    accounting = _PassAccounting()
    value, error = _simpson_panel(
        problem,
        options,
        QuadratureOptions(),
        left,
        right,
        estimate,
        tolerance=1e-12,
        depth=0,
        edge_index=2,
        accounting=accounting,
    )
    assert value == pytest.approx(exact, rel=0.0, abs=1e-14)
    assert error <= 1e-15
    assert accounting.refinements == 0 and accounting.midpoints == 2


def test_depth_exhaustion_is_reported_not_silently_accepted():
    problem = _circle_problem(lambda rotation: 1.0 + 0.5 * float(rotation[0, 0]))
    result = trace_fiber(problem)

    quadrature = integrate_fiber(
        problem,
        result,
        quadrature_options=QuadratureOptions(
            relative_tolerance=1e-15, maximum_refinement_depth=1
        ),
    )

    assert quadrature.status == "available"
    assert quadrature.maximum_depth_reached == 1
    assert len(quadrature.depth_exhausted_edges) > 0
    assert quadrature.refinement_failures == ()


# --- Step 3: convergence order -----------------------------------------------


def test_convergence_order_is_near_four_for_a_smooth_non_quadratic_weight():
    problem = _circle_problem(lambda rotation: 1.0 + 0.5 * float(rotation[0, 0]))
    result = trace_fiber(problem)

    order = estimate_convergence_order(problem, ContinuationOptions(), result, epsilon=EPSILON)

    assert order.order is not None and abs(order.order - 4.0) < 0.5
    assert order.median_edge_order is not None and abs(order.median_edge_order - 4.0) < 0.5
    assert order.low_order_edges == ()
    assert len(order.levels) == 3


def test_convergence_order_is_none_at_the_floating_point_floor():
    problem = _circle_problem(lambda _: 1.0)
    result = trace_fiber(problem)

    order = estimate_convergence_order(problem, ContinuationOptions(), result, epsilon=EPSILON)

    assert order.order is None
    assert "not estimable" in order.note
    assert order.levels[0] == pytest.approx(order.levels[2], abs=1e-13)


# --- Step 4: analytic circle values and reversed seed orientation ------------


def test_constant_weight_recovers_the_haar_identity_within_epsilon_and_error():
    problem = _circle_problem(lambda _: 1.0)
    result = trace_fiber(problem)

    quadrature = integrate_fiber(problem, result)

    assert quadrature.status == "available"
    assert quadrature.fiber_status == "closed"
    expected_raw = 2.0 * np.pi / (1.0 + EPSILON)  # J_perp == 1 on the circle
    assert abs(quadrature.raw_value - expected_raw) <= quadrature.raw_error_estimate + 1e-13
    assert quadrature.haar_to_dvol_g_factor == HAAR_TO_DVOL_G_FACTOR == 1.0 / (8.0 * np.pi**2)
    assert quadrature.value == quadrature.raw_value * HAAR_TO_DVOL_G_FACTOR
    assert abs(quadrature.value - 1.0 / (4.0 * np.pi)) <= (
        EPSILON / (4.0 * np.pi) + quadrature.error_estimate
    )
    assert quadrature.refinements == 0
    assert quadrature.component_completeness == "unknown"
    assert quadrature.coverage.startswith("partial")


def test_trigonometric_weight_matches_the_analytic_integral_within_error():
    problem = _circle_problem(lambda rotation: 1.0 + 0.5 * float(rotation[0, 0]))
    result = trace_fiber(problem)

    quadrature = integrate_fiber(problem, result)

    expected_raw = 2.0 * np.pi / (1.0 + EPSILON)  # int_0^{2pi} (1 + cos(theta)/2) dtheta
    assert abs(quadrature.raw_value - expected_raw) <= quadrature.raw_error_estimate
    assert quadrature.raw_error_estimate < 1e-7 * expected_raw
    assert quadrature.refinement_failures == ()
    assert quadrature.depth_exhausted_edges == ()


def test_reversed_seed_orientation_keeps_arclength_and_integral():
    problem = _circle_problem(lambda rotation: 1.0 + 0.5 * float(rotation[0, 0]))
    forward_options = ContinuationOptions()
    reverse_options = ContinuationOptions(initial_tangent_sign=-1)
    forward = trace_fiber(problem, forward_options)
    reverse = trace_fiber(problem, reverse_options)

    assert float(np.dot(forward.tangents[0], reverse.tangents[0])) < 0.0
    assert reverse.status == forward.status == FiberStatus.CLOSED
    assert reverse.arclength_increments.sum() == pytest.approx(
        forward.arclength_increments.sum(), abs=1e-12
    )
    forward_quadrature = integrate_fiber(problem, forward, forward_options)
    reverse_quadrature = integrate_fiber(problem, reverse, reverse_options)
    assert abs(forward_quadrature.value - reverse_quadrature.value) <= (
        forward_quadrature.error_estimate + reverse_quadrature.error_estimate
    )


# --- Step 5: canonical pixel fiber ---------------------------------------------


def test_canonical_pixel_quadrature_converges_without_silent_degradation(canonical):
    _, result, quadrature = canonical

    assert result.status == FiberStatus.CLOSED
    assert quadrature.status == "available"
    assert quadrature.fiber_status == "closed"
    assert quadrature.value > 0.0
    assert np.isfinite(quadrature.error_estimate)
    assert quadrature.error_estimate < 1e-8 * quadrature.value
    assert quadrature.value == quadrature.raw_value * HAAR_TO_DVOL_G_FACTOR
    assert quadrature.epsilon == 1e-6
    assert quadrature.factor_names == INTEGRAND_FACTOR_NAMES
    # No silent fallback: every refinement node was retracted and no edge hit
    # the depth limit (the entry_measure slope jumps need depth 16 at 1e-8).
    assert quadrature.refinement_failures == ()
    assert quadrature.depth_exhausted_edges == ()
    assert 12 <= quadrature.maximum_depth_reached <= 20
    assert quadrature.node_count > len(result.poses)
    assert quadrature.refinements > 0
    # Piecewise-smooth integrand: smooth edges show Simpson's order 4, the
    # slope jumps of entry_measure pull the global uniform order down.
    assert quadrature.median_edge_convergence_order is not None
    assert abs(quadrature.median_edge_convergence_order - 4.0) < 0.5
    assert 1 <= len(quadrature.low_order_edges) <= 12
    assert quadrature.convergence_order_estimate is not None
    assert quadrature.convergence_order_estimate < 3.0


@pytest.mark.parametrize("initial_step", [0.03, 0.08])
def test_canonical_pixel_integral_is_invariant_under_initial_step(canonical, initial_step):
    problem, _, reference = canonical
    options = ContinuationOptions(initial_step=initial_step)
    result = trace_fiber(problem, options)
    assert result.status == FiberStatus.CLOSED

    quadrature = integrate_fiber(problem, result, options)

    # Invariants only: sample counts are a controller observation, not a criterion.
    assert quadrature.refinement_failures == ()
    assert quadrature.depth_exhausted_edges == ()
    assert abs(quadrature.value - reference.value) <= (
        quadrature.error_estimate + reference.error_estimate
    )


@pytest.mark.parametrize("relative_tolerance", [1e-6, 1e-9])
def test_canonical_pixel_integral_is_invariant_under_refinement_tolerance(
    canonical, relative_tolerance
):
    problem, result, reference = canonical

    quadrature = integrate_fiber(
        problem,
        result,
        quadrature_options=QuadratureOptions(relative_tolerance=relative_tolerance),
    )

    assert quadrature.refinement_failures == ()
    assert quadrature.depth_exhausted_edges == ()
    assert abs(quadrature.value - reference.value) <= (
        quadrature.error_estimate + reference.error_estimate
    )
    if relative_tolerance < reference.relative_tolerance:
        assert quadrature.error_estimate < reference.error_estimate


def test_canonical_pixel_integral_is_invariant_under_reversed_orientation(canonical):
    problem, forward, reference = canonical
    options = ContinuationOptions(initial_tangent_sign=-1)
    reverse = trace_fiber(problem, options)

    assert reverse.status == FiberStatus.CLOSED
    assert float(np.dot(forward.tangents[0], reverse.tangents[0])) < 0.0
    quadrature = integrate_fiber(problem, reverse, options)
    assert quadrature.refinement_failures == ()
    assert abs(quadrature.value - reference.value) <= (
        quadrature.error_estimate + reference.error_estimate
    )
