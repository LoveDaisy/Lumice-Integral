from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from lumice_integral.analytic import BODY_AXIS, direction_map, tangent_basis
from lumice_integral.continuation import (
    FiberProblem,
    FiberStatus,
    TargetChart,
    TerminationReason,
    local_residual_jacobian,
    target_residual,
    trace_fiber,
)
from lumice_integral.so3 import exp


def central_difference_jacobian(function, point, step=1e-5):
    columns = []
    for axis in np.eye(point.size):
        columns.append((function(point + step * axis) - function(point - step * axis)) / (2 * step))
    return np.stack(columns, axis=1)


def test_ad_jacobian_matches_central_difference():
    rotation = jnp.eye(3, dtype=jnp.float64)
    target = BODY_AXIS
    basis = tangent_basis(target)

    problem = FiberProblem(
        path="analytic-body-axis",
        incident_direction=jnp.array([1.0, 0.0, 0.0], dtype=jnp.float64),
        target_chart=TargetChart(target, basis),
        direction_evaluator=direction_map,
        seed=rotation,
    )
    actual = np.asarray(local_residual_jacobian(problem, rotation))
    expected = central_difference_jacobian(
        lambda delta: np.asarray(
            target_residual(problem, rotation @ exp(jnp.asarray(delta)))
        ),
        np.zeros(3),
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)
    assert np.linalg.matrix_rank(actual) == 2


def test_analytic_fiber_closes_with_small_residual():
    initial = jnp.eye(3, dtype=jnp.float64)
    target = BODY_AXIS
    problem = FiberProblem(
        path="analytic-body-axis",
        incident_direction=jnp.array([1.0, 0.0, 0.0], dtype=jnp.float64),
        target_chart=TargetChart(target, tangent_basis(target)),
        direction_evaluator=direction_map,
        seed=initial,
    )
    result = trace_fiber(problem)

    assert result.status == FiberStatus.CLOSED
    assert result.reason == TerminationReason.CLOSED_LOOP
    assert result.residual_norms.max() <= 1e-10
    assert result.closure_diagnostics.seed_distance <= 1e-8
    np.testing.assert_allclose(
        result.arclength_increments.sum(), 2 * np.pi, rtol=0, atol=1e-8
    )
