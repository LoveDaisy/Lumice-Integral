from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from lumice_integral.analytic import tangent_basis
from lumice_integral.continuation import FiberStatus, TerminationReason, trace_fiber, trace_implicit_fiber
from lumice_integral.optics import (
    minimum_deviation_incident,
    path_3_5,
    path_3_5_domain,
    path_3_5_problem,
)
from lumice_integral.so3 import exp
from test_analytic_fiber import central_difference_jacobian


def test_symmetric_3_5_path_is_regular_and_unit_length():
    evaluation = path_3_5(jnp.eye(3, dtype=jnp.float64), minimum_deviation_incident())

    np.testing.assert_allclose(np.linalg.norm(evaluation.direction), 1.0, atol=1e-14)
    assert evaluation.entry.discriminant > 0.0
    assert evaluation.exit.discriminant > 0.0
    assert evaluation.entry.incidence_cosine > 0.0
    assert evaluation.exit.incidence_cosine > 0.0


def test_3_5_direction_jacobian_matches_central_difference():
    rotation = exp(jnp.array([0.15, 0.08, -0.05], dtype=jnp.float64))
    incident = minimum_deviation_incident()
    target = path_3_5(rotation, incident).direction
    basis = tangent_basis(target)

    def local_residual(delta):
        direction = path_3_5(rotation @ exp(delta), incident).direction
        return basis.T @ (direction - target)

    point = np.zeros(3)
    actual = np.asarray(jax.jacfwd(local_residual)(jnp.asarray(point)))
    expected = central_difference_jacobian(
        lambda delta: np.asarray(local_residual(jnp.asarray(delta))), point
    )

    relative_error = np.linalg.norm(actual - expected) / np.linalg.norm(expected)
    assert relative_error <= 1e-6
    assert np.linalg.matrix_rank(actual) == 2
    assert np.linalg.svd(actual, compute_uv=False)[-1] >= 0.05


def test_3_5_regular_fiber_closes():
    initial = exp(jnp.array([0.15, 0.08, -0.05], dtype=jnp.float64))
    incident = minimum_deviation_incident()
    target = path_3_5(initial, incident).direction
    basis = tangent_basis(target)

    def pose_residual(rotation):
        direction = path_3_5(rotation, incident).direction
        return basis.T @ (direction - target)

    result = trace_implicit_fiber(initial, pose_residual)

    assert result.residual_norms.max() <= 1e-10
    assert result.closure_error <= 1e-8
    assert result.steps == 145


def test_safe_3_5_adapter_traces_a_regular_component():
    initial = exp(jnp.array([0.15, 0.08, -0.05], dtype=jnp.float64))
    problem = path_3_5_problem(initial, minimum_deviation_incident())

    result = trace_fiber(problem)

    assert result.status == FiberStatus.CLOSED
    assert result.reason == TerminationReason.CLOSED_LOOP
    assert result.residual_norms.max() <= 1e-10
    assert all(
        margins["entry_snell_discriminant"] > 0.0
        and margins["exit_snell_discriminant"] > 0.0
        for margins in result.branch_diagnostics.accepted_margins
    )


def test_3_5_tir_is_reported_before_the_unsafe_exit_square_root():
    initial = exp(
        jnp.array(
            [0.5447316801391622, -1.506228738763967, -1.190186580432801],
            dtype=jnp.float64,
        )
    )
    incident = minimum_deviation_incident()
    domain = path_3_5_domain(initial, incident)
    problem = path_3_5_problem(
        initial,
        incident,
        target_direction=jnp.array([0.0, 0.0, 1.0], dtype=jnp.float64),
    )

    result = trace_fiber(problem)

    assert not domain.valid
    assert domain.event_kind == "tir_boundary"
    assert domain.margins["entry_snell_discriminant"] > 0.0
    assert domain.margins["exit_snell_discriminant"] < 0.0
    assert result.status == FiberStatus.EVENT_TERMINATED
    assert result.reason == TerminationReason.TIR_BOUNDARY
    assert result.terminal_payload.event.margin < 0.0
