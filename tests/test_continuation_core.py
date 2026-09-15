from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from lumice_integral.analytic import BODY_AXIS, direction_map, tangent_basis
from lumice_integral.continuation import (
    ContinuationOptions,
    DomainEvaluation,
    EventCandidate,
    FiberProblem,
    FiberStatus,
    TargetChart,
    TerminationReason,
    _correct_trial,
    _evaluate_regular_state,
)
from lumice_integral.so3 import exp


def analytic_problem(
    *,
    seed: jnp.ndarray | None = None,
    domain_evaluator=None,
) -> FiberProblem:
    target = BODY_AXIS
    return FiberProblem(
        path="analytic-body-axis",
        incident_direction=jnp.array([1.0, 0.0, 0.0], dtype=jnp.float64),
        target_chart=TargetChart(target, tangent_basis(target)),
        direction_evaluator=direction_map,
        domain_and_event_evaluator=domain_evaluator,
        seed=jnp.eye(3, dtype=jnp.float64) if seed is None else seed,
    )


def test_reference_schema_declares_scope_conventions_and_reason_alphabet():
    problem = analytic_problem(
        domain_evaluator=lambda _: DomainEvaluation(
            valid=False,
            margins={"tir": -0.1},
            event=EventCandidate(TerminationReason.TIR_BOUNDARY, -0.1),
        )
    )
    options = ContinuationOptions()

    assert problem.path == "analytic-body-axis"
    assert problem.pose_metric_and_measure == "right-invariant-so3/dvol_g"
    assert problem.domain_and_event_evaluator(problem.seed).event.kind == "tir_boundary"
    assert options.dtype == "float64"
    assert set(FiberStatus) == {
        FiberStatus.CLOSED,
        FiberStatus.EVENT_TERMINATED,
        FiberStatus.NUMERICAL_FAILURE,
        FiberStatus.BUDGET_EXHAUSTED,
    }


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("seed", jnp.eye(3, dtype=jnp.float32), "float64"),
        (
            "incident_direction",
            jnp.array([2.0, 0.0, 0.0], dtype=jnp.float64),
            "unit length",
        ),
    ],
)
def test_problem_rejects_invalid_reference_inputs(field, value, message):
    arguments = {
        "path": "analytic-body-axis",
        "incident_direction": jnp.array([1.0, 0.0, 0.0], dtype=jnp.float64),
        "target_chart": TargetChart(BODY_AXIS, tangent_basis(BODY_AXIS)),
        "direction_evaluator": direction_map,
        "seed": jnp.eye(3, dtype=jnp.float64),
    }
    arguments[field] = value

    with pytest.raises(ValueError, match=message):
        FiberProblem(**arguments)


def test_target_chart_rejects_nonunit_target_and_antipode_neighborhood_limit():
    basis = tangent_basis(BODY_AXIS)
    with pytest.raises(ValueError, match="unit length"):
        TargetChart(2.0 * BODY_AXIS, basis)
    with pytest.raises(ValueError, match="minimum_dot"):
        TargetChart(BODY_AXIS, basis, minimum_dot=-1.0)


def test_reference_options_reject_float32_and_invalid_step_bounds():
    with pytest.raises(ValueError, match="float64"):
        ContinuationOptions(dtype="float32")
    with pytest.raises(ValueError, match="step sizes"):
        ContinuationOptions(minimum_step=0.1, initial_step=0.05)


def test_schema_inputs_are_numpy_float64_compatible():
    problem = analytic_problem(seed=jnp.asarray(np.eye(3), dtype=jnp.float64))
    assert np.asarray(problem.seed).dtype == np.float64
    assert np.asarray(problem.target_chart.direction).dtype == np.float64


def test_regular_state_reports_analytic_singular_values_and_tangent():
    problem = analytic_problem()
    state = _evaluate_regular_state(problem, ContinuationOptions(), problem.seed)

    assert state.accepted
    assert state.jacobian.shape == (2, 3)
    np.testing.assert_allclose(
        state.jacobian_diagnostic.singular_values, (1.0, 1.0), atol=1e-14
    )
    assert state.jacobian_diagnostic.normal_jacobian == pytest.approx(1.0)
    assert state.jacobian_diagnostic.rank == 2
    np.testing.assert_allclose(np.linalg.norm(state.tangent), 1.0, atol=1e-14)


def test_tangent_orientation_is_continuous():
    problem = analytic_problem()
    options = ContinuationOptions()
    first = _evaluate_regular_state(problem, options, problem.seed)
    second_pose = problem.seed @ exp(0.1 * first.tangent)
    second = _evaluate_regular_state(
        problem, options, second_pose, previous_tangent=first.tangent
    )

    assert second.accepted
    assert float(jnp.dot(first.tangent, second.tangent)) >= 0.0


def test_rank_deficient_direction_map_has_typed_event_and_diagnostics():
    target = BODY_AXIS
    problem = FiberProblem(
        path="constant-map",
        incident_direction=jnp.array([1.0, 0.0, 0.0], dtype=jnp.float64),
        target_chart=TargetChart(target, tangent_basis(target)),
        direction_evaluator=lambda _: target,
        seed=jnp.eye(3, dtype=jnp.float64),
    )
    state = _evaluate_regular_state(problem, ContinuationOptions(), problem.seed)

    assert not state.accepted
    assert state.reason == TerminationReason.RANK_LOSS
    assert state.event.kind == TerminationReason.RANK_LOSS
    assert state.jacobian_diagnostic.rank == 0
    assert state.jacobian_diagnostic.normal_jacobian == 0.0


def test_bordered_corrector_converges_and_preserves_phase():
    problem = analytic_problem()
    options = ContinuationOptions()
    initial = _evaluate_regular_state(problem, options, problem.seed)
    predicted = problem.seed @ exp(
        jnp.array([0.03, -0.02, 0.04], dtype=jnp.float64)
    )

    outcome = _correct_trial(
        problem, options, problem.seed, predicted, initial.tangent
    )

    assert outcome.accepted
    assert outcome.residual_norm <= options.residual_tolerance
    assert outcome.correction_norm <= options.maximum_correction
    assert outcome.tangent_dot >= options.minimum_tangent_dot


def test_bordered_corrector_iteration_budget_is_typed_failure():
    problem = analytic_problem()
    options = ContinuationOptions(
        residual_tolerance=1e-16,
        corrector_maximum_iterations=1,
    )
    initial = _evaluate_regular_state(problem, options, problem.seed)
    predicted = problem.seed @ exp(
        jnp.array([0.1, -0.08, 0.03], dtype=jnp.float64)
    )

    outcome = _correct_trial(
        problem, options, problem.seed, predicted, initial.tangent
    )

    assert not outcome.accepted
    assert outcome.reason == TerminationReason.CORRECTOR_FAILURE
    assert outcome.iterations == 1
    assert outcome.residual_norm > options.residual_tolerance
