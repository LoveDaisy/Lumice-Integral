from __future__ import annotations

from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest

import lumice_integral.continuation as continuation
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
    _correct_closure,
    _evaluate_regular_state,
    _adapt_accepted_step,
    trace_fiber,
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

    result = trace_fiber(problem)
    assert result.status == FiberStatus.EVENT_TERMINATED
    assert result.reason == TerminationReason.RANK_LOSS
    assert result.terminal_payload.event.details["sigma_2"] == 0.0


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


def test_bordered_corrector_requires_a_small_final_newton_update():
    problem = analytic_problem()
    options = ContinuationOptions(
        corrector_maximum_iterations=3,
        corrector_update_tolerance=1e-16,
    )
    initial = _evaluate_regular_state(problem, options, problem.seed)
    predicted = problem.seed @ exp(
        jnp.array([0.03, -0.02, 0.04], dtype=jnp.float64)
    )

    outcome = _correct_trial(
        problem, options, problem.seed, predicted, initial.tangent
    )

    assert not outcome.accepted
    assert outcome.reason == TerminationReason.CORRECTOR_FAILURE
    assert outcome.residual_norm <= options.residual_tolerance
    assert outcome.update_norm > options.corrector_update_tolerance


@pytest.mark.parametrize(
    ("name", "direction", "reason"),
    [
        (
            "nonunit",
            lambda rotation: 2.0 * direction_map(rotation),
            TerminationReason.INVALID_NUMERICAL_INPUT,
        ),
        (
            "outside_chart",
            lambda rotation: -direction_map(rotation),
            TerminationReason.CHART_BOUNDARY,
        ),
    ],
)
def test_trial_corrector_preflights_smooth_output_before_ad(name, direction, reason):
    calls = 0

    def counted_direction(rotation):
        nonlocal calls
        calls += 1
        return direction(rotation)

    base = analytic_problem()
    problem = FiberProblem(
        path=f"trial-{name}",
        incident_direction=base.incident_direction,
        target_chart=base.target_chart,
        direction_evaluator=counted_direction,
        seed=base.seed,
    )
    initial = _evaluate_regular_state(base, ContinuationOptions(), base.seed)

    outcome = _correct_trial(
        problem,
        ContinuationOptions(),
        base.seed,
        base.seed,
        initial.tangent,
    )

    assert not outcome.accepted
    assert outcome.reason == reason
    assert calls == 1
    if reason == TerminationReason.CHART_BOUNDARY:
        assert outcome.event is not None
        assert outcome.event.kind == reason


def test_residual_headroom_participates_in_step_adaptation():
    problem = analytic_problem()
    options = ContinuationOptions(residual_tolerance=1e-8)
    initial = _evaluate_regular_state(problem, options, problem.seed)
    accepted = _correct_trial(
        problem,
        options,
        problem.seed,
        problem.seed @ exp(0.01 * initial.tangent),
        initial.tangent,
    )
    assert accepted.accepted

    near_residual_limit = replace(
        accepted,
        iterations=1,
        correction_norm=0.001,
        tangent_dot=0.99,
        residual_norm=0.75 * options.residual_tolerance,
    )
    assert _adapt_accepted_step(0.04, near_residual_limit, options) < 0.04


def test_adaptive_analytic_trace_closes_with_quadrature_ready_geometry():
    result = trace_fiber(analytic_problem())

    assert result.status == FiberStatus.CLOSED
    assert result.reason == TerminationReason.CLOSED_LOOP
    assert result.component_completeness == "unknown"
    assert result.poses.shape[1:] == (3, 3)
    assert result.arclength_increments.shape == (result.poses.shape[0] - 1,)
    assert result.residual_norms.shape == (result.poses.shape[0],)
    assert result.tangents.shape == (result.poses.shape[0], 3)
    assert np.all(result.arclength_increments >= 0.0)
    assert result.residual_norms.max() <= 1e-10
    assert result.closure_diagnostics.final_correction_accepted
    np.testing.assert_allclose(
        result.closure_diagnostics.accumulated_arclength,
        2.0 * np.pi,
        rtol=0.0,
        atol=1e-8,
    )


def test_large_initial_step_is_rejected_then_recovers_by_shrinking():
    options = ContinuationOptions(
        initial_step=0.6,
        maximum_step=0.6,
        maximum_advance=0.2,
    )
    result = trace_fiber(analytic_problem(), options)

    assert result.status == FiberStatus.CLOSED
    rejected = [trial for trial in result.step_diagnostics if not trial.accepted]
    assert rejected
    assert rejected[0].proposed_step == pytest.approx(0.6)
    assert any(trial.accepted for trial in result.step_diagnostics)


@pytest.mark.parametrize(
    ("options", "reason"),
    [
        (ContinuationOptions(maximum_accepted_steps=1), TerminationReason.STEP_BUDGET),
        (ContinuationOptions(maximum_arclength=0.01), TerminationReason.ARCLENGTH_BUDGET),
        (ContinuationOptions(maximum_evaluations=1), TerminationReason.EVALUATION_BUDGET),
    ],
)
def test_trace_distinguishes_budget_exhaustion(options, reason):
    result = trace_fiber(analytic_problem(), options)

    assert result.status == FiberStatus.BUDGET_EXHAUSTED
    assert result.reason == reason
    assert result.terminal_payload.last_accepted_pose is not None


def test_closure_evaluation_budget_is_never_exceeded():
    options = ContinuationOptions(maximum_evaluations=111)
    result = trace_fiber(analytic_problem(), options)

    assert result.reason == TerminationReason.EVALUATION_BUDGET
    assert result.closure_diagnostics.final_correction_attempted
    assert result.terminal_payload.evaluations == options.maximum_evaluations


def test_closure_preserves_a_regular_state_event_reason_and_payload():
    target = BODY_AXIS
    problem = FiberProblem(
        path="closure-rank-loss",
        incident_direction=jnp.array([1.0, 0.0, 0.0], dtype=jnp.float64),
        target_chart=TargetChart(target, tangent_basis(target)),
        direction_evaluator=lambda _: target,
        seed=jnp.eye(3, dtype=jnp.float64),
    )
    options = ContinuationOptions()

    outcome = _correct_closure(
        problem,
        options,
        problem.seed,
        problem.seed,
        jnp.array([0.0, 0.0, 1.0], dtype=jnp.float64),
        jnp.array([0.0, 0.0, 1.0], dtype=jnp.float64),
    )

    assert not outcome.accepted
    assert outcome.reason == TerminationReason.RANK_LOSS
    assert outcome.event is not None
    assert outcome.event.kind == TerminationReason.RANK_LOSS


def test_closure_requires_a_small_final_newton_update():
    problem = analytic_problem()
    options = ContinuationOptions(
        closure_maximum_iterations=1,
        corrector_update_tolerance=1e-16,
    )
    initial = _evaluate_regular_state(problem, options, problem.seed)
    current = problem.seed @ exp(0.01 * initial.tangent)

    outcome = _correct_closure(
        problem,
        options,
        current,
        problem.seed,
        initial.tangent,
        initial.tangent,
    )

    assert not outcome.accepted
    assert outcome.reason == TerminationReason.CORRECTOR_FAILURE
    assert outcome.residual_norm <= options.residual_tolerance
    assert outcome.update_norm > options.corrector_update_tolerance


@pytest.mark.parametrize(
    ("name", "direction", "reason"),
    [
        (
            "nonunit",
            lambda rotation: 2.0 * direction_map(rotation),
            TerminationReason.INVALID_NUMERICAL_INPUT,
        ),
        (
            "outside_chart",
            lambda rotation: -direction_map(rotation),
            TerminationReason.CHART_BOUNDARY,
        ),
    ],
)
def test_closure_corrector_preflights_smooth_output_before_ad(
    name, direction, reason
):
    calls = 0

    def counted_direction(rotation):
        nonlocal calls
        calls += 1
        return direction(rotation)

    base = analytic_problem()
    problem = FiberProblem(
        path=f"closure-{name}",
        incident_direction=base.incident_direction,
        target_chart=base.target_chart,
        direction_evaluator=counted_direction,
        seed=base.seed,
    )
    initial = _evaluate_regular_state(base, ContinuationOptions(), base.seed)

    outcome = _correct_closure(
        problem,
        ContinuationOptions(),
        base.seed,
        base.seed,
        initial.tangent,
        initial.tangent,
    )

    assert not outcome.accepted
    assert outcome.reason == reason
    assert calls == 1
    if reason == TerminationReason.CHART_BOUNDARY:
        assert outcome.event is not None
        assert outcome.event.kind == reason


def test_closure_failure_history_survives_a_later_success(monkeypatch):
    original_correct_closure = continuation._correct_closure
    calls = 0

    def reject_first_closure(*args, **kwargs):
        nonlocal calls
        outcome = original_correct_closure(*args, **kwargs)
        if calls == 0:
            calls += 1
            assert outcome.accepted
            assert outcome.state is not None
            return replace(
                outcome,
                accepted=False,
                rejected_pose=np.asarray(outcome.state.rotation),
                reason=TerminationReason.CORRECTOR_FAILURE,
                message="forced recoverable closure failure",
            )
        calls += 1
        return outcome

    monkeypatch.setattr(continuation, "_correct_closure", reject_first_closure)

    result = trace_fiber(analytic_problem())

    assert result.status == FiberStatus.CLOSED
    assert len(result.closure_attempt_diagnostics) >= 2
    first_attempt = result.closure_attempt_diagnostics[0]
    assert first_attempt.reason == TerminationReason.CORRECTOR_FAILURE
    assert first_attempt.rejected_pose is not None
    assert np.isfinite(first_attempt.residual_norm)
    assert np.isfinite(first_attempt.update_norm)
    assert not first_attempt.gates_passed
    assert result.closure_attempt_diagnostics[-1].accepted
    assert result.closure_diagnostics.attempts == result.closure_attempt_diagnostics


def test_trace_reports_step_underflow_separately_from_trigger():
    options = ContinuationOptions(
        initial_step=0.04,
        minimum_step=0.03,
        maximum_step=0.04,
        maximum_advance=0.01,
    )
    result = trace_fiber(analytic_problem(), options)

    assert result.status == FiberStatus.NUMERICAL_FAILURE
    assert result.reason == TerminationReason.STEP_UNDERFLOW
    assert result.step_diagnostics[-1].reason == "corrector_failure"


def test_trace_reports_corrector_failure_after_bounded_recovery():
    options = ContinuationOptions(
        maximum_advance=0.01,
        maximum_retries=0,
    )
    result = trace_fiber(analytic_problem(), options)

    assert result.status == FiberStatus.NUMERICAL_FAILURE
    assert result.reason == TerminationReason.CORRECTOR_FAILURE
    assert len(result.step_diagnostics) == 1


def test_antipode_algebraic_root_is_rejected_by_chart_gate():
    target = BODY_AXIS
    problem = FiberProblem(
        path="antipode-map",
        incident_direction=jnp.array([1.0, 0.0, 0.0], dtype=jnp.float64),
        target_chart=TargetChart(target, tangent_basis(target)),
        direction_evaluator=lambda _: -target,
        seed=jnp.eye(3, dtype=jnp.float64),
    )

    result = trace_fiber(problem)

    assert result.status == FiberStatus.EVENT_TERMINATED
    assert result.reason == TerminationReason.CHART_BOUNDARY
    assert result.terminal_payload.event.margin < 0.0


def test_incompatible_tangent_cannot_pass_final_closure_correction():
    problem = analytic_problem()
    options = ContinuationOptions()
    initial = _evaluate_regular_state(problem, options, problem.seed)

    outcome = _correct_closure(
        problem,
        options,
        problem.seed,
        problem.seed,
        initial.tangent,
        -initial.tangent,
    )

    assert not outcome.accepted
    assert outcome.reason == TerminationReason.TOPOLOGY_AMBIGUITY
    assert outcome.tangent_dot < options.closure_tangent_dot


def test_known_event_precedes_unsafe_direction_evaluation():
    calls = {"direction": 0}

    def domain(rotation):
        margin = -float(rotation[1, 0])
        if margin < 0.0:
            return DomainEvaluation(
                valid=False,
                margins={"tir": margin},
                event=EventCandidate(TerminationReason.TIR_BOUNDARY, margin),
            )
        return DomainEvaluation(valid=True, margins={"tir": margin})

    def direction(rotation):
        calls["direction"] += 1
        return direction_map(rotation)

    base = analytic_problem(domain_evaluator=domain)
    problem = FiberProblem(
        path=base.path,
        incident_direction=base.incident_direction,
        target_chart=base.target_chart,
        direction_evaluator=direction,
        domain_and_event_evaluator=domain,
        seed=base.seed,
    )
    result = trace_fiber(problem)

    assert result.status == FiberStatus.EVENT_TERMINATED
    assert result.reason == TerminationReason.TIR_BOUNDARY
    assert calls["direction"] == 2


def test_nonfinite_trial_remains_nonfinite_after_retry_exhaustion():
    def direction(rotation):
        unsafe = rotation[1, 0] > 0.0
        return direction_map(rotation) + jnp.where(unsafe, jnp.nan, 0.0)

    base = analytic_problem()
    problem = FiberProblem(
        path=base.path,
        incident_direction=base.incident_direction,
        target_chart=base.target_chart,
        direction_evaluator=direction,
        seed=base.seed,
    )
    result = trace_fiber(problem, ContinuationOptions(maximum_retries=2))

    assert result.status == FiberStatus.NUMERICAL_FAILURE
    assert result.reason == TerminationReason.NON_FINITE
    assert len(result.step_diagnostics) == 3
