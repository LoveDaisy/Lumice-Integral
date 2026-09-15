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
)


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
