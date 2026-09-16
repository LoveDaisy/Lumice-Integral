"""Consumer-facing conformance tests for the Phase I reference core.

These tests organize evidence by the language-independent contract rather than
by private continuation helpers.  Implementation-level tests remain in the
module-specific test files.
"""

from __future__ import annotations

from dataclasses import fields, replace

import jax.numpy as jnp
import numpy as np
import pytest

from lumice_integral.analytic import BODY_AXIS, direction_map, tangent_basis
from lumice_integral.continuation import (
    ContinuationOptions,
    DomainEvaluation,
    EventCandidate,
    FiberProblem,
    FiberResult,
    FiberStatus,
    TargetChart,
    TerminationReason,
    trace_fiber,
)
from lumice_integral.optics import path_3_5
from lumice_integral.so3 import exp


def _analytic_problem(*, direction_evaluator=direction_map) -> FiberProblem:
    return FiberProblem(
        path="analytic-body-axis",
        incident_direction=jnp.array([1.0, 0.0, 0.0], dtype=jnp.float64),
        target_chart=TargetChart(BODY_AXIS, tangent_basis(BODY_AXIS)),
        direction_evaluator=direction_evaluator,
        seed=jnp.eye(3, dtype=jnp.float64),
    )


def _pairwise_rotation_distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    relative = np.einsum("aij,bik->abjk", left, right)
    cosine = np.clip(
        (np.trace(relative, axis1=-2, axis2=-1) - 1.0) / 2.0,
        -1.0,
        1.0,
    )
    skew = np.stack(
        (
            relative[..., 2, 1] - relative[..., 1, 2],
            relative[..., 0, 2] - relative[..., 2, 0],
            relative[..., 1, 0] - relative[..., 0, 1],
        ),
        axis=-1,
    ) / 2.0
    return np.arctan2(np.linalg.norm(skew, axis=-1), cosine)


def _rotation_set_distance(left: np.ndarray, right: np.ndarray) -> float:
    distances = _pairwise_rotation_distances(left, right)
    return float(
        max(
            np.max(np.min(distances, axis=1)),
            np.max(np.min(distances, axis=0)),
        )
    )


def _oracle_incident_3_5(refractive_index: float = 1.31) -> np.ndarray:
    external_angle = np.arcsin(refractive_index * 0.5)
    return np.array([-np.cos(external_angle), np.sin(external_angle), 0.0])


def _oracle_tangent_basis(direction: np.ndarray) -> np.ndarray:
    """Construct a target basis without the production analytic helper."""
    reference = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(direction, reference))) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])
    first = np.cross(reference, direction)
    first /= np.linalg.norm(first)
    second = np.cross(direction, first)
    return np.column_stack((first, second))


def _oracle_path_3_5(
    rotation: np.ndarray,
    incident: np.ndarray,
    refractive_index: float = 1.31,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Direct NumPy ray oracle independent of production optics adapters."""
    entry_normal = rotation @ np.array([1.0, 0.0, 0.0])
    exit_normal = rotation @ np.array([-0.5, np.sqrt(3.0) / 2.0, 0.0])
    entry_index = 1.0 / refractive_index
    entry_cosine = -float(np.dot(entry_normal, incident))
    entry_discriminant = 1.0 - entry_index**2 * (1.0 - entry_cosine**2)
    assert entry_cosine > 0.0
    assert entry_discriminant > 0.0
    internal = entry_index * incident + (
        entry_index * entry_cosine - np.sqrt(entry_discriminant)
    ) * entry_normal
    exit_cosine = float(np.dot(exit_normal, internal))
    exit_discriminant = 1.0 - refractive_index**2 * (1.0 - exit_cosine**2)
    assert exit_cosine > 0.0
    assert exit_discriminant > 0.0
    outgoing = refractive_index * internal - (
        refractive_index * exit_cosine - np.sqrt(exit_discriminant)
    ) * exit_normal
    return internal, outgoing, {
        "entry_incidence_cosine": entry_cosine,
        "entry_snell_discriminant": entry_discriminant,
        "exit_incidence_cosine": exit_cosine,
        "exit_snell_discriminant": exit_discriminant,
    }


def _oracle_domain_3_5(
    rotation: np.ndarray,
    incident: np.ndarray,
    refractive_index: float = 1.31,
) -> DomainEvaluation:
    """Test-side event gate derived independently from the optics adapter."""
    rotation = np.asarray(rotation)
    entry_normal = rotation @ np.array([1.0, 0.0, 0.0])
    exit_normal = rotation @ np.array([-0.5, np.sqrt(3.0) / 2.0, 0.0])
    entry_index = 1.0 / refractive_index
    entry_cosine = -float(np.dot(entry_normal, incident))
    entry_discriminant = 1.0 - entry_index**2 * (1.0 - entry_cosine**2)
    margins = {
        "entry_incidence_cosine": entry_cosine,
        "entry_snell_discriminant": entry_discriminant,
    }
    if entry_cosine <= 0.0:
        event = EventCandidate(
            TerminationReason.PATH_INFEASIBLE,
            entry_cosine,
            "oracle: ray does not enter face 3",
            margins,
        )
        return DomainEvaluation(False, margins, event)
    if entry_discriminant <= 0.0:
        event = EventCandidate(
            TerminationReason.TIR_BOUNDARY,
            entry_discriminant,
            "oracle: entry Snell boundary",
            margins,
        )
        return DomainEvaluation(False, margins, event)

    internal = entry_index * incident + (
        entry_index * entry_cosine - np.sqrt(entry_discriminant)
    ) * entry_normal
    exit_cosine = float(np.dot(exit_normal, internal))
    exit_discriminant = 1.0 - refractive_index**2 * (1.0 - exit_cosine**2)
    margins = {
        **margins,
        "exit_incidence_cosine": exit_cosine,
        "exit_snell_discriminant": exit_discriminant,
    }
    if exit_cosine <= 0.0:
        event = EventCandidate(
            TerminationReason.PATH_INFEASIBLE,
            exit_cosine,
            "oracle: ray does not leave face 5",
            margins,
        )
        return DomainEvaluation(False, margins, event)
    if exit_discriminant <= 0.0:
        event = EventCandidate(
            TerminationReason.TIR_BOUNDARY,
            exit_discriminant,
            "oracle: exit Snell boundary",
            margins,
        )
        return DomainEvaluation(False, margins, event)
    return DomainEvaluation(True, margins)


@pytest.fixture(scope="module")
def analytic_sweep():
    problem = _analytic_problem()
    return {
        step: trace_fiber(problem, ContinuationOptions(initial_step=step))
        for step in (0.04, 0.08, 0.12)
    }


@pytest.fixture(scope="module")
def optical_fixture():
    seed = exp(jnp.array([0.15, 0.08, -0.05], dtype=jnp.float64))
    incident_numpy = _oracle_incident_3_5()
    _, target_numpy, _ = _oracle_path_3_5(np.asarray(seed), incident_numpy)
    basis_numpy = _oracle_tangent_basis(target_numpy)
    incident = jnp.asarray(incident_numpy, dtype=jnp.float64)
    refractive_index = jnp.asarray(1.31, dtype=jnp.float64)

    return FiberProblem(
        path="oracle-constructed-3-5:n=1.31",
        incident_direction=incident,
        target_chart=TargetChart(
            jnp.asarray(target_numpy, dtype=jnp.float64),
            jnp.asarray(basis_numpy, dtype=jnp.float64),
        ),
        direction_evaluator=lambda rotation: path_3_5(
            rotation, incident, refractive_index
        ).direction,
        domain_and_event_evaluator=lambda rotation: _oracle_domain_3_5(
            np.asarray(rotation), incident_numpy
        ),
        seed=seed,
    )


@pytest.fixture(scope="module")
def optical_sweep(optical_fixture):
    return {
        step: trace_fiber(optical_fixture, ContinuationOptions(initial_step=step))
        for step in (0.03, 0.04, 0.08)
    }


@pytest.fixture(scope="module")
def optical_controller_sweep(optical_fixture, optical_sweep):
    return {
        "reference": optical_sweep[0.04],
        "perturbed": trace_fiber(
            optical_fixture,
            ContinuationOptions(
                initial_step=0.04,
                minimum_step=2e-5,
                maximum_step=0.10,
                shrink_factor=0.4,
                growth_factor=1.15,
                maximum_retries=10,
            ),
        ),
    }


def test_public_result_schema_preserves_units_shapes_dtype_and_availability():
    result = trace_fiber(_analytic_problem())
    sample_count = len(result.poses)

    assert result.status == FiberStatus.CLOSED
    assert result.reason == TerminationReason.CLOSED_LOOP
    assert result.component_scope == "one component reached from one seed"
    assert result.component_completeness == "unknown"
    assert result.poses.shape == (sample_count, 3, 3)
    assert result.tangents.shape == (sample_count, 3)
    assert result.residual_norms.shape == (sample_count,)
    assert result.arclength_increments.shape == (sample_count - 1,)
    assert result.poses.dtype == np.float64
    assert result.tangents.dtype == np.float64
    assert result.residual_norms.dtype == np.float64
    assert result.arclength_increments.dtype == np.float64
    assert np.all(result.arclength_increments >= 0.0)
    assert len(result.jacobian_diagnostics) == sample_count
    assert result.conventions == {
        "coordinate_sign": "phase1-v1",
        "pose_representation": "float64 rotation matrix (3, 3)",
        "metric_measure": "right-invariant-so3/dvol_g",
        "dtype": "float64",
        "arclength_unit": "radian",
        "solver_options_version": "reference-continuation-v1",
        "evaluation_unit": (
            "one gated pose or corrector iterate including smooth value and AD work"
        ),
    }
    assert set(result.weight_observables) == {
        "rho_pose",
        "entry_measure",
        "visibility",
        "fresnel_transmission",
        "path_validity",
        "source_factor",
        "pixel_factor",
        "other_radiometric",
    }
    assert set(result.weight_observables.values()) == {"unavailable"}


def test_public_schema_contains_every_contract_result_field():
    assert {field.name for field in fields(FiberResult)} == {
        "status",
        "reason",
        "poses",
        "arclength_increments",
        "residual_norms",
        "tangents",
        "jacobian_diagnostics",
        "step_diagnostics",
        "branch_diagnostics",
        "closure_diagnostics",
        "closure_attempt_diagnostics",
        "terminal_payload",
        "conventions",
        "weight_observables",
        "component_scope",
        "component_completeness",
    }
    assert set(FiberStatus) == {
        FiberStatus.CLOSED,
        FiberStatus.EVENT_TERMINATED,
        FiberStatus.NUMERICAL_FAILURE,
        FiberStatus.BUDGET_EXHAUSTED,
    }
    assert ContinuationOptions().dtype == "float64"


def test_analytic_circle_truth_closure_and_haar_normalization(analytic_sweep):
    for result in analytic_sweep.values():
        assert result.status == FiberStatus.CLOSED
        assert result.reason == TerminationReason.CLOSED_LOOP
        assert result.residual_norms.max() <= 1e-11
        assert result.closure_diagnostics.section_crossed
        assert result.closure_diagnostics.final_correction_accepted
        assert result.closure_diagnostics.seed_distance <= 1e-12
        np.testing.assert_allclose(
            result.arclength_increments.sum(), 2.0 * np.pi, rtol=0.0, atol=1e-12
        )
        np.testing.assert_allclose(
            result.arclength_increments.sum() / (8.0 * np.pi**2),
            1.0 / (4.0 * np.pi),
            rtol=0.0,
            atol=2e-14,
        )
        assert all(diagnostic.rank == 2 for diagnostic in result.jacobian_diagnostics)
        np.testing.assert_allclose(
            [diagnostic.singular_values for diagnostic in result.jacobian_diagnostics],
            1.0,
            rtol=0.0,
            atol=2e-14,
        )
        np.testing.assert_allclose(
            [diagnostic.normal_jacobian for diagnostic in result.jacobian_diagnostics],
            1.0,
            rtol=0.0,
            atol=2e-14,
        )
        tangent_dots = np.sum(result.tangents[:-1] * result.tangents[1:], axis=1)
        assert np.all(tangent_dots > 0.999999999999)


def test_analytic_circle_is_invariant_under_orthogonal_target_basis(analytic_sweep):
    problem = _analytic_problem()
    reference = analytic_sweep[0.04]
    basis = np.asarray(problem.target_chart.basis)
    transforms = (
        np.array([[0.0, -1.0], [1.0, 0.0]]),
        np.array([[1.0, 0.0], [0.0, -1.0]]),
    )

    for transform in transforms:
        changed_problem = FiberProblem(
            path=problem.path,
            incident_direction=problem.incident_direction,
            target_chart=TargetChart(
                problem.target_chart.direction,
                jnp.asarray(basis @ transform, dtype=jnp.float64),
                minimum_dot=problem.target_chart.minimum_dot,
            ),
            direction_evaluator=problem.direction_evaluator,
            seed=problem.seed,
        )
        changed = trace_fiber(changed_problem)

        assert changed.status == reference.status
        assert changed.reason == reference.reason
        assert abs(float(np.dot(changed.tangents[0], reference.tangents[0]))) >= (
            1.0 - 1e-14
        )
        np.testing.assert_allclose(
            changed.jacobian_diagnostics[0].singular_values,
            reference.jacobian_diagnostics[0].singular_values,
            rtol=0.0,
            atol=2e-14,
        )
        assert changed.jacobian_diagnostics[0].normal_jacobian == pytest.approx(
            reference.jacobian_diagnostics[0].normal_jacobian, abs=2e-14
        )
        assert _rotation_set_distance(changed.poses, reference.poses) <= 1e-12
        assert changed.arclength_increments.sum() == pytest.approx(
            reference.arclength_increments.sum(), abs=2e-12
        )


def test_synthetic_3_5_trace_matches_independent_direct_ray_oracle(
    optical_fixture, optical_sweep
):
    result = optical_sweep[0.04]
    seed = np.asarray(exp(jnp.array([0.15, 0.08, -0.05], dtype=jnp.float64)))
    incident = _oracle_incident_3_5()
    _, target, _ = _oracle_path_3_5(seed, incident)
    target_basis = _oracle_tangent_basis(target)

    np.testing.assert_allclose(optical_fixture.incident_direction, incident)
    np.testing.assert_allclose(optical_fixture.target_chart.direction, target)
    np.testing.assert_allclose(optical_fixture.target_chart.basis, target_basis)

    assert result.status == FiberStatus.CLOSED
    assert result.reason == TerminationReason.CLOSED_LOOP
    assert result.component_completeness == "unknown"
    for index, rotation in enumerate(result.poses):
        internal, outgoing, margins = _oracle_path_3_5(rotation, incident)
        np.testing.assert_allclose(np.linalg.norm(internal), 1.0, atol=3e-15)
        np.testing.assert_allclose(np.linalg.norm(outgoing), 1.0, atol=3e-15)
        assert float(np.dot(outgoing, target)) > 0.999999999999
        np.testing.assert_allclose(outgoing, target, rtol=0.0, atol=4e-15)
        np.testing.assert_allclose(
            np.linalg.norm(target_basis.T @ (outgoing - target)),
            result.residual_norms[index],
            rtol=0.0,
            atol=4e-16,
        )
        for name, value in margins.items():
            assert result.branch_diagnostics.accepted_margins[index][name] == (
                pytest.approx(value, abs=3e-15)
            )


def test_synthetic_3_5_safe_step_sweep_converges_without_fixed_step_count(
    optical_sweep,
):
    lengths = []
    for result in optical_sweep.values():
        sample_count = len(result.poses)
        lengths.append(float(result.arclength_increments.sum()))
        assert result.status == FiberStatus.CLOSED
        assert result.reason == TerminationReason.CLOSED_LOOP
        assert result.residual_norms.max() <= 1e-11
        assert result.closure_diagnostics.seed_distance <= 2e-13
        assert result.closure_diagnostics.accumulated_arclength == pytest.approx(
            lengths[-1], abs=2e-14
        )
        assert result.arclength_increments.shape == (sample_count - 1,)
        assert np.all(result.arclength_increments >= 0.0)
        assert len(result.branch_diagnostics.accepted_margins) == sample_count
        assert all(diagnostic.rank == 2 for diagnostic in result.jacobian_diagnostics)
        assert min(
            diagnostic.normal_jacobian
            for diagnostic in result.jacobian_diagnostics
        ) > 0.024

    assert max(lengths) - min(lengths) <= 0.0017


def test_synthetic_3_5_controller_threshold_perturbation_converges_consistently(
    optical_controller_sweep,
):
    reference = optical_controller_sweep["reference"]
    perturbed = optical_controller_sweep["perturbed"]

    assert perturbed.status == reference.status == FiberStatus.CLOSED
    assert perturbed.reason == reference.reason == TerminationReason.CLOSED_LOOP
    assert perturbed.residual_norms.max() <= 1e-11
    assert perturbed.closure_diagnostics.seed_distance <= 2e-13
    assert abs(
        float(perturbed.arclength_increments.sum())
        - float(reference.arclength_increments.sum())
    ) <= 0.0015
    assert _rotation_set_distance(perturbed.poses, reference.poses) <= 0.008
    assert abs(float(np.dot(perturbed.tangents[0], reference.tangents[0]))) >= (
        1.0 - 2e-14
    )


def test_synthetic_3_5_is_invariant_under_orthogonal_target_basis(
    optical_fixture, optical_sweep
):
    reference = optical_sweep[0.04]
    basis = np.asarray(optical_fixture.target_chart.basis)
    transforms = (
        np.array([[0.0, -1.0], [1.0, 0.0]]),
        np.array([[1.0, 0.0], [0.0, -1.0]]),
    )

    for transform in transforms:
        changed_problem = replace(
            optical_fixture,
            target_chart=replace(
                optical_fixture.target_chart,
                basis=jnp.asarray(basis @ transform, dtype=jnp.float64),
            ),
        )
        changed = trace_fiber(changed_problem)

        assert changed.status == reference.status
        assert changed.reason == reference.reason
        assert abs(float(np.dot(changed.tangents[0], reference.tangents[0]))) >= (
            1.0 - 2e-14
        )
        np.testing.assert_allclose(
            changed.jacobian_diagnostics[0].singular_values,
            reference.jacobian_diagnostics[0].singular_values,
            rtol=0.0,
            atol=2e-14,
        )
        assert changed.jacobian_diagnostics[0].normal_jacobian == pytest.approx(
            reference.jacobian_diagnostics[0].normal_jacobian, abs=2e-14
        )
        assert _rotation_set_distance(changed.poses, reference.poses) <= 2e-11
        assert changed.arclength_increments.sum() == pytest.approx(
            reference.arclength_increments.sum(), abs=2e-11
        )


def test_public_rank_loss_and_antipode_events_retain_causal_payload():
    constant = _analytic_problem(direction_evaluator=lambda _: BODY_AXIS)
    rank_loss = trace_fiber(constant)
    antipode = trace_fiber(
        _analytic_problem(direction_evaluator=lambda _: -BODY_AXIS)
    )

    assert rank_loss.status == FiberStatus.EVENT_TERMINATED
    assert rank_loss.reason == TerminationReason.RANK_LOSS
    assert rank_loss.terminal_payload.event is not None
    assert rank_loss.terminal_payload.event.details["sigma_2"] == 0.0
    assert rank_loss.terminal_payload.event.details["normal_jacobian"] == 0.0
    assert rank_loss.terminal_payload.event.details["rank"] == 0.0
    assert not rank_loss.poses.size
    assert antipode.status == FiberStatus.EVENT_TERMINATED
    assert antipode.reason == TerminationReason.CHART_BOUNDARY
    assert antipode.terminal_payload.event is not None
    assert antipode.terminal_payload.event.margin < 0.0
    assert not antipode.poses.size


def test_known_domain_event_precedes_unsafe_smooth_evaluation():
    direction_calls = 0

    def domain(rotation):
        margin = -float(rotation[1, 0])
        if margin < 0.0:
            return DomainEvaluation(
                valid=False,
                margins={"tir": margin},
                event=EventCandidate(TerminationReason.TIR_BOUNDARY, margin),
            )
        return DomainEvaluation(valid=True, margins={"tir": margin})

    def counted_direction(rotation):
        nonlocal direction_calls
        direction_calls += 1
        return direction_map(rotation)

    base = _analytic_problem(direction_evaluator=counted_direction)
    problem = replace(base, domain_and_event_evaluator=domain)
    result = trace_fiber(problem)

    assert result.status == FiberStatus.EVENT_TERMINATED
    assert result.reason == TerminationReason.TIR_BOUNDARY
    assert result.terminal_payload.event is not None
    assert result.terminal_payload.event.margin < 0.0
    assert result.terminal_payload.rejected_pose is not None
    assert result.terminal_payload.last_accepted_pose is not None
    assert direction_calls == 2


def test_public_corrector_nonconvergence_and_step_underflow_are_distinct():
    corrector = trace_fiber(
        _analytic_problem(),
        ContinuationOptions(maximum_advance=0.01, maximum_retries=0),
    )
    underflow = trace_fiber(
        _analytic_problem(),
        ContinuationOptions(
            initial_step=0.04,
            minimum_step=0.03,
            maximum_step=0.04,
            maximum_advance=0.01,
        ),
    )

    assert corrector.status == FiberStatus.NUMERICAL_FAILURE
    assert corrector.reason == TerminationReason.CORRECTOR_FAILURE
    assert len(corrector.step_diagnostics) == 1
    assert not corrector.step_diagnostics[0].accepted
    assert corrector.terminal_payload.rejected_pose is not None
    assert underflow.status == FiberStatus.NUMERICAL_FAILURE
    assert underflow.reason == TerminationReason.STEP_UNDERFLOW
    assert underflow.step_diagnostics[-1].reason == "corrector_failure"
    assert "below minimum" in underflow.terminal_payload.message


def test_public_nonfinite_evaluator_output_never_appears_closed():
    def nonfinite_after_seed(rotation):
        unsafe = rotation[1, 0] > 0.0
        return direction_map(rotation) + jnp.where(unsafe, jnp.nan, 0.0)

    result = trace_fiber(
        _analytic_problem(direction_evaluator=nonfinite_after_seed),
        ContinuationOptions(maximum_retries=2),
    )

    assert result.status == FiberStatus.NUMERICAL_FAILURE
    assert result.reason == TerminationReason.NON_FINITE
    assert len(result.step_diagnostics) == 3
    assert all(not diagnostic.accepted for diagnostic in result.step_diagnostics)
    assert result.terminal_payload.rejected_pose is not None
    assert result.terminal_payload.event is None


@pytest.mark.parametrize(
    ("options", "reason"),
    (
        (ContinuationOptions(maximum_accepted_steps=1), TerminationReason.STEP_BUDGET),
        (ContinuationOptions(maximum_arclength=0.01), TerminationReason.ARCLENGTH_BUDGET),
        (ContinuationOptions(maximum_evaluations=1), TerminationReason.EVALUATION_BUDGET),
    ),
)
def test_public_budget_termination_retains_partial_geometry(options, reason):
    result = trace_fiber(_analytic_problem(), options)

    assert result.status == FiberStatus.BUDGET_EXHAUSTED
    assert result.reason == reason
    assert result.terminal_payload.last_accepted_pose is not None
    assert result.poses.shape[0] >= 1
    assert result.arclength_increments.shape == (result.poses.shape[0] - 1,)
    assert result.component_completeness == "unknown"
    assert result.terminal_payload.message
