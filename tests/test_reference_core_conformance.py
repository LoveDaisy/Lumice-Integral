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
from lumice_integral.camera import linear_pixel_outgoing_direction
from lumice_integral.canonical_scene import (
    CANONICAL_REFRACTIVE_INDEX,
    CANONICAL_RENDER,
    canonical_incident_direction,
)
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
from lumice_integral.optics import path_3_5, path_3_5_problem
from lumice_integral.so3 import exp
from lumice_integral.weights import WeightObservable

# Reference defaults of the closure extent gate before
# task-continuation-gates-and-fixtures (absolute: 40 steps and pi arclength).
# Passed literally to reproduce the pre-fix behaviour on the same fixtures.
_LEGACY_CLOSURE_GATE = dict(closure_minimum_steps=40, closure_minimum_arclength=np.pi)


def _analytic_problem(*, direction_evaluator=direction_map) -> FiberProblem:
    return FiberProblem(
        path="analytic-body-axis",
        incident_direction=jnp.array([1.0, 0.0, 0.0], dtype=jnp.float64),
        target_chart=TargetChart(BODY_AXIS, tangent_basis(BODY_AXIS)),
        direction_evaluator=direction_evaluator,
        seed=jnp.eye(3, dtype=jnp.float64),
    )


# --- Analytic short loops: the conjugation circle --------------------------
#
# Every map of the form ``G(R v)`` (the body-axis map above, a double-mirror
# reflection, ...) has fibers that are cosets of a one-parameter subgroup and
# hence length ``2 pi``, so a short analytic loop needs a map that reads more
# than one body vector.  ``_conjugation_map`` reads the rotation angle
# ``phi`` and the axis component ``sin(phi) u . a`` of the pose itself:
# ``F(R) = normalize((cos phi, sin(phi) u . a, 1))``.  Its fiber through
# ``Rot(u0, theta)`` is the conjugation circle ``t -> Rot(exp(t a) u0, theta)``
# of rotations by the fixed angle ``theta`` about axes on the cone
# ``u . a = cos beta``.  Its body angular velocity ``R^T a - a`` has the
# constant norm ``2 sin(theta / 2) sin(beta)``, so the loop length is exactly
# ``4 pi sin(theta / 2) sin(beta)`` under the section 5.1 metric: two knobs
# that place the length anywhere below ``2 pi``.  Both ``(cos phi, sin phi u)``
# are smooth functions of ``R`` away from angle ``pi``.
_CONJUGATION_CONE_AXIS = jnp.array([0.0, 0.0, 1.0], dtype=jnp.float64)
_CONJUGATION_ANGLE = np.pi / 2.0


def _conjugation_map(rotation):
    cos_angle = (jnp.trace(rotation) - 1.0) / 2.0
    sin_angle_axis = jnp.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ]
    ) / 2.0
    lifted = jnp.array([cos_angle, sin_angle_axis @ _CONJUGATION_CONE_AXIS, 1.0])
    return lifted / jnp.linalg.norm(lifted)


def _conjugation_loop_length(cone_angle: float) -> float:
    return 4.0 * np.pi * np.sin(_CONJUGATION_ANGLE / 2.0) * np.sin(cone_angle)


def _conjugation_cone_angle(loop_length: float) -> float:
    return float(np.arcsin(loop_length / (4.0 * np.pi * np.sin(_CONJUGATION_ANGLE / 2.0))))


def _conjugation_problem(loop_length: float) -> FiberProblem:
    cone_angle = _conjugation_cone_angle(loop_length)
    axis = jnp.array([np.sin(cone_angle), 0.0, np.cos(cone_angle)], dtype=jnp.float64)
    seed = exp(_CONJUGATION_ANGLE * axis)
    target = _conjugation_map(seed)
    return FiberProblem(
        path=f"analytic-conjugation-circle:L={loop_length}",
        incident_direction=jnp.array([1.0, 0.0, 0.0], dtype=jnp.float64),
        target_chart=TargetChart(target, tangent_basis(target)),
        direction_evaluator=_conjugation_map,
        seed=seed,
    )


# --- ch06 strip pixels: short loops and a boundary-hugging loop -------------
#
# Seeds frozen from ``discovery.discover_components`` on the canonical scene
# (``rng_seed=20260916``, 400k prescan samples, 2 deg, 0.3 rad; the
# ``tests/test_discovery.py`` parameters), task-continuation-gates-and-fixtures
# Step 0.  Column 126 rows 100/150/224 sit on the lit band below the inner
# caustic where the 3-5 loop is shorter than pi; column 150 rows 700/780 are
# the strip's lower band where the loop runs parallel to the exit TIR boundary
# (``exit_snell_discriminant`` near 0.0175 for the whole lower half of the
# loop).  Exponential coordinates, ``so3.exp`` gives the seed pose.
_STRIP_PIXEL_SEED_COORDINATES: dict[tuple[int, int], tuple[float, float, float]] = {
    (100, 126): (-1.5066578217593831, 0.39664283175724213, 0.0890374620046349),
    (150, 126): (-1.4208326302791248, 0.614453957148551, -0.0473172496332926),
    (224, 126): (-1.5933577786781372, -0.12433770957520761, 0.4211233203809581),
    (700, 150): (-1.614678779370118, -0.3852552208498327, 1.2983286011517758),
    (780, 150): (-1.68895386947514, 0.14376477560238074, 1.4516482547130902),
}
# Single-traversal metric lengths under the reference defaults (Mac, float64);
# the pre-fix absolute gate closed these three on the second traversal at
# exactly twice these values (issue evidence: 3.291 / 4.751 / 6.222).
_STRIP_SHORT_LOOP_LENGTHS = {
    (100, 126): 1.645239,
    (150, 126): 2.375620,
    (224, 126): 3.111244,
}


def _strip_pixel_problem(row: int, column: int) -> FiberProblem:
    return path_3_5_problem(
        exp(jnp.asarray(_STRIP_PIXEL_SEED_COORDINATES[(row, column)], dtype=jnp.float64)),
        jnp.asarray(canonical_incident_direction(), dtype=jnp.float64),
        target_direction=jnp.asarray(
            linear_pixel_outgoing_direction(row, column, **CANONICAL_RENDER),
            dtype=jnp.float64,
        ),
        refractive_index=jnp.asarray(CANONICAL_REFRACTIVE_INDEX, dtype=jnp.float64),
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


_CONJUGATION_SWEEP_STEPS = (0.01, 0.02, 0.04, 0.2)


@pytest.fixture(scope="module")
def conjugation_sweep():
    # ``maximum_step=step`` so that the step actually sets the resolution
    # (the controller otherwise grows every run to the same 0.12 cap).
    return {
        (length, step): trace_fiber(
            _conjugation_problem(length),
            ContinuationOptions(initial_step=step, maximum_step=step),
        )
        for length in (1.0, 2.0)
        for step in _CONJUGATION_SWEEP_STEPS
    }


@pytest.fixture(scope="module")
def strip_short_loops():
    return {
        pixel: trace_fiber(_strip_pixel_problem(*pixel))
        for pixel in _STRIP_SHORT_LOOP_LENGTHS
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
        "haar_to_dvol_g_factor": "1/(8*pi**2)",
        "coarea_denominator": (
            "normal_jacobian J_perp in jacobian_diagnostics; never folded into weights"
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
    # C14: with no evaluator registered every factor is explicitly unavailable
    # (no unit, no normalization, no stand-in values), never silently one.
    for observable in result.weight_observables.values():
        assert isinstance(observable, WeightObservable)
        assert observable == WeightObservable("unavailable")


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
    # Sampled-pose set distance between two discretisations of one traversal
    # of the 0.964 loop (0.0093 observed; the earlier 0.008 bound was measured
    # on four traversals, whose interleaved samples lay closer).
    assert _rotation_set_distance(perturbed.poses, reference.poses) <= 0.012
    assert abs(float(np.dot(perturbed.tangents[0], reference.tangents[0]))) >= (
        1.0 - 2e-14
    )


def test_analytic_conjugation_loops_shorter_than_pi_close_on_the_first_traversal(
    conjugation_sweep,
):
    """(a) Loops of length 1.0 and 2.0 close once, with the closed-form length."""
    for (length, step), result in conjugation_sweep.items():
        assert result.status == FiberStatus.CLOSED
        assert result.reason == TerminationReason.CLOSED_LOOP
        assert result.residual_norms.max() <= 1e-11
        assert result.closure_diagnostics.section_crossed
        assert result.closure_diagnostics.final_correction_accepted
        assert result.closure_diagnostics.seed_distance <= 1e-12
        assert all(diagnostic.rank == 2 for diagnostic in result.jacobian_diagnostics)
        # Chord lengths of the sampled polygon under-estimate the metric
        # length by O(h^2): within 1e-2 relative even at the 0.2 cap.
        traced = float(result.arclength_increments.sum())
        assert 0.0 < length - traced <= 1e-2 * length
    for length in (1.0, 2.0):
        # Second-order convergence to the closed-form length over the capped
        # steps 0.01 / 0.02 / 0.04 (error ratio about 4 per halving); the
        # finest is within 2e-4 relative.
        errors = [
            length - float(conjugation_sweep[(length, step)].arclength_increments.sum())
            for step in _CONJUGATION_SWEEP_STEPS[:3]
        ]
        assert errors[0] <= 2e-4 * length
        for coarse, fine in zip(errors[1:], errors[:-1]):
            assert 3.0 <= coarse / fine <= 5.0


def test_legacy_absolute_closure_gate_traverses_the_analytic_short_loops_repeatedly(
    conjugation_sweep,
):
    """Counterexample: the pre-fix gate closes at the first traversal past pi."""
    for length, traversals in ((1.0, 4), (2.0, 2)):
        legacy = trace_fiber(
            _conjugation_problem(length), ContinuationOptions(**_LEGACY_CLOSURE_GATE)
        )
        reference = conjugation_sweep[(length, 0.04)]
        assert legacy.reason == TerminationReason.CLOSED_LOOP
        assert float(legacy.arclength_increments.sum()) == pytest.approx(
            traversals * float(reference.arclength_increments.sum()), rel=1e-3
        )


def test_strip_short_loops_close_at_their_single_traversal_length(strip_short_loops):
    """(b) Column 126 rows 100/150/224 recover the single-loop lengths."""
    for pixel, result in strip_short_loops.items():
        assert result.status == FiberStatus.CLOSED
        assert result.reason == TerminationReason.CLOSED_LOOP
        assert result.residual_norms.max() <= 1e-11
        assert result.closure_diagnostics.section_crossed
        assert result.closure_diagnostics.final_correction_accepted
        assert result.closure_diagnostics.seed_distance <= 1e-12
        assert float(result.arclength_increments.sum()) == pytest.approx(
            _STRIP_SHORT_LOOP_LENGTHS[pixel], abs=1e-6
        )


@pytest.mark.parametrize("pixel", sorted(_STRIP_SHORT_LOOP_LENGTHS))
def test_legacy_absolute_closure_gate_doubles_the_strip_short_loops(
    strip_short_loops, pixel
):
    """(c) Double traversal: the pre-fix defaults on the same seeds, literally.

    This is the real behaviour of the reference defaults before
    task-continuation-gates-and-fixtures on rows 58-225 of the ch06 strip,
    not a constructed scenario: the loop is shorter than pi, so the absolute
    gate let the trace pass its seed once and close on the second return.
    """
    legacy = trace_fiber(_strip_pixel_problem(*pixel), ContinuationOptions(**_LEGACY_CLOSURE_GATE))
    single = float(strip_short_loops[pixel].arclength_increments.sum())
    assert legacy.reason == TerminationReason.CLOSED_LOOP
    # The second traversal is sampled on a different polygon, hence rtol 1e-3
    # (the arclength tolerance the retired fingerprint dedup of ``discovery`` used).
    assert float(legacy.arclength_increments.sum()) == pytest.approx(2.0 * single, rel=1e-3)
    assert legacy.closure_diagnostics.seed_distance <= 1e-12
    assert single == pytest.approx(_STRIP_SHORT_LOOP_LENGTHS[pixel], abs=1e-6)


@pytest.mark.parametrize("pixel", [(700, 150), (780, 150)])
def test_strip_boundary_hugging_loops_no_longer_exhaust_the_step_budget(pixel):
    """(d) Loops running parallel to the exit TIR boundary close normally.

    Before the rate-based event slowdown these seeds crawled at
    ``minimum_step`` from about the 80th accepted step until the 4000-step
    budget ran out (``explore-continuation-degenerate-stall-diagnosis``).
    """
    options = ContinuationOptions()
    result = trace_fiber(_strip_pixel_problem(*pixel), options)
    assert result.reason != TerminationReason.STEP_BUDGET
    assert result.status == FiberStatus.CLOSED
    assert result.reason == TerminationReason.CLOSED_LOOP
    assert result.residual_norms.max() <= 1e-11
    assert result.closure_diagnostics.seed_distance <= 1e-12
    accepted = [d for d in result.step_diagnostics if d.accepted]
    assert not any(d.proposed_step <= options.minimum_step for d in accepted)
    margins = [d.event_margins["exit_snell_discriminant"] for d in accepted]
    assert min(margins) < options.event_slowdown_margin
    assert min(margins) > 0.0


def test_analytic_circle_closes_on_the_first_traversal_with_a_large_step():
    """(e) A 0.2 initial step cannot jump the section or close early."""
    result = trace_fiber(
        _analytic_problem(), ContinuationOptions(initial_step=0.2, maximum_step=0.2)
    )
    assert result.status == FiberStatus.CLOSED
    assert result.reason == TerminationReason.CLOSED_LOOP
    assert result.residual_norms.max() <= 1e-11
    assert result.closure_diagnostics.section_crossed
    assert result.closure_diagnostics.final_correction_accepted
    assert result.closure_diagnostics.seed_distance <= 1e-12
    np.testing.assert_allclose(
        result.closure_diagnostics.accumulated_arclength, 2.0 * np.pi, rtol=0.0, atol=1e-12
    )
    # One crossing only: the seed-relative transverse coordinate (the axis
    # component of ``seed^T pose`` along the seed tangent) changes sign exactly
    # once along the sampled loop, at the closing edge.
    seed = np.asarray(result.poses[0])
    tangent = np.asarray(result.tangents[0])
    sections = []
    for pose in result.poses[1:]:
        relative = seed.T @ np.asarray(pose)
        skew = np.array(
            [
                relative[2, 1] - relative[1, 2],
                relative[0, 2] - relative[2, 0],
                relative[1, 0] - relative[0, 1],
            ]
        ) / 2.0
        sections.append(float(np.dot(tangent, skew)))
    sign_changes = sum(
        1 for left, right in zip(sections[:-1], sections[1:]) if left * right < 0.0
    )
    assert sign_changes == 1


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
