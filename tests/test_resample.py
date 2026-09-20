"""Spline resampling of traced fibers (task-resample-and-integrate Step 1)."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from lumice_integral.analytic import BODY_AXIS, direction_map, tangent_basis
from lumice_integral.canonical_scene import canonical_pixel_problem
from lumice_integral.continuation import ContinuationOptions, FiberProblem, FiberStatus, TargetChart, trace_fiber
from lumice_integral.resample import (
    FiberSpline,
    evaluate_spline,
    fiber_spline,
    resample_fiber,
    resample_spline,
    uniform_parameters,
)
from lumice_integral.so3 import rotation_distance


def _circle_problem() -> FiberProblem:
    return FiberProblem(
        path="analytic-body-axis",
        incident_direction=jnp.array([1.0, 0.0, 0.0], dtype=jnp.float64),
        target_chart=TargetChart(BODY_AXIS, tangent_basis(BODY_AXIS)),
        direction_evaluator=direction_map,
        seed=jnp.eye(3, dtype=jnp.float64),
    )


def _circle_angle(rotation: np.ndarray) -> float:
    return float(np.arctan2(rotation[1, 0], rotation[0, 0]))


@pytest.fixture(scope="module")
def canonical():
    problem = canonical_pixel_problem()
    return problem, trace_fiber(problem)


def test_closed_spline_drops_the_duplicated_closing_sample_and_wraps_to_the_seed(canonical):
    _, result = canonical
    assert result.status == FiberStatus.CLOSED
    spline = fiber_spline(result)

    assert spline.closed
    assert spline.segment_count == len(result.poses) - 1
    assert spline.total == pytest.approx(float(np.sum(result.arclength_increments)), abs=1e-15)
    np.testing.assert_allclose(spline.quaternions[-1], spline.quaternions[0], atol=0.0)
    # Sign continuity along the loop, including the wrapped neighbour.
    dots = np.sum(spline.quaternions[1:] * spline.quaternions[:-1], axis=1)
    assert np.all(dots > 0.0)


def test_spline_interpolates_every_accepted_pose_with_its_own_tangent(canonical):
    _, result = canonical
    spline = fiber_spline(result)
    at_knots = resample_spline(spline, spline.knots)

    poses = np.asarray(result.poses)
    for index in range(len(poses)):
        assert float(rotation_distance(at_knots.rotations[index], poses[index])) < 1e-14
    tangents = np.asarray(result.tangents)[:-1]
    assert np.min(np.sum(at_knots.phase_tangents[:-1] * tangents, axis=1)) > 1.0 - 1e-12


def test_resampled_predictors_are_uniform_in_parameter_and_close_to_the_fiber(canonical):
    problem, result = canonical
    predictors = resample_fiber(result, 129)

    assert predictors.rotations.shape == (129, 3, 3) and predictors.closed
    np.testing.assert_allclose(np.diff(predictors.parameters), predictors.total / 128, rtol=1e-12)
    identity_defect = np.einsum("nji,njk->nik", predictors.rotations, predictors.rotations) - np.eye(3)
    assert np.max(np.abs(identity_defect)) < 1e-14
    assert float(rotation_distance(predictors.rotations[-1], predictors.rotations[0])) < 1e-14
    # Off the fiber by no more than the spline's own error (Step 1 probe: 1e-7..3e-6
    # against the retracted midpoints, 300x closer than the chord midpoints).
    residual = np.asarray(
        [np.linalg.norm(np.asarray(problem.target_chart.basis).T @ (np.asarray(problem.direction_evaluator(jnp.asarray(r))) - np.asarray(problem.target_chart.direction))) for r in predictors.rotations]
    )
    assert residual.max() < 5e-6


def test_analytic_circle_resamples_onto_its_exact_parametrisation():
    result = trace_fiber(_circle_problem())
    predictors = resample_fiber(result, 65)

    # On the geodesic circle the chord equals the arclength, so the spline
    # parameter is the angle and the body velocity is the unit e3 tangent.
    angles = np.unwrap([_circle_angle(r) for r in predictors.rotations])
    sign = np.sign(angles[1] - angles[0])
    np.testing.assert_allclose(sign * (angles - angles[0]), predictors.parameters, atol=2e-9)
    np.testing.assert_allclose(np.linalg.norm(predictors.body_velocities, axis=1), 1.0, atol=2e-9)


def test_open_arc_keeps_both_end_samples_exactly(canonical):
    problem, _ = canonical
    result = trace_fiber(problem, ContinuationOptions(maximum_accepted_steps=20))
    assert result.status == FiberStatus.BUDGET_EXHAUSTED
    spline = fiber_spline(result)
    predictors = resample_fiber(result, 33)

    assert not spline.closed and not predictors.closed
    assert spline.segment_count == len(result.poses) - 1
    assert float(rotation_distance(predictors.rotations[0], result.poses[0])) < 1e-14
    assert float(rotation_distance(predictors.rotations[-1], result.poses[-1])) < 1e-14


def test_spline_evaluation_rejects_parameters_outside_the_range(canonical):
    _, result = canonical
    spline = fiber_spline(result)
    with pytest.raises(ValueError):
        evaluate_spline(spline, np.array([-1e-3]))
    with pytest.raises(ValueError):
        uniform_parameters(spline, 1)
    assert isinstance(spline, FiberSpline)


# --- bucketed knot batches (task-pixel-cost-shape-stable-kernels Step 2.2) ---


def test_bucket_length_is_the_next_power_of_two():
    from lumice_integral.resample import _bucket_length

    assert [_bucket_length(n) for n in (0, 1, 2, 3, 4, 5, 8, 9, 1000, 1024, 1025)] == [
        1, 1, 2, 4, 4, 8, 8, 16, 1024, 1024, 2048,
    ]


@pytest.mark.parametrize("count", [1, 3, 8, 37])
def test_padded_knot_batches_match_the_unpadded_kernels_bit_for_bit(count):
    """Padding members are elementwise-isolated: the first ``count`` rows are exactly the plain kernel's.

    Against the *un-jitted* per-element functions the compiled kernels differ
    by XLA's fusion (last-bit contractions), which is a property of ``jit``,
    not of the padding; that link is checked at a few ulp.
    """
    from lumice_integral.resample import (
        _quaternion_derivative_batch,
        _quaternion_derivative_kernel,
        _quaternion_from_rotation_batch,
        _quaternion_from_rotation_kernel,
    )
    from lumice_integral.so3 import exp, quaternion_derivative, quaternion_from_rotation

    rng = np.random.default_rng(count)
    rotations = np.stack([np.asarray(exp(jnp.asarray(v))) for v in rng.normal(size=(count, 3))])
    tangents = rng.normal(size=(count, 3))

    quaternions = _quaternion_from_rotation_batch(rotations)
    assert quaternions.shape == (count, 4)
    np.testing.assert_array_equal(quaternions, np.asarray(_quaternion_from_rotation_kernel(jnp.asarray(rotations))))
    eager = np.stack([np.asarray(quaternion_from_rotation(jnp.asarray(r))) for r in rotations])
    np.testing.assert_allclose(quaternions, eager, rtol=0.0, atol=4e-16)

    derivatives = _quaternion_derivative_batch(quaternions, tangents)
    assert derivatives.shape == (count, 4)
    np.testing.assert_array_equal(
        derivatives, np.asarray(_quaternion_derivative_kernel(jnp.asarray(quaternions), jnp.asarray(tangents)))
    )
    eager = np.stack(
        [np.asarray(quaternion_derivative(jnp.asarray(q), jnp.asarray(t))) for q, t in zip(quaternions, tangents)]
    )
    assert np.all(np.abs(derivatives - eager) <= 4e-16 * (1.0 + np.abs(eager)))
