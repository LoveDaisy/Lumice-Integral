from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lumice_integral.so3 import exp, log, rotation_distance, rotation_distances


@pytest.mark.parametrize("angle", [0.0, 1e-9, 1e-5, 1e-3, 0.1, 1.0, 3.0])
def test_log_inverts_exp_on_the_injectivity_domain(angle):
    axis = np.array([0.3, -0.5, 0.8])
    axis /= np.linalg.norm(axis)
    vector = jnp.asarray(angle * axis, dtype=jnp.float64)

    recovered = np.asarray(log(exp(vector)))

    np.testing.assert_allclose(recovered, angle * axis, rtol=0.0, atol=4e-16 * (1.0 + angle))
    assert float(rotation_distance(jnp.eye(3), exp(vector))) == pytest.approx(angle, abs=4e-16 * (1.0 + angle))


def test_log_is_zero_and_ad_safe_at_the_identity():
    np.testing.assert_array_equal(np.asarray(log(jnp.eye(3, dtype=jnp.float64))), 0.0)
    # d/dv log(exp(v)) at v = 0 is the identity: both branches must be AD safe.
    jacobian = np.asarray(jax.jacfwd(lambda v: log(exp(v)))(jnp.zeros(3, dtype=jnp.float64)))
    np.testing.assert_allclose(jacobian, np.eye(3), rtol=0.0, atol=1e-15)


# --- host-side batch distance (task-pixel-cost-shape-stable-kernels) ----------


def test_rotation_distances_matches_the_jax_kernel_pairwise():
    """The numpy batch is the same formula as ``rotation_distance``: ulp-level agreement."""
    rng = np.random.default_rng(20260920)
    axes = rng.normal(size=(64, 3))
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    # Angles across the range, including the near-identity and near-pi ends.
    angles = np.concatenate(
        ([0.0, 1e-9, 1e-6, 1e-3, np.pi - 1e-6, np.pi], rng.uniform(0.0, np.pi, size=58))
    )
    rotations = np.stack([np.asarray(exp(jnp.asarray(a * v))) for a, v in zip(angles, axes)])
    left = np.asarray(exp(jnp.array([0.4, -0.7, 0.2], dtype=jnp.float64)))

    batch = rotation_distances(left, rotations)
    pairwise = np.array([float(rotation_distance(jnp.asarray(left), jnp.asarray(r))) for r in rotations])

    assert batch.shape == (64,) and batch.dtype == np.float64
    # Absolute agreement scaled by the angle (both sides lose the same
    # O(eps) in the matrix product), not a fixed single-machine tolerance.
    assert np.all(np.abs(batch - pairwise) <= 8e-16 * (1.0 + pairwise)), np.abs(batch - pairwise).max()
    # And the identity member is at the same distance the pairwise kernel reports for it.
    assert rotation_distances(left, left[None])[0] == pytest.approx(0.0, abs=1e-15)


def test_rotation_distances_accepts_jax_inputs_and_rejects_bad_shapes():
    left = exp(jnp.array([0.1, 0.2, 0.3], dtype=jnp.float64))
    rights = jnp.stack([left, jnp.eye(3, dtype=jnp.float64)])
    distances = rotation_distances(left, rights)
    assert isinstance(distances, np.ndarray)
    assert distances[0] == pytest.approx(0.0, abs=1e-15)
    assert distances[1] == pytest.approx(float(rotation_distance(left, jnp.eye(3))), abs=1e-15)
    assert rotation_distances(np.eye(3), np.zeros((0, 3, 3))).shape == (0,)
    with pytest.raises(ValueError):
        rotation_distances(np.eye(3), np.eye(3))
    with pytest.raises(ValueError):
        rotation_distances(np.eye(4), np.zeros((2, 3, 3)))


# --- quaternion chart (task-resample-and-integrate Step 1) --------------------


@pytest.mark.parametrize("angle", [0.0, 1e-6, 0.3, 2.0, np.pi - 1e-9, np.pi])
def test_quaternion_round_trip_including_angles_near_pi(angle):
    from lumice_integral.so3 import quaternion_from_rotation, rotation_from_quaternion

    axis = np.array([0.3, -0.5, 0.8])
    axis /= np.linalg.norm(axis)
    rotation = exp(jnp.asarray(angle * axis, dtype=jnp.float64))

    quaternion = np.asarray(quaternion_from_rotation(rotation))
    recovered = np.asarray(rotation_from_quaternion(jnp.asarray(quaternion)))

    assert np.linalg.norm(quaternion) == pytest.approx(1.0, abs=1e-15)
    np.testing.assert_allclose(recovered, np.asarray(rotation), rtol=0.0, atol=4e-15)
    # (w, x, y, z) with w = cos(angle / 2) up to the overall sign.
    assert abs(abs(quaternion[0]) - abs(np.cos(angle / 2.0))) < 1e-14


def test_quaternion_derivative_matches_the_right_trivialized_body_velocity():
    from lumice_integral.so3 import (
        quaternion_derivative,
        quaternion_from_rotation,
        rotation_from_quaternion,
        vee,
    )

    quaternion = quaternion_from_rotation(exp(jnp.array([0.3, -0.2, 0.5], dtype=jnp.float64)))
    omega = jnp.array([0.1, 0.4, -0.3], dtype=jnp.float64)
    rotation, rotation_rate = jax.jvp(
        rotation_from_quaternion, (quaternion,), (quaternion_derivative(quaternion, omega),)
    )
    # dR/ds = R hat(omega)  <=>  vee(R^T dR/ds) = omega
    np.testing.assert_allclose(np.asarray(vee(rotation.T @ rotation_rate)), np.asarray(omega), atol=1e-15)


def test_continuous_quaternion_signs_flips_only_where_consecutive_dots_are_negative():
    from lumice_integral.so3 import continuous_quaternion_signs

    base = np.array([[1.0, 0.0, 0.0, 0.0], [0.9, 0.1, 0.0, 0.0], [0.8, 0.2, 0.1, 0.0], [0.7, 0.3, 0.1, 0.1]])
    flipped = base.copy()
    flipped[1] *= -1.0
    flipped[2] *= -1.0

    signed = continuous_quaternion_signs(flipped)

    np.testing.assert_array_equal(signed, base)
    np.testing.assert_array_equal(continuous_quaternion_signs(base), base)
    with pytest.raises(ValueError):
        continuous_quaternion_signs(np.zeros((3, 3)))
