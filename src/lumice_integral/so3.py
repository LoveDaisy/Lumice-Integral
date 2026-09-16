"""Small SO(3) helpers using right-trivialized tangent coordinates."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array, lax


def _rodrigues_taylor(value: Array) -> tuple[Array, Array]:
    value_squared = value * value
    a = 1.0 - value / 6.0 + value_squared / 120.0
    b = 0.5 - value / 24.0 + value_squared / 720.0
    return a, b


def _rodrigues_regular(value: Array) -> tuple[Array, Array]:
    theta = jnp.sqrt(value)
    return jnp.sin(theta) / theta, (1.0 - jnp.cos(theta)) / value


def hat(vector: Array) -> Array:
    """Return the skew matrix whose action is ``vector x operand``."""
    x, y, z = vector
    return jnp.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=vector.dtype,
    )


def exp(rotation_vector: Array) -> Array:
    """Map a rotation vector to SO(3) with a zero-safe Rodrigues formula."""
    theta_squared = jnp.dot(rotation_vector, rotation_vector)
    generator = hat(rotation_vector)

    a, b = lax.cond(
        theta_squared < 1e-8,
        _rodrigues_taylor,
        _rodrigues_regular,
        theta_squared,
    )
    return jnp.eye(3, dtype=rotation_vector.dtype) + a * generator + b * (
        generator @ generator
    )


def _log_taylor(theta_squared: Array) -> Array:
    return 1.0 + theta_squared / 6.0 + 7.0 * theta_squared * theta_squared / 360.0


def _log_regular(theta_squared: Array) -> Array:
    theta = jnp.sqrt(theta_squared)
    return theta / jnp.sin(theta)


def log(rotation: Array) -> Array:
    """Map a rotation to its rotation vector on the injectivity domain.

    Zero-safe like :func:`exp`; the angle is recovered from ``arctan2`` as in
    :func:`rotation_distance`.  Rotations with angle close to ``pi`` are not
    supported (the axis becomes ill-conditioned there).
    """
    cosine = jnp.clip((jnp.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    skew_vector = jnp.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ]
    ) / 2.0
    sine_squared = jnp.dot(skew_vector, skew_vector)
    # Zero-safe norm: keeps the AD graph finite at the identity.
    positive = sine_squared > 0.0
    sine = jnp.where(positive, jnp.sqrt(jnp.where(positive, sine_squared, 1.0)), 0.0)
    theta = jnp.arctan2(sine, cosine)
    scale = lax.cond(theta * theta < 1e-8, _log_taylor, _log_regular, theta * theta)
    return scale * skew_vector


def rotation_distance(left: Array, right: Array) -> Array:
    """Return the geodesic angle between two rotation matrices."""
    relative = left.T @ right
    cosine = jnp.clip((jnp.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    skew_vector = jnp.array(
        [
            relative[2, 1] - relative[1, 2],
            relative[0, 2] - relative[2, 0],
            relative[1, 0] - relative[0, 1],
        ]
    ) / 2.0
    sine = jnp.linalg.norm(skew_vector)
    return jnp.arctan2(sine, cosine)
