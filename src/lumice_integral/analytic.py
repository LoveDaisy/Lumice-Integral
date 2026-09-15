"""Analytic SO(3) fiber used as the first continuation fixture."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from .so3 import exp


BODY_AXIS = jnp.array([0.0, 0.0, 1.0], dtype=jnp.float64)


def tangent_basis(direction: Array) -> Array:
    """Construct a deterministic orthonormal basis of a sphere tangent plane."""
    direction = direction / jnp.linalg.norm(direction)
    candidates = jnp.eye(3, dtype=direction.dtype)
    reference = candidates[jnp.argmin(jnp.abs(candidates @ direction))]
    first = jnp.cross(direction, reference)
    first = first / jnp.linalg.norm(first)
    second = jnp.cross(direction, first)
    return jnp.stack((first, second), axis=1)


def direction_map(rotation: Array) -> Array:
    """Map a crystal pose to the world direction of its body z axis."""
    return rotation @ BODY_AXIS


def residual(rotation: Array, target: Array, basis: Array) -> Array:
    """Represent the sphere constraint in a fixed target tangent chart."""
    return basis.T @ (direction_map(rotation) - target)


def residual_after_update(
    delta: Array, rotation: Array, target: Array, basis: Array
) -> Array:
    return residual(rotation @ exp(delta), target, basis)
