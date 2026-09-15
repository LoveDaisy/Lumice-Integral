"""Reference predictor-corrector continuation on SO(3)."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from .analytic import residual, residual_after_update
from .so3 import exp, rotation_distance


@dataclass(frozen=True)
class TraceResult:
    rotations: np.ndarray
    residual_norms: np.ndarray
    closure_error: float
    length: float


def constraint_jacobian(rotation: Array, target: Array, basis: Array) -> Array:
    """Differentiate the local residual in right-trivialized coordinates."""
    zero = jnp.zeros(3, dtype=rotation.dtype)
    return jax.jacfwd(residual_after_update, argnums=0)(
        zero, rotation, target, basis
    )


def null_tangent(jacobian: Array, previous: Array | None = None) -> Array:
    """Return an oriented unit vector spanning a rank-two Jacobian's nullspace."""
    _, _, vh = jnp.linalg.svd(jacobian, full_matrices=True)
    tangent = vh[-1]
    if previous is not None:
        tangent = jnp.where(jnp.dot(tangent, previous) < 0.0, -tangent, tangent)
    return tangent / jnp.linalg.norm(tangent)


def correct(
    predicted: Array,
    target: Array,
    basis: Array,
    tangent: Array,
    *,
    tolerance: float = 1e-13,
    max_iterations: int = 8,
) -> Array:
    """Correct normal to the predicted fiber tangent using bordered Newton steps."""
    rotation = predicted
    for _ in range(max_iterations):
        value = residual(rotation, target, basis)
        if float(jnp.linalg.norm(value)) <= tolerance:
            break
        jacobian = constraint_jacobian(rotation, target, basis)
        bordered = jnp.concatenate((jacobian, tangent[jnp.newaxis, :]), axis=0)
        step = jnp.linalg.solve(bordered, jnp.concatenate((-value, jnp.zeros(1))))
        rotation = rotation @ exp(step)
    return rotation


def trace_closed_fiber(
    initial: Array,
    target: Array,
    basis: Array,
    *,
    steps: int = 128,
) -> TraceResult:
    """Trace one known full turn of the analytic fiber."""
    step_size = 2.0 * jnp.pi / steps
    rotation = initial
    tangent = null_tangent(constraint_jacobian(rotation, target, basis))
    rotations = [np.asarray(rotation)]
    residual_norms = [float(jnp.linalg.norm(residual(rotation, target, basis)))]

    for _ in range(steps):
        predicted = rotation @ exp(step_size * tangent)
        rotation = correct(predicted, target, basis, tangent)
        tangent = null_tangent(
            constraint_jacobian(rotation, target, basis), previous=tangent
        )
        rotations.append(np.asarray(rotation))
        residual_norms.append(
            float(jnp.linalg.norm(residual(rotation, target, basis)))
        )

    return TraceResult(
        rotations=np.stack(rotations),
        residual_norms=np.asarray(residual_norms),
        closure_error=float(rotation_distance(initial, rotation)),
        length=float(steps * step_size),
    )

