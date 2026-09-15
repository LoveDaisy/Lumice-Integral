"""Reference predictor-corrector continuation on SO(3)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

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
    steps: int
    detected_gap: float


ResidualFunction = Callable[[Array], Array]


def residual_jacobian(rotation: Array, residual_function: ResidualFunction) -> Array:
    """Differentiate an arbitrary two-component residual on SO(3)."""
    zero = jnp.zeros(3, dtype=rotation.dtype)
    return jax.jacfwd(lambda delta: residual_function(rotation @ exp(delta)))(zero)


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
        steps=steps,
        detected_gap=float(rotation_distance(initial, rotation)),
    )


def _section_coordinate(initial: Array, rotation: Array, tangent: Array) -> Array:
    relative = initial.T @ rotation
    skew_vector = jnp.array(
        [
            relative[2, 1] - relative[1, 2],
            relative[0, 2] - relative[2, 0],
            relative[1, 0] - relative[0, 1],
        ]
    ) / 2.0
    return jnp.dot(tangent, skew_vector)


def trace_implicit_fiber(
    initial: Array,
    residual_function: ResidualFunction,
    *,
    step_size: float = 0.02,
    max_steps: int = 2000,
    min_steps: int = 100,
    closure_search_radius: float = 0.25,
    tolerance: float = 1e-11,
) -> TraceResult:
    """Trace a regular implicit fiber until it returns through the start section."""
    evaluate = jax.jit(residual_function)
    differentiate = jax.jit(lambda rotation: residual_jacobian(rotation, residual_function))
    rotation = initial
    initial_tangent = null_tangent(differentiate(rotation))
    tangent = initial_tangent
    rotations = [np.asarray(rotation)]
    residual_norms = [float(jnp.linalg.norm(evaluate(rotation)))]
    previous_section = float(_section_coordinate(initial, rotation, initial_tangent))
    minimum_gap = float("inf")
    section_crossings = 0

    for step_index in range(1, max_steps + 1):
        predicted = rotation @ exp(
            jnp.asarray(step_size, dtype=rotation.dtype) * tangent
        )
        rotation = predicted
        for _ in range(8):
            value = evaluate(rotation)
            if float(jnp.linalg.norm(value)) <= tolerance:
                break
            jacobian = differentiate(rotation)
            bordered = jnp.concatenate(
                (jacobian, tangent[jnp.newaxis, :]), axis=0
            )
            update = jnp.linalg.solve(
                bordered,
                jnp.concatenate((-value, jnp.zeros(1, dtype=rotation.dtype))),
            )
            rotation = rotation @ exp(update)

        residual_norm = float(jnp.linalg.norm(evaluate(rotation)))
        if not np.isfinite(residual_norm):
            raise RuntimeError(f"non-finite residual at step {step_index}")
        if residual_norm > tolerance:
            raise RuntimeError(
                f"corrector failed at step {step_index}: residual={residual_norm}"
            )
        tangent = null_tangent(differentiate(rotation), previous=tangent)
        rotations.append(np.asarray(rotation))
        residual_norms.append(residual_norm)

        section = float(_section_coordinate(initial, rotation, initial_tangent))
        gap = float(rotation_distance(initial, rotation))
        crossed = previous_section != 0.0 and previous_section * section <= 0.0
        if crossed:
            section_crossings += 1
        if step_index >= min_steps:
            minimum_gap = min(minimum_gap, gap)
        if (
            step_index >= min_steps
            and crossed
            and gap <= closure_search_radius
            and float(jnp.dot(tangent, initial_tangent)) > 0.0
        ):
            detected_gap = gap

            def closing_system(delta):
                candidate = rotation @ exp(delta)
                return jnp.concatenate(
                    (
                        residual_function(candidate),
                        jnp.atleast_1d(
                            _section_coordinate(initial, candidate, initial_tangent)
                        ),
                    )
                )

            delta = jnp.zeros(3, dtype=rotation.dtype)
            for _ in range(8):
                value = closing_system(delta)
                if float(jnp.linalg.norm(value)) <= tolerance:
                    break
                jacobian = jax.jacfwd(closing_system)(delta)
                delta = delta + jnp.linalg.solve(jacobian, -value)
            rotation = rotation @ exp(delta)
            rotations.append(np.asarray(rotation))
            residual_norms.append(float(jnp.linalg.norm(evaluate(rotation))))
            return TraceResult(
                rotations=np.stack(rotations),
                residual_norms=np.asarray(residual_norms),
                closure_error=float(rotation_distance(initial, rotation)),
                length=float(step_index * step_size + detected_gap),
                steps=step_index,
                detected_gap=detected_gap,
            )
        previous_section = section

    raise RuntimeError(
        "fiber did not close within "
        f"{max_steps} steps: min_gap={minimum_gap}, "
        f"section_crossings={section_crossings}"
    )
