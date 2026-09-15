"""Independent differentiable optics for fixed smooth ray paths."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array


FACE_3_NORMAL = jnp.array([1.0, 0.0, 0.0], dtype=jnp.float64)
FACE_5_NORMAL = jnp.array(
    [-0.5, jnp.sqrt(jnp.asarray(3.0, dtype=jnp.float64)) / 2.0, 0.0]
)
ICE_REFRACTIVE_INDEX = jnp.asarray(1.31, dtype=jnp.float64)


class Refraction(NamedTuple):
    direction: Array
    discriminant: Array
    incidence_cosine: Array


class PathEvaluation(NamedTuple):
    direction: Array
    entry: Refraction
    exit: Refraction


def refract_smooth(
    direction: Array, normal_toward_incident: Array, relative_index: Array
) -> Refraction:
    """Evaluate Snell refraction on a caller-validated non-TIR branch."""
    incidence_cosine = -jnp.dot(normal_toward_incident, direction)
    discriminant = 1.0 - relative_index**2 * (1.0 - incidence_cosine**2)
    transmitted = relative_index * direction + (
        relative_index * incidence_cosine - jnp.sqrt(discriminant)
    ) * normal_toward_incident
    return Refraction(transmitted, discriminant, incidence_cosine)


def path_3_5(
    rotation: Array,
    incident_direction: Array,
    refractive_index: Array = ICE_REFRACTIVE_INDEX,
) -> PathEvaluation:
    """Trace refraction through hexagonal-prism side faces 3 then 5."""
    entry_normal = rotation @ FACE_3_NORMAL.astype(rotation.dtype)
    exit_normal = rotation @ FACE_5_NORMAL.astype(rotation.dtype)
    refractive_index = jnp.asarray(refractive_index, dtype=rotation.dtype)
    entry = refract_smooth(
        incident_direction, entry_normal, 1.0 / refractive_index
    )
    exit = refract_smooth(entry.direction, -exit_normal, refractive_index)
    return PathEvaluation(exit.direction, entry, exit)


def minimum_deviation_incident(
    refractive_index: Array = ICE_REFRACTIVE_INDEX,
) -> Array:
    """Return the in-plane incident ray for a symmetric 60-degree prism path."""
    refractive_index = jnp.asarray(refractive_index)
    internal_angle = jnp.pi / 6.0
    external_angle = jnp.arcsin(refractive_index * jnp.sin(internal_angle))
    return jnp.array(
        [-jnp.cos(external_angle), jnp.sin(external_angle), 0.0],
        dtype=refractive_index.dtype,
    )
