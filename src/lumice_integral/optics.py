"""Independent differentiable optics for fixed smooth ray paths."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Mapping, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

if TYPE_CHECKING:
    from .continuation import FiberProblem


FACE_3_NORMAL = jnp.array([1.0, 0.0, 0.0], dtype=jnp.float64)
FACE_5_NORMAL = jnp.array(
    [-0.5, jnp.sqrt(jnp.asarray(3.0, dtype=jnp.float64)) / 2.0, 0.0]
)
ICE_REFRACTIVE_INDEX = jnp.asarray(1.31, dtype=jnp.float64)


def _require_float64_reference_input(name: str, value: Array) -> None:
    if np.asarray(value).dtype != np.float64:
        raise ValueError(f"{name} must use float64 for the reference solver")


def _require_positive_finite_scalar(name: str, value: Array) -> float:
    """Validate host-side scalar inputs before 3-5 feasibility arithmetic."""
    value_array = np.asarray(value)
    if value_array.shape != ():
        raise ValueError(f"{name} must be a finite positive scalar")
    try:
        scalar = float(value_array)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite positive scalar") from error
    if not np.isfinite(scalar) or scalar <= 0.0:
        raise ValueError(f"{name} must be a finite positive scalar")
    return scalar


class Refraction(NamedTuple):
    direction: Array
    discriminant: Array
    incidence_cosine: Array


class PathEvaluation(NamedTuple):
    direction: Array
    entry: Refraction
    exit: Refraction


@dataclass(frozen=True)
class PathDomainCheck:
    """Host-side feasibility and event evidence for the fixed 3-5 branch."""

    valid: bool
    margins: Mapping[str, float] = field(default_factory=dict)
    event_kind: str | None = None
    event_margin: float = float("inf")
    message: str = ""


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


def path_3_5_domain(
    rotation: Array,
    incident_direction: Array,
    refractive_index: Array = ICE_REFRACTIVE_INDEX,
) -> PathDomainCheck:
    """Check the 3-5 branch on the host before either square root is taken."""
    rotation_array = np.asarray(rotation, dtype=np.float64)
    incident = np.asarray(incident_direction, dtype=np.float64)
    index = _require_positive_finite_scalar("refractive_index", refractive_index)
    entry_normal = rotation_array @ np.asarray(FACE_3_NORMAL)
    exit_normal = rotation_array @ np.asarray(FACE_5_NORMAL)
    entry_index = 1.0 / index
    entry_cosine = -float(np.dot(entry_normal, incident))
    entry_discriminant = 1.0 - entry_index**2 * (1.0 - entry_cosine**2)
    margins = {
        "entry_incidence_cosine": entry_cosine,
        "entry_snell_discriminant": entry_discriminant,
    }
    if not all(np.isfinite(value) for value in margins.values()):
        return PathDomainCheck(
            False,
            margins,
            "non_finite",
            float("nan"),
            "entry feasibility calculation is non-finite",
        )
    if entry_cosine <= 0.0:
        return PathDomainCheck(
            False,
            margins,
            "path_infeasible",
            entry_cosine,
            "incident ray does not enter through face 3",
        )
    if entry_discriminant <= 0.0:
        return PathDomainCheck(
            False,
            margins,
            "tir_boundary",
            entry_discriminant,
            "entry Snell discriminant is non-positive",
        )

    entry_direction = entry_index * incident + (
        entry_index * entry_cosine - np.sqrt(entry_discriminant)
    ) * entry_normal
    exit_cosine = float(np.dot(exit_normal, entry_direction))
    exit_discriminant = 1.0 - index**2 * (1.0 - exit_cosine**2)
    margins = {
        **margins,
        "exit_incidence_cosine": exit_cosine,
        "exit_snell_discriminant": exit_discriminant,
    }
    if not all(np.isfinite(value) for value in margins.values()):
        return PathDomainCheck(
            False,
            margins,
            "non_finite",
            float("nan"),
            "exit feasibility calculation is non-finite",
        )
    if exit_cosine <= 0.0:
        return PathDomainCheck(
            False,
            margins,
            "path_infeasible",
            exit_cosine,
            "internal ray does not leave through face 5",
        )
    if exit_discriminant <= 0.0:
        return PathDomainCheck(
            False,
            margins,
            "tir_boundary",
            exit_discriminant,
            "exit Snell discriminant is non-positive",
        )
    return PathDomainCheck(True, margins)


DOMAIN_MARGIN_NAMES = (
    "entry_incidence_cosine",
    "entry_snell_discriminant",
    "exit_incidence_cosine",
    "exit_snell_discriminant",
)


class BatchDomainCheck(NamedTuple):
    """Host-side 3-5 feasibility of many rotations at once (see :func:`path_3_5_domain_batch`)."""

    valid: np.ndarray
    margins: Mapping[str, np.ndarray]
    direction: np.ndarray


def path_3_5_domain_batch(
    rotations: Array,
    incident_direction: Array,
    refractive_index: Array = ICE_REFRACTIVE_INDEX,
) -> BatchDomainCheck:
    """Vectorised 3-5 feasibility: ``valid`` iff all four margins are positive.

    The single authority for the batch form of the four smooth-branch gates
    that :func:`path_3_5_domain` applies one pose at a time
    (``entry_incidence_cosine > 0``, ``entry_snell_discriminant > 0``,
    ``exit_incidence_cosine > 0``, ``exit_snell_discriminant > 0``); the
    prescan table and component discovery both go through here so the two
    forms cannot drift apart.  Margins are the cosines and Snell
    discriminants :func:`path_3_5` evaluates on its way through both faces,
    read straight off its :class:`PathEvaluation` (no second derivation); a
    non-finite margin compares ``False`` and therefore invalidates the pose,
    matching the scalar ``non_finite`` verdict.  ``direction`` is the outgoing
    direction of every pose, meaningful only where ``valid`` holds.  Runs as
    one eager ``jax.vmap`` over ``rotations`` of shape ``(N, 3, 3)``.
    """
    rotation_array = jnp.asarray(rotations, dtype=jnp.float64)
    if rotation_array.ndim != 3 or rotation_array.shape[1:] != (3, 3):
        raise ValueError("rotations must have shape (N, 3, 3)")
    incident = jnp.asarray(np.asarray(incident_direction, dtype=np.float64))
    index = jnp.asarray(_require_positive_finite_scalar("refractive_index", refractive_index))
    evaluation = jax.vmap(lambda r: path_3_5(r, incident, index))(rotation_array)
    margins = {
        "entry_incidence_cosine": np.asarray(evaluation.entry.incidence_cosine),
        "entry_snell_discriminant": np.asarray(evaluation.entry.discriminant),
        "exit_incidence_cosine": np.asarray(evaluation.exit.incidence_cosine),
        "exit_snell_discriminant": np.asarray(evaluation.exit.discriminant),
    }
    valid = np.ones(rotation_array.shape[0], dtype=bool)
    for name in DOMAIN_MARGIN_NAMES:
        valid &= margins[name] > 0
    return BatchDomainCheck(valid, margins, np.asarray(evaluation.direction))


def fresnel_unpolarized_transmittance(
    n1: float, cos_i: float, n2: float, cos_t: float
) -> float:
    """Unpolarized (s/p averaged) power transmittance of one planar interface.

    ``n1``/``cos_i`` describe the incident side, ``n2``/``cos_t`` the
    transmitted side; the cosines are the ones ``refract_smooth`` reports
    (``incidence_cosine`` and ``sqrt(discriminant)``).  For normal incidence
    this reduces to ``1 - ((n1 - n2) / (n1 + n2))**2``.  The caller must be on
    a non-TIR branch (``cos_t`` real); no clamping is performed here.
    """
    r_s = (n1 * cos_i - n2 * cos_t) / (n1 * cos_i + n2 * cos_t)
    r_p = (n2 * cos_i - n1 * cos_t) / (n2 * cos_i + n1 * cos_t)
    return 1.0 - 0.5 * (r_s * r_s + r_p * r_p)


def fresnel_transmission_3_5(
    rotation: Array,
    incident_direction: Array,
    refractive_index: Array = ICE_REFRACTIVE_INDEX,
) -> float:
    """Product of the face-3 entry and face-5 exit unpolarized transmittances.

    Reuses the host-side cosines and Snell discriminants that
    :func:`path_3_5_domain` already evaluates (``cos_t = sqrt(discriminant)``)
    instead of re-deriving the refraction.  Outside the smooth 3-5 domain
    (TIR, back-face entry or exit) the path transmits no power and ``0.0`` is
    returned; :func:`path_3_5_domain` remains the place to read *why*.
    """
    check = path_3_5_domain(rotation, incident_direction, refractive_index)
    if not check.valid:
        return 0.0
    index = float(np.asarray(refractive_index))
    margins = check.margins
    entry = fresnel_unpolarized_transmittance(
        1.0,
        margins["entry_incidence_cosine"],
        index,
        float(np.sqrt(margins["entry_snell_discriminant"])),
    )
    exit = fresnel_unpolarized_transmittance(
        index,
        margins["exit_incidence_cosine"],
        1.0,
        float(np.sqrt(margins["exit_snell_discriminant"])),
    )
    return float(entry * exit)


def fresnel_transmission_3_5_batch(
    rotations: Array,
    incident_direction: Array,
    refractive_index: Array = ICE_REFRACTIVE_INDEX,
) -> np.ndarray:
    """Batch form of :func:`fresnel_transmission_3_5` over ``(N, 3, 3)`` rotations.

    Reads the cosines and Snell discriminants off :func:`path_3_5_domain_batch`
    (the batch authority of the smooth-domain gates) and applies the same
    :func:`fresnel_unpolarized_transmittance` arithmetic elementwise; invalid
    poses get ``0.0`` and their (possibly negative) discriminants are never
    square-rooted, so no ``RuntimeWarning``/NaN leaks into valid entries.
    """
    check = path_3_5_domain_batch(rotations, incident_direction, refractive_index)
    index = float(np.asarray(refractive_index))
    valid = check.valid
    margins = check.margins
    entry_cosine = np.where(valid, margins["entry_incidence_cosine"], 1.0)
    exit_cosine = np.where(valid, margins["exit_incidence_cosine"], 1.0)
    entry_transmitted = np.sqrt(np.where(valid, margins["entry_snell_discriminant"], 1.0))
    exit_transmitted = np.sqrt(np.where(valid, margins["exit_snell_discriminant"], 1.0))
    entry = fresnel_unpolarized_transmittance(1.0, entry_cosine, index, entry_transmitted)
    exit = fresnel_unpolarized_transmittance(index, exit_cosine, 1.0, exit_transmitted)
    return np.where(valid, entry * exit, 0.0).astype(np.float64)


def path_3_5_problem(
    seed: Array,
    incident_direction: Array,
    *,
    target_direction: Array | None = None,
    refractive_index: Array = ICE_REFRACTIVE_INDEX,
    target_minimum_dot: float = 0.0,
) -> FiberProblem:
    """Adapt the smooth 3-5 branch and its host event gate to continuation."""
    from .analytic import tangent_basis
    from .continuation import (
        DomainEvaluation,
        EventCandidate,
        FiberProblem,
        TargetChart,
        TerminationReason,
    )

    _require_float64_reference_input("seed", seed)
    _require_float64_reference_input("incident_direction", incident_direction)
    _require_float64_reference_input("refractive_index", refractive_index)
    _require_positive_finite_scalar("refractive_index", refractive_index)
    if target_direction is not None:
        _require_float64_reference_input("target_direction", target_direction)
    seed = jnp.asarray(seed)
    incident_direction = jnp.asarray(incident_direction)
    refractive_index = jnp.asarray(refractive_index)

    def direction_evaluator(rotation: Array) -> Array:
        return path_3_5(rotation, incident_direction, refractive_index).direction

    def domain_evaluator(rotation: Array) -> DomainEvaluation:
        check = path_3_5_domain(rotation, incident_direction, refractive_index)
        event = (
            EventCandidate(
                TerminationReason(check.event_kind),
                check.event_margin,
                check.message,
                check.margins,
            )
            if check.event_kind is not None
            else None
        )
        return DomainEvaluation(check.valid, check.margins, event)

    seed_domain = path_3_5_domain(seed, incident_direction, refractive_index)
    if target_direction is None:
        if not seed_domain.valid:
            raise ValueError(
                "target_direction is required when the 3-5 seed is outside "
                f"the smooth domain: {seed_domain.event_kind}"
            )
        target_direction = direction_evaluator(seed)
    target_direction = jnp.asarray(target_direction)
    return FiberProblem(
        path=f"3-5:n={float(refractive_index):.8g}",
        incident_direction=incident_direction,
        target_chart=TargetChart(
            target_direction,
            tangent_basis(target_direction),
            minimum_dot=target_minimum_dot,
        ),
        direction_evaluator=direction_evaluator,
        domain_and_event_evaluator=domain_evaluator,
        seed=seed,
    )


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
