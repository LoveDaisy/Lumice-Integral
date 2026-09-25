"""Independent differentiable optics for fixed smooth ray paths of the hexagonal prism.

A ray path is a face sequence ``(entry, *internal_reflections, exit)``.  The
smooth branch refracts in through the entry face, reflects off each internal
face and refracts out through the exit face.  Every interface splits the
power (Fresnel, unpolarized): the path keeps the transmitted part at entry
and exit and the reflected part at each internal face, which is total
(``R = 1``) under TIR and partial otherwise.  A partial internal reflection
is a weight (:func:`fresnel_transmission_path`), not a domain boundary; only
the entry/exit Snell discriminants are TIR boundaries.  ``path_direction`` /
``path_domain`` / ``path_domain_batch`` / ``fresnel_transmission_path`` /
``path_problem`` take the face sequence explicitly; the ``path_3_5*`` names
are the same functions at ``faces == PATH_3_5_FACES`` (thin wrappers, not a
second implementation), kept because the ch06 fixtures and the strip
pipeline are pinned to them bit for bit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Mapping, NamedTuple, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

if TYPE_CHECKING:
    from .continuation import FiberProblem


_HALF_SQRT_3 = jnp.sqrt(jnp.asarray(3.0, dtype=jnp.float64)) / 2.0
# Body-frame outward normals of the hexagonal prism, face numbering of
# ``geometry.core`` (basal 1 = +c, 2 = -c; side face 3+i at azimuth i*60 deg).
# Written with the exact rationals +-1/2 and +-sqrt(3)/2 rather than cos/sin so
# that the 3-5 evaluations stay bit-identical to the pre-generalisation
# constants; ``tests/test_geometry_core.py`` pins every entry against
# ``HexPrism().normal`` (the geometry package's Newell normals) to 1e-15.
HEXPRISM_BODY_NORMALS: Mapping[int, Array] = {
    1: jnp.array([0.0, 0.0, 1.0], dtype=jnp.float64),
    2: jnp.array([0.0, 0.0, -1.0], dtype=jnp.float64),
    3: jnp.array([1.0, 0.0, 0.0], dtype=jnp.float64),
    4: jnp.array([0.5, _HALF_SQRT_3, 0.0]),
    5: jnp.array([-0.5, _HALF_SQRT_3, 0.0]),
    6: jnp.array([-1.0, 0.0, 0.0], dtype=jnp.float64),
    7: jnp.array([-0.5, -_HALF_SQRT_3, 0.0]),
    8: jnp.array([0.5, -_HALF_SQRT_3, 0.0]),
}
FACE_3_NORMAL = HEXPRISM_BODY_NORMALS[3]
FACE_5_NORMAL = HEXPRISM_BODY_NORMALS[5]
ICE_REFRACTIVE_INDEX = jnp.asarray(1.31, dtype=jnp.float64)
PATH_3_5_FACES = (3, 5)


def normalize_faces(faces: Sequence[int]) -> tuple[int, ...]:
    """``(entry, *internal_reflections, exit)`` as a tuple of known hexagonal-prism face numbers."""
    normalized = tuple(int(face) for face in faces)
    if len(normalized) < 2:
        raise ValueError("faces must be (entry, *reflections, exit) with at least two faces")
    unknown = [face for face in normalized if face not in HEXPRISM_BODY_NORMALS]
    if unknown:
        raise ValueError(f"unknown hexagonal-prism face numbers {unknown}; expected 1-8")
    return normalized


def path_id_of(faces: Sequence[int]) -> str:
    """Face sequence -> path id (``(3, 1, 2, 5) -> "3-1-2-5"``); inverse of :func:`faces_of_path_id`."""
    return "-".join(str(face) for face in normalize_faces(faces))


def faces_of_path_id(path_id: str) -> tuple[int, ...]:
    """Path id -> face sequence (``"3-1-2-5" -> (3, 1, 2, 5)``); the single parser of the id format."""
    try:
        parts = tuple(int(part) for part in str(path_id).split("-"))
    except ValueError as error:
        raise ValueError(f"malformed path_id {path_id!r}; expected dash-joined face numbers like '3-5'") from error
    return normalize_faces(parts)


def problem_path_label(faces: Sequence[int], refractive_index: float) -> str:
    """``FiberProblem.path`` of a fixed face sequence at one index (``"3-5:n=1.31"``)."""
    return f"{path_id_of(faces)}:n={float(refractive_index):.8g}"


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


class InternalReflection(NamedTuple):
    """One internal face reflection (total or partial).

    ``incidence_cosine = d . N`` (``N`` the world outward normal, ``d`` the
    internal ray) must be positive for the ray to reach the face from inside;
    ``tir_discriminant = n^2 (1 - cos^2) - 1`` is positive where the
    reflection is total.  It is a diagnostic and the input of the face's
    reflectance, not a domain gate: where it is non-positive the reflection
    is partial and ``cos_t = sqrt(-tir_discriminant)`` is the escaping ray's
    cosine (:func:`internal_reflectance`).
    """

    direction: Array
    tir_discriminant: Array
    incidence_cosine: Array


class PathEvaluation(NamedTuple):
    direction: Array
    entry: Refraction
    exit: Refraction
    internal: tuple[InternalReflection, ...] = ()


@dataclass(frozen=True)
class PathDomainCheck:
    """Host-side feasibility and event evidence for one fixed smooth branch."""

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


def reflect_internal(
    direction: Array, outward_normal: Array, refractive_index: Array
) -> InternalReflection:
    """Mirror ``direction`` off an internal face (total or partial reflection alike)."""
    incidence_cosine = jnp.dot(outward_normal, direction)
    tir_discriminant = refractive_index**2 * (1.0 - incidence_cosine**2) - 1.0
    reflected = direction - 2.0 * incidence_cosine * outward_normal
    return InternalReflection(reflected, tir_discriminant, incidence_cosine)


def path_direction(
    rotation: Array,
    faces: Sequence[int],
    incident_direction: Array,
    refractive_index: Array = ICE_REFRACTIVE_INDEX,
) -> PathEvaluation:
    """Trace the smooth branch of ``faces`` through the rotated prism (module docstring).

    World face normals are ``rotation @ body_normal``; the entry refraction
    uses the outward normal (it points toward the incident medium), the exit
    refraction its negative, and each internal reflection the outward normal
    with the internal ray hitting it from inside.  ``faces`` is static (a
    Python tuple); the JAX trace unrolls one step per face.
    """
    faces = normalize_faces(faces)
    dtype = rotation.dtype
    refractive_index = jnp.asarray(refractive_index, dtype=dtype)
    entry_normal = rotation @ HEXPRISM_BODY_NORMALS[faces[0]].astype(dtype)
    entry = refract_smooth(incident_direction, entry_normal, 1.0 / refractive_index)
    direction = entry.direction
    internal: list[InternalReflection] = []
    for face in faces[1:-1]:
        reflection = reflect_internal(
            direction, rotation @ HEXPRISM_BODY_NORMALS[face].astype(dtype), refractive_index
        )
        internal.append(reflection)
        direction = reflection.direction
    exit_normal = rotation @ HEXPRISM_BODY_NORMALS[faces[-1]].astype(dtype)
    exit = refract_smooth(direction, -exit_normal, refractive_index)
    return PathEvaluation(exit.direction, entry, exit, tuple(internal))


def path_3_5(
    rotation: Array,
    incident_direction: Array,
    refractive_index: Array = ICE_REFRACTIVE_INDEX,
) -> PathEvaluation:
    """Trace refraction through hexagonal-prism side faces 3 then 5 (:func:`path_direction`)."""
    return path_direction(rotation, PATH_3_5_FACES, incident_direction, refractive_index)


def _internal_margin_names(step: int) -> tuple[str, str]:
    return (f"internal_{step}_incidence_cosine", f"internal_{step}_tir_discriminant")


def domain_margin_names(faces: Sequence[int]) -> tuple[str, ...]:
    """Margin names of :func:`path_domain` for ``faces``, in evaluation order.

    ``entry_*`` (incidence cosine, Snell discriminant), then
    ``internal_{k}_*`` (incidence cosine, TIR discriminant) for the k-th
    internal reflection, then ``exit_*``.  Which of them gate the smooth
    branch is :func:`validity_margin_names`; the internal TIR discriminants
    are diagnostics only.  For ``PATH_3_5_FACES`` this is :data:`DOMAIN_MARGIN_NAMES`.
    """
    faces = normalize_faces(faces)
    names: list[str] = ["entry_incidence_cosine", "entry_snell_discriminant"]
    for step in range(1, len(faces) - 1):
        names.extend(_internal_margin_names(step))
    names.extend(("exit_incidence_cosine", "exit_snell_discriminant"))
    return tuple(names)


def validity_margin_names(faces: Sequence[int]) -> tuple[str, ...]:
    """The margins of :func:`domain_margin_names` that must be positive on the smooth branch.

    Listed explicitly rather than filtered by name: the entry incidence
    cosine and Snell discriminant, each internal reflection's incidence
    cosine (the ray reaches the face from inside), the exit incidence cosine
    and Snell discriminant.  ``internal_{k}_tir_discriminant`` is left out:
    a non-total internal reflection lowers the path's power
    (:func:`fresnel_transmission_path`) but keeps the pose in the domain.
    """
    faces = normalize_faces(faces)
    names: list[str] = ["entry_incidence_cosine", "entry_snell_discriminant"]
    for step in range(1, len(faces) - 1):
        names.append(_internal_margin_names(step)[0])
    names.extend(("exit_incidence_cosine", "exit_snell_discriminant"))
    return tuple(names)


def path_domain(
    rotation: Array,
    faces: Sequence[int],
    incident_direction: Array,
    refractive_index: Array = ICE_REFRACTIVE_INDEX,
) -> PathDomainCheck:
    """Check the smooth branch of ``faces`` on the host before any square root is taken.

    Gates in ray order (:func:`validity_margin_names`), stopping at the
    first failure with the margins evaluated so far: the entry incidence
    cosine and Snell discriminant, each internal reflection's incidence
    cosine (``path_infeasible`` when the internal ray does not reach the
    face), then the exit incidence cosine and Snell discriminant.  Each
    internal reflection's TIR discriminant is recorded in ``margins`` as a
    diagnostic but gates nothing: a partial reflection keeps the branch.  So
    ``valid`` does not imply full power; the path's power factor is
    :func:`fresnel_transmission_path`.  Event kinds are the two existing
    :class:`.continuation.TerminationReason` values; which face the event
    belongs to is in the margin name and the message.
    """
    faces = normalize_faces(faces)
    rotation_array = np.asarray(rotation, dtype=np.float64)
    incident = np.asarray(incident_direction, dtype=np.float64)
    index = _require_positive_finite_scalar("refractive_index", refractive_index)
    entry_normal = rotation_array @ np.asarray(HEXPRISM_BODY_NORMALS[faces[0]])
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
            f"incident ray does not enter through face {faces[0]}",
        )
    if entry_discriminant <= 0.0:
        return PathDomainCheck(
            False,
            margins,
            "tir_boundary",
            entry_discriminant,
            "entry Snell discriminant is non-positive",
        )

    direction = entry_index * incident + (
        entry_index * entry_cosine - np.sqrt(entry_discriminant)
    ) * entry_normal
    for step, face in enumerate(faces[1:-1], start=1):
        normal = rotation_array @ np.asarray(HEXPRISM_BODY_NORMALS[face])
        cosine = float(np.dot(normal, direction))
        discriminant = index**2 * (1.0 - cosine**2) - 1.0
        cosine_name, discriminant_name = _internal_margin_names(step)
        margins = {**margins, cosine_name: cosine, discriminant_name: discriminant}
        if not (np.isfinite(cosine) and np.isfinite(discriminant)):
            return PathDomainCheck(
                False,
                margins,
                "non_finite",
                float("nan"),
                f"internal reflection {step} feasibility calculation is non-finite",
            )
        if cosine <= 0.0:
            return PathDomainCheck(
                False,
                margins,
                "path_infeasible",
                cosine,
                f"internal ray does not reach face {face} from inside (reflection {step})",
            )
        direction = direction - 2.0 * cosine * normal

    exit_normal = rotation_array @ np.asarray(HEXPRISM_BODY_NORMALS[faces[-1]])
    exit_cosine = float(np.dot(exit_normal, direction))
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
            f"internal ray does not leave through face {faces[-1]}",
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


def path_3_5_domain(
    rotation: Array,
    incident_direction: Array,
    refractive_index: Array = ICE_REFRACTIVE_INDEX,
) -> PathDomainCheck:
    """Check the 3-5 branch on the host before either square root is taken (:func:`path_domain`)."""
    return path_domain(rotation, PATH_3_5_FACES, incident_direction, refractive_index)


# The four margins of the 3-5 branch (``domain_margin_names(PATH_3_5_FACES)``).
DOMAIN_MARGIN_NAMES = domain_margin_names(PATH_3_5_FACES)


class BatchDomainCheck(NamedTuple):
    """Host-side feasibility of many rotations at once (see :func:`path_domain_batch`)."""

    valid: np.ndarray
    margins: Mapping[str, np.ndarray]
    direction: np.ndarray


def path_domain_batch(
    rotations: Array,
    faces: Sequence[int],
    incident_direction: Array,
    refractive_index: Array = ICE_REFRACTIVE_INDEX,
) -> BatchDomainCheck:
    """Vectorised feasibility of ``faces``: ``valid`` iff every validity margin is positive.

    The single authority for the batch form of the smooth-branch gates that
    :func:`path_domain` applies one pose at a time (``margins`` carries every
    name of :func:`domain_margin_names`, ``valid`` tests those of
    :func:`validity_margin_names`); the S^2 event store and component discovery
    both go through here so the two forms cannot drift apart.  Margins are
    the cosines and discriminants :func:`path_direction` evaluates on its way
    through the faces, read straight off its :class:`PathEvaluation` (no
    second derivation); a non-finite margin compares ``False`` and therefore
    invalidates the pose, matching the scalar ``non_finite`` verdict.
    ``direction`` is the outgoing direction of every pose, meaningful only
    where ``valid`` holds.  Runs as one eager ``jax.vmap`` over ``rotations``
    of shape ``(N, 3, 3)``.
    """
    faces = normalize_faces(faces)
    rotation_array = jnp.asarray(rotations, dtype=jnp.float64)
    if rotation_array.ndim != 3 or rotation_array.shape[1:] != (3, 3):
        raise ValueError("rotations must have shape (N, 3, 3)")
    incident = jnp.asarray(np.asarray(incident_direction, dtype=np.float64))
    index = jnp.asarray(_require_positive_finite_scalar("refractive_index", refractive_index))
    evaluation = jax.vmap(lambda r: path_direction(r, faces, incident, index))(rotation_array)
    margins = {
        "entry_incidence_cosine": np.asarray(evaluation.entry.incidence_cosine),
        "entry_snell_discriminant": np.asarray(evaluation.entry.discriminant),
    }
    for step, reflection in enumerate(evaluation.internal, start=1):
        cosine_name, discriminant_name = _internal_margin_names(step)
        margins[cosine_name] = np.asarray(reflection.incidence_cosine)
        margins[discriminant_name] = np.asarray(reflection.tir_discriminant)
    margins["exit_incidence_cosine"] = np.asarray(evaluation.exit.incidence_cosine)
    margins["exit_snell_discriminant"] = np.asarray(evaluation.exit.discriminant)
    valid = np.ones(rotation_array.shape[0], dtype=bool)
    for name in validity_margin_names(faces):
        valid &= margins[name] > 0
    return BatchDomainCheck(valid, margins, np.asarray(evaluation.direction))


def path_3_5_domain_batch(
    rotations: Array,
    incident_direction: Array,
    refractive_index: Array = ICE_REFRACTIVE_INDEX,
) -> BatchDomainCheck:
    """Vectorised 3-5 feasibility: ``valid`` iff all four margins are positive (:func:`path_domain_batch`)."""
    return path_domain_batch(rotations, PATH_3_5_FACES, incident_direction, refractive_index)


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


def internal_reflectance(refractive_index: float, incidence_cosine, tir_discriminant):
    """Unpolarized power reflectance of one internal face, ``1`` under TIR.

    ``incidence_cosine``/``tir_discriminant`` are the face's
    :class:`InternalReflection` margins.  Where ``tir_discriminant > 0`` the
    reflection is total; otherwise the escaping ray's cosine is
    ``sqrt(-tir_discriminant)`` and ``R = 1 - T(n -> 1)``.  This is
    Lumice's ``GetReflectRatio(d, n)`` with ``d = -tir_discriminant / cos^2``
    (the same per-interface s/p average, ``docs/conventions.md`` #18).
    Scalars or arrays; the total branch never takes a square root.
    """
    total = tir_discriminant > 0
    cos_i = np.where(total, 1.0, incidence_cosine)
    cos_t = np.sqrt(np.where(total, 1.0, -tir_discriminant))
    partial = 1.0 - fresnel_unpolarized_transmittance(refractive_index, cos_i, 1.0, cos_t)
    return np.where(total, 1.0, partial)


def fresnel_transmission_path(
    rotation: Array,
    faces: Sequence[int],
    incident_direction: Array,
    refractive_index: Array = ICE_REFRACTIVE_INDEX,
) -> float:
    """Power factor of ``faces``: entry ``T`` x each internal ``R_k`` x exit ``T``.

    Unpolarized (s/p averaged) per interface; ``R_k = 1`` where the internal
    reflection is total (:func:`internal_reflectance`), so a path whose
    reflections are all total gets the entry and exit transmittances only.
    The name is kept on purpose: this is the path's power transmission, and
    the internal factors are what it always should have contained.  Reuses
    the host-side cosines and discriminants that :func:`path_domain` already
    evaluates (``cos_t = sqrt(discriminant)``) instead of re-deriving the
    refraction.  Outside the smooth domain (entry/exit TIR, back-face entry,
    exit or internal face) the path transmits no power and ``0.0`` is
    returned; :func:`path_domain` remains the place to read *why*.
    """
    check = path_domain(rotation, faces, incident_direction, refractive_index)
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
    power = entry * exit
    for step in range(1, len(normalize_faces(faces)) - 1):
        cosine_name, discriminant_name = _internal_margin_names(step)
        power = power * float(internal_reflectance(index, margins[cosine_name], margins[discriminant_name]))
    return float(power)


def fresnel_transmission_3_5(
    rotation: Array,
    incident_direction: Array,
    refractive_index: Array = ICE_REFRACTIVE_INDEX,
) -> float:
    """Face-3 entry times face-5 exit unpolarized transmittance (:func:`fresnel_transmission_path`)."""
    return fresnel_transmission_path(rotation, PATH_3_5_FACES, incident_direction, refractive_index)


def fresnel_transmission_path_batch(
    rotations: Array,
    faces: Sequence[int],
    incident_direction: Array,
    refractive_index: Array = ICE_REFRACTIVE_INDEX,
) -> np.ndarray:
    """Batch form of :func:`fresnel_transmission_path` over ``(N, 3, 3)`` rotations.

    Reads the cosines and discriminants off :func:`path_domain_batch` (the
    batch authority of the smooth-domain gates) and applies the same
    :func:`fresnel_unpolarized_transmittance` / :func:`internal_reflectance`
    arithmetic elementwise; invalid poses get ``0.0`` and their (possibly
    negative) discriminants are never square-rooted, so no
    ``RuntimeWarning``/NaN leaks into valid entries.
    """
    check = path_domain_batch(rotations, faces, incident_direction, refractive_index)
    index = float(np.asarray(refractive_index))
    valid = check.valid
    margins = check.margins
    entry_cosine = np.where(valid, margins["entry_incidence_cosine"], 1.0)
    exit_cosine = np.where(valid, margins["exit_incidence_cosine"], 1.0)
    entry_transmitted = np.sqrt(np.where(valid, margins["entry_snell_discriminant"], 1.0))
    exit_transmitted = np.sqrt(np.where(valid, margins["exit_snell_discriminant"], 1.0))
    entry = fresnel_unpolarized_transmittance(1.0, entry_cosine, index, entry_transmitted)
    exit = fresnel_unpolarized_transmittance(index, exit_cosine, 1.0, exit_transmitted)
    power = entry * exit
    for step in range(1, len(normalize_faces(faces)) - 1):
        cosine_name, discriminant_name = _internal_margin_names(step)
        cosine = np.where(valid, margins[cosine_name], 1.0)
        discriminant = np.where(valid, margins[discriminant_name], 1.0)
        power = power * internal_reflectance(index, cosine, discriminant)
    return np.where(valid, power, 0.0).astype(np.float64)


def fresnel_transmission_3_5_batch(
    rotations: Array,
    incident_direction: Array,
    refractive_index: Array = ICE_REFRACTIVE_INDEX,
) -> np.ndarray:
    """Batch form of :func:`fresnel_transmission_3_5` (:func:`fresnel_transmission_path_batch`)."""
    return fresnel_transmission_path_batch(rotations, PATH_3_5_FACES, incident_direction, refractive_index)


# A fiber meets a Snell boundary tangentially: the outgoing direction is fixed
# along it, so ``exit_snell_discriminant = (d . n_exit)^2`` falls quadratically
# in arclength, and the direction map's square root leaves the corrector
# unable to converge in the last ~1e-10 of margin (A60-10 arcs ended in
# ``step_underflow`` at random, task phase1-partial-reflection-domain).
# :func:`path_problem` therefore puts the ``tir_boundary`` event at this
# margin; the arclength cut off is ~sqrt(tolerance / curvature), ~1e-4 rad,
# and is in the quadrature's endpoint truncation estimate.
SNELL_EVENT_TOLERANCE = 1e-8
SNELL_MARGIN_NAMES = ("entry_snell_discriminant", "exit_snell_discriminant")


def path_problem(
    seed: Array,
    faces: Sequence[int],
    incident_direction: Array,
    *,
    target_direction: Array | None = None,
    refractive_index: Array = ICE_REFRACTIVE_INDEX,
    target_minimum_dot: float = 0.0,
) -> FiberProblem:
    """Adapt the smooth branch of ``faces`` and its host event gate to continuation.

    The domain evaluator is :func:`path_domain` with two adaptations for
    continuation: its margins are the event margins
    (:func:`validity_margin_names`, no internal TIR discriminant) and a
    Snell discriminant at or below :data:`SNELL_EVENT_TOLERANCE` is already
    the ``tir_boundary`` event.
    ``FiberProblem.path`` is :func:`problem_path_label` (``"3-1-2-5:n=1.31"``);
    the evaluator closures are fresh per call, so batch callers share one
    problem per face sequence through :func:`.discovery.retarget_problem`.
    """
    from .analytic import tangent_basis
    from .continuation import (
        DomainEvaluation,
        EventCandidate,
        FiberProblem,
        TargetChart,
        TerminationReason,
    )

    faces = normalize_faces(faces)
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
        return path_direction(rotation, faces, incident_direction, refractive_index).direction

    # Continuation steers by its margins (event approach step limit, arc-end
    # truncation estimate), so it gets the event margins only: the internal
    # TIR discriminants gate nothing (a partial reflection keeps the branch)
    # and stay in the event details as diagnostics.
    event_margin_names = validity_margin_names(faces)

    def domain_evaluator(rotation: Array) -> DomainEvaluation:
        check = path_domain(rotation, faces, incident_direction, refractive_index)
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
        margins = {name: check.margins[name] for name in event_margin_names if name in check.margins}
        if check.valid:
            name = min(SNELL_MARGIN_NAMES, key=lambda key: margins[key])
            if margins[name] <= SNELL_EVENT_TOLERANCE:
                return DomainEvaluation(
                    False,
                    margins,
                    EventCandidate(
                        TerminationReason.TIR_BOUNDARY,
                        margins[name],
                        f"{name} within SNELL_EVENT_TOLERANCE of the critical angle",
                        check.margins,
                    ),
                )
        return DomainEvaluation(check.valid, margins, event)

    seed_domain = path_domain(seed, faces, incident_direction, refractive_index)
    if target_direction is None:
        if not seed_domain.valid:
            raise ValueError(
                f"target_direction is required when the {path_id_of(faces)} seed is outside "
                f"the smooth domain: {seed_domain.event_kind}"
            )
        target_direction = direction_evaluator(seed)
    target_direction = jnp.asarray(target_direction)
    return FiberProblem(
        path=problem_path_label(faces, float(refractive_index)),
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


def path_3_5_problem(
    seed: Array,
    incident_direction: Array,
    *,
    target_direction: Array | None = None,
    refractive_index: Array = ICE_REFRACTIVE_INDEX,
    target_minimum_dot: float = 0.0,
) -> FiberProblem:
    """Adapt the smooth 3-5 branch and its host event gate to continuation (:func:`path_problem`)."""
    return path_problem(
        seed,
        PATH_3_5_FACES,
        incident_direction,
        target_direction=target_direction,
        refractive_index=refractive_index,
        target_minimum_dot=target_minimum_dot,
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
