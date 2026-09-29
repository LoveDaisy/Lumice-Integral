"""LI -> Lumice parity fixtures of the analytic module A v0 (``docs/analytic-parity-fixtures.md``).

Three fixture kinds, one JSON file each, for Lumice's ``liblumice_analytic``
(``doc/analytic-api.md`` section 4) to replay without reading LI source:

- ``evaluate_path``: one crystal, one path, one pose: validity, outgoing
  direction, body-frame segment directions, per-interface transmittances and
  the total Fresnel factor (:func:`.optics.trace_path`,
  :func:`.optics.path_domain`, :func:`.optics.fresnel_transmission_path`);
- ``trace_fiber``: :func:`.continuation.trace_fiber` from a seed on the
  fiber, in both tangent orientations unless the first closes, compared as
  curves, never step by step;
- ``seed_search``: :func:`.discovery.discover_components` on the band it
  carries (contract ``docs/phase1-math-contract.md`` section 9.5), compared
  by funnel counts, counters and components.

Each fixture carries its LI rev, the SHA-256 of ``docs/conventions.md``,
``symmetry_semantics = "none"`` (a concrete face sequence, no reduction), the
crystal as the Lumice closed-form scalars of ``LUMICE_ANALYTIC_Crystal``, its
inputs, its expected output and a tolerance with its basis for every
compared quantity.  The points are chosen on the sphere of ``u`` (the sun in
the crystal frame) by :func:`choose_point` per matrix cell; the pose follows
from ``u`` (:func:`pose_of_u`) and the target from the pose, so every seed
lies on its fiber to rounding.  This module only reads the solvers; it adds
no numerical behaviour to them.
"""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import asdict, dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any, Mapping, Sequence

import jax.numpy as jnp
import numpy as np

from .continuation import ContinuationOptions, FiberResult, FiberStatus, local_residual_jacobian, trace_fiber
from .discovery import ComponentDiscoveryResult, discover_components, distance_to_curve, retarget_problem
from .geometry import HexPrism, Polyhedron, Pyramid, entry_measure
from .geometry.pyramid import miller_indices_to_c_over_a
from .optics import (
    face_normals,
    fresnel_transmission_path,
    fresnel_unpolarized_transmittance,
    internal_reflectance,
    normalize_faces,
    path_direction,
    path_domain,
    path_domain_batch,
    path_id_of,
    path_problem,
    trace_path,
    validity_margin_names,
)
from .provenance import git_commit, sha256_of
from .s2_store import S2EventStore, StoreSeeds, build_event_store, event_rotations
from .so3 import rotation_distances

FORMAT = "lumice-integral/analytic-parity"
SCHEMA_VERSION = 1
# Module A takes one concrete face sequence and performs no symmetry reduction
# (Lumice doc/analytic-api.md section 3.3 rule 2; docs/conventions.md #21).
SYMMETRY_SEMANTICS = "none"
CATEGORIES = ("random", "critical", "near_boundary")

REPO_ROOT = Path(__file__).resolve().parents[2]
CONVENTIONS_PATH = REPO_ROOT / "docs" / "conventions.md"

# The target half-plane: every pose puts its outgoing direction in the vertical half-plane through the
# propagation direction s, above it (the plane of test_discovery.target_at_deviation).
AZIMUTH_REFERENCE = np.array([0.0, 0.0, 1.0])

# Tolerances and their basis (docs/analytic-parity-fixtures.md section 5 is the table of the same values).
KINEMATIC_ATOL = 1e-12
KINEMATIC_BASIS = (
    "float64 evaluation of the same closed-form refraction/reflection chain from bit-identical JSON inputs: a few "
    "ulp per interface (1e-12 leaves ~1e3 ulp of room), times 1/(2 sqrt(d)) for the smallest Snell discriminant d "
    "of the pose, the derivative of the refracted cosine sqrt(d) (kinematic_atol)"
)


def kinematic_atol(margins: Mapping[str, float]) -> float:
    """:data:`KINEMATIC_ATOL` widened by ``1/(2 sqrt(d))`` at the smallest Snell discriminant ``d`` below 1/4."""
    snell = [value for name, value in margins.items() if name.endswith("snell_discriminant") and value > 0.0]
    smallest = min(snell, default=1.0)
    return KINEMATIC_ATOL * max(1.0, 0.5 / math.sqrt(smallest))


JACOBIAN_BASIS = (
    "J_perp = sigma_1 sigma_2 of the 2 x 3 right-trivialised residual Jacobian (contract section 5.4) is one "
    "derivative above the directions, so the direction tolerance's Snell amplification 1/(2 sqrt(d)) enters squared: "
    "|error| <= 1e-12 max(1, 1/(4 d)) max(1, |value|) per value (jacobian_rtol); measured: LI's AD value agrees with "
    "an independent central difference (h = 1e-6) to 6e-9 relative, and a 1e-14 rad pose nudge moves it by at most "
    "2.5e-12 relative away from Snell boundaries (4e-8 at d ~ 1e-8)"
)


def jacobian_rtol(margins: Mapping[str, float]) -> float:
    """Relative tolerance of ``J_perp`` and the singular values: :func:`kinematic_atol`'s amplification, squared."""
    snell = [value for name, value in margins.items() if name.endswith("snell_discriminant") and value > 0.0]
    smallest = min(snell, default=1.0)
    return KINEMATIC_ATOL * max(1.0, 0.25 / smallest)


CURVE_DISTANCE_TOL = 0.012
CURVE_DISTANCE_BASIS = (
    "docs/phase1-math-contract.md section 10.1: bidirectional sampled-pose set distance below 0.012 rad between "
    "two step-controller settings on the 3-5 loop; measured here against the geodesic polyline "
    "(curve_distance), which removes the chord-sampling part of that distance"
)
CURVE_DENSIFY_SPACING = 1e-3
ARCLENGTH_RTOL = 2e-3
ARCLENGTH_BASIS = (
    "docs/phase1-math-contract.md section 10.1: discrete lengths of one 3-5 loop span 0.0017 on 0.9645 "
    "(1.8e-3 relative) across initial steps 0.03-0.08; an arc's length is compared only from the same seed "
    "(section 9.5.6)"
)


# ------------------------------------------------------------------ crystal
def prism_crystal(height: float, face_distance: Sequence[float] = (1.0,) * 6) -> dict[str, Any]:
    """``LUMICE_ANALYTIC_Crystal`` of a Lumice prism (``HexPrism.from_lumice``); unused fields are zero."""
    return {
        "kind": "prism",
        "height": float(height),
        "face_distance": [float(f) for f in face_distance],
        "upper_h": 0.0,
        "lower_h": 0.0,
        "upper_wedge_deg": 0.0,
        "lower_wedge_deg": 0.0,
    }


def miller_wedge_deg(indices: Sequence[int]) -> float:
    """The Lumice wedge angle (degrees) of Miller indices ``(h, 0, l)``, the inverse of ``wedge_angle_to_c_over_a``."""
    c_over_a = miller_indices_to_c_over_a(indices)
    if c_over_a is None:
        raise ValueError(f"Miller indices {tuple(indices)!r} have no cone")
    return float(np.degrees(np.arctan(np.sqrt(3.0) / 2.0 / c_over_a)))


def pyramid_crystal(
    prism_h: float,
    upper_h: float,
    lower_h: float,
    upper_wedge_deg: float,
    lower_wedge_deg: float,
    face_distance: Sequence[float] = (1.0,) * 6,
) -> dict[str, Any]:
    """``LUMICE_ANALYTIC_Crystal`` of a Lumice pyramid (``Pyramid.from_lumice`` with wedge angles)."""
    return {
        "kind": "pyramid",
        "height": float(prism_h),
        "face_distance": [float(f) for f in face_distance],
        "upper_h": float(upper_h),
        "lower_h": float(lower_h),
        "upper_wedge_deg": float(upper_wedge_deg),
        "lower_wedge_deg": float(lower_wedge_deg),
    }


def build_crystal(spec: Mapping[str, Any]) -> Polyhedron:
    """The LI crystal of a fixture's ``crystal`` block (hexagon edge ``a = 1``, LI's scale)."""
    face_distance = tuple(float(f) for f in spec["face_distance"])
    if spec["kind"] == "prism":
        return HexPrism.from_lumice(spec["height"], face_distance)
    if spec["kind"] == "pyramid":
        return Pyramid.from_lumice(
            spec["height"],
            spec["upper_h"],
            spec["lower_h"],
            face_distance=face_distance,
            upper_wedge_deg=spec["upper_wedge_deg"],
            lower_wedge_deg=spec["lower_wedge_deg"],
        )
    raise ValueError(f"unknown crystal kind {spec['kind']!r}")


# ------------------------------------------------------------------ JSON
def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value)
    if isinstance(value, str) or value is None:
        return str(value) if value is not None else None
    raise TypeError(f"not JSON-serialisable: {type(value).__name__}")


def dumps(document: Mapping[str, Any]) -> str:
    """Canonical text: sorted keys, shortest round-trip float repr (bit-exact float64), no NaN, trailing newline."""
    return json.dumps(_jsonable(document), indent=1, sort_keys=True, allow_nan=False) + "\n"


def write_json(path: Path, document: Mapping[str, Any]) -> None:
    Path(path).write_text(dumps(document))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def fixture_provenance() -> dict[str, Any]:
    """LI rev, whether the tracked tree matched it, and the SHA-256 of the conventions table."""
    try:
        clean = subprocess.run(["git", "diff", "--quiet", "HEAD"], cwd=REPO_ROOT, check=False).returncode == 0
    except OSError:
        clean = False
    return {
        "li_rev": git_commit(REPO_ROOT) or "unknown",
        "li_tracked_tree_clean": clean,
        "conventions_sha256": sha256_of(CONVENTIONS_PATH),
    }


def _header(kind: str, provenance: Mapping[str, Any], cell: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "format": FORMAT,
        "schema_version": SCHEMA_VERSION,
        "fixture_kind": kind,
        "symmetry_semantics": SYMMETRY_SEMANTICS,
        "provenance": dict(provenance),
        "cell": dict(cell),
    }


def _tolerance(value: float, basis: str) -> dict[str, Any]:
    return {"value": float(value), "basis": basis}


# ------------------------------------------------------------------ scene and points on S^2
@dataclass(frozen=True)
class Scene:
    """One crystal, one concrete path, one index and one sun: everything but the pose."""

    crystal_spec: Mapping[str, Any]
    faces: tuple[int, ...]
    refractive_index: float
    sun_direction: tuple[float, float, float]

    @cached_property
    def crystal(self) -> Polyhedron:
        return build_crystal(self.crystal_spec)

    @property
    def path_id(self) -> str:
        return path_id_of(self.faces, self.crystal)

    @property
    def sun(self) -> np.ndarray:
        return np.asarray(self.sun_direction, dtype=np.float64)

    @property
    def incident_direction(self) -> np.ndarray:
        """``s = -s_hat``, the propagation direction (the API's ``incident_direction``)."""
        return -self.sun


def body_outgoing(scene: Scene, u: np.ndarray) -> tuple[np.ndarray, float]:
    """``phi = Phi_P(-u)`` (body-frame outgoing propagation) and ``D = angle(phi, -u)`` for a valid ``u``."""
    normals = face_normals(scene.crystal, scene.faces)
    phi = np.asarray(
        trace_path(jnp.eye(3), normals, jnp.asarray(-u), jnp.asarray(scene.refractive_index)).direction
    )
    return phi, float(np.arccos(np.clip(phi @ -u, -1.0, 1.0)))


def domain_at_u(scene: Scene, u: np.ndarray):
    """:func:`.optics.path_domain` at the identity pose with the sun at ``u`` (depends on ``u`` only)."""
    return path_domain(np.eye(3), scene.faces, -u, scene.refractive_index, crystal=scene.crystal)


def minimum_validity_margin(margins: Mapping[str, float], faces: Sequence[int]) -> tuple[str, float]:
    """The smallest of the validity margins present (``path_domain`` stops at the first failing one)."""
    present = [(name, float(margins[name])) for name in validity_margin_names(faces) if name in margins]
    return min(present, key=lambda item: item[1])


def admissible_u(scene: Scene, u: np.ndarray) -> bool:
    """Discovery's gates at ``u``: the path domain is valid and the finite crystal's entry measure is positive."""
    if not domain_at_u(scene, u).valid:
        return False
    return entry_measure(np.eye(3), scene.faces, -u, scene.crystal, n_ice=scene.refractive_index).value > 0.0


def pose_of_u(scene: Scene, u: np.ndarray) -> np.ndarray:
    """The pose with ``R u = s_hat`` whose outgoing direction lies in the :data:`AZIMUTH_REFERENCE` half-plane.

    :func:`.s2_store.event_rotations` of the single event ``(u, phi, D)``: the
    pose discovery gives a store event (contract section 9.5.3).
    """
    phi, deviation = body_outgoing(scene, u)
    return event_rotations(u[None], phi[None], np.array([deviation]), scene.sun, AZIMUTH_REFERENCE)[0]


def _unit(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector)


def _geodesic(u: np.ndarray, tangent: np.ndarray, angle: float) -> np.ndarray:
    return _unit(np.cos(angle) * u + np.sin(angle) * tangent)


def _fixed_tangent(u: np.ndarray) -> np.ndarray:
    """A tangent at ``u`` fixed by ``u`` alone (``u x z``, or ``u x x`` near the poles)."""
    axis = np.array([0.0, 0.0, 1.0]) if abs(u[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    return _unit(np.cross(u, axis))


def _bisect(predicate, inside: float, outside: float, iterations: int = 80) -> tuple[float, float]:
    """Shrink ``[inside, outside]`` (``predicate(inside)`` true, ``predicate(outside)`` false) around the switch."""
    for _ in range(iterations):
        middle = 0.5 * (inside + outside)
        if middle in (inside, outside):
            break
        if predicate(middle):
            inside = middle
        else:
            outside = middle
    return inside, outside


class PointUnavailable(Exception):
    """A matrix cell whose category has no real geometric point on this path (recorded, not faked)."""


def random_u(scene: Scene, rng_seed: int, max_draws: int = 100_000) -> tuple[np.ndarray, int]:
    """The first admissible direction of ``numpy.random.default_rng(rng_seed)`` normals (uniform on ``S^2``)."""
    rng = np.random.default_rng(rng_seed)
    for draw in range(1, max_draws + 1):
        u = _unit(rng.standard_normal(3))
        if admissible_u(scene, u):
            return u, draw
    raise PointUnavailable(f"no admissible u in {max_draws} draws for {scene.path_id}")


def critical_u(scene: Scene, offset_deg: float, step_deg: float = 0.05, max_deg: float = 30.0) -> tuple[np.ndarray, dict]:
    """A point of the level set ``D = D* +- offset`` next to an interior extremum ``D*`` of ``D_P``.

    The extremum is the first ``minimum``/``maximum`` of
    :attr:`.dp_field.DPField.interior_critical_points`; the level is on its lit
    side (above a minimum, below a maximum), reached along the geodesic from
    the extremum in the direction :func:`_fixed_tangent` and bisected.
    """
    from .dp_field import DPField  # the field layer is only needed for this category

    field_ = DPField.build(scene.crystal, scene.faces, scene.refractive_index)
    extrema = [point for point in field_.interior_critical_points if point.kind in ("minimum", "maximum")]
    if not extrema:
        kinds = sorted({point.kind for point in field_.interior_critical_points})
        raise PointUnavailable(
            f"D_P of {scene.path_id} has no interior extremum on this crystal (interior critical points: "
            f"{kinds or 'none'}; critical values are boundary ones only: DPField.interior_critical_points)"
        )
    extremum = extrema[0]
    centre = _unit(np.asarray(extremum.position, dtype=np.float64))
    sign = 1.0 if extremum.kind == "minimum" else -1.0
    level = extremum.value + sign * math.radians(offset_deg)
    tangent = _fixed_tangent(centre)

    def below_level(angle: float) -> bool:
        u = _geodesic(centre, tangent, angle)
        return domain_at_u(scene, u).valid and sign * (body_outgoing(scene, u)[1] - level) < 0.0

    angle = 0.0
    while below_level(angle + math.radians(step_deg)):
        angle += math.radians(step_deg)
        if angle > math.radians(max_deg):
            raise PointUnavailable(f"level {math.degrees(level):.4f} deg not reached within {max_deg} deg of the extremum")
    inside, _ = _bisect(below_level, angle, angle + math.radians(step_deg))
    u = _geodesic(centre, tangent, inside)
    if not admissible_u(scene, u):
        raise PointUnavailable("the level-set point next to the extremum is not admissible (entry measure 0)")
    return u, {
        "extremum_kind": extremum.kind,
        "extremum_deviation_deg": math.degrees(extremum.value),
        "extremum_u": centre,
        "level_offset_deg": sign * offset_deg,
        "geodesic_angle_deg": math.degrees(inside),
    }


def near_boundary_u(
    scene: Scene, start: np.ndarray, margin: float, step_deg: float = 0.5
) -> tuple[np.ndarray, np.ndarray, dict]:
    """A valid ``u`` whose smallest validity margin is ``margin``, next to the edge of ``U_P``, and its mirror outside.

    From ``start`` (valid) along the geodesic in the direction
    :func:`_fixed_tangent`: march to the first invalid step, bisect the edge
    ``theta_b``, then bisect the smallest validity margin down to
    ``margin`` between the last valid step and ``theta_b``; the outside point
    is the mirror image across ``theta_b``.
    """
    tangent = _fixed_tangent(start)

    def valid(angle: float) -> bool:
        return domain_at_u(scene, _geodesic(start, tangent, angle)).valid

    def margin_above(angle: float) -> bool:
        check = domain_at_u(scene, _geodesic(start, tangent, angle))
        return check.valid and minimum_validity_margin(check.margins, scene.faces)[1] > margin

    angle = 0.0
    step = math.radians(step_deg)
    while valid(angle + step):
        angle += step
        if angle > math.pi:
            raise PointUnavailable("the geodesic never leaves U_P")
    edge, _ = _bisect(valid, angle, angle + step)
    if not margin_above(angle):
        raise PointUnavailable("the last valid step is already inside the requested margin")
    _, inner = _bisect(margin_above, angle, edge)
    u_in = _geodesic(start, tangent, inner)
    u_out = _geodesic(start, tangent, 2.0 * edge - inner)
    name, value = minimum_validity_margin(domain_at_u(scene, u_in).margins, scene.faces)
    return u_in, u_out, {
        "start_u": start,
        "edge_geodesic_angle_deg": math.degrees(edge),
        "inside_geodesic_angle_deg": math.degrees(inner),
        "nearest_margin": name,
        "nearest_margin_value": value,
        "outside_failure": domain_at_u(scene, u_out).message,
    }


# ------------------------------------------------------------------ matrix cells
@dataclass(frozen=True)
class Cell:
    """One row of the fixture matrix: a scene, a point category and how the point is chosen."""

    scene: Scene
    category: str
    rationale: str
    rng_seed: int = 20260928
    critical_offset_deg: float = 0.5
    boundary_margin: float = 1e-3
    store_n: int | None = 100_000
    seed_search_skip: str | None = None
    band_half_width_deg: float = 0.2
    cluster_radius_rad: float = 0.3
    distance_threshold: float = ContinuationOptions.closure_distance

    def __post_init__(self) -> None:
        if self.category not in CATEGORIES:
            raise ValueError(f"category must be one of {CATEGORIES}")
        if (self.store_n is None) == (self.seed_search_skip is None):
            raise ValueError("give exactly one of store_n and seed_search_skip")

    @property
    def name(self) -> str:
        return f"{self.scene.path_id}__{self.category}"


@dataclass(frozen=True)
class PointChoice:
    u: np.ndarray
    pose: np.ndarray
    target: np.ndarray
    selection: dict[str, Any]
    outside_pose: np.ndarray | None = None


def choose_point(cell: Cell) -> PointChoice:
    """The cell's ``u`` (module docstring), its pose and the target it produces; :class:`PointUnavailable` if none."""
    scene = cell.scene
    outside = None
    if cell.category == "random":
        u, draws = random_u(scene, cell.rng_seed)
        selection = {"method": "first admissible uniform draw", "rng": "numpy.random.default_rng", "rng_seed": cell.rng_seed, "draws": draws}
    elif cell.category == "critical":
        u, selection = critical_u(scene, cell.critical_offset_deg)
        selection = {"method": "lit side of an interior extremum of D_P", **selection}
    else:
        start, draws = random_u(scene, cell.rng_seed)
        u, u_out, selection = near_boundary_u(scene, start, cell.boundary_margin)
        selection = {
            "method": "bisected to a validity margin from the random cell's start",
            "rng_seed": cell.rng_seed,
            "draws": draws,
            "target_margin": cell.boundary_margin,
            **selection,
        }
        outside = pose_of_u(scene, u_out)
    pose = pose_of_u(scene, u)
    target = np.asarray(path_direction(pose, scene.faces, scene.incident_direction, scene.refractive_index, crystal=scene.crystal).direction)
    phi, deviation = body_outgoing(scene, u)
    selection = {**selection, "u": u, "deviation_deg": math.degrees(deviation)}
    return PointChoice(u, pose, target, selection, outside)


def _scene_input(scene: Scene) -> dict[str, Any]:
    return {
        "crystal": dict(scene.crystal_spec),
        "faces": list(scene.faces),
        "refractive_index": scene.refractive_index,
        "incident_direction": scene.incident_direction,
    }


def _scene_of(fixture_input: Mapping[str, Any]) -> Scene:
    incident = np.asarray(fixture_input["incident_direction"], dtype=np.float64)
    return Scene(
        fixture_input["crystal"],
        tuple(int(f) for f in fixture_input["faces"]),
        float(fixture_input["refractive_index"]),
        tuple(float(x) for x in -incident),
    )


# ------------------------------------------------------------------ EvaluatePath
def failed_gate(margins: Mapping[str, float], faces: Sequence[int]) -> dict[str, Any] | None:
    """The first validity gate (in :func:`.optics.validity_margin_names` order) whose margin is not positive."""
    for name in validity_margin_names(faces):
        if name in margins and not margins[name] > 0.0:
            return {"name": name, "value": float(margins[name])}
    return None


def pointwise_observables(scene: Scene, pose: np.ndarray) -> dict[str, Any]:
    """Validity, branch margins and the normal Jacobian at ``pose``: what a trace reports per accepted pose.

    ``branch_margins`` are the validity margins (:func:`.optics.validity_margin_names`, the event margins
    of :func:`.optics.path_problem`, contract section 9.3 ``branch_diagnostics``), present only where the
    pose is valid; an invalid pose reports its ``failed_gate`` instead.  ``normal_jacobian`` is
    ``J_perp = sigma_1 sigma_2`` of the right-trivialised residual Jacobian at the pose's own outgoing
    direction (contract section 5.4): it depends on the pose and the path only, not on a target, and it is
    unavailable (``None``, ``jacobian_available = False``) where the path is not valid.
    """
    faces = normalize_faces(scene.faces, scene.crystal)
    check = path_domain(pose, faces, scene.incident_direction, scene.refractive_index, crystal=scene.crystal)
    if not check.valid:
        return {
            "valid": False,
            "check": check,
            "branch_margins": None,
            "failed_gate": failed_gate(check.margins, faces),
            "jacobian_available": False,
            "normal_jacobian": None,
            "singular_values": None,
        }
    outgoing = np.asarray(path_direction(pose, faces, scene.incident_direction, scene.refractive_index, crystal=scene.crystal).direction)
    problem = retarget_problem(_scene_problem(scene), outgoing, pose)
    singular_values = np.linalg.svd(np.asarray(local_residual_jacobian(problem, jnp.asarray(pose, dtype=jnp.float64))), compute_uv=False)
    return {
        "valid": True,
        "check": check,
        "branch_margins": {name: float(check.margins[name]) for name in validity_margin_names(faces)},
        "failed_gate": None,
        "jacobian_available": True,
        "normal_jacobian": float(singular_values[0] * singular_values[1]),
        "singular_values": singular_values,
    }


_PROBLEM_CACHE: dict[tuple, Any] = {}


def _scene_problem(scene: Scene):
    """One :func:`.optics.path_problem` per scene, retargeted per pose: its evaluator closures key the JIT cache."""
    key = (json.dumps(_jsonable(scene.crystal_spec), sort_keys=True), scene.faces, scene.refractive_index, scene.sun_direction)
    if key not in _PROBLEM_CACHE:
        _PROBLEM_CACHE[key] = path_problem(
            jnp.eye(3, dtype=jnp.float64),
            scene.faces,
            jnp.asarray(scene.incident_direction),
            target_direction=jnp.asarray(scene.sun, dtype=jnp.float64),
            refractive_index=jnp.asarray(scene.refractive_index, dtype=jnp.float64),
            crystal=scene.crystal,
        )
    return _PROBLEM_CACHE[key]


POINTWISE_FIELDS = ("branch_margins", "failed_gate", "jacobian_available", "normal_jacobian", "singular_values")


def evaluate_path(scene: Scene, pose: np.ndarray) -> dict[str, Any]:
    """The ``LUMICE_ANALYTIC_PathEvaluation`` fields of ``pose`` (outgoing etc. only where ``valid``).

    Wave 2 adds :data:`POINTWISE_FIELDS` (:func:`pointwise_observables`); the v0 fields are unchanged.
    """
    faces = normalize_faces(scene.faces, scene.crystal)
    observables = pointwise_observables(scene, pose)
    check = observables.pop("check")
    pointwise = {key: observables[key] for key in POINTWISE_FIELDS}
    name, value = minimum_validity_margin(check.margins, faces)
    diagnostics = {"validity_margins": dict(check.margins), "nearest_margin": name, "nearest_margin_value": value}
    if not check.valid:
        return {"valid": False, "fresnel_transmission": 0.0, "diagnostics": {**diagnostics, "message": check.message}, **pointwise}
    index = scene.refractive_index
    evaluation = path_direction(pose, faces, scene.incident_direction, index, crystal=scene.crystal)
    rotation = np.asarray(pose)
    world = [scene.incident_direction, np.asarray(evaluation.entry.direction)]
    world += [np.asarray(step.direction) for step in evaluation.internal]
    world.append(np.asarray(evaluation.exit.direction))
    margins = check.margins
    transmittances = [
        fresnel_unpolarized_transmittance(1.0, margins["entry_incidence_cosine"], index, math.sqrt(margins["entry_snell_discriminant"]))
    ]
    for step in evaluation.internal:
        transmittances.append(float(internal_reflectance(index, float(step.incidence_cosine), float(step.tir_discriminant))))
    transmittances.append(
        fresnel_unpolarized_transmittance(index, margins["exit_incidence_cosine"], 1.0, math.sqrt(margins["exit_snell_discriminant"]))
    )
    return {
        "valid": True,
        "outgoing_direction": np.asarray(evaluation.direction),
        "segment_directions": np.stack([rotation.T @ direction for direction in world]),
        "interface_transmittances": np.asarray(transmittances, dtype=np.float64),
        "fresnel_transmission": fresnel_transmission_path(pose, faces, scene.incident_direction, index, crystal=scene.crystal),
        "diagnostics": diagnostics,
        **pointwise,
    }


def build_evaluate_path_fixture(
    scene: Scene, pose: np.ndarray, cell: Mapping[str, Any], provenance: Mapping[str, Any]
) -> dict[str, Any]:
    fixture = _header("evaluate_path", provenance, cell)
    fixture["input"] = {**_scene_input(scene), "pose": np.asarray(pose).reshape(9)}
    expected = evaluate_path(scene, np.asarray(pose))
    diagnostics = expected["diagnostics"]
    atol = kinematic_atol(diagnostics["validity_margins"])
    fixture["expected"] = expected
    fixture["tolerance"] = {
        "valid": _tolerance(
            0.0,
            f"exact boolean; the nearest gate {diagnostics['nearest_margin']} has margin "
            f"{diagnostics['nearest_margin_value']:.3g}, far above the float64 rounding of a margin (~1e-15)",
        ),
        **{key: _tolerance(atol, KINEMATIC_BASIS) for key in ("outgoing_direction", "segment_directions", "interface_transmittances", "fresnel_transmission")},
        **pointwise_tolerances(diagnostics["validity_margins"]),
    }
    return fixture


def pointwise_tolerances(margins: Mapping[str, float]) -> dict[str, Any]:
    """Tolerances of :data:`POINTWISE_FIELDS` at a pose with these margins (``evaluate_path`` and trace poses)."""
    atol = kinematic_atol(margins)
    jacobian = _tolerance(jacobian_rtol(margins), "relative, " + JACOBIAN_BASIS)
    return {
        "branch_margins": _tolerance(
            atol, "absolute, per named margin, the same key set; the margins are cosines and discriminants of the chain: " + KINEMATIC_BASIS
        ),
        "failed_gate": _tolerance(atol, "the gate name exactly, its (non-positive) margin as branch_margins"),
        "jacobian_available": _tolerance(0.0, "exact boolean: available exactly where the path is valid (contract section 9.3)"),
        "normal_jacobian": jacobian,
        "singular_values": jacobian,
    }


# ------------------------------------------------------------------ curves
# Deliberately not ``jax.vmap(so3.log)``/``jax.vmap(so3.exp)``: curve lengths vary per
# fixture cell, and a vmap over a new leading dimension recompiles per distinct shape
# (the per-pixel XLA recompilation cost this codebase has already paid down elsewhere,
# see AGENTS.md's render_ch06_strip.py history). This host-numpy pair is the same
# closed-form Rodrigues map as ``so3.log``/``so3.exp``, restated batched and
# shape-polymorphic for that reason.
def _log_rotations(rotations: np.ndarray) -> np.ndarray:
    """Rotation vectors of ``(N, 3, 3)`` rotations below pi (host numpy; chords here are <= 0.12 rad)."""
    skew = 0.5 * np.stack(
        [rotations[:, 2, 1] - rotations[:, 1, 2], rotations[:, 0, 2] - rotations[:, 2, 0], rotations[:, 1, 0] - rotations[:, 0, 1]],
        axis=1,
    )
    sine = np.linalg.norm(skew, axis=1)
    cosine = np.clip((np.trace(rotations, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    angle = np.arctan2(sine, cosine)
    scale = np.where(sine > 1e-15, angle / np.where(sine > 1e-15, sine, 1.0), 1.0)
    return skew * scale[:, None]


def _exp_rotations(vectors: np.ndarray) -> np.ndarray:
    angle = np.linalg.norm(vectors, axis=1)
    safe = np.where(angle > 1e-15, angle, 1.0)
    a = np.where(angle > 1e-15, np.sin(angle) / safe, 1.0)
    b = np.where(angle > 1e-15, (1.0 - np.cos(angle)) / safe**2, 0.5)
    x, y, z = vectors.T
    zero = np.zeros_like(x)
    hat = np.stack([np.stack([zero, -z, y], 1), np.stack([z, zero, -x], 1), np.stack([-y, x, zero], 1)], 1)
    return np.eye(3) + a[:, None, None] * hat + b[:, None, None] * hat @ hat


def densify_curve(poses: np.ndarray, closed: bool, spacing: float = CURVE_DENSIFY_SPACING) -> np.ndarray:
    """The geodesic polyline through ``poses`` (``(N, 3, 3)``) sampled at most ``spacing`` apart."""
    poses = np.asarray(poses, dtype=np.float64).reshape(-1, 3, 3)
    if closed and len(poses) > 1:
        poses = np.concatenate((poses, poses[:1]))
    if len(poses) < 2:
        return poses
    steps = _log_rotations(np.einsum("nji,njk->nik", poses[:-1], poses[1:]))
    pieces = []
    for start, step in zip(poses[:-1], steps):
        count = max(1, math.ceil(np.linalg.norm(step) / spacing))
        fractions = np.arange(count) / count
        pieces.append(start @ _exp_rotations(fractions[:, None] * step[None]))
    pieces.append(poses[-1:])
    return np.concatenate(pieces)


def curve_distance(
    a: np.ndarray, a_closed: bool, b: np.ndarray, b_closed: bool, spacing: float = CURVE_DENSIFY_SPACING
) -> float:
    """Symmetric Hausdorff distance between two pose polylines (geodesic chords, sampled ``spacing`` apart)."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 3, 3)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 3, 3)
    if len(a) == 0 or len(b) == 0:  # a trace that accepted no pose (seed rejected) has no curve
        return 0.0 if len(a) == len(b) else math.inf
    dense_a = densify_curve(a, a_closed, spacing)
    dense_b = densify_curve(b, b_closed, spacing)
    a_to_b = max(float(np.min(rotation_distances(pose, dense_b))) for pose in np.asarray(a).reshape(-1, 3, 3))
    b_to_a = max(float(np.min(rotation_distances(pose, dense_a))) for pose in np.asarray(b).reshape(-1, 3, 3))
    return max(a_to_b, b_to_a)


# ------------------------------------------------------------------ TraceFiber
def continuation_options_dict(options: ContinuationOptions) -> dict[str, Any]:
    return asdict(options)


def _trace_record(result: FiberResult, sign: int, scene: Scene) -> dict[str, Any]:
    """One trace: the v0 point lists plus, per accepted pose, its branch margins and normal Jacobian (wave 2)."""
    poses = np.asarray(result.poses, dtype=np.float64)
    names = validity_margin_names(normalize_faces(scene.faces, scene.crystal))
    jacobians = result.jacobian_diagnostics
    margins = result.branch_diagnostics.accepted_margins
    if not len(jacobians) == len(margins) == len(poses):
        raise AssertionError(f"per-pose diagnostics {len(jacobians)} / {len(margins)} do not match {len(poses)} poses")
    return {
        "initial_tangent_sign": sign,
        "status": str(result.status),
        "reason": str(result.reason),
        "poses": poses.reshape(-1, 9),
        "crystal_frame_sun_directions": np.einsum("nji,j->ni", poses, -scene.incident_direction),
        "arclength_increments": np.asarray(result.arclength_increments, dtype=np.float64),
        "residual_norms": np.asarray(result.residual_norms, dtype=np.float64),
        "tangents": np.asarray(result.tangents, dtype=np.float64),
        "arclength": float(np.sum(result.arclength_increments)),
        "branch_margin_names": list(names),
        "branch_margins": np.asarray([[margin[name] for name in names] for margin in margins], dtype=np.float64).reshape(-1, len(names)),
        "jacobian_available": [bool(diagnostic.available) for diagnostic in jacobians],
        "normal_jacobian": np.asarray([diagnostic.normal_jacobian for diagnostic in jacobians], dtype=np.float64),
        "singular_values": np.asarray([diagnostic.singular_values for diagnostic in jacobians], dtype=np.float64).reshape(-1, 2),
    }


def run_traces(scene: Scene, seed: np.ndarray, target: np.ndarray, options: ContinuationOptions) -> list[dict[str, Any]]:
    """The forward trace, plus the backward one (sign reversed) unless the forward trace closed."""
    problem = path_problem(
        jnp.asarray(seed, dtype=jnp.float64),
        scene.faces,
        jnp.asarray(scene.incident_direction),
        target_direction=jnp.asarray(target, dtype=jnp.float64),
        refractive_index=jnp.asarray(scene.refractive_index, dtype=jnp.float64),
        crystal=scene.crystal,
    )
    forward = trace_fiber(problem, options)
    traces = [_trace_record(forward, options.initial_tangent_sign, scene)]
    if forward.status != FiberStatus.CLOSED:
        backward_options = ContinuationOptions(**{**asdict(options), "initial_tangent_sign": -options.initial_tangent_sign})
        backward = trace_fiber(problem, backward_options)
        traces.append(_trace_record(backward, backward_options.initial_tangent_sign, scene))
    return traces


def traced_curve(traces: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, bool]:
    """The curve the traces describe: a closed forward trace, or backward reversed then forward (section 9.5.5)."""
    forward = np.asarray(traces[0]["poses"], dtype=np.float64).reshape(-1, 3, 3)
    if traces[0]["status"] == FiberStatus.CLOSED:
        return forward, True
    if len(traces) == 1:
        return forward, False
    backward = np.asarray(traces[1]["poses"], dtype=np.float64).reshape(-1, 3, 3)
    return np.concatenate((backward[1:][::-1], forward)), False


def build_trace_fiber_fixture(
    scene: Scene,
    seed: np.ndarray,
    target: np.ndarray,
    cell: Mapping[str, Any],
    provenance: Mapping[str, Any],
    options: ContinuationOptions | None = None,
    reference_traces: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """A ``trace_fiber`` fixture; ``reference_traces`` (the same seed under unlimited options) for a budget fixture.

    When the traces end ``budget_exhausted`` the fixture carries ``expected.reference_curve`` (the curve of
    ``reference_traces``) and a ``budget_extent`` tolerance, and the curve and length are compared by
    containment and by the budget's own extent instead of by equality (:func:`_compare_budget_traces`).
    """
    options = options or ContinuationOptions()
    fixture = _header("trace_fiber", provenance, cell)
    fixture["input"] = {
        **_scene_input(scene),
        "target_direction": target,
        "seed_pose": np.asarray(seed).reshape(9),
        "continuation": continuation_options_dict(options),
    }
    traces = run_traces(scene, seed, target, options)
    fixture["expected"] = {"traces": traces}
    tau = options.residual_tolerance + options.relative_residual_tolerance
    fixture["tolerance"] = {
        "status_reason": _tolerance(
            0.0,
            "exact; with two traces the pair {reason of each} is compared unordered, because the seed tangent sign is "
            "deterministic per LAPACK build only (docs/phase1-math-contract.md section 10.1, seed orientation)",
        ),
        "curve_distance_rad": _tolerance(CURVE_DISTANCE_TOL, CURVE_DISTANCE_BASIS),
        "arclength_relative": _tolerance(ARCLENGTH_RTOL, ARCLENGTH_BASIS),
        "residual_norm_bound": _tolerance(tau, "every accepted pose: residual_tolerance + relative_residual_tolerance (section 10.1)"),
        **TRACE_POINTWISE_TOLERANCES,
    }
    if any(trace["status"] == FiberStatus.BUDGET_EXHAUSTED for trace in traces):
        if reference_traces is None:
            raise ValueError("a budget-exhausted trace_fiber fixture needs the unbudgeted reference traces")
        curve, closed = traced_curve(reference_traces)
        fixture["expected"]["reference_curve"] = {"poses": curve.reshape(-1, 9), "closed": closed}
        fixture["tolerance"]["budget_extent"] = _tolerance(0.0, BUDGET_EXTENT_BASIS)
    return fixture


TRACE_POINTWISE_TOLERANCES = {
    "pointwise_consistency": _tolerance(
        KINEMATIC_ATOL,
        "per trace and pose i: jacobian_available[i] is true, and normal_jacobian[i], singular_values[i] and "
        "branch_margins[i] equal the backend's own EvaluatePath at poses[i] within the evaluate_path tolerances at "
        "that pose (value 1e-12 is their base: kinematic_atol for margins, jacobian_rtol for J_perp); a per-sample "
        "comparison with LI is not made, because two backends step differently",
    ),
    "accepted_pose_regularity": _tolerance(
        0.0,
        "exact: at every accepted pose every branch margin is positive and normal_jacobian > 0 (contract sections "
        "6.1 and 5.4: an accepted pose is on the smooth branch and regular)",
    ),
}
BUDGET_EXTENT_BASIS = (
    "budget_exhausted traces (contract section 9.4; partial geometry, section 9.3): step_budget ends with exactly "
    "maximum_accepted_steps + 1 poses (the seed plus one pose per accepted step); arclength_budget ends with "
    "maximum_arclength - maximum_advance < arclength <= maximum_arclength (the next edge would have crossed the "
    "budget); evaluation_budget ends with at least one pose (what one evaluation unit counts is backend-internal, "
    "section 9.2); every pose lies within curve_distance_rad of expected.reference_curve, the same seed traced "
    "without the budget.  Curve equality and arclength equality are not compared: two step controllers reach "
    "different extents inside one budget"
)


# ------------------------------------------------------------------ seed search
_STORE_CACHE: dict[tuple, S2EventStore] = {}


def scene_store(scene: Scene, n: int) -> S2EventStore:
    """The antipodal-Fibonacci store of the scene's path at ``n`` points (in memory, cached per process)."""
    key = (json.dumps(_jsonable(scene.crystal_spec), sort_keys=True), scene.faces, scene.refractive_index, n)
    if key not in _STORE_CACHE:
        _STORE_CACHE[key] = build_event_store(scene.crystal, scene.refractive_index, [scene.faces], n, run_checks=False)
    return _STORE_CACHE[key]


@dataclass(frozen=True, eq=False)
class BandSeeds:
    """A :class:`.s2_store.StoreSeeds` stand-in over one exported band (``u``, ``phi``, ``D`` in pool order).

    ``discover_components`` reads ``faces``/``incident_direction``/
    ``refractive_index``/``crystal`` and ``candidates``; the candidates are
    :func:`.s2_store.event_rotations` of the band, exactly the store's
    construction, for the one target and half width the band was cut for.
    """

    scene: Scene
    u: np.ndarray
    phi: np.ndarray
    deviation: np.ndarray
    target: np.ndarray
    half_width_rad: float

    @property
    def faces(self) -> tuple[int, ...]:
        return self.scene.faces

    @property
    def incident_direction(self) -> np.ndarray:
        return self.scene.incident_direction

    @property
    def refractive_index(self) -> float:
        return self.scene.refractive_index

    @property
    def crystal(self) -> Polyhedron:
        return self.scene.crystal

    def candidates(self, target_direction: np.ndarray, half_width_rad: float) -> tuple[np.ndarray, np.ndarray]:
        if not (np.array_equal(target_direction, self.target) and half_width_rad == self.half_width_rad):
            raise ValueError("a BandSeeds serves only the target and half width its band was cut for")
        if len(self.deviation) == 0:
            return np.zeros((0, 3, 3)), np.zeros(0)
        delta = float(np.arccos(np.clip(self.target @ self.incident_direction, -1.0, 1.0)))
        rotations = event_rotations(self.u, self.phi, self.deviation, self.scene.sun, self.target)
        return rotations, np.abs(self.deviation - delta)


def _store_band(store: S2EventStore, scene: Scene, target: np.ndarray, half_width_rad: float):
    """The band ``|D_i - delta| <= b`` as :meth:`.s2_store.StoreSeeds.candidates` cuts it."""
    delta = float(np.arccos(np.clip(target @ scene.incident_direction, -1.0, 1.0)))
    band = store.band_slice(delta - half_width_rad, np.nextafter(delta + half_width_rad, np.inf))
    return (
        np.asarray(band.u, dtype=np.float64).reshape(-1, 3),
        np.asarray(band.phi, dtype=np.float64).reshape(-1, 3),
        np.asarray(band.D, dtype=np.float64).reshape(-1),
    )


def _component_curve(component) -> tuple[np.ndarray, bool]:
    return np.asarray(component.result.poses, dtype=np.float64), component.kind == "closed"


def discovery_record(result: ComponentDiscoveryResult) -> dict[str, Any]:
    """The v0 parity subset of a discovery result (contract section 9.5.8), curves as pose lists."""
    components = []
    for component in result.components:
        poses, _ = _component_curve(component)
        record = {
            "kind": component.kind,
            "seed": np.asarray(component.seed).reshape(9),
            "status": str(component.status),
            "reason": str(component.reason),
            "start_reason": None if component.start_reason is None else str(component.start_reason),
            "arclength": float(component.arclength),
            "curve_poses": poses.reshape(-1, 9),
        }
        if component.kind == "arc":
            record["seed_index"] = int(component.result.seed_index)
        components.append(record)
    return {
        "completeness": result.completeness,
        "pool_count": result.pool_count,
        "extra_seed_count": result.extra_seed_count,
        "raw_cluster_count": result.raw_cluster_count,
        "admissible_count": result.admissible_count,
        "events": dict(result.events),
        "components": components,
        "incomplete": [
            {"cause": item.cause, "seed": np.asarray(item.seed).reshape(9), "status": str(item.status), "reason": str(item.reason)}
            for item in result.incomplete
        ],
    }


def build_seed_search_fixture(
    cell: Cell, target: np.ndarray, cell_record: Mapping[str, Any], provenance: Mapping[str, Any]
) -> dict[str, Any]:
    scene = cell.scene
    options = ContinuationOptions()
    store = scene_store(scene, cell.store_n)
    seeds = StoreSeeds(store, scene.faces, scene.sun)
    half_width = math.radians(cell.band_half_width_deg)
    u, phi, deviation = _store_band(store, scene, target, half_width)
    rotations, _ = seeds.candidates(target, half_width)
    if not np.array_equal(rotations, BandSeeds(scene, u, phi, deviation, target, half_width).candidates(target, half_width)[0]):
        raise AssertionError("the exported band does not reproduce the store's candidate poses")
    result = discover_components(
        target,
        seeds,
        continuation=options,
        band_half_width_deg=cell.band_half_width_deg,
        cluster_radius_rad=cell.cluster_radius_rad,
        distance_threshold=cell.distance_threshold,
    )
    fixture = _header("seed_search", provenance, cell_record)
    fixture["input"] = {
        **_scene_input(scene),
        "target_direction": target,
        "sample": {
            "sampler": "antipodal Fibonacci lattice (docs/phase1-math-contract.md section 9.5.2), kept events w > 0",
            "n": cell.store_n,
            "band_u": u,
            "band_phi": phi,
            "band_deviation": deviation,
        },
        "extra_seeds": [],
        "band_half_width_deg": cell.band_half_width_deg,
        "cluster_radius_rad": cell.cluster_radius_rad,
        "distance_threshold": cell.distance_threshold,
        "continuation": continuation_options_dict(options),
    }
    fixture["expected"] = discovery_record(result)
    fixture["tolerance"] = {
        "counts_and_events": _tolerance(0.0, "exact: the pool is the exported band (section 9.5.8)"),
        "kind_status_reasons": _tolerance(0.0, "exact, per component in trace order (section 9.5.5); an arc's {reason, start_reason} unordered"),
        "seed_to_curve_rad": _tolerance(
            cell.distance_threshold,
            "the call's distance_threshold: the contract's own 'this pose is on that component' test (section 9.5.4 step 5)",
        ),
        "curve_distance_rad": _tolerance(CURVE_DISTANCE_TOL, CURVE_DISTANCE_BASIS),
        "closed_arclength_relative": _tolerance(ARCLENGTH_RTOL, ARCLENGTH_BASIS + "; arcs are not compared"),
    }
    return fixture


# ------------------------------------------------------------------ verification (read back, recompute, compare)
@dataclass
class Check:
    """Comparison outcomes of one fixture; ``failures`` is empty when every quantity is within tolerance."""

    fixture: str
    failures: list[str] = field(default_factory=list)
    compared: int = 0

    def expect(self, ok: bool, message: str) -> None:
        self.compared += 1
        if not ok:
            self.failures.append(message)


def _close(check: Check, name: str, got: Any, expected: Any, atol: float) -> None:
    got = np.asarray(got, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    error = float(np.max(np.abs(got - expected))) if got.shape == expected.shape and got.size else (0.0 if got.shape == expected.shape else math.inf)
    check.expect(error <= atol, f"{name}: max |error| {error:.3e} > {atol:.1e}")


def verify_evaluate_path(fixture: Mapping[str, Any], name: str = "") -> Check:
    check = Check(name)
    scene = _scene_of(fixture["input"])
    got = evaluate_path(scene, np.asarray(fixture["input"]["pose"], dtype=np.float64).reshape(3, 3))
    expected = fixture["expected"]
    tolerance = fixture["tolerance"]
    check.expect(got["valid"] == expected["valid"], f"valid: {got['valid']} != {expected['valid']}")
    if got["valid"] and expected["valid"]:
        for key in ("outgoing_direction", "segment_directions", "interface_transmittances", "fresnel_transmission"):
            _close(check, key, got[key], expected[key], tolerance[key]["value"])
    else:
        _close(check, "fresnel_transmission", got["fresnel_transmission"], expected["fresnel_transmission"], 0.0)
    if "jacobian_available" in expected:  # wave 2 fields; a v0 fixture has none of them
        _compare_pointwise(check, "", got, expected, tolerance)
    return check


def _close_relative(check: Check, name: str, got: Any, expected: Any, rtol: float) -> None:
    got = np.asarray(got, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    if got.shape != expected.shape:
        check.expect(False, f"{name}: shape {got.shape} != {expected.shape}")
        return
    error = float(np.max(np.abs(got - expected) / np.maximum(1.0, np.abs(expected)))) if got.size else 0.0
    check.expect(error <= rtol, f"{name}: max |error|/max(1, |value|) {error:.3e} > {rtol:.1e}")


def _compare_pointwise(check: Check, where: str, got: Mapping[str, Any], expected: Mapping[str, Any], tolerance: Mapping[str, Any]) -> None:
    """:data:`POINTWISE_FIELDS` of one pose: availability and gate names exact, values within their tolerances."""
    check.expect(
        got["jacobian_available"] == expected["jacobian_available"],
        f"{where}jacobian_available: {got['jacobian_available']} != {expected['jacobian_available']}",
    )
    if got["jacobian_available"] and expected["jacobian_available"]:
        rtol = tolerance["normal_jacobian"]["value"]
        _close_relative(check, f"{where}normal_jacobian", got["normal_jacobian"], expected["normal_jacobian"], rtol)
        _close_relative(check, f"{where}singular_values", got["singular_values"], expected["singular_values"], tolerance["singular_values"]["value"])
    else:
        check.expect(
            expected["normal_jacobian"] is None and expected["singular_values"] is None,
            f"{where}normal_jacobian present although jacobian_available is false",
        )
    got_margins, expected_margins = got["branch_margins"], expected["branch_margins"]
    if got_margins is None or expected_margins is None:
        check.expect(got_margins is None and expected_margins is None, f"{where}branch_margins: one side is unavailable")
    else:
        check.expect(sorted(got_margins) == sorted(expected_margins), f"{where}branch_margins: names {sorted(got_margins)} != {sorted(expected_margins)}")
        if sorted(got_margins) == sorted(expected_margins):
            names = sorted(expected_margins)
            _close(check, f"{where}branch_margins", [got_margins[n] for n in names], [expected_margins[n] for n in names], tolerance["branch_margins"]["value"])
    got_gate, expected_gate = got["failed_gate"], expected["failed_gate"]
    if got_gate is None or expected_gate is None:
        check.expect(got_gate is None and expected_gate is None, f"{where}failed_gate: {got_gate} != {expected_gate}")
    else:
        check.expect(got_gate["name"] == expected_gate["name"], f"{where}failed_gate: {got_gate['name']} != {expected_gate['name']}")
        _close(check, f"{where}failed_gate", got_gate["value"], expected_gate["value"], tolerance["failed_gate"]["value"])


def curve_containment(poses: np.ndarray, curve: np.ndarray, closed: bool, spacing: float = CURVE_DENSIFY_SPACING) -> float:
    """The largest distance from a pose of ``poses`` to the densified ``curve`` (one-sided; 0 without poses)."""
    poses = np.asarray(poses, dtype=np.float64).reshape(-1, 3, 3)
    if len(poses) == 0:
        return 0.0
    dense = densify_curve(curve, closed, spacing)
    if len(dense) == 0:
        return math.inf
    return max(float(np.min(rotation_distances(pose, dense))) for pose in poses)


def _compare_traces(check: Check, got: Sequence[Mapping[str, Any]], fixture: Mapping[str, Any]) -> None:
    expected = fixture["expected"]["traces"]
    tolerance = fixture["tolerance"]
    check.expect(len(got) == len(expected), f"trace count {len(got)} != {len(expected)}")
    check.expect(
        sorted((t["status"], t["reason"]) for t in got) == sorted((t["status"], t["reason"]) for t in expected),
        f"status/reason {[(t['status'], t['reason']) for t in got]} != {[(t['status'], t['reason']) for t in expected]}",
    )
    if "budget_extent" in tolerance:
        _compare_budget_traces(check, got, fixture)
    else:
        curve, closed = traced_curve(got)
        reference, reference_closed = traced_curve(expected)
        distance = curve_distance(curve, closed, reference, reference_closed)
        limit = tolerance["curve_distance_rad"]["value"]
        check.expect(distance <= limit, f"curve distance {distance:.3e} > {limit}")
        length = sum(t["arclength"] for t in got)
        reference_length = sum(t["arclength"] for t in expected)
        rtol = tolerance["arclength_relative"]["value"]
        check.expect(abs(length - reference_length) <= rtol * reference_length + 1e-12, f"arclength {length} vs {reference_length}")
    bound = tolerance["residual_norm_bound"]["value"]
    worst = max((float(np.max(t["residual_norms"])) for t in got if len(t["residual_norms"])), default=0.0)
    check.expect(worst <= bound, f"residual norm {worst:.3e} > {bound:.1e}")
    if "pointwise_consistency" in tolerance:  # wave 2 recipes; a v0 fixture has none of them
        _compare_trace_pointwise(check, got, fixture)


def _compare_budget_traces(check: Check, got: Sequence[Mapping[str, Any]], fixture: Mapping[str, Any]) -> None:
    """The :data:`BUDGET_EXTENT_BASIS` recipe: the budget's own extent, and containment in the unbudgeted curve."""
    options = fixture["input"]["continuation"]
    reference = fixture["expected"]["reference_curve"]
    reference_curve = np.asarray(reference["poses"], dtype=np.float64).reshape(-1, 3, 3)
    limit = fixture["tolerance"]["curve_distance_rad"]["value"]
    for index, trace in enumerate(got):
        poses = np.asarray(trace["poses"], dtype=np.float64).reshape(-1, 3, 3)
        if trace["reason"] == "step_budget":
            expected_count = options["maximum_accepted_steps"] + 1
            check.expect(len(poses) == expected_count, f"trace {index}: step_budget with {len(poses)} poses != {expected_count}")
        elif trace["reason"] == "arclength_budget":
            low, high = options["maximum_arclength"] - options["maximum_advance"], options["maximum_arclength"]
            check.expect(low < trace["arclength"] <= high, f"trace {index}: arclength_budget at {trace['arclength']} outside ({low}, {high}]")
        elif trace["reason"] == "evaluation_budget":
            check.expect(len(poses) >= 1, f"trace {index}: evaluation_budget without partial geometry")
        distance = curve_containment(poses, reference_curve, bool(reference["closed"]))
        check.expect(distance <= limit, f"trace {index}: {distance:.3e} from the unbudgeted curve > {limit}")


def _compare_trace_pointwise(check: Check, got: Sequence[Mapping[str, Any]], fixture: Mapping[str, Any]) -> None:
    """Self-consistency of the per-pose arrays with EvaluatePath, and regularity of every accepted pose."""
    scene = _scene_of(fixture["input"])
    for index, trace in enumerate(got):
        poses = np.asarray(trace["poses"], dtype=np.float64).reshape(-1, 3, 3)
        names = list(trace["branch_margin_names"])
        margins = np.asarray(trace["branch_margins"], dtype=np.float64).reshape(len(poses), len(names))
        jacobian = np.asarray(trace["normal_jacobian"], dtype=np.float64)
        singular = np.asarray(trace["singular_values"], dtype=np.float64).reshape(-1, 2)
        available = list(trace["jacobian_available"])
        check.expect(
            len(jacobian) == len(singular) == len(available) == len(poses),
            f"trace {index}: per-pose arrays {len(jacobian)}/{len(singular)}/{len(available)} for {len(poses)} poses",
        )
        if not len(jacobian) == len(singular) == len(available) == len(poses):
            continue
        regular = all(available) and bool(np.all(margins > 0.0)) and bool(np.all(jacobian > 0.0))
        check.expect(regular, f"trace {index}: an accepted pose is unavailable, has a non-positive margin or J_perp <= 0")
        for i, pose in enumerate(poses):
            reference = pointwise_observables(scene, pose)
            mine = {
                "jacobian_available": bool(available[i]),
                "normal_jacobian": float(jacobian[i]),
                "singular_values": singular[i],
                "branch_margins": dict(zip(names, margins[i])),
                "failed_gate": None,
            }
            before = len(check.failures)
            _compare_pointwise(check, f"trace {index} pose {i}: ", mine, reference, pointwise_tolerances(reference["check"].margins))
            if len(check.failures) > before:
                break  # the first inconsistent pose names the defect; the rest would repeat it


def verify_trace_fiber(fixture: Mapping[str, Any], name: str = "") -> Check:
    check = Check(name)
    data = fixture["input"]
    scene = _scene_of(data)
    options = ContinuationOptions(**data["continuation"])
    seed = np.asarray(data["seed_pose"], dtype=np.float64).reshape(3, 3)
    target = np.asarray(data["target_direction"], dtype=np.float64)
    got = run_traces(scene, seed, target, options)
    _compare_traces(check, got, fixture)
    return check


def verify_seed_search(fixture: Mapping[str, Any], name: str = "") -> Check:
    check = Check(name)
    data = fixture["input"]
    scene = _scene_of(data)
    target = np.asarray(data["target_direction"], dtype=np.float64)
    sample = data["sample"]
    half_width = math.radians(data["band_half_width_deg"])
    seeds = BandSeeds(
        scene,
        np.asarray(sample["band_u"], dtype=np.float64).reshape(-1, 3),
        np.asarray(sample["band_phi"], dtype=np.float64).reshape(-1, 3),
        np.asarray(sample["band_deviation"], dtype=np.float64).reshape(-1),
        target,
        half_width,
    )
    result = discover_components(
        target,
        seeds,
        continuation=ContinuationOptions(**data["continuation"]),
        extra_seeds=[np.asarray(s, dtype=np.float64).reshape(3, 3) for s in data["extra_seeds"]],
        band_half_width_deg=data["band_half_width_deg"],
        cluster_radius_rad=data["cluster_radius_rad"],
        distance_threshold=data["distance_threshold"],
    )
    got = discovery_record(result)
    expected = fixture["expected"]
    tolerance = fixture["tolerance"]
    for key in ("completeness", "pool_count", "extra_seed_count", "raw_cluster_count", "admissible_count", "events"):
        check.expect(got[key] == expected[key], f"{key}: {got[key]} != {expected[key]}")
    check.expect(len(got["components"]) == len(expected["components"]), "component count differs")
    check.expect(
        [item["cause"] for item in got["incomplete"]] == [item["cause"] for item in expected["incomplete"]],
        "incomplete causes differ",
    )
    for index, (mine, reference) in enumerate(zip(got["components"], expected["components"])):
        closed = reference["kind"] == "closed"
        reference_curve = np.asarray(reference["curve_poses"], dtype=np.float64).reshape(-1, 3, 3)
        check.expect(mine["kind"] == reference["kind"] and mine["status"] == reference["status"], f"component {index}: kind/status")
        check.expect(
            sorted(filter(None, (mine["reason"], mine["start_reason"]))) == sorted(filter(None, (reference["reason"], reference["start_reason"]))),
            f"component {index}: end reasons",
        )
        seed_distance = distance_to_curve(np.asarray(mine["seed"]).reshape(3, 3), reference_curve)
        limit = tolerance["seed_to_curve_rad"]["value"]
        check.expect(seed_distance <= limit, f"component {index}: seed {seed_distance:.3e} from the curve > {limit}")
        distance = curve_distance(np.asarray(mine["curve_poses"]).reshape(-1, 3, 3), closed, reference_curve, closed)
        limit = tolerance["curve_distance_rad"]["value"]
        check.expect(distance <= limit, f"component {index}: curve distance {distance:.3e} > {limit}")
        if closed:
            rtol = tolerance["closed_arclength_relative"]["value"]
            check.expect(
                abs(mine["arclength"] - reference["arclength"]) <= rtol * reference["arclength"],
                f"component {index}: arclength {mine['arclength']} vs {reference['arclength']}",
            )
    return check


VERIFIERS = {"evaluate_path": verify_evaluate_path, "trace_fiber": verify_trace_fiber, "seed_search": verify_seed_search}


def verify_fixture(path: Path) -> Check:
    """Read one fixture back, recompute it with this checkout, compare within the fixture's own tolerances."""
    fixture = read_json(path)
    if fixture.get("format") != FORMAT or fixture.get("schema_version") != SCHEMA_VERSION:
        return Check(Path(path).name, [f"not a schema {SCHEMA_VERSION} {FORMAT} fixture"], 1)
    return VERIFIERS[fixture["fixture_kind"]](fixture, Path(path).name)


# ------------------------------------------------------------------ export
MANIFEST = "manifest.json"


def export_cell(cell: Cell, output_dir: Path, provenance: Mapping[str, Any]) -> dict[str, Any]:
    """Write the fixtures of one cell; returns its manifest entry (files, skips, selection)."""
    scene = cell.scene
    entry: dict[str, Any] = {
        "name": cell.name,
        "path": scene.path_id,
        "crystal": dict(scene.crystal_spec),
        "category": cell.category,
        "rationale": cell.rationale,
        "files": [],
        "skipped": [],
    }
    try:
        choice = choose_point(cell)
    except PointUnavailable as error:
        entry["skipped"].append({"fixture": "all", "reason": str(error)})
        return entry
    entry["selection"] = choice.selection
    record = {"name": cell.name, "path": scene.path_id, "category": cell.category, "rationale": cell.rationale, "selection": choice.selection}

    def write(suffix: str, document: Mapping[str, Any]) -> None:
        name = f"{cell.name}__{suffix}.json"
        write_json(Path(output_dir) / name, document)
        entry["files"].append(name)

    write("evaluate_path__point", build_evaluate_path_fixture(scene, choice.pose, {**record, "pose_label": "point"}, provenance))
    if choice.outside_pose is not None:
        write(
            "evaluate_path__outside",
            build_evaluate_path_fixture(scene, choice.outside_pose, {**record, "pose_label": "outside (mirror across the edge of U_P)"}, provenance),
        )
    trace = build_trace_fiber_fixture(scene, choice.pose, choice.target, record, provenance)
    write("trace_fiber", trace)
    curve, _ = traced_curve(trace["expected"]["traces"])
    margins = [minimum_validity_margin(path_domain(pose, scene.faces, scene.incident_direction, scene.refractive_index, crystal=scene.crystal).margins, scene.faces)[1] for pose in curve]
    nearest = curve[int(np.argmin(margins))]
    write(
        "evaluate_path__curve_min_margin",
        build_evaluate_path_fixture(scene, nearest, {**record, "pose_label": "the traced pose with the smallest validity margin"}, provenance),
    )
    if cell.seed_search_skip is not None:
        entry["skipped"].append({"fixture": "seed_search", "reason": cell.seed_search_skip})
    else:
        write("seed_search", build_seed_search_fixture(cell, choice.target, record, provenance))
    return entry


# ------------------------------------------------------------------ edge cells (wave 2)
EDGE_POINTS = ("critical_offset", "extremum", "random", "antipodal_target", "target")


@dataclass(frozen=True)
class OptionVariant:
    """A ``trace_fiber`` fixture from the edge cell's seed under changed options.

    ``kind`` is ``"perturbation"`` (controller settings around the reference defaults: the export asserts the
    traced curve still equals the default one within the fixture tolerances, contract C06) or ``"limit"`` (a
    budget or a step/corrector limit that ends the trace early: C09, C11).
    """

    label: str
    kind: str
    overrides: tuple[tuple[str, Any], ...]

    def __post_init__(self) -> None:
        if self.kind not in ("perturbation", "limit"):
            raise ValueError("kind must be 'perturbation' or 'limit'")

    def options(self) -> ContinuationOptions:
        return ContinuationOptions(**dict(self.overrides))


@dataclass(frozen=True)
class EdgeCell:
    """A wave 2 edge case (``docs/analytic-parity-fixtures.md`` section 6.1), kept apart from the 3 x 3 matrix.

    ``point`` says where the seed comes from: ``critical_offset`` (the lit side of an interior extremum of
    ``D_P`` at ``critical_offset_deg``, as the matrix's ``critical`` category), ``extremum`` (the extremum
    itself: a rank-deficient seed), ``random`` (the matrix's random draw), ``antipodal_target`` (the random
    pose aimed at ``-d``), or ``target`` (``target`` given; the seeds are the components that seed search on
    the ``store_n`` sample returns, one ``trace_fiber`` each).  ``serves`` names the contract section 11 rows.
    """

    label: str
    scene: Scene
    point: str
    serves: tuple[str, ...]
    rationale: str
    critical_offset_deg: float = 0.5
    rng_seed: int = 20260928
    target: tuple[float, float, float] | None = None
    seed_search: bool = False
    variants: tuple[OptionVariant, ...] = ()
    store_n: int = 100_000
    band_half_width_deg: float = 0.2
    cluster_radius_rad: float = 0.3
    distance_threshold: float = ContinuationOptions.closure_distance

    def __post_init__(self) -> None:
        if self.point not in EDGE_POINTS:
            raise ValueError(f"point must be one of {EDGE_POINTS}")
        if (self.point == "target") != (self.target is not None):
            raise ValueError("give target exactly for point='target'")
        if self.point == "target" and not self.seed_search:
            raise ValueError("point='target' takes its seeds from seed search")

    @property
    def name(self) -> str:
        return f"{self.scene.path_id}__{self.label}"

    @property
    def search_cell(self) -> Cell:
        """The seed-search parameters as a matrix :class:`Cell` (``build_seed_search_fixture`` reads them)."""
        return Cell(
            self.scene,
            "random",
            self.rationale,
            store_n=self.store_n,
            band_half_width_deg=self.band_half_width_deg,
            cluster_radius_rad=self.cluster_radius_rad,
            distance_threshold=self.distance_threshold,
        )


def _extremum_u(scene: Scene) -> tuple[np.ndarray, dict]:
    from .dp_field import DPField

    extrema = [point for point in DPField.build(scene.crystal, scene.faces, scene.refractive_index).interior_critical_points if point.kind in ("minimum", "maximum")]
    if not extrema:
        raise PointUnavailable(f"D_P of {scene.path_id} has no interior extremum on this crystal")
    centre = _unit(np.asarray(extrema[0].position, dtype=np.float64))
    return centre, {"method": "the first interior extremum of D_P itself", "extremum_kind": extrema[0].kind, "extremum_deviation_deg": math.degrees(extrema[0].value)}


def _outgoing(scene: Scene, pose: np.ndarray) -> np.ndarray:
    return np.asarray(path_direction(pose, scene.faces, scene.incident_direction, scene.refractive_index, crystal=scene.crystal).direction)


def edge_seeds(cell: EdgeCell) -> tuple[list[np.ndarray], np.ndarray, dict[str, Any]]:
    """The seed poses, the target and the selection record of an edge cell (``target`` cells run seed search)."""
    scene = cell.scene
    if cell.point == "target":
        target = np.asarray(cell.target, dtype=np.float64)
        store = scene_store(scene, cell.store_n)
        result = discover_components(
            target,
            StoreSeeds(store, scene.faces, scene.sun),
            continuation=ContinuationOptions(),
            band_half_width_deg=cell.band_half_width_deg,
            cluster_radius_rad=cell.cluster_radius_rad,
            distance_threshold=cell.distance_threshold,
        )
        if not result.components:
            raise PointUnavailable(f"seed search at the target returned no component ({result.completeness})")
        seeds = [np.asarray(component.seed, dtype=np.float64) for component in result.components]
        deviation = math.degrees(math.acos(float(np.clip(target @ scene.incident_direction, -1.0, 1.0))))
        return seeds, target, {"method": "the component seeds of seed search at the given target", "deviation_deg": deviation, "components": len(seeds)}
    if cell.point == "critical_offset":
        u, selection = critical_u(scene, cell.critical_offset_deg)
        selection = {"method": "lit side of an interior extremum of D_P", **selection}
    elif cell.point == "extremum":
        u, selection = _extremum_u(scene)
    else:
        u, draws = random_u(scene, cell.rng_seed)
        selection = {"method": "first admissible uniform draw", "rng": "numpy.random.default_rng", "rng_seed": cell.rng_seed, "draws": draws}
    pose = pose_of_u(scene, u)
    target = _outgoing(scene, pose)
    if cell.point == "antipodal_target":
        target = -target
        selection = {**selection, "target": "the antipode -d of the pose's own outgoing direction d"}
    return [pose], target, {**selection, "u": u, "deviation_deg": math.degrees(body_outgoing(scene, u)[1])}


def _curve_extreme_poses(scene: Scene, curve: np.ndarray) -> dict[str, np.ndarray]:
    """The traced poses with the smallest branch margin and the smallest ``J_perp`` (evaluate_path points)."""
    observables = [pointwise_observables(scene, pose) for pose in curve]
    margins = [min(item["branch_margins"].values()) for item in observables]
    jacobians = [item["normal_jacobian"] for item in observables]
    return {"curve_min_margin": curve[int(np.argmin(margins))], "curve_min_jacobian": curve[int(np.argmin(jacobians))]}


def export_edge_cell(cell: EdgeCell, output_dir: Path, provenance: Mapping[str, Any]) -> dict[str, Any]:
    """Write the fixtures of one edge cell; returns its manifest entry."""
    scene = cell.scene
    entry: dict[str, Any] = {
        "name": cell.name,
        "path": scene.path_id,
        "crystal": dict(scene.crystal_spec),
        "label": cell.label,
        "point": cell.point,
        "serves": list(cell.serves),
        "rationale": cell.rationale,
        "files": [],
        "skipped": [],
    }
    try:
        seeds, target, selection = edge_seeds(cell)
    except PointUnavailable as error:
        entry["skipped"].append({"fixture": "all", "reason": str(error)})
        return entry
    entry["selection"] = selection
    record = {"name": cell.name, "path": scene.path_id, "category": "edge", "label": cell.label, "serves": list(cell.serves), "rationale": cell.rationale, "selection": selection}

    def write(suffix: str, document: Mapping[str, Any]) -> None:
        name = f"{cell.name}__{suffix}.json"
        write_json(Path(output_dir) / name, document)
        entry["files"].append(name)

    if cell.point != "target":
        write("evaluate_path__point", build_evaluate_path_fixture(scene, seeds[0], {**record, "pose_label": "point"}, provenance))
    base_traces = None
    for index, seed in enumerate(seeds):
        suffix = "trace_fiber" if len(seeds) == 1 else f"trace_fiber__component_{index}"
        trace = build_trace_fiber_fixture(scene, seed, target, {**record, "seed_label": suffix}, provenance)
        write(suffix, trace)
        base_traces = base_traces or trace["expected"]["traces"]
    curve, _ = traced_curve(base_traces)
    if len(curve):
        for label, pose in _curve_extreme_poses(scene, curve).items():
            text = {"curve_min_margin": "the traced pose with the smallest branch margin", "curve_min_jacobian": "the traced pose with the smallest J_perp"}[label]
            write(f"evaluate_path__{label}", build_evaluate_path_fixture(scene, pose, {**record, "pose_label": text}, provenance))
    for variant in cell.variants:
        options = variant.options()
        fixture = build_trace_fiber_fixture(
            scene, seeds[0], target, {**record, "variant": variant.label, "variant_kind": variant.kind}, provenance, options, reference_traces=base_traces
        )
        if variant.kind == "perturbation":
            check = Check(variant.label)
            _compare_traces(check, fixture["expected"]["traces"], {**fixture, "expected": {"traces": base_traces}})
            if check.failures:
                raise AssertionError(f"{cell.name} {variant.label}: the perturbed trace left the default one: {check.failures}")
        write(f"trace_fiber__{variant.label}", fixture)
    if cell.seed_search:
        write("seed_search", build_seed_search_fixture(cell.search_cell, target, record, provenance))
    return entry


def export_matrix(
    cells: Sequence[Cell], output_dir: Path, edge_cells: Sequence[EdgeCell] = (), band_sum_cells: Sequence[BandSumCell] = ()
) -> dict[str, Any]:
    """Export every cell (edge cell, band-sum cell) into ``output_dir`` and write :data:`MANIFEST`; returns the manifest.

    The manifest lists the matrix under ``cells``, the edge cells under ``edge_cells`` and the module B
    fixtures under ``band_sum_cells``; a key is absent when it has no entry, so a matrix-only export has the
    v0 manifest and an export without band-sum cells the wave 2 one, byte for byte.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    provenance = fixture_provenance()
    manifest = {
        "format": FORMAT,
        "schema_version": SCHEMA_VERSION,
        "symmetry_semantics": SYMMETRY_SEMANTICS,
        "provenance": provenance,
        "cells": [export_cell(cell, output_dir, provenance) for cell in cells],
    }
    if edge_cells:
        manifest["edge_cells"] = [export_edge_cell(cell, output_dir, provenance) for cell in edge_cells]
    if band_sum_cells:
        manifest["band_sum_cells"] = [export_band_sum_cell(cell, output_dir, provenance) for cell in band_sum_cells]
    write_json(output_dir / MANIFEST, manifest)
    return manifest


def verify_directory(output_dir: Path) -> list[Check]:
    """:func:`verify_fixture` of every file the manifest lists (matrix, edge and band-sum cells)."""
    manifest = read_json(Path(output_dir) / MANIFEST)
    entries = [*manifest["cells"], *manifest.get("edge_cells", []), *manifest.get("band_sum_cells", [])]
    return [verify_fixture(Path(output_dir) / name) for cell in entries for name in cell["files"]]


# ------------------------------------------------------------------ band sum (module B, docs/band-sum-contract.md)
# One fixture per cell: a single concrete path, one pose density, a pixel table given as directions (any
# projection) and the estimator's per-pixel output.  Layer 1 replays the estimator on the band events the
# fixture carries (tight: summation order); layer 2 regenerates the sample from the section 9.5.2 lattice
# (allowances derived per pixel from the events within float reach of a band end or a gate).
BAND_SUM_KIND = "band_sum"
BAND_SUM_SAMPLER = "antipodal Fibonacci lattice (docs/phase1-math-contract.md section 9.5.2), kept events w = A T > 0"
BAND_SUM_VALUE_RTOL = 1e-10
BAND_SUM_VALUE_BASIS = (
    "the same sum over bit-identical events (layer 1): summation order and the rounding of rho's arccos/atan2/exp; "
    "LI's scatter and gather forms agree to 1e-12 on values and 3e-12 on K_eff near 1 (docs/phase2.md section 8), "
    "a narrow density amplifies d(theta) by |theta - mean| / sigma^2; 1e-10 relative leaves >= 30x room; the "
    "density's normalisation integrals I, Q must be accurate to 1e-12 relative (docs/band-sum-contract.md section 4.3)"
)
BAND_SUM_EDGE_EPSILON_RAD = 1e-9
BAND_SUM_GATE_EPSILON = 1e-9
BAND_SUM_WEIGHT_EPSILON = 1e-9
BAND_SUM_LAYER2_BASIS = (
    "layer 2 (the backend regenerates the sample): an event can change sides only within float reach of a band end "
    "(|D - delta_lo|, |D - delta_hi| <= edge_epsilon_rad), a validity gate (|smallest validity margin| <= gate_epsilon) "
    "or the w > 0 gate (0 < w <= weight_epsilon; w -> 0 continuously at every validity gate: A ~ cos at entry and "
    "every face the corridor projects, T_exit -> 0 at exit TIR); per pixel expected.pixels[].allowance counts those "
    "lattice points and bounds their effect on K, K_rho_pos, the value and K_eff (docs/band-sum-contract.md section 7.2); "
    "float64 D of the same closed-form chain differs by <= 1e-13 rad away from D = 0, pi, so 1e-9 leaves >= 1e4 room"
)
BAND_SUM_SUBNORMAL_BASIS = (
    "K_rho_pos counts c_i = w_i rho_i > 0 in IEEE-754 double with gradual underflow; a band event whose c_i is "
    "subnormal in LI (0 < c_i < 2.2250738585072014e-308) may be 0 on a backend that flushes subnormals, so each pixel "
    "allows allowance.K_rho_pos_subnormal of them; its value contribution is below 1e-300"
)
RANK0_SIGMAS = 5.0
RANK0_HAAR_CHECK_SAMPLES = 1_000_000


def linear_pixel_table(render: Mapping[str, Any], pixels: Sequence[tuple[int, int]]) -> dict[str, Any]:
    """Outgoing directions of linear-lens pixels: centre, four corners in cyclic order, the solid angle.

    The solid angle is :func:`.path_class.pixel_solid_angle` (centre approximation ``cos^3 / scale^2``).
    """
    from .camera import linear_pixel_outgoing_direction
    from .path_class import pixel_solid_angle

    cyclic = ((-0.5, -0.5), (-0.5, 0.5), (0.5, 0.5), (0.5, -0.5))
    return {
        "projection": {"kind": "linear", "render": dict(render)},
        "labels": [[int(r), int(c)] for r, c in pixels],
        "centre": np.array([linear_pixel_outgoing_direction(r, c, **render) for r, c in pixels]),
        "corners": np.array([[linear_pixel_outgoing_direction(r + dr, c + dc, **render) for dr, dc in cyclic] for r, c in pixels]),
        "solid_angle": np.array([pixel_solid_angle(render, r, c) for r, c in pixels]),
    }


def lambert_view(centre_sky: Sequence[float], field_radius_deg: float, size: int) -> dict[str, Any]:
    """A single-disk Lambert azimuthal equal-area view (Lumice ``doc/prototypes/analyze-workspace.html`` ``makeView``).

    ``centre_sky`` is the sky direction at the disk centre; screen ``up`` is the zenith projected on the
    tangent plane (``[-1, 0, 0]`` when the centre is within ``acos(0.999)`` of the zenith), ``right = centre x
    up``; the disk of radius ``0.492 size`` pixels reaches ``field_radius_deg`` from the centre.  An example
    expansion for the fixtures, not part of the contract (``docs/band-sum-contract.md`` section 3.3).
    """
    centre = np.asarray(centre_sky, dtype=np.float64)
    centre = centre / np.linalg.norm(centre)
    up = np.array([-1.0, 0.0, 0.0]) if abs(centre[2]) > 0.999 else _unit(np.array([0.0, 0.0, 1.0]) - centre * centre[2])
    radius = size * 0.492
    return {
        "centre": centre,
        "up": up,
        "right": np.cross(centre, up),
        "half": size / 2.0,
        "k": radius / (2.0 * math.sin(min(math.radians(field_radius_deg), math.pi) / 2.0)),
    }


def lambert_sky_direction(view: Mapping[str, Any], x: float, y: float) -> np.ndarray:
    """The sky direction at screen point ``(x, y)`` (``y`` down), the prototype's ``vInv`` without its disk clip."""
    big_x, big_y = (x - view["half"]) / view["k"], (view["half"] - y) / view["k"]
    rho = math.hypot(big_x, big_y)
    if rho < 1e-12:
        return np.array(view["centre"], dtype=np.float64)
    theta = 2.0 * math.asin(min(1.0, rho / 2.0))
    tangent = _unit(view["right"] * (big_x / rho) + view["up"] * (big_y / rho))
    return _unit(view["centre"] * math.cos(theta) + tangent * math.sin(theta))


def lambert_pixel_table(
    centre_sky: Sequence[float], field_radius_deg: float, size: int, pixels: Sequence[tuple[int, int]]
) -> dict[str, Any]:
    """Outgoing directions ``d = -sky`` of Lambert pixels ``(y, x)`` covering ``[x, x+1) x [y, y+1)``.

    Corners in cyclic order; the solid angle is exact for an equal-area map, ``1 / k^2`` per unit pixel.
    """
    view = lambert_view(centre_sky, field_radius_deg, size)
    cyclic = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
    return {
        "projection": {
            "kind": "lambert_azimuthal_equal_area",
            "centre_sky": [float(v) for v in view["centre"]],
            "field_radius_deg": float(field_radius_deg),
            "size": int(size),
            "label_meaning": "(y, x): pixel [x, x + 1) x [y, y + 1), y down, up = zenith on the tangent plane, right = centre x up",
        },
        "labels": [[int(y), int(x)] for y, x in pixels],
        "centre": np.array([-lambert_sky_direction(view, x + 0.5, y + 0.5) for y, x in pixels]),
        "corners": np.array([[-lambert_sky_direction(view, x + dx, y + dy) for dx, dy in cyclic] for y, x in pixels]),
        "solid_angle": np.full(len(pixels), 1.0 / view["k"] ** 2),
    }


def pixel_table_of(projection: Mapping[str, Any], pixels: Sequence[tuple[int, int]]) -> dict[str, Any]:
    if projection["kind"] == "linear":
        return linear_pixel_table(projection["render"], pixels)
    if projection["kind"] == "lambert_azimuthal_equal_area":
        return lambert_pixel_table(projection["centre_sky"], projection["field_radius_deg"], projection["size"], pixels)
    raise ValueError(f"unknown projection {projection['kind']!r}")


def pixel_contains(corners: np.ndarray, direction: np.ndarray) -> bool:
    """``direction`` lies in the spherical quadrilateral of the cyclic ``corners`` (great-circle edges, closed).

    Every ``det(c_k, c_{k+1}, x)`` has one sign (zero allowed) and ``x`` is on the corners' side of the sphere
    (the sign test alone also accepts the antipode).
    """
    corners = np.asarray(corners, dtype=np.float64)
    x = np.asarray(direction, dtype=np.float64)
    signs = np.array([np.linalg.det(np.stack([corners[k], corners[(k + 1) % 4], x])) for k in range(4)])
    return bool(corners.sum(axis=0) @ x > 0.0 and (np.all(signs >= 0.0) or np.all(signs <= 0.0)))


def density_of(spec: Mapping[str, Any]):
    from .pose_density import build_pose_density

    return build_pose_density(spec["family"], **{key: value for key, value in spec.items() if key != "family"})


def _density_block(spec: Mapping[str, Any]) -> dict[str, Any]:
    """The fixture's ``pose_density``: family, resolved degree parameters, LI's normalisation integrals (informative)."""
    from .pose_density import resolve_pose_density_parameters

    density = density_of(spec)
    block: dict[str, Any] = {"family": spec["family"], **resolve_pose_density_parameters(**dict(spec))}
    informative = {}
    if hasattr(density, "marginal_integral"):
        informative["zenith_marginal_integral"] = density.marginal_integral
    if hasattr(density, "roll_integral"):
        informative["zenith_marginal_integral"] = density.zenith.marginal_integral
        informative["roll_marginal_integral"] = density.roll_integral
    if informative:
        block["normalization_informative"] = informative
    return block


def _density_max(density) -> float:
    """The largest value of ``rho_H`` (the Gaussian profiles peak at 1)."""
    if hasattr(density, "roll_integral"):
        return 2.0 / density.zenith.marginal_integral * 2.0 * math.pi / density.roll_integral
    if hasattr(density, "marginal_integral"):
        return 2.0 / density.marginal_integral
    return 1.0


@dataclass(frozen=True)
class BandSumCell:
    """A module B fixture: one concrete path, one pose density, a pixel table (``docs/band-sum-contract.md``).

    ``projection`` is how LI expands the table (``linear`` with a Lumice ``render`` block, or
    ``lambert_azimuthal_equal_area``); ``pixels`` are its ``(row, column)`` / ``(y, x)`` labels.  A rank-0
    path's point mass is the lattice mean when ``rank0_sample_count`` is ``None`` (deterministic, the random
    density only) and LI's Haar-stream estimate of that many samples otherwise (a statistical comparison).
    """

    label: str
    scene: Scene
    density: Mapping[str, Any]
    projection: Mapping[str, Any]
    pixels: tuple[tuple[int, int], ...]
    n: int
    rationale: str
    rank0_sample_count: int | None = None

    @property
    def name(self) -> str:
        return f"{self.scene.path_id}__band_sum_{self.label}"


def _identity_group(faces: tuple[int, ...]):
    from .path_class import StoreGroup, Transport

    return StoreGroup((faces,), (Transport((faces,), None),))


def _singular(table: Mapping[str, Any], scene: Scene) -> np.ndarray:
    s = scene.incident_direction
    return np.array([pixel_contains(quad, s) or pixel_contains(quad, -s) for quad in np.asarray(table["corners"], dtype=np.float64)])


def _bands(table: Mapping[str, Any], scene: Scene):
    from .band_sum import pixel_bands_from_directions

    labels = np.asarray(table["labels"], dtype=np.int64).reshape(-1, 2)
    return pixel_bands_from_directions(
        np.asarray(table["centre"], dtype=np.float64), np.asarray(table["corners"], dtype=np.float64), scene.sun, labels[:, 0], labels[:, 1]
    )


def band_sum_pixels(scene: Scene, density, table: Mapping[str, Any], events, n: int, form: str) -> list[dict[str, Any]]:
    """The rank-2 per-pixel output of ``docs/band-sum-contract.md`` section 6 on ``events`` (an :class:`.s2_store.S2Events`).

    ``form`` is ``"scatter"`` (the production :func:`.band_sum.scatter_store`) or ``"gather"`` (the oracle
    :func:`.band_sum.class_band_sum_of_band`); a singular pixel (it contains ``s`` or ``-s``) has no value.
    """
    from .band_sum import ScatterSums, class_band_sum_of_band, scatter_results, scatter_store

    group = _identity_group(scene.faces)
    bands = _bands(table, scene)
    singular = _singular(table, scene)
    regular = np.flatnonzero(~singular)
    if form == "scatter":
        part = bands.take(regular)
        sums = ScatterSums.zeros(len(part))
        scatter_store(events, group, density, part, sums)
        results = scatter_results(part, sums, n)
    elif form == "gather":
        centres, corners = np.asarray(table["centre"], dtype=np.float64), np.asarray(table["corners"], dtype=np.float64)
        from .band_sum import band_of_pixel_directions

        results = [
            class_band_sum_of_band([(events.arrays(), group)], scene.sun, density, band_of_pixel_directions(centres[i], list(corners[i]), scene.sun), n)
            for i in regular
        ]
    else:
        raise ValueError("form must be 'scatter' or 'gather'")
    by_pixel = dict(zip(regular.tolist(), results))
    out = []
    for i, label in enumerate(np.asarray(table["labels"]).reshape(-1, 2).tolist()):
        record: dict[str, Any] = {"label": label, "delta": bands.delta[i], "delta_lo": bands.lo[i], "delta_hi": bands.hi[i]}
        if singular[i]:
            record.update(status="singular", value=None, K=None, K_rho_pos=None, K_eff=None, total=None, square=None)
        else:
            r = by_pixel[i]
            record.update(status="ok", value=r.value, K=r.K, K_rho_pos=r.K_rho_pos, K_eff=r.K_eff, total=r.total, square=r.square)
        out.append(record)
    return out


def _events_of(store: S2EventStore):
    from .s2_store import S2Events

    e = store.events
    return S2Events(*(np.asarray(a, dtype=np.float64) for a in (e.u, e.phi, e.D, e.w)))


def _band_union(events, bands, regular: np.ndarray) -> np.ndarray:
    """Store indices of every event in a regular pixel's band (``searchsorted``, left-closed right-open), in store order."""
    first = np.searchsorted(events.D, bands.lo[regular])
    stop = np.searchsorted(events.D, bands.hi[regular])
    return np.unique(np.concatenate([np.arange(a, b) for a, b in zip(first, stop)] + [np.zeros(0, dtype=np.int64)])).astype(np.int64)


def _lattice_fields(scene: Scene, n: int) -> dict[str, np.ndarray]:
    """``D``, ``w`` and the smallest validity margin at every lattice point (all ``n``, kept or not)."""
    from .s2_store import align_rotations, evaluate_fields, store_lattice

    u = store_lattice(n)
    rotations = align_rotations(u, scene.sun)
    fields = evaluate_fields(rotations, scene.sun, scene.crystal, scene.refractive_index, [scene.faces])
    margins = path_domain_batch(rotations, scene.faces, scene.incident_direction, scene.refractive_index, crystal=scene.crystal).margins
    smallest = np.min(np.stack([np.asarray(margins[name], dtype=np.float64) for name in validity_margin_names(scene.faces)]), axis=0)
    return {"D": np.asarray(fields["D"]), "w": np.asarray(fields["w"]), "phi": np.asarray(fields["phi"]), "u": u, "margin": smallest}


def _contributions(scene: Scene, density, centre: np.ndarray, u: np.ndarray, phi: np.ndarray, d: np.ndarray, w: np.ndarray) -> np.ndarray:
    if len(w) == 0:
        return np.zeros(0)
    return w * density.evaluate_batch(event_rotations(u, phi, d, scene.sun, centre))


def _kish_range(total: float, square: float, count: int, candidates: np.ndarray, flips: int) -> float:
    """How far Kish ``total^2 / square`` can move when the candidate contributions are added or removed."""
    k_eff = total * total / square if square > 0.0 else 0.0
    if flips == 0:
        return 0.0
    spread, squares = float(np.sum(np.abs(candidates))), float(np.sum(candidates**2))
    high = (abs(total) + spread) ** 2 / (square - squares) if square - squares > 0.0 else math.inf
    low = max(abs(total) - spread, 0.0) ** 2 / (square + squares) if square + squares > 0.0 else 0.0
    return float(min(max(high - k_eff, k_eff - low), k_eff + count + flips))


def band_sum_allowances(
    scene: Scene, density, table: Mapping[str, Any], events, n: int, pixels: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any] | None]:
    """Per regular pixel, the layer-2 allowances (``BAND_SUM_LAYER2_BASIS``) and the subnormal count of its band events."""
    fields = _lattice_fields(scene, n)
    d, w, margin = fields["D"], fields["w"], fields["margin"]
    finite = np.isfinite(d)
    gate = np.abs(margin) <= BAND_SUM_GATE_EPSILON
    small = (w > 0.0) & (w <= BAND_SUM_WEIGHT_EPSILON)
    unplaced = int(np.count_nonzero(gate & ~finite))  # at a gate, D not finite: counted against every pixel
    centres = np.asarray(table["centre"], dtype=np.float64)
    rho_max = _density_max(density)
    out: list[dict[str, Any] | None] = []
    for i, pixel in enumerate(pixels):
        if pixel["status"] != "ok":
            out.append(None)
            continue
        lo, hi = pixel["delta_lo"], pixel["delta_hi"]
        near = finite & (d >= lo - BAND_SUM_EDGE_EPSILON_RAD) & (d <= hi + BAND_SUM_EDGE_EPSILON_RAD)
        edge = near & (w > 0.0) & ((np.abs(d - lo) <= BAND_SUM_EDGE_EPSILON_RAD) | (np.abs(d - hi) <= BAND_SUM_EDGE_EPSILON_RAD))
        gated = near & (gate | small)
        candidate = edge | gated
        kept = candidate & (w > 0.0)
        flips = int(np.count_nonzero(candidate)) + unplaced
        c_kept = _contributions(scene, density, centres[i], fields["u"][kept], fields["phi"][kept], d[kept], w[kept])
        a, b = np.searchsorted(events.D, [lo, hi])
        c_band = _contributions(scene, density, centres[i], events.u[a:b], events.phi[a:b], events.D[a:b], events.w[a:b])
        unkept = flips - int(np.count_nonzero(kept))
        width, delta = hi - lo, pixel["delta"]
        scale = 2.0 * math.pi * n * width * math.sin(delta)
        bound = float(np.sum(np.abs(c_kept))) + unkept * BAND_SUM_WEIGHT_EPSILON * rho_max
        subnormal = int(np.count_nonzero((c_band > 0.0) & (c_band < np.finfo(np.float64).tiny)))
        out.append(
            {
                "K_layer2": flips,
                "K_rho_pos_subnormal": subnormal,
                "K_rho_pos_layer2": flips + subnormal,
                "value_layer2": bound / scale if scale > 0.0 else (0.0 if bound == 0.0 else math.inf),
                "K_eff_layer2": _kish_range(pixel["total"], pixel["square"], pixel["K"], np.append(c_kept, np.full(unkept, BAND_SUM_WEIGHT_EPSILON * rho_max)), flips),
                "candidates": {"band_end": int(np.count_nonzero(edge)), "gate": int(np.count_nonzero(gated)), "gate_without_finite_D": unplaced},
            }
        )
    return out


def rank0_point_mass(cell: BandSumCell, samples: int | None = None) -> dict[str, Any]:
    """The rank-0 point mass ``m = E_Haar[[valid] rho A T]`` (lattice mean, or LI's Haar stream; section 5)."""
    from .path_class import RANK0_RNG_SEED, estimate_rank0_contribution

    scene = cell.scene
    count = samples or cell.rank0_sample_count
    if count is None:
        w = np.asarray(scene_store(scene, cell.n).events.w, dtype=np.float64)
        return {"m": float(np.sum(w)) / cell.n, "method": "lattice_mean", "n": cell.n}
    estimate = estimate_rank0_contribution(
        scene.crystal, scene.faces, scene.incident_direction, scene.refractive_index, density_of(cell.density), sample_count=count
    )
    return {
        "m": estimate.value,
        "method": "haar_stream",
        "error_estimate": estimate.error_estimate,
        "sample_count": estimate.sample_count,
        "rng_seed": RANK0_RNG_SEED,
        "rng": "numpy.random.default_rng(rng_seed); path_class.haar_domain_batches, 200000 per batch",
    }


def rank0_pixels(scene: Scene, table: Mapping[str, Any], m: float) -> list[dict[str, Any]]:
    s = scene.incident_direction
    out = []
    for label, quad, omega in zip(np.asarray(table["labels"]).reshape(-1, 2).tolist(), np.asarray(table["corners"]), np.asarray(table["solid_angle"])):
        lit = pixel_contains(quad, s)
        out.append({"label": label, "status": "point_mass" if lit else "ok", "value": m / float(omega) if lit else 0.0})
    return out


def build_band_sum_fixture(cell: BandSumCell, provenance: Mapping[str, Any]) -> dict[str, Any]:
    from .geometry import halo_map_rank

    scene = cell.scene
    density = density_of(cell.density)
    table = pixel_table_of(cell.projection, cell.pixels)
    rank = halo_map_rank(scene.crystal, scene.faces)
    record = {"name": cell.name, "path": scene.path_id, "label": cell.label, "rationale": cell.rationale, "rank": rank}
    fixture = _header(BAND_SUM_KIND, provenance, record)
    fixture["input"] = {
        **_scene_input(scene),
        "pose_density": _density_block(cell.density),
        "sample": {"sampler": BAND_SUM_SAMPLER, "n": cell.n},
        "pixels": table,
    }
    if rank == 0:
        mass = rank0_point_mass(cell)
        pixels = rank0_pixels(scene, table, mass["m"])
        statistical = mass["method"] == "haar_stream"
        if not statistical:
            fixture["input"]["events"] = {"w": np.asarray(scene_store(scene, cell.n).events.w, dtype=np.float64)}
            check = rank0_point_mass(cell, RANK0_HAAR_CHECK_SAMPLES)
            mass["haar_check_informative"] = {key: check[key] for key in ("m", "error_estimate", "sample_count", "rng_seed")}
        fixture["expected"] = {"rank": 0, "point_mass": mass, "pixels": pixels}
        fixture["tolerance"] = {
            "point_mass_relative": _tolerance(
                RANK0_SIGMAS * mass["error_estimate"] / mass["m"] if statistical else BAND_SUM_VALUE_RTOL,
                (
                    f"statistical: {RANK0_SIGMAS:g} standard errors of LI's Haar-stream mean, relative; a backend widens it by "
                    "its own error, |m - m_LI| <= 5 sqrt(sigma_LI^2 + sigma_backend^2) (docs/band-sum-contract.md section 5)"
                )
                if statistical
                else "the lattice mean sum(w)/N over the fixture's w (layer 1) or the regenerated lattice (layer 2): "
                "summation order, and the w -> 0 continuity at every gate for layer 2 (" + BAND_SUM_LAYER2_BASIS + ")",
            ),
            "status": _tolerance(0.0, "exact: the point mass sits on the pixel containing s (section 5), every other pixel is 0"),
        }
        return fixture
    store = scene_store(scene, cell.n)
    events = _events_of(store)
    pixels = band_sum_pixels(scene, density, table, events, cell.n, "scatter")
    regular = np.flatnonzero(~_singular(table, scene))
    union = _band_union(events, _bands(table, scene), regular)
    fixture["input"]["events"] = {"u": events.u[union], "phi": events.phi[union], "deviation": events.D[union], "w": events.w[union]}
    for pixel, allowance in zip(pixels, band_sum_allowances(scene, density, table, events, cell.n, pixels)):
        pixel["allowance"] = allowance
    fixture["expected"] = {"rank": 2, "pixels": pixels}
    fixture["tolerance"] = {
        "value_relative": _tolerance(BAND_SUM_VALUE_RTOL, BAND_SUM_VALUE_BASIS),
        "K_eff_relative": _tolerance(BAND_SUM_VALUE_RTOL, BAND_SUM_VALUE_BASIS),
        "K": _tolerance(0.0, "exact in layer 1: left-closed right-open band on the fixture's own D values (section 4.2)"),
        "K_rho_pos": _tolerance(0.0, "per pixel allowance.K_rho_pos_subnormal in layer 1: " + BAND_SUM_SUBNORMAL_BASIS),
        "status": _tolerance(0.0, "exact: singular iff the pixel contains s or -s (section 4.1)"),
        "edge_epsilon_rad": _tolerance(BAND_SUM_EDGE_EPSILON_RAD, BAND_SUM_LAYER2_BASIS),
        "gate_epsilon": _tolerance(BAND_SUM_GATE_EPSILON, BAND_SUM_LAYER2_BASIS),
        "weight_epsilon": _tolerance(BAND_SUM_WEIGHT_EPSILON, BAND_SUM_LAYER2_BASIS),
    }
    return fixture


def compare_band_sum_pixels(check: Check, where: str, got: Sequence[Mapping[str, Any]], fixture: Mapping[str, Any], layer: int) -> None:
    """``docs/analytic-parity-fixtures.md`` section 4, ``band_sum``: statuses exact, then per pixel within the layer's allowance."""
    expected = fixture["expected"]["pixels"]
    rtol = fixture["tolerance"]["value_relative"]["value"]
    check.expect(len(got) == len(expected), f"{where}pixel count {len(got)} != {len(expected)}")
    for mine, reference in zip(got, expected):
        label = f"{where}pixel {reference['label']}"
        check.expect(mine["status"] == reference["status"], f"{label}: status {mine['status']} != {reference['status']}")
        if reference["status"] != "ok" or mine["status"] != "ok":
            check.expect(mine["value"] is None, f"{label}: a singular pixel has no value")
            continue
        allowance = reference["allowance"]
        k_tol = 0 if layer == 1 else allowance["K_layer2"]
        pos_tol = allowance["K_rho_pos_subnormal"] if layer == 1 else allowance["K_rho_pos_layer2"]
        value_tol = rtol * abs(reference["value"]) + (0.0 if layer == 1 else allowance["value_layer2"])
        k_eff_tol = rtol * abs(reference["K_eff"]) + (0.0 if layer == 1 else allowance["K_eff_layer2"])
        check.expect(abs(mine["K"] - reference["K"]) <= k_tol, f"{label}: K {mine['K']} vs {reference['K']} (allowance {k_tol})")
        check.expect(
            abs(mine["K_rho_pos"] - reference["K_rho_pos"]) <= pos_tol, f"{label}: K_rho_pos {mine['K_rho_pos']} vs {reference['K_rho_pos']} (allowance {pos_tol})"
        )
        check.expect(abs(mine["value"] - reference["value"]) <= value_tol, f"{label}: value {mine['value']!r} vs {reference['value']!r} (tolerance {value_tol:.3e})")
        check.expect(abs(mine["K_eff"] - reference["K_eff"]) <= k_eff_tol, f"{label}: K_eff {mine['K_eff']!r} vs {reference['K_eff']!r} (tolerance {k_eff_tol:.3e})")


def _cell_of(fixture: Mapping[str, Any]) -> BandSumCell:
    data = fixture["input"]
    table = data["pixels"]
    density = {key: value for key, value in data["pose_density"].items() if key != "normalization_informative"}
    point_mass = fixture["expected"].get("point_mass", {})
    return BandSumCell(
        fixture["cell"]["label"],
        _scene_of(data),
        density,
        table["projection"],
        tuple(tuple(label) for label in table["labels"]),
        int(data["sample"]["n"]),
        fixture["cell"]["rationale"],
        point_mass.get("sample_count") if point_mass.get("method") == "haar_stream" else None,
    )


def verify_band_sum(fixture: Mapping[str, Any], name: str = "") -> Check:
    """LI's read-back of a ``band_sum`` fixture: the pixel table, then layer 1 (both forms) and layer 2."""
    from .s2_store import S2Events

    check = Check(name)
    cell = _cell_of(fixture)
    scene, data = cell.scene, fixture["input"]
    table = pixel_table_of(cell.projection, cell.pixels)
    for key in ("centre", "corners", "solid_angle"):
        check.expect(np.array_equal(np.asarray(table[key]), np.asarray(data["pixels"][key])), f"pixel table {key} is not LI's expansion of the projection")
    density = density_of(cell.density)
    if fixture["expected"]["rank"] == 0:
        mass = fixture["expected"]["point_mass"]
        rtol = fixture["tolerance"]["point_mass_relative"]["value"]
        if mass["method"] == "lattice_mean":
            layer1 = float(np.sum(np.asarray(data["events"]["w"], dtype=np.float64))) / cell.n
            check.expect(abs(layer1 - mass["m"]) <= rtol * abs(mass["m"]), f"layer 1 point mass {layer1!r} vs {mass['m']!r}")
            haar = mass["haar_check_informative"]
            check.expect(abs(mass["m"] - haar["m"]) <= RANK0_SIGMAS * haar["error_estimate"], "the lattice mean is not the Haar mean (definition check)")
        got = rank0_point_mass(cell)
        check.expect(abs(got["m"] - mass["m"]) <= rtol * abs(mass["m"]), f"point mass {got['m']!r} vs {mass['m']!r}")
        pixels = rank0_pixels(scene, table, got["m"])
        for mine, reference in zip(pixels, fixture["expected"]["pixels"]):
            check.expect(mine["status"] == reference["status"], f"pixel {reference['label']}: status")
            check.expect(abs(mine["value"] - reference["value"]) <= rtol * abs(reference["value"]), f"pixel {reference['label']}: value")
        return check
    given = data["events"]
    events = S2Events(*(np.asarray(given[key], dtype=np.float64) for key in ("u", "phi", "deviation", "w")))
    events = S2Events(events.u.reshape(-1, 3), events.phi.reshape(-1, 3), events.D, events.w)
    for form in ("scatter", "gather"):
        compare_band_sum_pixels(check, f"layer 1 {form}: ", band_sum_pixels(scene, density, table, events, cell.n, form), fixture, 1)
    store_events = _events_of(scene_store(scene, cell.n))
    regular = np.flatnonzero(~_singular(table, scene))
    union = _band_union(store_events, _bands(table, scene), regular)
    same = all(np.array_equal(getattr(store_events, key)[union], getattr(events, key)) for key in ("u", "phi", "D", "w"))
    check.expect(same, "the fixture's events are not the regenerated sample's band events")
    compare_band_sum_pixels(check, "layer 2: ", band_sum_pixels(scene, density, table, store_events, cell.n, "scatter"), fixture, 2)
    return check


VERIFIERS[BAND_SUM_KIND] = verify_band_sum


def export_band_sum_cell(cell: BandSumCell, output_dir: Path, provenance: Mapping[str, Any]) -> dict[str, Any]:
    """Write one ``band_sum`` fixture; returns its manifest entry."""
    name = f"{cell.name}.json"
    fixture = build_band_sum_fixture(cell, provenance)
    write_json(Path(output_dir) / name, fixture)
    return {
        "name": cell.name,
        "path": cell.scene.path_id,
        "crystal": dict(cell.scene.crystal_spec),
        "label": cell.label,
        "rank": fixture["cell"]["rank"],
        "pose_density": fixture["input"]["pose_density"]["family"],
        "projection": cell.projection["kind"],
        "rationale": cell.rationale,
        "files": [name],
        "skipped": [],
    }

__all__ = [
    "BAND_SUM_KIND",
    "CATEGORIES",
    "FORMAT",
    "MANIFEST",
    "SCHEMA_VERSION",
    "SYMMETRY_SEMANTICS",
    "BandSeeds",
    "BandSumCell",
    "Cell",
    "Check",
    "EdgeCell",
    "OptionVariant",
    "PointChoice",
    "PointUnavailable",
    "Scene",
    "band_sum_allowances",
    "band_sum_pixels",
    "build_band_sum_fixture",
    "build_crystal",
    "build_evaluate_path_fixture",
    "build_seed_search_fixture",
    "build_trace_fiber_fixture",
    "choose_point",
    "compare_band_sum_pixels",
    "curve_distance",
    "dumps",
    "evaluate_path",
    "export_band_sum_cell",
    "export_cell",
    "export_edge_cell",
    "export_matrix",
    "fixture_provenance",
    "lambert_pixel_table",
    "linear_pixel_table",
    "miller_wedge_deg",
    "pixel_contains",
    "prism_crystal",
    "pyramid_crystal",
    "rank0_point_mass",
    "verify_band_sum",
    "verify_directory",
    "verify_fixture",
]
