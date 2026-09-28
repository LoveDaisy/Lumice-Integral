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

from .continuation import ContinuationOptions, FiberResult, FiberStatus, trace_fiber
from .discovery import ComponentDiscoveryResult, discover_components, distance_to_curve
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

# Tolerances and their basis (docs/analytic-parity-fixtures.md section 4 is the table of the same values).
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


def random_u(scene: Scene, rng_seed: int, max_draws: int = 100_000) -> tuple[np.ndarray, int]:
    """The first admissible direction of ``numpy.random.default_rng(rng_seed)`` normals (uniform on ``S^2``)."""
    rng = np.random.default_rng(rng_seed)
    for draw in range(1, max_draws + 1):
        u = _unit(rng.standard_normal(3))
        if admissible_u(scene, u):
            return u, draw
    raise RuntimeError(f"no admissible u in {max_draws} draws for {scene.path_id}")


class PointUnavailable(Exception):
    """A matrix cell whose category has no real geometric point on this path (recorded, not faked)."""


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
def evaluate_path(scene: Scene, pose: np.ndarray) -> dict[str, Any]:
    """The ``LUMICE_ANALYTIC_PathEvaluation`` fields of ``pose`` (outgoing etc. only where ``valid``)."""
    faces = normalize_faces(scene.faces, scene.crystal)
    check = path_domain(pose, faces, scene.incident_direction, scene.refractive_index, crystal=scene.crystal)
    name, value = minimum_validity_margin(check.margins, faces)
    diagnostics = {"validity_margins": dict(check.margins), "nearest_margin": name, "nearest_margin_value": value}
    if not check.valid:
        return {"valid": False, "fresnel_transmission": 0.0, "diagnostics": {**diagnostics, "message": check.message}}
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
    }
    return fixture


# ------------------------------------------------------------------ curves
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
    dense_a = densify_curve(a, a_closed, spacing)
    dense_b = densify_curve(b, b_closed, spacing)
    a_to_b = max(float(np.min(rotation_distances(pose, dense_b))) for pose in np.asarray(a).reshape(-1, 3, 3))
    b_to_a = max(float(np.min(rotation_distances(pose, dense_a))) for pose in np.asarray(b).reshape(-1, 3, 3))
    return max(a_to_b, b_to_a)


# ------------------------------------------------------------------ TraceFiber
def continuation_options_dict(options: ContinuationOptions) -> dict[str, Any]:
    return asdict(options)


def _trace_record(result: FiberResult, sign: int, incident: np.ndarray) -> dict[str, Any]:
    poses = np.asarray(result.poses, dtype=np.float64)
    return {
        "initial_tangent_sign": sign,
        "status": str(result.status),
        "reason": str(result.reason),
        "poses": poses.reshape(-1, 9),
        "crystal_frame_sun_directions": np.einsum("nji,j->ni", poses, -incident),
        "arclength_increments": np.asarray(result.arclength_increments, dtype=np.float64),
        "residual_norms": np.asarray(result.residual_norms, dtype=np.float64),
        "tangents": np.asarray(result.tangents, dtype=np.float64),
        "arclength": float(np.sum(result.arclength_increments)),
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
    traces = [_trace_record(forward, options.initial_tangent_sign, scene.incident_direction)]
    if forward.status != FiberStatus.CLOSED:
        backward_options = ContinuationOptions(**{**asdict(options), "initial_tangent_sign": -options.initial_tangent_sign})
        backward = trace_fiber(problem, backward_options)
        traces.append(_trace_record(backward, backward_options.initial_tangent_sign, scene.incident_direction))
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
) -> dict[str, Any]:
    options = options or ContinuationOptions()
    fixture = _header("trace_fiber", provenance, cell)
    fixture["input"] = {
        **_scene_input(scene),
        "target_direction": target,
        "seed_pose": np.asarray(seed).reshape(9),
        "continuation": continuation_options_dict(options),
    }
    fixture["expected"] = {"traces": run_traces(scene, seed, target, options)}
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
    }
    return fixture


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
    return check


def _compare_traces(check: Check, got: Sequence[Mapping[str, Any]], expected: Sequence[Mapping[str, Any]], tolerance: Mapping[str, Any]) -> None:
    check.expect(len(got) == len(expected), f"trace count {len(got)} != {len(expected)}")
    check.expect(
        sorted((t["status"], t["reason"]) for t in got) == sorted((t["status"], t["reason"]) for t in expected),
        f"status/reason {[(t['status'], t['reason']) for t in got]} != {[(t['status'], t['reason']) for t in expected]}",
    )
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


def verify_trace_fiber(fixture: Mapping[str, Any], name: str = "") -> Check:
    check = Check(name)
    data = fixture["input"]
    scene = _scene_of(data)
    options = ContinuationOptions(**data["continuation"])
    seed = np.asarray(data["seed_pose"], dtype=np.float64).reshape(3, 3)
    target = np.asarray(data["target_direction"], dtype=np.float64)
    got = run_traces(scene, seed, target, options)
    _compare_traces(check, got, fixture["expected"]["traces"], fixture["tolerance"])
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


def export_matrix(cells: Sequence[Cell], output_dir: Path) -> dict[str, Any]:
    """Export every cell into ``output_dir`` and write :data:`MANIFEST`; returns the manifest."""
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
    write_json(output_dir / MANIFEST, manifest)
    return manifest


def verify_directory(output_dir: Path) -> list[Check]:
    """:func:`verify_fixture` of every file the manifest lists."""
    manifest = read_json(Path(output_dir) / MANIFEST)
    return [verify_fixture(Path(output_dir) / name) for cell in manifest["cells"] for name in cell["files"]]


__all__ = [
    "CATEGORIES",
    "FORMAT",
    "MANIFEST",
    "SCHEMA_VERSION",
    "SYMMETRY_SEMANTICS",
    "BandSeeds",
    "Cell",
    "Check",
    "PointChoice",
    "PointUnavailable",
    "Scene",
    "build_crystal",
    "build_evaluate_path_fixture",
    "build_seed_search_fixture",
    "build_trace_fiber_fixture",
    "choose_point",
    "curve_distance",
    "dumps",
    "evaluate_path",
    "export_cell",
    "export_matrix",
    "fixture_provenance",
    "miller_wedge_deg",
    "prism_crystal",
    "pyramid_crystal",
    "verify_directory",
    "verify_fixture",
]
