"""Production data for the canonical pose-fiber illustration."""

from __future__ import annotations

from pathlib import Path
import shlex
from typing import Any, Mapping

import numpy as np

from _canonical_data import IllustrationDataError
from _illustration_io import IllustrationData, load_pair, validate_required_arrays, write_pair


FIGURE_ID = "pose-fiber"
OUTPUT_NAME = f"{FIGURE_ID}.png"
SCHEMA_VERSION = "lumice-integral.illustration.pose-fiber/v1"

_REQUIRED_ARRAYS = {
    "u_curve": ((513, 3), "floating"),
    "relative_rotvec": ((513, 3), "floating"),
    "selected_indices": ((4,), "integer"),
    "selected_s": ((4,), "floating"),
    "selected_rotations": ((4, 3, 3), "floating"),
    "ray_points": ((4, 4, 3), "floating"),
    "ray_segment_directions": ((4, 3, 3), "floating"),
    "ray_face_numbers": ((4, 2), "integer"),
    "ray_face_margins": ((4, 2), "floating"),
    "ray_direction_errors": ((4, 3), "floating"),
    "snell_tangential_errors": ((4, 2), "floating"),
    "source_world": ((3,), "floating"),
    "incident_direction": ((3,), "floating"),
    "target_direction": ((3,), "floating"),
    "vertices_body": ((12, 3), "floating"),
    "vertices_world": ((4, 12, 3), "floating"),
    "face_numbers": ((None,), "integer"),
    "face_normals_world": ((4, None, 3), "floating"),
    "face_centers_world": ((4, None, 3), "floating"),
}

_ARRAY_SEMANTICS = {
    "u_curve": ("dimensionless", "Production fiber projection u = R^T s_hat"),
    "relative_rotvec": ("radian", "Presentation coordinates log(R_0^T R)"),
    "selected_indices": ("index", "Shared numbered positions in the presentation grid"),
    "selected_s": ("radian", "Production numerical arclength at numbered positions"),
    "selected_rotations": ("dimensionless", "Body-to-world rotations at numbered positions"),
    "ray_points": ("crystal length", "Tail, face-3 hit, face-5 hit, and head in world coordinates"),
    "ray_segment_directions": ("dimensionless", "Normalized incident, internal, and outgoing ray segments"),
    "ray_face_numbers": ("face id", "Entry and exit faces returned by Polyhedron.intersect_ray"),
    "ray_face_margins": ("crystal length", "In-face margins at the two physical hits"),
    "ray_direction_errors": ("dimensionless", "L2 errors against production incident/internal/outgoing directions"),
    "snell_tangential_errors": ("dimensionless", "Independent entry and exit tangential Snell residuals"),
    "source_world": ("dimensionless", "Unit direction toward the Sun"),
    "incident_direction": ("dimensionless", "Fixed incident propagation direction"),
    "target_direction": ("dimensionless", "Fixed outgoing propagation direction d"),
    "vertices_body": ("crystal length", "Canonical crystal vertices in body coordinates"),
    "vertices_world": ("crystal length", "Canonical crystal vertices at numbered poses"),
    "face_numbers": ("face id", "Canonical crystal face numbers"),
    "face_normals_world": ("dimensionless", "World outward normals at numbered poses"),
    "face_centers_world": ("crystal length", "World face centroids at numbered poses"),
}


def data_directory(data_root: Path | str) -> Path:
    return Path(data_root) / FIGURE_ID


def recompute_command(data_root: Path | str) -> str:
    root = Path(__file__).resolve().parents[2]
    return shlex.join(
        [
            "uv",
            "run",
            "--project",
            str(root),
            "python",
            str(Path(__file__).with_name("pose_fiber.py").resolve()),
            "--recompute",
            "--data-dir",
            str(Path(data_root).resolve()),
        ]
    )


def _fail(directory: Path, reason: str, data_root: Path) -> IllustrationDataError:
    return IllustrationDataError(
        f"invalid {FIGURE_ID} data at {directory}: {reason}. Recompute with: {recompute_command(data_root)}"
    )


def _validate(
    directory: Path,
    metadata: Any,
    arrays: Mapping[str, np.ndarray],
    digest: str,
    *,
    data_root: Path,
) -> None:
    fail = lambda reason: _fail(directory, reason, data_root)
    if not isinstance(metadata, Mapping):
        raise fail("metadata.json root must be an object")
    if metadata.get("schema") != SCHEMA_VERSION or metadata.get("figure_id") != FIGURE_ID:
        raise fail("schema or figure identity is wrong")
    payload = metadata.get("payload")
    if not isinstance(payload, Mapping) or payload.get("file") != "arrays.npz":
        raise fail("payload file declaration is missing or wrong")
    if payload.get("sha256") != digest:
        raise fail("arrays.npz SHA-256 does not match metadata")
    validate_required_arrays(arrays, _REQUIRED_ARRAYS, fail=fail)
    if not np.array_equal(arrays["selected_indices"], [0, 128, 256, 384]):
        raise fail("selected_indices do not match the shared presentation grid")
    if not np.all(arrays["ray_face_numbers"] == [3, 5]):
        raise fail("ray face sequence is not 3 to 5")


def load_data(data_root: Path | str) -> IllustrationData:
    data_root = Path(data_root)
    directory = data_directory(data_root)
    return load_pair(
        directory,
        fail=lambda reason: _fail(directory, reason, data_root),
        validate=lambda path, metadata, arrays, digest: _validate(
            path, metadata, arrays, digest, data_root=data_root
        ),
    )


def _tangential(vector: np.ndarray, normal: np.ndarray) -> np.ndarray:
    return vector - float(vector @ normal) * normal


def recompute_data(
    data_root: Path | str,
    canonical_directory: Path | str | None = None,
) -> IllustrationData:
    from _canonical_data import load_data as load_canonical_data
    from _canonical_data import recompute_data as recompute_canonical_data
    from lumice_integral.canonical_scene import CANONICAL_REFRACTIVE_INDEX, canonical_crystal
    from lumice_integral.geometry import Polyhedron
    from lumice_integral.geometry.entry_measure import entry_measure
    from lumice_integral.geometry.feasibility import corridor_intersection, corridor_polygons, perp_bases
    from lumice_integral.optics import path_direction

    data_root = Path(data_root)
    if canonical_directory is None:
        canonical_directory = recompute_canonical_data(data_root).directory
    common = load_canonical_data(canonical_directory)
    shared = common.arrays
    selected_indices = shared["selected_indices"]
    rotations = shared["presentation_rotations"][selected_indices]
    source = shared["source_world"]
    incident = shared["incident_direction"]
    target = shared["target_direction"]
    crystal = canonical_crystal()
    polygons, _ = corridor_polygons(crystal, [3, 5])

    ray_points = []
    segment_directions = []
    face_numbers = []
    face_margins = []
    direction_errors = []
    snell_errors = []
    for rotation in rotations:
        measure = entry_measure(
            rotation,
            [3, 5],
            incident,
            crystal,
            n_ice=CANONICAL_REFRACTIVE_INDEX,
        )
        if measure.status != "ok" or measure.internal_direction is None:
            raise RuntimeError(f"numbered pose has no valid 3-5 entry measure: {measure.status}")
        internal_body = measure.internal_direction
        basis_u, basis_w = perp_bases(internal_body[None, :])
        intersection = corridor_intersection(polygons, internal_body[None, :])
        count = int(intersection.count[0])
        if count < 3:
            raise RuntimeError("numbered pose has an empty representative-ray corridor")
        overlap = intersection.points[0, :count]
        origin_body = np.mean(overlap, axis=0) @ np.stack([basis_u[0], basis_w[0]])
        hit = crystal.intersect_ray(origin_body, internal_body)
        if hit is None:
            raise RuntimeError("representative internal line does not intersect the canonical crystal")
        t_entry, entry_face, t_exit, exit_face = hit
        if [entry_face.number, exit_face.number] != [3, 5]:
            raise RuntimeError(
                f"representative internal line hit {[entry_face.number, exit_face.number]}, expected [3, 5]"
            )
        entry_body = origin_body + t_entry * internal_body
        exit_body = origin_body + t_exit * internal_body
        margins = np.array(
            [crystal.face_margin(entry_face, entry_body), crystal.face_margin(exit_face, exit_body)]
        )
        if np.min(margins) <= 1e-6:
            raise RuntimeError(f"representative ray is too close to an edge: {np.min(margins):.3g}")

        entry_world = rotation @ entry_body
        exit_world = rotation @ exit_body
        internal_world = rotation @ internal_body
        points = np.stack(
            [entry_world - 1.05 * incident, entry_world, exit_world, exit_world + 1.05 * target]
        )
        directions = np.diff(points, axis=0)
        directions /= np.linalg.norm(directions, axis=1)[:, None]
        production = path_direction(
            rotation,
            [3, 5],
            incident,
            CANONICAL_REFRACTIVE_INDEX,
            crystal=crystal,
        )
        production_internal = np.asarray(production.entry.direction, dtype=np.float64)
        production_outgoing = np.asarray(production.direction, dtype=np.float64)
        errors = np.array(
            [
                np.linalg.norm(directions[0] - incident),
                np.linalg.norm(directions[1] - production_internal),
                np.linalg.norm(directions[2] - production_outgoing),
            ]
        )
        if np.max(errors) >= 1e-10 or np.linalg.norm(production_outgoing - target) >= 1e-10:
            raise RuntimeError(f"representative ray direction error {np.max(errors):.3g}")
        entry_normal = rotation @ crystal.normal(entry_face)
        exit_normal = rotation @ crystal.normal(exit_face)
        independent_snell = np.array(
            [
                np.linalg.norm(
                    _tangential(incident, entry_normal)
                    - CANONICAL_REFRACTIVE_INDEX * _tangential(internal_world, entry_normal)
                ),
                np.linalg.norm(
                    CANONICAL_REFRACTIVE_INDEX * _tangential(internal_world, exit_normal)
                    - _tangential(target, exit_normal)
                ),
            ]
        )
        if np.max(independent_snell) >= 1e-10:
            raise RuntimeError(f"independent Snell residual {np.max(independent_snell):.3g}")
        ray_points.append(points)
        segment_directions.append(directions)
        face_numbers.append([entry_face.number, exit_face.number])
        face_margins.append(margins)
        direction_errors.append(errors)
        snell_errors.append(independent_snell)

    vertices_body = shared["vertices_body"]
    normals_body = shared["face_normals_body"]
    centers_body = shared["face_centers_body"]
    arrays = {
        "u_curve": shared["presentation_u"],
        "relative_rotvec": shared["presentation_relative_rotvec"],
        "selected_indices": selected_indices,
        "selected_s": shared["selected_s"],
        "selected_rotations": rotations,
        "ray_points": np.asarray(ray_points),
        "ray_segment_directions": np.asarray(segment_directions),
        "ray_face_numbers": np.asarray(face_numbers, dtype=np.int64),
        "ray_face_margins": np.asarray(face_margins),
        "ray_direction_errors": np.asarray(direction_errors),
        "snell_tangential_errors": np.asarray(snell_errors),
        "source_world": source,
        "incident_direction": incident,
        "target_direction": target,
        "vertices_body": vertices_body,
        "vertices_world": np.stack([vertices_body @ rotation.T for rotation in rotations]),
        "face_numbers": shared["face_numbers"],
        "face_normals_world": np.stack([normals_body @ rotation.T for rotation in rotations]),
        "face_centers_world": np.stack([centers_body @ rotation.T for rotation in rotations]),
    }
    metadata = {
        "schema": SCHEMA_VERSION,
        "figure_id": FIGURE_ID,
        "fixture": common.metadata["fixture"],
        "geometry": common.metadata["geometry"],
        "source": {
            "canonical_data_sha256": common.metadata["payload"]["sha256"],
            "component_scope": common.metadata["source"]["component_scope"],
            "component_completeness": common.metadata["source"]["component_completeness"],
        },
        "ray_construction": {
            "path": [3, 5],
            "representative_point": "vertex mean of the convex production corridor intersection in the perpendicular plane",
            "physical_hits": "Polyhedron.intersect_ray on the body-frame internal line",
            "display_extension": "1.05 crystal-length units before entry and after exit; not part of the physical solve",
        },
        "validation": {
            "max_direction_error_l2": float(np.max(arrays["ray_direction_errors"])),
            "max_snell_tangential_error_l2": float(np.max(arrays["snell_tangential_errors"])),
            "minimum_hit_face_margin": float(np.min(arrays["ray_face_margins"])),
            "face_sequences": arrays["ray_face_numbers"],
        },
        "conventions": {
            "u": "u = R^T s_hat points toward the source",
            "incident": "orange ray follows incident propagation -s_hat",
            "outgoing": "blue ray follows outgoing propagation d; the apparent sky point is -d",
            "pose_coordinates": "log(R_0^T R) in radians; not fiber arclength",
        },
        "provenance": {
            "canonical_revision": common.metadata["provenance"]["lumice_integral"],
            "recompute_command": recompute_command(data_root),
        },
    }
    directory = data_directory(data_root)
    write_pair(
        directory,
        arrays,
        metadata,
        semantics=_ARRAY_SEMANTICS,
        validate=lambda path, saved_metadata, saved_arrays, digest: _validate(
            path, saved_metadata, saved_arrays, digest, data_root=data_root
        ),
    )
    return load_data(data_root)
