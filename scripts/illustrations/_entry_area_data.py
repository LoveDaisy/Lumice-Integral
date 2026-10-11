"""Production data for the canonical entry-area illustration."""

from __future__ import annotations

from pathlib import Path
import shlex
from typing import Any, Mapping

import numpy as np

from _illustration_io import (
    IllustrationData,
    IllustrationDataError,
    load_pair,
    validate_required_arrays,
    write_pair,
)


FIGURE_ID = "entry-area"
OUTPUT_NAME = f"{FIGURE_ID}.png"
SCHEMA_VERSION = "lumice-integral.illustration.entry-area/v1"

_REQUIRED_ARRAYS = {
    "curve_s": ((None,), "floating"),
    "curve_area_over_a2": ((None,), "floating"),
    "sample_s": ((3,), "floating"),
    "sample_rotations": ((3, 3, 3), "floating"),
    "projected_faces": ((3, 2, 4, 2), "floating"),
    "overlap": ((3, 4, 2), "floating"),
    "area_over_a2": ((3,), "floating"),
    "area_perp_over_a2": ((3,), "floating"),
    "internal_directions": ((3, 3), "floating"),
    "cosine_ratio": ((3,), "floating"),
    "upper_edges": ((3, 2, 2), "floating"),
    "support_faces": ((3, 2), "integer"),
    "normal_jacobian": ((3,), "floating"),
    "outgoing_errors": ((3,), "floating"),
}

_ARRAY_SEMANTICS = {
    "curve_s": ("radian", "Production numerical SO(3) arclength"),
    "curve_area_over_a2": ("dimensionless", "Production effective entry area divided by a squared"),
    "sample_s": ("radian", "Before, alignment, and after positions on the arclength ruler"),
    "sample_rotations": ("dimensionless", "Retracted rotations at the three sample positions"),
    "projected_faces": ("crystal length", "Entry and exit corridor faces projected perpendicular to the internal ray"),
    "overlap": ("crystal length", "Convex projected-face intersection A_perp"),
    "area_over_a2": ("dimensionless", "Effective incident-beam entry area divided by a squared"),
    "area_perp_over_a2": ("dimensionless", "Internal perpendicular overlap area divided by a squared"),
    "internal_directions": ("dimensionless", "Body-frame internal propagation directions"),
    "cosine_ratio": ("dimensionless", "cos(theta_i) / cos(theta_t)"),
    "upper_edges": ("crystal length", "Active upper edge, or coincident edge at B"),
    "support_faces": ("face id", "Upper and lower limiting face numbers"),
    "normal_jacobian": ("dimensionless", "Production J_perp at A, B, and C"),
    "outgoing_errors": ("dimensionless", "L2 outgoing-direction residual at A, B, and C"),
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
            str(Path(__file__).with_name("entry_area.py").resolve()),
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
    if len(arrays["curve_s"]) != len(arrays["curve_area_over_a2"]):
        raise fail("area curve arrays do not align")
    if not np.array_equal(arrays["support_faces"], [[5, 3], [3, 5], [3, 5]]):
        raise fail("support-face exchange record is not the canonical A/B/C pattern")


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


def _shoelace(points: np.ndarray) -> float:
    x, y = points.T
    return float(abs(x @ np.roll(y, -1) - y @ np.roll(x, -1)) / 2.0)


def _support_face(projected: np.ndarray, overlap: np.ndarray, *, upper: bool) -> tuple[int, np.ndarray, float]:
    mids = (overlap + np.roll(overlap, -1, axis=0)) / 2.0
    edge_index = int(np.argmax(mids[:, 1]) if upper else np.argmin(mids[:, 1]))
    midpoint = mids[edge_index]
    distances = []
    for face_index, polygon in enumerate(projected):
        for vertex in range(len(polygon)):
            start = polygon[vertex]
            end = polygon[(vertex + 1) % len(polygon)]
            edge = end - start
            offset = midpoint - start
            distance = abs(edge[0] * offset[1] - edge[1] * offset[0]) / np.linalg.norm(edge)
            distances.append((float(distance), face_index))
    distances.sort()
    return distances[0][1], overlap[[edge_index, (edge_index + 1) % len(overlap)]], distances[1][0]


def recompute_data(
    data_root: Path | str,
    canonical_directory: Path | str | None = None,
) -> IllustrationData:
    from scipy.optimize import brentq
    from scipy.spatial.transform import Rotation, Slerp

    from _canonical_data import load_data as load_canonical_data
    from _canonical_data import recompute_data as recompute_canonical_data
    from lumice_integral.canonical_scene import (
        CANONICAL_REFRACTIVE_INDEX,
        canonical_crystal,
        canonical_pixel_problem,
    )
    from lumice_integral.continuation import retract_to_fiber_batch
    from lumice_integral.geometry.entry_measure import entry_measure, entry_measure_batch
    from lumice_integral.geometry.feasibility import corridor_intersection, corridor_polygons, perp_bases
    import jax

    data_root = Path(data_root)
    if canonical_directory is None:
        canonical_directory = recompute_canonical_data(data_root).directory
    common = load_canonical_data(canonical_directory)
    shared = common.arrays
    rotations = shared["presentation_rotations"]
    s = shared["presentation_s"]
    problem = canonical_pixel_problem()
    crystal = canonical_crystal()
    a = float(crystal.a)
    incident = shared["incident_direction"]
    target = shared["target_direction"]
    interpolator = Slerp(s, Rotation.from_matrix(rotations))

    def pose_at(position: float) -> tuple[np.ndarray, float, float]:
        index = min(max(int(np.searchsorted(s, position) - 1), 0), len(s) - 2)
        tangent = Rotation.from_matrix(rotations[index].T @ rotations[index + 1]).as_rotvec()
        tangent /= np.linalg.norm(tangent)
        predicted = interpolator([position]).as_matrix()
        result = retract_to_fiber_batch(problem, predicted, tangent[None, :], iterations=3)
        if not np.all(result.finite) or np.max(result.residual_after) >= 1e-11:
            raise RuntimeError("entry-area interpolation failed production retraction")
        return result.rotations[0], float(result.normal_jacobians[0]), float(result.residual_after[0])

    def geometry(rotation: np.ndarray):
        result = entry_measure(
            rotation,
            [3, 5],
            incident,
            crystal,
            n_ice=CANONICAL_REFRACTIVE_INDEX,
        )
        if result.status != "ok" or result.internal_direction is None:
            raise RuntimeError(f"entry-area sample is not feasible: {result.status}")
        polygons, _ = corridor_polygons(crystal, [3, 5])
        basis_u, basis_w = perp_bases(result.internal_direction[None, :])
        basis = np.stack([basis_u[0], basis_w[0]], axis=-1)
        projected = np.asarray([polygon @ basis for polygon in polygons])
        intersection = corridor_intersection(polygons, result.internal_direction[None, :])
        overlap = intersection.points[0, : intersection.count[0]]
        if overlap.shape != (4, 2):
            raise RuntimeError(f"canonical entry-area overlap changed shape to {overlap.shape}")
        return result, projected, overlap

    internal_z = np.array(
        [
            entry_measure(
                rotation,
                [3, 5],
                incident,
                crystal,
                n_ice=CANONICAL_REFRACTIVE_INDEX,
            ).internal_direction[2]
            for rotation in rotations
        ]
    )
    crossings = np.flatnonzero(internal_z[:-1] * internal_z[1:] <= 0.0)
    if len(crossings) == 0:
        raise RuntimeError("no internal-direction z crossing exists on the canonical component")
    crossing = int(crossings[np.argmin(abs((s[crossings] + s[crossings + 1]) / 2.0 - 1.91))])

    def root_function(position: float) -> float:
        rotation, _, _ = pose_at(position)
        return float(geometry(rotation)[0].internal_direction[2])

    root = brentq(root_function, float(s[crossing]), float(s[crossing + 1]), xtol=2e-13)
    sample_s = np.array([root - 0.20, root, root + 0.20])
    pose_records = [pose_at(position) for position in sample_s]
    sample_rotations = np.asarray([record[0] for record in pose_records])
    jacobians = np.asarray([record[1] for record in pose_records])
    records = [geometry(rotation) for rotation in sample_rotations]

    projected_faces = np.asarray([record[1] for record in records])
    overlaps = np.asarray([record[2] for record in records])
    results = [record[0] for record in records]
    upper = [_support_face(projected, overlap, upper=True) for projected, overlap in zip(projected_faces, overlaps)]
    lower = [_support_face(projected, overlap, upper=False) for projected, overlap in zip(projected_faces, overlaps)]
    upper_faces = [item[0] for item in upper]
    lower_faces = [item[0] for item in lower]
    if upper_faces[0] != 1 or upper_faces[2] != 0 or lower_faces[0] != 0 or lower_faces[2] != 1:
        raise RuntimeError(
            "canonical entry-area support faces no longer exchange around the selected root: "
            f"upper={upper_faces}, lower={lower_faces}"
        )
    if upper[1][2] >= 1e-10 or lower[1][2] >= 1e-10:
        raise RuntimeError("entry and exit boundaries no longer coincide at the selected root")

    area = np.asarray([result.value for result in results])
    area_perp = np.asarray([result.area_perp_internal for result in results])
    ratio = np.asarray([result.cosine_incident / result.cosine_internal for result in results])
    shoelace_errors = np.asarray(
        [abs(_shoelace(overlap) - result.area_perp_internal) for result, overlap in zip(results, overlaps)]
    )
    conversion_errors = np.abs(area_perp * ratio - area)
    if np.max(shoelace_errors) >= 1e-12 * a**2 or np.max(conversion_errors) >= 1e-10 * a**2:
        raise RuntimeError("entry-area projection or cosine conversion check failed")

    production_area = entry_measure_batch(
        shared["quadrature_rotations"],
        [3, 5],
        incident,
        crystal,
        n_ice=CANONICAL_REFRACTIVE_INDEX,
    )
    full_loop_error = float(np.max(np.abs(production_area - shared["entry_measure"])))
    if full_loop_error >= 1e-10 * a**2:
        raise RuntimeError("entry-area batch disagrees with the shared production factor")
    outgoing = np.asarray(jax.vmap(problem.direction_evaluator)(sample_rotations))
    outgoing_errors = np.linalg.norm(outgoing - target, axis=1)
    if np.max(outgoing_errors) >= 1e-10:
        raise RuntimeError("entry-area sample rotations left the canonical fiber")

    slopes = []
    center_area = area[1]
    for step in (0.002, 0.001):
        left = geometry(pose_at(root - step)[0])[0].value
        right = geometry(pose_at(root + step)[0])[0].value
        slopes.append([(center_area - left) / step, (right - center_area) / step])
    slopes = np.asarray(slopes)
    if slopes[-1, 0] <= 0.1 or slopes[-1, 1] >= -0.1:
        raise RuntimeError("entry-area one-sided slopes no longer identify a local kink")
    if abs(results[1].internal_direction[2]) >= 1e-10 or jacobians[1] <= 1e-3:
        raise RuntimeError("entry-area alignment is missing or became a Jacobian critical point")

    arrays = {
        "curve_s": shared["quadrature_s"],
        "curve_area_over_a2": shared["entry_measure"] / a**2,
        "sample_s": sample_s,
        "sample_rotations": sample_rotations,
        "projected_faces": projected_faces,
        "overlap": overlaps,
        "area_over_a2": area / a**2,
        "area_perp_over_a2": area_perp / a**2,
        "internal_directions": np.asarray([result.internal_direction for result in results]),
        "cosine_ratio": ratio,
        "upper_edges": np.asarray([item[1] for item in upper]),
        "support_faces": np.asarray([[5, 3], [3, 5], [3, 5]], dtype=np.int64),
        "normal_jacobian": jacobians,
        "outgoing_errors": outgoing_errors,
    }
    metadata = {
        "schema": SCHEMA_VERSION,
        "figure_id": FIGURE_ID,
        "fixture": common.metadata["fixture"],
        "source": {
            "canonical_data_sha256": common.metadata["payload"]["sha256"],
            "component_scope": common.metadata["source"]["component_scope"],
            "component_completeness": common.metadata["source"]["component_completeness"],
        },
        "event": {
            "definition": "internal_direction.z = 0, where projected basal-edge levels coincide",
            "s_rad": root,
            "bracket_s_rad": [float(s[crossing]), float(s[crossing + 1])],
            "sample_spacing_rad": 0.20,
            "interpolation": "Slerp on the production presentation rotations followed by three production retraction iterations",
        },
        "validation": {
            "event_internal_z": float(arrays["internal_directions"][1, 2]),
            "one_sided_area_slopes_a2_per_rad": slopes,
            "slope_probe_steps_rad": [0.002, 0.001],
            "max_shoelace_area_error_a2": float(np.max(shoelace_errors) / a**2),
            "max_cosine_conversion_error_a2": float(np.max(conversion_errors) / a**2),
            "max_area_difference_vs_production_a2": full_loop_error / a**2,
            "max_outgoing_direction_error_l2": float(np.max(outgoing_errors)),
            "normal_jacobian_at_event": float(jacobians[1]),
            "upper_support_before_after": [5, 3],
            "lower_support_before_after": [3, 5],
        },
        "conventions": {
            "projection": "orthogonal along the internal ray; common physical scale and production perp_bases",
            "area": "A_perp is the projected overlap; A = A_perp cos(theta_i) / cos(theta_t)",
            "s": "production numerical arclength used as the interpolation coordinate",
            "scope": "B is one clipping-boundary alignment, not a Jacobian critical point or a claim about all kinks",
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
