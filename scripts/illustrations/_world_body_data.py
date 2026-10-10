"""Self-contained data contract for the world/body coordinate-frame illustration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any, Mapping

import numpy as np


FIGURE_ID = "world-body-frames"
OUTPUT_NAME = f"{FIGURE_ID}.png"
SCHEMA_VERSION = "lumice-integral.illustration.world-body-frames/v1"
ARRAYS_FILE = "arrays.npz"
METADATA_FILE = "metadata.json"
DENSE_NODE_COUNT = 513
RETRACTION_ITERATIONS = 2
SELECTED_INDICES = np.array([0, 128, 256, 384], dtype=np.int64)
ORIGINAL_ACCEPTED_REVISION = "98b4a8c7ec70e519f3744279840c2814f871ade7"
ORIGINAL_PROTOTYPE_SHA256 = "eb8bb4850835faeb2442a5f25bee1fd9271bcc887d077a62e7999b17bd8c6a62"
ORIGINAL_ACCEPTED_IMAGE_SHA256 = "e0fe35655484acd6b76ced2d9286e1b03e25340211c13e64eb99bc0695add81e"
WRITING_VISUAL_REFERENCE_REVISION = "b7784b8c96c742ee4f49f2277fe424bfda213ae4"

_REQUIRED_ARRAYS = {
    "selected_dense_indices": ((4,), "integer"),
    "rotations": ((4, 3, 3), "floating"),
    "source_world": ((3,), "floating"),
    "source_body": ((4, 3), "floating"),
    "target_direction": ((3,), "floating"),
    "outgoing_directions": ((4, 3), "floating"),
    "vertices_body": ((12, 3), "floating"),
    "vertices_world": ((4, 12, 3), "floating"),
    "face_numbers": (None, "integer"),
    "face_normals_body": (None, "floating"),
    "face_normals_world": (None, "floating"),
    "face_centers_body": (None, "floating"),
    "face_centers_world": (None, "floating"),
}

_ARRAY_SEMANTICS = {
    "selected_dense_indices": ("index", "Indices in the 513-point presentation resampling grid"),
    "rotations": ("dimensionless", "Body-to-world rotation matrices R_i"),
    "source_world": ("dimensionless", "Unit direction toward the source in world coordinates"),
    "source_body": ("dimensionless", "Unit directions u_i = R_i^T source_world"),
    "target_direction": ("dimensionless", "Canonical outgoing propagation direction"),
    "outgoing_directions": ("dimensionless", "Production halo-map directions at the selected poses"),
    "vertices_body": ("crystal length", "Canonical crystal vertices in body coordinates"),
    "vertices_world": ("crystal length", "Canonical crystal vertices transformed by each R_i"),
    "face_numbers": ("face id", "Production crystal face numbers in payload order"),
    "face_normals_body": ("dimensionless", "Production outward face normals in body coordinates"),
    "face_normals_world": ("dimensionless", "Outward face normals transformed by each R_i"),
    "face_centers_body": ("crystal length", "Production face centroids in body coordinates"),
    "face_centers_world": ("crystal length", "Face centroids transformed by each R_i"),
}


class IllustrationDataError(ValueError):
    pass


@dataclass(frozen=True)
class IllustrationData:
    directory: Path
    metadata: Mapping[str, Any]
    arrays: Mapping[str, np.ndarray]


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
            str(Path(__file__).with_name("world_body_frames.py").resolve()),
            "--recompute",
            "--data-dir",
            str(Path(data_root).resolve()),
        ]
    )


def _fail(directory: Path, reason: str, data_root: Path) -> IllustrationDataError:
    return IllustrationDataError(
        f"invalid {FIGURE_ID} data at {directory}: {reason}. "
        f"Recompute with: {recompute_command(data_root)}"
    )


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"metadata cannot encode {type(value).__name__}")


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _git_state(root: Path) -> dict[str, Any]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, check=True, capture_output=True, text=True
        ).stdout.strip()
    )
    return {"revision": revision, "tracked_or_untracked_dirty": dirty}


def _array_metadata(arrays: Mapping[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "unit": _ARRAY_SEMANTICS[name][0],
            "semantic": _ARRAY_SEMANTICS[name][1],
        }
        for name, array in arrays.items()
    }


def _validate_pair(
    directory: Path,
    metadata: Any,
    arrays: Mapping[str, np.ndarray],
    *,
    data_root: Path,
    arrays_sha256: str,
) -> None:
    if not isinstance(metadata, Mapping):
        raise _fail(directory, "metadata.json root must be an object", data_root)
    if metadata.get("schema") != SCHEMA_VERSION:
        raise _fail(directory, f"unsupported schema {metadata.get('schema')!r}", data_root)
    if metadata.get("figure_id") != FIGURE_ID:
        raise _fail(directory, f"unexpected figure_id {metadata.get('figure_id')!r}", data_root)
    payload = metadata.get("payload")
    if not isinstance(payload, Mapping) or payload.get("file") != ARRAYS_FILE:
        raise _fail(directory, "payload file declaration is missing or wrong", data_root)
    if payload.get("sha256") != arrays_sha256:
        raise _fail(directory, "arrays.npz SHA-256 does not match metadata", data_root)
    missing_arrays = sorted(set(_REQUIRED_ARRAYS) - set(arrays))
    if missing_arrays:
        raise _fail(directory, f"missing required arrays {missing_arrays}", data_root)

    for name, (shape, kind) in _REQUIRED_ARRAYS.items():
        array = arrays[name]
        if shape is not None and array.shape != shape:
            raise _fail(directory, f"{name} has shape {array.shape}, expected {shape}", data_root)
        if kind == "integer" and array.dtype.kind not in "iu":
            raise _fail(directory, f"{name} must use an integer dtype", data_root)
        if kind == "floating" and array.dtype.kind != "f":
            raise _fail(directory, f"{name} must use a floating dtype", data_root)
        if not np.all(np.isfinite(array)):
            raise _fail(directory, f"{name} contains non-finite values", data_root)

    face_numbers = arrays["face_numbers"]
    if (
        face_numbers.ndim != 1
        or len(face_numbers) == 0
        or len(set(face_numbers.tolist())) != len(face_numbers)
    ):
        raise _fail(directory, "face_numbers must be a non-empty unique one-dimensional array", data_root)
    face_count = len(face_numbers)
    face_shapes = {
        "face_normals_body": (face_count, 3),
        "face_normals_world": (4, face_count, 3),
        "face_centers_body": (face_count, 3),
        "face_centers_world": (4, face_count, 3),
    }
    for name, shape in face_shapes.items():
        if arrays[name].shape != shape:
            raise _fail(directory, f"{name} has shape {arrays[name].shape}, expected {shape}", data_root)

    geometry = metadata.get("geometry")
    if not isinstance(geometry, Mapping):
        raise _fail(directory, "geometry declaration is missing", data_root)
    face_vertex_ids = geometry.get("face_vertex_ids")
    if not isinstance(face_vertex_ids, list) or len(face_vertex_ids) != face_count:
        raise _fail(directory, "face_vertex_ids do not match the face count", data_root)
    vertex_count = len(arrays["vertices_body"])
    if any(
        not isinstance(ids, list)
        or len(ids) < 3
        or len(set(ids)) != len(ids)
        or any(
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or index >= vertex_count
            for index in ids
        )
        for ids in face_vertex_ids
    ):
        raise _fail(directory, "face_vertex_ids contain invalid or repeated vertex indices", data_root)

    edges = geometry.get("edges")
    if not isinstance(edges, list) or not edges:
        raise _fail(directory, "edges must be a non-empty list", data_root)
    for edge in edges:
        if (
            not isinstance(edge, list)
            or len(edge) != 2
            or any(
                isinstance(index, bool)
                or not isinstance(index, int)
                or index < 0
                or index >= vertex_count
                for index in edge
            )
            or edge[0] == edge[1]
        ):
            raise _fail(directory, "edges contain invalid or repeated vertex indices", data_root)
        if not any(set(edge).issubset(ids) for ids in face_vertex_ids):
            raise _fail(directory, f"edge {tuple(edge)} has no adjacent face", data_root)


def load_data(data_root: Path | str) -> IllustrationData:
    data_root = Path(data_root)
    directory = data_directory(data_root)
    metadata_path = directory / METADATA_FILE
    arrays_path = directory / ARRAYS_FILE
    if not metadata_path.is_file() or not arrays_path.is_file():
        missing = [path.name for path in (metadata_path, arrays_path) if not path.is_file()]
        raise _fail(directory, f"missing {', '.join(missing)}", data_root)
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise _fail(directory, f"cannot read metadata.json ({error})", data_root) from error
    try:
        with np.load(arrays_path, allow_pickle=False) as payload:
            arrays = {name: np.array(payload[name], copy=True) for name in payload.files}
    except (OSError, ValueError) as error:
        raise _fail(directory, f"cannot read arrays.npz ({error})", data_root) from error
    try:
        arrays_hash = sha256(arrays_path.read_bytes()).hexdigest()
    except OSError as error:
        raise _fail(directory, f"cannot hash arrays.npz ({error})", data_root) from error
    _validate_pair(directory, metadata, arrays, data_root=data_root, arrays_sha256=arrays_hash)
    return IllustrationData(directory=directory, metadata=metadata, arrays=arrays)


def _write_data(data_root: Path, arrays: Mapping[str, np.ndarray], metadata: Mapping[str, Any]) -> None:
    directory = data_directory(data_root)
    directory.mkdir(parents=True, exist_ok=True)
    temporary_arrays = directory / ".arrays.npz.tmp"
    temporary_metadata = directory / ".metadata.json.tmp"
    with temporary_arrays.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    arrays_hash = sha256(temporary_arrays.read_bytes()).hexdigest()
    complete_metadata = {
        **metadata,
        "payload": {
            "file": ARRAYS_FILE,
            "sha256": arrays_hash,
            "arrays": _array_metadata(arrays),
        },
    }
    temporary_metadata.write_text(
        json.dumps(_json_value(complete_metadata), allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with np.load(temporary_arrays, allow_pickle=False) as payload:
        check_arrays = {name: np.array(payload[name], copy=True) for name in payload.files}
    check_metadata = json.loads(temporary_metadata.read_text(encoding="utf-8"))
    _validate_pair(directory, check_metadata, check_arrays, data_root=data_root, arrays_sha256=arrays_hash)
    temporary_arrays.replace(directory / ARRAYS_FILE)
    temporary_metadata.replace(directory / METADATA_FILE)


def recompute_data(data_root: Path | str) -> IllustrationData:
    from lumice_integral.canonical_scene import (
        canonical_crystal,
        canonical_fixture_metadata,
        canonical_pixel_problem,
        canonical_sun_direction,
        canonical_target_direction,
    )
    from lumice_integral.continuation import (
        ContinuationOptions,
        FiberStatus,
        retract_to_fiber_batch,
        trace_fiber,
    )
    from lumice_integral.figure_data import SCHEMA_VERSION as FIBER_SCHEMA_VERSION
    from lumice_integral.figure_data import export_fiber_figure_data
    from lumice_integral.resample import resample_fiber
    from lumice_integral.weights import evaluate_weights_batch
    import jax

    data_root = Path(data_root)
    root = Path(__file__).resolve().parents[2]
    problem = canonical_pixel_problem()
    options = ContinuationOptions()
    traced = trace_fiber(problem, options)
    if traced.status is not FiberStatus.CLOSED:
        raise RuntimeError(f"canonical fiber did not close: {traced.status.value}: {traced.terminal_payload.message}")

    upstream_directory = data_root / "canonical-fiber"
    upstream = export_fiber_figure_data(
        traced,
        upstream_directory,
        fixture=canonical_fixture_metadata(),
        provenance={"purpose": f"upstream geometry for {FIGURE_ID}"},
    )
    upstream_metadata = json.loads(upstream.metadata.read_text(encoding="utf-8"))
    if upstream_metadata["schema"] != FIBER_SCHEMA_VERSION:
        raise RuntimeError(f"unexpected upstream fiber schema {upstream_metadata['schema']!r}")

    predictors = resample_fiber(traced, DENSE_NODE_COUNT)
    retraction = retract_to_fiber_batch(
        problem,
        predictors.rotations,
        predictors.phase_tangents,
        iterations=RETRACTION_ITERATIONS,
    )
    rotations_dense = np.asarray(retraction.rotations, dtype=np.float64)
    if not np.all(retraction.finite):
        raise RuntimeError(f"{np.count_nonzero(~retraction.finite)} retracted nodes are non-finite")
    max_retraction_residual = float(np.max(retraction.residual_after))
    if max_retraction_residual > 1e-10:
        raise RuntimeError(f"retracted-node residual {max_retraction_residual:.3g} exceeds 1e-10")
    validity = evaluate_weights_batch(problem.weight_evaluators, rotations_dense, ("path_validity",)).values[
        "path_validity"
    ]
    if not np.all(validity == 1.0):
        raise RuntimeError(f"{np.count_nonzero(validity != 1.0)} retracted nodes fail production path validity")

    outgoing_dense = np.asarray(jax.vmap(problem.direction_evaluator)(rotations_dense), dtype=np.float64)
    target = np.asarray(canonical_target_direction(), dtype=np.float64)
    max_outgoing_residual = float(np.max(np.linalg.norm(outgoing_dense - target, axis=1)))
    if max_outgoing_residual > 1e-10:
        raise RuntimeError(f"outgoing-direction residual {max_outgoing_residual:.3g} exceeds 1e-10")

    rotations = rotations_dense[SELECTED_INDICES]
    source_world = np.asarray(canonical_sun_direction(), dtype=np.float64)
    source_body = np.einsum("nji,j->ni", rotations, source_world)
    crystal = canonical_crystal()
    vertices_body = np.asarray(crystal.vertices, dtype=np.float64)
    face_numbers = np.array([face.number for face in crystal.faces], dtype=np.int64)
    face_normals_body = np.stack([crystal.normal(face) for face in crystal.faces])
    face_centers_body = np.stack([crystal.centroid(face) for face in crystal.faces])
    arrays = {
        "selected_dense_indices": SELECTED_INDICES,
        "rotations": rotations,
        "source_world": source_world,
        "source_body": source_body,
        "target_direction": target,
        "outgoing_directions": outgoing_dense[SELECTED_INDICES],
        "vertices_body": vertices_body,
        "vertices_world": np.stack([vertices_body @ rotation.T for rotation in rotations]),
        "face_numbers": face_numbers,
        "face_normals_body": face_normals_body,
        "face_normals_world": np.stack([face_normals_body @ rotation.T for rotation in rotations]),
        "face_centers_body": face_centers_body,
        "face_centers_world": np.stack([face_centers_body @ rotation.T for rotation in rotations]),
    }
    source_body_definition_error = float(
        np.max(np.abs(source_body - np.einsum("nji,j->ni", rotations, source_world)))
    )
    source_body_round_trip_error = float(
        np.max(np.abs(np.einsum("nij,nj->ni", rotations, source_body) - source_world))
    )
    vertices_round_trip_error = float(
        max(
            np.max(np.abs(world @ rotation - vertices_body))
            for world, rotation in zip(arrays["vertices_world"], rotations)
        )
    )
    normals_round_trip_error = float(
        max(
            np.max(np.abs(world @ rotation - face_normals_body))
            for world, rotation in zip(arrays["face_normals_world"], rotations)
        )
    )
    centers_round_trip_error = float(
        max(
            np.max(np.abs(world @ rotation - face_centers_body))
            for world, rotation in zip(arrays["face_centers_world"], rotations)
        )
    )
    rotation_orthogonality_error = float(
        np.max(np.abs(rotations.transpose(0, 2, 1) @ rotations - np.eye(3)))
    )
    rotation_determinant_error = float(np.max(np.abs(np.linalg.det(rotations) - 1.0)))
    source_world_norm_error = float(abs(np.linalg.norm(source_world) - 1.0))
    target_direction_norm_error = float(abs(np.linalg.norm(target) - 1.0))
    outgoing_direction_norm_error = float(
        np.max(np.abs(np.linalg.norm(outgoing_dense[SELECTED_INDICES], axis=1) - 1.0))
    )
    face_normal_norm_error = float(
        np.max(np.abs(np.linalg.norm(face_normals_body, axis=1) - 1.0))
    )
    if max(rotation_orthogonality_error, rotation_determinant_error) > 1e-10:
        raise RuntimeError("selected presentation rotations are not elements of SO(3)")
    if max(
        source_body_definition_error,
        source_body_round_trip_error,
        vertices_round_trip_error,
        normals_round_trip_error,
        centers_round_trip_error,
    ) > 1e-12:
        raise RuntimeError("selected presentation coordinates do not round-trip within 1e-12")
    if max(
        source_world_norm_error,
        target_direction_norm_error,
        outgoing_direction_norm_error,
        face_normal_norm_error,
    ) > 1e-12:
        raise RuntimeError("selected presentation directions are not unit vectors within 1e-12")

    coordinate_checks = {
        "source_body_definition_max_abs": source_body_definition_error,
        "source_body_round_trip_max_abs": source_body_round_trip_error,
        "vertices_round_trip_max_abs": vertices_round_trip_error,
        "face_normals_round_trip_max_abs": normals_round_trip_error,
        "face_centers_round_trip_max_abs": centers_round_trip_error,
        "rotation_orthogonality_max_abs": rotation_orthogonality_error,
        "rotation_determinant_max_abs": rotation_determinant_error,
        "source_world_norm_error": source_world_norm_error,
        "target_direction_norm_error": target_direction_norm_error,
        "outgoing_direction_norm_max_abs": outgoing_direction_norm_error,
        "face_normal_norm_max_abs": face_normal_norm_error,
        "dense_outgoing_direction_max_l2": max_outgoing_residual,
        "dense_retraction_residual_max_l2": max_retraction_residual,
        "dense_minimum_path_validity": float(np.min(validity)),
    }
    metadata = {
        "schema": SCHEMA_VERSION,
        "figure_id": FIGURE_ID,
        "fixture": upstream_metadata["fixture"],
        "sampling": {
            "meaning": "uniform cumulative-chord Hermite presentation grid, then fixed production retraction",
            "dense_node_count": DENSE_NODE_COUNT,
            "retraction_iterations": RETRACTION_ITERATIONS,
            "selected_indices": SELECTED_INDICES,
            "claim": "presentation sampling; not a new quadrature-convergence claim",
        },
        "conventions": {
            "rotation": "R maps body coordinates to world coordinates",
            "source_world": "s_hat points toward the Sun",
            "incident_propagation": "incident_direction = -s_hat",
            "source_body": "u_i = R_i^T s_hat",
        },
        "geometry": {
            "face_numbers": face_numbers,
            "face_vertex_ids": [list(face.vertex_ids) for face in crystal.faces],
            "edges": [list(edge) for edge in crystal.edges],
        },
        "validation": coordinate_checks,
        "source": {
            "upstream_schema": upstream_metadata["schema"],
            "upstream_relative_directory": "../canonical-fiber",
            "upstream_metadata_sha256": sha256(upstream.metadata.read_bytes()).hexdigest(),
            "upstream_arrays_sha256": upstream_metadata["payload"]["sha256"],
            "component_scope": traced.component_scope,
            "component_completeness": traced.component_completeness,
        },
        "provenance": {
            "lumice_integral": _git_state(root),
            "python": sys.version,
            "packages": {
                name: _package_version(name) for name in ("jax", "numpy", "scipy")
            },
            "continuation_options": asdict(options),
            "original_accepted_revision": ORIGINAL_ACCEPTED_REVISION,
            "original_prototype_sha256": ORIGINAL_PROTOTYPE_SHA256,
            "original_accepted_image_sha256": ORIGINAL_ACCEPTED_IMAGE_SHA256,
            "visual_reference": {
                "repository": "Writing-Lab/modern-ice-halo-series",
                "revision": WRITING_VISUAL_REFERENCE_REVISION,
                "paths": ["halo_notes/draw/projection.py", "halo_notes/draw/scene.py"],
                "use": "projection and crystal-visibility behavior only; no runtime dependency or copied framework",
            },
            "recompute_command": recompute_command(data_root),
        },
    }
    _write_data(data_root, arrays, metadata)
    return load_data(data_root)
