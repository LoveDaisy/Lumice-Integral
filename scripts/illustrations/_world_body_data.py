"""Self-contained data contract for the world/body coordinate-frame illustration."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import shlex
from typing import Any, Mapping
from zipfile import BadZipFile

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
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise _fail(directory, f"cannot read metadata.json ({error})", data_root) from error
    try:
        with np.load(arrays_path, allow_pickle=False) as payload:
            arrays = {name: np.array(payload[name], copy=True) for name in payload.files}
    except (OSError, EOFError, ValueError, BadZipFile) as error:
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


def _recompute_from_canonical(
    data_root: Path,
    canonical_directory: Path,
) -> IllustrationData:
    from _canonical_data import load_data as load_canonical_data

    common = load_canonical_data(canonical_directory)
    common_arrays = common.arrays
    common_metadata = common.metadata
    rotations = common_arrays["presentation_rotations"][SELECTED_INDICES]
    source_world = common_arrays["source_world"]
    source_body = common_arrays["presentation_u"][SELECTED_INDICES]
    vertices_body = common_arrays["vertices_body"]
    face_numbers = common_arrays["face_numbers"]
    face_normals_body = common_arrays["face_normals_body"]
    face_centers_body = common_arrays["face_centers_body"]
    target = common_arrays["target_direction"]
    arrays = {
        "selected_dense_indices": SELECTED_INDICES,
        "rotations": rotations,
        "source_world": source_world,
        "source_body": source_body,
        "target_direction": target,
        "outgoing_directions": common_arrays["presentation_outgoing"][SELECTED_INDICES],
        "vertices_body": vertices_body,
        "vertices_world": np.stack([vertices_body @ rotation.T for rotation in rotations]),
        "face_numbers": face_numbers,
        "face_normals_body": face_normals_body,
        "face_normals_world": np.stack([face_normals_body @ rotation.T for rotation in rotations]),
        "face_centers_body": face_centers_body,
        "face_centers_world": np.stack([face_centers_body @ rotation.T for rotation in rotations]),
    }
    coordinate_checks = {
        "source_body_definition_max_abs": float(
            np.max(np.abs(source_body - np.einsum("nji,j->ni", rotations, source_world)))
        ),
        "source_body_round_trip_max_abs": float(
            np.max(np.abs(np.einsum("nij,nj->ni", rotations, source_body) - source_world))
        ),
        "vertices_round_trip_max_abs": float(
            max(
                np.max(np.abs(world @ rotation - vertices_body))
                for world, rotation in zip(arrays["vertices_world"], rotations)
            )
        ),
        "face_normals_round_trip_max_abs": float(
            max(
                np.max(np.abs(world @ rotation - face_normals_body))
                for world, rotation in zip(arrays["face_normals_world"], rotations)
            )
        ),
        "face_centers_round_trip_max_abs": float(
            max(
                np.max(np.abs(world @ rotation - face_centers_body))
                for world, rotation in zip(arrays["face_centers_world"], rotations)
            )
        ),
        "rotation_orthogonality_max_abs": float(
            np.max(np.abs(rotations.transpose(0, 2, 1) @ rotations - np.eye(3)))
        ),
        "rotation_determinant_max_abs": float(np.max(np.abs(np.linalg.det(rotations) - 1.0))),
        "source_world_norm_error": float(abs(np.linalg.norm(source_world) - 1.0)),
        "target_direction_norm_error": float(abs(np.linalg.norm(target) - 1.0)),
        "outgoing_direction_norm_max_abs": float(
            np.max(np.abs(np.linalg.norm(arrays["outgoing_directions"], axis=1) - 1.0))
        ),
        "face_normal_norm_max_abs": float(
            np.max(np.abs(np.linalg.norm(face_normals_body, axis=1) - 1.0))
        ),
        "dense_outgoing_direction_max_l2": common_metadata["validation"][
            "presentation_outgoing_direction_max_l2"
        ],
        "dense_retraction_residual_max_l2": common_metadata["validation"][
            "presentation_retraction_residual_max_l2"
        ],
        "dense_minimum_path_validity": common_metadata["validation"]["minimum_path_validity"],
    }
    if max(
        coordinate_checks["source_body_definition_max_abs"],
        coordinate_checks["source_body_round_trip_max_abs"],
        coordinate_checks["vertices_round_trip_max_abs"],
        coordinate_checks["face_normals_round_trip_max_abs"],
        coordinate_checks["face_centers_round_trip_max_abs"],
    ) > 1e-12:
        raise RuntimeError("selected presentation coordinates do not round-trip within 1e-12")
    if max(
        coordinate_checks["rotation_orthogonality_max_abs"],
        coordinate_checks["rotation_determinant_max_abs"],
    ) > 1e-10:
        raise RuntimeError("selected presentation rotations are not elements of SO(3)")
    if max(
        coordinate_checks["source_world_norm_error"],
        coordinate_checks["target_direction_norm_error"],
        coordinate_checks["outgoing_direction_norm_max_abs"],
        coordinate_checks["face_normal_norm_max_abs"],
    ) > 1e-12:
        raise RuntimeError("selected presentation directions are not unit vectors within 1e-12")

    metadata = {
        "schema": SCHEMA_VERSION,
        "figure_id": FIGURE_ID,
        "fixture": common_metadata["fixture"],
        "sampling": {
            "meaning": "uniform cumulative-chord Hermite presentation grid, then fixed production retraction",
            "dense_node_count": DENSE_NODE_COUNT,
            "retraction_iterations": RETRACTION_ITERATIONS,
            "selected_indices": SELECTED_INDICES,
            "claim": "presentation sampling; not a new quadrature-convergence claim",
        },
        "conventions": common_metadata["conventions"],
        "geometry": common_metadata["geometry"],
        "validation": coordinate_checks,
        "source": {
            "upstream_schema": common_metadata["source"]["upstream_schema"],
            "upstream_relative_directory": "../canonical-fiber",
            "upstream_metadata_sha256": common_metadata["source"]["upstream_metadata_sha256"],
            "upstream_arrays_sha256": common_metadata["source"]["upstream_arrays_sha256"],
            "canonical_data_relative_directory": "../canonical-data",
            "canonical_data_sha256": common_metadata["payload"]["sha256"],
            "component_scope": common_metadata["source"]["component_scope"],
            "component_completeness": common_metadata["source"]["component_completeness"],
        },
        "provenance": {
            **common_metadata["provenance"],
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


def recompute_data(
    data_root: Path | str,
    canonical_directory: Path | str | None = None,
) -> IllustrationData:
    from _canonical_data import recompute_data as recompute_canonical_data

    data_root = Path(data_root)
    if canonical_directory is None:
        canonical_directory = recompute_canonical_data(data_root).directory
    return _recompute_from_canonical(data_root, Path(canonical_directory))
