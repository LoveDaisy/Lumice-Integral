"""Shared JSON/NPZ helpers for rebuildable illustration caches."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Callable, Mapping
from zipfile import BadZipFile

import numpy as np


class IllustrationDataError(ValueError):
    pass


@dataclass(frozen=True)
class IllustrationData:
    directory: Path
    metadata: Mapping[str, Any]
    arrays: Mapping[str, np.ndarray]


def json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TypeError(f"metadata cannot encode {type(value).__name__}")


def array_metadata(
    arrays: Mapping[str, np.ndarray],
    semantics: Mapping[str, tuple[str, str]],
) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "unit": semantics[name][0],
            "semantic": semantics[name][1],
        }
        for name, array in arrays.items()
    }


def load_pair(
    directory: Path,
    *,
    fail: Callable[[str], IllustrationDataError],
    validate: Callable[[Path, Any, Mapping[str, np.ndarray], str], None],
) -> IllustrationData:
    metadata_path = directory / "metadata.json"
    arrays_path = directory / "arrays.npz"
    if not metadata_path.is_file() or not arrays_path.is_file():
        missing = [path.name for path in (metadata_path, arrays_path) if not path.is_file()]
        raise fail(f"missing {', '.join(missing)}")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise fail(f"cannot read metadata.json ({error})") from error
    try:
        with np.load(arrays_path, allow_pickle=False) as payload:
            arrays = {name: np.array(payload[name], copy=True) for name in payload.files}
    except (OSError, EOFError, ValueError, BadZipFile) as error:
        raise fail(f"cannot read arrays.npz ({error})") from error
    try:
        digest = sha256(arrays_path.read_bytes()).hexdigest()
    except OSError as error:
        raise fail(f"cannot hash arrays.npz ({error})") from error
    validate(directory, metadata, arrays, digest)
    return IllustrationData(directory=directory, metadata=metadata, arrays=arrays)


def write_pair(
    directory: Path,
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
    *,
    semantics: Mapping[str, tuple[str, str]],
    validate: Callable[[Path, Any, Mapping[str, np.ndarray], str], None],
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    temporary_arrays = directory / ".arrays.npz.tmp"
    temporary_metadata = directory / ".metadata.json.tmp"
    with temporary_arrays.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    digest = sha256(temporary_arrays.read_bytes()).hexdigest()
    complete_metadata = {
        **metadata,
        "payload": {
            "file": "arrays.npz",
            "sha256": digest,
            "arrays": array_metadata(arrays, semantics),
        },
    }
    temporary_metadata.write_text(
        json.dumps(json_value(complete_metadata), allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with np.load(temporary_arrays, allow_pickle=False) as payload:
        check_arrays = {name: np.array(payload[name], copy=True) for name in payload.files}
    check_metadata = json.loads(temporary_metadata.read_text(encoding="utf-8"))
    validate(directory, check_metadata, check_arrays, digest)
    temporary_arrays.replace(directory / "arrays.npz")
    temporary_metadata.replace(directory / "metadata.json")


def validate_required_arrays(
    arrays: Mapping[str, np.ndarray],
    required: Mapping[str, tuple[tuple[int | None, ...] | None, str]],
    *,
    fail: Callable[[str], IllustrationDataError],
) -> None:
    missing = sorted(set(required) - set(arrays))
    if missing:
        raise fail(f"missing required arrays {missing}")
    for name, (shape, kind) in required.items():
        array = arrays[name]
        if shape is not None:
            if array.ndim != len(shape) or any(
                expected is not None and actual != expected
                for actual, expected in zip(array.shape, shape)
            ):
                raise fail(f"{name} has shape {array.shape}, expected {shape}")
        if kind == "integer" and array.dtype.kind not in "iu":
            raise fail(f"{name} must use an integer dtype")
        if kind == "floating" and array.dtype.kind != "f":
            raise fail(f"{name} must use a floating dtype")
        if kind == "boolean" and array.dtype.kind != "b":
            raise fail(f"{name} must use a boolean dtype")
        if array.dtype.kind in "fciu" and not np.all(np.isfinite(array)):
            raise fail(f"{name} contains non-finite values")
