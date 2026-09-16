"""Versioned, solver-independent figure-data export."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .continuation import FiberResult


SCHEMA_VERSION = "lumice-integral.figure-data/v1"


@dataclass(frozen=True)
class FigureDataFiles:
    metadata: Path
    arrays: Path


def _json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TypeError(f"figure-data metadata cannot encode {type(value).__name__}")


def _branch_margin_arrays(result: FiberResult) -> tuple[list[str], np.ndarray]:
    if len(result.branch_diagnostics.accepted_margins) != len(result.poses):
        raise ValueError("figure-data branch margins do not align with poses")
    names = sorted(
        {
            name
            for margins in result.branch_diagnostics.accepted_margins
            for name in margins
        }
    )
    values = np.full((len(result.poses), len(names)), np.nan, dtype=np.float64)
    for row, margins in enumerate(result.branch_diagnostics.accepted_margins):
        for column, name in enumerate(names):
            if name in margins:
                values[row, column] = margins[name]
    return names, values


def _arrays(result: FiberResult) -> tuple[dict[str, np.ndarray], list[str]]:
    margin_names, margins = _branch_margin_arrays(result)
    singular_values = np.asarray(
        [diagnostic.singular_values for diagnostic in result.jacobian_diagnostics],
        dtype=np.float64,
    ).reshape((-1, 2))
    normal_jacobian = np.asarray(
        [diagnostic.normal_jacobian for diagnostic in result.jacobian_diagnostics],
        dtype=np.float64,
    )
    rank = np.asarray(
        [diagnostic.rank for diagnostic in result.jacobian_diagnostics],
        dtype=np.int64,
    )
    condition = np.asarray(
        [diagnostic.condition for diagnostic in result.jacobian_diagnostics],
        dtype=np.float64,
    )
    increments = np.asarray(result.arclength_increments, dtype=np.float64)
    sample_count = len(result.poses)
    if len(increments) != max(0, sample_count - 1):
        raise ValueError("figure-data arclength increments do not align with poses")
    cumulative_arclength = (
        np.concatenate((np.zeros(1, dtype=np.float64), np.cumsum(increments)))
        if sample_count
        else np.empty(0, dtype=np.float64)
    )
    arrays = {
        "poses": np.asarray(result.poses, dtype=np.float64),
        "tangents": np.asarray(result.tangents, dtype=np.float64),
        "residual_norms": np.asarray(result.residual_norms, dtype=np.float64),
        "arclength_increments": increments,
        "cumulative_arclength": cumulative_arclength,
        "jacobian_singular_values": singular_values,
        "normal_jacobian": normal_jacobian,
        "jacobian_rank": rank,
        "jacobian_condition": condition,
        "branch_margins": margins,
    }
    sample_arrays = (
        "poses",
        "tangents",
        "residual_norms",
        "cumulative_arclength",
        "jacobian_singular_values",
        "normal_jacobian",
        "jacobian_rank",
        "jacobian_condition",
        "branch_margins",
    )
    mismatched = {
        name: array.shape
        for name, array in arrays.items()
        if name in sample_arrays and len(array) != sample_count
    }
    if mismatched:
        raise ValueError(f"figure-data sample arrays do not align: {mismatched}")
    return arrays, margin_names


def _array_metadata(arrays: Mapping[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    units = {
        "arclength_increments": "radian",
        "cumulative_arclength": "radian",
    }
    semantics = {
        "poses": "ordered SO(3) rotation matrices",
        "tangents": "oriented right-trivialized unit tangents",
        "residual_norms": "target-chart residual norms",
        "arclength_increments": "SO(3) metric edge lengths",
        "cumulative_arclength": "cumulative SO(3) metric arclength",
        "jacobian_singular_values": "two singular values of the local residual Jacobian",
        "normal_jacobian": "normal Jacobian J_perp of the halo map",
        "jacobian_rank": "estimated local residual-Jacobian rank",
        "jacobian_condition": "estimated local residual-Jacobian condition number",
        "branch_margins": "named path-domain margins; columns are declared in metadata",
    }
    return {
        name: {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "unit": units.get(name, "dimensionless"),
            "semantic": semantics[name],
        }
        for name, array in arrays.items()
    }


def export_fiber_figure_data(
    result: FiberResult,
    output_directory: Path | str,
    *,
    fixture: Mapping[str, Any],
    provenance: Mapping[str, Any] | None = None,
) -> FigureDataFiles:
    """Write one ``FiberResult`` as JSON metadata plus an NPZ array payload."""
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    arrays_path = output_directory / "arrays.npz"
    metadata_path = output_directory / "metadata.json"
    fixture_metadata = _json_value(fixture)
    provenance_metadata = _json_value(provenance or {})
    arrays, margin_names = _arrays(result)

    temporary_arrays = output_directory / ".arrays.npz.tmp"
    with temporary_arrays.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary_arrays.replace(arrays_path)
    array_sha256 = sha256(arrays_path.read_bytes()).hexdigest()

    metadata = {
        "schema": SCHEMA_VERSION,
        "fixture": fixture_metadata,
        "provenance": provenance_metadata,
        "result": {
            "status": result.status.value,
            "reason": result.reason.value,
            "component_scope": result.component_scope,
            "component_completeness": result.component_completeness,
            "sample_count": len(result.poses),
            "conventions": _json_value(result.conventions),
            "weight_observables": _json_value(result.weight_observables),
            "closure": {
                "accumulated_arclength": result.closure_diagnostics.accumulated_arclength,
                "seed_distance": result.closure_diagnostics.seed_distance,
                "section_crossed": result.closure_diagnostics.section_crossed,
                "tangent_dot": result.closure_diagnostics.tangent_dot,
                "final_correction_accepted": (
                    result.closure_diagnostics.final_correction_accepted
                ),
            },
        },
        "payload": {
            "file": arrays_path.name,
            "sha256": array_sha256,
            "arrays": _array_metadata(arrays),
            "branch_margin_columns": margin_names,
        },
    }
    temporary_metadata = output_directory / ".metadata.json.tmp"
    temporary_metadata.write_text(
        json.dumps(
            _json_value(metadata),
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary_metadata.replace(metadata_path)
    return FigureDataFiles(metadata_path, arrays_path)
