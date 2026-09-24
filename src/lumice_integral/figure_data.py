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
from .quadrature import ResampledQuadratureResult, pointwise_integrand
from .weights import WeightObservable


# v2 (task-single-fiber-physical-integrand): ``result.weight_observables`` values
# are objects ``{status, unit, normalization, array}`` instead of bare status
# strings, and every available factor adds a ``weight_<name>`` sample array.
# Still v2 (task-single-fiber-line-quadrature): the optional ``result.quadrature``
# object and ``integrand`` sample array are pure additions; no existing field
# changed type.
# v3 (task-resample-and-integrate): ``result.quadrature`` describes the
# resampled fixed-grid quadrature; the adaptive method's fields (refinements,
# maximum_refinement_depth, maximum_depth_reached, convergence order fields,
# refinement_failures, depth_exhausted_edges) are gone and the grid/retraction
# evidence fields replace them.  Every other field and array is unchanged.
SCHEMA_VERSION = "lumice-integral.figure-data/v3"
WEIGHT_ARRAY_PREFIX = "weight_"
INTEGRAND_ARRAY_NAME = "integrand"


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


def _weight_array_name(name: str) -> str:
    return f"{WEIGHT_ARRAY_PREFIX}{name}"


def _weight_metadata(result: FiberResult) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    for name, observable in result.weight_observables.items():
        if not isinstance(observable, WeightObservable):
            raise TypeError(f"figure-data expects WeightObservable for {name!r}")
        metadata[name] = {
            "status": observable.status,
            "unit": observable.unit,
            "normalization": observable.normalization,
            "array": (
                _weight_array_name(name) if observable.status == "available" else None
            ),
        }
    return metadata


def _weight_arrays(result: FiberResult) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for name, observable in result.weight_observables.items():
        if observable.status != "available":
            continue
        values = np.asarray(observable.values, dtype=np.float64)
        if values.shape != (len(result.poses),):
            raise ValueError(f"figure-data weight {name!r} does not align with poses")
        arrays[_weight_array_name(name)] = values
    return arrays


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


def _quadrature_metadata(quadrature: ResampledQuadratureResult) -> dict[str, Any]:
    return {
        "status": quadrature.status,
        "method": quadrature.method,
        "fiber_status": quadrature.fiber_status,
        "coverage": quadrature.coverage,
        "component_completeness": quadrature.component_completeness,
        "density_factor_name": quadrature.density_factor_name,
        "factor_names": list(quadrature.factor_names),
        "epsilon": quadrature.epsilon,
        "relative_tolerance": quadrature.relative_tolerance,
        "initial_node_count": quadrature.initial_node_count,
        "maximum_node_count": quadrature.maximum_node_count,
        "retraction_iterations": quadrature.retraction_iterations,
        "node_count": quadrature.node_count,
        "refinement_rounds": quadrature.refinement_rounds,
        "node_count_exhausted": quadrature.node_count_exhausted,
        "node_count_history": [[count, value] for count, value in quadrature.node_count_history],
        "value": quadrature.value,
        "error_estimate": quadrature.error_estimate,
        "raw_value": quadrature.raw_value,
        "raw_error_estimate": quadrature.raw_error_estimate,
        "haar_to_dvol_g_factor": quadrature.haar_to_dvol_g_factor,
        "residual_before_max": quadrature.residual_before_max,
        "residual_before_median": quadrature.residual_before_median,
        "residual_after_max": quadrature.residual_after_max,
        "residual_after_median": quadrature.residual_after_median,
        "non_finite_node_count": quadrature.non_finite_node_count,
        "endpoint_truncation_estimate": quadrature.endpoint_truncation_estimate,
        "endpoint_truncation_note": quadrature.endpoint_truncation_note,
        # ``factor_seconds`` (wall clock) is deliberately not exported: the
        # canonical export must stay byte-identical across runs (section 6).
        "integrand_array": (
            INTEGRAND_ARRAY_NAME if quadrature.status == "available" else None
        ),
    }


def _integrand_arrays(
    result: FiberResult, quadrature: ResampledQuadratureResult | None
) -> dict[str, np.ndarray]:
    if quadrature is None or quadrature.status != "available":
        return {}
    values = pointwise_integrand(result, epsilon=quadrature.epsilon)
    return {INTEGRAND_ARRAY_NAME: np.asarray(values, dtype=np.float64)}


def _arrays(
    result: FiberResult, quadrature: ResampledQuadratureResult | None
) -> tuple[dict[str, np.ndarray], list[str]]:
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
        **_weight_arrays(result),
        **_integrand_arrays(result, quadrature),
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
        *(name for name in arrays if name.startswith(WEIGHT_ARRAY_PREFIX)),
        INTEGRAND_ARRAY_NAME,
    )
    mismatched = {
        name: array.shape
        for name, array in arrays.items()
        if name in sample_arrays and len(array) != sample_count
    }
    if mismatched:
        raise ValueError(f"figure-data sample arrays do not align: {mismatched}")
    return arrays, margin_names


def _array_metadata(
    arrays: Mapping[str, np.ndarray],
    result: FiberResult,
    quadrature: ResampledQuadratureResult | None,
) -> dict[str, dict[str, Any]]:
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
    for name, observable in result.weight_observables.items():
        if observable.status == "available":
            array_name = _weight_array_name(name)
            units[array_name] = observable.unit
            semantics[array_name] = (
                f"named physical factor {name!r} evaluated at every accepted pose; "
                "see result.weight_observables for its normalization"
            )
    if quadrature is not None and INTEGRAND_ARRAY_NAME in arrays:
        units[INTEGRAND_ARRAY_NAME] = (
            "length^2 (the entry_measure unit; rho_pose, fresnel_transmission, "
            "path_validity and normal_jacobian are dimensionless)"
        )
        semantics[INTEGRAND_ARRAY_NAME] = (
            "partial physical integrand rho_pose * "
            f"{' * '.join(quadrature.factor_names)} / (normal_jacobian + epsilon) "
            "at every accepted pose; epsilon, the Haar convention and the "
            "integrated value are declared in result.quadrature and "
            "result.conventions; normal_jacobian stays unregularised"
        )
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
    quadrature: ResampledQuadratureResult | None = None,
) -> FigureDataFiles:
    """Write one ``FiberResult`` as JSON metadata plus an NPZ array payload.

    With ``quadrature`` the metadata gains ``result.quadrature`` and, when the
    integral is available, the arrays gain the pointwise ``integrand``;
    without it the output is exactly the v2 layout of the previous stage.
    """
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    arrays_path = output_directory / "arrays.npz"
    metadata_path = output_directory / "metadata.json"
    fixture_metadata = _json_value(fixture)
    provenance_metadata = _json_value(provenance or {})
    arrays, margin_names = _arrays(result, quadrature)

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
            "weight_observables": _weight_metadata(result),
            "quadrature": (
                _quadrature_metadata(quadrature) if quadrature is not None else None
            ),
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
            "arrays": _array_metadata(arrays, result, quadrature),
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


# Chapter-10 verdicts (task ch10-numerical-verdicts): a schema of its own, not a v4 of the fiber schema --
# a verdict is a statement with its numbers and arrays, not a FiberResult.
VERDICT_SCHEMA_VERSION = "lumice-integral.ch10-verdict/v1"


def export_verdict_figure_data(
    verdict: Any,
    output_directory: Path | str,
    *,
    provenance: Mapping[str, Any] | None = None,
) -> FigureDataFiles:
    """Write one :class:`.ch10_verdicts.Verdict` as ``metadata.json`` plus ``arrays.npz`` (same file discipline as fibers).

    The metadata holds the verdict's name, status, statement, numbers and
    parameters, the caller's ``provenance`` and the payload's SHA-256 with
    each array's shape, dtype and note.
    """
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    arrays_path = output_directory / "arrays.npz"
    metadata_path = output_directory / "metadata.json"
    arrays = {name: np.asarray(array) for name, array in verdict.arrays.items()}
    missing = sorted(set(arrays) - set(verdict.array_notes))
    if missing:
        raise ValueError(f"verdict {verdict.name!r} has arrays without a note: {missing}")

    temporary_arrays = output_directory / ".arrays.npz.tmp"
    with temporary_arrays.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary_arrays.replace(arrays_path)
    array_sha256 = sha256(arrays_path.read_bytes()).hexdigest()

    metadata = {
        "schema": VERDICT_SCHEMA_VERSION,
        "verdict": verdict.name,
        "status": verdict.status,
        "statement": verdict.statement,
        "numbers": verdict.numbers,
        "parameters": verdict.parameters,
        "provenance": provenance or {},
        "payload": {
            "file": arrays_path.name,
            "sha256": array_sha256,
            "arrays": {
                name: {"shape": list(array.shape), "dtype": str(array.dtype), "note": verdict.array_notes[name]}
                for name, array in arrays.items()
            },
        },
    }
    temporary_metadata = output_directory / ".metadata.json.tmp"
    temporary_metadata.write_text(
        json.dumps(_json_value(metadata), allow_nan=False, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_metadata.replace(metadata_path)
    return FigureDataFiles(metadata_path, arrays_path)
