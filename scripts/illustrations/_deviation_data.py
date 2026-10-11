"""Production data for the deviation-level-set illustration."""

from __future__ import annotations

import json
from pathlib import Path
import shlex
from typing import Any, Mapping

import numpy as np

from _canonical_data import IllustrationDataError
from _illustration_io import IllustrationData, load_pair, validate_required_arrays, write_pair


FIGURE_ID = "deviation-level-set"
OUTPUT_NAME = f"{FIGURE_ID}.png"
SCHEMA_VERSION = "lumice-integral.illustration.deviation-level-set/v1"
SPHERE_SHAPE = (121, 241)
CHART_SHAPE = (321, 321)

_REQUIRED_ARRAYS = {
    "sphere_points": ((*SPHERE_SHAPE, 3), "floating"),
    "sphere_d_deg": (SPHERE_SHAPE, "floating"),
    "sphere_valid": (SPHERE_SHAPE, "boolean"),
    "chart_x": (CHART_SHAPE, "floating"),
    "chart_y": (CHART_SHAPE, "floating"),
    "chart_d_deg": (CHART_SHAPE, "floating"),
    "chart_valid": (CHART_SHAPE, "boolean"),
    "chart_center": ((3,), "floating"),
    "chart_e1": ((3,), "floating"),
    "chart_e2": ((3,), "floating"),
    "u_curve": ((513, 3), "floating"),
    "curve_xy": ((513, 2), "floating"),
    "selected_indices": ((4,), "integer"),
    "delta_rad": ((), "floating"),
}

_ARRAY_SEMANTICS = {
    "sphere_points": ("dimensionless", "Unit-sphere field sample positions"),
    "sphere_d_deg": ("degree", "D_P on the sphere; zero-filled outside sphere_valid"),
    "sphere_valid": ("boolean", "Production optical-domain mask U_P on the sphere grid"),
    "chart_x": ("dimensionless", "Orthographic chart coordinate u dot e1"),
    "chart_y": ("dimensionless", "Orthographic chart coordinate u dot e2"),
    "chart_d_deg": ("degree", "D_P in the chart; zero-filled outside chart_valid"),
    "chart_valid": ("boolean", "Production optical-domain mask U_P in the chart"),
    "chart_center": ("dimensionless", "Unit direction at the chart center"),
    "chart_e1": ("dimensionless", "First orthographic chart basis vector"),
    "chart_e2": ("dimensionless", "Second orthographic chart basis vector"),
    "u_curve": ("dimensionless", "Production fiber projection u = R^T s_hat"),
    "curve_xy": ("dimensionless", "Fiber projection in the orthographic chart"),
    "selected_indices": ("index", "Shared numbered positions in the presentation grid"),
    "delta_rad": ("radian", "Canonical incident-to-outgoing deviation"),
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
            str(Path(__file__).with_name("deviation_level_set.py").resolve()),
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
    if not np.all(arrays["sphere_valid"] <= np.isfinite(arrays["sphere_d_deg"])):
        raise fail("valid sphere samples must have finite field values")
    if not np.all(arrays["chart_valid"] <= np.isfinite(arrays["chart_d_deg"])):
        raise fail("valid chart samples must have finite field values")
    if not np.array_equal(arrays["selected_indices"], [0, 128, 256, 384]):
        raise fail("selected_indices do not match the shared presentation grid")


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


def recompute_data(
    data_root: Path | str,
    canonical_directory: Path | str | None = None,
) -> IllustrationData:
    from _canonical_data import load_data as load_canonical_data
    from _canonical_data import recompute_data as recompute_canonical_data
    from lumice_integral.canonical_scene import (
        CANONICAL_REFRACTIVE_INDEX,
        canonical_crystal,
    )
    from lumice_integral.dp_field import DPField

    data_root = Path(data_root)
    if canonical_directory is None:
        canonical_directory = recompute_canonical_data(data_root).directory
    common = load_canonical_data(canonical_directory)
    u_curve = common.arrays["presentation_u"]
    selected_indices = common.arrays["selected_indices"]
    incident = common.arrays["incident_direction"]
    target = common.arrays["target_direction"]
    delta = float(np.arctan2(np.linalg.norm(np.cross(incident, target)), incident @ target))
    field = DPField.build(canonical_crystal(), [3, 5], CANONICAL_REFRACTIVE_INDEX)

    def evaluate(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        flat = points.reshape(-1, 3)
        valid = field.valid_batch(flat)
        values = np.zeros(len(flat), dtype=np.float64)
        values[valid] = np.degrees(field.d_p_batch(flat[valid]))
        if not np.all(np.isfinite(values[valid])):
            raise RuntimeError("D_P returned non-finite values inside the optical domain")
        return values.reshape(points.shape[:-1]), valid.reshape(points.shape[:-1])

    longitude, latitude = np.meshgrid(
        np.linspace(-np.pi, np.pi, SPHERE_SHAPE[1]),
        np.linspace(-np.pi / 2.0, np.pi / 2.0, SPHERE_SHAPE[0]),
    )
    sphere_points = np.stack(
        [
            np.cos(latitude) * np.cos(longitude),
            np.cos(latitude) * np.sin(longitude),
            np.sin(latitude),
        ],
        axis=-1,
    )
    sphere_d, sphere_valid = evaluate(sphere_points)
    center = np.mean(u_curve, axis=0)
    center /= np.linalg.norm(center)
    e1 = np.cross([0.0, 0.0, 1.0], center)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(center, e1)
    x, y = np.meshgrid(
        np.linspace(-0.94, 0.94, CHART_SHAPE[1]),
        np.linspace(-0.94, 0.94, CHART_SHAPE[0]),
    )
    rim = x * x + y * y < 0.995**2
    chart_points = (
        x[..., None] * e1
        + y[..., None] * e2
        + np.sqrt(np.maximum(1.0 - x * x - y * y, 0.0))[..., None] * center
    )
    chart_d = np.zeros(CHART_SHAPE, dtype=np.float64)
    chart_valid = np.zeros(CHART_SHAPE, dtype=bool)
    values, valid = evaluate(chart_points[rim])
    chart_d[rim], chart_valid[rim] = values, valid
    curve_xy = np.stack([u_curve @ e1, u_curve @ e2], axis=-1)
    curve_values = field.d_p_batch(u_curve)
    level_error = float(np.max(np.abs(curve_values - delta)))
    if not np.all(field.valid_batch(u_curve)) or level_error >= 1e-10:
        raise RuntimeError(f"fiber projection is not on the canonical D_P level: {level_error:.3g}")

    arrays = {
        "sphere_points": sphere_points,
        "sphere_d_deg": sphere_d,
        "sphere_valid": sphere_valid,
        "chart_x": x,
        "chart_y": y,
        "chart_d_deg": chart_d,
        "chart_valid": chart_valid,
        "chart_center": center,
        "chart_e1": e1,
        "chart_e2": e2,
        "u_curve": u_curve,
        "curve_xy": curve_xy,
        "selected_indices": selected_indices,
        "delta_rad": np.asarray(delta),
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
        "field": {
            "implementation": "DPField.build(..., [3, 5], n=1.31)",
            "sphere_grid": list(SPHERE_SHAPE),
            "chart_grid": list(CHART_SHAPE),
            "chart_projection": "orthographic hemisphere; not equal-area",
            "outside_value": "array value is zero-filled and ignored wherever the paired valid mask is false",
        },
        "validation": {
            "fiber_projected_level_error_rad": level_error,
            "fiber_all_inside_optical_domain": True,
            "delta_deg": float(np.degrees(delta)),
        },
        "conventions": {
            "colour": "angular deviation D_P in degrees, not image brightness",
            "grey": "outside the optical formula domain U_P, not every zero-weight pose",
            "highlight": "production fiber projection, not a grid contour or completeness certificate",
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
