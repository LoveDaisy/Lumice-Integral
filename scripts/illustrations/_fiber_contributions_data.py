"""Production data for the canonical fiber-contributions illustration."""

from __future__ import annotations

from pathlib import Path
import shlex
from typing import Any, Mapping

import numpy as np
from scipy.integrate import cumulative_simpson

from _canonical_data import IllustrationDataError
from _illustration_io import IllustrationData, load_pair, validate_required_arrays, write_pair


FIGURE_ID = "fiber-contributions"
OUTPUT_NAME = f"{FIGURE_ID}.png"
SCHEMA_VERSION = "lumice-integral.illustration.fiber-contributions/v1"

_REQUIRED_ARRAYS = {
    "s": ((None,), "floating"),
    "parameters": ((None,), "floating"),
    "speed": ((None,), "floating"),
    "rho_pose": ((None,), "floating"),
    "entry_area_over_a2": ((None,), "floating"),
    "transmission": ((None,), "floating"),
    "jacobian_factor": ((None,), "floating"),
    "product_over_a2": ((None,), "floating"),
    "selected_s": ((4,), "floating"),
}

_ARRAY_SEMANTICS = {
    "s": ("radian", "Production numerical SO(3) arclength on the final quadrature grid"),
    "parameters": ("spline parameter", "Final quadrature spline parameters"),
    "speed": ("radian / spline parameter", "Production ds/dt"),
    "rho_pose": ("dimensionless", "Production pose-density factor rho"),
    "entry_area_over_a2": ("dimensionless", "Production effective entry area divided by a squared"),
    "transmission": ("dimensionless", "Production Fresnel transmission T"),
    "jacobian_factor": ("dimensionless", "K = 1 / (J_perp + epsilon)"),
    "product_over_a2": ("dimensionless", "rho times A/a squared times T times K"),
    "selected_s": ("radian", "Shared numbered presentation positions on the same numerical arclength ruler"),
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
            str(Path(__file__).with_name("fiber_contributions.py").resolve()),
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
    count = len(arrays["s"])
    if any(len(arrays[name]) != count for name in _REQUIRED_ARRAYS if name != "selected_s"):
        raise fail("factor arrays do not align")
    if np.any(np.diff(arrays["s"]) <= 0.0):
        raise fail("s must be strictly increasing")
    if any(np.any(arrays[name] <= 0.0) for name in ("rho_pose", "entry_area_over_a2", "transmission", "jacobian_factor", "product_over_a2")):
        raise fail("logarithmic factor curves must be strictly positive")


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
    from lumice_integral.canonical_scene import canonical_crystal
    from lumice_integral.quadrature import HAAR_TO_DVOL_G_FACTOR

    data_root = Path(data_root)
    if canonical_directory is None:
        canonical_directory = recompute_canonical_data(data_root).directory
    common = load_canonical_data(canonical_directory)
    shared = common.arrays
    quadrature = common.metadata["quadrature"]
    a = float(canonical_crystal().a)
    epsilon = float(quadrature["epsilon"])
    jacobian_factor = 1.0 / (shared["normal_jacobian"] + epsilon)
    product = (
        shared["rho_pose"]
        * (shared["entry_measure"] / a**2)
        * shared["fresnel_transmission"]
        * jacobian_factor
    )
    expected = shared["integrand"] / a**2
    relative_error = float(np.max(np.abs(product - expected) / expected))
    if relative_error >= 1e-12:
        raise RuntimeError(f"factor product differs from production integrand by {relative_error:.3g}")
    if not np.all(shared["path_validity"] == 1.0):
        raise RuntimeError("canonical contribution figure cannot omit a non-unit path-validity factor")
    if not np.allclose(
        shared["quadrature_rotations"][0], shared["quadrature_rotations"][-1], rtol=0.0, atol=1e-12
    ):
        raise RuntimeError("canonical quadrature grid does not close at the same pose")
    cumulative_raw = cumulative_simpson(
        shared["integrand"] * shared["quadrature_speed"],
        x=shared["quadrature_parameters"],
        initial=0.0,
    )
    integrated = float(cumulative_raw[-1] * HAAR_TO_DVOL_G_FACTOR / a**2)
    integral_error = abs(integrated - float(quadrature["value"]) / a**2)
    if integral_error >= 1e-10 * max(1.0, abs(integrated)):
        raise RuntimeError(f"factor-grid integral differs from production result by {integral_error:.3g}")

    arrays = {
        "s": shared["quadrature_s"],
        "parameters": shared["quadrature_parameters"],
        "speed": shared["quadrature_speed"],
        "rho_pose": shared["rho_pose"],
        "entry_area_over_a2": shared["entry_measure"] / a**2,
        "transmission": shared["fresnel_transmission"],
        "jacobian_factor": jacobian_factor,
        "product_over_a2": product,
        "selected_s": shared["selected_s"],
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
        "quadrature": {
            "epsilon": epsilon,
            "value_over_a2": float(quadrature["value"]) / a**2,
            "error_estimate_over_a2": float(quadrature["error_estimate"]) / a**2,
            "refined_value_over_a2": float(quadrature["refined_value"]) / a**2,
            "refined_error_estimate_over_a2": float(quadrature["refined_error_estimate"]) / a**2,
            "node_count": int(quadrature["node_count"]),
            "node_count_exhausted": bool(quadrature["node_count_exhausted"]),
            "haar_to_dvol_g_factor": HAAR_TO_DVOL_G_FACTOR,
        },
        "validation": {
            "factor_product_max_relative_error": relative_error,
            "factor_grid_integral_absolute_error": integral_error,
            "path_validity_is_unity": True,
            "closed_endpoint_rotation_max_abs": float(
                np.max(
                    np.abs(
                        shared["quadrature_rotations"][0]
                        - shared["quadrature_rotations"][-1]
                    )
                )
            ),
            "fiber_length_rad": float(shared["quadrature_s"][-1]),
        },
        "conventions": {
            "area_unit": "a^2, with a the canonical hexagon edge length",
            "normalization": "no individual factor is peak-normalized",
            "arclength": "production ds/dt integrated on the final quadrature grid",
            "closure": "s=0 and s=L denote the same pose",
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
