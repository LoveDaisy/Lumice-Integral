"""Shared production data for the accepted canonical theory illustrations."""

from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any, Mapping

import numpy as np

from _illustration_io import (
    IllustrationData,
    IllustrationDataError,
    load_pair,
    validate_required_arrays,
    write_pair,
)


FIGURE_ID = "canonical-data"
SCHEMA_VERSION = "lumice-integral.illustration.canonical-data/v1"
PRESENTATION_NODE_COUNT = 513
RETRACTION_ITERATIONS = 2
SELECTED_INDICES = np.array([0, 128, 256, 384], dtype=np.int64)
ORIGINAL_ACCEPTED_REVISION = "98b4a8c7ec70e519f3744279840c2814f871ade7"

_REQUIRED_ARRAYS = {
    "accepted_rotations": ((None, 3, 3), "floating"),
    "accepted_tangents": ((None, 3), "floating"),
    "presentation_parameters": ((PRESENTATION_NODE_COUNT,), "floating"),
    "presentation_rotations": ((PRESENTATION_NODE_COUNT, 3, 3), "floating"),
    "presentation_tangents": ((PRESENTATION_NODE_COUNT, 3), "floating"),
    "presentation_u": ((PRESENTATION_NODE_COUNT, 3), "floating"),
    "presentation_outgoing": ((PRESENTATION_NODE_COUNT, 3), "floating"),
    "presentation_relative_rotvec": ((PRESENTATION_NODE_COUNT, 3), "floating"),
    "presentation_s": ((PRESENTATION_NODE_COUNT,), "floating"),
    "presentation_speed": ((PRESENTATION_NODE_COUNT,), "floating"),
    "selected_indices": ((4,), "integer"),
    "selected_s": ((4,), "floating"),
    "quadrature_parameters": ((None,), "floating"),
    "quadrature_rotations": ((None, 3, 3), "floating"),
    "quadrature_tangents": ((None, 3), "floating"),
    "quadrature_u": ((None, 3), "floating"),
    "quadrature_s": ((None,), "floating"),
    "quadrature_speed": ((None,), "floating"),
    "integrand": ((None,), "floating"),
    "normal_jacobian": ((None,), "floating"),
    "rho_pose": ((None,), "floating"),
    "entry_measure": ((None,), "floating"),
    "fresnel_transmission": ((None,), "floating"),
    "path_validity": ((None,), "floating"),
    "source_world": ((3,), "floating"),
    "incident_direction": ((3,), "floating"),
    "target_direction": ((3,), "floating"),
    "vertices_body": ((12, 3), "floating"),
    "face_numbers": ((None,), "integer"),
    "face_normals_body": ((None, 3), "floating"),
    "face_centers_body": ((None, 3), "floating"),
}

_ARRAY_SEMANTICS = {
    "accepted_rotations": ("dimensionless", "Production accepted body-to-world rotations"),
    "accepted_tangents": ("dimensionless", "Production accepted right-trivialized fiber tangents"),
    "presentation_parameters": ("spline parameter", "Uniform cumulative-chord spline parameters"),
    "presentation_rotations": ("dimensionless", "Retracted rotations on the 513-point presentation grid"),
    "presentation_tangents": ("dimensionless", "Production fiber tangents on the presentation grid"),
    "presentation_u": ("dimensionless", "Body-frame source directions u = R^T s_hat"),
    "presentation_outgoing": ("dimensionless", "Production outgoing propagation directions"),
    "presentation_relative_rotvec": ("radian", "log(R_0^T R) presentation coordinates"),
    "presentation_s": ("radian", "Numerical SO(3) arclength from production ds/dt"),
    "presentation_speed": ("radian / spline parameter", "Production ds/dt on the presentation grid"),
    "selected_indices": ("index", "Shared numbered positions in the presentation grid"),
    "selected_s": ("radian", "Numerical arclengths at the shared numbered positions"),
    "quadrature_parameters": ("spline parameter", "Final production quadrature grid parameters"),
    "quadrature_rotations": ("dimensionless", "Retracted rotations on the final quadrature grid"),
    "quadrature_tangents": ("dimensionless", "Production fiber tangents on the quadrature grid"),
    "quadrature_u": ("dimensionless", "Body-frame source directions on the quadrature grid"),
    "quadrature_s": ("radian", "Numerical SO(3) arclength on the quadrature grid"),
    "quadrature_speed": ("radian / spline parameter", "Production ds/dt on the quadrature grid"),
    "integrand": ("length^2", "Production physical integrand before ds and Haar conversion"),
    "normal_jacobian": ("dimensionless", "Unregularized normal Jacobian J_perp"),
    "rho_pose": ("dimensionless", "Production pose-density factor"),
    "entry_measure": ("length^2", "Production effective entry area"),
    "fresnel_transmission": ("dimensionless", "Production Fresnel transmission factor"),
    "path_validity": ("dimensionless", "Production path-validity factor"),
    "source_world": ("dimensionless", "Unit direction s_hat toward the Sun"),
    "incident_direction": ("dimensionless", "Incident propagation direction -s_hat"),
    "target_direction": ("dimensionless", "Canonical outgoing propagation direction d"),
    "vertices_body": ("crystal length", "Canonical crystal vertices in body coordinates"),
    "face_numbers": ("face id", "Canonical crystal face numbers"),
    "face_normals_body": ("dimensionless", "Canonical crystal outward face normals"),
    "face_centers_body": ("crystal length", "Canonical crystal face centroids"),
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
            str(Path(__file__).with_name("regen.py").resolve()),
            "--mode",
            "recompute",
            "--data-dir",
            str(Path(data_root).resolve()),
            "--output-dir",
            str((Path(data_root) / "unused-figures").resolve()),
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
    if metadata.get("schema") != SCHEMA_VERSION:
        raise fail(f"unsupported schema {metadata.get('schema')!r}")
    if metadata.get("figure_id") != FIGURE_ID:
        raise fail(f"unexpected figure_id {metadata.get('figure_id')!r}")
    payload = metadata.get("payload")
    if not isinstance(payload, Mapping) or payload.get("file") != "arrays.npz":
        raise fail("payload file declaration is missing or wrong")
    if payload.get("sha256") != digest:
        raise fail("arrays.npz SHA-256 does not match metadata")
    validate_required_arrays(arrays, _REQUIRED_ARRAYS, fail=fail)

    presentation_count = len(arrays["presentation_parameters"])
    quadrature_count = len(arrays["quadrature_parameters"])
    for name in (
        "presentation_rotations",
        "presentation_tangents",
        "presentation_u",
        "presentation_outgoing",
        "presentation_relative_rotvec",
        "presentation_s",
        "presentation_speed",
    ):
        if len(arrays[name]) != presentation_count:
            raise fail(f"{name} does not align with the presentation grid")
    for name in (
        "quadrature_rotations",
        "quadrature_tangents",
        "quadrature_u",
        "quadrature_s",
        "quadrature_speed",
        "integrand",
        "normal_jacobian",
        "rho_pose",
        "entry_measure",
        "fresnel_transmission",
        "path_validity",
    ):
        if len(arrays[name]) != quadrature_count:
            raise fail(f"{name} does not align with the quadrature grid")
    if np.any(np.diff(arrays["presentation_s"]) <= 0.0):
        raise fail("presentation_s must be strictly increasing")
    if np.any(np.diff(arrays["quadrature_s"]) <= 0.0):
        raise fail("quadrature_s must be strictly increasing")
    indices = arrays["selected_indices"]
    if not np.array_equal(indices, SELECTED_INDICES):
        raise fail(f"selected_indices must be {SELECTED_INDICES.tolist()}")
    if not np.array_equal(arrays["selected_s"], arrays["presentation_s"][indices]):
        raise fail("selected_s does not match presentation_s at selected_indices")


def load_data(directory: Path | str) -> IllustrationData:
    directory = Path(directory)
    data_root = directory.parent if directory.name == FIGURE_ID else directory
    resolved = directory if directory.name == FIGURE_ID else data_directory(directory)
    return load_pair(
        resolved,
        fail=lambda reason: _fail(resolved, reason, data_root),
        validate=lambda path, metadata, arrays, digest: _validate(
            path, metadata, arrays, digest, data_root=data_root
        ),
    )


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


def _rotation_vectors(rotations: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    relative = np.einsum("ij,njk->nik", rotations[0].T, rotations)
    vectors = Rotation.from_matrix(relative).as_rotvec()
    if np.max(np.linalg.norm(vectors, axis=1)) >= np.pi - 0.1:
        raise RuntimeError("canonical presentation curve leaves the selected rotation-vector chart")
    return vectors


def recompute_data(data_root: Path | str) -> IllustrationData:
    from scipy.integrate import cumulative_simpson

    from lumice_integral.canonical_scene import (
        canonical_crystal,
        canonical_fixture_metadata,
        canonical_incident_direction,
        canonical_pixel_problem,
        canonical_sun_direction,
        canonical_target_direction,
    )
    from lumice_integral.continuation import ContinuationOptions, FiberStatus, retract_to_fiber_batch, trace_fiber
    from lumice_integral.figure_data import export_fiber_figure_data
    from lumice_integral.quadrature import (
        INTEGRAND_FACTOR_NAMES,
        ResampleOptions,
        _evaluate_grid_nodes,
        integrate_fiber_resampled,
    )
    from lumice_integral.resample import fiber_spline, resample_spline, uniform_parameters
    from lumice_integral.weights import evaluate_weights_batch
    import jax

    data_root = Path(data_root)
    root = Path(__file__).resolve().parents[2]
    problem = canonical_pixel_problem()
    continuation_options = ContinuationOptions()
    traced = trace_fiber(problem, continuation_options)
    if traced.status is not FiberStatus.CLOSED:
        raise RuntimeError(f"canonical fiber did not close: {traced.status.value}: {traced.terminal_payload.message}")

    quadrature_options = ResampleOptions()
    quadrature = integrate_fiber_resampled(problem, traced, quadrature_options)
    if quadrature.status != "available" or quadrature.node_count_exhausted:
        raise RuntimeError("canonical production quadrature did not converge within its configured grid")
    refined_options = ResampleOptions(relative_tolerance=1e-5)
    refined = integrate_fiber_resampled(problem, traced, refined_options)
    if refined.status != "available" or refined.node_count_exhausted:
        raise RuntimeError("canonical refined quadrature did not converge")

    upstream = export_fiber_figure_data(
        traced,
        data_root / "canonical-fiber",
        fixture=canonical_fixture_metadata(),
        provenance={"purpose": "upstream production fiber for accepted theory illustrations"},
        quadrature=quadrature,
    )
    upstream_metadata = json.loads(upstream.metadata.read_text(encoding="utf-8"))

    spline = fiber_spline(traced)

    def evaluate_grid(node_count: int):
        parameters = uniform_parameters(spline, node_count)
        nodes = _evaluate_grid_nodes(problem, spline, parameters, quadrature_options)
        predictors = resample_spline(spline, parameters)
        retraction = retract_to_fiber_batch(
            problem,
            predictors.rotations,
            predictors.phase_tangents,
            iterations=quadrature_options.retraction_iterations,
        )
        weights = evaluate_weights_batch(
            problem.weight_evaluators,
            retraction.rotations,
            ("rho_pose", *INTEGRAND_FACTOR_NAMES),
        ).values
        if not np.all(retraction.finite & nodes.finite):
            raise RuntimeError("canonical resampling produced non-finite nodes")
        if np.max(np.abs(retraction.residual_after)) > 1e-10:
            raise RuntimeError("canonical resampling retraction residual exceeds 1e-10")
        if np.any(nodes.speed <= 0.0):
            raise RuntimeError("canonical resampling speed is not strictly positive")
        if not np.all(weights["path_validity"] == 1.0):
            raise RuntimeError("canonical resampling contains an invalid finite-crystal path")
        product = (
            weights["rho_pose"]
            * weights["entry_measure"]
            * weights["fresnel_transmission"]
            * weights["path_validity"]
            / (retraction.normal_jacobians + quadrature_options.epsilon)
        )
        relative_error = np.max(np.abs(product - nodes.integrand) / nodes.integrand)
        if relative_error > 1e-12:
            raise RuntimeError(f"canonical integrand factor assembly differs by {relative_error:.3g}")
        arclength = cumulative_simpson(nodes.speed, x=parameters, initial=0.0)
        return parameters, nodes, predictors, retraction, weights, np.asarray(arclength)

    presentation = evaluate_grid(PRESENTATION_NODE_COUNT)
    production = evaluate_grid(quadrature.node_count)
    p_parameters, p_nodes, _, p_retraction, _, p_s = presentation
    q_parameters, q_nodes, _, q_retraction, q_weights, q_s = production

    source_world = np.asarray(canonical_sun_direction(), dtype=np.float64)
    target = np.asarray(canonical_target_direction(), dtype=np.float64)
    presentation_outgoing = np.asarray(
        jax.vmap(problem.direction_evaluator)(p_retraction.rotations), dtype=np.float64
    )
    outgoing_error = float(np.max(np.linalg.norm(presentation_outgoing - target, axis=1)))
    if outgoing_error > 1e-10:
        raise RuntimeError(f"canonical outgoing residual {outgoing_error:.3g} exceeds 1e-10")

    crystal = canonical_crystal()
    face_numbers = np.array([face.number for face in crystal.faces], dtype=np.int64)
    arrays = {
        "accepted_rotations": np.asarray(traced.poses, dtype=np.float64),
        "accepted_tangents": np.asarray(traced.tangents, dtype=np.float64),
        "presentation_parameters": p_parameters,
        "presentation_rotations": p_retraction.rotations,
        "presentation_tangents": p_retraction.tangents,
        "presentation_u": np.einsum("nji,j->ni", p_retraction.rotations, source_world),
        "presentation_outgoing": presentation_outgoing,
        "presentation_relative_rotvec": _rotation_vectors(p_retraction.rotations),
        "presentation_s": p_s,
        "presentation_speed": p_nodes.speed,
        "selected_indices": SELECTED_INDICES,
        "selected_s": p_s[SELECTED_INDICES],
        "quadrature_parameters": q_parameters,
        "quadrature_rotations": q_retraction.rotations,
        "quadrature_tangents": q_retraction.tangents,
        "quadrature_u": np.einsum("nji,j->ni", q_retraction.rotations, source_world),
        "quadrature_s": q_s,
        "quadrature_speed": q_nodes.speed,
        "integrand": q_nodes.integrand,
        "normal_jacobian": q_retraction.normal_jacobians,
        "rho_pose": q_weights["rho_pose"],
        "entry_measure": q_weights["entry_measure"],
        "fresnel_transmission": q_weights["fresnel_transmission"],
        "path_validity": q_weights["path_validity"],
        "source_world": source_world,
        "incident_direction": np.asarray(canonical_incident_direction(), dtype=np.float64),
        "target_direction": target,
        "vertices_body": np.asarray(crystal.vertices, dtype=np.float64),
        "face_numbers": face_numbers,
        "face_normals_body": np.stack([crystal.normal(face) for face in crystal.faces]),
        "face_centers_body": np.stack([crystal.centroid(face) for face in crystal.faces]),
    }
    raw_from_grid = float(cumulative_simpson(q_nodes.weighted, x=q_parameters, initial=0.0)[-1])
    raw_grid_error = abs(raw_from_grid - quadrature.raw_value)
    if raw_grid_error > 1e-12 * max(1.0, abs(quadrature.raw_value)):
        raise RuntimeError(f"canonical quadrature grid reconstruction differs by {raw_grid_error:.3g}")

    metadata = {
        "schema": SCHEMA_VERSION,
        "figure_id": FIGURE_ID,
        "fixture": canonical_fixture_metadata(),
        "sampling": {
            "presentation_node_count": PRESENTATION_NODE_COUNT,
            "selected_indices": SELECTED_INDICES,
            "retraction_iterations": RETRACTION_ITERATIONS,
            "presentation_meaning": "uniform cumulative-chord Hermite grid retracted onto the production fiber",
            "quadrature_node_count": quadrature.node_count,
            "quadrature_grid_independent_of_presentation": quadrature.node_count != PRESENTATION_NODE_COUNT,
        },
        "quadrature": {
            **asdict(quadrature),
            "refined_relative_tolerance": refined.relative_tolerance,
            "refined_node_count": refined.node_count,
            "refined_value": refined.value,
            "refined_error_estimate": refined.error_estimate,
            "default_to_refined_absolute_difference": abs(quadrature.value - refined.value),
            "grid_reconstruction_raw_absolute_difference": raw_grid_error,
        },
        "conventions": {
            "rotation": "R maps body coordinates to world coordinates",
            "source_world": "s_hat points toward the Sun",
            "incident_propagation": "incident_direction = -s_hat",
            "source_body": "u = R^T s_hat",
            "arclength": "cumulative Simpson integral of production ds/dt",
        },
        "geometry": {
            "face_numbers": face_numbers,
            "face_vertex_ids": [list(face.vertex_ids) for face in crystal.faces],
            "edges": [list(edge) for edge in crystal.edges],
        },
        "validation": {
            "presentation_retraction_residual_max_l2": float(np.max(p_retraction.residual_after)),
            "quadrature_retraction_residual_max_l2": float(np.max(q_retraction.residual_after)),
            "presentation_outgoing_direction_max_l2": outgoing_error,
            "integrand_factor_product_max_relative_error": float(
                np.max(
                    np.abs(
                        arrays["rho_pose"]
                        * arrays["entry_measure"]
                        * arrays["fresnel_transmission"]
                        * arrays["path_validity"]
                        / (arrays["normal_jacobian"] + quadrature.epsilon)
                        - arrays["integrand"]
                    )
                    / arrays["integrand"]
                )
            ),
            "minimum_path_validity": float(np.min(arrays["path_validity"])),
            "minimum_speed": float(min(np.min(p_nodes.speed), np.min(q_nodes.speed))),
            "minimum_normal_jacobian": float(np.min(arrays["normal_jacobian"])),
        },
        "source": {
            "upstream_schema": upstream_metadata["schema"],
            "upstream_relative_directory": "../canonical-fiber",
            "upstream_metadata_sha256": sha256(upstream.metadata.read_bytes()).hexdigest(),
            "upstream_arrays_sha256": upstream_metadata["payload"]["sha256"],
            "component_scope": traced.component_scope,
            "component_completeness": traced.component_completeness,
            "original_accepted_revision": ORIGINAL_ACCEPTED_REVISION,
        },
        "provenance": {
            "lumice_integral": _git_state(root),
            "python": sys.version,
            "packages": {name: _package_version(name) for name in ("jax", "numpy", "scipy")},
            "continuation_options": asdict(continuation_options),
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
    return load_data(directory)
