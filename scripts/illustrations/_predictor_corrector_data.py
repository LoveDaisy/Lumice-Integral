"""Production replay data for the canonical predictor-corrector illustration."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import shlex
from typing import Any, Mapping

import numpy as np

from _canonical_data import IllustrationDataError
from _illustration_io import IllustrationData, load_pair, validate_required_arrays, write_pair


FIGURE_ID = "predictor-corrector"
OUTPUT_NAME = f"{FIGURE_ID}.png"
SCHEMA_VERSION = "lumice-integral.illustration.predictor-corrector/v1"
SEED_PATH = Path(__file__).with_name("data") / "predictor_corrector_seed.json"
SEED_ROTATION_TOLERANCE_RAD = 1e-10

_REQUIRED_ARRAYS = {
    "accepted": ((4, 3, 3), "floating"),
    "predicted": ((3, 3, 3), "floating"),
    "tangents": ((4, 3), "floating"),
    "local_basis": ((3, 2), "floating"),
    "local_curve": ((None, 2), "floating"),
    "local_accepted": ((4, 2), "floating"),
    "local_predicted": ((3, 2), "floating"),
    "global_curve": ((513, 3), "floating"),
    "global_segment": ((4, 3), "floating"),
    "sky": ((3, 3), "floating"),
    "sky_offsets_deg": ((3, 2), "floating"),
    "sky_basis": ((2, 3), "floating"),
    "angular_errors_deg": ((3,), "floating"),
    "phase_condition_errors": ((3,), "floating"),
    "accepted_outgoing_errors": ((4,), "floating"),
}

_ARRAY_SEMANTICS = {
    "accepted": ("dimensionless", "Four sequential accepted body-to-world rotations A through D"),
    "predicted": ("dimensionless", "Three production tangent predictions"),
    "tangents": ("dimensionless", "Production fiber tangents at accepted states"),
    "local_basis": ("dimensionless", "Fixed tangent-normal basis at A for the local pose chart"),
    "local_curve": ("radian", "Production fiber in the fixed local pose chart"),
    "local_accepted": ("radian", "Accepted states A through D in the local pose chart"),
    "local_predicted": ("radian", "Predicted states in the local pose chart"),
    "global_curve": ("radian", "Full canonical fiber in log(R_0^T R) coordinates"),
    "global_segment": ("radian", "Accepted replay states in the global presentation chart"),
    "sky": ("dimensionless", "Apparent sky directions -d for A, first prediction, and B"),
    "sky_offsets_deg": ("degree", "Sky tangent-plane offsets from the fixed target"),
    "sky_basis": ("dimensionless", "Sky tangent-plane basis"),
    "angular_errors_deg": ("degree", "Angular miss from the fixed sky target"),
    "phase_condition_errors": ("radian", "Absolute phase-condition residual for each corrected state"),
    "accepted_outgoing_errors": ("dimensionless", "L2 outgoing residuals at accepted states"),
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
            str(Path(__file__).with_name("predictor_corrector.py").resolve()),
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
    if len(arrays["local_curve"]) < 4:
        raise fail("local_curve is too short to render")


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


def _require_seed_rotation_match(current: np.ndarray, traced: np.ndarray) -> float:
    from scipy.spatial.transform import Rotation

    error = float(Rotation.from_matrix(current.T @ traced).magnitude())
    if error >= SEED_ROTATION_TOLERANCE_RAD:
        raise RuntimeError(
            "frozen predictor seed no longer matches the current canonical accepted trace: "
            f"{error:.3g} rad"
        )
    return error


def recompute_data(
    data_root: Path | str,
    canonical_directory: Path | str | None = None,
) -> IllustrationData:
    from scipy.spatial.transform import Rotation

    from _canonical_data import load_data as load_canonical_data
    from _canonical_data import recompute_data as recompute_canonical_data
    from lumice_integral import continuation
    from lumice_integral.canonical_scene import canonical_pixel_problem
    import jax

    data_root = Path(data_root)
    if canonical_directory is None:
        canonical_directory = recompute_canonical_data(data_root).directory
    common = load_canonical_data(canonical_directory)
    seed_bytes = SEED_PATH.read_bytes()
    seed = json.loads(seed_bytes)
    if seed.get("schema") != "lumice-integral.illustration.predictor-corrector-seed/v1":
        raise RuntimeError("unsupported predictor-corrector seed schema")
    current = np.asarray(seed["current_rotation"], dtype=np.float64)
    tangent = np.asarray(seed["phase_tangent"], dtype=np.float64)
    step = float(seed["step_rad"])
    accepted_index = int(seed["source"]["historical_accepted_index"])
    _require_seed_rotation_match(current, common.arrays["accepted_rotations"][accepted_index])
    tangent_alignment = float(abs(tangent @ common.arrays["accepted_tangents"][accepted_index]))
    if tangent_alignment < 1.0 - 1e-12:
        raise RuntimeError(f"frozen predictor tangent no longer matches the component: {tangent_alignment:.12g}")

    problem = canonical_pixel_problem()
    options = continuation.ContinuationOptions()
    accepted = [current]
    tangents = [tangent]
    predicted = []
    phase_errors = []
    for _ in range(3):
        trial = np.asarray(continuation._predictor_kernel(current, tangent, np.asarray(step)))
        outcome = continuation._correct_trial(problem, options, current, trial, tangent)
        if not outcome.accepted or outcome.state is None:
            raise RuntimeError(f"frozen predictor-corrector replay was rejected: {outcome.message}")
        following = np.asarray(outcome.state.rotation)
        phase_errors.append(
            abs(float(tangent @ Rotation.from_matrix(trial.T @ following).as_rotvec()))
        )
        predicted.append(trial)
        accepted.append(following)
        current = following
        tangent = np.asarray(outcome.state.tangent)
        tangents.append(tangent)
    accepted = np.asarray(accepted)
    predicted = np.asarray(predicted)
    tangents = np.asarray(tangents)

    historical_prediction = np.asarray(seed["historical_predicted_rotation"])
    historical_first_corrected = np.asarray(seed["historical_first_corrected_rotation"])
    prediction_error = float(np.max(np.abs(predicted[0] - historical_prediction)))
    corrected_error = float(np.max(np.abs(accepted[1] - historical_first_corrected)))
    if prediction_error >= 1e-12 or corrected_error >= 1e-12:
        raise RuntimeError(
            "frozen replay drifted from the accepted process anchor: "
            f"prediction={prediction_error:.3g}, corrected={corrected_error:.3g}"
        )

    target = common.arrays["target_direction"]
    accepted_outgoing = np.asarray(jax.vmap(problem.direction_evaluator)(accepted))
    accepted_errors = np.linalg.norm(accepted_outgoing - target, axis=1)
    if np.max(accepted_errors) >= 1e-10 or np.max(phase_errors) >= 1e-11:
        raise RuntimeError("replayed accepted states fail outgoing or phase residual gates")

    local_tangent = tangents[0] / np.linalg.norm(tangents[0])
    correction = Rotation.from_matrix(accepted[0].T @ accepted[1]).as_rotvec()
    normal = correction - local_tangent * (correction @ local_tangent)
    normal /= np.linalg.norm(normal)
    basis = np.stack([local_tangent, normal], axis=-1)

    def local(rotations: np.ndarray) -> np.ndarray:
        return Rotation.from_matrix(accepted[0].T @ rotations).as_rotvec() @ basis

    presentation_rotations = common.arrays["presentation_rotations"]
    all_local = local(presentation_rotations)
    norms = Rotation.from_matrix(accepted[0].T @ presentation_rotations).magnitude()
    mask = (all_local[:, 0] > -0.025) & (all_local[:, 0] < 0.16) & (norms < 0.3)
    local_curve = np.vstack([all_local[mask], local(accepted)])
    local_curve = local_curve[np.argsort(local_curve[:, 0])]

    selected = np.stack([accepted[0], predicted[0], accepted[1]])
    outgoing = np.asarray(jax.vmap(problem.direction_evaluator)(selected))
    sky = -outgoing
    sky_target = -target
    e1 = np.cross([0.0, 0.0, 1.0], sky_target)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(sky_target, e1)
    offsets = np.degrees(
        np.stack(
            [
                np.arctan2(sky @ e1, sky @ sky_target),
                np.arctan2(sky @ e2, sky @ sky_target),
            ],
            axis=-1,
        )
    )
    angular_errors = np.degrees(
        np.arctan2(np.linalg.norm(np.cross(sky, sky_target), axis=1), sky @ sky_target)
    )
    if np.max(angular_errors[[0, 2]]) >= 1e-9:
        raise RuntimeError("accepted sky states do not return to the fixed target")
    historical_miss = 0.04783524364181246
    if abs(float(angular_errors[1]) - historical_miss) >= 1e-6:
        raise RuntimeError(
            f"predicted sky miss {angular_errors[1]:.12g} deg differs from accepted anchor {historical_miss}"
        )

    global_origin = presentation_rotations[0]
    arrays = {
        "accepted": accepted,
        "predicted": predicted,
        "tangents": tangents,
        "local_basis": basis,
        "local_curve": local_curve,
        "local_accepted": local(accepted),
        "local_predicted": local(predicted),
        "global_curve": common.arrays["presentation_relative_rotvec"],
        "global_segment": Rotation.from_matrix(global_origin.T @ accepted).as_rotvec(),
        "sky": sky,
        "sky_offsets_deg": offsets,
        "sky_basis": np.stack([e1, e2]),
        "angular_errors_deg": angular_errors,
        "phase_condition_errors": np.asarray(phase_errors),
        "accepted_outgoing_errors": accepted_errors,
    }
    metadata = {
        "schema": SCHEMA_VERSION,
        "figure_id": FIGURE_ID,
        "fixture": common.metadata["fixture"],
        "source": {
            "canonical_data_sha256": common.metadata["payload"]["sha256"],
            "component_scope": common.metadata["source"]["component_scope"],
            "component_completeness": common.metadata["source"]["component_completeness"],
            "seed_file": str(SEED_PATH.relative_to(Path(__file__).resolve().parents[2])),
            "seed_sha256": sha256(seed_bytes).hexdigest(),
            "seed_provenance": seed["source"],
        },
        "replay": {
            "implementation": "three sequential continuation._predictor_kernel / _correct_trial calls",
            "step_rad": step,
            "continuation_options": {
                key: value
                for key, value in vars(options).items()
            },
            "private_api_scope": "narrow illustration diagnostic; no solver behavior or runtime patching",
        },
        "validation": {
            "historical_prediction_max_matrix_error": prediction_error,
            "historical_first_corrected_max_matrix_error": corrected_error,
            "accepted_outgoing_direction_max_l2": float(np.max(accepted_errors)),
            "phase_condition_error_max_rad": float(np.max(phase_errors)),
            "predicted_sky_miss_deg": float(angular_errors[1]),
            "accepted_sky_miss_max_deg": float(np.max(angular_errors[[0, 2]])),
            "current_component_tangent_alignment": tangent_alignment,
        },
        "conventions": {
            "pose_view": "fixed 2D projection of log(R_A^T R), equal scales; not fiber arclength",
            "sky_view": "tangent-plane angular offsets around apparent target -d, in degrees",
            "correction": "two outgoing residuals plus a phase condition; not a scalar D_P gradient",
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
