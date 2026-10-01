"""Fixed, machine-readable diagnostic references for selected halo ray paths.

This module is an assembler, not another optical solver.  It records two
deliberately narrow reference cases from the public LI computation chains:

``random_regular``
    Randomly oriented regular prisms.  The 3-5 inner edge is checked both by
    the analytic minimum-deviation formula and by :mod:`.focusing`; 3-1-5
    keeps its solar-side dispersion edge, its still-candidate focusing label,
    its antisolar internal-TIR blue band, and the assessed red exit gate as
    separate records.

``plate_rhombic_9``
    Filled in by :func:`plate_reference`: the fixed rhombic prism and ideal
    horizontal-plate family used for the two 120-degree target points.

Every conclusion carries a status and a coverage statement.  Arrays which
identify poses or samples stay outside the JSON-ready metadata and are
returned in :class:`DiagnosticReference.arrays`.  No function imports,
links, or invokes Lumice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from . import chromatic, focusing, optics
from .camera import incident_direction_from_sun, rotation_about_axis, sun_direction
from .dp_field import DPField
from .geometry import HexPrism, Polyhedron, fold_matrix, halo_map_rank, wedge_angle_deg
from .path_weight import entry_and_power, weighted_power
from .pose_density import build_pose_density
from .s2_store import align_rotations
from .symmetry.reflection_group import commutes_with_rz

SCHEMA_VERSION = "lumice-integral.raypath-diagnostic-reference/v1"
RANDOM_LATTICE_N = 20_000
N_RED = chromatic.N_RED
N_BLUE = chromatic.N_BLUE
INDEX_ENDPOINTS = {"red": N_RED, "blue": N_BLUE}
PLATE_FACE_DISTANCE = (1.5, 1.0, 1.0, 1.5, 1.0, 1.0)
PLATE_SUN_ALTITUDE_DEG = 9.0
PLATE_SUN_AZIMUTH_DEG = 180.0
PLATE_TARGET_AZIMUTH_DEG = {"plus": 300.0, "minus": 60.0}
PLATE_CLASSES = {"white": (1, 3, 4, 2), "blue": (1, 3, 5, 2)}
PLATE_COARSE_N = 4096
PLATE_FINE_N = 8192
SUPPORT_ENDPOINT_TOLERANCE_RAD = 1.0e-8
TARGET_RESIDUAL_TOLERANCE = 1.0e-10


@dataclass(frozen=True, eq=False)
class DiagnosticReference:
    """JSON-ready metadata plus named numerical arrays."""

    metadata: dict[str, Any]
    arrays: dict[str, np.ndarray] = field(default_factory=dict)


def minimum_deviation_deg(index: float) -> float:
    """Minimum deviation of a 60-degree ice prism, in degrees.

    This analytic chain is independent of the AD/field critical-point chain:
    ``2 asin(n sin(30 deg)) - 60 deg``.
    """

    index = float(index)
    argument = index * np.sin(np.radians(30.0))
    if not np.isfinite(index) or not 0.0 <= argument <= 1.0:
        raise ValueError("index must give a real 60-degree-prism minimum deviation")
    return float(np.degrees(2.0 * np.arcsin(argument) - np.radians(60.0)))


def _feature_as_json(feature: chromatic.ChromaticFeature) -> dict[str, Any]:
    return {
        "kind": feature.kind,
        "source": feature.source,
        "color": feature.color,
        "visible": bool(feature.visible),
        "positive_fraction": feature.positive_fraction,
        "delta_red_deg": float(np.degrees(feature.delta_red)),
        "delta_blue_deg": float(np.degrees(feature.delta_blue)),
        "shift_deg": float(np.degrees(feature.shift)),
        "spread_deg": float(np.degrees(feature.spread)),
        "direction_dispersion_deg": float(np.degrees(feature.direction_dispersion)),
        "contrast": feature.contrast,
        "weight": feature.weight,
        "lit_fraction": feature.lit_fraction,
        "score": feature.score,
    }


def _onset(table: focusing.WavelengthCriticalTable, source: str) -> focusing.WavelengthOnsetShift:
    matches = [row for row in table.onsets if row.source == source]
    if len(matches) != 1:
        raise ValueError(f"expected one {source!r} onset for {table.path}, found {len(matches)}")
    return matches[0]


def _internal_reflection_counterfactual(
    crystal: Polyhedron,
    faces: Sequence[int],
    lattice_n: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Compare production ``A*T`` with the same path after removing only internal ``R`` factors.

    Samples are the blue-index TIR-onset curve.  Geometry, domain checks,
    entry measure, and entry/exit Fresnel transmission are unchanged; only
    the product of internal reflectances is omitted in the counterfactual.
    """

    fields = {
        label: DPField.build(crystal, faces, index, lattice_n=lattice_n)
        for label, index in INDEX_ENDPOINTS.items()
    }
    blue_kinks = [kink for kink in fields["blue"].weight_kinks if kink.margin == "internal_1_tir_discriminant"]
    if len(blue_kinks) != 1 or not blue_kinks[0].arcs:
        raise ValueError("3-1-5 blue internal-reflection kink is unavailable")
    points = np.asarray(blue_kinks[0].points, dtype=np.float64)
    probe_sun = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    rotations = align_rotations(points, probe_sun)
    incident = incident_direction_from_sun(probe_sun)

    arrays: dict[str, np.ndarray] = {
        "random_315_counterfactual_u": points,
        "random_315_counterfactual_rotations": rotations,
    }
    summary: dict[str, Any] = {
        "sample": "blue-index internal_1_tir_discriminant curve",
        "sample_count": int(len(points)),
        "removed_factor": "internal_1_reflectance",
        "unchanged": ["path_domain", "entry_measure_A", "entry_transmission", "exit_transmission", "sample_poses"],
        "indices": {},
    }
    for label, index in INDEX_ENDPOINTS.items():
        area, production_power = entry_and_power(rotations, faces, incident, index, crystal=crystal)
        check = optics.path_domain_batch(rotations, faces, incident, index, crystal=crystal)
        valid = np.asarray(check.valid)
        margins = check.margins
        entry = optics.fresnel_unpolarized_transmittance(
            1.0,
            np.where(valid, margins["entry_incidence_cosine"], 1.0),
            index,
            np.sqrt(np.where(valid, margins["entry_snell_discriminant"], 1.0)),
        )
        exit_ = optics.fresnel_unpolarized_transmittance(
            index,
            np.where(valid, margins["exit_incidence_cosine"], 1.0),
            1.0,
            np.sqrt(np.where(valid, margins["exit_snell_discriminant"], 1.0)),
        )
        without_internal_r = np.asarray(area) * np.where(valid, entry * exit_, 0.0)
        production = np.asarray(area) * np.asarray(production_power)
        arrays[f"random_315_counterfactual_A_{label}"] = np.asarray(area)
        arrays[f"random_315_counterfactual_T_{label}"] = np.asarray(production_power)
        arrays[f"random_315_counterfactual_AT_{label}"] = production
        arrays[f"random_315_counterfactual_AT_without_internal_R_{label}"] = without_internal_r
        arrays[f"random_315_counterfactual_valid_{label}"] = valid
        summary["indices"][label] = {
            "n": index,
            "valid_count": int(np.count_nonzero(valid)),
            "production_mean_AT": float(np.mean(production)),
            "without_internal_R_mean_AT": float(np.mean(without_internal_r)),
        }
    red = summary["indices"]["red"]
    blue = summary["indices"]["blue"]
    summary["blue_red_ratio"] = {
        "production": blue["production_mean_AT"] / red["production_mean_AT"],
        "without_internal_R": blue["without_internal_R_mean_AT"] / red["without_internal_R_mean_AT"],
    }
    return summary, arrays


def random_orientation_reference(*, lattice_n: int = RANDOM_LATTICE_N) -> DiagnosticReference:
    """Build the regular-prism random-orientation part of the reference."""

    if lattice_n < 1000:
        raise ValueError("lattice_n must be at least 1000 for the diagnostic reference")
    crystal = HexPrism(a=1.0, h=1.0)
    density = build_pose_density("random")
    table_35 = focusing.wavelength_critical_table(crystal, (3, 5), density, INDEX_ENDPOINTS)
    table_315 = focusing.wavelength_critical_table(crystal, (3, 1, 5), density, INDEX_ENDPOINTS)
    inner_35 = _onset(table_35, "interior_minimum")
    candidate_315 = _onset(table_315, "boundary_extremum")
    diagnostic_315 = chromatic.diagnose(crystal, (3, 1, 5), lattice_n=lattice_n)
    kink = [feature for feature in diagnostic_315.features if feature.kind == "edge"]
    gate = [feature for feature in diagnostic_315.features if feature.kind == "gate_edge"]
    if len(kink) != 1 or len(gate) != 1:
        raise ValueError(f"expected one 3-1-5 kink and one moving gate, found {len(kink)} and {len(gate)}")
    analytic = {label: minimum_deviation_deg(index) for label, index in INDEX_ENDPOINTS.items()}
    field_values = inner_35.values_deg
    residuals = {label: field_values[label] - analytic[label] for label in INDEX_ENDPOINTS}
    counterfactual, arrays = _internal_reflection_counterfactual(crystal, (3, 1, 5), lattice_n)

    features = [
        {
            "id": "random_regular.3-5.inner_edge",
            "path": "3-5",
            "location": "solar-side circular inner edge",
            "mechanism": "ordinary minimum-deviation dispersion",
            "evidence_status": "confirmed",
            "indices": dict(INDEX_ENDPOINTS),
            "position_deg": analytic,
            "field_position_deg": dict(field_values),
            "analytic_minus_field_residual_deg": {label: -residuals[label] for label in INDEX_ENDPOINTS},
            "focusing_profile": inner_35.profile,
            "jacobian_focusing": inner_35.jacobian_focusing,
            "interpretation": "finite jump at a non-degenerate minimum; not a divergent Jacobian caustic",
            "tolerance": {
                "absolute_deg": 2.0e-8,
                "source": "float64 analytic formula versus independently evaluated DPField critical point",
            },
        },
        {
            "id": "random_regular.3-1-5.solar_dispersion_edge",
            "path": "3-1-5",
            "location": "solar side",
            "mechanism": "ordinary minimum-deviation dispersion",
            "evidence_status": "confirmed",
            "indices": dict(INDEX_ENDPOINTS),
            "position_deg": analytic,
            "interpretation": "the path retains a solar-side red-to-blue dispersion edge; this is separate from the antisolar TIR band",
        },
        {
            "id": "random_regular.3-1-5.solar_caustic_candidate",
            "path": "3-1-5",
            "location": "solar side",
            "mechanism": "DPField boundary critical record",
            "evidence_status": "candidate",
            "position_deg": dict(candidate_315.values_deg),
            "focusing_profile": candidate_315.profile,
            "jacobian_focusing": candidate_315.jacobian_focusing,
            "interpretation": "field classification alone is insufficient evidence of a visible caustic",
        },
        {
            "id": "random_regular.3-1-5.antisolar_tir_blue_band",
            "path": "3-1-5",
            "location": "antisolar side",
            "mechanism": "internal-reflection Fresnel TIR kink",
            "evidence_status": "confirmed",
            **_feature_as_json(kink[0]),
        },
        {
            "id": "random_regular.3-1-5.exit_gate",
            "path": "3-1-5",
            "location": "assessed moving exit gate",
            "mechanism": "exit Snell gate",
            "evidence_status": "confirmed",
            "interpretation": "red candidate assessed as not visible because its D spread overwhelms its shift",
            **_feature_as_json(gate[0]),
        },
    ]
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "case_id": "random_regular",
        "configuration": {
            "crystal": {"kind": "HexPrism", "a": 1.0, "h": 1.0, "face_distance": [1.0] * 6},
            "pose_measure": "normalised Haar on SO(3)",
            "indices": dict(INDEX_ENDPOINTS),
            "lattice_n": int(lattice_n),
        },
        "features": features,
        "internal_reflection_counterfactual": counterfactual,
        "coverage": {
            "status": "finite numerical reference",
            "complete_for": ["3-5 minimum-deviation endpoint pair", "reported 3-1-5 DPField kink and moving gates"],
            "not_evaluated": [
                "finite solar disc convolution",
                "absolute visual prominence of the candidate caustic",
                "all-sky feature enumeration",
            ],
            "kink_walk_complete": bool(diagnostic_315.coverage_complete),
        },
    }
    return DiagnosticReference(metadata, arrays)


def _rz_rotations(theta: np.ndarray) -> np.ndarray:
    theta = np.asarray(theta, dtype=np.float64)
    rotations = np.zeros(theta.shape + (3, 3), dtype=np.float64)
    cosine, sine = np.cos(theta), np.sin(theta)
    rotations[..., 0, 0] = cosine
    rotations[..., 0, 1] = -sine
    rotations[..., 1, 0] = sine
    rotations[..., 1, 1] = cosine
    rotations[..., 2, 2] = 1.0
    return rotations


def _midpoint_grid(count: int) -> tuple[np.ndarray, np.ndarray]:
    theta = 2.0 * np.pi * (np.arange(count, dtype=np.float64) + 0.5) / count
    return theta, _rz_rotations(theta)


def _target_assignment(fixed_direction: np.ndarray, targets: Mapping[str, np.ndarray]) -> tuple[str, dict[str, float]]:
    residuals = {name: float(np.linalg.norm(fixed_direction - target)) for name, target in targets.items()}
    matches = [name for name, residual in residuals.items() if residual <= TARGET_RESIDUAL_TOLERANCE]
    if len(matches) > 1:
        raise ValueError(f"fixed direction ambiguously matches targets: {residuals}")
    return (matches[0] if matches else "other"), residuals


def _one_weight(
    theta: float,
    faces: Sequence[int],
    incident: np.ndarray,
    index: float,
    crystal: Polyhedron,
) -> float:
    value = weighted_power(_rz_rotations(np.array([theta])), faces, incident, index, crystal=crystal)
    return float(value[0])


def _support_intervals(
    theta: np.ndarray,
    weights: np.ndarray,
    faces: Sequence[int],
    incident: np.ndarray,
    index: float,
    crystal: Polyhedron,
) -> list[dict[str, Any]]:
    """Periodic positive-support intervals with transition brackets refined by bisection.

    Discovery is limited to the supplied midpoint grid.  Bisection refines
    only transitions that grid already found; it is not a topology or
    completeness certificate.
    """

    active = np.asarray(weights) > 0.0
    if not np.any(active):
        return []
    if np.all(active):
        return [{"start_rad": 0.0, "end_rad": 2.0 * np.pi, "wraps_period": False, "transition_brackets": []}]

    transitions: list[dict[str, Any]] = []
    count = len(theta)
    for left_index in range(count):
        right_index = (left_index + 1) % count
        if bool(active[left_index]) == bool(active[right_index]):
            continue
        lo = float(theta[left_index])
        hi = float(theta[right_index])
        if right_index == 0:
            hi += 2.0 * np.pi
        left_active = bool(active[left_index])
        right_active = bool(active[right_index])
        left_weight = float(weights[left_index])
        right_weight = float(weights[right_index])
        while hi - lo > SUPPORT_ENDPOINT_TOLERANCE_RAD:
            mid = 0.5 * (lo + hi)
            mid_weight = _one_weight(mid % (2.0 * np.pi), faces, incident, index, crystal)
            if (mid_weight > 0.0) == left_active:
                lo, left_weight = mid, mid_weight
            else:
                hi, right_weight = mid, mid_weight
        transitions.append(
            {
                "theta_rad": float(0.5 * (lo + hi) % (2.0 * np.pi)),
                "left_active": left_active,
                "right_active": right_active,
                "bracket": [lo, hi],
                "bracket_width_rad": hi - lo,
                "left_weight": left_weight,
                "right_weight": right_weight,
            }
        )
    transitions.sort(key=lambda item: item["theta_rad"])
    starts = [item for item in transitions if not item["left_active"] and item["right_active"]]
    ends = [item for item in transitions if item["left_active"] and not item["right_active"]]
    if len(starts) != len(ends):
        raise RuntimeError(f"periodic support transitions do not alternate for {tuple(faces)}")
    intervals: list[dict[str, Any]] = []
    for start in starts:
        following = [end for end in ends if end["theta_rad"] > start["theta_rad"]]
        end = following[0] if following else ends[0]
        end_unwrapped = end["theta_rad"]
        wraps = end_unwrapped <= start["theta_rad"]
        if wraps:
            end_unwrapped += 2.0 * np.pi
        intervals.append(
            {
                "start_rad": start["theta_rad"],
                "end_rad": end_unwrapped,
                "wraps_period": wraps,
                "transition_brackets": [start, end],
            }
        )
    return intervals


def _member_record(
    crystal: Polyhedron,
    faces: tuple[int, ...],
    incident: np.ndarray,
    targets: Mapping[str, np.ndarray],
    grids: Mapping[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]]]:
    fold = fold_matrix(crystal, faces)
    fixed = fold @ incident
    assignment, target_residuals = _target_assignment(fixed, targets)
    weights: dict[str, dict[str, np.ndarray]] = {"coarse": {}, "fine": {}}
    energies: dict[str, Any] = {}
    support: dict[str, Any] = {}
    for label, index in INDEX_ENDPOINTS.items():
        for resolution, (_, rotations) in grids.items():
            weights[resolution][label] = weighted_power(rotations, faces, incident, index, crystal=crystal)
        coarse = float(np.mean(weights["coarse"][label]))
        fine = float(np.mean(weights["fine"][label]))
        energies[label] = {"coarse": coarse, "fine": fine, "absolute_difference": abs(fine - coarse)}
        support[label] = _support_intervals(
            grids["fine"][0], weights["fine"][label], faces, incident, index, crystal
        )
    production_residual: dict[str, float | None] = {}
    production_valid_count: dict[str, int] = {}
    for label, index in INDEX_ENDPOINTS.items():
        physical = np.flatnonzero(weights["fine"][label] > 0.0)
        if len(physical) == 0:
            production_residual[label] = None
            production_valid_count[label] = 0
            continue
        selected = physical[np.linspace(0, len(physical) - 1, min(3, len(physical)), dtype=int)]
        probe_rotations = grids["fine"][1][selected]
        check = optics.path_domain_batch(probe_rotations, faces, incident, index, crystal=crystal)
        if not np.all(check.valid):
            raise RuntimeError(f"positive A*T sample is outside the optical domain for {faces} at n={index}")
        production_residual[label] = float(np.max(np.linalg.norm(check.direction - fixed, axis=1)))
        production_valid_count[label] = int(len(selected))
    record = {
        "faces": list(faces),
        "orbit_kind": "L1/PBD",
        "halo_map_rank": halo_map_rank(crystal, faces),
        "wedge_deg": wedge_angle_deg(crystal, faces),
        "fold_matrix": fold.tolist(),
        "commutator_norm_at_37deg": float(
            np.linalg.norm(fold @ _rz_rotations(np.array([np.radians(37.0)]))[0] - _rz_rotations(np.array([np.radians(37.0)]))[0] @ fold)
        ),
        "commutes_with_rz": bool(commutes_with_rz(fold)),
        "fixed_outgoing_direction": fixed.tolist(),
        "target_assignment": assignment,
        "target_residuals": target_residuals,
        "production_direction_residual_max": production_residual,
        "production_probe_valid_count": production_valid_count,
        "energy": energies,
        "support_intervals": support,
    }
    return record, weights


def _summarise_target_energy(members: Sequence[dict[str, Any]]) -> dict[str, Any]:
    energy: dict[str, Any] = {}
    for target in ("plus", "minus", "other"):
        energy[target] = {}
        selected = [member for member in members if member["target_assignment"] == target]
        for label in INDEX_ENDPOINTS:
            coarse = sum(member["energy"][label]["coarse"] for member in selected)
            fine = sum(member["energy"][label]["fine"] for member in selected)
            energy[target][label] = {
                "coarse": coarse,
                "fine": fine,
                "absolute_difference": abs(fine - coarse),
            }
    energy["class_total"] = {}
    for label in INDEX_ENDPOINTS:
        coarse = sum(energy[target][label]["coarse"] for target in ("plus", "minus", "other"))
        fine = sum(energy[target][label]["fine"] for target in ("plus", "minus", "other"))
        energy["class_total"][label] = {
            "coarse": coarse,
            "fine": fine,
            "absolute_difference": abs(fine - coarse),
        }
    ratios = {}
    for target in ("plus", "minus", "other"):
        red = energy[target]["red"]["fine"]
        blue = energy[target]["blue"]["fine"]
        if red > 0.0:
            ratios[target] = {"status": "available", "blue_red": blue / red}
        elif blue > 0.0:
            ratios[target] = {"status": "one_sided", "blue_red": None}
        else:
            ratios[target] = {"status": "unavailable", "blue_red": None}
    energy["target_ratios"] = ratios
    return energy


def _class_snapshots(
    label: str,
    members: Sequence[dict[str, Any]],
    weights_by_member: Mapping[tuple[int, ...], dict[str, dict[str, np.ndarray]]],
    grids: Mapping[str, tuple[np.ndarray, np.ndarray]],
    crystal: Polyhedron,
    sun: np.ndarray,
    incident: np.ndarray,
    targets: Mapping[str, np.ndarray],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    selected = next(
        member
        for member in members
        if member["energy"]["red"]["fine"] > 0.0 and member["energy"]["blue"]["fine"] > 0.0
    )
    faces = tuple(selected["faces"])
    weights = weights_by_member[faces]
    common = (weights["fine"]["red"] > 0.0) & (weights["fine"]["blue"] > 0.0)
    candidates = np.flatnonzero(common)
    if len(candidates) < 3:
        raise RuntimeError(f"{label} class has no sampled common red/blue support arc")
    chosen = candidates[np.linspace(0, len(candidates) - 1, 5, dtype=int)]
    theta = grids["fine"][0][chosen]
    rotations = grids["fine"][1][chosen]
    body_sun = np.einsum("nji,j->ni", rotations, sun)
    directions = {}
    residuals = {}
    for index_label, index in INDEX_ENDPOINTS.items():
        check = optics.path_domain_batch(rotations, faces, incident, index, crystal=crystal)
        directions[index_label] = check.direction
        residuals[index_label] = np.linalg.norm(check.direction - targets[selected["target_assignment"]], axis=1)

    negative_rotation = rotation_about_axis(sun, 10.0) @ rotations[0]
    negative_body_sun = negative_rotation.T @ sun
    negative_checks = {
        index_label: optics.path_domain_batch(
            negative_rotation[None, :, :], faces, incident, index, crystal=crystal
        )
        for index_label, index in INDEX_ENDPOINTS.items()
    }
    negative = {
        "left_world_sun_axis_rotation_deg": 10.0,
        "body_sun_projection_residual": float(np.linalg.norm(negative_body_sun - body_sun[0])),
        "horizontal_family_residual": float(np.linalg.norm(negative_rotation @ np.array([0.0, 0.0, 1.0]) - np.array([0.0, 0.0, 1.0]))),
        "target_residual": {
            index_label: float(np.linalg.norm(check.direction[0] - targets[selected["target_assignment"]]))
            for index_label, check in negative_checks.items()
        },
        "valid": {index_label: bool(check.valid[0]) for index_label, check in negative_checks.items()},
    }
    prefix = f"plate_{label}_snapshots"
    arrays = {
        f"{prefix}_theta_rad": theta,
        f"{prefix}_u": body_sun,
        f"{prefix}_rotations": rotations,
        f"{prefix}_outgoing_red": directions["red"],
        f"{prefix}_outgoing_blue": directions["blue"],
        f"{prefix}_A_red": entry_and_power(rotations, faces, incident, N_RED, crystal=crystal)[0],
        f"{prefix}_T_red": entry_and_power(rotations, faces, incident, N_RED, crystal=crystal)[1],
        f"{prefix}_A_blue": entry_and_power(rotations, faces, incident, N_BLUE, crystal=crystal)[0],
        f"{prefix}_T_blue": entry_and_power(rotations, faces, incident, N_BLUE, crystal=crystal)[1],
    }
    metadata = {
        "member": list(faces),
        "target_assignment": selected["target_assignment"],
        "array_prefix": prefix,
        "pose_count": int(len(theta)),
        "max_R_u_minus_sun": float(np.max(np.linalg.norm(np.einsum("nij,nj->ni", rotations, body_sun) - sun, axis=1))),
        "max_R_ez_minus_ez": float(
            np.max(np.linalg.norm(np.einsum("nij,j->ni", rotations, np.array([0.0, 0.0, 1.0])) - np.array([0.0, 0.0, 1.0]), axis=1))
        ),
        "max_target_residual": {key: float(np.max(value)) for key, value in residuals.items()},
        "negative_pose": negative,
    }
    return metadata, arrays


def plate_reference(
    *,
    coarse_n: int = PLATE_COARSE_N,
    fine_n: int = PLATE_FINE_N,
    auxiliary_samples: int = 4096,
) -> DiagnosticReference:
    """Build the fixed rhombic-prism, ideal-horizontal-plate 120-degree reference."""

    if coarse_n < 128 or fine_n < 2 * coarse_n:
        raise ValueError("plate reference requires coarse_n >= 128 and fine_n >= 2 * coarse_n")
    crystal = HexPrism(a=1.0, h=2.0, face_distance=PLATE_FACE_DISTANCE)
    sun = sun_direction(PLATE_SUN_ALTITUDE_DEG, PLATE_SUN_AZIMUTH_DEG)
    incident = incident_direction_from_sun(sun)
    sky_targets = {
        name: sun_direction(PLATE_SUN_ALTITUDE_DEG, azimuth)
        for name, azimuth in PLATE_TARGET_AZIMUTH_DEG.items()
    }
    targets = {name: incident_direction_from_sun(sky) for name, sky in sky_targets.items()}
    grids = {"coarse": _midpoint_grid(coarse_n), "fine": _midpoint_grid(fine_n)}
    arrays: dict[str, np.ndarray] = {}
    classes: dict[str, Any] = {}
    for class_label, representative in PLATE_CLASSES.items():
        records: list[dict[str, Any]] = []
        member_weights: dict[tuple[int, ...], dict[str, dict[str, np.ndarray]]] = {}
        for faces in chromatic.class_members(representative):
            record, weights = _member_record(crystal, faces, incident, targets, grids)
            if record["wedge_deg"] > 1.0e-9 or not record["commutes_with_rz"]:
                record["algorithm_status"] = "algorithm_not_covered"
            else:
                record["algorithm_status"] = "covered_constant_direction"
            records.append(record)
            member_weights[faces] = weights
        covered = [record for record in records if record["algorithm_status"] == "covered_constant_direction"]
        if len(covered) != len(records):
            raise RuntimeError(f"fixed {class_label} class contains members outside the constant-direction algorithm")
        if not any(record["energy"]["red"]["fine"] > 0.0 for record in records):
            raise RuntimeError(f"fixed {class_label} class has no physically lit red member")
        energy = _summarise_target_energy(records)
        snapshots, snapshot_arrays = _class_snapshots(
            class_label, records, member_weights, grids, crystal, sun, incident, targets
        )
        arrays.update(snapshot_arrays)
        auxiliary = chromatic.diagnose_class(
            crystal,
            representative,
            chromatic.PlateFamily(PLATE_SUN_ALTITUDE_DEG, 0.0, auxiliary_samples, 3),
        )
        classes[class_label] = {
            "representative": list(representative),
            "orbit_kind": "L1/PBD label orbit; no L2 physical-equivalence claim",
            "members": records,
            "lit_members": {
                label: [record["faces"] for record in records if record["energy"][label]["fine"] > 0.0]
                for label in INDEX_ENDPOINTS
            },
            "target_energy": energy,
            "snapshots": snapshots,
            "auxiliary_diagnose_class": {
                "role": "qualitative class-level cross-check, not the per-target energy authority",
                "samples": auxiliary_samples,
                "lit_members": {
                    key: [list(member) for member in value] for key, value in auxiliary.lit_members.items()
                },
                "kind": auxiliary.verdict.kind,
                "color": auxiliary.verdict.color,
                "ratio": None if auxiliary.verdict.tint is None else auxiliary.verdict.tint.ratio,
            },
        }

    spherical_separation = {
        name: float(
            np.degrees(
                np.arctan2(np.linalg.norm(np.cross(sun, sky)), float(np.dot(sun, sky)))
            )
        )
        for name, sky in sky_targets.items()
    }
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "case_id": "plate_rhombic_9",
        "configuration": {
            "crystal": {
                "kind": "HexPrism",
                "a": 1.0,
                "h": 2.0,
                "face_distance": list(PLATE_FACE_DISTANCE),
            },
            "sun": {"altitude_deg": PLATE_SUN_ALTITUDE_DEG, "azimuth_deg": PLATE_SUN_AZIMUTH_DEG},
            "pose_measure": "Rz(theta), theta uniform under dtheta/(2*pi); c axis exactly vertical",
            "indices": dict(INDEX_ENDPOINTS),
            "quadrature": {"coarse_midpoints": coarse_n, "fine_midpoints": fine_n},
            "support_endpoint_tolerance_rad": SUPPORT_ENDPOINT_TOLERANCE_RAD,
            "target_residual_tolerance": TARGET_RESIDUAL_TOLERANCE,
        },
        "targets": {
            name: {
                "relative_solar_azimuth_deg": 120.0 if name == "plus" else -120.0,
                "sky_azimuth_deg": PLATE_TARGET_AZIMUTH_DEG[name],
                "sky_direction": sky_targets[name].tolist(),
                "propagation_direction": targets[name].tolist(),
                "spherical_separation_from_sun_deg": spherical_separation[name],
            }
            for name in ("plus", "minus")
        },
        "classes": classes,
        "coverage": {
            "status": "resolution_limited",
            "grid_step_rad": 2.0 * np.pi / fine_n,
            "endpoint_bracket_max_width_rad": SUPPORT_ENDPOINT_TOLERANCE_RAD,
            "statement": "bisection refines only support transitions found on the periodic fine grid",
            "not_claimed": [
                "absence of support intervals narrower than one grid step",
                "general SO(3) target relation",
                "L2 physical equivalence of the L1/PBD orbit",
                "rank-0 collapse",
            ],
        },
    }
    return DiagnosticReference(metadata, arrays)


def build_reference(
    *,
    random_lattice_n: int = RANDOM_LATTICE_N,
    plate_coarse_n: int = PLATE_COARSE_N,
    plate_fine_n: int = PLATE_FINE_N,
    auxiliary_samples: int = 4096,
) -> DiagnosticReference:
    """Build both fixed cases into the versioned top-level reference."""

    random = random_orientation_reference(lattice_n=random_lattice_n)
    plate = plate_reference(coarse_n=plate_coarse_n, fine_n=plate_fine_n, auxiliary_samples=auxiliary_samples)
    return DiagnosticReference(
        {
            "schema_version": SCHEMA_VERSION,
            "configuration": {"index_endpoints": dict(INDEX_ENDPOINTS)},
            "cases": {"random_regular": random.metadata, "plate_rhombic_9": plate.metadata},
            "limitations": [
                "fixed diagnostic cases, not a general feature enumerator",
                "numerical convergence evidence and residuals, never exact numerical results",
                "no Lumice runtime or production dependency",
            ],
        },
        {**random.arrays, **plate.arrays},
    )


__all__ = [
    "DiagnosticReference",
    "INDEX_ENDPOINTS",
    "N_BLUE",
    "N_RED",
    "PLATE_CLASSES",
    "PLATE_COARSE_N",
    "PLATE_FACE_DISTANCE",
    "PLATE_FINE_N",
    "RANDOM_LATTICE_N",
    "SCHEMA_VERSION",
    "SUPPORT_ENDPOINT_TOLERANCE_RAD",
    "TARGET_RESIDUAL_TOLERANCE",
    "build_reference",
    "minimum_deviation_deg",
    "plate_reference",
    "random_orientation_reference",
]
