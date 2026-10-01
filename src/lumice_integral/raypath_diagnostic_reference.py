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
from .camera import incident_direction_from_sun
from .dp_field import DPField
from .geometry import HexPrism, Polyhedron
from .path_weight import entry_and_power
from .pose_density import build_pose_density
from .s2_store import align_rotations

SCHEMA_VERSION = "lumice-integral.raypath-diagnostic-reference/v1"
RANDOM_LATTICE_N = 20_000
N_RED = chromatic.N_RED
N_BLUE = chromatic.N_BLUE
INDEX_ENDPOINTS = {"red": N_RED, "blue": N_BLUE}


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


__all__ = [
    "DiagnosticReference",
    "INDEX_ENDPOINTS",
    "N_BLUE",
    "N_RED",
    "RANDOM_LATTICE_N",
    "SCHEMA_VERSION",
    "minimum_deviation_deg",
    "random_orientation_reference",
]
