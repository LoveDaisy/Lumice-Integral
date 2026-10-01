"""Mechanism-level tests for the fixed ray-path diagnostic reference."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from lumice_integral import raypath_diagnostic_reference as reference
from lumice_integral.provenance import sha256_of


@pytest.fixture(scope="module")
def random_case() -> reference.DiagnosticReference:
    return reference.random_orientation_reference(lattice_n=5000)


def _feature(case: reference.DiagnosticReference, identifier: str) -> dict:
    return next(feature for feature in case.metadata["features"] if feature["id"] == identifier)


def test_3_5_inner_edge_has_independent_analytic_and_field_evidence(random_case) -> None:
    feature = _feature(random_case, "random_regular.3-5.inner_edge")
    assert feature["evidence_status"] == "confirmed"
    assert feature["focusing_profile"] == "finite_jump"
    assert not feature["jacobian_focusing"]
    for label in ("red", "blue"):
        assert feature["position_deg"][label] == pytest.approx(feature["field_position_deg"][label], abs=2e-8)
    assert feature["position_deg"]["blue"] > feature["position_deg"]["red"]


def test_3_1_5_preserves_solar_and_antisolar_records(random_case) -> None:
    solar = _feature(random_case, "random_regular.3-1-5.solar_dispersion_edge")
    candidate = _feature(random_case, "random_regular.3-1-5.solar_caustic_candidate")
    blue = _feature(random_case, "random_regular.3-1-5.antisolar_tir_blue_band")
    gate = _feature(random_case, "random_regular.3-1-5.exit_gate")
    assert solar["evidence_status"] == "confirmed" and solar["position_deg"]["blue"] > solar["position_deg"]["red"]
    assert candidate["evidence_status"] == "candidate"
    assert candidate["jacobian_focusing"] and candidate["focusing_profile"] == "degenerate"
    assert blue["source"] == "internal_1_tir_discriminant"
    assert blue["color"] == "blue" and blue["visible"]
    assert blue["spread_deg"] < blue["shift_deg"]
    assert gate["source"] == "exit_snell_discriminant"
    assert gate["color"] == "red" and not gate["visible"]


def test_removing_only_internal_R_destroys_the_blue_counterfactual(random_case) -> None:
    counterfactual = random_case.metadata["internal_reflection_counterfactual"]
    ratios = counterfactual["blue_red_ratio"]
    assert ratios["production"] > 1.8
    assert ratios["without_internal_R"] < 0.9
    assert counterfactual["indices"]["red"]["valid_count"] == counterfactual["sample_count"]
    assert counterfactual["indices"]["blue"]["valid_count"] == counterfactual["sample_count"]
    assert np.array_equal(
        random_case.arrays["random_315_counterfactual_valid_red"],
        random_case.arrays["random_315_counterfactual_valid_blue"],
    )


def test_index_mutation_moves_the_analytic_edge() -> None:
    baseline = reference.minimum_deviation_deg(reference.N_RED)
    assert reference.minimum_deviation_deg(reference.N_RED + 0.001) > baseline
    with pytest.raises(ValueError):
        reference.minimum_deviation_deg(3.0)


@pytest.fixture(scope="module")
def plate_case() -> reference.DiagnosticReference:
    return reference.plate_reference(coarse_n=256, fine_n=512, auxiliary_samples=512)


def test_plate_members_obey_the_constant_direction_structure(plate_case) -> None:
    for class_record in plate_case.metadata["classes"].values():
        assert len(class_record["members"]) == 24
        assignments = {member["target_assignment"] for member in class_record["members"]}
        assert assignments == {"plus", "minus"}
        for member in class_record["members"]:
            assert member["algorithm_status"] == "covered_constant_direction"
            assert member["wedge_deg"] <= 1e-9
            assert member["commutes_with_rz"]
            residuals = [value for value in member["production_direction_residual_max"].values() if value is not None]
            if residuals:
                assert max(residuals) < 5e-14
            else:
                assert member["production_probe_valid_count"] == {"red": 0, "blue": 0}
            assert member["halo_map_rank"] == 2


def test_plate_target_energy_is_partitioned_per_sky_point(plate_case) -> None:
    classes = plate_case.metadata["classes"]
    for label, expected_ratio in (("white", 1.0), ("blue", 1.5)):
        record = classes[label]
        assert record["lit_members"]["red"] and record["lit_members"]["blue"]
        energy = record["target_energy"]
        for index_label in ("red", "blue"):
            split = sum(energy[target][index_label]["fine"] for target in ("plus", "minus", "other"))
            assert split == pytest.approx(energy["class_total"][index_label]["fine"], abs=1e-15)
        for target in ("plus", "minus"):
            ratio = energy["target_ratios"][target]
            assert ratio["status"] == "available"
            assert ratio["blue_red"] == pytest.approx(expected_ratio, rel=0.08)
        assert energy["target_ratios"]["other"]["status"] == "unavailable"


def test_plate_snapshots_stay_on_target_and_negative_pose_breaks_family(plate_case) -> None:
    for label, record in plate_case.metadata["classes"].items():
        snapshots = record["snapshots"]
        assert max(snapshots["max_target_residual"].values()) < 5e-14
        assert snapshots["max_R_u_minus_sun"] < 5e-16
        assert snapshots["max_R_ez_minus_ez"] < 5e-16
        negative = snapshots["negative_pose"]
        assert negative["body_sun_projection_residual"] < 5e-16
        assert negative["horizontal_family_residual"] > 0.01
        assert min(negative["target_residual"].values()) > 0.01
        arrays = plate_case.arrays
        prefix = snapshots["array_prefix"]
        assert arrays[f"{prefix}_rotations"].shape == (5, 3, 3)
        assert np.all(arrays[f"{prefix}_A_red"] > 0.0)
        assert np.all(arrays[f"{prefix}_T_red"] > 0.0)


def test_plate_support_reports_numerical_brackets_and_coverage(plate_case) -> None:
    coverage = plate_case.metadata["coverage"]
    assert coverage["status"] == "resolution_limited"
    assert "absence of support intervals narrower than one grid step" in coverage["not_claimed"]
    for class_record in plate_case.metadata["classes"].values():
        lit = [member for member in class_record["members"] if member["energy"]["red"]["fine"] > 0.0]
        assert lit
        for member in lit:
            intervals = member["support_intervals"]["red"]
            assert intervals
            for interval in intervals:
                for transition in interval["transition_brackets"]:
                    assert transition["bracket_width_rad"] <= reference.SUPPORT_ENDPOINT_TOLERANCE_RAD
                    assert transition["left_active"] != transition["right_active"]


def test_plate_pose_constraint_mutation_is_detected(plate_case) -> None:
    for class_record in plate_case.metadata["classes"].values():
        negative = class_record["snapshots"]["negative_pose"]
        assert negative["body_sun_projection_residual"] < 5e-16
        assert min(negative["target_residual"].values()) > reference.TARGET_RESIDUAL_TOLERANCE


def test_exporter_writes_json_npz_and_provenance(tmp_path, random_case, plate_case) -> None:
    script_path = Path(__file__).parents[1] / "scripts" / "export_raypath_diagnostic_reference.py"
    spec = importlib.util.spec_from_file_location("export_raypath_diagnostic_reference", script_path)
    assert spec is not None and spec.loader is not None
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)
    assembled = reference.DiagnosticReference(
        {
            "schema_version": reference.SCHEMA_VERSION,
            "cases": {"random_regular": random_case.metadata, "plate_rhombic_9": plate_case.metadata},
        },
        {**random_case.arrays, **plate_case.arrays},
    )
    files = exporter.write_reference(assembled, tmp_path, {"test": True})
    payload = json.loads(files["reference"].read_text())
    provenance = json.loads(files["provenance"].read_text())
    with np.load(files["arrays"], allow_pickle=False) as arrays:
        assert sorted(arrays.files) == sorted(assembled.arrays)
        assert arrays["plate_blue_snapshots_rotations"].shape == (5, 3, 3)
    assert payload["schema_version"] == reference.SCHEMA_VERSION
    assert payload["cases"]["plate_rhombic_9"]["classes"]["blue"]["target_energy"] == plate_case.metadata["classes"]["blue"]["target_energy"]
    assert provenance["test"] and len(provenance["files"]["arrays.npz"]["sha256"]) == 64


def test_recorded_fixture_pins_semantic_scalars_and_arrays() -> None:
    fixture = Path(__file__).parent / "data" / "raypath-diagnostic-reference"
    payload = json.loads((fixture / "reference.json").read_text())
    provenance = json.loads((fixture / "provenance.json").read_text())
    assert payload["schema_version"] == reference.SCHEMA_VERSION
    assert not provenance["sources_modified_in_worktree"]
    for filename in ("reference.json", "arrays.npz"):
        assert provenance["files"][filename]["sha256"] == sha256_of(fixture / filename)

    features = {feature["id"]: feature for feature in payload["cases"]["random_regular"]["features"]}
    inner = features["random_regular.3-5.inner_edge"]
    for label, index in reference.INDEX_ENDPOINTS.items():
        assert inner["position_deg"][label] == pytest.approx(reference.minimum_deviation_deg(index), abs=2e-8)
    assert features["random_regular.3-1-5.solar_caustic_candidate"]["evidence_status"] == "candidate"
    counterfactual = payload["cases"]["random_regular"]["internal_reflection_counterfactual"]["blue_red_ratio"]
    assert counterfactual["production"] == pytest.approx(2.072278481972593, abs=1e-12)
    assert counterfactual["without_internal_R"] == pytest.approx(0.753492919873792, abs=1e-12)

    classes = payload["cases"]["plate_rhombic_9"]["classes"]
    expected_ratios = {"white": 1.0098063886104218, "blue": 1.525349437614488}
    for class_label, expected in expected_ratios.items():
        for target in ("plus", "minus"):
            energy = classes[class_label]["target_energy"]
            coarse_ratio = energy[target]["blue"]["coarse"] / energy[target]["red"]["coarse"]
            fine_ratio = energy["target_ratios"][target]["blue_red"]
            assert fine_ratio == pytest.approx(expected, abs=max(2.0 * abs(fine_ratio - coarse_ratio), 1e-12))
        assert len(classes[class_label]["lit_members"]["red"]) == 12
        assert len(classes[class_label]["lit_members"]["blue"]) == 12

    with np.load(fixture / "arrays.npz", allow_pickle=False) as arrays:
        assert sorted(arrays.files) == sorted(payload["array_store"]["arrays"])
        for label in ("white", "blue"):
            rotations = arrays[f"plate_{label}_snapshots_rotations"]
            assert rotations.shape == (5, 3, 3)
            assert np.max(np.abs(rotations @ np.swapaxes(rotations, 1, 2) - np.eye(3))) < 5e-16
