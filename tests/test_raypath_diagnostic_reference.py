"""Mechanism-level tests for the fixed ray-path diagnostic reference."""

from __future__ import annotations

import numpy as np
import pytest

from lumice_integral import raypath_diagnostic_reference as reference


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
