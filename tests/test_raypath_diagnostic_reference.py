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
