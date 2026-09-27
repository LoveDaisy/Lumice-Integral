"""``spectrum.illuminant`` data against Lumice ``src/util/illuminant_data.hpp`` / ``illuminant.cpp``, exactly."""

from __future__ import annotations

import pytest
from _lumice_source import float_array, float_constant, lumice_source

from lumice_integral.spectrum import illuminant
from lumice_integral.spectrum.data import cie_daylight_basis
from lumice_integral.spectrum.illuminant import IlluminantType

pytestmark = pytest.mark.slow  # slow: reads the Lumice checkout outside the repository


@pytest.mark.parametrize("lumice_name, ours", [("kDaylightS0", "DAYLIGHT_S0"), ("kDaylightS1", "DAYLIGHT_S1"), ("kDaylightS2", "DAYLIGHT_S2")])
def test_daylight_basis_matches_lumice_source_exactly(lumice_name: str, ours: str) -> None:
    source = [float(v) for v in float_array(lumice_source("src/util/illuminant_data.hpp"), lumice_name)]
    table = getattr(cie_daylight_basis, ours)
    assert len(source) == len(table) == 107
    differing = [(300 + 5 * i, a, b) for i, (a, b) in enumerate(zip(source, table)) if a != b]
    assert differing == []


def test_scalar_constants_match_lumice_source() -> None:
    text = lumice_source("src/util/illuminant_data.hpp")
    for name, value in [
        ("kDaylightLambdaMin", cie_daylight_basis.DAYLIGHT_MIN_WAVELENGTH_NM),
        ("kDaylightLambdaMax", cie_daylight_basis.DAYLIGHT_MAX_WAVELENGTH_NM),
        ("kDaylightLambdaStep", cie_daylight_basis.DAYLIGHT_STEP_NM),
        ("kCctD50", illuminant.DAYLIGHT_CCT_K[IlluminantType.D50]),
        ("kCctD55", illuminant.DAYLIGHT_CCT_K[IlluminantType.D55]),
        ("kCctD65", illuminant.DAYLIGHT_CCT_K[IlluminantType.D65]),
        ("kCctD75", illuminant.DAYLIGHT_CCT_K[IlluminantType.D75]),
        ("kIlluminantATemp", illuminant.ILLUMINANT_A_TEMPERATURE_K),
        ("kIlluminantAC2", illuminant.ILLUMINANT_A_C2_NM_K),
        ("kIlluminantARefWl", illuminant.ILLUMINANT_A_REFERENCE_NM),
    ]:
        assert float(float_constant(text, name)) == value, name


def test_formula_literals_match_lumice_source() -> None:
    """The CIE 015 polynomial and mixing literals, as they appear in ``illuminant.cpp``."""
    text = lumice_source("src/util/illuminant.cpp")
    for literal in [
        "0.244063f + 0.09911e3f * t_inv + 2.9678e6f * t_inv2 - 4.6070e9f * t_inv3",
        "0.237040f + 0.24748e3f * t_inv + 1.9018e6f * t_inv2 - 2.0064e9f * t_inv3",
        "cct <= 7000.0f",
        "-3.000f * x_d * x_d + 2.870f * x_d - 0.275f",
        "0.0241f + 0.2562f * x_d - 0.7341f * y_d",
        "(-1.3515f - 1.7703f * x_d + 5.9114f * y_d) / denom",
        "(0.0300f - 31.4424f * x_d + 30.0717f * y_d) / denom",
    ]:
        assert literal in text, literal
    assert [t.value for t in IlluminantType] == ["D50", "D55", "D65", "D75", "A", "E"]
    for name in ("D50", "D55", "D65", "D75", "A", "E"):
        assert f'"{name}"' in lumice_source("src/util/illuminant_data.hpp")
