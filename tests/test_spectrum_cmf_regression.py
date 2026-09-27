"""``spectrum.data.cie_1931_cmf`` against Lumice ``src/util/color_data.hpp``: all 471 x 3 values, exactly."""

from __future__ import annotations

import pytest
from _lumice_source import float_array, float_constant, lumice_source

from lumice_integral.spectrum.data import cie_1931_cmf

pytestmark = pytest.mark.slow  # slow: reads the Lumice checkout outside the repository


@pytest.mark.parametrize("lumice_name, ours", [("kCmfX", "CMF_X"), ("kCmfY", "CMF_Y"), ("kCmfZ", "CMF_Z")])
def test_cmf_table_matches_lumice_source_exactly(lumice_name: str, ours: str) -> None:
    source = [float(v) for v in float_array(lumice_source("src/util/color_data.hpp"), lumice_name)]
    table = getattr(cie_1931_cmf, ours)
    assert len(source) == len(table) == 471
    differing = [(360 + i, a, b) for i, (a, b) in enumerate(zip(source, table)) if a != b]
    assert differing == []


def test_cmf_range_matches_lumice_source() -> None:
    text = lumice_source("src/core/color_util.hpp")
    assert int(float_constant(text, "kCmfMinWavelength")) == cie_1931_cmf.CMF_MIN_WAVELENGTH_NM
    assert int(float_constant(text, "kCmfMaxWavelength")) == cie_1931_cmf.CMF_MAX_WAVELENGTH_NM
