"""``spectrum.cmf``: the CIE 1931 2 degree table and Lumice's nearest-row lookup."""

from __future__ import annotations

import pytest

from lumice_integral.spectrum.cmf import lookup
from lumice_integral.spectrum.data.cie_1931_cmf import CMF_X, CMF_Y, CMF_Z


def test_table_shape() -> None:
    assert len(CMF_X) == len(CMF_Y) == len(CMF_Z) == 471


def test_ybar_550_matches_the_absolute_scale_probe() -> None:
    """``scripts/probe_absolute_scale.py::YBAR_550``, an anchor typed in before this table existed."""
    assert lookup(550.0)[1] == 0.9949501


def test_standard_cie_1931_landmarks() -> None:
    """``ybar(555) = 1`` (its maximum) and two 5 nm rows of the CIE 1931 table near the ``xbar`` / ``zbar`` peaks."""
    assert lookup(555.0)[1] == 1.0
    assert lookup(600.0)[0] == 1.0622
    assert lookup(445.0)[2] == 1.7826
    assert max(CMF_Y) == 1.0


def test_lookup_rounds_to_the_nearest_nm_like_lumice() -> None:
    """``int(wl + 0.5)``: ties go up, no interpolation."""
    assert lookup(549.5) == lookup(550.0)
    assert lookup(549.49) == lookup(549.0)
    assert lookup(550.49) == lookup(550.0)


@pytest.mark.parametrize("wavelength_nm", [359.49, 830.5, 300.0, 900.0])
def test_lookup_out_of_range_returns_zero(wavelength_nm: float) -> None:
    assert lookup(wavelength_nm) == (0.0, 0.0, 0.0)


def test_range_ends_are_inside() -> None:
    assert lookup(359.5) == (CMF_X[0], CMF_Y[0], CMF_Z[0])
    assert lookup(830.0) == (CMF_X[-1], CMF_Y[-1], CMF_Z[-1])
