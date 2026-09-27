"""``spectrum.illuminant``: Lumice's standard-illuminant SPDs."""

from __future__ import annotations

import numpy as np
import pytest

from lumice_integral.spectrum.data.cie_daylight_basis import DAYLIGHT_S0, DAYLIGHT_S1, DAYLIGHT_S2
from lumice_integral.spectrum.illuminant import IlluminantType, daylight_chromaticity, spd

DAYLIGHT = (IlluminantType.D50, IlluminantType.D55, IlluminantType.D65, IlluminantType.D75)


def test_basis_shape() -> None:
    assert len(DAYLIGHT_S0) == len(DAYLIGHT_S1) == len(DAYLIGHT_S2) == 107


def test_illuminant_e_is_one_in_range_and_zero_outside() -> None:
    assert all(spd(IlluminantType.E, w) == 1.0 for w in (300.0, 550.0, 830.0))
    assert spd(IlluminantType.E, 299.9) == 0.0 and spd(IlluminantType.E, 830.1) == 0.0


@pytest.mark.parametrize("illuminant", list(IlluminantType))
def test_every_illuminant_is_zero_outside_300_830(illuminant: IlluminantType) -> None:
    assert spd(illuminant, 299.0) == 0.0 and spd(illuminant, 831.0) == 0.0


@pytest.mark.parametrize("illuminant", [*DAYLIGHT, IlluminantType.A])
def test_normalised_to_100_at_560nm(illuminant: IlluminantType) -> None:
    """S1 and S2 vanish at 560 nm, so the daylight SPDs are ``S0(560) = 100`` there; A by construction."""
    assert spd(illuminant, 560.0) == pytest.approx(100.0, abs=1e-12)


def test_d65_chromaticity_is_the_cie_white_point() -> None:
    """CIE 015 daylight at 6504 K lands on the D65 white point ``(0.3127, 0.3290)`` (the formula gives ``y = 0.32908``)."""
    x, y = daylight_chromaticity(6504.0)
    assert abs(x - 0.3127) < 1e-4 and abs(y - 0.3290) < 2e-4


def test_d65_against_the_cie_table() -> None:
    """Tabulated D65 (CIE 015) at 5 nm grid points; the reconstruction differs by the CIE's own rounding (< 0.1 %)."""
    tabulated = {400.0: 82.7549, 450.0: 117.008, 500.0: 109.354, 600.0: 90.0062, 700.0: 71.6091}
    for wavelength, value in tabulated.items():
        assert spd(IlluminantType.D65, wavelength) == pytest.approx(value, rel=1e-3)


def test_daylight_interpolates_linearly_between_grid_points() -> None:
    left, right = spd(IlluminantType.D65, 500.0), spd(IlluminantType.D65, 505.0)
    assert spd(IlluminantType.D65, 502.5) == pytest.approx(0.5 * (left + right), rel=1e-14)


def test_lower_colour_temperatures_are_redder() -> None:
    ratios = [spd(t, 700.0) / spd(t, 450.0) for t in (*DAYLIGHT[::-1], IlluminantType.A)]
    assert all(a < b for a, b in zip(ratios, ratios[1:]))


def test_illuminant_a_rises_monotonically_across_the_visible() -> None:
    values = [spd(IlluminantType.A, w) for w in np.linspace(380.0, 780.0, 41)]
    assert all(a < b for a, b in zip(values, values[1:]))
