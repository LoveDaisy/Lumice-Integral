"""``spectrum.dispersion``: Lumice's Sellmeier ``n(lambda)``, anchored at ``n(550)`` and against Lumice's float evaluation."""

from __future__ import annotations

import numpy as np
import pytest

from lumice_integral.spectrum.dispersion import SELLMEIER_COEFFICIENTS, refractive_index

from _lumice_source import float_array, float_constant, lumice_source

LUMICE_N_550_FLOAT_COEFFICIENTS = 1.3110129100622272
"""``IceRefractiveIndex::Get(550.0)`` as Lumice computes it (float coefficients), measured by compiling the source."""


def _lumice_float_coefficient_evaluation(wavelength_nm: float) -> float:
    """``IceRefractiveIndex::Get`` with its C++ promotions: float coefficients, float ``C * 1e-2f`` products, double rest."""
    b1, b2, c1, c2 = (float(np.float32(c)) for c in SELLMEIER_COEFFICIENTS)
    c1 = float(np.float32(c1) * np.float32(1e-2))
    c2 = float(np.float32(c2) * np.float32(1e2))
    wl = wavelength_nm / 1e3
    return float(np.sqrt(1.0 + b1 / (1 - c1 / wl / wl) + b2 / (1 - c2 / wl / wl)))


def test_n_550_matches_the_explore_anchor() -> None:
    assert refractive_index(550.0) == 1.3110129170742788


def test_n_550_matches_the_value_the_scripts_pass_for_lumice() -> None:
    """``scripts/probe_absolute_scale.py::LUMICE_REFRACTIVE_INDEX_550`` and ``--refractive-index 1.3110129``."""
    assert abs(refractive_index(550.0) - 1.3110129) < 2e-8


def test_the_float_coefficient_shift_stays_below_1e8_over_the_pool_band() -> None:
    assert _lumice_float_coefficient_evaluation(550.0) == LUMICE_N_550_FLOAT_COEFFICIENTS
    wavelengths = np.linspace(380.0, 780.0, 401)
    shift = max(abs(refractive_index(w) - _lumice_float_coefficient_evaluation(w)) for w in wavelengths)
    assert 0.0 < shift < 1e-8


def test_normal_dispersion_over_the_visible_band() -> None:
    values = [refractive_index(w) for w in np.linspace(380.0, 780.0, 81)]
    assert all(a > b for a, b in zip(values, values[1:]))
    assert 1.30 < values[-1] < values[0] < 1.33


@pytest.mark.parametrize("wavelength_nm", [349.9, 900.1, 200.0, 1200.0])
def test_outside_lumice_range_returns_one(wavelength_nm: float) -> None:
    assert refractive_index(wavelength_nm) == 1.0


def test_range_ends_are_inside() -> None:
    assert refractive_index(350.0) > 1.3 and refractive_index(900.0) > 1.3


@pytest.mark.slow  # slow: reads the Lumice checkout outside the repository
def test_coefficients_and_range_match_lumice_source() -> None:
    text = lumice_source("src/core/optics.hpp")
    assert tuple(float(v) for v in float_array(text, "kCoefAvr")) == SELLMEIER_COEFFICIENTS
    assert float(float_constant(text, "kMinWaveLength")) == 350.0
    assert float(float_constant(text, "kMaxWaveLength")) == 900.0
    body = lumice_source("src/core/optics.cpp")
    assert "kCoefAvr[2] * 1e-2f / wave_length / wave_length" in body
    assert "kCoefAvr[3] * 1e2f / wave_length / wave_length" in body
