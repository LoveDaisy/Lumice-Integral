"""Relative spectral power distributions of the CIE standard illuminants Lumice offers.

Authority: Lumice ``src/util/illuminant.cpp::GetIlluminantSpd`` with the data
of ``src/util/illuminant_data.hpp`` (basis functions transcribed in
:mod:`.data.cie_daylight_basis`).  Lumice evaluates in C++ ``float``; this
module evaluates the same formulas in float64.

- ``D50`` / ``D55`` / ``D65`` / ``D75``: CIE 015:2018 daylight, CCT -> ``x_D``
  (two branches split at 7000 K) -> ``y_D`` -> mixing coefficients ``M1, M2``
  -> ``S0 + M1 S1 + M2 S2``, linearly interpolated on the 5 nm grid of
  ``[300, 830]`` nm (normalised to about 100 at 560 nm); zero outside.
- ``A``: Planck at 2856 K, ``100 (560/lambda)^5 (e^{c2/(T 560)} - 1) / (e^{c2/(T lambda)} - 1)``
  with ``c2 = 1.4388e7`` nm K, restricted to ``[300, 830]`` nm.
- ``E``: ``1`` on ``[300, 830]`` nm, zero outside.

:func:`band_mean_spd` is Lumice's ``MeanIlluminantWeight``: the SPD averaged
over the band ``[380, 780]`` nm the pool slices (and the per-batch sampler)
cover.  Lumice charges it, not the weight of any drawn wavelength, to the
``emitted_energy`` normalisation denominator (``src/core/simulator.cpp``).
"""

from __future__ import annotations

import enum
import math

from .data.cie_daylight_basis import (
    DAYLIGHT_MAX_WAVELENGTH_NM,
    DAYLIGHT_MIN_WAVELENGTH_NM,
    DAYLIGHT_S0,
    DAYLIGHT_S1,
    DAYLIGHT_S2,
    DAYLIGHT_STEP_NM,
)


class IlluminantType(enum.Enum):
    """Lumice's ``IlluminantType``; the values are its JSON names."""

    D50 = "D50"
    D55 = "D55"
    D65 = "D65"
    D75 = "D75"
    A = "A"
    E = "E"


DAYLIGHT_CCT_K = {
    IlluminantType.D50: 5003.0,
    IlluminantType.D55: 5503.0,
    IlluminantType.D65: 6504.0,
    IlluminantType.D75: 7504.0,
}
"""``kCctD50`` ... ``kCctD75`` (CIE 015:2018, revised Planck constants)."""

ILLUMINANT_A_TEMPERATURE_K = 2856.0
ILLUMINANT_A_C2_NM_K = 1.4388e7
ILLUMINANT_A_REFERENCE_NM = 560.0


def daylight_chromaticity(cct_k: float) -> tuple[float, float]:
    """``(x_D, y_D)`` of CIE daylight at correlated colour temperature ``cct_k``."""
    t = 1.0 / cct_k
    if cct_k <= 7000.0:
        x = 0.244063 + 0.09911e3 * t + 2.9678e6 * t**2 - 4.6070e9 * t**3
    else:
        x = 0.237040 + 0.24748e3 * t + 1.9018e6 * t**2 - 2.0064e9 * t**3
    return x, -3.000 * x * x + 2.870 * x - 0.275


def daylight_mixing_coefficients(cct_k: float) -> tuple[float, float]:
    """``(M1, M2)`` weighting the basis functions ``S1``, ``S2``."""
    x, y = daylight_chromaticity(cct_k)
    denominator = 0.0241 + 0.2562 * x - 0.7341 * y
    return (-1.3515 - 1.7703 * x + 5.9114 * y) / denominator, (0.0300 - 31.4424 * x + 30.0717 * y) / denominator


def _daylight_spd(cct_k: float, wavelength_nm: float) -> float:
    m1, m2 = daylight_mixing_coefficients(cct_k)
    fi = (wavelength_nm - DAYLIGHT_MIN_WAVELENGTH_NM) / DAYLIGHT_STEP_NM
    i0 = int(fi)
    frac = fi - i0
    last = len(DAYLIGHT_S0) - 1
    if i0 >= last:
        i0, frac = last, 0.0
    i1 = min(i0 + 1, last)
    s0, s1, s2 = (table[i0] + frac * (table[i1] - table[i0]) for table in (DAYLIGHT_S0, DAYLIGHT_S1, DAYLIGHT_S2))
    return s0 + m1 * s1 + m2 * s2


def _illuminant_a_spd(wavelength_nm: float) -> float:
    c2, t, ref = ILLUMINANT_A_C2_NM_K, ILLUMINANT_A_TEMPERATURE_K, ILLUMINANT_A_REFERENCE_NM
    return 100.0 * (ref / wavelength_nm) ** 5 * math.expm1(c2 / (t * ref)) / math.expm1(c2 / (t * wavelength_nm))


def spd(illuminant: IlluminantType, wavelength_nm: float) -> float:
    """Relative SPD of ``illuminant`` at ``wavelength_nm``; zero outside ``[300, 830]`` nm."""
    if wavelength_nm < DAYLIGHT_MIN_WAVELENGTH_NM or wavelength_nm > DAYLIGHT_MAX_WAVELENGTH_NM:
        return 0.0
    if illuminant in DAYLIGHT_CCT_K:
        return _daylight_spd(DAYLIGHT_CCT_K[illuminant], wavelength_nm)
    if illuminant is IlluminantType.A:
        return _illuminant_a_spd(wavelength_nm)
    if illuminant is IlluminantType.E:
        return 1.0
    raise ValueError(f"unknown illuminant {illuminant!r}")


BAND_MEAN_MIN_NM = 380.0
BAND_MEAN_WIDTH_NM = 400.0
BAND_MEAN_SIMPSON_INTERVALS = 8000
"""0.05 nm steps: the 5 nm daylight grid points are Simpson panel edges, so the piecewise-linear SPDs integrate exactly."""


def band_mean_spd(illuminant: IlluminantType) -> float:
    """``(1 / 400) int_380^780 spd(lambda) d lambda``: Lumice's ``MeanIlluminantWeight`` (``illuminant.cpp::ComputeMeanSpd``).

    Lumice averages ``GetIlluminantSpd`` in C++ ``float`` on 4000001 equally
    spaced points of the band, endpoints included; this is the integral it
    approximates, by composite Simpson (exact for the daylight SPDs and ``E``,
    ``~1e-15`` relative for ``A``).  The two differ by the endpoint weight of
    the discrete average and Lumice's ``float``, ``O(1e-7)`` relative.
    """
    count = BAND_MEAN_SIMPSON_INTERVALS
    step = BAND_MEAN_WIDTH_NM / count
    total = 0.0
    for i in range(count + 1):
        coefficient = 1.0 if i in (0, count) else (4.0 if i % 2 else 2.0)
        total += coefficient * spd(illuminant, BAND_MEAN_MIN_NM + i * step)
    return total * step / 3.0 / BAND_MEAN_WIDTH_NM
