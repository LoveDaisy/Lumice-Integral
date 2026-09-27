"""Refractive index of ice ``n(lambda)``: Lumice's Sellmeier fit, reproduced without Lumice.

Authority: Lumice ``src/core/optics.cpp::IceRefractiveIndex::Get`` with the
coefficients ``kCoefAvr`` of ``src/core/optics.hpp`` (B1, B2, C1, C2; the
average of the ordinary and extraordinary fits)::

    n^2 = 1 + B1 / (1 - C1 * 1e-2 / lambda^2) + B2 / (1 - C2 * 1e2 / lambda^2),  lambda in micrometres

Evaluated in float64 on the decimal coefficients, which gives
``n(550) = 1.3110129170742788`` (the anchor recorded by
explore-spectral-conventions).  Lumice holds the coefficients and the
``C * 1e-2`` / ``C * 1e2`` products as C++ ``float``, which moves its own
value to ``1.3110129100622272`` at 550 nm (below ``1e-8`` over the visible
band, ``tests/test_spectrum_dispersion.py``), and its wavelength pool then
rounds ``n`` to ``float`` (half an ulp, up to ``6e-8``).
"""

from __future__ import annotations

import math

SELLMEIER_COEFFICIENTS = (0.701777, 1.091144, 0.884400, 0.796950)
"""``kCoefAvr``: ``(B1, B2, C1, C2)``."""

MIN_WAVELENGTH_NM = 350.0
MAX_WAVELENGTH_NM = 900.0


def refractive_index(wavelength_nm: float) -> float:
    """``n(lambda)`` of ice; ``1.0`` outside ``[350, 900]`` nm.

    The ``1.0`` fallback is Lumice's own boundary handling (``kMinWaveLength`` /
    ``kMaxWaveLength``), kept so pools built on either side agree; it is not
    a physical statement about ice outside the fitted range.
    """
    if wavelength_nm < MIN_WAVELENGTH_NM or wavelength_nm > MAX_WAVELENGTH_NM:
        return 1.0
    b1, b2, c1, c2 = SELLMEIER_COEFFICIENTS
    wl_um = wavelength_nm / 1e3
    n2 = 1.0
    n2 += b1 / (1 - c1 * 1e-2 / wl_um / wl_um)
    n2 += b2 / (1 - c2 * 1e2 / wl_um / wl_um)
    return math.sqrt(n2)
