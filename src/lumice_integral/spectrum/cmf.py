"""CIE 1931 2 degree colour-matching functions ``(xbar, ybar, zbar)`` at a wavelength.

Authority: Lumice ``src/core/backend/wl_pool.hpp::ComputeCmf`` on the table of
``src/util/color_data.hpp`` (transcribed in :mod:`.data.cie_1931_cmf`): the
wavelength is rounded to the nearest integer nm (``int(wl + 0.5)``) and looked
up in ``[360, 830]``; outside that range all three are zero.  No
interpolation between the 1 nm rows.
"""

from __future__ import annotations

import math

from .data.cie_1931_cmf import CMF_MAX_WAVELENGTH_NM, CMF_MIN_WAVELENGTH_NM, CMF_X, CMF_Y, CMF_Z


def lookup(wavelength_nm: float) -> tuple[float, float, float]:
    """``(xbar, ybar, zbar)`` of the 1 nm row nearest ``wavelength_nm``; ``(0, 0, 0)`` outside ``[360, 830]`` nm."""
    key = math.floor(wavelength_nm + 0.5)
    if key < CMF_MIN_WAVELENGTH_NM or key > CMF_MAX_WAVELENGTH_NM:
        return (0.0, 0.0, 0.0)
    index = key - CMF_MIN_WAVELENGTH_NM
    return (CMF_X[index], CMF_Y[index], CMF_Z[index])
