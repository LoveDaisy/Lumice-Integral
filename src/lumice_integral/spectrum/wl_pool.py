"""The M-slot wavelength pool: Lumice's one reduction of every spectrum config.

Authority: Lumice ``src/core/backend/wl_pool.hpp::ComputeWlPool``.  Both kinds
of ``light_source.spectrum`` reduce to ``M`` entries ``(n, spd_weight, xbar,
ybar, zbar)``:

- illuminant: the midpoints ``lambda_i = 380 + (i + 1/2) 400 / M`` of ``M``
  equal slices of ``[380, 780]`` nm, each with :func:`.dispersion.refractive_index`,
  :func:`.illuminant.spd` and :func:`.cmf.lookup` at ``lambda_i``;
- discrete: one wavelength with a given weight, its entry replicated ``M``
  times (Lumice draws a slot uniformly, so the copies keep that draw
  well-defined).

Lumice computes the pool in C++ ``float`` and stores ``float`` entries; this
module stays in float64 (``docs/conventions.md``, spectral row).

:func:`emitted_weight` is the other half of the normalisation: every ray
carries its slot's ``spd_weight``, while Lumice's ``emitted_energy`` charges
each emitted ray with one fixed weight, the illuminant's band mean
(:func:`.illuminant.band_mean_spd`) or the discrete ``weight``.  An image in
units of ``raw / emitted_energy`` is therefore ``(1 / M) sum_i spd_i CMF_i V_i
/ emitted_weight`` (the slot is drawn uniformly), in which a discrete pool's
``M`` and ``weight`` cancel.
"""

from __future__ import annotations

import dataclasses

from .cmf import lookup
from .dispersion import refractive_index
from .illuminant import IlluminantType, band_mean_spd, spd

POOL_BAND_MIN_NM = 380.0
POOL_BAND_WIDTH_NM = 400.0

WAVELENGTH_COUNT_PRESETS: dict[str, int] = {
    "panel_wide_fov": 5,
    "panel_narrow_fov": 9,
    "writing_canonical_strip": 33,
}
"""Lower bounds on ``M`` for the three rendering contexts measured by explore-spectral-conventions.

``panel_wide_fov``: the panel at fov 32-100 deg; ``panel_narrow_fov``: the
panel at fov 16 deg; ``writing_canonical_strip``: the writing series'
canonical strip (fov 6 deg).  They are anchor values, not a general
fov -> ``M`` rule: the dispersion step between adjacent slices has to stay
below a pixel, so ``M`` depends on fov and resolution together, and only
these three points were checked.
"""


@dataclasses.dataclass(frozen=True)
class WlPoolEntry:
    """One slot of the pool: Lumice's ``WlEntry`` plus the wavelength it was computed at."""

    wavelength_nm: float
    refractive_index: float
    spd_weight: float
    cmf_x: float
    cmf_y: float
    cmf_z: float


def _entry(wavelength_nm: float, spd_weight: float) -> WlPoolEntry:
    x, y, z = lookup(wavelength_nm)
    return WlPoolEntry(wavelength_nm, refractive_index(wavelength_nm), spd_weight, x, y, z)


def pool_wavelengths(m: int) -> tuple[float, ...]:
    """The ``M`` slice midpoints ``380 + (i + 1/2) 400 / M`` nm."""
    if m < 1:
        raise ValueError(f"the pool needs M >= 1 slots, got {m}")
    return tuple(POOL_BAND_MIN_NM + (i + 0.5) * POOL_BAND_WIDTH_NM / m for i in range(m))


def wavelength_pool(
    m: int,
    *,
    illuminant: IlluminantType | None = None,
    discrete_wavelength_nm: float | None = None,
    discrete_weight: float = 1.0,
) -> tuple[WlPoolEntry, ...]:
    """The ``M``-slot pool of an illuminant, or of one discrete wavelength; exactly one of the two must be given."""
    if (illuminant is None) == (discrete_wavelength_nm is None):
        raise ValueError("give exactly one of illuminant= and discrete_wavelength_nm=")
    if illuminant is not None:
        return tuple(_entry(wl, spd(illuminant, wl)) for wl in pool_wavelengths(m))
    if m < 1:
        raise ValueError(f"the pool needs M >= 1 slots, got {m}")
    return (_entry(float(discrete_wavelength_nm), float(discrete_weight)),) * m


def emitted_weight(*, illuminant: IlluminantType | None = None, discrete_weight: float | None = None) -> float:
    """The weight Lumice charges ``emitted_energy`` per emitted ray; exactly one of the two must be given.

    ``illuminant``: :func:`.illuminant.band_mean_spd` (``MeanIlluminantWeight``,
    ``src/core/simulator.cpp``); discrete: the wavelength's own ``weight``
    (``wl_param.weight_``).  The arguments mirror :func:`wavelength_pool`'s.
    """
    if (illuminant is None) == (discrete_weight is None):
        raise ValueError("give exactly one of illuminant= and discrete_weight=")
    return band_mean_spd(illuminant) if illuminant is not None else float(discrete_weight)
