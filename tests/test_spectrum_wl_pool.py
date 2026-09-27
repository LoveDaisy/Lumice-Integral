"""``spectrum.wl_pool``: the M-slot pool of ``ComputeWlPool`` in its illuminant and discrete forms."""

from __future__ import annotations

import pytest

from lumice_integral.spectrum import WAVELENGTH_COUNT_PRESETS, IlluminantType, WlPoolEntry, wavelength_pool
from lumice_integral.spectrum.cmf import lookup
from lumice_integral.spectrum.dispersion import refractive_index
from lumice_integral.spectrum.illuminant import spd
from lumice_integral.spectrum.wl_pool import pool_wavelengths


@pytest.mark.parametrize("m", [1, 5, 9, 33, 64])
def test_illuminant_mode_samples_slice_midpoints(m: int) -> None:
    pool = wavelength_pool(m, illuminant=IlluminantType.D65)
    assert len(pool) == m
    for i, entry in enumerate(pool):
        wl = 380.0 + (i + 0.5) * 400.0 / m
        assert entry == WlPoolEntry(wl, refractive_index(wl), spd(IlluminantType.D65, wl), *lookup(wl))


def test_midpoints_tile_380_780() -> None:
    wavelengths = pool_wavelengths(5)
    assert wavelengths == (420.0, 500.0, 580.0, 660.0, 740.0)
    assert pool_wavelengths(1) == (580.0,)


def test_discrete_mode_replicates_one_entry() -> None:
    pool = wavelength_pool(4, discrete_wavelength_nm=550.0, discrete_weight=0.25)
    assert len(pool) == 4 and len(set(pool)) == 1
    assert pool[0] == WlPoolEntry(550.0, refractive_index(550.0), 0.25, *lookup(550.0))
    assert pool[0].refractive_index == 1.3110129170742788


def test_rejects_ambiguous_or_missing_mode() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        wavelength_pool(5)
    with pytest.raises(ValueError, match="exactly one"):
        wavelength_pool(5, illuminant=IlluminantType.E, discrete_wavelength_nm=550.0)


@pytest.mark.parametrize("kwargs", [{"illuminant": IlluminantType.E}, {"discrete_wavelength_nm": 550.0}])
def test_rejects_an_empty_pool(kwargs) -> None:
    with pytest.raises(ValueError, match="M >= 1"):
        wavelength_pool(0, **kwargs)


def test_wavelength_count_presets_are_the_explore_summary_values() -> None:
    assert WAVELENGTH_COUNT_PRESETS == {"panel_wide_fov": 5, "panel_narrow_fov": 9, "writing_canonical_strip": 33}


def test_the_pool_band_lies_inside_every_table() -> None:
    """380-780 nm is inside the CMF (360-830), SPD (300-830) and dispersion (350-900) ranges: no zeroed slot."""
    for entry in wavelength_pool(255, illuminant=IlluminantType.A):
        assert entry.refractive_index > 1.3 and entry.spd_weight > 0.0 and entry.cmf_y > 0.0
