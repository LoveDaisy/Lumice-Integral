"""``spectrum.store``: one S^2 store per distinct ``n(lambda)`` through the existing ``build_or_load`` cache."""

from __future__ import annotations

import dataclasses

import pytest

from lumice_integral import s2_store
from lumice_integral.canonical_scene import canonical_crystal
from lumice_integral.spectrum import IlluminantType, wavelength_pool
from lumice_integral.spectrum.store import build_wavelength_pool_stores

SMALL_N = 2_000


def _stores(tmp_path, pool):
    return build_wavelength_pool_stores(canonical_crystal(), [(3, 5)], SMALL_N, pool, base_dir=tmp_path, run_checks=False)


def test_each_wavelength_gets_the_store_of_its_own_refractive_index(tmp_path) -> None:
    pool = wavelength_pool(3, illuminant=IlluminantType.E)
    stores = _stores(tmp_path, pool)
    assert list(stores) == [entry.wavelength_nm for entry in pool]
    for entry in pool:
        assert stores[entry.wavelength_nm].spec.refractive_index == entry.refractive_index
    keys = {store.spec.cache_key() for store in stores.values()}
    assert len(keys) == 3
    assert {p.name for p in tmp_path.iterdir() if not p.name.startswith(".")} == keys


def test_the_stores_are_the_ones_build_or_load_returns(tmp_path) -> None:
    """No second cache: a direct ``build_or_load`` with the same ``n`` loads the directory the pool built."""
    entry = wavelength_pool(3, illuminant=IlluminantType.E)[0]
    stores = _stores(tmp_path, [entry])
    direct = s2_store.build_or_load(canonical_crystal(), entry.refractive_index, [(3, 5)], SMALL_N, base_dir=tmp_path, run_checks=False)
    assert direct.spec == stores[entry.wavelength_nm].spec
    assert len(list(tmp_path.iterdir())) == 1


def test_equal_refractive_indices_share_one_store(tmp_path) -> None:
    first, second = wavelength_pool(2, illuminant=IlluminantType.E)
    twin = dataclasses.replace(second, refractive_index=first.refractive_index)
    stores = _stores(tmp_path, [first, twin])
    assert stores[first.wavelength_nm] is stores[twin.wavelength_nm]
    assert len(list(tmp_path.iterdir())) == 1


def test_a_discrete_pool_builds_one_store(tmp_path) -> None:
    stores = _stores(tmp_path, wavelength_pool(4, discrete_wavelength_nm=550.0))
    assert len(stores) == 1 and stores[550.0].spec.refractive_index == pytest.approx(1.3110129, abs=2e-8)
