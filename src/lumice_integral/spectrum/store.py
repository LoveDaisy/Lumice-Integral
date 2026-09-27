"""One S^2 event store per pool wavelength, through the existing :func:`..s2_store.build_or_load`.

The refractive index is already a store parameter (it enters
:meth:`..s2_store.S2StoreSpec.cache_key`), so several wavelengths are several
calls with different ``n(lambda)``: no second cache, no schema change.  This
and :mod:`.xyz_band_sum` are the modules of :mod:`lumice_integral.spectrum`
that depend on the rest of the package (checked by
``tests/test_spectrum_dependency_direction.py``).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..geometry import HexPrism
from ..s2_store import DEFAULT_CACHE_DIR, S2EventStore, build_or_load
from .wl_pool import WlPoolEntry


def build_wavelength_pool_stores(
    crystal: HexPrism,
    members: Sequence[Sequence[int]],
    n: int,
    pool: Sequence[WlPoolEntry],
    *,
    base_dir: Path = DEFAULT_CACHE_DIR,
    **build_or_load_kwargs: Any,
) -> dict[float, S2EventStore]:
    """``{wavelength_nm: store}`` for every entry of ``pool``, one :func:`build_or_load` per distinct ``n``.

    Entries with the same refractive index (compared exactly: equal ``n``
    means the same build parameters, hence the same cache key) share one store
    object; a discrete pool, ``M`` copies of one entry, maps to a single store.
    """
    by_index: dict[float, S2EventStore] = {}
    stores: dict[float, S2EventStore] = {}
    for entry in pool:
        if entry.refractive_index not in by_index:
            by_index[entry.refractive_index] = build_or_load(
                crystal, entry.refractive_index, members, n, base_dir=base_dir, **build_or_load_kwargs
            )
        stores[entry.wavelength_nm] = by_index[entry.refractive_index]
    return stores
