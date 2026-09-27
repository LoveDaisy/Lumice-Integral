"""Spectral conventions of Lumice, reproduced independently: dispersion, colour matching, illuminants, the wavelength pool.

Authority: the Lumice source (``docs/conventions.md``, spectral row); the
values are transcribed or recomputed here, never read from Lumice at run
time.  Everything except ``store`` is pure Python in float64 and never
imports JAX, numpy or another :mod:`lumice_integral` module (checked by
``tests/test_spectrum_dependency_direction.py``):

- ``dispersion``: ``refractive_index(lambda)``, Lumice's Sellmeier fit of ice.
- ``cmf``: the CIE 1931 2 degree colour-matching functions, 360-830 nm at 1 nm.
- ``illuminant``: ``IlluminantType`` (D50 / D55 / D65 / D75 / A / E) and their SPDs.
- ``wl_pool``: the ``M``-slot wavelength pool every spectrum config reduces to,
  and ``WAVELENGTH_COUNT_PRESETS`` (lower bounds on ``M`` per rendering context).
- ``store``: one S^2 event store per pool wavelength through
  :func:`lumice_integral.s2_store.build_or_load` (the only edge to ``s2_store``);
  not imported here, so ``import lumice_integral.spectrum`` stays light.
- ``xyz_band_sum``: colour band sums, one monochrome
  :func:`lumice_integral.band_sum.render_band_sum_window` per distinct
  ``n(lambda)`` summed to CIE XYZ, and their output directory; the edge to
  ``band_sum``, likewise not imported here.
"""

from . import cmf, dispersion, illuminant, wl_pool
from .illuminant import IlluminantType
from .wl_pool import WAVELENGTH_COUNT_PRESETS, WlPoolEntry, emitted_weight, wavelength_pool

__all__ = [
    "WAVELENGTH_COUNT_PRESETS",
    "IlluminantType",
    "WlPoolEntry",
    "cmf",
    "dispersion",
    "emitted_weight",
    "illuminant",
    "wavelength_pool",
    "wl_pool",
]
