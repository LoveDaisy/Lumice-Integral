"""Primitives of the absolute radiometric scale probe (``scripts/probe_absolute_scale.py``).

The probe's conversion ``raw / E = K_p V``, ``K_p = N_sym * ybar * Omega_p / (S / 2)``,
rests on three closed-form ingredients checked here: the total surface area
``S`` of the canonical prism, the on-axis linear-lens pixel solid angle (which
must equal the ``axis_solid_angle`` of the Lumice sidecar of task
``lumice-raw-profile-oracle``, ``1.7438296140426246e-07``), and the
``ybar(550)`` constant duplicated in ``scripts/compare_strip_v2.py``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from lumice_integral.canonical_scene import CANONICAL_RENDER, canonical_crystal

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
LUMICE_SIDECAR_AXIS_SOLID_ANGLE = 1.7438296140426246e-07


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def probe():
    return _load("probe_absolute_scale")


def test_total_surface_area_matches_the_prism_closed_form(probe):
    crystal = canonical_crystal()  # a = 1, h = 2
    a, h = crystal.a, crystal.h
    assert probe.total_surface_area(crystal) == pytest.approx(6.0 * a * h + 3.0 * np.sqrt(3.0) * a * a, rel=1e-12)
    assert probe.total_surface_area(crystal) == pytest.approx(17.196152422706632, rel=1e-12)


def test_on_axis_pixel_solid_angle_equals_the_lumice_sidecar(probe):
    render = dict(CANONICAL_RENDER)
    centre = probe.pixel_solid_angle(render["height"] // 2, render["width"] // 2, render)
    assert centre == pytest.approx(LUMICE_SIDECAR_AXIS_SOLID_ANGLE, rel=1e-6)
    corner = probe.pixel_solid_angle(0, 0, render)
    assert corner < centre


def test_ybar_constant_is_shared_with_compare_strip_v2(probe):
    compare = _load("compare_strip_v2")
    assert compare.YBAR_550 == probe.YBAR_550
    assert compare.LUMICE_SYMMETRY_FOLD == probe.LUMICE_SYMMETRY_FOLD


def test_pixel_constant_depends_on_the_pixel_through_its_solid_angle_only(probe):
    render = dict(CANONICAL_RENDER)
    s = probe.total_surface_area(canonical_crystal())
    centre = probe.pixel_constant(render["height"] // 2, render["width"] // 2, render, n_sym=12, surface_area=s)
    assert centre == pytest.approx(12 * probe.YBAR_550 * LUMICE_SIDECAR_AXIS_SOLID_ANGLE / (0.5 * s), rel=1e-6)
    for row, column in ((150, 106), (300, 126), (450, 146)):
        k = probe.pixel_constant(row, column, render, n_sym=12, surface_area=s)
        assert k / centre == pytest.approx(probe.pixel_solid_angle(row, column, render) / LUMICE_SIDECAR_AXIS_SOLID_ANGLE, rel=1e-6)


def test_merged_relative_noise_matches_its_own_formula(probe):
    rng = np.random.default_rng(0)
    a = 100.0 + rng.normal(0, 5.0, 5000)
    b = 100.0 + rng.normal(0, 5.0, 5000)
    expected = float(np.std((a - b) / (0.5 * (a + b))) / 2.0)
    assert probe.merged_relative_noise(a, b) == pytest.approx(expected, rel=1e-12)
    assert probe.merged_relative_noise(a, b) == pytest.approx(probe.merged_relative_noise(b, a), rel=1e-12)
    assert probe.merged_relative_noise(a, a) == pytest.approx(0.0, abs=1e-12)
