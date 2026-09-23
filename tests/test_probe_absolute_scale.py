"""Primitives of the absolute radiometric scale probe (``scripts/probe_absolute_scale.py``).

The probe's conversion ``raw / E = N_sym * ybar * Omega_p * V~`` rests on three
closed-form ingredients checked here: the projected silhouette ``A_tot`` of the
canonical prism, the on-axis linear-lens pixel solid angle (which must equal
the ``axis_solid_angle`` of the Lumice sidecar of task
``lumice-raw-profile-oracle``, ``1.7438296140426246e-07``), and the
``ybar(550)`` constant duplicated in ``scripts/compare_strip_v2.py``.  One
pixel pins ``A_eff = V / V~`` to the value recorded in
``docs/ch06-reference-fixture.md`` section 7, stage 4.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from lumice_integral.canonical_scene import CANONICAL_RENDER, canonical_crystal, canonical_incident_direction
from lumice_integral.strip_pixel import PixelOptions, canonical_strip_scene, pixel_target

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
LUMICE_SIDECAR_AXIS_SOLID_ANGLE = 1.7438296140426246e-07
TEST_PRESCAN_SAMPLES = 400_000


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def probe():
    return _load("probe_absolute_scale")


def test_silhouette_area_matches_the_prism_closed_forms(probe):
    crystal = canonical_crystal()  # a = 1, h = 2
    a, h = crystal.a, crystal.h
    along_c = probe.silhouette_area(crystal, np.array([[0.0, 0.0, -1.0]]))[0]
    # HexPrism puts a side-face normal on x and a vertex on y
    along_face_normal = probe.silhouette_area(crystal, np.array([[-1.0, 0.0, 0.0]]))[0]
    along_vertex = probe.silhouette_area(crystal, np.array([[0.0, -1.0, 0.0]]))[0]
    assert along_c == pytest.approx(1.5 * np.sqrt(3.0) * a * a, rel=1e-12)
    assert along_vertex == pytest.approx(np.sqrt(3.0) * a * h, rel=1e-12)
    assert along_face_normal == pytest.approx(2.0 * a * h, rel=1e-12)
    # convex body: the front-facing sum is half the absolute sum over every face
    s = np.array([0.3, -0.5, 0.8])
    s /= np.linalg.norm(s)
    front = probe.silhouette_area(crystal, s[None])[0]
    back = probe.silhouette_area(crystal, -s[None])[0]
    assert front == pytest.approx(back, rel=1e-12)


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


def test_effective_silhouette_of_a_bright_band_pixel(probe):
    crystal = canonical_crystal()
    scene = canonical_strip_scene(prescan_sample_count=TEST_PRESCAN_SAMPLES)
    scene_lumice = canonical_strip_scene(
        prescan_table=scene.prescan_table,
        pose_density=probe.PerSilhouetteDensity(scene.pose_density, crystal, canonical_incident_direction()),
    )
    value, value_lumice, found = probe.pixel_values(scene, scene_lumice, pixel_target(scene.render, 300, 126), PixelOptions(), ())
    assert found.incomplete_count == 0 and len(found.components) == 1
    # recorded by the probe run of 2026-09-23 (absolute_scale_pixels.csv, row 300 of column 126: 4.3507)
    assert value / value_lumice == pytest.approx(4.3507, rel=1e-3)
