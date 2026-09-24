"""``scripts/compare_lumice_family.py``: its vectorised pixel solid angles are the probe's, on any linear camera."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from lumice_integral.canonical_scene import CANONICAL_RENDER

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "render",
    [
        dict(CANONICAL_RENDER),
        {"width": 321, "height": 161, "fov_deg": 32.0, "view": {"azimuth": 0.0, "elevation": 15.0}},
        {"width": 401, "height": 401, "fov_deg": 100.0, "view": {"azimuth": 0.0, "elevation": 15.0}},
    ],
)
def test_pixel_solid_angles_match_the_probe(render):
    probe = _load("probe_absolute_scale")
    family = _load("compare_lumice_family")
    grid = family.pixel_solid_angles(render)
    assert grid.shape == (render["height"], render["width"])
    for row, column in ((0, 0), (render["height"] // 2, render["width"] // 2), (render["height"] - 1, render["width"] - 3), (7, render["width"] // 3)):
        assert grid[row, column] == pytest.approx(probe.pixel_solid_angle(row, column, render), rel=1e-12)
