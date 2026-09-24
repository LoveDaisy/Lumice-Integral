"""``scripts/compare_lumice_family.py``: its vectorised pixel solid angles are the probe's, on any linear camera."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from lumice_integral.canonical_scene import CANONICAL_RENDER

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def family():
    return _load("compare_lumice_family")


@pytest.mark.parametrize(
    "render",
    [
        dict(CANONICAL_RENDER),
        {"width": 321, "height": 161, "fov_deg": 32.0, "view": {"azimuth": 0.0, "elevation": 15.0}},
        {"width": 401, "height": 401, "fov_deg": 100.0, "view": {"azimuth": 0.0, "elevation": 15.0}},
    ],
)
def test_pixel_solid_angles_match_the_probe(render, family):
    probe = _load("probe_absolute_scale")
    grid = family.pixel_solid_angles(render)
    assert grid.shape == (render["height"], render["width"])
    for row, column in ((0, 0), (render["height"] // 2, render["width"] // 2), (render["height"] - 1, render["width"] - 3), (7, render["width"] // 3)):
        assert grid[row, column] == pytest.approx(probe.pixel_solid_angle(row, column, render), rel=1e-12)


def test_family_imports_pixel_solid_angles_and_merged_relative_noise_rather_than_redefining_them(family):
    """a56: these must be re-exported from probe_absolute_scale, not a second local implementation.

    ``__module__`` is set once, at function-definition time in ``probe_absolute_scale.py``, and is
    unaffected by which module *object* currently sits under that name in ``sys.modules`` (this test
    file's ``_load`` helper reloads modules by file path, which would make a plain identity check
    order-dependent); so it is the robust way to assert "``family`` imports this, it does not define
    a second copy of it".
    """
    assert family.pixel_solid_angles.__module__ == "probe_absolute_scale"
    assert family.merged_relative_noise.__module__ == "probe_absolute_scale"


def test_flux_ratio_sums_measured_over_predicted_on_the_mask(family):
    measured = np.array([1.0, 2.0, 3.0, 100.0])
    predicted = np.array([2.0, 2.0, 3.0, 100.0])
    mask = np.array([True, True, True, False])
    assert family.flux_ratio(measured, predicted, mask) == pytest.approx(6.0 / 7.0)


def test_flux_ratio_is_nan_when_predicted_is_zero_on_the_mask(family):
    measured = np.array([1.0, 2.0])
    predicted = np.array([0.0, 0.0])
    mask = np.array([True, True])
    assert np.isnan(family.flux_ratio(measured, predicted, mask))


def test_profile_rms_of_identical_profiles_is_zero(family):
    a = np.array([0.0, 1.0, 2.0, 10.0, 2.0, 1.0, 0.0])
    rms, lit = family.profile_rms(a, a.copy(), floor=0.05)
    assert rms == pytest.approx(0.0, abs=1e-12)
    assert lit == int((a / a.max() > 0.05).sum())


def test_profile_rms_matches_a_direct_computation_on_shifted_profiles(family):
    a = np.array([0.0, 5.0, 10.0, 5.0, 0.0])
    b = np.array([0.0, 4.0, 8.0, 6.0, 0.0])
    floor = 0.1
    na, nb = a / a.max(), b / b.max()
    lit = (na > floor) | (nb > floor)
    expected_rms = float(np.sqrt(np.mean((na[lit] - nb[lit]) ** 2)))
    rms, count = family.profile_rms(a, b, floor)
    assert rms == pytest.approx(expected_rms)
    assert count == int(lit.sum())


def test_profile_rms_raises_when_a_profile_is_all_zero(family):
    zero = np.zeros(5)
    nonzero = np.array([0.0, 1.0, 2.0, 1.0, 0.0])
    with pytest.raises(ValueError):
        family.profile_rms(zero, nonzero, floor=0.05)
    with pytest.raises(ValueError):
        family.profile_rms(nonzero, zero, floor=0.05)
