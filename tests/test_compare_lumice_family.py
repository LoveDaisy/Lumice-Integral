"""``scripts/compare_lumice_family.py``: its vectorised pixel solid angles are the probe's, on any linear camera."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from lumice_integral.canonical_scene import CANONICAL_RENDER, canonical_crystal
from lumice_integral.geometry import HexPrism
from lumice_integral.path_class import build_path_class

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


D3H = HexPrism.from_lumice(1.0, (1.0, 1.2, 1.0, 1.2, 1.0, 1.2))
G2 = HexPrism.from_lumice(1.0, (1.0, 1.3, 0.7, 1.9, 1.1, 0.4))


def _config(filters: list[dict], top: int) -> dict:
    return {"filter": filters, "scene": {"scattering": [{"prob": 0, "entries": [{"crystal": 1, "proportion": 100, "filter": top}]}]}}


def _exact(members) -> list[dict]:
    return [{"id": k + 1, "type": "raypath", "raypath": list(m), "action": "filter_in"} for k, m in enumerate(members)]


def _complex(members) -> dict:
    parts = _exact(members)
    return _config([*parts, {"id": 99, "type": "complex", "composition": [p["id"] for p in parts]}], 99)


def _members(crystal, path):
    return [list(m) for m in build_path_class(crystal, path).members]


def test_check_filter_accepts_pbd_only_on_the_regular_prism(family):
    pbd = _config([{"id": 1, "type": "raypath", "raypath": [3, 5], "symmetry": "PBD", "action": "filter_in"}], 1)
    assert family.check_filter(pbd, [3, 5], _members(canonical_crystal(), [3, 5]), canonical_crystal()) == "PBD"
    with pytest.raises(SystemExit, match="smaller than D6h"):
        family.check_filter(pbd, [3, 5], _members(D3H, [3, 5]), D3H)
    with pytest.raises(SystemExit, match="Lumice PBD admits"):  # a single-path render is not the PBD class
        family.check_filter(pbd, [3, 5], [[3, 5]], canonical_crystal())


@pytest.mark.parametrize(("crystal", "path"), [(D3H, [3, 5]), (D3H, [3, 5, 6, 7]), (G2, [3, 5]), (canonical_crystal(), [3, 5])])
def test_check_filter_accepts_exact_members_alone_or_ored(family, crystal, path):
    members = _members(crystal, path)
    assert family.check_filter(_complex(members[::-1]), path, members, crystal) == "complex"
    if len(members) == 1:
        assert family.check_filter(_config(_exact(members), 1), path, members, crystal) == "exact"


@pytest.mark.parametrize(
    "change",
    [
        lambda m: m[:-1],  # a member missing
        lambda m: [*m, [4, 6, 7, 8]],  # an extra path (a PBD image that is not a G_true image on D3h)
        lambda m: [m[0][::-1], *m[1:]],  # a member's faces in the wrong order
        lambda m: [*m, m[0]],  # a member twice
    ],
)
def test_check_filter_rejects_a_different_member_set(family, change):
    members = _members(D3H, [3, 5, 6, 7])
    with pytest.raises(SystemExit, match="admits"):
        family.check_filter(_complex(change(members)), [3, 5, 6, 7], members, D3H)


def test_check_filter_rejects_folded_or_non_raypath_parts(family):
    members = _members(D3H, [3, 5])
    config = _complex(members)
    config["filter"][0]["symmetry"] = "P"
    with pytest.raises(SystemExit, match="without symmetry"):
        family.check_filter(config, [3, 5], members, D3H)
    nested = _complex(members)
    nested["filter"][-1]["composition"][0] = [1, 2]
    with pytest.raises(SystemExit, match="without symmetry"):
        family.check_filter(nested, [3, 5], members, D3H)
