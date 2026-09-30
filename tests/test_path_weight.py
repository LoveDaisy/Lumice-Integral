"""``path_weight``: the one ``A T`` kernel, against the two factor authorities called directly (an independent oracle)."""

from __future__ import annotations

import ast

import numpy as np
import pytest

from lumice_integral import chromatic, optics, path_weight, s2_store, weights
from lumice_integral.camera import incident_direction_from_sun
from lumice_integral.geometry import HexPrism, Pyramid
from lumice_integral.geometry.entry_measure import entry_measure_batch
from lumice_integral.pose_density import HaarUniformPoseDensity
from lumice_integral.so3 import haar_rotations

from _dependency_direction import PACKAGE_ROOT

SUN = np.array([0.3, -0.2, 0.93]) / np.linalg.norm([0.3, -0.2, 0.93])
CRYSTALS = {
    "prism": HexPrism(),
    "rhombic": HexPrism(a=1.0, h=2.0, face_distance=[1.5, 1.0, 1.0, 1.5, 1.0, 1.0]),
    "pyramid": Pyramid.from_lumice(1.0, 0.3, 0.3),
}
INDICES = (1.307, 1.31, 1.317)
PATHS = ((3, 5), (3, 1, 5), (1, 3, 5, 2))


@pytest.fixture(scope="module")
def rotations() -> np.ndarray:
    return haar_rotations(3000, np.random.default_rng(5))


def _oracle(rotations, faces, crystal, index):
    s = incident_direction_from_sun(SUN)
    area = entry_measure_batch(rotations, faces, s, crystal, n_ice=index)
    power = optics.fresnel_transmission_path_batch(rotations, faces, s, index, crystal=crystal)
    return area, power


@pytest.mark.parametrize("name", list(CRYSTALS))
@pytest.mark.parametrize("index", INDICES)
def test_kernel_and_its_callers_equal_the_factor_authorities_bit_for_bit(rotations, name, index) -> None:
    crystal = CRYSTALS[name]
    s = incident_direction_from_sun(SUN)
    fields = s2_store.evaluate_fields(rotations, SUN, crystal, index, list(PATHS))
    expected_w = None
    for m, faces in enumerate(PATHS):
        area, power = _oracle(rotations, faces, crystal, index)
        assert np.any(area * power > 0.0), faces  # the comparison is not vacuous
        got_area, got_power = path_weight.entry_and_power(rotations, faces, s, index, crystal=crystal)
        np.testing.assert_array_equal(got_area, area)
        np.testing.assert_array_equal(got_power, power)
        np.testing.assert_array_equal(path_weight.weighted_power(rotations, faces, s, index, crystal=crystal), area * power)
        np.testing.assert_array_equal(chromatic.weighted_power(rotations, faces, s, index, crystal=crystal), area * power)
        np.testing.assert_array_equal(fields["A"][m], area)
        np.testing.assert_array_equal(fields["T"][m], power)
        expected_w = area * power if expected_w is None else expected_w + area * power
        evaluators = weights.build_path_weight_evaluators(
            faces=faces, incident_direction=s, refractive_index=index, crystal=crystal, pose_density=HaarUniformPoseDensity()
        )
        np.testing.assert_array_equal(evaluators["entry_measure"].evaluate_batch(rotations), area)
        np.testing.assert_array_equal(evaluators["fresnel_transmission"].evaluate_batch(rotations), power)
    np.testing.assert_array_equal(fields["w"], expected_w)


def test_chromatic_weighted_power_is_the_kernel() -> None:
    assert chromatic.weighted_power is path_weight.weighted_power


def test_the_factor_authorities_are_called_together_only_in_the_kernel() -> None:
    """a56: outside :mod:`path_weight` no production module calls both ``entry_measure_batch`` and the path power."""
    both = {"entry_measure_batch", "fresnel_transmission_path_batch"}
    offenders = []
    for path in PACKAGE_ROOT.rglob("*.py"):
        if path.name == "path_weight.py":
            continue
        called = set()
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Call):
                func = node.func
                called.add(func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None))
        if both <= called:
            offenders.append(str(path.relative_to(PACKAGE_ROOT)))
    assert offenders == []


@pytest.mark.parametrize("index", [float("nan"), -1.0])
def test_weighted_power_validates_like_path_domain_batch(index):
    """The random-orientation class branch dropped its direct ``path_domain_batch`` call; ``weighted_power``
    must keep rejecting what it rejects (it reaches it through ``fresnel_transmission_path_batch``)."""
    from lumice_integral.geometry import HexPrism

    crystal = HexPrism.from_ratio(2.0)
    rotations = np.stack([np.eye(3)] * 2)
    s = np.array([0.0, 0.0, -1.0])
    with pytest.raises(ValueError):
        optics.path_domain_batch(rotations, (3, 1, 6), s, index, crystal=crystal)
    with pytest.raises(ValueError):
        path_weight.weighted_power(rotations, (3, 1, 6), s, index, crystal=crystal)
