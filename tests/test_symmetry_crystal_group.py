"""The crystal's own symmetry group ``G_true`` (``symmetry.crystal_group``) on the design explore's samples."""

from __future__ import annotations

import numpy as np
import pytest

from lumice_integral.geometry import HexPrism, Pyramid
from lumice_integral.symmetry import reflection_group as rg
from lumice_integral.symmetry.crystal_group import check_group, true_symmetry_group
from lumice_integral.symmetry.signature import D6H


@pytest.mark.parametrize("crystal,order", [
    (HexPrism(), 24),                                        # D6h
    (HexPrism(1.0, 2.7), 24),
    (HexPrism(1.0, 1.0, (1, 1.2, 1, 1.2, 1, 1.2)), 12),      # unequal hexagon, D3h
    (HexPrism(1.0, 1.0, (1, 2, 1, 2, 1, 2)), 12),            # critical: a triangle, D3h
    (HexPrism(1.0, 1.0, (1.9, 1, 1, 1.9, 1, 1)), 8),         # D2h
    (HexPrism(1.0, 1.0, (2, 1, 1, 2, 1, 1)), 8),             # critical: a rhombus, D2h
    (HexPrism(1.0, 1.0, (1.0, 1.3, 0.7, 1.9, 1.1, 0.4)), 2), # {E, sigma_h}
    (Pyramid(), 24),
    (Pyramid(0.5, 3.0, 1.1, 0.2), 24),
])
def test_group_order(crystal, order):
    group = true_symmetry_group(crystal)
    assert len(group) == order
    check_group(group)   # the function checks it too; kept here as an independent statement of the axioms


def test_the_group_is_a_subset_of_d6h_in_its_order_and_contains_sigma_h():
    group = true_symmetry_group(HexPrism(1.0, 1.0, (1.0, 1.3, 0.7, 1.9, 1.1, 0.4)))
    keys = [rg.key(g) for g in D6H]
    indices = [keys.index(rg.key(g)) for g in group]
    assert indices == sorted(indices)
    assert {rg.key(g) for g in group} == {rg.key(np.eye(3)), rg.key(np.diag([1.0, 1.0, -1.0]))}


def test_offsets_are_measured_from_the_centroid():
    """``[1, 1, 1, 1.5, 1.5, 1.5]`` is the triangle-truncated hexagon of ``[1.25 ± 0.25 alternating]`` moved off the
    axis: about its own centre it is D3h, about the origin it would only keep a mirror."""
    crystal = HexPrism(1.0, 1.0, (1, 1, 1, 1.5, 1.5, 1.5))
    assert len(true_symmetry_group(crystal)) == 12
    assert len(true_symmetry_group(crystal.transformed(translation=[0.3, -0.2, 0.5]))) == 12


def test_the_regular_prism_group_is_all_of_d6h():
    assert [rg.key(g) for g in true_symmetry_group(HexPrism())] == [rg.key(g) for g in D6H]


def test_check_group_rejects_a_non_group():
    with pytest.raises(AssertionError, match="identity"):
        check_group((rg.Rz(60),))
    with pytest.raises(AssertionError, match="product|inverse"):
        check_group((np.eye(3), rg.Rz(60)))
