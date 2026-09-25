"""``D6h`` equivariance of the field layer: a member's critical set is the representative's moved by ``u -> g u``."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from lumice_integral.canonical_scene import canonical_crystal
from lumice_integral.dp_field import DPField
from lumice_integral.path_class import build_path_class, hexprism_symmetry_matrices, path_class_symmetry

N = 1.31
IMPROPER = [e for e in hexprism_symmetry_matrices() if np.linalg.det(e) < 0.0]


def _element(representative, member, elements=None) -> np.ndarray:
    path_class = build_path_class(canonical_crystal(), representative)
    return path_class_symmetry(path_class, symmetry_elements=elements)[member]


@pytest.mark.parametrize(
    ("representative", "member", "elements", "det"),
    [
        ((3, 5), (3, 7), None, 1.0),  # C2' rotation, the first match of the default search
        ((3, 5), (3, 7), IMPROPER, -1.0),  # the same member reached by a mirror
        ((3, 5, 6, 7, 3), (4, 8, 7, 6, 4), None, 1.0),
        ((3, 5, 6, 7, 3), (4, 8, 7, 6, 4), IMPROPER, -1.0),
    ],
)
def test_member_critical_set_is_the_transported_representative(representative, member, elements, det) -> None:
    g = _element(representative, member, elements)
    assert np.linalg.det(g) == pytest.approx(det)
    crystal = canonical_crystal()
    rep = DPField.build(crystal, representative, N)
    other = DPField.build(crystal, member, N)
    moved = rep.critical_set.transported(g)
    mine = other.critical_set
    # corners and interior points are Newton-exact
    exact = (dataclasses.replace(mine, boundary=()), dataclasses.replace(moved, boundary=()))
    angle, value = exact[0].mismatch(exact[1])
    assert angle <= 1e-9 and value <= 5e-8
    # loop extrema: values to the ~1e-8 of the exit-TIR pieces; positions of a flat extremum only to ~sqrt of that
    angle, value = mine.mismatch(moved)
    assert angle <= 3e-4 and value <= 5e-8
    assert len(other.corners) == len(rep.corners)
    assert len(other.interior_critical_points) == len(rep.interior_critical_points)
    for mine, theirs in zip(other.interval_partition(), rep.interval_partition()):
        assert mine[2:] == theirs[2:]
        assert mine.lower == pytest.approx(theirs.lower, abs=5e-8) and mine.upper == pytest.approx(theirs.upper, abs=5e-8)


def test_field_values_transport() -> None:
    """``D_{gPg^-1}(g u) = D_P(u)`` pointwise, proper and improper ``g``."""
    crystal = canonical_crystal()
    rep = DPField.build(crystal, (3, 5), N)
    other = DPField.build(crystal, (3, 7), N)
    u = np.random.default_rng(2).normal(size=(4096, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    for elements in (None, IMPROPER):
        g = _element((3, 5), (3, 7), elements)
        np.testing.assert_allclose(other.d_p_batch(u @ g.T), rep.d_p_batch(u), atol=1e-13, equal_nan=True)
