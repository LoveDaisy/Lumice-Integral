"""``path_class.path_class_symmetry``: the proper ``D6h`` element that maps the representative onto each member."""

from __future__ import annotations

import numpy as np

from lumice_integral.canonical_scene import canonical_crystal
from lumice_integral.path_class import (
    _hexprism_normals,
    _symmetry_image_of_faces,
    build_path_class,
    hexprism_symmetry_matrices,
    path_class_symmetry,
)


def test_class_3_5_is_one_proper_orbit() -> None:
    crystal = canonical_crystal()
    path_class = build_path_class(crystal, (3, 5))
    symmetry = path_class_symmetry(path_class, crystal)
    normals = _hexprism_normals(crystal)
    assert set(symmetry) == set(path_class.members) and len(symmetry) == 12
    assert np.array_equal(symmetry[(3, 5)], np.eye(3))
    for member, g in symmetry.items():
        assert g is not None, member
        assert np.isclose(np.linalg.det(g), 1.0)
        assert np.allclose(g @ g.T, np.eye(3), atol=1e-14)
        assert _symmetry_image_of_faces(g, path_class.representative, normals) == member


def test_class_3_1_2_5_needs_improper_elements_for_half_its_members() -> None:
    """24 members, trivial stabiliser: the 12 proper elements reach 12, the other 12 need a mirror."""
    crystal = canonical_crystal()
    path_class = build_path_class(crystal, (3, 1, 2, 5))
    symmetry = path_class_symmetry(path_class, crystal)
    assert len(symmetry) == 24
    missing = [member for member, g in symmetry.items() if g is None]
    assert len(missing) == 12
    assert (3, 2, 1, 5) in missing  # the basal swap z -> -z alone
    normals = _hexprism_normals(crystal)
    improper = [e for e in hexprism_symmetry_matrices() if np.linalg.det(e) < 0.0]
    for member in missing:
        assert any(_symmetry_image_of_faces(e, path_class.representative, normals) == member for e in improper)


def test_restricted_elements_fall_back_to_none() -> None:
    """Dropping one proper element leaves exactly the member it reached without a ``g``."""
    crystal = canonical_crystal()
    path_class = build_path_class(crystal, (3, 5))
    full = path_class_symmetry(path_class, crystal)
    dropped_member = (5, 7)
    dropped = full[dropped_member]
    elements = [e for e in hexprism_symmetry_matrices() if not np.array_equal(e, dropped)]
    assert len(elements) == 23
    restricted = path_class_symmetry(path_class, crystal, symmetry_elements=elements)
    assert restricted[dropped_member] is None
    assert all(restricted[m] is not None for m in path_class.members if m != dropped_member)
