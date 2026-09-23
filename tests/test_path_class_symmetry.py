"""``path_class.path_class_symmetry``: a ``D6h`` element (proper first, else improper) mapping the representative onto each member."""

from __future__ import annotations

import numpy as np
import pytest

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
        assert np.isclose(np.linalg.det(g), 1.0)
        assert np.allclose(g @ g.T, np.eye(3), atol=1e-14)
        assert _symmetry_image_of_faces(g, path_class.representative, normals) == member


def test_class_3_1_2_5_gets_mirrors_for_the_members_no_rotation_reaches() -> None:
    """24 members, trivial stabiliser: the 12 proper elements reach 12, the other 12 get a mirror."""
    crystal = canonical_crystal()
    path_class = build_path_class(crystal, (3, 1, 2, 5))
    symmetry = path_class_symmetry(path_class, crystal)
    assert len(symmetry) == 24
    normals = _hexprism_normals(crystal)
    improper = [member for member, g in symmetry.items() if np.linalg.det(g) < 0.0]
    assert len(improper) == 12
    assert (3, 2, 1, 5) in improper  # the basal swap z -> -z alone
    for member, g in symmetry.items():
        assert np.allclose(g @ g.T, np.eye(3), atol=1e-14)
        assert _symmetry_image_of_faces(g, path_class.representative, normals) == member
    proper = [e for e in hexprism_symmetry_matrices() if np.linalg.det(e) > 0.0]
    for member in improper:
        assert not any(_symmetry_image_of_faces(e, path_class.representative, normals) == member for e in proper)


def test_restricted_elements_fall_back_to_another_element_or_fail() -> None:
    """Dropping one proper element: its member gets the mirror that also reaches it; dropping both is an error."""
    crystal = canonical_crystal()
    path_class = build_path_class(crystal, (3, 5))
    normals = _hexprism_normals(crystal)
    full = path_class_symmetry(path_class, crystal)
    dropped_member = (5, 7)
    dropped = full[dropped_member]
    elements = [e for e in hexprism_symmetry_matrices() if not np.array_equal(e, dropped)]
    assert len(elements) == 23
    restricted = path_class_symmetry(path_class, crystal, symmetry_elements=elements)
    assert np.linalg.det(restricted[dropped_member]) < 0.0
    assert _symmetry_image_of_faces(restricted[dropped_member], path_class.representative, normals) == dropped_member
    assert all(np.array_equal(restricted[m], full[m]) for m in path_class.members if m != dropped_member)
    reaching = [e for e in hexprism_symmetry_matrices() if _symmetry_image_of_faces(e, (3, 5), normals) == dropped_member]
    elements = [e for e in hexprism_symmetry_matrices() if not any(np.array_equal(e, r) for r in reaching)]
    with pytest.raises(RuntimeError, match="not a D6h image"):
        path_class_symmetry(path_class, crystal, symmetry_elements=elements)
