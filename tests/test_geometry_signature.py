"""Path-level invariants ``wedge_angle_deg`` / ``halo_map_rank`` on the hexagonal prism.

The angle baselines are the family wedge angles of the writing repository's
``halo_notes.math.signature`` (0 deg parallel, 60 deg for the 22-deg family,
90 deg for the 46-deg family); that module is not imported, only its
documented values are asserted.  The oracle side uses the closed-form
normals and ``path_matrix`` of ``_geometry_oracles``.
"""

from __future__ import annotations

import numpy as np
import pytest

from lumice_integral.geometry import HexPrism, halo_map_rank, wedge_angle_deg

from _geometry_oracles import NORMALS, path_matrix, refraction_cancels

CRYSTAL = HexPrism(1.0, 0.8)


@pytest.mark.parametrize(
    "faces, expected_deg",
    [
        ((3, 6), 0.0),
        ((1, 2), 0.0),
        ((3, 5), 60.0),
        ((3, 7), 60.0),
        ((3, 1, 2, 5), 60.0),
        ((1, 3), 90.0),
        ((3, 1), 90.0),
        ((3, 4), 120.0),
        ((1, 3, 2), 0.0),
    ],
)
def test_known_hexprism_wedge_angles(faces, expected_deg):
    assert wedge_angle_deg(CRYSTAL, faces) == pytest.approx(expected_deg, abs=1e-9)


@pytest.mark.parametrize("faces", [(3, 5), (3, 7), (3, 1, 2, 5), (1, 3, 2), (3, 1, 5, 7, 4), (1, 2, 3, 5, 1)])
def test_wedge_angle_matches_the_oracle_matrix(faces):
    expected = np.degrees(np.arccos(-NORMALS[faces[0]] @ (path_matrix(faces).T @ NORMALS[faces[-1]])))
    assert wedge_angle_deg(CRYSTAL, faces) == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize("faces", [(1, 2), (2, 1), (3, 6), (4, 7), (5, 8)])
def test_a0_06_family_is_rank_zero(faces):
    assert refraction_cancels(faces)
    assert np.allclose(path_matrix(faces), np.eye(3))
    assert halo_map_rank(CRYSTAL, faces) == 0


@pytest.mark.parametrize("faces", [(3, 5), (3, 7), (3, 1, 2, 5), (1, 3), (3, 4)])
def test_deviating_paths_are_rank_two(faces):
    assert halo_map_rank(CRYSTAL, faces) == 2


def test_parallel_path_with_non_identity_fold_matrix_is_rank_two():
    # 1-3-2 refracts out parallel to the incident ray in the *body* frame, but
    # M = S_3 != I rotates the world direction with the pose: rank 2, not 0.
    faces = (1, 3, 2)
    assert refraction_cancels(faces)
    assert not np.allclose(path_matrix(faces), np.eye(3))
    assert wedge_angle_deg(CRYSTAL, faces) == pytest.approx(0.0, abs=1e-9)
    assert halo_map_rank(CRYSTAL, faces) == 2


def test_faces_must_have_entry_and_exit():
    with pytest.raises(ValueError):
        wedge_angle_deg(CRYSTAL, (3,))
    with pytest.raises(ValueError):
        halo_map_rank(CRYSTAL, (3,))
