"""``lumice_integral.symmetry.attitude``.

The writing repository's ``test_attitude.py`` compares against its drawing
package and a ch3 chapter helper (not migrated); here the independent
references are a hand-written ``Ry``, the orthonormality of the pose, and
the optics trace of a parallel ray path (:func:`.optics.path_domain_batch`):
for a path whose refractions cancel, the sky point is exactly
``R M R^T s_hat``.
"""

from __future__ import annotations

import numpy as np
import pytest

from lumice_integral import optics
from lumice_integral.camera import incident_direction_from_sun, sun_direction
from lumice_integral.prescan import haar_rotations
from lumice_integral.symmetry import reflection_group as rg
from lumice_integral.symmetry.attitude import Ry, column_attitude, sky_direction


def test_ry_is_the_right_handed_rotation_about_y():
    for deg in (90.0, 37.0, -120.0):
        t = np.radians(deg)
        want = np.array([[np.cos(t), 0.0, np.sin(t)], [0.0, 1.0, 0.0], [-np.sin(t), 0.0, np.cos(t)]])
        assert np.abs(Ry(deg) - want).max() < 1e-15
    np.testing.assert_allclose(Ry(90.0) @ [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], atol=1e-15)  # z -> x


@pytest.mark.parametrize("psi,theta", [(0.0, 0.0), (33.0, 210.0), (-70.0, 5.0)])
def test_column_attitude_is_a_rotation_with_horizontal_c_axis_at_azimuth_psi(psi, theta):
    R = column_attitude(psi, theta)
    assert np.abs(R.T @ R - np.eye(3)).max() < 1e-12 and abs(np.linalg.det(R) - 1.0) < 1e-12
    assert np.abs(R[:, 2] - [np.cos(np.radians(psi)), np.sin(np.radians(psi)), 0.0]).max() < 1e-12


@pytest.mark.parametrize("faces,number", [((3, 1, 6), 11), ((1, 3, 5, 2), 4), ((3, 5, 6, 7, 3), 2), ((3, 1, 4, 5), 7)])
def test_sky_direction_of_a_parallel_path_is_the_traced_sky_point(faces, number):
    """``sky_direction(M, R, s_hat)`` equals ``-(traced outgoing direction)`` wherever the path is valid.

    The identity holds for any pose ``R``, so Haar poses are used; the paths
    are ch3 representatives that the optics gates admit (several others,
    e.g. ``3-1-5-7-4``, need a non-TIR internal reflection and never are).
    """
    M = rg.path_matrix(faces)
    assert rg.refraction_cancels(faces) and rg.identify(M).number == number
    sun = sun_direction(20.0)
    poses = haar_rotations(4000, np.random.default_rng(7))
    world = optics.path_domain_batch(poses, faces, incident_direction_from_sun(sun), 1.31)
    valid = np.asarray(world.valid)
    assert np.count_nonzero(valid) > 20
    sky = -np.asarray(world.direction)[valid]
    want = np.stack([sky_direction(M, R, sun) for R in poses[valid]])
    np.testing.assert_allclose(sky, want, atol=1e-12)


def test_theta_drops_out_for_elements_commuting_with_rz_and_not_otherwise():
    """#3 (``b R_z(-120)``) commutes with ``R_z(theta)``: the sky point ignores theta; #2 (``S_90``) does not."""
    sun = sun_direction(22.0)
    br = rg.by_number(3).matrix
    points = [sky_direction(br, column_attitude(40.0, th), sun) for th in (0.0, 17.0, 100.0, 259.5)]
    assert max(np.abs(p - points[0]).max() for p in points) < 1e-12
    s_matrix = rg.by_number(2).matrix
    p0, p1 = (sky_direction(s_matrix, column_attitude(40.0, th), sun) for th in (10.0, 50.0))
    assert np.linalg.norm(p0 - p1) > np.radians(2.0)


def test_sky_direction_batch_matches_single():
    M = rg.by_number(8).matrix
    R = column_attitude(12.0, 34.0)
    suns = np.stack([sun_direction(e) for e in (0.0, 10.0, 40.0)])
    batch = sky_direction(M, R, suns)
    assert batch.shape == (3, 3)
    for k, sv in enumerate(suns):
        assert np.abs(batch[k] - sky_direction(M, R, sv)).max() < 1e-15
