"""The conventions table (``docs/conventions.md``): each row's numeric check.

Lumice and the writing series are read as evidence only; the few formulas
used from them are transcribed here with their source, never imported.
The writing series' pose construction ``column_attitude`` now lives in this
repository (:mod:`lumice_integral.symmetry.attitude`) and is imported as the
object under test; the Lumice chain stays transcribed.
"""

from __future__ import annotations

import numpy as np
import pytest

from lumice_integral import optics
from lumice_integral.camera import incident_direction_from_sun, sun_direction
from lumice_integral.canonical_scene import CANONICAL_REFRACTIVE_INDEX, canonical_crystal, canonical_sun_direction
from lumice_integral.geometry import HexPrism, fold_matrix
from lumice_integral.path_class import hexprism_symmetry_matrices, phi_key
from lumice_integral.pose_density import c_axis_roll, c_axis_zenith
from lumice_integral.s2_store import align_rotations, evaluate_fields, store_lattice
from lumice_integral.symmetry.attitude import column_attitude


def rz(deg: float) -> np.ndarray:
    t = np.radians(deg)
    return np.array([[np.cos(t), -np.sin(t), 0.0], [np.sin(t), np.cos(t), 0.0], [0.0, 0.0, 1.0]])


def ry(deg: float) -> np.ndarray:
    t = np.radians(deg)
    return np.array([[np.cos(t), 0.0, np.sin(t)], [0.0, 1.0, 0.0], [-np.sin(t), 0.0, np.cos(t)]])


def lumice_chain(az_deg: float, zenith_deg: float, roll_deg: float) -> np.ndarray:
    """Lumice ``doc/coordinate-convention.md`` section 6: ``Rz(az - 180) . Ry(-zenith) . Rz(roll)``."""
    return rz(az_deg - 180.0) @ ry(-zenith_deg) @ rz(roll_deg)


# ------------------------------------------------------------ crystal frame
def test_face_numbering_is_lumices():
    """Face 1 = +z (c axis), face 3 + i at azimuth i * 60 deg (Lumice ``geo3d_closedform.hpp`` kHexFaceCos/Sin)."""
    lumice_cos = [1.0, 0.5, -0.5, -1.0, -0.5, 0.5]
    lumice_sin = [0.0, np.sqrt(3.0) / 2.0, np.sqrt(3.0) / 2.0, 0.0, -np.sqrt(3.0) / 2.0, -np.sqrt(3.0) / 2.0]
    crystal = HexPrism.from_ratio(2.0)
    np.testing.assert_allclose(crystal.normal(crystal.face(1)), [0.0, 0.0, 1.0], atol=1e-15)
    np.testing.assert_allclose(crystal.normal(crystal.face(2)), [0.0, 0.0, -1.0], atol=1e-15)
    for i in range(6):
        np.testing.assert_allclose(crystal.normal(crystal.face(3 + i)), [lumice_cos[i], lumice_sin[i], 0.0], atol=1e-15)


# ------------------------------------------------------------ pose chain
@pytest.mark.parametrize("az", [0.0, 37.0, 180.0, -95.0])
@pytest.mark.parametrize("roll", [0.0, 20.0, 180.0, -73.0])
def test_column_attitude_is_the_lumice_chain_with_theta_equal_roll_minus_180(az, roll):
    """``R_Lumice(az, 90, roll) = column_attitude(psi = az, theta = roll - 180)`` (``Rz(180)`` conjugation)."""
    np.testing.assert_allclose(lumice_chain(az, 90.0, roll), column_attitude(az, roll - 180.0), atol=1e-15)
    rotation = column_attitude(az, roll - 180.0)
    assert c_axis_zenith(rotation) == pytest.approx(np.pi / 2.0, abs=1e-15)
    offset = (c_axis_roll(rotation) - np.radians(roll) + np.pi) % (2.0 * np.pi) - np.pi
    assert abs(offset) <= 1e-12


def test_roll_zero_puts_face_3_up_and_theta_zero_puts_it_down():
    """Horizontal c axis: Lumice / LI ``roll = 0`` has face 3 (body +x) on top; ``column_attitude`` theta = 0 below."""
    face_3 = np.array([1.0, 0.0, 0.0])
    np.testing.assert_allclose(lumice_chain(30.0, 90.0, 0.0) @ face_3, [0.0, 0.0, 1.0], atol=1e-15)
    np.testing.assert_allclose(column_attitude(30.0, 0.0) @ face_3, [0.0, 0.0, -1.0], atol=1e-15)


@pytest.mark.parametrize(
    "az, zenith, roll, n1_world, n3_world",
    [
        (0.0, 0.0, 0.0, [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]),  # Case A
        (0.0, 90.0, 0.0, [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),  # Case B = section 8, the Parry default
        (90.0, 90.0, 0.0, [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]),  # Case C
        (0.0, 0.0, 90.0, [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]),  # Case D
    ],
    ids=["A", "B", "C", "D"],
)
def test_chain_output_quick_reference(az, zenith, roll, n1_world, n3_world):
    """Lumice ``doc/coordinate-convention.md`` appendix "Chain Output Quick Reference" (``N1 = R e_z``, ``N3 = R e_x``)."""
    rotation = lumice_chain(az, zenith, roll)
    np.testing.assert_allclose(rotation @ [0.0, 0.0, 1.0], n1_world, atol=1e-15)
    np.testing.assert_allclose(rotation @ [1.0, 0.0, 0.0], n3_world, atol=1e-15)


def test_parry_default_holds_face_3_up_over_the_whole_azimuth_circle():
    """Section 5.3: at the Parry means (zenith 90, roll 0) ``N3`` is world ``+z`` for every azimuth, the c axis ``N1``
    is horizontal at azimuth ``az`` (section 8 at ``az = 0``: ``N1 = +x``) and ``pose_density`` reads zenith 90, roll 0.

    Roll 180 (``column_attitude``'s theta = 0) puts face 6 (body ``-x``) up instead: the same vertical body line.
    """
    for az in np.arange(0.0, 360.0, 7.5):
        rotation = lumice_chain(az, 90.0, 0.0)
        np.testing.assert_allclose(rotation @ [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], atol=1e-15)
        t = np.radians(az)
        np.testing.assert_allclose(rotation @ [0.0, 0.0, 1.0], [np.cos(t), np.sin(t), 0.0], atol=1e-15)
        assert c_axis_zenith(rotation) == pytest.approx(np.pi / 2.0, abs=1e-15)
        assert c_axis_roll(rotation) == pytest.approx(0.0, abs=1e-15)
        np.testing.assert_allclose(lumice_chain(az, 90.0, 180.0) @ [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], atol=1e-15)


def parry_support_deviations(faces, roll_deg: float, elevation_deg: float, samples: int = 720) -> np.ndarray:
    """``D`` (degrees) over the valid poses of the Parry sigma = 0 support (zenith 90, ``roll_deg``, azimuth uniform)."""
    rotations = np.stack([lumice_chain(az, 90.0, roll_deg) for az in np.linspace(0.0, 360.0, samples, endpoint=False)])
    sun = sun_direction(elevation_deg)
    check = optics.path_domain_batch(
        rotations, faces, incident_direction_from_sun(sun), CANONICAL_REFRACTIVE_INDEX, crystal=canonical_crystal()
    )
    sky = -np.asarray(check.direction)[np.asarray(check.valid)]
    return np.degrees(np.arccos(np.clip(sky @ sun, -1.0, 1.0)))


@pytest.mark.parametrize("elevation", [5.0, 15.0, 40.0])
def test_parry_roll_zero_pins_the_face_6_mirror_path_not_1_3_2(elevation):
    """Task parry-roll-convention-check: which labelled path the Parry family pins depends on roll 0 vs 180.

    At Lumice's roll 0 face 3 is the top face, so ``1-3-2`` (internal reflection on face 3) has no valid pose on
    the sigma = 0 support, while ``1-6-2`` (on the bottom face 6) runs the whole azimuth circle at one deviation,
    ``D = 2 h`` (the subsun): its fold matrix is the vertical-face mirror ``S_x``, which commutes with rotations
    about the vertical body axis ``+x``.  Roll 180 (``column_attitude`` theta = 0) swaps the two labels.
    ``1-4-2`` (mirror of a tilted face) and the 22-degree ``3-5`` entering the top face (``6-8`` at roll 180)
    stay spread.
    """
    crystal = canonical_crystal()
    np.testing.assert_allclose(fold_matrix(crystal, (1, 6, 2)), np.diag([-1.0, 1.0, 1.0]), atol=1e-15)
    for roll, pinned, empty, wedge in ((0.0, (1, 6, 2), (1, 3, 2), (3, 5)), (180.0, (1, 3, 2), (1, 6, 2), (6, 8))):
        assert parry_support_deviations(empty, roll, elevation).size == 0
        deviations = parry_support_deviations(pinned, roll, elevation)
        assert deviations.size > 100
        np.testing.assert_allclose(deviations, 2.0 * elevation, atol=1e-9)
        for spread in ((1, 4, 2), wedge):
            deviations = parry_support_deviations(spread, roll, elevation)
            assert deviations.size > 100 and np.ptp(deviations) > 5.0, spread


# ------------------------------------------------------------ sun and u
def test_sun_direction_is_the_writing_series_sun_vector():
    """``halo_notes/math/attitude.py::sun_vector``: ``(cos S, 0, sin S)``, toward the sun; light travels along ``-s_hat``."""
    for elevation in (0.0, 15.0, 22.0, 60.0):
        t = np.radians(elevation)
        np.testing.assert_allclose(sun_direction(elevation), [np.cos(t), 0.0, np.sin(t)], atol=1e-15)
    np.testing.assert_array_equal(incident_direction_from_sun(canonical_sun_direction()), -canonical_sun_direction())


def test_store_u_phi_and_d_are_framework_theorem_8():
    """``u = R^-1 s_hat``, ``phi = Phi_P(-u)``, sky point ``x = -R phi``, ``D_P(u) = angle(-Phi_P(-u), u) = angle(x, s_hat)``.

    ``Phi_P`` evaluated independently of the store: :func:`.optics.path_domain_batch`
    in the world frame with the propagation direction ``-s_hat``.
    """
    sun = canonical_sun_direction()
    u = store_lattice(4000)
    rotations = align_rotations(u, sun)
    np.testing.assert_allclose(np.einsum("nji,j->ni", rotations, sun), u, atol=1e-15)  # u = R^-1 s_hat
    fields = evaluate_fields(rotations, sun, canonical_crystal(), CANONICAL_REFRACTIVE_INDEX, [(3, 5)])
    valid = fields["valid"]
    assert np.count_nonzero(valid) > 100
    world = optics.path_domain_batch(rotations, (3, 5), incident_direction_from_sun(sun), CANONICAL_REFRACTIVE_INDEX)
    np.testing.assert_array_equal(np.asarray(world.valid), valid)
    outgoing = np.asarray(world.direction)[valid]
    np.testing.assert_allclose(np.einsum("nij,nj->ni", rotations[valid], fields["phi"][valid]), outgoing, atol=1e-14)
    sky = -outgoing  # the light point: the outgoing propagation direction reversed
    deviation = fields["D"][valid]
    np.testing.assert_allclose(np.cos(deviation), sky @ sun, atol=1e-12)  # angle(x, s_hat)
    np.testing.assert_allclose(np.cos(deviation), np.sum(-fields["phi"][valid] * u[valid], axis=1), atol=1e-12)
    assert np.all(np.degrees(deviation) > 20.0)  # 3-5 is a 60-degree wedge: the 22-degree halo, not 158


# ------------------------------------------------------------ D6h labels
@pytest.mark.parametrize("faces", [(3, 5), (3, 1, 2, 5), (1, 3, 5, 2), (3, 6, 4, 8), (1, 2)])
def test_phi_key_indexes_the_d6h_tuple_by_matrix_not_by_published_number(faces):
    """Task 19's interface: ``phi_key[0]`` indexes :func:`hexprism_symmetry_matrices` (``symmetry.signature.D6H`` order).

    Only the matrix is meaningful across projects
    (``lumice_integral.symmetry.reflection_group.identify(M)`` gives the
    published #1-#12); the index is not a published number.  Every prism
    fold matrix is one of the 24 and one of the 12 of the group ``G``
    generated by the four mirrors (``M^2`` has order dividing 3 and
    ``det M = +-1``).
    """
    crystal = canonical_crystal()
    M = fold_matrix(crystal, faces)
    index = phi_key(crystal, faces)[0]
    np.testing.assert_allclose(hexprism_symmetry_matrices()[index], M, atol=1e-12)
    xy = M[:2, :2]
    assert abs(M[2, 2]) == pytest.approx(1.0) and np.allclose(M[:2, 2], 0.0) and np.allclose(M[2, :2], 0.0)
    # G: the xy part is the identity, a rotation by +-120 deg or a mirror at 90 / +-30 deg (never +-60 / 180 deg).
    if np.linalg.det(xy) > 0.0:
        angle = np.degrees(np.arctan2(xy[1, 0], xy[0, 0]))
        assert min(abs(angle - a) for a in (0.0, 120.0, -120.0)) < 1e-9, angle
    else:
        axis = np.degrees(np.arctan2(xy[1, 0], xy[0, 0])) / 2.0  # the invariant line of the reflection
        assert min(abs((axis - a + 90.0) % 180.0 - 90.0) for a in (90.0, 30.0, -30.0)) < 1e-9, axis
