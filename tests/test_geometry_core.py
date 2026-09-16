"""Face-numbering and convention cross-checks between ``lumice_integral.geometry`` and ``optics``.

The blueprint had no standalone ``core`` test file (its ``core.py`` assertions live in the unfold /
pyramid / feasibility tests).  This file adds the one check the migration acceptance criteria call
for explicitly: the hexagonal-prism face chirality must be bit-for-bit the one ``optics.py`` hard-codes.
"""

import numpy as np
import pytest

from lumice_integral.geometry import BASAL_BOTTOM, BASAL_TOP, HexPrism, N_ICE, PRISM_FACES, rotation
from lumice_integral.optics import FACE_3_NORMAL, FACE_5_NORMAL, ICE_REFRACTIVE_INDEX

from _geometry_oracles import NORMALS


def test_face_3_and_5_normals_match_optics_constants():
    """``optics.path_3_5`` uses ``FACE_3_NORMAL`` / ``FACE_5_NORMAL`` as body-frame normals; the geometry
    package derives the same faces from vertex rings via Newell's formula.  Both must agree exactly, or
    ``entry_measure`` and the direction map would silently live in mirrored crystals."""
    c = HexPrism()
    np.testing.assert_allclose(c.normal(c.face(3)), np.asarray(FACE_3_NORMAL), atol=1e-15)
    np.testing.assert_allclose(c.normal(c.face(5)), np.asarray(FACE_5_NORMAL), atol=1e-15)


@pytest.mark.parametrize("number", [BASAL_TOP, BASAL_BOTTOM, *PRISM_FACES])
def test_all_hexprism_normals_match_closed_form(number):
    """Face ``3 + i`` has azimuth ``i * 60`` degrees counter-clockwise from +x; basal faces are +-z."""
    c = HexPrism(a=1.3, h=0.4)
    np.testing.assert_allclose(c.normal(c.face(number)), NORMALS[number], atol=1e-15)


def test_sixty_degree_rotation_shifts_prism_face_number():
    """Rotating the crystal by +60 degrees about c maps face ``3+i`` onto where face ``3+i+1`` was."""
    c = HexPrism()
    r = c.transformed(rotation([0, 0, 1], 60.0))
    for i in range(6):
        np.testing.assert_allclose(r.normal(r.face(3 + i)), c.normal(c.face(3 + (i + 1) % 6)), atol=1e-15)


def test_n_ice_matches_optics_refractive_index():
    assert N_ICE == float(ICE_REFRACTIVE_INDEX)
