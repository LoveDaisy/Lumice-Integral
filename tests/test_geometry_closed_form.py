"""Closed-form hexagonal cross-section (``geometry.closed_form``) and the prism / pyramid built on it.

Oracles: the legacy vertex formulas (verbatim, the pre-closed-form ``HexPrism`` / ``Pyramid`` rings) for
the regular shapes, bit for bit; and a brute-force half-plane intersection (every pair of lines, keep the
feasible points) that shares nothing with the per-face interval test, for the presence mask and the ring.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from lumice_integral.geometry import BASAL_BOTTOM, BASAL_TOP, HexPrism, Pyramid, hex_cross_section
from lumice_integral.geometry.closed_form import SIDE_NORMALS_2D


# ---- legacy formulas -------------------------------------------------------------------------------------

def _legacy_hexprism_vertices(a: float, h: float) -> np.ndarray:
    ang = np.deg2rad(-30.0 + 60.0 * np.arange(6))
    ring = np.stack([a * np.cos(ang), a * np.sin(ang)], axis=1)
    return np.vstack([np.column_stack([ring, np.full(6, h / 2)]), np.column_stack([ring, np.full(6, -h / 2)])])


def _legacy_pyramid_vertices(a: float, h: float, c_over_a: float, tip_ratio: float) -> np.ndarray:
    cone_h = a * c_over_a * tip_ratio
    tip_a = a * (1.0 - tip_ratio)
    ang = np.deg2rad(-30.0 + 60.0 * np.arange(6))
    cs = np.stack([np.cos(ang), np.sin(ang)], axis=1)

    def ring(radius, z):
        return np.column_stack([radius * cs, np.full(6, z)])

    return np.vstack([ring(a, h / 2), ring(a, -h / 2), ring(tip_a, h / 2 + cone_h), ring(tip_a, -(h / 2 + cone_h))])


@pytest.mark.parametrize("a,h", [(1.0, 1.0), (0.5, 1.0), (1.0, 2.0), (0.37, 4.1), (2.3, 0.2)])
def test_regular_hexprism_is_the_legacy_prism_bit_for_bit(a, h):
    crystal = HexPrism(a, h)
    assert np.array_equal(crystal.vertices, _legacy_hexprism_vertices(a, h))
    assert [f.number for f in crystal.faces] == [1, 2, 3, 4, 5, 6, 7, 8]
    assert crystal.face(BASAL_TOP).vertex_ids == tuple(range(6))
    assert crystal.face(BASAL_BOTTOM).vertex_ids == (11, 10, 9, 8, 7, 6)
    for i in range(6):
        j = (i + 1) % 6
        assert crystal.face(3 + i).vertex_ids == (6 + i, 6 + j, j, i)
    assert crystal.face_distance_ratios == (1.0,) * 6


@pytest.mark.parametrize("a,h,c_over_a,tip_ratio", [(1.0, 1.0, 1.6288, 0.5), (0.5, 2.0, 1.6288, 0.2),
                                                    (1.3, 0.4, 0.9, 0.9), (0.7, 3.0, 2.4, 0.05)])
def test_pyramid_is_the_legacy_pyramid_bit_for_bit(a, h, c_over_a, tip_ratio):
    assert np.array_equal(Pyramid(a, h, c_over_a, tip_ratio).vertices, _legacy_pyramid_vertices(a, h, c_over_a, tip_ratio))


def test_face_distance_one_is_the_regular_apothem():
    """Lumice's ``face_distance`` unit is the regular apothem ``a·√3/2`` (``HexPrism(a)``'s side-plane distance)."""
    for a in (1.0, 0.5, 2.3):
        crystal = HexPrism(a, 1.0)
        for i in range(6):
            face = crystal.face(3 + i)
            assert crystal.normal(face) @ crystal.face_vertices(face)[0] == pytest.approx(a * np.sqrt(3.0) / 2.0, abs=1e-15)
        assert np.allclose(hex_cross_section(a).offsets, a * np.sqrt(3.0) / 2.0, rtol=0.0, atol=1e-15)


def test_from_lumice_height_is_over_the_circumscribed_diameter():
    crystal = HexPrism.from_lumice(1.0, a=0.5)
    assert crystal.h == pytest.approx(1.0)
    assert np.array_equal(crystal.vertices, HexPrism(0.5, 1.0).vertices)
    assert HexPrism.from_lumice(1.3, [1, 2, 1, 2, 1, 2]).face_distance_ratios == (1.0, 2.0, 1.0, 2.0, 1.0, 2.0)


# ---- presence mask ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("face_distance,present", [
    ((1, 1.2, 1, 1.2, 1, 1.2), (0, 1, 2, 3, 4, 5)),       # writing ch8 fig 7: an unequal hexagon
    ((1, 1.9, 1, 1.9, 1, 1.9), (0, 1, 2, 3, 4, 5)),
    ((1, 2, 1, 2, 1, 2), (0, 2, 4)),                      # far faces exactly touch the triangle's corners
    ((1, 2.5, 1, 2.5, 1, 2.5), (0, 2, 4)),
    ((1, 0.5, 1, 0.5, 1, 0.5), (1, 3, 5)),                # the complementary critical triangle
    ((1.9, 1, 1, 1.9, 1, 1), (0, 1, 2, 3, 4, 5)),
    ((2, 1, 1, 2, 1, 1), (1, 2, 4, 5)),                   # far pair exactly touches the rhombus' corners
    ((1.0, 1.3, 0.7, 1.9, 1.1, 0.4), (0, 1, 2, 4, 5)),    # the asymmetric sample of the design explore
    ((1, 1, 0.01, 1, 1, 1), (0, 1, 2, 3, 4, 5)),          # a near face grows, its neighbours shrink
    ((1, 1, -0.2, 1, 1, 1), (0, 2, 4, 5)),                # a negative distance removes the neighbours
])
def test_presence_mask(face_distance, present):
    section = hex_cross_section(1.0, face_distance)
    assert section.present == present
    assert section.face_present == tuple(i in present for i in range(6))
    crystal = HexPrism(1.0, 1.0, face_distance)
    assert sorted(f.number for f in crystal.faces) == [1, 2] + [3 + i for i in present]
    for i in set(range(6)) - set(present):
        with pytest.raises(KeyError):
            crystal.face(3 + i)


def test_critical_distance_is_decided_on_both_sides():
    """The far faces of ``[1, d, 1, d, 1, d]`` vanish exactly at the triangle's corner radius ``d = 2``."""
    for d, count in ((2.0 - 1e-6, 6), (2.0, 3), (2.0 + 1e-6, 3)):
        assert len(hex_cross_section(1.0, (1, d, 1, d, 1, d)).present) == count


@pytest.mark.parametrize("face_distance", [
    (1, 1, -1.5, 1, 1, 1), (1, 1, -1, 1, 1, -1), (-1,) * 6, (0,) * 6, (1, -1, 1, -1, 1, -1),
    (1, 1, -0.5, -0.9, -0.9, 1),    # every opposite pair has positive width, yet the three slabs miss
])
def test_no_area_is_rejected(face_distance):
    with pytest.raises(ValueError, match="no area"):
        hex_cross_section(1.0, face_distance)
    with pytest.raises(ValueError, match="no area"):
        HexPrism(1.0, 1.0, face_distance)


@pytest.mark.parametrize("bad", [(1, 1, 1, 1, 1), (1, 1, 1, 1, 1, np.nan), (1, 1, 1, 1, 1, np.inf)])
def test_malformed_face_distance_is_rejected(bad):
    with pytest.raises(ValueError, match="six finite"):
        hex_cross_section(1.0, bad)


def test_bad_lengths_are_rejected():
    with pytest.raises(ValueError):
        HexPrism(0.0, 1.0)
    with pytest.raises(ValueError):
        HexPrism(1.0, -1.0)


# ---- independent oracle: brute-force half-plane intersection --------------------------------------------

def _brute_force_polygon(offsets: np.ndarray, tol: float = 1e-9) -> tuple[set[int], np.ndarray]:
    """Every pairwise line intersection that satisfies all six half-planes; faces carrying two distinct ones."""
    points = []
    for i, j in itertools.combinations(range(6), 2):
        m = SIDE_NORMALS_2D[[i, j]]
        if abs(np.linalg.det(m)) < 1e-12:
            continue
        x = np.linalg.solve(m, offsets[[i, j]])
        if np.all(SIDE_NORMALS_2D @ x <= offsets + tol):
            points.append(x)
    points = np.array(points)
    scale = np.max(np.abs(offsets))
    faces = set()
    for i in range(6):
        on = points[np.abs(points @ SIDE_NORMALS_2D[i] - offsets[i]) <= tol * scale]
        if len(on) and np.max(np.linalg.norm(on[:, None] - on[None], axis=-1)) > 1e-7 * scale:
            faces.add(i)
    return faces, points


def test_random_cross_sections_agree_with_brute_force():
    rng = np.random.default_rng(20260927)
    checked = 0
    for _ in range(3000):
        face_distance = rng.uniform(-0.8, 2.5, 6)
        try:
            section = hex_cross_section(1.0, face_distance)
        except ValueError:
            continue
        faces, points = _brute_force_polygon(section.offsets)
        assert set(section.present) == faces, face_distance
        # every ring corner is a brute-force vertex, and every brute-force vertex is a ring corner
        distance = np.linalg.norm(section.ring[:, None] - points[None], axis=-1)
        assert distance.min(axis=1).max() < 1e-9
        assert distance.min(axis=0).max() < 1e-9
        checked += 1
    assert checked > 1500


@pytest.mark.parametrize("face_distance", [
    (1, 1, 1, 1, 1, 1), (1, 1.2, 1, 1.2, 1, 1.2), (1, 2, 1, 2, 1, 2), (2, 1, 1, 2, 1, 1),
    (1.0, 1.3, 0.7, 1.9, 1.1, 0.4), (1, 1, -0.2, 1, 1, 1), (0.3, 1.7, -0.1, 2.2, 0.9, 0.6),
])
def test_prism_is_a_closed_convex_polyhedron(face_distance):
    crystal = HexPrism(1.0, 1.7, face_distance)
    v, e, f = len(crystal.vertices), len(crystal.edges), len(crystal.faces)
    assert v - e + f == 2
    for face in crystal.faces:
        for point in crystal.face_vertices(face):
            assert crystal.contains(point, eps=1e-12)
        # outward: the centroid is strictly inside every face plane
        assert crystal.face_distance(face, crystal.centroid()) < 0.0
        assert all(len(crystal.edge_faces(edge)) == 2 for edge in crystal.edges)
    # side face normals are the fixed star, whatever the distances
    for face in crystal.faces:
        if face.number >= 3:
            assert np.allclose(crystal.normal(face)[:2], SIDE_NORMALS_2D[face.number - 3], atol=1e-12)


def test_transformed_and_mirrored_keep_the_face_set_and_distances():
    crystal = HexPrism(1.0, 1.0, (2, 1, 1, 2, 1, 1))
    moved = crystal.transformed(np.eye(3), [0.1, 0.2, 0.3])
    mirrored = crystal.mirrored(crystal.face(4))
    for copy in (moved, mirrored):
        assert type(copy) is HexPrism
        assert copy.face_distance_ratios == crystal.face_distance_ratios
        assert [f.number for f in copy.faces] == [f.number for f in crystal.faces]


# ---- the eroded cross-section of a cone (cone_sweep) ---------------------------------------------------------

from scipy.optimize import linprog  # noqa: E402

from lumice_integral.geometry.closed_form import SIDE_NORMALS_2D, cone_sweep, inset_corner  # noqa: E402


def _lp_apex(fd, a=1.0) -> float:
    """Independent oracle for the natural apex: max m with n_i·u / apothem + m <= fd_i (scipy's LP, not an enumeration)."""
    constraints = np.column_stack([SIDE_NORMALS_2D / (a * np.sqrt(3.0) / 2.0), np.ones(6)])
    result = linprog([0.0, 0.0, -1.0], A_ub=constraints, b_ub=fd, bounds=[(None, None)] * 3)
    return -result.fun


def test_inset_corner_at_zero_is_the_ring_bit_for_bit():
    for fd in [(1,) * 6, (1.0, 1.3, 0.7, 1.9, 1.1, 0.4), (1, 1, -0.2, 1, 1, 1)]:
        section = hex_cross_section(0.7, fd)
        p = section.present
        ring = np.stack([inset_corner(0.7, fd, p[k - 1], p[k], 0.0) for k in range(len(p))])
        np.testing.assert_array_equal(ring, section.ring)


def test_regular_cone_has_no_events_and_a_point_apex_at_one():
    sweep = cone_sweep(1.0)
    assert sweep.events == () and sweep.apex_present == tuple(range(6))
    assert sweep.m_apex == pytest.approx(1.0, abs=1e-15)
    assert len(sweep.apex_points) == 1 and sweep.apex_corner == (0,) * 6
    np.testing.assert_allclose(sweep.apex_points[0], 0.0, atol=1e-15)


def test_alternating_hexagon_loses_its_long_faces_then_closes_as_a_triangle():
    """``[1, 1.2]`` alternating: the faces at 1.2 shrink to zero at m = 0.8 (analytic: the triangle of the faces at 1
    has corner radius 2·apothem, the long faces sit 0.2 further out and move in at the triangle's rate), the
    triangle closes at m = 1."""
    sweep = cone_sweep(1.0, (1, 1.2, 1, 1.2, 1, 1.2))
    assert [event.dying for event in sweep.events] == [(1,), (3,), (5,)]
    assert all(event.m == pytest.approx(0.8, abs=1e-12) for event in sweep.events)
    assert sweep.m_apex == pytest.approx(1.0, abs=1e-12) and sweep.apex_present == (0, 2, 4)
    assert len(sweep.apex_points) == 1


def test_a_thin_cross_section_closes_on_a_ridge():
    """Faces 0 and 3 at 0.5: the width closes at m = 0.5 while the hexagon is still long, a ridge of two points."""
    sweep = cone_sweep(1.0, (0.5, 1, 1, 0.5, 1, 1))
    assert sweep.m_apex == pytest.approx(0.5, abs=1e-12)
    assert len(sweep.apex_points) == 2
    np.testing.assert_allclose(sweep.apex_points[:, 0], 0.0, atol=1e-12)   # on the y axis, between faces 0 and 3


@pytest.mark.parametrize("seed", range(4))
def test_sweep_agrees_with_an_lp_and_with_the_cross_section_at_every_height(seed):
    rng = np.random.default_rng(seed)
    checked = 0
    while checked < 60:
        fd = rng.uniform(-0.6, 2.4, 6)
        try:
            sweep = cone_sweep(1.0, fd)
        except ValueError:
            continue
        checked += 1
        assert sweep.m_apex == pytest.approx(_lp_apex(fd), abs=1e-9)
        for fraction in (0.1, 0.45, 0.8, 0.99):
            m = fraction * sweep.m_apex
            if any(abs(event.m - m) < 1e-6 for event in sweep.events):
                continue
            assert hex_cross_section(1.0, fd - m).present == sweep.present_at(m)
        for event in sweep.events:   # a real corner death: the dying run's lines pass through the new corner
            offsets = np.sqrt(3.0) / 2.0 * (fd - event.m)
            for i in (event.before, *event.dying, event.after):
                assert SIDE_NORMALS_2D[i] @ event.xy == pytest.approx(offsets[i], abs=1e-9)
