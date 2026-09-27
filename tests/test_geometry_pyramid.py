import numpy as np
import pytest

from lumice_integral.geometry import C_OVER_A_ICE, HexPrism, Polyhedron, Pyramid
from lumice_integral.geometry.pyramid import LOWER_PYRAMID_FACES, UPPER_PYRAMID_FACES, pyramid_face_angle

from _geometry_oracles import pyramid_reference_normals


@pytest.fixture(scope="module")
def reference_normals() -> dict[int, np.ndarray]:
    """闭式法向 oracle（蓝本用写作仓 ``docs/checks/check_pyramid.py`` 真跑一遍；本仓改用同一公式的独立实现）。"""
    ref = pyramid_reference_normals(C_OVER_A_ICE)
    assert len(ref) == 20
    return ref


def test_face_numbers_match_lumice_layout():
    p = Pyramid()
    assert sorted(f.number for f in p.faces) == [1, 2, *range(3, 9), *range(13, 19), *range(23, 29)]


def test_normals_match_closed_form_oracle(reference_normals):
    p = Pyramid(a=1.0, h=0.7)
    for number, ref in reference_normals.items():
        assert np.allclose(p.normal(p.face(number)), ref, atol=1e-12), number


def test_face_angle_is_62_degrees():
    assert pyramid_face_angle() == pytest.approx(62.001, abs=1e-3)
    p = Pyramid()
    for number in UPPER_PYRAMID_FACES:
        assert np.degrees(np.arccos(p.normal(p.face(number))[2])) == pytest.approx(pyramid_face_angle())
    for number in LOWER_PYRAMID_FACES:
        assert np.degrees(np.arccos(-p.normal(p.face(number))[2])) == pytest.approx(pyramid_face_angle())


@pytest.mark.parametrize("tip_ratio", [0.2, 0.5, 0.9])
def test_outward_normals_and_convexity(tip_ratio):
    p = Pyramid(a=1.0, h=0.5, tip_ratio=tip_ratio)
    c = p.centroid()
    for f in p.faces:
        assert p.normal(f) @ (p.centroid(f) - c) > 0, f.number
        # 凸：所有顶点都在每个面的内侧
        assert np.all(p.normal(f) @ (p.vertices - p.face_vertices(f)[0]).T <= 1e-12), f.number


def test_euler_characteristic():
    p = Pyramid()
    assert len(p.vertices) - len(p.edges) + len(p.faces) == 2
    assert len(p.vertices) == 24 and len(p.edges) == 42 and len(p.faces) == 20


def test_prism_band_agrees_with_hexprism():
    """侧面 3–8 的法向与顶点环与同参数 HexPrism 完全一致（锥晶 = 六棱柱两端加锥台）。"""
    p, q = Pyramid(a=1.3, h=0.4), HexPrism(a=1.3, h=0.4)
    for number in range(3, 9):
        assert np.allclose(p.normal(p.face(number)), q.normal(q.face(number)))
        assert np.allclose(p.face_vertices(p.face(number)), q.face_vertices(q.face(number)))


def test_cone_planes_pass_through_prism_ring_and_apex():
    """锥面所在平面过棱柱环顶点，且延长后交于轴上 ``h/2 + a·c_over_a`` 处（理论锥顶）。"""
    a, h, ca = 1.0, 0.6, 1.6288
    p = Pyramid(a=a, h=h, c_over_a=ca, tip_ratio=0.3)
    apex = np.array([0.0, 0.0, h / 2 + a * ca])
    for number in UPPER_PYRAMID_FACES:
        assert p.face_distance(p.face(number), apex) == pytest.approx(0.0, abs=1e-12)
        assert p.face_distance(p.face(number), -apex) != pytest.approx(0.0, abs=1e-6)


def test_tip_ratio_validation():
    with pytest.raises(ValueError):
        Pyramid(tip_ratio=0.0)
    with pytest.raises(ValueError):
        Pyramid(tip_ratio=1.0)


def test_transformed_and_mirrored_keep_type_and_fields():
    p = Pyramid(a=1.2, h=0.3, tip_ratio=0.4)
    q = p.transformed(translation=[1, 2, 3])
    m = p.mirrored(p.face(13))
    for r in (q, m):
        assert isinstance(r, Pyramid) and isinstance(r, Polyhedron)
        assert (r.a, r.h, r.c_over_a, r.tip_ratio) == (p.a, p.h, p.c_over_a, p.tip_ratio)
    # 镜像后外法向仍朝外（顶点环序已同步反转）
    for f in m.faces:
        assert m.normal(f) @ (m.centroid(f) - m.centroid()) > 0, f.number
    assert np.allclose(m.normal(m.face(13)), -p.normal(p.face(13)))


# ---- Lumice pyramid semantics: Miller indices, from_lumice ---------------------------------------------------

from lumice_integral.geometry.closed_form import SIDE_NORMALS_2D  # noqa: E402
from lumice_integral.geometry.pyramid import (  # noqa: E402
    LUMICE_MILLER_C_OVER_A,
    miller_indices_to_c_over_a,
    wedge_angle_is_legal,
    wedge_angle_to_c_over_a,
)

ALL_FACES = {1, 2, *range(3, 9), *range(13, 19), *range(23, 29)}


def _wedge_deg(c_over_a: float) -> float:
    return 90.0 - pyramid_face_angle(c_over_a)


def test_default_miller_indices_are_the_28_degree_face():
    """``(1, 0, 1)`` is Lumice's default ``{1, 0, -1, 1}`` face: wedge ≈ 28.0° to the c axis, i.e. the 62.0° face angle."""
    c = miller_indices_to_c_over_a((1, 0, 1))
    assert c == LUMICE_MILLER_C_OVER_A
    assert _wedge_deg(c) == pytest.approx(28.0, abs=0.01)
    assert pyramid_face_angle(c) == pytest.approx(pyramid_face_angle(C_OVER_A_ICE), abs=0.01)
    assert miller_indices_to_c_over_a((2, 0, 3)) == pytest.approx(2 * LUMICE_MILLER_C_OVER_A / 3)
    # the direct wedge path is the same relation
    assert wedge_angle_to_c_over_a(_wedge_deg(c)) == pytest.approx(c, rel=1e-12)


def test_miller_h_zero_means_no_cone():
    assert miller_indices_to_c_over_a((0, 0, 1)) is None
    assert miller_indices_to_c_over_a((0, 0, 0)) is None


@pytest.mark.parametrize("indices, match", [
    ((1, 0, -1, 1), "exactly three"),      # the GUI's four-index label
    ((1, 0), "exactly three"),
    ((1, 1, 1), "k = 1"),
    ((-1, 0, 1), "negative"),
    ((1, 0, -1), "negative"),
    ((1.5, 0, 1), "whole number"),
    ((1, 0, 0), "wedge angle"),            # l = 0: 0 degrees
    ((1, 0, 2000), "wedge angle"),         # 89.95 degrees
    ((2000, 0, 1), "wedge angle"),         # 0.02 degrees
])
def test_illegal_miller_indices_are_refused(indices, match):
    with pytest.raises(ValueError, match=match):
        miller_indices_to_c_over_a(indices)


def test_wedge_bounds_are_inclusive_in_float32():
    """Lumice compares ``0.1f <= alpha <= 89.9f``: both endpoints build a cone, whatever their double rounding."""
    assert wedge_angle_is_legal(0.1) and wedge_angle_is_legal(89.9)
    assert wedge_angle_is_legal(float(np.float32(89.9))) and wedge_angle_is_legal(float(np.float32(0.1)))
    assert not wedge_angle_is_legal(0.0999) and not wedge_angle_is_legal(89.9001) and not wedge_angle_is_legal(0.0)


def _assert_closed_convex(p: Pyramid) -> None:
    """Independent geometry: Euler, every edge on two faces, outward normals, every vertex inside every face plane."""
    assert len(p.vertices) - len(p.edges) + len(p.faces) == 2
    assert all(len(p.edge_faces(e)) == 2 for e in p.edges)
    scale = float(np.max(np.linalg.norm(p.vertices, axis=1)))
    c = p.centroid()
    for f in p.faces:
        assert len(f.vertex_ids) >= 3, f.number
        assert p.normal(f) @ (p.centroid(f) - c) > 0, f.number
        assert np.all(p.normal(f) @ (p.vertices - p.face_vertices(f)[0]).T <= 1e-12 * scale), f.number


def _faces(p: Pyramid) -> set[int]:
    return {f.number for f in p.faces}


def test_symmetric_regular_shape_delegates_to_the_legacy_constructor(monkeypatch):
    """The symmetric regular subset is ``Pyramid(a, h, c_over_a, tip_ratio)`` itself: the erosion path is not run."""
    def refuse(*args, **kwargs):
        raise AssertionError("the erosion construction must not run for the symmetric regular subset")

    monkeypatch.setattr(Pyramid, "_from_cones", classmethod(refuse))
    p = Pyramid.from_lumice(0.7, 0.4, 0.4, face_distance=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0], a=1.3)
    q = Pyramid(a=1.3, h=2 * 1.3 * 0.7, c_over_a=LUMICE_MILLER_C_OVER_A, tip_ratio=0.4)
    np.testing.assert_array_equal(p.vertices, q.vertices)
    assert [f for f in p.faces] == [f for f in q.faces]
    assert (p.c_over_a, p.tip_ratio) == (q.c_over_a, q.tip_ratio)
    assert p.shape.upper_h == p.shape.lower_h == 0.4 and p.shape.face_distance == (1.0,) * 6


@pytest.mark.parametrize("kwargs", [
    dict(upper_h=0.4, lower_h=0.3),                          # heights differ
    dict(upper_h=0.4, lower_h=0.4, lower_indices=(2, 0, 3)),  # indices differ
    dict(upper_h=0.4, lower_h=0.4, face_distance=(1, 1, 1, 1, 1, 1.0001)),
    dict(upper_h=0.0, lower_h=0.0),                          # symmetric, no cones: tip_ratio 0 is outside (0, 1)
    dict(upper_h=1.0, lower_h=1.0),                          # symmetric, full apex: tip_ratio 1 is outside (0, 1)
])
def test_everything_else_takes_the_erosion_construction(monkeypatch, kwargs):
    calls = []
    original = Pyramid._from_cones.__func__

    def counting(cls, *args):
        calls.append(args)
        return original(cls, *args)

    monkeypatch.setattr(Pyramid, "_from_cones", classmethod(counting))
    p = Pyramid.from_lumice(0.7, **kwargs)
    assert len(calls) == 1 and p.c_over_a is None and p.tip_ratio is None
    _assert_closed_convex(p)


@pytest.mark.parametrize("upper_h, lower_h", [(0.5, 0.3), (0.2, 0.9), (0.5, 0.5)])
def test_erosion_construction_matches_the_legacy_cone_on_a_regular_section(upper_h, lower_h):
    """Each side of an asymmetric regular-hexagon crystal is the legacy frustum of its own tip ratio (an independent
    derivation: similar triangles there, the eroded star here)."""
    a, prism_h = 0.8, 0.6
    p = Pyramid._from_cones(a, 2 * a * prism_h, (1.0,) * 6, (C_OVER_A_ICE, upper_h), (C_OVER_A_ICE, lower_h))
    for fraction, numbers in ((upper_h, (1, *UPPER_PYRAMID_FACES)), (lower_h, (2, *LOWER_PYRAMID_FACES))):
        legacy = Pyramid(a=a, h=2 * a * prism_h, c_over_a=C_OVER_A_ICE, tip_ratio=fraction)
        for number in (*numbers, *range(3, 9)):
            ours, theirs = p.face_vertices(p.face(number)), legacy.face_vertices(legacy.face(number))
            np.testing.assert_allclose(ours, theirs, rtol=0.0, atol=1e-14)


def test_cone_face_normals_are_the_closed_form_direction():
    """Face 13+i / 23+i: normal ∝ (n_i, ±(√3/2)/c_over_a) whatever the heights and ``face_distance``."""
    p = Pyramid.from_lumice(0.5, 0.6, 1.0, (2, 0, 3), (1, 0, 1), face_distance=(1.0, 1.3, 0.7, 1.9, 1.1, 0.4))
    for sign, numbers, c in ((1, UPPER_PYRAMID_FACES, p.shape.upper_c_over_a), (-1, LOWER_PYRAMID_FACES, p.shape.lower_c_over_a)):
        for i, number in enumerate(numbers):
            if number not in _faces(p):
                continue
            expected = np.append(SIDE_NORMALS_2D[i], sign * np.sqrt(3.0) / 2.0 / c)
            np.testing.assert_allclose(p.normal(p.face(number)), expected / np.linalg.norm(expected), atol=1e-12)


@pytest.mark.parametrize("prism_h, upper_h, lower_h, faces", [
    (0.7, 0.0, 0.0, set(range(1, 9))),                                        # both sides capped: the prism
    (0.7, 1.0, 1.0, set(range(3, 9)) | set(range(13, 19)) | set(range(23, 29))),   # full apex both sides, no basal
    (0.7, 0.0, 1.0, {1, *range(3, 9), *range(23, 29)}),                        # capped above, apex below
    (0.7, 1.7, 0.4, ALL_FACES - {1}),                                          # >= 1 is the same full apex
    (0.0, 1.0, 0.0, {2, *range(13, 19)}),                                      # single hexagonal pyramid on a base
    (0.0, 0.5, 0.0, {1, 2, *range(13, 19)}),                                   # a single frustum
    (0.0, 1.0, 1.0, set(range(13, 19)) | set(range(23, 29))),                  # bipyramid
    (0.0, 0.3, 0.6, {1, 2, *range(13, 19), *range(23, 29)}),                   # two frusta joined at z = 0
    (0.7, -0.5, -0.3, ALL_FACES),                                              # negative heights fold
    (0.7, 5e-6, 0.3, ALL_FACES - set(range(13, 19))),                          # at or below 1e-5: no cone
])
def test_legality_boundaries(prism_h, upper_h, lower_h, faces):
    p = Pyramid.from_lumice(prism_h, upper_h, lower_h)
    assert _faces(p) == faces
    _assert_closed_convex(p)


def test_prism_band_of_a_capped_crystal_is_the_prism():
    p, q = Pyramid.from_lumice(0.7, face_distance=(1.0, 1.3, 0.7, 1.9, 1.1, 0.4)), \
        HexPrism.from_lumice(0.7, (1.0, 1.3, 0.7, 1.9, 1.1, 0.4))
    assert _faces(p) == {f.number for f in q.faces}
    for f in q.faces:
        np.testing.assert_array_equal(p.normal(p.face(f.number)), q.normal(f))
        assert {tuple(v) for v in p.face_vertices(p.face(f.number))} == {tuple(v) for v in q.face_vertices(f)}


def test_zero_volume_and_absent_faces():
    with pytest.raises(ValueError, match="zero volume"):
        Pyramid.from_lumice(0.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="zero volume"):
        Pyramid.from_lumice(0.0, 0.5, 0.5, (0, 0, 1), (0, 0, 1))     # h == 0 indices: no cones either
    with pytest.raises(ValueError, match="no area"):
        Pyramid.from_lumice(0.5, 0.5, 0.5, face_distance=(1, -1, 1, -1, 1, -1))
    p = Pyramid.from_lumice(0.5, 0.0, 0.5)
    with pytest.raises(KeyError):
        p.face(13)
    q = Pyramid.from_lumice(0.5, 0.5, 0.5, upper_wedge_deg=90.0, lower_wedge_deg=28.0)   # out of range: no cone
    assert _faces(q) == ALL_FACES - set(range(13, 19))


def test_apex_snap_band():
    """A truncation within ``APEX_SNAP_REL`` of the apex becomes the apex (Lumice ``ApexCollapsedAt``): no basal face."""
    fd = (1, 1.2, 1, 1.2, 1, 1.2)
    assert 1 in _faces(Pyramid.from_lumice(0.5, 0.999, 0.5, face_distance=fd))
    assert 1 not in _faces(Pyramid.from_lumice(0.5, 0.999999, 0.5, face_distance=fd))


@pytest.mark.parametrize("seed", range(3))
def test_random_shapes_are_closed_convex_solids(seed):
    rng = np.random.default_rng(100 + seed)
    built = 0
    while built < 40:
        fd = tuple(rng.uniform(-0.3, 2.0, 6))
        heights = [float(rng.choice([0.0, rng.uniform(0.05, 0.95), 1.0])) for _ in range(2)]
        try:
            p = Pyramid.from_lumice(float(rng.choice([0.0, rng.uniform(0.1, 1.5)])), *heights,
                                    face_distance=fd, upper_wedge_deg=float(rng.uniform(10, 80)),
                                    lower_wedge_deg=float(rng.uniform(10, 80)))
        except ValueError:
            continue
        built += 1
        _assert_closed_convex(p)


def test_from_lumice_transformed_keeps_its_shape():
    p = Pyramid.from_lumice(0.5, 0.3, 1.0, face_distance=(1, 1.2, 1, 1.2, 1, 1.2))
    for r in (p.transformed(translation=[1, 2, 3]), p.mirrored(p.face(13))):
        assert isinstance(r, Pyramid) and r.shape == p.shape and (r.c_over_a, r.tip_ratio) == (None, None)
