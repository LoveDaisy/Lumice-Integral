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
