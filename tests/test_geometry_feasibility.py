import math

import numpy as np
import pytest

from lumice_integral.geometry import HexPrism, N_ICE, perp_basis
from lumice_integral.geometry import feasibility as fz
from lumice_integral.geometry.feasibility import (COS_CRITICAL, LatLonGrid, admissible_directions, area_eps,
                                                  corridor_mask, corridor_polygons, entry_points, external_directions,
                                                  is_feasible, perp_bases)

from _geometry_oracles import refract, trace_faces

GRID = LatLonGrid(90, 180)      # 2°，n_phi 是 6 的倍数：D6h 下精确不变
PLATE = HexPrism.from_ratio(0.2)


def _batch(*polys2d):
    """把若干二维多边形（每个 (V, 2)）当成"每方向一个"的一批，逐个裁剪求交，返回各行面积。"""
    batch = fz._initial_batch(fz._ensure_ccw(np.asarray(polys2d[0], float)[None]))
    for p in polys2d[1:]:
        batch = fz._clip_by_polygon(batch, fz._ensure_ccw(np.asarray(p, float)[None]))
    return batch.area()[0], batch.count[0]


SQUARE = [(0, 0), (1, 0), (1, 1), (0, 1)]


# ---- Step 1：批量 Sutherland–Hodgman 的红绿对 ----------------------------------------------


def test_clip_disjoint_squares_is_empty():
    area, cnt = _batch(SQUARE, [(2, 0), (3, 0), (3, 1), (2, 1)])
    assert area == 0.0 and cnt < 3


def test_clip_overlapping_squares_area_is_analytic():
    area, _ = _batch(SQUARE, [(0.5, 0.5), (1.5, 0.5), (1.5, 1.5), (0.5, 1.5)])
    assert area == pytest.approx(0.25)


def test_clip_triangle_inside_square_returns_triangle():
    tri = [(0.2, 0.2), (0.8, 0.2), (0.5, 0.7)]
    area, cnt = _batch(SQUARE, tri)
    assert cnt == 3 and area == pytest.approx(0.5 * 0.6 * 0.5)
    # 交换裁剪顺序结果相同
    assert _batch(tri, SQUARE)[0] == pytest.approx(area)


def test_clip_edge_touching_squares_has_zero_area():
    """只共一条边、内部不相交：交集退化为线段，面积 0，按 eps 判不可行。"""
    area, _ = _batch(SQUARE, [(1, 0), (2, 0), (2, 1), (1, 1)])
    assert area == pytest.approx(0.0, abs=1e-15)


def test_clockwise_input_is_flipped_to_ccw():
    cw = np.asarray(SQUARE[::-1], float)[None]
    ccw = fz._ensure_ccw(cw)
    q = np.roll(ccw, -1, axis=1)
    signed = 0.5 * (ccw[..., 0] * q[..., 1] - ccw[..., 1] * q[..., 0]).sum(axis=1)
    assert signed[0] == pytest.approx(1.0)
    # 顺时针输入不翻转就会把交集裁空
    assert _batch(SQUARE[::-1], SQUARE)[0] == pytest.approx(1.0)


def test_batch_rows_are_independent():
    """同一批里不同行各自裁剪：一行相交、一行不相交。"""
    subj = fz._initial_batch(np.asarray([SQUARE, SQUARE], float))
    clip = np.asarray([[(0.5, 0.5), (1.5, 0.5), (1.5, 1.5), (0.5, 1.5)], [(2, 0), (3, 0), (3, 1), (2, 1)]], float)
    out = fz._clip_by_polygon(subj, clip)
    assert out.area() == pytest.approx([0.25, 0.0])


# ---- 网格与基 -------------------------------------------------------------------------


def test_grid_weights_cover_sphere():
    assert GRID.weights.sum() == pytest.approx(4 * math.pi, rel=1e-3)
    assert np.allclose(np.linalg.norm(GRID.directions, axis=1), 1.0)


def test_grid_refinement_nests_cells():
    fine = GRID.refined(1)
    up = GRID.upsample(np.arange(GRID.size))
    # 子格 (2i+di, 2j+dj) 的父格是 (i, j)
    k = np.arange(fine.size)
    parent = (k // fine.n_phi // 2) * GRID.n_phi + (k % fine.n_phi) // 2
    assert np.array_equal(up, parent)
    assert fine.weights.sum() == pytest.approx(4 * math.pi, rel=1e-3)


def test_perp_bases_matches_scalar_perp_basis():
    d = GRID.directions[::911]
    u, w = perp_bases(d)
    for k in range(len(d)):
        us, ws = perp_basis(d[k])
        assert np.allclose(u[k], us) and np.allclose(w[k], ws)


# ---- Step 2：ch2 两个算例 ------------------------------------------------------------------


def test_1_3_2_feasible_with_near_normal_incidence():
    """ch2：1-3-2 几何 + 光学都可行，优化器给出 cosθ = 0.998（入射角 ≈ 3.6°）：贴着底面棱边近法向入射。
    本实现的可行集是 -z 附近半个锥帽（面积 ≈ 1.12 sr），最小入射角随网格细化趋于 0（只受 eps 限制）——3.6° 是
    ch2 内点法障碍项把点挡在多边形边界内侧的产物，是最小值的一个上界，不是最小值本身。"""
    res = admissible_directions(PLATE, (1, 3, 2), GRID)
    assert res.mask.any() and is_feasible(PLATE, (1, 3, 2), GRID)
    assert res.min_incidence_deg <= 3.6 + 1.0
    assert np.allclose(res.argmin_direction, [0, 0, -1], atol=0.05)
    assert 1.0 < res.area_sr < 1.3


def test_1_3_4_geometric_ok_but_only_a_sliver_under_critical_angle():
    """ch2：1-3-4 几何可行、最小入射角 74.5° > 49.8°，光学不可行。本实现：几何走廊非空；「首尾入（出）射角较大者」
    的最小值 ≈ 49.3°，刚好压在临界角之下——对应贴着面 3 与面 4 公共棱、以 ~89.8° 掠射面 3 的光线（trace 白盒可复现，
    见 test_entry_points_are_traceable_witnesses），可行方向集只是一条 < 0.001 sr 的细缝。ch2 的 74.5° 同样是障碍项
    把解挡在棱边之外得到的局部最优。"""
    cm = corridor_mask(PLATE, (1, 3, 4), GRID)
    assert cm.any()
    res = admissible_directions(PLATE, (1, 3, 4), GRID)
    assert 49.0 < res.min_incidence_deg < math.degrees(math.acos(COS_CRITICAL))
    assert 0 < res.area_sr < 1e-3


def test_optical_only_rejection_3_4():
    """3-4：相邻侧面，几何上一条直线能穿过两面，但入射 / 出射临界角锥相距 120° > 2θc，光学不可行。"""
    assert corridor_mask(PLATE, (3, 4), GRID).any()
    assert not is_feasible(PLATE, (3, 4), GRID)
    assert admissible_directions(PLATE, (3, 4), GRID).min_incidence_deg == pytest.approx(60.0, abs=0.5)


def test_thin_plate_rejects_1_3_5_but_column_accepts():
    """1-3-5 需要从顶面下到侧面 3 再走到侧面 5：片晶太薄走不到，柱晶可以（h/a 依赖是几何约束的核心）。"""
    assert not is_feasible(PLATE, (1, 3, 5), GRID)
    assert is_feasible(HexPrism.from_ratio(2.0), (1, 3, 5), GRID)


def test_single_face_external_reflection():
    for a in (1, 3):
        res = admissible_directions(PLATE, (a,), GRID)
        n_a = PLATE.normal(PLATE.face(a))
        assert res.area_sr == pytest.approx(2 * math.pi)
        assert res.min_incidence_deg == 0.0
        assert np.array_equal(res.mask, -(GRID.directions @ n_a) > 0)
        assert is_feasible(PLATE, (a,), GRID)


def test_corridor_mask_parent_restriction_is_consistent():
    """给 parent_mask 只在其内重算，结果与全量重算在该范围内逐位相等、范围外恒 False。"""
    parent = corridor_mask(PLATE, (3, 1), GRID)
    full = corridor_mask(PLATE, (3, 1, 5), GRID)
    inc = corridor_mask(PLATE, (3, 1, 5), GRID, parent_mask=parent)
    assert np.array_equal(inc, full & parent)
    assert not inc[~parent].any()


def test_corridor_polygons_shape_and_exit_normal():
    polys, n_tilde_b = corridor_polygons(PLATE, (1, 3, 2))
    assert [len(p) for p in polys] == [6, 4, 6]
    # 镜面上的点是不动点：点集相同，只是幽灵的顶点环反了向
    assert np.allclose(polys[1][::-1], PLATE.face_vertices(PLATE.face(3)))
    assert np.allclose(n_tilde_b, [0, 0, -1])


# ---- 面积收敛与 min_incidence 细化 ---------------------------------------------------------


@pytest.mark.parametrize("faces", [(1, 3, 2), (3, 1, 5), (3, 5, 8), (1, 3)])
def test_area_converges_under_grid_refinement(faces):
    """网格加密一级（2° → 1°）后可行方向集面积变化 < 5%。"""
    coarse = admissible_directions(PLATE, faces, GRID, refine_levels=1).area_sr
    fine = admissible_directions(PLATE, faces, GRID.refined(1), refine_levels=1).area_sr
    assert abs(fine - coarse) / fine < 0.05


def test_min_incidence_refinement_improves_on_coarse_grid():
    """3-5（对称平行光路）：目标 max(∠(-d, n_3), ∠(d, -n_3)) 的真值 30°（入 / 出射面夹角 60° 的一半），局部细化
    后误差应小于粗格半径。"""
    coarse = admissible_directions(PLATE, (3, 5), GRID, refine_levels=0).min_incidence_deg
    fine = admissible_directions(PLATE, (3, 5), GRID, refine_levels=3).min_incidence_deg
    assert abs(fine - 30.0) <= abs(coarse - 30.0) + 1e-9
    assert abs(fine - 30.0) < 0.3


# ---- 逆 Snell 与构造性见证 ------------------------------------------------------------------


@pytest.mark.parametrize("faces", [(1, 3, 2), (3, 1, 5), (3, 5, 8)])
def test_external_directions_roundtrip_snell(faces):
    res = admissible_directions(PLATE, faces, GRID, refine_levels=0)
    d = GRID.directions[res.mask][::7]
    ext = external_directions(PLATE, faces, d)
    n_a = PLATE.normal(PLATE.face(faces[0]))
    assert np.isfinite(ext).all()
    assert np.allclose(np.linalg.norm(ext, axis=1), 1.0)
    assert (ext @ n_a < 0).all()
    for k in range(len(d)):
        assert np.allclose(refract(ext[k], n_a, 1.0, N_ICE), d[k], atol=1e-9)


def test_external_directions_nan_beyond_critical_angle():
    n_a = PLATE.normal(PLATE.face(1))
    d = np.array([[0.0, 0.0, -1.0], [math.sin(math.radians(60)), 0.0, -math.cos(math.radians(60))]])
    ext = external_directions(PLATE, (1, 2), d)
    assert np.allclose(ext[0], -n_a) and np.isnan(ext[1]).all()


@pytest.mark.parametrize("faces", [(1, 3, 2), (1, 3, 4), (3, 4, 5), (3, 1, 5), (1, 2, 3), (3, 6, 3)])
def test_entry_points_are_traceable_witnesses(faces):
    """每个可行方向给出的入射点见证，喂给独立实现的 :func:`trace_faces` 后面序列必须恰好等于 ``faces``。"""
    res = admissible_directions(PLATE, faces, GRID, refine_levels=0)
    idx = np.flatnonzero(res.mask)
    assert len(idx) > 0
    d = GRID.directions_at(idx)
    p, ext = entry_points(PLATE, faces, d), external_directions(PLATE, faces, d)
    for k in range(len(idx)):
        assert trace_faces(PLATE, p[k] - 3 * ext[k], ext[k], len(faces)) == list(faces)


def test_eps_is_relative_to_crystal_scale():
    assert area_eps(HexPrism(2.0, 0.4)) == pytest.approx(4 * area_eps(HexPrism(1.0, 0.2)))
