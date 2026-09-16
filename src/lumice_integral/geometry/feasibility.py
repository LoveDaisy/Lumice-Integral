"""光路的几何 + 光学可行性判定（作者 2022 年 matlab 版「计算机遍历-搜索-优化」的 python 复活版）。

Provenance：迁移自写作仓 ``halo_notes.geometry.feasibility``（2026-09-16），算法与公开名称 1:1 保留。
下文的「ch2」指写作系列第 2 章（2022，matlab 已失）——它把「是否存在一条直线贯穿展开走廊里所有多边形的内部」
写成 ``fmincon`` 内点法优化；这里换成**方向扫描 + 精确凸多边形投影求交**：固定内部直线方向 $\\mathbf d$，把 :func:`unfold_faces` 给出的
走廊多边形 $P_1\\ldots P_N$ 沿 $\\mathbf d$ 正交投影到垂直于 $\\mathbf d$ 的平面，逐个 Sutherland–Hodgman 裁剪
求交——交集有正面积 ⇔ 存在方向为 $\\mathbf d$ 的直线穿过每个 $P_i$ 内部（几何约束）。在球面经纬网格上扫一遍
$\\mathbf d$ 就得到整个**可行方向集**（它是方向映射 $\\Phi_P$ 的定义域），而不只是一个可行点。

光学约束用临界角判据（与写作系列 signature 分类的 $\\Phi$ 定义域同源）：
入射 $-\\mathbf d\\cdot\\mathbf n_a \\ge \\cos\\theta_c$、出射 $\\mathbf d\\cdot\\tilde{\\mathbf n}_b \\ge \\cos\\theta_c$，
$\\theta_c = \\arcsin(1/n)$，$n$ = :data:`lumice_integral.geometry.core.N_ICE`。

两种掩码（``typing.NewType`` 区分，DFS 剪枝只能用前者）：

- :func:`corridor_mask` → :data:`CorridorMask`：几何走廊 ∧ 入射约束，**不含出射约束**——末面在枚举里还会继续
  延伸成内反射面，出射约束只对「就在这里出射」的候选有意义；
- :func:`admissible_directions` → :data:`AdmissibleMask`：再叠加出射约束，是「以 ``faces`` 为一条完整光路」的
  可行方向集，附球面面积与「最大入射角最小者」（ch2 的优化目标）。

走廊多边形取法：``faces = [a, m_1..m_k, b]`` 展开后走廊 = ``bodies[0]`` 的面 ``a``、``bodies[j]`` 的面 ``m_j``、
``bodies[-1]`` 的面 ``b``（镜面上的点是镜像不动点，``bodies[j]`` 的面 ``m_j`` 与 ``bodies[j-1]`` 的同号面点集相同，
取哪个只影响法向朝向，不影响投影）。

所有裁剪都对**一批方向**向量化：多边形数组形状 ``(N, V, 2)``、每行有效顶点数 ``(N,)``，逐条边裁剪后按稳定
argsort 压缩——纯 python 逐方向裁剪在 ``max_len=8`` 的枚举里撑不住。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NewType, Sequence

import numpy as np

from .core import N_ICE, Polyhedron, Vec3
from .unfold import unfold_faces

CorridorMask = NewType("CorridorMask", np.ndarray)
"""几何走廊 ∧ 入射光学约束的方向掩码（``(N,) bool``，与网格同序）。DFS 剪枝维护的就是这个语义——但出于性能考虑
（本函数对应的 :func:`corridor_mask` 每次调用都从头展开+投影整条 ``faces``），DFS
（:mod:`lumice_integral.geometry.enumerate`）不逐节点调用 :func:`corridor_mask`，而是用同一套裁剪原语增量维护等价状态，
两者逐位一致由 :func:`~lumice_integral.geometry.enumerate.corridor_index_for_path` 与对应回归测试机械核验，见该函数
docstring。"""

AdmissibleMask = NewType("AdmissibleMask", np.ndarray)
"""在 :data:`CorridorMask` 之上再叠加出射光学约束的掩码：以 ``faces`` 为完整光路的可行方向集。"""

COS_CRITICAL = math.sqrt(1.0 - 1.0 / (N_ICE * N_ICE))
"""$\\cos\\theta_c$，$\\theta_c = \\arcsin(1/n) \\approx 49.8°$：内部方向与界面法向夹角不超过它才能折射出去。"""

EPS_REL = 1e-6
"""走廊交集面积阈值的默认相对系数：``eps = EPS_REL × (晶体最短棱长)²``，与晶体尺度无关。"""


# ---- 球面经纬网格 -------------------------------------------------------------------


@dataclass(frozen=True)
class LatLonGrid:
    """S² 上的经纬网格：极角 θ ∈ (0, π) 均分 ``n_theta`` 格、方位角 φ ∈ [0, 2π) 均分 ``n_phi`` 格，取格中心为
    方向样本，立体角权重 ``sinθ·Δθ·Δφ``。平铺下标 ``k = i * n_phi + j``（``i`` 为 θ 格、``j`` 为 φ 格）。

    ``n_phi`` 取 6 的倍数时网格在 D6h（六棱柱点群）下精确不变，对称等价的光路给出逐位相同的掩码。
    """

    n_theta: int
    n_phi: int

    def __post_init__(self) -> None:
        if self.n_theta < 1 or self.n_phi < 1:
            raise ValueError("grid must have at least one cell in each direction")

    @property
    def size(self) -> int:
        return self.n_theta * self.n_phi

    @property
    def d_theta(self) -> float:
        return math.pi / self.n_theta

    @property
    def d_phi(self) -> float:
        return 2.0 * math.pi / self.n_phi

    def theta(self, i: np.ndarray | int) -> np.ndarray:
        return (np.asarray(i, dtype=float) + 0.5) * self.d_theta

    def phi(self, j: np.ndarray | int) -> np.ndarray:
        return (np.asarray(j, dtype=float) + 0.5) * self.d_phi

    def directions_at(self, flat_index: np.ndarray) -> np.ndarray:
        """给定平铺下标的方向单位向量 ``(M, 3)``。"""
        k = np.asarray(flat_index, dtype=int)
        th, ph = self.theta(k // self.n_phi), self.phi(k % self.n_phi)
        return np.column_stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])

    def weights_at(self, flat_index: np.ndarray) -> np.ndarray:
        k = np.asarray(flat_index, dtype=int)
        return np.sin(self.theta(k // self.n_phi)) * self.d_theta * self.d_phi

    @property
    def directions(self) -> np.ndarray:
        """全部格中心方向 ``(N, 3)``。"""
        return self.directions_at(np.arange(self.size))

    @property
    def weights(self) -> np.ndarray:
        """全部格的立体角权重 ``(N,)``，总和 ≈ 4π。"""
        return self.weights_at(np.arange(self.size))

    def refined(self, levels: int = 1) -> "LatLonGrid":
        """每个格切成 ``2×2``、共 ``levels`` 级后的细网格（粗格 ``(i, j)`` 的子格为 ``(2i..2i+1, 2j..2j+1)``）。"""
        f = 2 ** levels
        return LatLonGrid(self.n_theta * f, self.n_phi * f)

    def boundary_cells(self, mask: np.ndarray) -> np.ndarray:
        """``mask``（平铺 ``(N,) bool``）里与某个四邻格取值不同的格的平铺下标（φ 方向循环、θ 方向不循环）。"""
        m = np.asarray(mask, dtype=bool).reshape(self.n_theta, self.n_phi)
        diff = np.zeros_like(m)
        diff[:, :] |= m != np.roll(m, 1, axis=1)
        diff[:, :] |= m != np.roll(m, -1, axis=1)
        diff[1:, :] |= m[1:] != m[:-1]
        diff[:-1, :] |= m[:-1] != m[1:]
        return np.flatnonzero(diff.ravel())

    def upsample(self, mask: np.ndarray) -> np.ndarray:
        """把平铺掩码按 ``refined(1)`` 的格序展开（每格复制成 2×2 子格）。"""
        m = np.asarray(mask).reshape(self.n_theta, self.n_phi)
        return np.repeat(np.repeat(m, 2, axis=0), 2, axis=1).ravel()


def perp_bases(directions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """:func:`lumice_integral.geometry.core.perp_basis` 的批量版：每个方向一对正交基 ``(u, w)``，``(u, w, d̂)`` 右手系，
    选取规则逐位相同（helper 取 +z，``|d_z| ≥ 0.9`` 时改取 +x）。"""
    d = np.asarray(directions, dtype=float)
    d = d / np.linalg.norm(d, axis=1, keepdims=True)
    near_z = np.abs(d[:, 2]) >= 0.9
    helper = np.where(near_z[:, None], np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]))
    u = np.cross(d, helper)
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    w = np.cross(d, u)
    return u, w


# ---- 批量凸多边形裁剪 -----------------------------------------------------------------


@dataclass
class _PolyBatch:
    """一批（每方向一个）二维凸多边形：``points (N, V, 2)``，每行前 ``count[n]`` 个顶点有效、逆时针排列。"""

    points: np.ndarray
    count: np.ndarray

    @property
    def n(self) -> int:
        return self.points.shape[0]

    def area(self) -> np.ndarray:
        """鞋带公式面积 ``(N,)``（顶点数 < 3 的行为 0）。"""
        N, V, _ = self.points.shape
        if V == 0:
            return np.zeros(N)
        idx = np.arange(V)[None, :]
        valid = idx < self.count[:, None]
        nxt = np.where(idx + 1 < self.count[:, None], idx + 1, 0)
        p = self.points
        q = np.take_along_axis(p, nxt[:, :, None], axis=1)
        cross = p[:, :, 0] * q[:, :, 1] - p[:, :, 1] * q[:, :, 0]
        return 0.5 * np.where(valid, cross, 0.0).sum(axis=1)

    def take(self, rows: np.ndarray) -> "_PolyBatch":
        return _PolyBatch(self.points[rows], self.count[rows])


def _project(points3d: np.ndarray, u: np.ndarray, w: np.ndarray) -> np.ndarray:
    """把同一个三维多边形 ``(V, 3)`` 沿每个方向投影：返回 ``(N, V, 2)``，坐标 ``(p·u, p·w)``（投影中心取原点）。"""
    return np.stack([points3d @ u.T, points3d @ w.T], axis=-1).transpose(1, 0, 2)


def _ensure_ccw(poly: np.ndarray) -> np.ndarray:
    """``(N, V, 2)`` 每行按有符号面积翻成逆时针（投影是仿射变换，会按手性翻转环序）。"""
    q = np.roll(poly, -1, axis=1)
    signed = 0.5 * (poly[:, :, 0] * q[:, :, 1] - poly[:, :, 1] * q[:, :, 0]).sum(axis=1)
    out = poly.copy()
    out[signed < 0] = poly[signed < 0, ::-1]
    return out


def _clip_halfplane(batch: _PolyBatch, a: np.ndarray, b: np.ndarray) -> _PolyBatch:
    """Sutherland–Hodgman 的一步：每行用有向边 ``a[n] → b[n]`` 的左侧半平面裁剪该行多边形。"""
    N, V, _ = batch.points.shape
    if V == 0:
        return batch
    p = batch.points
    idx = np.arange(V)[None, :]
    valid = idx < batch.count[:, None]
    nxt = np.where(idx + 1 < batch.count[:, None], idx + 1, 0)
    q = np.take_along_axis(p, nxt[:, :, None], axis=1)
    e = (b - a)[:, None, :]
    rel_p, rel_q = p - a[:, None, :], q - a[:, None, :]
    s_p = e[:, :, 0] * rel_p[:, :, 1] - e[:, :, 1] * rel_p[:, :, 0]
    s_q = e[:, :, 0] * rel_q[:, :, 1] - e[:, :, 1] * rel_q[:, :, 0]
    in_p, in_q = s_p >= 0, s_q >= 0
    keep_p = valid & in_p
    crossing = valid & (in_p != in_q)
    denom = np.where(crossing, s_p - s_q, 1.0)
    t = np.where(crossing, s_p / denom, 0.0)
    x = p + t[:, :, None] * (q - p)
    # 输出槽位 2i = 顶点 i（若在内侧），2i+1 = 边 i→i+1 与裁剪线的交点（若跨越）；顺序即保持环序
    out = np.empty((N, 2 * V, 2))
    out[:, 0::2], out[:, 1::2] = p, x
    out_valid = np.empty((N, 2 * V), dtype=bool)
    out_valid[:, 0::2], out_valid[:, 1::2] = keep_p, crossing
    order = np.argsort(~out_valid, axis=1, kind="stable")
    count = out_valid.sum(axis=1)
    width = int(count.max()) if N else 0
    order = order[:, :width]
    return _PolyBatch(np.take_along_axis(out, order[:, :, None], axis=1), count)


def _clip_by_polygon(batch: _PolyBatch, clip: np.ndarray) -> _PolyBatch:
    """用每行的凸裁剪多边形 ``clip (N, M, 2)``（逆时针）依次裁剪 ``batch``。"""
    M = clip.shape[1]
    for i in range(M):
        batch = _clip_halfplane(batch, clip[:, i], clip[:, (i + 1) % M])
    return batch


def _initial_batch(poly: np.ndarray) -> _PolyBatch:
    """把逆时针的 ``(N, V, 2)`` 投影多边形包成裁剪起点。"""
    N, V, _ = poly.shape
    return _PolyBatch(poly.copy(), np.full(N, V, dtype=int))


# ---- 走廊与判据 --------------------------------------------------------------------


def corridor_polygons(crystal: Polyhedron, faces: Sequence[int]) -> tuple[list[np.ndarray], Vec3]:
    """展开走廊的多边形序列（各 ``(V_i, 3)``，展开空间世界坐标）与展开后的出射面外法向 ``ñ_b``。"""
    bodies, n_tilde_b = unfold_faces(crystal, faces)
    polys = [bodies[0].face_vertices(bodies[0].face(faces[0]))]
    polys += [bodies[j].face_vertices(bodies[j].face(faces[j])) for j in range(1, len(faces) - 1)]
    polys.append(bodies[-1].face_vertices(bodies[-1].face(faces[-1])))
    return polys, n_tilde_b


def min_edge_length(crystal: Polyhedron) -> float:
    v = crystal.vertices
    return float(min(np.linalg.norm(v[a] - v[b]) for a, b in crystal.edges))


def area_eps(crystal: Polyhedron, eps_rel: float = EPS_REL) -> float:
    """走廊交集的面积阈值：``eps_rel × 最短棱长²``。"""
    return eps_rel * min_edge_length(crystal) ** 2


def corridor_intersection(polys: Sequence[np.ndarray], directions: np.ndarray) -> _PolyBatch:
    """沿每个方向把走廊多边形投影并归约求交，返回每方向的交集多边形（面积由 :meth:`_PolyBatch.area` 给）。"""
    u, w = perp_bases(directions)
    batch = _initial_batch(_ensure_ccw(_project(polys[0], u, w)))
    for poly in polys[1:]:
        batch = _clip_by_polygon(batch, _ensure_ccw(_project(poly, u, w)))
    return batch


def geometric_ok(polys: Sequence[np.ndarray], directions: np.ndarray, eps: float) -> np.ndarray:
    """几何约束：沿 ``directions`` 各方向走廊投影交集面积 > ``eps``，返回 ``(N,) bool``。"""
    d = np.asarray(directions, dtype=float).reshape(-1, 3)
    if len(d) == 0:
        return np.zeros(0, dtype=bool)
    return corridor_intersection(polys, d).area() > eps


def entry_ok(n_a: Vec3, directions: np.ndarray, cos_tc: float = COS_CRITICAL) -> np.ndarray:
    """入射光学约束 $-\\mathbf d\\cdot\\mathbf n_a \\ge \\cos\\theta_c$。"""
    return -(np.asarray(directions) @ n_a) >= cos_tc


def exit_ok(n_tilde_b: Vec3, directions: np.ndarray, cos_tc: float = COS_CRITICAL) -> np.ndarray:
    """出射光学约束 $\\mathbf d\\cdot\\tilde{\\mathbf n}_b \\ge \\cos\\theta_c$。"""
    return (np.asarray(directions) @ n_tilde_b) >= cos_tc


def incidence_objective_deg(n_a: Vec3, n_tilde_b: Vec3, directions: np.ndarray) -> np.ndarray:
    """ch2 优化目标：首尾两面上入（出）射角的较大者，``max(arccos(-d·n_a), arccos(d·ñ_b))``（度）。"""
    d = np.asarray(directions)
    c = np.minimum(-(d @ n_a), d @ n_tilde_b)
    return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))


def corridor_mask(crystal: Polyhedron, faces: Sequence[int], grid: LatLonGrid, parent_mask: np.ndarray | None = None,
                  eps: float | None = None) -> CorridorMask:
    """几何走廊 ∧ 入射约束的方向掩码。``parent_mask`` 给定时只在其为 ``True`` 的方向上重算（前缀走廊只会
    变窄，父掩码为 ``False`` 处子掩码必为 ``False``）。单面序列（外反射）没有走廊：掩码 = 半球 ``-d·n_a > 0``。

    注意：本函数每次调用都从头对整条 ``faces`` 重新展开（:func:`corridor_polygons` 内部重新调用
    :func:`~lumice_integral.geometry.unfold.unfold_faces`）、重新投影裁剪；``parent_mask`` 只缩小候选方向集合，不复用
    父节点已裁剪出的走廊交集多边形，是独立于路径长度的「从头算」入口，适合单条路径核验或非 DFS 场景。
    :mod:`lumice_integral.geometry.enumerate` 的 DFS 为避免这一路径每步都重复展开的 O(depth) 开销，维护自己的增量走廊
    状态（见 :func:`~lumice_integral.geometry.enumerate.corridor_index_for_path`），语义与本函数逐位一致，由测试机械
    核验（``test_enumerate.py::test_dfs_incremental_corridor_matches_corridor_mask``）。"""
    faces = [int(f) for f in faces]
    eps = area_eps(crystal) if eps is None else eps
    n_a = crystal.normal(crystal.face(faces[0]))
    mask = np.zeros(grid.size, dtype=bool)
    cand = np.arange(grid.size) if parent_mask is None else np.flatnonzero(parent_mask)
    d = grid.directions_at(cand)
    if len(faces) == 1:
        mask[cand] = -(d @ n_a) > 0
        return CorridorMask(mask)
    ok = entry_ok(n_a, d)
    if ok.any():
        polys, _ = corridor_polygons(crystal, faces)
        ok[ok] = geometric_ok(polys, d[ok], eps)
    mask[cand] = ok
    return CorridorMask(mask)


@dataclass(frozen=True)
class AdmissibleResult:
    """:func:`admissible_directions` 的结果。

    - ``mask``：粗网格上的可行方向掩码；
    - ``area_sr``：可行方向集的球面面积（边界细化 ``refine_levels`` 级后的估计，单位球面度）；
    - ``min_incidence_deg``：几何走廊上「首尾两面入（出）射角较大者」的最小值（ch2 的优化目标；局部细化后）；
      ``< 49.8°`` ⇔ 光学可行。单面序列为 0；
    - ``argmin_direction``：取到该最小值的内部方向。
    """

    mask: AdmissibleMask
    area_sr: float
    min_incidence_deg: float
    argmin_direction: Vec3


def _refine_area(grid: LatLonGrid, mask: np.ndarray, evaluate, levels: int) -> float:
    """边界自适应细化的面积估计：每级把掩码上采样一倍，只在可行 / 不可行翻转处的格重新评估。"""
    fine, g = np.asarray(mask, dtype=bool), grid
    for _ in range(levels):
        fine, g = g.upsample(fine), g.refined(1)
        idx = g.boundary_cells(fine)
        if len(idx):
            fine[idx] = evaluate(g.directions_at(idx))
    return float((fine * g.weights).sum())


def _refine_min(grid: LatLonGrid, geom_mask: np.ndarray, objective, geometric, levels: int) -> tuple[float, Vec3]:
    """在几何可行格上取目标最小者，再在「接近最小值的全部候选格」上做 2×2 局部细化 ``levels`` 级（不是只沿
    全局 argmin 细化：可行窗口很窄时粗网格 argmin 可能落在错误的局部极值旁）。返回 ``(最小值, argmin 方向)``。"""
    cand = np.flatnonzero(geom_mask)
    if len(cand) == 0:
        return math.inf, np.full(3, np.nan)
    g = grid
    dirs = g.directions_at(cand)
    f = objective(dirs)
    best, best_dir = float(f.min()), dirs[int(f.argmin())]
    margin = 2.0 * math.degrees(max(g.d_theta, g.d_phi))
    for _ in range(levels):
        keep = cand[f <= best + margin]
        i, j = keep // g.n_phi, keep % g.n_phi
        gi = np.concatenate([2 * i, 2 * i + 1, 2 * i, 2 * i + 1])
        gj = np.concatenate([2 * j, 2 * j, 2 * j + 1, 2 * j + 1])
        g = g.refined(1)
        cand = gi * g.n_phi + gj
        dirs = g.directions_at(cand)
        ok = geometric(dirs)
        cand, dirs = cand[ok], dirs[ok]
        if len(cand) == 0:
            break
        f = objective(dirs)
        k = int(f.argmin())
        if f[k] < best:
            best, best_dir = float(f[k]), dirs[k]
        margin /= 2.0
    return best, best_dir


def admissible_directions(crystal: Polyhedron, faces: Sequence[int], grid: LatLonGrid,
                          corridor: CorridorMask | None = None, eps: float | None = None,
                          refine_levels: int = 2) -> AdmissibleResult:
    """以 ``faces`` 为一条完整光路的可行方向集（几何走廊 ∧ 入射约束 ∧ 出射约束）。

    ``corridor`` 给定时（枚举器路径）在它之上叠加出射约束，``min_incidence_deg`` 也只在它上面取——当结果非空时
    这与在全部几何可行方向上取一致（最小值必落在两个光学约束都满足的区域内），结果为空时它只是上界。
    不给时从头算全网格的几何掩码，``min_incidence_deg`` 是 ch2 意义上的精确定义。单面序列（外反射）：可行集 =
    半球，面积 2π，``min_incidence_deg = 0``。"""
    faces = [int(f) for f in faces]
    eps = area_eps(crystal) if eps is None else eps
    n_a = crystal.normal(crystal.face(faces[0]))
    if len(faces) == 1:
        mask = corridor_mask(crystal, faces, grid, corridor, eps)
        return AdmissibleResult(AdmissibleMask(mask), 2.0 * math.pi, 0.0, -n_a)

    polys, n_tilde_b = corridor_polygons(crystal, faces)

    def geometric(d: np.ndarray) -> np.ndarray:
        return geometric_ok(polys, d, eps)

    def objective(d: np.ndarray) -> np.ndarray:
        return incidence_objective_deg(n_a, n_tilde_b, d)

    def evaluate(d: np.ndarray) -> np.ndarray:
        ok = entry_ok(n_a, d) & exit_ok(n_tilde_b, d)
        if ok.any():
            ok[ok] = geometric(d[ok])
        return ok

    if corridor is None:
        geom = geometric(grid.directions)
        corridor = CorridorMask(geom & entry_ok(n_a, grid.directions))
    else:
        geom = np.asarray(corridor, dtype=bool)
    mask = np.asarray(corridor, dtype=bool) & exit_ok(n_tilde_b, grid.directions)
    area = _refine_area(grid, mask, evaluate, refine_levels) if mask.any() else 0.0
    best, best_dir = _refine_min(grid, geom, objective, geometric, refine_levels)
    return AdmissibleResult(AdmissibleMask(mask), area, best, best_dir)


def is_feasible(crystal: Polyhedron, faces: Sequence[int], grid: LatLonGrid, eps: float | None = None) -> bool:
    """几何 + 光学都可行（可行方向集非空）。"""
    return bool(admissible_directions(crystal, faces, grid, eps=eps, refine_levels=0).mask.any())


def entry_points(crystal: Polyhedron, faces: Sequence[int], directions: np.ndarray) -> np.ndarray:
    """每个内部方向对应的一个具体入射点 ``(N, 3)``：走廊投影交集多边形的质心沿该方向抬回入射面 ``a`` 所在平面。
    从这一点沿该方向出发的直线穿过走廊每个多边形的内部（交集非空时），是「可行」的构造性见证，可直接喂给
    独立的射线追迹（``tests/test_geometry_feasibility.py`` 里基于 :meth:`Polyhedron.intersect_ray` 的 oracle）复核。
    交集为空的方向给 ``NaN``。"""
    faces = [int(f) for f in faces]
    d = np.asarray(directions, dtype=float).reshape(-1, 3)
    polys, _ = corridor_polygons(crystal, faces)
    batch = corridor_intersection(polys, d)
    N, V, _ = batch.points.shape
    valid = np.arange(V)[None, :] < batch.count[:, None]
    cnt = np.maximum(batch.count, 1)
    centroid = np.where(valid[:, :, None], batch.points, 0.0).sum(axis=1) / cnt[:, None]
    u, w = perp_bases(d)
    base = centroid[:, :1] * u + centroid[:, 1:] * w          # 垂直于 d 的平面上的点
    n_a = crystal.normal(crystal.face(faces[0]))
    p0 = crystal.face_vertices(crystal.face(faces[0]))[0]
    t = ((p0 - base) @ n_a) / (d @ n_a)
    out = base + t[:, None] * d
    out[batch.count < 3] = np.nan
    return out


def external_directions(crystal: Polyhedron, faces: Sequence[int], directions: np.ndarray,
                        n_ice: float = N_ICE) -> np.ndarray:
    """内部方向 → 晶体外入射方向（逆 Snell）：切向分量乘以 ``n_ice``，法向分量按单位长反解、指向晶体内
    （``d_ext·n_a < 0``）。内部方向超临界角（``n·sinθ_int > 1``）的行为 ``NaN``。单面序列（外反射）原样返回。"""
    faces = [int(f) for f in faces]
    d = np.asarray(directions, dtype=float).reshape(-1, 3)
    if len(faces) == 1:
        return d.copy()
    n_a = crystal.normal(crystal.face(faces[0]))
    normal_part = d @ n_a
    tangent = (d - normal_part[:, None] * n_a[None, :]) * n_ice
    t2 = (tangent ** 2).sum(axis=1)
    cos_ext = np.sqrt(np.clip(1.0 - t2, 0.0, None))
    out = tangent - cos_ext[:, None] * n_a[None, :]
    out[(t2 > 1.0) | (normal_part >= 0)] = np.nan
    return out


__all__ = ["AdmissibleMask", "AdmissibleResult", "COS_CRITICAL", "CorridorMask", "EPS_REL", "LatLonGrid",
           "admissible_directions", "area_eps", "corridor_intersection", "corridor_mask", "corridor_polygons",
           "entry_ok", "entry_points", "exit_ok", "external_directions", "geometric_ok", "incidence_objective_deg",
           "is_feasible", "min_edge_length", "perp_bases"]
