"""按字长枚举几何 + 光学都可行的光路（写作系列 ch2「穷举、规约、剪枝搜索一波流」的 python 复活版）。

Provenance：迁移自写作仓 ``halo_notes.geometry.enumerate``（2026-09-16），算法与公开名称 1:1 保留。

DFS 按面序列延伸（相邻面不重复，入射面 / 出射面均可为任意面；单面序列 = 外反射，按 ch2 计入）。每个节点持有
:data:`~lumice_integral.geometry.feasibility.CorridorMask`（几何走廊 ∧ 入射约束）及每个可行方向上的走廊投影交集
多边形；追加一个面时，末面变成镜面、走廊只多出一个多边形（新幽灵上的新面），所以子节点只需把父节点的交集多边形
再裁一次——**单调性**（走廊只会变窄）保证父掩码为空即可剪掉整棵子树。出射光学约束只在「就在这里出射」的候选上
叠加（:func:`~lumice_integral.geometry.feasibility.admissible_directions` 的语义），不参与剪枝。

对称去重：``symmetry_orbit`` 是调用方注入的「面序列 → 对称轨道」回调（六棱柱典型取 D6 × 上下互换，
即 Lumice ``PBD`` 对称；``tests/test_geometry_enumerate.py`` 内置了一份），本包不内置任何点群实现；
一般多面体传 ``None`` 不去重。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import numpy as np

from .core import Polyhedron, Vec3
from .feasibility import (AdmissibleMask, CorridorMask, LatLonGrid, _clip_by_polygon, _ensure_ccw, _initial_batch,
                          _PolyBatch, _project, admissible_directions, area_eps, entry_ok, exit_ok,
                          incidence_objective_deg, perp_bases)

SymmetryOrbit = Callable[[Sequence[int]], Iterable[Sequence[int]]]


@dataclass(frozen=True)
class RaypathRecord:
    """一条可行光路：面序列、可行方向掩码（网格同序）、可行方向集球面面积、「首尾入（出）射角较大者」的最小值。

    ``area_sr``/``min_incidence_deg`` 的精度取决于产出它的 ``enumerate_raypaths[_with_stats]`` 调用传的
    ``refine_levels``（默认 0 = 粗网格估计，未做边界自适应细化）；跨模块复用这两个字段前，先确认调用方是否需要
    显式传 ``refine_levels > 0`` 换取精细值。"""

    faces: tuple[int, ...]
    admissible_mask: AdmissibleMask
    area_sr: float
    min_incidence_deg: float

    @property
    def length(self) -> int:
        return len(self.faces)

    @property
    def label(self) -> str:
        return "-".join(str(f) for f in self.faces)


@dataclass
class EnumerationStats:
    """剪枝统计。``attempted`` = 尝试扩展（父节点 + 一个新面）的次数，``pruned`` = 其中走廊掩码为空、整棵子树被
    剪掉的次数；``prune_rate = pruned / attempted``。``feasible_by_length`` 按字长计几何 + 光学都可行（去重前）的条数，
    ``geometric_by_length`` 计几何可行（走廊掩码非空）的节点数。"""

    attempted: int = 0
    pruned: int = 0
    feasible_by_length: dict[int, int] = field(default_factory=dict)
    geometric_by_length: dict[int, int] = field(default_factory=dict)

    @property
    def prune_rate(self) -> float:
        return self.pruned / self.attempted if self.attempted else 0.0


@dataclass
class _Node:
    """DFS 节点：面序列、当前幽灵晶体、掩码为 True 的方向的平铺下标、这些方向上的走廊投影交集多边形。

    ``index``/``batch`` 内部全程是裸 ``np.ndarray``，不套 :data:`~lumice_integral.geometry.feasibility.CorridorMask`
    ``NewType``——后者只标注 :func:`~lumice_integral.geometry.feasibility.corridor_mask` 等公开函数签名与
    :class:`RaypathRecord` 的边界，不覆盖 DFS 内部剪枝这条热路径；这条路径的正确性由
    ``test_dfs_incremental_corridor_matches_corridor_mask`` 单独兜底。"""

    faces: tuple[int, ...]
    body: Polyhedron
    index: np.ndarray
    batch: _PolyBatch


def _ghost_for(node: _Node) -> Polyhedron:
    """节点末面变镜面后的幽灵晶体（入射节点——``faces`` 只有一个面——原样返回原晶体）。"""
    if len(node.faces) == 1:
        return node.body
    return node.body.mirrored(node.body.face(node.faces[-1]))


def _extend_batch(ghost: Polyhedron, index: np.ndarray, batch: _PolyBatch, f: int, u_all: np.ndarray,
                  w_all: np.ndarray, eps: float) -> tuple[np.ndarray, _PolyBatch] | None:
    """把走廊沿新面 ``f`` 再裁一次：DFS 单步扩展与 :func:`corridor_index_for_path` 共用的唯一实现（a56：
    走廊只会变窄这一单调剪枝逻辑只在这里写一份）。裁剪后交集全空则返回 ``None``（整棵子树可剪）。"""
    u, w = u_all[index], w_all[index]
    face = ghost.face(f)
    poly = _ensure_ccw(_project(ghost.face_vertices(face), u, w))
    new_batch = _clip_by_polygon(batch, poly)
    keep = new_batch.area() > eps
    if not keep.any():
        return None
    return index[keep], new_batch.take(np.flatnonzero(keep))


def corridor_index_for_path(crystal: Polyhedron, faces: Sequence[int], grid: LatLonGrid,
                            eps: float | None = None) -> np.ndarray:
    """按 DFS 增量剪枝用的同一套原语（:func:`_ghost_for` / :func:`_extend_batch`），独立沿给定的**完整**面序列
    走一遍，返回走廊（几何 ∧ 入射约束，:data:`~lumice_integral.geometry.feasibility.CorridorMask` 语义）仍非空的方向
    平铺下标集合。

    与 :func:`~lumice_integral.geometry.feasibility.corridor_mask` 的关系：两者语义相同（同一 ``eps``、同一入射约束），
    但 :func:`~lumice_integral.geometry.feasibility.corridor_mask` 每次调用都从头展开、投影整条 ``faces``（O(depth) 每
    调用），本函数复用 DFS 沿路径增量裁剪的批状态（O(1) 每步）——是 DFS 实际剪枝路径为何不直接调用
    ``corridor_mask`` 的具体原因。两者逐位一致由 ``test_enumerate.py::test_dfs_incremental_corridor_matches_corridor_mask``
    机械核验，不靠人工审查兜底。
    """
    faces = [int(f) for f in faces]
    eps = area_eps(crystal) if eps is None else eps
    all_dirs = grid.directions
    face0 = crystal.face(faces[0])
    n_a = crystal.normal(face0)
    if len(faces) == 1:
        # 与 corridor_mask 的单面分支一致：外反射没有临界角约束，可行集是半球，不是 entry_ok。
        return np.flatnonzero(-(all_dirs @ n_a) > 0)
    u_all, w_all = perp_bases(all_dirs)
    index = np.flatnonzero(entry_ok(n_a, all_dirs))
    poly = _ensure_ccw(_project(crystal.face_vertices(face0), u_all[index], w_all[index]))
    node = _Node((faces[0],), crystal, index, _initial_batch(poly))
    for f in faces[1:]:
        ghost = _ghost_for(node)
        ext = _extend_batch(ghost, node.index, node.batch, f, u_all, w_all, eps)
        if ext is None:
            return np.array([], dtype=int)
        idx, batch = ext
        node = _Node(node.faces + (f,), ghost, idx, batch)
    return node.index


def enumerate_raypaths_with_stats(crystal: Polyhedron, max_len: int, grid: LatLonGrid, *,
                                  entry_faces: Sequence[int] | None = None, symmetry_orbit: SymmetryOrbit | None = None,
                                  eps: float | None = None,
                                  refine_levels: int = 0) -> tuple[list[RaypathRecord], EnumerationStats]:
    """枚举长度 ≤ ``max_len`` 的全部可行光路，附剪枝统计。

    - ``entry_faces``：只从这些面入射（默认全部面）。六棱柱在 PBD 去重下只需 ``(1, 3)``；
    - ``symmetry_orbit``：面序列 → 其对称轨道，给定时每个轨道只保留字典序最小者（若该代表自身未被枚举到——
      比如 ``entry_faces`` 没含它的入射面——则保留轨道里已枚举到的最小者）；
    - ``eps``：走廊交集面积阈值，默认 :func:`~lumice_integral.geometry.feasibility.area_eps`；
    - ``refine_levels``：emit 前对候选做边界细化的级数（0 = 用粗网格面积与最小入射角，最快）。

    输出按 (长度, 字典序) 排序。``max_len`` 语义上是「枚举字长上限」，但每个入射面恒会先产出一条长度 1 的
    外反射记录（物理上总存在），即使 ``max_len < 1``；只有继续延伸（长度 ≥ 2）才受 ``max_len`` 约束。当前
    调用点都传 ``max_len ≥ 1``，未覆盖该边界。
    """
    eps = area_eps(crystal) if eps is None else eps
    stats = EnumerationStats()
    records: list[RaypathRecord] = []
    all_dirs = grid.directions
    u_all, w_all = perp_bases(all_dirs)
    weights = grid.weights
    faces_all = [f.number for f in crystal.faces]
    entries = faces_all if entry_faces is None else [int(f) for f in entry_faces]

    def emit(faces: tuple[int, ...], n_a: Vec3, n_tilde_b: Vec3, index: np.ndarray) -> None:
        d = all_dirs[index]
        ok = exit_ok(n_tilde_b, d)
        if not ok.any():
            return
        mask = np.zeros(grid.size, dtype=bool)
        mask[index[ok]] = True
        if refine_levels > 0:
            res = admissible_directions(crystal, faces, grid, CorridorMask(_full_mask(grid, index)), eps,
                                        refine_levels)
            area, best = res.area_sr, res.min_incidence_deg
        else:
            area = float(weights[index[ok]].sum())
            best = float(incidence_objective_deg(n_a, n_tilde_b, d[ok]).min())
        records.append(RaypathRecord(faces, AdmissibleMask(mask), area, best))
        stats.feasible_by_length[len(faces)] = stats.feasible_by_length.get(len(faces), 0) + 1

    def dfs(node: _Node, n_a: Vec3) -> None:
        if len(node.faces) >= max_len:
            return
        last = node.faces[-1]
        # 末面变镜面（入射面除外：[a] → [a, f] 时 f 仍在原晶体上）；所有子节点共用这一个新幽灵
        ghost = _ghost_for(node)
        for f in faces_all:
            if f == last:
                continue
            stats.attempted += 1
            ext = _extend_batch(ghost, node.index, node.batch, f, u_all, w_all, eps)
            if ext is None:
                stats.pruned += 1
                continue
            index, batch = ext
            child = _Node(node.faces + (f,), ghost, index, batch)
            stats.geometric_by_length[len(child.faces)] = stats.geometric_by_length.get(len(child.faces), 0) + 1
            emit(child.faces, n_a, ghost.normal(ghost.face(f)), child.index)
            dfs(child, n_a)

    for a in entries:
        face = crystal.face(a)
        n_a = crystal.normal(face)
        # 单面 = 外反射：没有走廊也没有临界角，可行集恒为半球
        hemi = np.flatnonzero(-(all_dirs @ n_a) > 0)
        mask = np.zeros(grid.size, dtype=bool)
        mask[hemi] = True
        records.append(RaypathRecord((a,), AdmissibleMask(mask), float(weights[hemi].sum()), 0.0))
        stats.feasible_by_length[1] = stats.feasible_by_length.get(1, 0) + 1
        if max_len < 2:
            continue
        root_index = np.flatnonzero(entry_ok(n_a, all_dirs))
        root_poly = _ensure_ccw(_project(crystal.face_vertices(face), u_all[root_index], w_all[root_index]))
        dfs(_Node((a,), crystal, root_index, _initial_batch(root_poly)), n_a)

    if symmetry_orbit is not None:
        records = _dedup(records, symmetry_orbit)
    records.sort(key=lambda r: (r.length, r.faces))
    return records, stats


def _full_mask(grid: LatLonGrid, index: np.ndarray) -> np.ndarray:
    mask = np.zeros(grid.size, dtype=bool)
    mask[index] = True
    return mask


def _dedup(records: list[RaypathRecord], symmetry_orbit: SymmetryOrbit) -> list[RaypathRecord]:
    """按轨道分组，每组保留面序列字典序最小的那条记录。"""
    by_orbit: dict[frozenset, RaypathRecord] = {}
    for r in records:
        key = frozenset(tuple(int(f) for f in s) for s in symmetry_orbit(r.faces))
        cur = by_orbit.get(key)
        if cur is None or r.faces < cur.faces:
            by_orbit[key] = r
    return list(by_orbit.values())


def enumerate_raypaths(crystal: Polyhedron, max_len: int, grid: LatLonGrid, *,
                       entry_faces: Sequence[int] | None = None, symmetry_orbit: SymmetryOrbit | None = None,
                       eps: float | None = None, refine_levels: int = 0) -> list[RaypathRecord]:
    """:func:`enumerate_raypaths_with_stats` 去掉统计的薄封装。"""
    return enumerate_raypaths_with_stats(crystal, max_len, grid, entry_faces=entry_faces, symmetry_orbit=symmetry_orbit,
                                         eps=eps, refine_levels=refine_levels)[0]


__all__ = ["EnumerationStats", "RaypathRecord", "SymmetryOrbit", "corridor_index_for_path", "enumerate_raypaths",
           "enumerate_raypaths_with_stats"]
