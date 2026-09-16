"""六方柱冰晶几何：顶点、面、面编号、外法向，以及刚体变换与射线求交。

Provenance：迁移自写作仓 ``halo_notes.geometry.core``（2026-09-16），公开名称 1:1 保留；本仓是这套几何的权威实现。

面编号沿用既有约定（与 Lumice ``doc/configuration.md`` 一致，也与本仓 ``optics.py`` 的
``FACE_3_NORMAL`` / ``FACE_5_NORMAL`` 逐位一致，由 ``tests/test_geometry_core.py`` 机械核验）：
底面 1（+c，即 +z）、2（-c）；侧面 3–8，面 ``3+i`` 的外法向方位角为 ``i·60°``
（从 +x 起逆时针），因此从 +c 轴俯视时 3→8 逆时针递增，
绕 c 轴转 60° 等价于侧面编号 +1 (mod 6)。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

Vec3 = np.ndarray

BASAL_TOP = 1
BASAL_BOTTOM = 2
PRISM_FACES = (3, 4, 5, 6, 7, 8)

N_ICE = 1.31  # 冰的折射率（可见光中段）；geometry 包内的唯一权威定义（feasibility / entry_measure 都从这里取）


@dataclass(frozen=True)
class Face:
    """多面体的一个面：编号 + 从外侧看逆时针排列的顶点索引环。"""

    number: int
    vertex_ids: tuple[int, ...]


def unit(v: Vec3) -> Vec3:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("zero-length vector")
    return v / n


def perp_basis(v: Sequence[float]) -> tuple[Vec3, Vec3]:
    """与 ``v`` 垂直的一对正交单位向量 ``(u, w)``，``(u, w, v̂)`` 成右手系。

    选取规则确定（不随机）：``u = v̂ × helper``，helper 取 +z（``v`` 接近 z 轴时改取 +x）。
    """
    a = unit(v)
    helper = np.array([0.0, 0.0, 1.0]) if abs(a[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = unit(np.cross(a, helper))
    w = np.cross(a, u)
    return u, w


def rotation(axis: Sequence[float], angle_deg: float) -> np.ndarray:
    """绕任意轴的旋转矩阵（Rodrigues 公式），右手系、角度制。"""
    k = unit(np.asarray(axis, dtype=float))
    t = np.deg2rad(angle_deg)
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(t) * kx + (1 - np.cos(t)) * kx @ kx


def rotation_between(src: Sequence[float], dst: Sequence[float]) -> np.ndarray:
    """把方向 ``src`` 转到 ``dst`` 的最小旋转矩阵。"""
    a, b = unit(src), unit(dst)
    c = np.cross(a, b)
    s = np.linalg.norm(c)
    if s < 1e-12:
        if a @ b > 0:
            return np.eye(3)
        # 反向：绕任一垂直轴转 180°
        helper = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
        return rotation(np.cross(a, helper), 180.0)
    return rotation(c, np.rad2deg(np.arctan2(s, a @ b)))


def rotation_from_frames(src_a: Sequence[float], src_b: Sequence[float],
                         dst_a: Sequence[float], dst_b: Sequence[float]) -> np.ndarray:
    """把方向 ``src_a`` 转到 ``dst_a``、同时把 ``src_b``（去掉沿 ``src_a`` 的分量后）转到
    ``dst_b``（同样去掉沿 ``dst_a`` 的分量）的旋转矩阵——用两对方向把姿态完全定死。
    典型用法：让晶体 c 轴指向画面某方向、再让某个侧面法向指向画面另一方向。
    """
    def frame(a, b):
        a = unit(a)
        b = np.asarray(b, dtype=float)
        b = unit(b - (b @ a) * a)
        return np.stack([a, b, np.cross(a, b)], axis=1)  # 列向量

    return frame(dst_a, dst_b) @ frame(src_a, src_b).T


class Polyhedron:
    """凸多面体：顶点数组 + 带编号的面。所有几何查询均基于顶点即时计算，
    所以刚体变换只需变换顶点。

    ``DIM`` 是顶点维数：这里是 3。半空间裁剪求交 / 包含判定 / 面距离 / 镜像 / 刚体变换这些只用点积的查询
    对维数不敏感（蓝本里曾有一个 ``DIM = 2`` 的多边形子类复用它们；本仓未迁移该子类），只有法向与"点在面内"
    判定（``normal`` / ``face_margin``）是三维专用写法。
    """

    DIM = 3

    def __init__(self, vertices: np.ndarray, faces: Iterable[Face]):
        self.vertices = np.asarray(vertices, dtype=float).reshape(-1, self.DIM)
        self.faces: tuple[Face, ...] = tuple(faces)
        self._by_number = {f.number: f for f in self.faces}
        if len(self._by_number) != len(self.faces):
            raise ValueError("duplicate face numbers")
        self._normals: dict[int, Vec3] = {}
        self._plane_cache: tuple[np.ndarray, np.ndarray] | None = None

    # ---- 基本查询 -------------------------------------------------------
    def face(self, number: int) -> Face:
        return self._by_number[number]

    def face_vertices(self, face: Face) -> np.ndarray:
        return self.vertices[list(face.vertex_ids)]

    def centroid(self, face: Face | None = None) -> Vec3:
        """面质心；``face=None`` 时返回整个体的顶点质心。"""
        pts = self.vertices if face is None else self.face_vertices(face)
        return pts.mean(axis=0)

    def normal(self, face: Face) -> Vec3:
        """面的单位外法向（Newell 公式；顶点环从外侧看逆时针）。

        顶点数组在实例生命周期内不变（变换 / 镜像都返回新实例），所以按面号缓存：
        射线求交与光路搜索（枚举 / 追迹要做上万次求交）反复查法向，
        逐次 ``np.cross`` 会占掉九成时间。
        """
        cached = self._normals.get(face.number)
        if cached is not None:
            return cached
        pts = self.face_vertices(face)
        nxt = np.roll(pts, -1, axis=0)
        n = unit(np.sum(np.cross(pts, nxt), axis=0))
        self._normals[face.number] = n
        return n

    def _planes(self) -> tuple[np.ndarray, np.ndarray]:
        """所有面的 (外法向 (F,3), 面上一点 (F,3))，与 ``self.faces`` 同序；缓存。"""
        if self._plane_cache is None:
            normals = np.stack([self.normal(f) for f in self.faces])
            points = np.stack([self.face_vertices(f)[0] for f in self.faces])
            self._plane_cache = (normals, points)
        return self._plane_cache

    @property
    def edges(self) -> tuple[tuple[int, int], ...]:
        """去重后的边（顶点索引对，小索引在前）。"""
        seen: dict[tuple[int, int], None] = {}
        for f in self.faces:
            ids = f.vertex_ids
            for a, b in zip(ids, ids[1:] + ids[:1]):
                seen[(min(a, b), max(a, b))] = None
        return tuple(seen)

    def edge_faces(self, edge: tuple[int, int]) -> tuple[Face, ...]:
        """与某条边相邻的面。"""
        a, b = edge
        out = []
        for f in self.faces:
            ids = f.vertex_ids
            pairs = {(min(x, y), max(x, y)) for x, y in zip(ids, ids[1:] + ids[:1])}
            if (min(a, b), max(a, b)) in pairs:
                out.append(f)
        return tuple(out)

    # ---- 变换 ------------------------------------------------------------
    def transformed(self, rotation: np.ndarray | None = None,
                    translation: Sequence[float] | None = None) -> "Polyhedron":
        """返回刚体变换后的副本：先绕原点旋转，再平移。"""
        v = self.vertices
        if rotation is not None:
            v = v @ np.asarray(rotation, dtype=float).T
        if translation is not None:
            v = v + np.asarray(translation, dtype=float)
        return type(self)._copy_with(self, v)

    def mirrored(self, face: Face) -> "Polyhedron":
        """返回关于 ``face`` 所在平面镜像后的副本（光路展开用的"幽灵晶体"）。

        镜像变换行列式为 -1，会把"从外侧看逆时针"的顶点环翻成顺时针，所以每个
        面的顶点环必须同步反转，否则 Newell 法向全部朝内、可见性与射线求交全错。
        面编号原样保留：幽灵晶体上的 5 号面仍叫 5。
        """
        n = self.normal(face)
        p0 = self.face_vertices(face)[0]
        v = self.vertices
        mirrored_v = v - 2.0 * ((v - p0) @ n)[:, None] * n[None, :]
        mirrored_faces = tuple(Face(f.number, tuple(reversed(f.vertex_ids))) for f in self.faces)
        return type(self)._copy_with(self, mirrored_v, mirrored_faces)

    def _copy_with(self, vertices: np.ndarray,
                   faces: Iterable[Face] | None = None) -> "Polyhedron":
        return Polyhedron(vertices, self.faces if faces is None else faces)

    # ---- 射线求交 --------------------------------------------------------
    def intersect_ray(self, origin: Sequence[float], direction: Sequence[float],
                      eps: float = 1e-9):
        """射线与凸体求交（半空间裁剪）。

        返回 ``(t_in, face_in, t_out, face_out)``；不相交返回 ``None``。
        起点在体内时 ``t_in < 0``、``face_in`` 为起点反向最近穿出的面。
        """
        o = np.asarray(origin, dtype=float)
        d = unit(direction)
        normals, points = self._planes()
        denom = normals @ d
        num = np.einsum("ij,ij->i", normals, points - o)  # 到平面的有符号距离（沿 d 的分子）
        parallel = np.abs(denom) < eps
        if np.any(num[parallel] < -eps):
            return None  # 与某面平行且在其外侧
        t = np.where(parallel, np.nan, num / np.where(parallel, 1.0, denom))
        entering = (denom < 0) & ~parallel   # 进入半空间
        leaving = (denom > 0) & ~parallel    # 离开半空间
        if not entering.any() or not leaving.any():
            return None
        i_in = int(np.nanargmax(np.where(entering, t, -np.inf)))
        i_out = int(np.nanargmin(np.where(leaving, t, np.inf)))
        t_in, t_out = float(t[i_in]), float(t[i_out])
        if t_in > t_out:
            return None
        return t_in, self.faces[i_in], t_out, self.faces[i_out]

    def contains(self, point: Sequence[float], eps: float = 1e-9) -> bool:
        p = np.asarray(point, dtype=float)
        return all(self.normal(f) @ (p - self.face_vertices(f)[0]) <= eps
                   for f in self.faces)

    def face_margin(self, face: Face, point: Sequence[float]) -> float:
        """点到面多边形各边（面内）的最小有符号距离：正 = 在面内部，0 = 恰在棱边上，
        负 = 在面外。只看面内分量，不检查点是否在面所在平面上（配合 :meth:`face_distance`）。
        用于判定光线"穿过面的内部"而不是擦边。"""
        q = np.asarray(point, dtype=float)
        n = self.normal(face)
        pts = self.face_vertices(face)
        nxt = np.roll(pts, -1, axis=0)
        # 顶点环从外侧看逆时针：边向量 × (点 - 边起点) 沿外法向为正 ⇔ 点在边的内侧
        edge = nxt - pts
        signed = np.cross(edge, q - pts) @ n / np.linalg.norm(edge, axis=1)
        return float(signed.min())

    def face_distance(self, face: Face, point: Sequence[float]) -> float:
        """点到面所在平面的有符号距离（沿外法向为正）。"""
        q = np.asarray(point, dtype=float)
        return float(self.normal(face) @ (q - self.face_vertices(face)[0]))


class HexPrism(Polyhedron):
    """六方柱冰晶。``a`` 为六边形边长（= 外接圆半径），``h`` 为柱高。

    片晶 ``h/a`` 小，柱晶 ``h/a`` 大。生成时 c 轴沿 +z、面 3 法向沿 +x，
    体心在原点；其他姿态用 :meth:`transformed` 得到。
    """

    def __init__(self, a: float = 1.0, h: float = 1.0):
        self.a = float(a)
        self.h = float(h)
        # 六边形顶点 k 位于方位角 -30° + 60°k，使面 3+i 的外法向落在 i·60°
        ang = np.deg2rad(-30.0 + 60.0 * np.arange(6))
        ring = np.stack([self.a * np.cos(ang), self.a * np.sin(ang)], axis=1)
        top = np.column_stack([ring, np.full(6, self.h / 2)])
        bottom = np.column_stack([ring, np.full(6, -self.h / 2)])
        vertices = np.vstack([top, bottom])  # 0–5 顶环，6–11 底环

        faces = [
            Face(BASAL_TOP, tuple(range(6))),
            Face(BASAL_BOTTOM, tuple(6 + k for k in range(5, -1, -1))),
        ]
        for i in range(6):
            j = (i + 1) % 6
            # 从外侧看逆时针：底-左、底-右、顶-右、顶-左
            faces.append(Face(3 + i, (6 + i, 6 + j, j, i)))
        super().__init__(vertices, faces)

    @classmethod
    def from_ratio(cls, ratio: float, a: float = 1.0) -> "HexPrism":
        """按高径比 ``h/a`` 构造。"""
        return cls(a=a, h=ratio * a)

    def _copy_with(self, vertices: np.ndarray,
                   faces: Iterable[Face] | None = None) -> "HexPrism":
        # 手动搬运字段以绕开 __init__（它会按 a/h 重新生成顶点，覆盖掉变换后的 vertices）；
        # HexPrism.__init__ 新增构造参数时必须同步在这里搬运，否则变换后的实例会悄悄丢字段。
        # ``faces`` 须透传：mirrored() 用它传入顶点环已反转的面。
        obj = HexPrism.__new__(HexPrism)
        obj.a, obj.h = self.a, self.h
        Polyhedron.__init__(obj, vertices, self.faces if faces is None else faces)
        return obj
