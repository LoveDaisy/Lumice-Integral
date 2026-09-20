"""按面序列展开光路（纯组合版，不依赖已追迹的光路）：:func:`unfold_faces` 把中间各面当镜子级联镜像出
幽灵晶体链，:func:`fold_matrix` 给出同一面序列的方向变换矩阵。

两者是同一件事在两个坐标系里的样子：幽灵链活在**世界坐标系**（每次关于"当前幽灵"的面平面镜像，
逐步平移旋转），方向矩阵 $M = S_{m_k}\\cdots S_{m_1}$ 活在**固定晶体坐标系**（法向取原晶体的）。
记 $L_k$ 为幽灵 $k$ 相对原晶体的线性部分，则 $L_k = S_{m_1}\\cdots S_{m_k} = M^{-1} = M^T$，
所以最后一个幽灵上出射面的法向 $\\tilde n_b = M^{-1} n_b$——这是 ``tests/test_geometry_unfold.py`` 的交叉断言。

Provenance：迁移自写作仓 ``halo_notes.geometry.unfold``（2026-09-16）；镜像逻辑只此一份，写作仓侧的绘图封装
今后调用本模块。

光路级不变量（2026-09-20，task-path-class-rendering-unit）：:func:`wedge_angle_deg` 是写作仓
``halo_notes.math.signature.wedge_angle`` 的最小子集迁移（楔角 $W = \\arccos(-n_a\\cdot M^T n_b)$，度），
:func:`halo_map_rank` 按 ch8 A0-06 族的规则把"$M = I$ 且 $W = 0$"的光路判为秩 0（出射方向与姿态无关，
晕图是太阳方向的点质量），其余判为秩 2。Φ-类等价、34 类表与 D6h 群本身不在此处（见 plan 默认假设 1）。
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .core import Polyhedron, Vec3


def _check_faces(faces: Sequence[int]) -> list[int]:
    faces = [int(f) for f in faces]
    if len(faces) < 2:
        raise ValueError("faces must be [entry, *reflections, exit] with at least entry and exit")
    return faces


def unfold_faces(crystal: Polyhedron, faces: Sequence[int]) -> tuple[list[Polyhedron], Vec3]:
    """``faces = [a, m_1, …, m_k, b]``：``a`` 入射面、``m_i`` 依次内反射、``b`` 出射面。

    返回 ``(bodies, n_tilde_b)``：``bodies[0]`` 是 ``crystal`` 本身，``bodies[j]`` 是 ``bodies[j-1]``
    关于其面 ``m_j`` 镜像后的幽灵（``j = 1..k``，镜像作用在前一个幽灵上而不是原晶体上），
    ``len(bodies) == k + 1``；展开空间里光线依次穿过 ``bodies[0]`` 的面 ``a``、``bodies[j]`` 的面 ``m_j``、
    ``bodies[-1]`` 的面 ``b``。``n_tilde_b`` 是 ``bodies[-1]`` 上出射面 ``b`` 的外法向（展开空间里的
    出射面法向）。无内反射（``k = 0``）时 ``bodies == [crystal]``。
    """
    faces = _check_faces(faces)
    bodies = [crystal]
    for number in faces[1:-1]:
        bodies.append(bodies[-1].mirrored(bodies[-1].face(number)))
    n_tilde_b = bodies[-1].normal(bodies[-1].face(faces[-1]))
    return bodies, n_tilde_b


def fold_matrix(crystal: Polyhedron, faces: Sequence[int]) -> np.ndarray:
    """中间面镜面矩阵按相遇顺序左乘：$M = S_{m_k}\\cdots S_{m_1}$，$S_n = I - 2nn^T$，``n`` 取
    ``crystal``（未镜像）的面外法向。六棱柱上与用解析面法向独立算出的镜面连乘（``tests/test_geometry_unfold.py``
    里的 oracle）逐位相等；无内反射时为单位阵。"""
    faces = _check_faces(faces)
    M = np.eye(3)
    for number in faces[1:-1]:
        n = crystal.normal(crystal.face(number))
        M = (np.eye(3) - 2.0 * np.outer(n, n)) @ M
    return M


WEDGE_ZERO_TOLERANCE_DEG = 1e-9


def wedge_angle_deg(crystal: Polyhedron, faces: Sequence[int]) -> float:
    """面序列的楔角 $W = \\arccos(-n_a \\cdot M^T n_b)$（度）：展开空间里入射面与出射面法向的夹角。

    ``M^T n_b`` 就是 :func:`unfold_faces` 返回的 ``n_tilde_b``（模块文档）；平行光路（首尾折射抵消）
    ``W = 0``，六棱柱 22° 晕族的两面光路（如 ``3-5``、``3-1-2-5``）``W = 60``，46° 族（如 ``1-3``）
    ``W = 90``。返回值单位为度，与 :class:`lumice_integral.path_class.PathClass.wedge_deg` 一致。
    """
    faces = _check_faces(faces)
    n_a = crystal.normal(crystal.face(faces[0]))
    n_b = crystal.normal(crystal.face(faces[-1]))
    minus_n_tilde_b = -(fold_matrix(crystal, faces).T @ n_b)
    # atan2 form: arccos loses ~sqrt(eps) near W = 0 (Newell normals put the
    # antipodal pairs 5-8 at 8.5e-7 deg), which is exactly where rank 0 is decided.
    return float(np.degrees(np.arctan2(np.linalg.norm(np.cross(n_a, minus_n_tilde_b)), n_a @ minus_n_tilde_b)))


def halo_map_rank(crystal: Polyhedron, faces: Sequence[int], *,
                  tolerance_deg: float = WEDGE_ZERO_TOLERANCE_DEG) -> int:
    """光路的晕图秩：``0`` 当且仅当 ``fold_matrix`` 为单位阵且楔角为 0（ch8 A0-06 族，如 ``1-2``、``3-6``），
    此时出射方向恒等于入射方向；否则 ``2``。秩 0 光路不进纤维流水线（``path_class`` 模块）。"""
    faces = _check_faces(faces)
    if not np.allclose(fold_matrix(crystal, faces), np.eye(3), rtol=0.0, atol=1e-12):
        return 2
    return 0 if wedge_angle_deg(crystal, faces) <= tolerance_deg else 2
