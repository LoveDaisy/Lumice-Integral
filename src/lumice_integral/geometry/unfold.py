"""按面序列展开光路（纯组合版，不依赖已追迹的光路）：:func:`unfold_faces` 把中间各面当镜子级联镜像出
幽灵晶体链，:func:`fold_matrix` 给出同一面序列的方向变换矩阵。

两者是同一件事在两个坐标系里的样子：幽灵链活在**世界坐标系**（每次关于"当前幽灵"的面平面镜像，
逐步平移旋转），方向矩阵 $M = S_{m_k}\\cdots S_{m_1}$ 活在**固定晶体坐标系**（法向取原晶体的）。
记 $L_k$ 为幽灵 $k$ 相对原晶体的线性部分，则 $L_k = S_{m_1}\\cdots S_{m_k} = M^{-1} = M^T$，
所以最后一个幽灵上出射面的法向 $\\tilde n_b = M^{-1} n_b$——这是 ``tests/test_geometry_unfold.py`` 的交叉断言。

Provenance：迁移自写作仓 ``halo_notes.geometry.unfold``（2026-09-16）；镜像逻辑只此一份，写作仓侧的绘图封装
今后调用本模块。
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
