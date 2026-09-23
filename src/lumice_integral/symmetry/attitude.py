"""柱晶姿态 → 天空方向（ch7 第 6 节、ch3 第 71 行三步旋转的最小实现；ch10/11 姿态空间接口的种子）。

- :func:`column_attitude`：$R = R_\\psi R_z(\\theta) = R_z(\\psi)\\,R_y(90°)\\,R_z(\\theta)$——先绕 c 轴自转 θ、再绕 y 轴放倒
  90°、最后绕 z 轴转到水平方位 ψ，与 ``03-parallel-raypaths/code/fig_rotation_steps.py`` 的三步矩阵同一约定（写作仓有单测逐项比对）。
  与 Lumice 姿态链的关系 ``R_Lumice(az, 90, roll) = column_attitude(az, roll - 180)`` 由 ``tests/test_conventions.py`` 核验。
- :func:`sky_direction`：$T\\,\\hat s = R M R^{-1}\\hat s$，``sun`` 用 ch3 约定「指向太阳的单位向量 $(\\cos\\Sigma, 0, \\sin\\Sigma)$」，
  返回天空点方向；$M = br$（#3）时逐点重现 ``_tricker_arc.tricker_arc``（ch3 第 104–110 行的 $\\hat r'$）。

只依赖 numpy。

Provenance：迁移自写作仓 ``halo_notes.math.attitude``（2026-09-24，task-symmetry-authority）。``Ry`` / ``column_attitude`` /
``sky_direction`` 1:1 保留；``sun_vector`` 不迁——同一语义在本仓的权威是 :func:`lumice_integral.camera.sun_direction`
（``(cos S, 0, sin S)``，指向太阳），写作仓侧切换时改调它。
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .reflection_group import Rz


def Ry(deg: float) -> np.ndarray:
    """绕 y 轴转 ``deg`` 度（右手系）。"""
    t = np.radians(deg)
    return np.array([[np.cos(t), 0.0, np.sin(t)], [0.0, 1.0, 0.0], [-np.sin(t), 0.0, np.cos(t)]])


def column_attitude(psi_deg: float, theta_deg: float) -> np.ndarray:
    """柱晶姿态矩阵 $R_z(\\psi)\\,R_y(90°)\\,R_z(\\theta)$：c 轴水平、方位角 ψ，绕 c 轴自转 θ。"""
    return Rz(psi_deg) @ Ry(90.0) @ Rz(theta_deg)


def sky_direction(M: np.ndarray, R: np.ndarray, sun: Sequence[float]) -> np.ndarray:
    """晶体系里的方向变换 $M$ 经姿态 $R$ 搬到天空：$R M R^{-1}\\hat s$（$R$ 正交，$R^{-1} = R^T$）。
    ``sun`` 可以是一个 (3,) 方向或 (N, 3) 一组方向，返回同形状。"""
    s = np.asarray(sun, dtype=float)
    T = np.asarray(R, float) @ np.asarray(M, float) @ np.asarray(R, float).T
    return s @ T.T


__all__ = ["Ry", "column_attitude", "sky_direction"]
