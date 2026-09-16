"""六方锥冰晶 :class:`Pyramid`：六棱柱两端各接一个截顶的六方锥台，2 底 + 6 侧 + 12 锥面 = 20 面。

Provenance：迁移自写作仓 ``halo_notes.geometry.pyramid``（2026-09-16）。下表的取证来源指向 Lumice 仓库
（``LoveDaisy/ice_halo_sim`` @ ``9232bccb``）的源码位置，是历史取证记录，本仓不包含这些文件。

面编号与 Lumice 逐面一致：

============  ==================================  ==========================================================
面号          语义                                取证来源
============  ==================================  ==========================================================
1 / 2         上 / 下底面（±c 即 ±z）             ``src/core/geo3d_closedform.cpp:749-750``
3+i (i=0..5)  棱柱侧面，外法向方位角 ``i·60°``    ``geo3d_closedform.cpp:752``；顶点表 ``:782-794``
13+i          上锥面（+z 侧），方位角同侧面 i     ``geo3d_closedform.cpp:753``；法向 xy 与侧面同向、z=+det ``:797-802``
23+i          下锥面（-z 侧），方位角同侧面 i     ``geo3d_closedform.cpp:754``；法向 z=-det ``:803-807``
============  ==================================  ==========================================================

slot 布局注释见 ``src/core/geo3d_closedform.hpp:202-207``；``doc/configuration.md:369-371``
（"upper pyramidal = 13–18, lower pyramidal = 23–28"）交叉核对一致。

锥面法向与 c 轴夹角 ``θ = arctan((2/√3)·c_over_a)``（``c_over_a = 1.6288`` 时 ``θ ≈ 62.0°``，
即 {10-11} 锥面；``tests/test_geometry_pyramid.py`` 用闭式法向公式独立对照）。
由"锥面过棱柱环相邻两顶点、法向如上"解得理论锥顶高出棱柱环 ``H = a·cos30°·tanθ = a·c_over_a``。

截顶量 ``tip_ratio``：锥台在从棱柱环到理论顶点的 ``tip_ratio`` 倍高度处截断（上下对称），语义与
Lumice ``upper_h`` / ``lower_h`` 的 ``(0, 1)`` 区间相同（``doc/configuration.md`` §Pyramid Shape Legality）。
默认 ``0.5`` 是占位值（Lumice 无默认可抄），锥晶正式立项时可能改；本模块只覆盖
"上下对称、两侧都截顶"这一种形状，单侧无锥 / 锥到顶点等变体留待后续任务。
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from .core import BASAL_BOTTOM, BASAL_TOP, Face, Polyhedron

C_OVER_A_ICE = 1.6288          # 冰的 c/a 轴比
UPPER_PYRAMID_FACES = (13, 14, 15, 16, 17, 18)
LOWER_PYRAMID_FACES = (23, 24, 25, 26, 27, 28)


def pyramid_face_angle(c_over_a: float = C_OVER_A_ICE) -> float:
    """锥面外法向与 +c 轴的夹角（度）：``arctan((2/√3)·c_over_a)``。"""
    return float(np.degrees(np.arctan(2.0 / np.sqrt(3.0) * c_over_a)))


class Pyramid(Polyhedron):
    """六方锥冰晶。``a`` 为六边形边长（= 棱柱环外接圆半径），``h`` 为中间棱柱段高度，
    ``c_over_a`` 定锥面倾角，``tip_ratio ∈ (0, 1)`` 定截顶位置（见模块 docstring）。

    与 :class:`HexPrism` 同一构造范式：子类 ``__init__`` 拼顶点与面后交给 ``Polyhedron.__init__``，
    ``_copy_with`` 手动搬运字段绕开 ``__init__``。生成姿态同 ``HexPrism``：c 轴沿 +z、面 3 法向沿 +x、体心在原点。
    """

    def __init__(self, a: float = 1.0, h: float = 1.0, c_over_a: float = C_OVER_A_ICE,
                 tip_ratio: float = 0.5):
        if not 0.0 < tip_ratio < 1.0:
            raise ValueError("tip_ratio must lie in (0, 1): both cones truncated, both basal faces present")
        self.a = float(a)
        self.h = float(h)
        self.c_over_a = float(c_over_a)
        self.tip_ratio = float(tip_ratio)
        cone_h = self.a * self.c_over_a * self.tip_ratio      # 锥台高（棱柱环到截顶面）
        tip_a = self.a * (1.0 - self.tip_ratio)               # 截顶小六边形边长（相似三角形）
        # 六边形顶点 k 位于方位角 -30° + 60°k，使面 3+i（及锥面 13+i / 23+i）的外法向落在 i·60°
        ang = np.deg2rad(-30.0 + 60.0 * np.arange(6))
        cs = np.stack([np.cos(ang), np.sin(ang)], axis=1)

        def ring(radius: float, z: float) -> np.ndarray:
            return np.column_stack([radius * cs, np.full(6, z)])

        vertices = np.vstack([
            ring(self.a, self.h / 2),                      # 0–5   棱柱顶环
            ring(self.a, -self.h / 2),                     # 6–11  棱柱底环
            ring(tip_a, self.h / 2 + cone_h),              # 12–17 上截顶小环
            ring(tip_a, -(self.h / 2 + cone_h)),           # 18–23 下截顶小环
        ])

        faces = [
            Face(BASAL_TOP, tuple(12 + k for k in range(6))),
            Face(BASAL_BOTTOM, tuple(18 + k for k in range(5, -1, -1))),
        ]
        for i in range(6):
            j = (i + 1) % 6
            # 从外侧看逆时针：底-左、底-右、顶-右、顶-左（与 HexPrism 侧面同一环序）
            faces.append(Face(3 + i, (6 + i, 6 + j, j, i)))
            faces.append(Face(UPPER_PYRAMID_FACES[i], (i, j, 12 + j, 12 + i)))
            faces.append(Face(LOWER_PYRAMID_FACES[i], (18 + i, 18 + j, 6 + j, 6 + i)))
        super().__init__(vertices, faces)

    def _copy_with(self, vertices: np.ndarray,
                   faces: Iterable[Face] | None = None) -> "Pyramid":
        # 同 HexPrism._copy_with：绕开 __init__ 搬字段；新增构造参数时必须同步搬运
        obj = Pyramid.__new__(Pyramid)
        obj.a, obj.h, obj.c_over_a, obj.tip_ratio = self.a, self.h, self.c_over_a, self.tip_ratio
        Polyhedron.__init__(obj, vertices, self.faces if faces is None else faces)
        return obj
