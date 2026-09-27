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
默认 ``0.5`` 是占位值（Lumice 无默认可抄），锥晶正式立项时可能改。``Pyramid(a, h, c_over_a, tip_ratio)``
只覆盖"上下对称、两侧都截顶、正六边形截面"；Lumice ``pyramid`` 的全部语义（独立的 ``upper_h`` / ``lower_h``、
独立的 Miller 指数或楔角、六个 ``face_distance``）由 :meth:`Pyramid.from_lumice` 构造，见其 docstring。
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Iterable, Sequence

import numpy as np

from .closed_form import PRESENT_REL_TOL, REGULAR_FACE_DISTANCE, cone_sweep, hex_cross_section, inset_corner
from .core import BASAL_BOTTOM, BASAL_TOP, Face, Polyhedron

C_OVER_A_ICE = 1.6288          # 冰的 c/a 轴比
UPPER_PYRAMID_FACES = (13, 14, 15, 16, 17, 18)
LOWER_PYRAMID_FACES = (23, 24, 25, 26, 27, 28)

# ---- Lumice ``pyramid`` 配置语义的常量（Lumice 仓只读取证；本仓独立实现，不 link / import）----------------
LUMICE_MILLER_C_OVER_A = 1.629
"""Miller 指数换算楔角所用的 c/a：Lumice ``kIceCrystalC``（``src/core/geo3d.hpp``）。与本仓 ``C_OVER_A_ICE``
（1.6288）差 1.2e-4 相对量；``from_lumice`` 的指数入口按 Lumice 的值换算，以便与 Lumice 配置逐字对应。"""
MIN_WEDGE_DEG, MAX_WEDGE_DEG = 0.1, 89.9
"""楔角（锥面与 c 轴的夹角）合法区间，两端含（``doc/configuration.md`` §Pyramid Shape Legality）。Lumice 以 float32
比较（``0.1f <= alpha <= 89.9f``），:func:`wedge_angle_is_legal` 同样在 float32 下判定，端点恰好落在边界上。"""
LUMICE_FLOAT_EPS = 1e-5
"""Lumice ``math::kFloatEps``：``upper_h`` / ``lower_h`` 不大于它时该侧无锥，``prism_h`` 不大于它时无棱柱段
（``src/core/geo3d_closedform.cpp`` ``ComputeClosedFormPyramidInner`` 的 ``has_upper`` / ``h2 > kFloatEps``）。"""
APEX_SNAP_REL = 5.0 * LUMICE_FLOAT_EPS
"""截顶离自然锥顶不到 ``APEX_SNAP_REL · max|face_distance − m|``（``m`` 单位）时吸附到锥顶、该侧无底面：
Lumice ``ApexCollapsedAt``（``kClosedFormGapToleranceCoefficient · kFloatEps``，``doc/configuration.md`` 所说
``upper_h`` 贴近 1 的吸附带）。"""


def pyramid_face_angle(c_over_a: float = C_OVER_A_ICE) -> float:
    """锥面外法向与 +c 轴的夹角（度）：``arctan((2/√3)·c_over_a)``。"""
    return float(np.degrees(np.arctan(2.0 / np.sqrt(3.0) * c_over_a)))


def wedge_angle_is_legal(wedge_deg: float) -> bool:
    """楔角是否在 ``[0.1°, 89.9°]`` 内（两端含，按 Lumice 的 float32 比较，见 :data:`MIN_WEDGE_DEG`）。"""
    return bool(np.float32(MIN_WEDGE_DEG) <= np.float32(wedge_deg) <= np.float32(MAX_WEDGE_DEG))


def wedge_angle_to_c_over_a(wedge_deg: float) -> float:
    """Lumice 楔角（锥面与 c 轴夹角，度）→ 本模块的 ``c_over_a = (√3/2)·cot(wedge)``（锥面法向与 c 轴夹角
    ``90° − wedge``，即 :func:`pyramid_face_angle` 的反函数）。"""
    return float(np.sqrt(3.0) / 2.0 / np.tan(np.radians(wedge_deg)))


def miller_indices_to_c_over_a(indices: Sequence[Real]) -> float | None:
    """Lumice ``upper_indices`` / ``lower_indices`` 的 ``(h, k, l)`` → ``c_over_a``；``h == 0`` 表示该侧无锥，返回 ``None``。

    Lumice 的楔角是 ``atan((√3/2)·(l/h) / c)``（``c`` = :data:`LUMICE_MILLER_C_OVER_A`，``src/core/geo3d.cpp``
    ``MillerIndexToWedgeAngleDeg``），代入 ``c_over_a = (√3/2)·cot(wedge)`` 得代数直接式 ``c_over_a = h·c / l``
    （Lumice 闭式路径 ``a1 = i1·c / (2·i4)`` 同一关系，免去 atan→tan 往返）。拒绝规则按 ``doc/configuration.md``
    §11 "Reading the Miller-Index Fallback Warning"（Lumice 拒绝后沿用旧角度并告警；本仓 fail-fast 抛 ``ValueError``）：
    不是恰好三个整数、``k != 0``、``h`` 或 ``l`` 为负、楔角落在 ``[0.1°, 89.9°]`` 之外。
    """
    values = tuple(indices)
    if len(values) != 3:
        raise ValueError(f"Miller indices must be exactly three integers (h, k, l), got {values!r} "
                         "(Lumice doc/configuration.md §11: a four-index Miller-Bravais label is refused)")
    for slot, value in zip("hkl", values):
        if isinstance(value, bool) or not isinstance(value, Real) or not float(value).is_integer():
            raise ValueError(f"Miller index {slot} = {value!r} is not a whole number (Lumice doc/configuration.md §11)")
    h, k, l = (int(v) for v in values)
    if k != 0:
        raise ValueError(f"Miller index k = {k} must be 0: a pyramidal face turned off the prism edges is not "
                         "expressible (Lumice doc/configuration.md §11)")
    if h < 0 or l < 0:
        raise ValueError(f"Miller indices {values!r}: h and l must not be negative (Lumice doc/configuration.md §11)")
    if h == 0:
        return None
    wedge = float(np.degrees(np.arctan(np.sqrt(3.0) / 2.0 * l / h / LUMICE_MILLER_C_OVER_A)))
    if not wedge_angle_is_legal(wedge):
        raise ValueError(f"Miller indices {values!r} make a wedge angle of {wedge:.4g} degrees, outside "
                         f"[{MIN_WEDGE_DEG}, {MAX_WEDGE_DEG}] (Lumice doc/configuration.md §11 / §Pyramid Shape Legality)")
    return h * LUMICE_MILLER_C_OVER_A / l


@dataclass(frozen=True)
class PyramidShape:
    """The Lumice ``pyramid`` shape a :class:`Pyramid` was built from (:meth:`Pyramid.from_lumice`).

    Heights are folded (``abs``) as Lumice folds them; ``upper_c_over_a`` / ``lower_c_over_a`` is ``None`` for a
    side without a cone (Miller ``h == 0``, or a direct wedge angle outside ``[0.1°, 89.9°]``).
    """

    prism_h: float
    upper_h: float
    lower_h: float
    upper_c_over_a: float | None
    lower_c_over_a: float | None
    face_distance: tuple[float, ...]


class Pyramid(Polyhedron):
    """六方锥冰晶。``a`` 为六边形边长（= 棱柱环外接圆半径），``h`` 为中间棱柱段高度，
    ``c_over_a`` 定锥面倾角，``tip_ratio ∈ (0, 1)`` 定截顶位置（见模块 docstring）。

    与 :class:`HexPrism` 同一构造范式：子类 ``__init__`` 拼顶点与面后交给 ``Polyhedron.__init__``，
    ``_copy_with`` 手动搬运字段绕开 ``__init__``。生成姿态同 ``HexPrism``：c 轴沿 +z、面 3 法向沿 +x、体心在原点。

    :meth:`from_lumice` 覆盖 Lumice ``pyramid`` 的全部语义；它构造的实例的 ``shape`` 记录所用的 Lumice 形状
    （直接用 ``__init__`` 构造时为 ``None``），非对称 / 非正六边形截面的实例 ``c_over_a`` 与 ``tip_ratio`` 为
    ``None``（两侧各有自己的值，见 ``shape``）。
    """

    shape: PyramidShape | None = None

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
        # 棱柱环与截顶小环都是正六边形横截面，与 HexPrism 同一闭式构造（closed_form.hex_cross_section）：
        # 顶点 k 位于方位角 -30° + 60°k，使面 3+i（及锥面 13+i / 23+i）的外法向落在 i·60°

        def ring(edge: float, z: float) -> np.ndarray:
            return np.column_stack([hex_cross_section(edge).ring, np.full(6, z)])

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

    @classmethod
    def from_lumice(cls, prism_h: float, upper_h: float = 0.0, lower_h: float = 0.0,
                    upper_indices: Sequence[Real] = (1, 0, 1), lower_indices: Sequence[Real] = (1, 0, 1),
                    face_distance: Sequence[float] = REGULAR_FACE_DISTANCE, a: float = 1.0, *,
                    upper_wedge_deg: float | None = None, lower_wedge_deg: float | None = None) -> "Pyramid":
        """按 Lumice ``pyramid`` 的 ``shape`` 构造（``doc/configuration.md`` §pyramid、§Pyramid Shape Legality、§11），
        参考六边形边长取 ``a``。

        - ``prism_h``：棱柱段高 / 参考外接圆直径，``h = 2a·prism_h``（同 :meth:`HexPrism.from_lumice`）；不大于
          :data:`LUMICE_FLOAT_EPS` 时无棱柱段（侧面 3–8 不存在，两锥台或锥台与底面在 ``z = 0`` 相接）。
        - ``upper_h`` / ``lower_h``：取绝对值（Lumice 折叠负值）。不大于 :data:`LUMICE_FLOAT_EPS` → 该侧无锥、
          以底面封口；``(0, 1)`` → 截顶，截在"棱柱肩部到自然锥顶"的该比例处（侵蚀量 ``m`` 的比例），有底面；
          ``>= 1`` 或截顶落在 :data:`APEX_SNAP_REL` 吸附带内 → 锥到自然锥顶（点或棱脊），该侧无底面。
        - ``upper_indices`` / ``lower_indices``：Miller ``(h, 0, l)``，经 :func:`miller_indices_to_c_over_a` 换算，
          非法组合抛 ``ValueError``，``h == 0`` 表示该侧无锥。``upper_wedge_deg`` / ``lower_wedge_deg`` 给出时
          取代对应的指数（Lumice 直接给楔角的入口，C API ``LUMICE_CrystalParam.upper_wedge_angle``）：
          ``[0.1°, 89.9°]`` 之外按 Lumice 语义静默视为无锥。
        - ``face_distance``：同 :class:`HexPrism`，作用于棱柱段与两个锥台的每一层截面——侵蚀量 ``m`` 处的截面是
          ``face_distance − m`` 的六方向星（:func:`.closed_form.cone_sweep`）。

        两侧都无锥且无棱柱段（零体积）、或 ``m = 0`` 截面无面积时抛 ``ValueError``（Lumice 丢弃这类晶体）。
        上下完全对称（同高、同 ``c_over_a``）、``face_distance`` 全为 1、有棱柱段且截顶比例落在 ``(0, 1)``
        未吸附时，直接委托 ``Pyramid(a, h, c_over_a, tip_ratio)``（同一构造只有一份实现）；其余走闭式侵蚀构造
        :meth:`_from_cones`。不存在的面（例如无锥一侧的 13–18 或 23–28）查询抛 ``KeyError``。
        """
        values = (prism_h, upper_h, lower_h)
        if not all(np.isfinite(v) for v in values):
            raise ValueError(f"prism_h, upper_h, lower_h must be finite, got {values!r}")
        prism_h, upper_h, lower_h = (abs(float(v)) for v in values)

        def side_c_over_a(indices: Sequence[Real], wedge_deg: float | None) -> float | None:
            if wedge_deg is None:
                return miller_indices_to_c_over_a(indices)
            if not wedge_angle_is_legal(float(wedge_deg)):
                return None
            return wedge_angle_to_c_over_a(float(wedge_deg))

        upper_c = side_c_over_a(upper_indices, upper_wedge_deg)
        lower_c = side_c_over_a(lower_indices, lower_wedge_deg)
        fd = tuple(float(f) for f in face_distance)
        shape = PyramidShape(prism_h, upper_h, lower_h, upper_c, lower_c, fd)
        upper = None if upper_c is None or upper_h <= LUMICE_FLOAT_EPS else (upper_c, upper_h)
        lower = None if lower_c is None or lower_h <= LUMICE_FLOAT_EPS else (lower_c, lower_h)
        h = 2.0 * float(a) * prism_h if prism_h > LUMICE_FLOAT_EPS else 0.0
        if upper is None and lower is None and h == 0.0:
            raise ValueError("a pyramid with no cone on either side and no prism band has zero volume "
                             "(Lumice doc/configuration.md §Pyramid Shape Legality)")
        if (upper is not None and upper == lower and h > 0.0 and fd == REGULAR_FACE_DISTANCE
                and upper_h < 1.0 and not _apex_snaps(fd, 1.0, upper_h)):
            crystal = cls(a=a, h=h, c_over_a=upper_c, tip_ratio=upper_h)
        else:
            crystal = cls._from_cones(float(a), h, fd, upper, lower)
        crystal.shape = shape
        return crystal

    @classmethod
    def _from_cones(cls, a: float, h: float, fd: tuple[float, ...],
                    upper: tuple[float, float] | None, lower: tuple[float, float] | None) -> "Pyramid":
        """闭式侵蚀构造：``upper`` / ``lower`` 为 ``(c_over_a, 高度比例)`` 或 ``None``（无锥），``h`` 为棱柱段全高（可为 0）。

        顶点全部由结构决定、不做顶点搜索或凸包：``m = 0`` 的截面环（肩部，``h = 0`` 时上下共用 ``z = 0``）、
        :func:`.closed_form.cone_sweep` 的每个角点死亡事件（一点）、截顶环或自然锥顶（点 / 棱脊两端）。锥面 ``i``
        的多边形由它起点角 ``(prev, i)`` 与终点角 ``(i, next)`` 两条轨迹上的顶点链围成：邻面死亡时轨迹在死亡点
        折向新邻面，自身死亡时两链在死亡点汇合。侵蚀量 ``m`` 处的高度为 ``±(h/2 + a·c_over_a·m)``（正六边形时
        ``m = 1`` 即理论锥顶 ``a·c_over_a``，与 ``__init__`` 同一关系）。
        """
        sweep = cone_sweep(a, fd)
        present = sweep.present
        n = len(present)
        tol_m = PRESENT_REL_TOL * float(np.max(np.abs(fd)))
        index: dict[tuple, int] = {}
        points: list[np.ndarray] = []

        def vid(key: tuple, xy: np.ndarray, z: float) -> int:
            if key not in index:
                index[key] = len(points)
                points.append(np.array([xy[0], xy[1], z]))
            return index[key]

        def shoulder(level: str, z: float, p: int, q: int) -> int:
            return vid((level, p, q), inset_corner(a, fd, p, q, 0.0), z)

        levels = {+1: "mid" if h == 0.0 else "top0", -1: "mid" if h == 0.0 else "bottom0"}
        faces: list[Face] = []
        if h > 0.0:
            for k, i in enumerate(present):
                p, q = present[k - 1], present[(k + 1) % n]
                # 从外侧看逆时针：底-左、底-右、顶-右、顶-左（同 HexPrism）
                faces.append(Face(3 + i, (shoulder("bottom0", -h / 2, p, i), shoulder("bottom0", -h / 2, i, q),
                                          shoulder("top0", h / 2, i, q), shoulder("top0", h / 2, p, i))))
        for sign, cone, basal, numbers in ((+1, upper, BASAL_TOP, UPPER_PYRAMID_FACES),
                                           (-1, lower, BASAL_BOTTOM, LOWER_PYRAMID_FACES)):
            level, z0 = levels[sign], sign * h / 2
            if cone is None:
                ring = [shoulder(level, z0, present[k - 1], present[k]) for k in range(n)]
                faces.append(Face(basal, tuple(ring if sign > 0 else ring[::-1])))
                continue
            c_over_a, fraction = cone
            m_top = min(fraction, 1.0) * sweep.m_apex
            collapsed = _apex_snaps(fd, sweep.m_apex, m_top)
            if collapsed:
                m_top = sweep.m_apex

            def z(m: float, sign: int = sign, c_over_a: float = c_over_a) -> float:
                return sign * (h / 2 + a * c_over_a * m)

            starts = {i: [shoulder(level, z0, present[k - 1], i)] for k, i in enumerate(present)}
            ends = {i: [shoulder(level, z0, i, present[(k + 1) % n])] for k, i in enumerate(present)}
            alive = list(present)
            for number, event in enumerate(sweep.events):
                if not collapsed and event.m > m_top + tol_m:
                    break
                if not collapsed and event.m >= m_top - tol_m:   # dies on the truncation plane: a top-ring corner
                    key = vid((sign, "top", event.before, event.after),
                              inset_corner(a, fd, event.before, event.after, m_top), z(m_top))
                else:
                    key = vid((sign, "event", number), event.xy, z(event.m))
                ends[event.before].append(key)
                starts[event.after].append(key)
                for i in event.dying:
                    starts[i].append(key)
                    ends[i].append(key)
                alive = [i for i in alive if i not in event.dying]
            if collapsed:
                if tuple(alive) != sweep.apex_present:
                    raise AssertionError(f"cone sweep apex faces {sweep.apex_present} != {tuple(alive)}")
                corners = [vid((sign, "apex", j), sweep.apex_points[j], z(m_top)) for j in sweep.apex_corner]
            else:
                corners = [vid((sign, "top", alive[k - 1], alive[k]),
                               inset_corner(a, fd, alive[k - 1], alive[k], m_top), z(m_top))
                           for k in range(len(alive))]
                faces.append(Face(basal, tuple(corners if sign > 0 else corners[::-1])))
            for k, i in enumerate(alive):
                starts[i].append(corners[k])
                ends[i].append(corners[(k + 1) % len(alive)])
            for i in present:
                # 从外侧看逆时针（上锥）：起点角肩部、终点链上行、起点链下行；下锥是它的 z 镜像，环序反转
                ring = _drop_repeats([starts[i][0], *ends[i], *starts[i][:0:-1]])
                faces.append(Face(numbers[i], tuple(ring if sign > 0 else ring[::-1])))
        faces.sort(key=lambda f: f.number)
        obj = cls.__new__(cls)
        obj.a, obj.h, obj.c_over_a, obj.tip_ratio = a, h, None, None
        Polyhedron.__init__(obj, np.array(points), faces)
        return obj

    def _copy_with(self, vertices: np.ndarray,
                   faces: Iterable[Face] | None = None) -> "Pyramid":
        # 同 HexPrism._copy_with：绕开 __init__ 搬字段；新增构造参数时必须同步搬运
        obj = Pyramid.__new__(Pyramid)
        obj.a, obj.h, obj.c_over_a, obj.tip_ratio = self.a, self.h, self.c_over_a, self.tip_ratio
        obj.shape = self.shape
        Polyhedron.__init__(obj, vertices, self.faces if faces is None else faces)
        return obj


def _apex_snaps(fd: Sequence[float], m_apex: float, m_top: float) -> bool:
    """截顶 ``m_top`` 是否落在自然锥顶 ``m_apex`` 的吸附带内（:data:`APEX_SNAP_REL`，Lumice ``ApexCollapsedAt``）。"""
    return m_apex - m_top <= APEX_SNAP_REL * float(np.max(np.abs(np.asarray(fd, dtype=float) - m_top)))


def _drop_repeats(ids: list[int]) -> list[int]:
    """去掉环上相邻的重复顶点（自身死亡的锥面两链在死亡点汇合、点状锥顶两角重合）。"""
    return [v for k, v in enumerate(ids) if v != ids[k - 1]]
