"""六棱柱内反射生成的方向变换群 G（ch7「为什么恰好是十二个」）：12 个矩阵的权威表，及其闭包 / 共轭类 /
特征值分类 / 与 $R_z(\\theta)$ 对易判定——原 `07-reflection-group/code/check_group.py` 的计算上提到此，
图 3 / 图 5 / 图 6 的脚本与单测都从这里取数，不各抄一份。

约定（与 ch3 标准坐标、:class:`lumice_integral.geometry.HexPrism` 一致）：面 1 / 2 是 ±z 底面，
面 ``3+i`` 的外法向方位角为 ``i·60°``。每面的镜面反射矩阵 $S_n = I - 2nn^T$，$S_n = S_{-n}$，故 8 个面只有 4 面镜。

12 个元素按 ch3 工作笔记「平行光路矩阵归总」编号 #1–#12，每个分解为 $xy$ 部分（6 种：$e$、$R_z(\\pm120°)$、
$S_{90°}$、$S_{\\pm30°}$，下标是反射轴在 $xy$ 平面内的方位角）× $z$ 符号（±1，$-1$ 即乘以底面镜 $b=\\mathrm{diag}(1,1,-1)$）。
正文第 3 节：$S_{90°}$ 是面 3/6 的镜子，$S_{30°}$ 是面 5/8，$S_{-30°}$ 是面 4/7（不变线 = 面法向方位角 + 90°）。

Provenance：迁移自写作仓 ``halo_notes.math.reflection_group``（2026-09-24，task-symmetry-authority），公开名称 1:1
保留；本仓是 G 的 12 元权威表。
"""

from __future__ import annotations

import functools
import itertools
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

# ---- 基本矩阵 ------------------------------------------------------------------


def Rz(deg: float) -> np.ndarray:
    """绕 z 轴逆时针转 ``deg`` 度。"""
    t = np.radians(deg)
    return np.array([[np.cos(t), -np.sin(t), 0.0], [np.sin(t), np.cos(t), 0.0], [0.0, 0.0, 1.0]])


def refl(n: Sequence[float]) -> np.ndarray:
    """法向为 ``n`` 的镜面反射 $S_n = I - 2nn^T$。"""
    v = np.asarray(n, dtype=float)
    v = v / np.linalg.norm(v)
    return np.eye(3) - 2.0 * np.outer(v, v)


def sxy(phi_deg: float) -> np.ndarray:
    """关于 xy 平面内过原点、方位角 ``phi_deg`` 的直线的反射 $S_\\phi$（z 不变）。
    它等于法向方位角为 ``phi_deg − 90°`` 的竖直镜面 ``refl``。"""
    t = np.radians(phi_deg)
    return np.array([[np.cos(2 * t), np.sin(2 * t), 0.0], [np.sin(2 * t), -np.cos(2 * t), 0.0], [0.0, 0.0, 1.0]])


B = np.diag([1.0, 1.0, -1.0])          # 底面镜 b


def key(M: np.ndarray) -> tuple:
    """矩阵的可哈希指纹（六位小数）。"""
    return tuple(np.round(np.asarray(M, dtype=float), 6).flatten())


# ---- 面法向与镜子 ----------------------------------------------------------------

NORMALS: dict[int, np.ndarray] = {1: np.array([0.0, 0.0, 1.0]), 2: np.array([0.0, 0.0, -1.0])}
for _k in range(6):
    _a = np.radians(60.0 * _k)
    NORMALS[3 + _k] = np.array([np.cos(_a), np.sin(_a), 0.0])
MIRRORS: dict[int, np.ndarray] = {k: refl(v) for k, v in NORMALS.items()}


def closure(generators: Iterable[np.ndarray]) -> tuple[list[np.ndarray], list[int]]:
    """由生成元生成的群闭包（左乘展开），返回 ``(元素列表, 按字长的累计个数)``。"""
    gens = list(generators)
    G = [np.eye(3)]
    seen = {key(np.eye(3))}
    frontier = [np.eye(3)]
    growth = [1]
    while frontier:
        nxt = []
        for g in frontier:
            for m in gens:
                h = m @ g
                if key(h) not in seen:
                    seen.add(key(h))
                    G.append(h)
                    nxt.append(h)
        frontier = nxt
        growth.append(len(seen))
    return G, growth


GROUP, GROWTH = closure(MIRRORS.values())   # |G| = 12，按字长增长 [1, 5, 10, 12, 12]


# ---- 12 个元素的权威表 ---------------------------------------------------------------

# xy 部分：符号 → (矩阵, mathtext)。顺序 = 正文第 92 行图 3 的列序
XY_ORDER: tuple[str, ...] = ("e", "r+", "r-", "s90", "s30", "s-30")
XY_MATRIX: dict[str, np.ndarray] = {
    "e": np.eye(3), "r+": Rz(120.0), "r-": Rz(-120.0),
    "s90": sxy(90.0), "s30": sxy(30.0), "s-30": sxy(-30.0),
}
XY_LATEX: dict[str, str] = {
    "e": r"$e$", "r+": r"$R_z(+120^\circ)$", "r-": r"$R_z(-120^\circ)$",
    "s90": r"$S_{90^\circ}$", "s30": r"$S_{30^\circ}$", "s-30": r"$S_{-30^\circ}$",
}
# 反射的不变线（方位角）与它代表的一对相对侧面：不变线 = 面法向方位角 + 90°
MIRROR_FACES: dict[str, tuple[int, int]] = {"s90": (3, 6), "s30": (5, 8), "s-30": (4, 7)}

# 正文第 17–24 行的 6 行表：发表类型 → (柱晶取向晕名, 片晶取向晕名)，逐字照抄
HALO_NAMES: dict[int, tuple[str, str]] = {
    1: ("（以对日点和反日点为中心）弥散", "映幻日环"),
    2: ("（以太阳和映日点为中心）弥散", "幻日环"),
    3: ("特里克尔弧", "映 120° 幻日"),
    4: ("映偕日弧", "120° 幻日"),
    5: ("幻日环", "映日"),
    6: ("太阳原像", "太阳原像"),
}


@dataclass(frozen=True)
class Element:
    """G 的一个元素：ch3 编号、$xy$ 符号、$z$ 符号、发表类型、典型光路（ch3 笔记「典型光路」行）。矩阵按需合成。"""

    number: int
    xy: str
    z_sign: int
    published_type: int
    representative_paths: tuple[tuple[int, ...], ...]

    @property
    def matrix(self) -> np.ndarray:
        return (B if self.z_sign < 0 else np.eye(3)) @ XY_MATRIX[self.xy]

    @property
    def column_halo(self) -> str:
        return HALO_NAMES[self.published_type][0]

    @property
    def plate_halo(self) -> str:
        return HALO_NAMES[self.published_type][1]

    @property
    def commutes_with_rz(self) -> bool:
        """与 $R_z(\\theta)$ 对易 ⇔ $xy$ 部分是旋转（含单位）。"""
        return commutes_with_rz(self.matrix)

    @property
    def latex(self) -> str:
        """mathtext：$xy$ 符号，$z=-1$ 时前面加 $b$（如 ``$b\\,R_z(-120^\\circ)$``；度号用 ``^\\circ``，Computer Modern 没有「°」字形）。"""
        xy = XY_LATEX[self.xy]
        return xy if self.z_sign > 0 else "$b\\," + xy[1:]


def _paths(*specs: str) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(c) for c in s.split("-")) for s in specs)


ELEMENTS: tuple[Element, ...] = (
    Element(1, "s90", -1, 1, _paths("1-2-3-1", "3-6-1-3")),
    Element(2, "s90", +1, 2, _paths("3", "1-3-2", "3-5-6-7-3")),
    Element(3, "r-", -1, 3, _paths("1-2-3-5-1", "3-1-5-7-4")),
    Element(4, "r-", +1, 4, _paths("1-3-5-2", "3-5-7-4")),
    Element(5, "r+", -1, 3, _paths("1-2-3-4-1", "3-1-6-4-8")),
    Element(6, "r+", +1, 4, _paths("1-3-4-2", "3-6-4-8")),
    Element(7, "s-30", -1, 1, _paths("1-3-4-2-5-1", "3-1-4-5", "3-5-7-1-3-5")),
    Element(8, "s-30", +1, 2, _paths("3-4-5", "1-3-4-5-2", "3-5-7-3-5")),
    Element(9, "s30", -1, 1, _paths("1-3-2-5-7-1", "3-1-5-7")),
    Element(10, "s30", +1, 2, _paths("1-3-4-6-2", "3-5-7")),
    Element(11, "e", -1, 5, _paths("1", "1-2-1", "3-1-6")),
    Element(12, "e", +1, 6, _paths("1-2", "3-6")),
)
BY_NUMBER: dict[int, Element] = {e.number: e for e in ELEMENTS}
MATRICES: dict[int, np.ndarray] = {e.number: e.matrix for e in ELEMENTS}
PUBLISHED: dict[int, int] = {e.number: e.published_type for e in ELEMENTS}


def by_number(n: int) -> Element:
    return BY_NUMBER[n]


def by_xy_z(xy: str, z_sign: int) -> Element:
    """图 3 网格的格子 (列符号, 行符号) → 元素。"""
    for e in ELEMENTS:
        if e.xy == xy and e.z_sign == z_sign:
            return e
    raise KeyError((xy, z_sign))


def identify(M: np.ndarray) -> Element:
    """任意矩阵 → 权威表里的元素（按指纹查，不在表里抛 ``KeyError``）。"""
    k = key(M)
    for e in ELEMENTS:
        if key(e.matrix) == k:
            return e
    raise KeyError("matrix is not one of the 12 elements of G")


# ---- 光路 → 矩阵 ----------------------------------------------------------------


def path_matrix(faces: Sequence[int]) -> np.ndarray:
    """平行光路（首尾面平行，折射抵消）的方向变换矩阵：中间各内反射面的镜面矩阵按相遇顺序左乘；
    单面 = 一次外反射；``1-2`` 这类无反射 = 单位阵。"""
    faces = [int(f) for f in faces]
    if len(faces) == 1:
        return MIRRORS[faces[0]]
    M = np.eye(3)
    for f in faces[1:-1]:
        M = MIRRORS[f] @ M
    return M


def refraction_cancels(faces: Sequence[int]) -> bool:
    r"""「平行光路」的精确判据：出射面法向 $n_{out} = -M\,n_{in}$（$M$ = 内反射镜面连乘）。

    首尾面字面平行（同面或对面）只是 $M n_{in} = \pm n_{in}$ 的特例；ch3 笔记的典型光路里 ``3-1-5-7-4``、``3-4-5`` 等
    10 条首尾面并不平行，但入射 / 出射两次折射同样精确抵消（$M$ 是正交阵、把入射面法向搬到出射面法向的反向，
    Snell 定律在正交变换下不变），方向变换仍是常数矩阵 $M$——正文与 ch3 说的「首尾面平行、折射抵消」指的就是这个条件。
    """
    faces = [int(f) for f in faces]
    if len(faces) == 1:
        return True                                       # 单面外反射：$-S_n n = n$ 自动成立
    return bool(np.allclose(NORMALS[faces[-1]], -path_matrix(faces) @ NORMALS[faces[0]]))


def class_of(number: int) -> list[int]:
    """``number`` 所在共轭类的全部编号（按 :func:`conjugacy_classes`）。"""
    return next(c for c in conjugacy_classes() if number in c)


@functools.lru_cache(maxsize=None)
def raypaths_of_class(number: int, max_len: int = 6) -> tuple[tuple[int, ...], ...]:
    """长度 ≤ ``max_len``、相邻面不重复、折射抵消（:func:`refraction_cancels`）、$M$ 落在 ``number`` 所在**共轭类**内的
    全部面序列（纯代数枚举，几何可达性交给追迹 / Lumice）。按 (长度, 字典序) 排序。"""
    targets = {key(MATRICES[n]) for n in class_of(number)}
    out: list[tuple[int, ...]] = []

    def walk(prefix: tuple[int, ...], M: np.ndarray) -> None:      # M = prefix[1:] 各面镜子的连乘（入射面不算）
        n_in = NORMALS[prefix[0]]
        if len(prefix) == 1:
            if key(MIRRORS[prefix[0]]) in targets:
                out.append(prefix)
        elif key(M) in targets and np.allclose(NORMALS[prefix[-1]], -M @ n_in):
            out.append(prefix)
        if len(prefix) == max_len:
            return
        last = prefix[-1]
        M_next = MIRRORS[last] @ M if len(prefix) > 1 else M     # 当前末面若不出射就是一次内反射
        for f in NORMALS:
            if f != last:
                walk(prefix + (f,), M_next)

    for f0 in NORMALS:
        walk((f0,), np.eye(3))
    return tuple(sorted(out, key=lambda s: (len(s), s)))


_B_SWAP = {1: 2, 2: 1, **{13 + i: 23 + i for i in range(6)}, **{23 + i: 13 + i for i in range(6)}}
"""上下互换（B）：底面 1↔2；锥晶再加上锥面 13+i ↔ 下锥面 23+i；侧面不动。"""


def pbd_orbit(faces: Sequence[int]) -> set[tuple[int, ...]]:
    """面序列在 Lumice ``symmetry: "PBD"`` 下的全部像：D6（P 旋转 + D 翻转，:mod:`lumice_integral.symmetry.group`）× 上下互换（B）。
    锥晶面号（13–18 / 23–28）同样适用：D6 逐环带作用，B 把上下锥面对调。"""
    from . import group as D6
    faces = tuple(int(f) for f in faces)
    out: set[tuple[int, ...]] = set()
    for g in D6.elements():
        im = tuple(D6.apply_path(g, faces))
        out.add(im)
        out.add(tuple(_B_SWAP.get(f, f) for f in im))
    return out


def representatives_under_pbd(seqs: Iterable[Sequence[int]]) -> tuple[tuple[int, ...], ...]:
    """按 PBD 轨道去重，每个轨道取字典序最小的序列作代表；输出按 (长度, 字典序) 排序。"""
    reps = {min(pbd_orbit(s)) for s in seqs}
    return tuple(sorted(reps, key=lambda s: (len(s), s)))


# ---- 分类 ------------------------------------------------------------------------


def conjugacy_classes(group: Sequence[np.ndarray] = GROUP) -> list[list[int]]:
    """按 ``group``（默认 G 自身；晶体对称操作给出同样的划分）做共轭得到的类，每类按编号升序，类按首元素排序。"""
    classes: dict[frozenset, list[int]] = {}
    for e in ELEMENTS:
        cls = frozenset(key(g @ e.matrix @ g.T) for g in group)
        classes.setdefault(cls, []).append(e.number)
    return sorted(classes.values())


def published_classes() -> list[list[int]]:
    """发表的 6 类（按 ``published_type`` 分组），与 :func:`conjugacy_classes` 同一排序。"""
    return sorted(sorted(e.number for e in ELEMENTS if e.published_type == t) for t in range(1, 7))


def eigenvalue_classes() -> list[list[int]]:
    """按特征值（O(3) 共轭不变量）分类：只有 5 类，#11 并入 #2/8/10。"""
    ev: dict[tuple, list[int]] = {}
    for e in ELEMENTS:
        ev.setdefault(tuple(np.round(np.sort_complex(np.linalg.eigvals(e.matrix)), 6)), []).append(e.number)
    return sorted(ev.values())


def commutes_with_rz(M: np.ndarray, probe_deg: float = 37.0) -> bool:
    """与 $R_z(\\theta)$ 对易（用一个非特殊角探针：对易性对所有 θ 同时成立或同时不成立）。"""
    R = Rz(probe_deg)
    return bool(np.allclose(M @ R, R @ M))


def commuting_with_rz() -> list[int]:
    """与 $R_z(\\theta)$ 对易的元素编号：{3, 4, 5, 6, 11, 12}（四锐两散）。"""
    return [e.number for e in ELEMENTS if e.commutes_with_rz]


def wedge_cosines(group: Sequence[np.ndarray] = GROUP) -> list[float]:
    """$\\langle n_a, M^{-1} n_b\\rangle$ 的全部可能取值（楔角余弦）。"""
    out = set()
    for M in group:
        for a, b in itertools.product(NORMALS, NORMALS):
            out.add(round(float(NORMALS[a] @ (M.T @ NORMALS[b])), 6))
    return sorted(out)


__all__ = ["B", "BY_NUMBER", "ELEMENTS", "Element", "GROUP", "GROWTH", "HALO_NAMES", "MATRICES", "MIRRORS",
           "MIRROR_FACES", "NORMALS", "PUBLISHED", "XY_LATEX", "XY_MATRIX", "XY_ORDER", "Rz", "by_number",
           "by_xy_z", "class_of", "closure", "commutes_with_rz", "commuting_with_rz", "conjugacy_classes",
           "eigenvalue_classes", "identify", "key", "path_matrix", "pbd_orbit", "published_classes", "raypaths_of_class",
           "refl", "refraction_cancels", "representatives_under_pbd", "sxy", "wedge_cosines"]
