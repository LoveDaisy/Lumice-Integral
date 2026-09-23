"""二面体群 D6 在六方柱侧面编号 3–8 上的置换表示（ch1「迷镜千幻不离宗」§群表）。

元素与记号（正文第 63 行）：旋转 $R^c_k$（k=0..5，$R^c_0=I$，左转 k·60°）、过面心轴翻转 $F^f_k$（k=3,4,5，
轴穿过面 k 与对面 k+3 的中点）、过顶点轴翻转 $F^d_{k,k+1}$（k=3,4,5，轴穿过面 k、k+1 的公共顶点）。
:func:`elements` 的顺序 = 已发表群表 1.7 的行序：I, R1..R5, F^f_3, F^d_{3,4}, F^f_4, F^d_{4,5}, F^f_5, F^d_{5,6}。

作用公式（面编号 n=3..8 按 mod 6 循环，底面 1、2 不动）：

    R^c_k(n)       ≡ n + k          (mod 6)          —— 正文第 43 行
    F^f_k(n)       ≡ 2k − n         (mod 6)
    F^d_{k,k+1}(n) ≡ 2k + 1 − n     (mod 6)

统一写法：把 6 个翻转按 :func:`elements` 顺序编号 m=0..5，则 F_m(n) ≡ m − n (mod 6)——相邻两个翻转轴相差
30°，F_m 的轴在 F_0（过面 3 中点）基础上转 m·30°。

正文第 76 行公式（`01-symmetry/迷镜千幻不离宗.md`）已改为与本模块一致（chore-figures-followup 改正：
原文 F^f_k(n) ≡ −n−k、F^d_{k,k+1}(n) ≡ −n+k 逐字套用会得到 F^f_3 ≡ F^d_{4,5} 重合、只剩 5 个不同翻转，
不能构成 D6，与已发表的图 1.6 及群表 1.7 矛盾）。

Provenance：迁移自写作仓 ``halo_notes.math.group``（2026-09-24，task-symmetry-authority），公开名称 1:1 保留；
本仓是这套置换表示的权威实现。

合成约定（正文第 39 行）：``compose(a, b)`` = 先做 a 再做 b，即作为映射是 ``n ↦ b(a(n))``；
``multiplication_table(elems)[i][j] == compose(elems[i], elems[j])``——"表 i 行 j 列 = 先 i 后 j"。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

SIDE_FACES = (3, 4, 5, 6, 7, 8)


@dataclass(frozen=True, order=True)
class Element:
    """D6 元素：``flip=False`` 时 n ↦ n + shift，``flip=True`` 时 n ↦ shift − n（面编号 mod 6）。"""

    flip: bool
    shift: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "shift", int(self.shift) % 6)

    @property
    def name(self) -> str:
        """纯文本名：``I`` / ``R1`` / ``Ff3`` / ``Fd34``。"""
        if not self.flip:
            return "I" if self.shift == 0 else f"R{self.shift}"
        k = 3 + self.shift // 2
        return f"Ff{k}" if self.shift % 2 == 0 else f"Fd{k}{k + 1}"

    def __repr__(self) -> str:
        return self.name


def R(k: int) -> Element:
    """旋转 $R^c_k$。"""
    return Element(False, k)


def Ff(k: int) -> Element:
    """过面 k 中点轴的翻转 $F^f_k$（k=3,4,5）。"""
    if k not in (3, 4, 5):
        raise ValueError(f"F^f_k requires k in 3..5, got {k}")
    return Element(True, 2 * (k - 3))


def Fd(k: int) -> Element:
    """过面 k、k+1 公共顶点轴的翻转 $F^d_{k,k+1}$（k=3,4,5）。"""
    if k not in (3, 4, 5):
        raise ValueError(f"F^d_{{k,k+1}} requires k in 3..5, got {k}")
    return Element(True, 2 * (k - 3) + 1)


IDENTITY = R(0)


def elements() -> list[Element]:
    """全部 12 个元素，顺序 = 群表 1.7 行序（前 6 个为 C6 子群）。"""
    return [R(k) for k in range(6)] + [Element(True, m) for m in range(6)]


PYRAMID_BANDS = {3: SIDE_FACES, 13: tuple(range(13, 19)), 23: tuple(range(23, 29))}
"""D6 作用的三个面号环带：棱柱侧面 3–8、上锥面 13–18、下锥面 23–28（锥晶面号见 ``lumice_integral.geometry.Pyramid``，
锥面 ``13+i`` / ``23+i`` 的方位角与侧面 ``3+i`` 相同，所以同一置换公式逐环带套用）。"""


def _band(face: int) -> int:
    for base, band in PYRAMID_BANDS.items():
        if face in band:
            return base
    raise ValueError(f"face number must be 1..8, 13..18 or 23..28, got {face}")


def apply(g: Element, face: int) -> int:
    """g 作用于面编号：底面 1、2 不动；侧面 3–8（以及锥晶的 13–18 / 23–28，各自环带内）按模块 docstring 的公式循环。"""
    if face in (1, 2):
        return face
    base = _band(face)
    i = face - base                                   # 环带内序号 0..5，与侧面 3+i 同方位角
    x = g.shift - (3 + i) if g.flip else (3 + i) + g.shift
    return base + (x - 3) % 6


def apply_path(g: Element, faces: Sequence[int]) -> list[int]:
    """g 逐面作用于一条光路（面序列），如 R^c_2: 1-5-2-7-1 → 1-7-2-3-1（正文第 45 行）。"""
    return [apply(g, f) for f in faces]


def compose(a: Element, b: Element) -> Element:
    """先做 a 再做 b（n ↦ b(a(n))）。结果由置换表查回元素，不靠代数化简。"""
    perm = tuple(apply(b, apply(a, n)) for n in SIDE_FACES)
    for g in elements():
        if tuple(apply(g, n) for n in SIDE_FACES) == perm:
            return g
    raise AssertionError(f"composition of {a} and {b} left D6: {perm}")  # pragma: no cover


def inverse(g: Element) -> Element:
    """逆元：翻转是对合，旋转 R_k 的逆是 R_{−k}。"""
    return g if g.flip else R(-g.shift)


def multiplication_table(elems: Sequence[Element]) -> list[list[Element]]:
    """``table[i][j] = compose(elems[i], elems[j])``——表 i 行 j 列 = 先 i 后 j。"""
    return [[compose(a, b) for b in elems] for a in elems]


def latex(g: Element) -> str:
    """mathtext 记号：``$I$``、``$R^c_1$``、``$F^f_3$``、``$F^d_{3,4}$``。"""
    if not g.flip:
        return "$I$" if g.shift == 0 else f"$R^c_{g.shift}$"
    k = 3 + g.shift // 2
    return f"$F^f_{k}$" if g.shift % 2 == 0 else f"$F^d_{{{k},{k + 1}}}$"


__all__ = ["Element", "Fd", "Ff", "IDENTITY", "R", "SIDE_FACES", "apply", "apply_path", "compose", "elements",
           "inverse", "latex", "multiplication_table"]
