"""光路 signature $(M,\\mathbf n_a,M^{-1}\\mathbf n_b)$ 的规范型与方向映射 $\\Phi$ 的采样指纹（ch8「三种楔子、十二种
折叠」）：原 ``08-wedge-times-fold/code/check_signature.py`` 内联的算法上提到此，该脚本改为薄封装调库；枚举器
（:mod:`lumice_integral.geometry.enumerate`）吐出的每条可行光路经 :func:`phi_class` 落到同一套类里，「上界怎么数的」与
「下界怎么判到哪一类」共用这一份规范化代码。

约定：光路 ``faces = [a, m_1..m_k, b]``，``M`` 为内反射镜面连乘（:func:`lumice_integral.geometry.unfold.fold_matrix`），
``a``/``b`` 为入射 / 出射面编号；面法向**只从传入的** ``crystal``（:class:`lumice_integral.geometry.Polyhedron`）取，
本模块不读任何模块级面表常量，锥晶等更一般的多面体走同一条路。商掉的对称群默认 :data:`D6H`（六方晶系点群，24 元，
元素矩阵由 :mod:`lumice_integral.symmetry.reflection_group` 的 ``Rz``/``sxy``/``B`` 拼出，不另写一份；全仓唯一一份 D6h
元素表，``path_class.hexprism_symmetry_matrices`` 直接返回它）——它必须是
``crystal`` 的对称群（把面法向集合映到自身），否则 :func:`face_of` 抛 ``ValueError``。

$\\Phi=\\mathcal S_{\\mathbf n_b}\\circ M\\circ\\mathcal S_{\\mathbf n_a}$ 的**定义域是判据的一部分**：入射须从外侧进面 $a$
（$\\langle\\mathbf d,\\mathbf n_a\\rangle<0$），出射须无 TIR 且从外侧离开面 $b$。定义域判据与可行性判定
（:func:`lumice_integral.geometry.feasibility.entry_ok` / :func:`~lumice_integral.geometry.feasibility.exit_ok`）是同一份实现：
两者都作用在**晶体内**方向上，$\\tilde{\\mathbf n}_b=M^{-1}\\mathbf n_b$。漏掉入射侧判据会把 60° 的 12 个 signature 类
错并成 6 个（一条光路与其时间反演在指纹上无法区分）。

楔角：:func:`wedge_angle` 与 :func:`lumice_integral.geometry.wedge_angle_deg` 共用同一份数值实现
（``geometry.unfold._wedge_angle_deg_from_normals``，``atan2`` 形式；写作仓原文是 ``arccos``，在 0° 附近丢 ~√ε）。

依赖方向：``symmetry → geometry`` 单向（``tests/test_symmetry_dependency_direction.py`` 用 ``ast`` 静态核验
``lumice_integral.geometry`` 不 import ``lumice_integral.symmetry``）。若未来 geometry 侧需要引用本模块（例如对
``RaypathRecord`` 直接分类），会构成循环依赖——届时应把共享的 snell / 临界角判据下沉到两者都能安全依赖的更底层位置，
而不是两侧各写一份判据。

Provenance：迁移自写作仓 ``halo_notes.math.signature``（2026-09-24，task-symmetry-authority），公开名称 1:1 保留；
唯一的实现改动是 :func:`wedge_angle` 改调上述共享楔角实现。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from ..geometry.core import N_ICE, Polyhedron
from ..geometry.feasibility import entry_ok, exit_ok
from ..geometry.unfold import _wedge_angle_deg_from_normals
from . import reflection_group as rg

A_MAX_DEG = 2 * math.degrees(math.asin(1 / N_ICE))
"""可透射楔角上限 $2\\arcsin(1/n)\\approx99.5°$，超过则楔角全反射（TIR 死角）。"""

# ---- D6h：ch7 的 D3h（G）不足以商 signature——法向方位角本身以 60° 为周期，还差绕 z 轴的旋转
# 与竖直镜；D6h = <Rz(60k), sxy(30k)> ∪ B·(前者)，共 24 元。
D6H: tuple[np.ndarray, ...] = tuple(
    g for k in range(6) for g in (rg.Rz(60 * k), rg.sxy(30 * k))
)
D6H = D6H + tuple(rg.B @ g for g in D6H)
assert len({rg.key(g) for g in D6H}) == 24, "D6h 应有 24 个元素"

Signature = tuple[tuple, int, int]
"""规范 signature：``(key(M), a, b)``——``M`` 的六位小数指纹（:func:`lumice_integral.symmetry.reflection_group.key`）与入 / 出面编号。"""

_rng = np.random.default_rng(0)  # 固定种子，保证指纹可复现
_SAMPLE = _rng.normal(size=(600, 3))
_SAMPLE /= np.linalg.norm(_SAMPLE, axis=1, keepdims=True)
"""指纹用的 600 个单位方向（固定种子 0）。"""


# ---- 面法向查表 ------------------------------------------------------------------


def _normal_table(crystal: Polyhedron) -> tuple[list[int], np.ndarray]:
    """``(面编号列表, 对应外法向 (F, 3))``。"""
    faces = list(crystal.faces)
    return [f.number for f in faces], np.array([crystal.normal(f) for f in faces])


def _face_lookup(numbers: list[int], normals: np.ndarray, v: np.ndarray, tol: float = 1e-6) -> int:
    i = int(np.argmin(np.linalg.norm(normals - v, axis=1)))
    if np.linalg.norm(normals[i] - v) > tol:
        raise ValueError(f"{v} is not a face normal of the crystal")
    return numbers[i]


def face_of(crystal: Polyhedron, v: np.ndarray) -> int:
    """法向量 ``v`` 对应的 ``crystal`` 面编号（最近法向，差范数 ≤ 1e-6）；不是任何面的法向抛 ``ValueError``。"""
    return _face_lookup(*_normal_table(crystal), np.asarray(v, dtype=float))


# ---- 楔角与族 ----------------------------------------------------------------------


def wedge_angle(crystal: Polyhedron, M: np.ndarray, a: int, b: int) -> float:
    """楔角 $A=\\arccos(-\\langle\\mathbf n_a, M^{-1}\\mathbf n_b\\rangle)$（度）。$M$ 正交，$M^{-1}=M^T$。
    数值上按 ``atan2`` 形式求（与 :func:`lumice_integral.geometry.wedge_angle_deg` 同一份实现）。"""
    n_a, n_b = crystal.normal(crystal.face(a)), crystal.normal(crystal.face(b))
    return _wedge_angle_deg_from_normals(n_a, -(np.asarray(M, dtype=float).T @ n_b))


FAMILY_NAMES: dict[float, str] = {0.0: "平行", 60.0: "22° 族", 90.0: "46° 族"}
"""六棱柱三个可透射楔角的族名（ch8 正文用词）。"""


def family(angle_deg: float, tol: float = 0.05) -> str | None:
    """楔角 → 族名：0° → 平行、60° → 22° 族、90° → 46° 族。其余角度（120°/180° 的 TIR 死角，以及锥晶的 28.0°、
    52.4° 等六棱柱之外的楔角）返回 ``None``——族名是六棱柱的叫法，可透射与否请另用 :data:`A_MAX_DEG` 判。"""
    for a, name in FAMILY_NAMES.items():
        if abs(angle_deg - a) <= tol:
            return name
    return None


# ---- signature 规范型 ------------------------------------------------------------


def canonical_signature(crystal: Polyhedron, M: np.ndarray, a: int, b: int,
                        group: Sequence[np.ndarray] = D6H) -> Signature:
    """$(M,\\mathbf n_a,\\mathbf n_b)$ 在 ``group`` 下的规范代表：对每个 $g$ 取 $(gMg^T,\\ g\\mathbf n_a,\\ g\\mathbf n_b)$
    的字典序最小者（面用编号表示）。"""
    n_a, n_b = crystal.normal(crystal.face(a)), crystal.normal(crystal.face(b))
    numbers, normals = _normal_table(crystal)
    best = None
    for g in group:
        t = (rg.key(g @ M @ g.T), _face_lookup(numbers, normals, g @ n_a), _face_lookup(numbers, normals, g @ n_b))
        if best is None or t < best:
            best = t
    return best


# ---- 方向映射 Φ 与采样指纹 ----------------------------------------------------------


def _refract(d: np.ndarray, n: np.ndarray, ratio: float) -> np.ndarray:
    """Snell 折射的向量形式（批量，``d (N, 3)``，``n`` 为界面外法向，``ratio = n_1/n_2``），只算方向、不判定义域：
    $\\mathbf d' = r\\mathbf d + (rc - \\sqrt{1 - r^2(1-c^2)})\\,\\mathbf n$，$c=-\\langle\\mathbf n,\\mathbf d\\rangle$。
    根号内为负（TIR）的行给 ``NaN``。定义域由调用方用 :func:`~lumice_integral.geometry.feasibility.entry_ok` /
    :func:`~lumice_integral.geometry.feasibility.exit_ok` 判。"""
    c = -(d @ n)
    k = 1.0 - ratio * ratio * (1.0 - c * c)
    root = np.sqrt(np.where(k >= 0, k, np.nan))
    return ratio * d + (ratio * c - root)[:, None] * n


def phi_batch(crystal: Polyhedron, M: np.ndarray, a: int, b: int, d: np.ndarray) -> np.ndarray:
    """:func:`phi` 的批量版：``d (N, 3)`` → ``(N, 3)``，定义域外的行为 ``NaN``。"""
    n_a, n_b = crystal.normal(crystal.face(a)), crystal.normal(crystal.face(b))
    d = np.asarray(d, dtype=float).reshape(-1, 3)
    out = np.full_like(d, np.nan)
    outside = (d @ n_a) < 0                                 # 从外侧进面 a
    if not outside.any():
        return out
    d1 = _refract(d[outside], n_a, 1.0 / N_ICE)             # 晶体内方向
    ok = entry_ok(n_a, d1) & exit_ok(M.T @ n_b, d1)         # 与可行性判定同一份光学判据（作用在晶体内方向上）
    d3 = _refract((M @ d1[ok].T).T, -n_b, N_ICE)            # 从内侧出射：法向取指向内部的 -n_b
    rows = np.flatnonzero(outside)[ok]
    out[rows] = d3
    return out


def phi(crystal: Polyhedron, M: np.ndarray, a: int, b: int, d: np.ndarray) -> np.ndarray | None:
    """$\\Phi(\\mathbf d) = \\mathcal S_{\\mathbf n_b}\\circ M\\circ\\mathcal S_{\\mathbf n_a}$，单个外部入射方向 ``d``；定义域外返回 ``None``。"""
    o = phi_batch(crystal, M, a, b, np.asarray(d, dtype=float).reshape(1, 3))[0]
    return None if np.isnan(o).any() else o


def phi_fingerprint(crystal: Polyhedron, M: np.ndarray, a: int, b: int,
                    group: Sequence[np.ndarray] = D6H) -> tuple:
    """$\\Phi$ 在 ``group`` 共轭下（$\\Phi_g(\\mathbf d) = g\\,\\Phi(g^{-1}\\mathbf d)$）的规范化采样指纹：对固定种子的 600 个
    方向逐个求值（三位小数，定义域外记 ``(9, 9, 9)``），取全部 $g$ 里字典序最小的一份。指纹相同 ⇔ 两条光路的方向映射
    （含定义域）在晶体对称下等价。"""
    best = None
    for g in group:
        o = phi_batch(crystal, M, a, b, _SAMPLE @ g)          # 行向量：g^T d ⇔ d @ g
        o = np.where(np.isnan(o), 9.0, np.round(o @ g.T, 3))  # g·o
        t = tuple(map(tuple, o))
        if best is None or t < best:
            best = t
    return best


# ---- Φ 类 --------------------------------------------------------------------------


@dataclass(frozen=True)
class PhiClass:
    """一条光路的 $\\Phi$ 类：楔角（一位小数）+ 指纹。相等 / 哈希只看这两项——同一 ``crystal`` 上任意两条方向映射在
    晶体对称下等价的光路给出相等的 :class:`PhiClass`，与运行顺序无关。``signature`` 是产生它的规范 signature（0° 时
    多个 signature 类并入同一 $\\Phi$ 类，这一项只是「本次算到的那一个」，不参与相等比较）。"""

    wedge_deg: float
    fingerprint: tuple = field(repr=False)
    signature: Signature = field(compare=False)

    @property
    def family(self) -> str | None:
        return family(self.wedge_deg)


def _crystal_key(crystal: Polyhedron) -> tuple:
    """按面法向（六位小数）给晶体一个可哈希键：同一套法向（如不同高径比的六棱柱）共享指纹缓存。"""
    return tuple((f.number, rg.key(crystal.normal(f))) for f in crystal.faces)


_FINGERPRINTS: dict[tuple, tuple] = {}
"""指纹缓存：``(晶体法向表键, 规范 signature) → 指纹``。"""


def fold_fingerprint(M: np.ndarray, group: Sequence[np.ndarray] = D6H) -> tuple:
    """平行光路（楔角 0°）用的「指纹」：$M$ 在 ``group`` 共轭下的规范代表 ``("fold", min_g key(gMg^T))``。

    平行光路 $W=I$、$\\Phi=M$ 是常数矩阵，ch7 / framework 定理 5′ 把它的类取为 $M$ 的共轭类（六棱柱 6 类），**不含定义域**：
    从哪个面进（``1-2`` 还是 ``3-6``）只决定亮哪一段，归 signature 管、$\\Phi$ 商掉——这是 14 → 6 的来由。若对 0° 也用带
    定义域的 :func:`phi_fingerprint`，14 个 signature 类一个都不并（定义域两两不同），与既定的 34 = 6 + 12 + 16 不符；本函数
    就是 ``check_signature.py`` 里「0° 按 ``identify(M).number`` 分组」那一步的库化，前缀 ``"fold"`` 保证不与采样指纹撞键。"""
    return ("fold", min(rg.key(g @ M @ g.T) for g in group))


def phi_class(crystal: Polyhedron, M: np.ndarray, a: int, b: int) -> PhiClass:
    """光路 ``(M, a, b)`` 所属的 $\\Phi$ 类：楔角 0° 用 :func:`fold_fingerprint`（$M$ 的共轭类），其余楔角用带定义域的
    :func:`phi_fingerprint`——与 ``check_signature.py`` / framework 定理 5′ 的判据逐字一致（六棱柱 6 + 12 + 16 = 34）。

    指纹按 ``(晶体法向表, 规范 signature)`` 缓存：可能出现的规范 signature 是有限的（六棱柱可透射 42 个 + TIR 死角 26 个），
    不随枚举光路数增长，24×600 次求值的指纹每个 signature 类只算一次；规范 signature 本身只要 24 次矩阵共轭 + 查表，
    逐条光路算得起。"""
    sig = canonical_signature(crystal, M, a, b)
    wedge = round(wedge_angle(crystal, M, a, b), 1)
    if wedge == 0.0:
        return PhiClass(wedge, fold_fingerprint(M), sig)
    cache_key = (_crystal_key(crystal), sig)
    fp = _FINGERPRINTS.get(cache_key)
    if fp is None:
        k, ca, cb = sig
        fp = _FINGERPRINTS[cache_key] = phi_fingerprint(crystal, np.array(k).reshape(3, 3), ca, cb)
    return PhiClass(wedge, fp, sig)


# ---- 六棱柱全类表（上界的全部格子，含不可达） -----------------------------------------------


@dataclass(frozen=True)
class PhiRow:
    """全类表的一行：``id`` 形如 ``A60-07``（楔角 + 该楔角内 1 起的序号，排序确定、与运行顺序无关），``folds`` 是该类
    里出现的折叠矩阵在 ch7 权威表里的编号（``M`` 不在 12 元表里时为空），``signatures`` / ``face_pairs`` 列出并入该类的
    全部规范 signature 及其入出面对（0° 类多对一，60°/90° 类一对一）。"""

    id: str
    wedge_deg: float
    signatures: tuple[Signature, ...]
    folds: tuple[int, ...]
    phi_class: PhiClass

    @property
    def family(self) -> str | None:
        return family(self.wedge_deg)

    @property
    def transmittable(self) -> bool:
        return self.wedge_deg < A_MAX_DEG

    @property
    def face_pairs(self) -> tuple[tuple[int, int], ...]:
        return tuple((a, b) for _, a, b in self.signatures)


def _fold_number(M: np.ndarray) -> int | None:
    try:
        return rg.identify(M).number
    except KeyError:
        return None


def phi_class_table(crystal: Polyhedron, fold_matrices: Sequence[np.ndarray] = rg.GROUP) -> tuple[PhiRow, ...]:
    """遍历 ``fold_matrices`` × 全部面对 ``(a, b)``，按 $\\Phi$ 类聚成全类表（含 TIR 死角的类，``transmittable`` 区分）。
    六棱柱默认 ``fold_matrices = G``（12 元），得 34 个可透射类，外加 120°/180° 两个死角类（定义域为空、指纹退化，
    同楔角的 signature 类全并成一类）。行序：楔角升序，同楔角内按 ``(min folds, signatures)`` 升序，序号由此而定。"""
    faces = [f.number for f in crystal.faces]
    by_class: dict[PhiClass, dict] = {}
    for M in fold_matrices:
        for a in faces:
            for b in faces:
                cls = phi_class(crystal, M, a, b)
                entry = by_class.setdefault(cls, {"sigs": set(), "folds": set()})
                entry["sigs"].add(cls.signature)
                n = _fold_number(M)
                if n is not None:
                    entry["folds"].add(n)
    rows = []
    for cls, entry in by_class.items():
        sigs, folds = tuple(sorted(entry["sigs"])), tuple(sorted(entry["folds"]))
        rows.append((cls.wedge_deg, folds[0] if folds else math.inf, sigs, folds, cls))
    rows.sort(key=lambda r: r[:3])
    out, counter = [], {}
    for wedge, _, sigs, folds, cls in rows:
        counter[wedge] = counter.get(wedge, 0) + 1
        out.append(PhiRow(f"A{wedge:g}-{counter[wedge]:02d}", wedge, sigs, folds, cls))
    return tuple(out)


__all__ = ["A_MAX_DEG", "D6H", "FAMILY_NAMES", "PhiClass", "PhiRow", "Signature", "canonical_signature", "face_of",
           "family", "fold_fingerprint", "phi", "phi_batch", "phi_class", "phi_class_table", "phi_fingerprint", "wedge_angle"]
