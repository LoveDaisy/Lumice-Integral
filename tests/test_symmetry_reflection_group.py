"""``lumice_integral.symmetry.reflection_group``（迁移自写作仓 ``halo_notes/math/tests/test_reflection_group.py``；
改 import，两份正文转录经 ``tests/_writing_repo.py`` 只读）。12 矩阵权威表：原 ``check_group.py`` 的 5 条断言逐条搬来，外加权威表与正文 6 行表 / ch3 笔记典型光路 / D6 置换表示的
交叉核对（两个模块对"同一个对称操作"的约定必须互相印证，见 ch7 图 6 的过滤器覆盖证明）。"""

from __future__ import annotations

import re

import numpy as np
import pytest

from lumice_integral.symmetry import group as D6
from lumice_integral.symmetry import reflection_group as rg

from _writing_repo import writing_file

# 两份正文转录核对读写作仓的已发表文本（只读证据，写作仓不在时跳过，见 ``tests/_writing_repo.py``）
CH7_TEXT = "07-reflection-group/为什么恰好是十二个.md"
CH3_NOTE = "03-parallel-raypaths/平行光路矩阵归总.md"


def test_closure_is_twelve_with_growth_1_5_10_12_12():
    assert len(rg.GROUP) == 12
    assert rg.GROWTH == [1, 5, 10, 12, 12]
    assert len(rg.NORMALS) == 8 and len({rg.key(m) for m in rg.MIRRORS.values()}) == 4   # 8 面 4 镜


def test_twelve_elements_are_exactly_the_closure():
    seen = {rg.key(g) for g in rg.GROUP}
    keys = {rg.key(e.matrix) for e in rg.ELEMENTS}
    assert all(k in seen for k in keys)
    assert len(keys) == 12 == len(rg.GROUP)
    assert [e.number for e in rg.ELEMENTS] == list(range(1, 13))
    assert {(e.xy, e.z_sign) for e in rg.ELEMENTS} == {(xy, z) for xy in rg.XY_ORDER for z in (+1, -1)}


def test_elements_match_ch3_note_matrices():
    """与 check_group.py 原 ``user`` 字典（ch3 笔记的 12 个矩阵）逐个相等。"""
    b, r, r2 = np.diag([1, 1, -1.0]), rg.Rz(120), rg.Rz(-120)
    user = {1: rg.sxy(90) @ b, 2: rg.sxy(90), 3: r2 @ b, 4: r2, 5: r @ b, 6: r,
            7: rg.sxy(-30) @ b, 8: rg.sxy(-30), 9: rg.sxy(30) @ b, 10: rg.sxy(30), 11: b, 12: np.eye(3)}
    for n, M in user.items():
        assert np.abs(rg.MATRICES[n] - M).max() < 1e-12, n
        assert rg.identify(M).number == n


def test_conjugacy_classes_equal_published_six():
    assert rg.conjugacy_classes() == rg.published_classes() == [[1, 7, 9], [2, 8, 10], [3, 5], [4, 6], [11], [12]]
    # 用晶体对称操作 D6（Rz(60k) 与竖直镜面）做共轭得到同样的 6 类
    d6 = [rg.Rz(60 * k) for k in range(6)] + [rg.sxy(30 * m) for m in range(6)]
    assert rg.conjugacy_classes(d6) == rg.conjugacy_classes()


def test_eigenvalue_classes_are_five_with_11_merged_into_2():
    ev = rg.eigenvalue_classes()
    assert len(ev) == 5 and [2, 8, 10, 11] in ev


def test_commuting_with_rz_is_3_4_5_6_11_12():
    assert rg.commuting_with_rz() == [3, 4, 5, 6, 11, 12]
    assert sorted({rg.PUBLISHED[i] for i in rg.commuting_with_rz()}) == [3, 4, 5, 6]
    for e in rg.ELEMENTS:   # 对易 ⇔ xy 部分是旋转
        assert e.commutes_with_rz == (e.xy in ("e", "r+", "r-"))
    assert rg.wedge_cosines() == [-1.0, -0.5, 0.0, 0.5, 1.0]


def test_sxy_is_the_vertical_mirror_at_phi_minus_90():
    """$S_\\phi$（不变线方位角 φ）= 法向方位角 φ−90° 的镜面；三条反射列对应的相对面对由此而来。"""
    for phi in (90.0, 30.0, -30.0, 12.5):
        t = np.radians(phi - 90.0)
        assert np.abs(rg.sxy(phi) - rg.refl([np.cos(t), np.sin(t), 0.0])).max() < 1e-12
    for xy, (a, b) in rg.MIRROR_FACES.items():
        assert np.abs(rg.XY_MATRIX[xy] - rg.MIRRORS[a]).max() < 1e-12
        assert np.abs(rg.XY_MATRIX[xy] - rg.MIRRORS[b]).max() < 1e-12
        assert (b - a) == 3   # 相对面


def _parse_text_table():
    rows = {}
    for line in writing_file(CH7_TEXT).read_text(encoding="utf-8").splitlines():
        m = re.match(r"\|\s*(\d)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*$", line)
        if m:
            rows[int(m.group(1))] = (m.group(2), m.group(3))
    return rows


def test_halo_names_transcribe_the_text_table():
    rows = _parse_text_table()
    assert set(rows) == set(range(1, 7))
    for t, (column, plate) in rows.items():
        assert rg.HALO_NAMES[t] == (column, plate), t
    for e in rg.ELEMENTS:
        assert (e.column_halo, e.plate_halo) == rows[e.published_type]


def test_representative_paths_transcribe_the_ch3_note_and_realise_their_matrices():
    note = writing_file(CH3_NOTE).read_text(encoding="utf-8")
    for e in rg.ELEMENTS:
        m = re.search(rf"^{e.number}: 典型光路[：:]?\s*(.+)$", note, flags=re.M)
        assert m, e.number
        spec = re.sub(r"（.*?）", "", m.group(1))   # 去掉「（和5镜像对称）」这类括注
        listed = [tuple(int(c) for c in s.split("-")) for s in re.findall(r"\d(?:-\d)*", spec)]
        assert list(e.representative_paths) == listed, (e.number, listed)
        for p in e.representative_paths:
            assert np.abs(rg.path_matrix(p) - e.matrix).max() < 1e-12, (e.number, p)


def d6_matrix(g: D6.Element) -> np.ndarray:
    """D6 置换表示的元素 → 它在方向上的 3×3 矩阵：旋转 R_k → Rz(60k)，翻转 F_m（轴在 30m°）→ sxy(30m)。"""
    return rg.Rz(60.0 * g.shift) if not g.flip else rg.sxy(30.0 * g.shift)


@pytest.mark.parametrize("g", D6.elements(), ids=lambda g: g.name)
def test_d6_face_permutation_agrees_with_direction_matrix(g):
    """桥梁：g 对面编号的置换 与 g 的方向矩阵对面法向的作用是同一件事——`apply(g, k)` 的面法向 == G_g · n_k。"""
    G = d6_matrix(g)
    for k in range(1, 9):
        assert np.abs(rg.NORMALS[D6.apply(g, k)] - G @ rg.NORMALS[k]).max() < 1e-12, (g, k)


@pytest.mark.parametrize("e", rg.ELEMENTS, ids=lambda e: f"#{e.number}")
def test_d6_images_of_representative_paths_cover_exactly_the_conjugacy_class(e):
    """把典型光路作 D6 全部像，其矩阵集合 == 该元素所在共轭类；且 1↔2 互换（B 对称）矩阵不变——
    Lumice ``symmetry="PBD"`` 过滤器覆盖且只覆盖该类。"""
    cls = next(c for c in rg.conjugacy_classes() if e.number in c)
    for p in e.representative_paths:
        images = {rg.identify(rg.path_matrix(D6.apply_path(g, p))).number for g in D6.elements()}
        assert images == set(cls), (e.number, p, images)
        swapped = [{1: 2, 2: 1}.get(f, f) for f in p]
        assert rg.identify(rg.path_matrix(swapped)).number == e.number
        for g in D6.elements():
            G = d6_matrix(g)
            assert np.abs(rg.path_matrix(D6.apply_path(g, p)) - G @ e.matrix @ G.T).max() < 1e-12


def test_latex_and_lookup():
    assert rg.by_xy_z("e", -1).number == 11 and rg.by_xy_z("s90", +1).number == 2
    assert rg.by_number(3).latex == r"$b\,R_z(-120^\circ)$" and rg.by_number(2).latex == r"$S_{90^\circ}$"
    with pytest.raises(KeyError):
        rg.identify(rg.Rz(60.0))


# ---- 类光路枚举（R2-1）：代表序列镜面积 ∈ 目标类；完备性——长度 ≤ L 的全部合格序列都被某个代表的 PBD 轨道覆盖 ----


def _brute_force(number: int, max_len: int) -> set[tuple[int, ...]]:
    """独立于 DFS 的暴力枚举（itertools.product），作为 raypaths_of_class 的对照。"""
    import itertools
    targets = {rg.key(rg.MATRICES[n]) for n in rg.class_of(number)}
    out = set()
    for L in range(1, max_len + 1):
        for s in itertools.product(range(1, 9), repeat=L):
            if any(s[i] == s[i + 1] for i in range(L - 1)):
                continue
            if rg.key(rg.path_matrix(s)) in targets and rg.refraction_cancels(s):
                out.add(s)
    return out


def test_refraction_cancels_is_the_parallel_raypath_criterion():
    """ch3 笔记的 28 条典型光路全部满足 n_out = −M n_in；其中 10 条首尾面并不字面平行（如 #3 的 3-1-5-7-4），
    字面「首尾面平行」会漏掉它们；反例：3-4-3（M = S_4 把 n_3 搬到 n_5）不合格。"""
    literal = 0
    for e in rg.ELEMENTS:
        for p in e.representative_paths:
            assert rg.refraction_cancels(p), (e.number, p)
            literal += abs(rg.NORMALS[p[0]] @ rg.NORMALS[p[-1]]) > 0.999
    assert literal == 28 - 10
    assert sum(len(e.representative_paths) for e in rg.ELEMENTS) == 28
    assert not rg.refraction_cancels((3, 4, 3)) and not rg.refraction_cancels((1, 3, 1))
    assert rg.refraction_cancels((3, 4, 5)) and rg.refraction_cancels((3, 1, 5, 7, 4))


@pytest.mark.parametrize("number", [3, 2, 11])
def test_raypaths_of_class_match_brute_force_and_representatives_realise_the_class(number):
    """(a) 每条代表序列：长度 ≤ 6、相邻面不同、折射抵消、镜面积 ∈ 目标共轭类；DFS 枚举 == 暴力枚举（长度 ≤ 4 全量对照）。"""
    cls = set(rg.class_of(number))
    seqs = rg.raypaths_of_class(number)
    assert set(rg.raypaths_of_class(number, 4)) == _brute_force(number, 4)
    reps = rg.representatives_under_pbd(seqs)
    assert 0 < len(reps) < len(seqs)
    for s in reps:
        assert 1 <= len(s) <= 6 and all(a != b for a, b in zip(s, s[1:]))
        assert rg.refraction_cancels(s)
        assert rg.identify(rg.path_matrix(s)).number in cls, (number, s)
    for p in rg.by_number(number).representative_paths:                      # ch3 的典型光路都在枚举里
        assert p in seqs


@pytest.mark.parametrize("number", [3, 2, 11])
def test_pbd_orbits_of_representatives_cover_the_class_exactly_once(number):
    """(b) 完备性：全部合格序列 == 各代表 PBD 轨道（`math.group.apply_path` + 上下互换）的并；轨道两两不交（无冗余代表）。"""
    seqs = set(rg.raypaths_of_class(number))
    reps = rg.representatives_under_pbd(seqs)
    covered: set[tuple[int, ...]] = set()
    for r in reps:
        orbit = set()
        for g in D6.elements():
            im = tuple(D6.apply_path(g, r))
            orbit.add(im)
            orbit.add(tuple({1: 2, 2: 1}.get(f, f) for f in im))
        assert orbit == rg.pbd_orbit(r)
        assert orbit <= seqs, (number, r)                                    # 轨道不跑出类外
        assert not (orbit & covered), (number, r)                            # 轨道互不相交
        covered |= orbit
    assert covered == seqs                                                   # 少一个代表这里就红


def test_class_sizes_are_stable():
    """枚举规模（长度 ≤ 6）：#3 类 864 条 / 36 代表，#2 类 1278 / 88，#11 类 330 / 22——改动枚举规则时这里先响。"""
    assert {n: (len(rg.raypaths_of_class(n)), len(rg.representatives_under_pbd(rg.raypaths_of_class(n))))
            for n in (3, 2, 11)} == {3: (864, 36), 2: (1278, 88), 11: (330, 22)}


def test_pbd_orbit_on_pyramid_faces():
    """锥晶面号：B 把 13+i ↔ 23+i、1 ↔ 2；D6 逐环带作用；轨道大小 = 24 / 稳定子；六棱柱面号的轨道不受影响。"""
    orbit = rg.pbd_orbit((13, 26))
    assert (23, 16) in orbit and (16, 23) in orbit and (14, 27) in orbit
    assert all(f in range(13, 19) or f in range(23, 29) for path in orbit for f in path)
    assert len(orbit) == 12                       # 13-26 是"上锥进对面下锥出"，稳定子阶 2（B∘R3 保持）
    assert (2, 23, 13) in rg.pbd_orbit((1, 13, 23)) and (1, 14, 24) in rg.pbd_orbit((1, 13, 23))
    assert rg.pbd_orbit((3, 5)) == rg.pbd_orbit((5, 3)) and len(rg.pbd_orbit((1, 3, 2, 1, 5))) == 24
