"""``lumice_integral.symmetry.group``（迁移自写作仓 ``halo_notes/math/tests/test_group.py``，只改 import）。D6 置换表示：群公理、正文两个例子、图 1.6 / 1.7 逐格转录（看图不看公式，旧图是已发表事实）。"""

import itertools

import pytest

from lumice_integral.symmetry.group import (IDENTITY, SIDE_FACES, Fd, Ff, R, apply, apply_path, compose, elements, inverse,
                                   latex, multiplication_table)

# ---- 旧图 1.7 `img/legacy/d6_table_shade.png` 逐格转录（12 行 × 12 列，行列顺序 = 表头顺序）----
# 记号：I / R1..R5 / Ff3 / Fd34 / Ff4 / Fd45 / Ff5 / Fd56
TABLE_1_7 = """
I    R1   R2   R3   R4   R5   Ff3  Fd34 Ff4  Fd45 Ff5  Fd56
R1   R2   R3   R4   R5   I    Fd56 Ff3  Fd34 Ff4  Fd45 Ff5
R2   R3   R4   R5   I    R1   Ff5  Fd56 Ff3  Fd34 Ff4  Fd45
R3   R4   R5   I    R1   R2   Fd45 Ff5  Fd56 Ff3  Fd34 Ff4
R4   R5   I    R1   R2   R3   Ff4  Fd45 Ff5  Fd56 Ff3  Fd34
R5   I    R1   R2   R3   R4   Fd34 Ff4  Fd45 Ff5  Fd56 Ff3
Ff3  Fd34 Ff4  Fd45 Ff5  Fd56 I    R1   R2   R3   R4   R5
Fd34 Ff4  Fd45 Ff5  Fd56 Ff3  R5   I    R1   R2   R3   R4
Ff4  Fd45 Ff5  Fd56 Ff3  Fd34 R4   R5   I    R1   R2   R3
Fd45 Ff5  Fd56 Ff3  Fd34 Ff4  R3   R4   R5   I    R1   R2
Ff5  Fd56 Ff3  Fd34 Ff4  Fd45 R2   R3   R4   R5   I    R1
Fd56 Ff3  Fd34 Ff4  Fd45 Ff5  R1   R2   R3   R4   R5   I
"""
# ---- 旧图 1.4 `img/legacy/c6_table.png` 逐格转录（6 × 6）----
TABLE_1_4 = """
I  R1 R2 R3 R4 R5
R1 R2 R3 R4 R5 I
R2 R3 R4 R5 I  R1
R3 R4 R5 I  R1 R2
R4 R5 I  R1 R2 R3
R5 I  R1 R2 R3 R4
"""
# ---- 旧图 1.6 `img/legacy/d6_f.png` 每个翻转把面 n 送到哪个位置（六个面板逐个读图）----
FLIPS_1_6 = {
    "Ff3": {3: 3, 4: 8, 5: 7, 6: 6, 7: 5, 8: 4},
    "Ff4": {3: 5, 4: 4, 5: 3, 6: 8, 7: 7, 8: 6},
    "Ff5": {3: 7, 4: 6, 5: 5, 6: 4, 7: 3, 8: 8},
    "Fd34": {3: 4, 4: 3, 5: 8, 6: 7, 7: 6, 8: 5},
    "Fd45": {3: 6, 4: 5, 5: 4, 6: 3, 7: 8, 8: 7},
    "Fd56": {3: 8, 4: 7, 5: 6, 6: 5, 7: 4, 8: 3},
}


def parse(table: str) -> list[list[str]]:
    return [line.split() for line in table.strip().splitlines()]


def names(table):
    return [[g.name for g in row] for row in table]


def test_twelve_distinct_elements_in_published_order():
    elems = elements()
    assert [g.name for g in elems] == ["I", "R1", "R2", "R3", "R4", "R5", "Ff3", "Fd34", "Ff4", "Fd45", "Ff5",
                                       "Fd56"]
    perms = {tuple(apply(g, n) for n in SIDE_FACES) for g in elems}
    assert len(perms) == 12


def test_group_axioms():
    elems = elements()
    for a, b in itertools.product(elems, repeat=2):
        assert compose(a, b) in elems                                  # 封闭
    for a, b, c in itertools.product(elems, repeat=3):
        assert compose(compose(a, b), c) == compose(a, compose(b, c))  # 结合
    for a in elems:
        assert compose(a, IDENTITY) == a and compose(IDENTITY, a) == a  # 单位
        assert compose(a, inverse(a)) == IDENTITY and compose(inverse(a), a) == IDENTITY  # 逆


def test_c6_is_a_subgroup():
    rots = elements()[:6]
    assert all(not g.flip for g in rots)
    assert all(compose(a, b) in rots for a in rots for b in rots)
    assert all(inverse(a) in rots for a in rots)


def test_bottom_faces_fixed_and_side_faces_permuted():
    for g in elements():
        assert apply(g, 1) == 1 and apply(g, 2) == 2
        assert sorted(apply(g, n) for n in SIDE_FACES) == list(SIDE_FACES)


def test_text_line_45_example_rc2():
    assert apply_path(R(2), [1, 5, 2, 7, 1]) == [1, 7, 2, 3, 1]


def test_text_line_71_example_ff3_then_fd45_is_rc3():
    assert compose(Ff(3), Fd(5 - 1)) == R(3)


def test_text_line_39_example_table_row2_col3():
    table = multiplication_table(elements()[:6])
    assert table[1][2] == R(3)  # 2 行 3 列（1 起）


def test_reflections_match_figure_1_6_transcription():
    by_name = {g.name: g for g in elements()}
    for name, mapping in FLIPS_1_6.items():
        assert {n: apply(by_name[name], n) for n in SIDE_FACES} == mapping, name


def test_c6_table_matches_figure_1_4_transcription():
    assert names(multiplication_table(elements()[:6])) == parse(TABLE_1_4)


def test_d6_table_matches_figure_1_7_transcription():
    got, want = names(multiplication_table(elements())), parse(TABLE_1_7)
    assert len(want) == 12 and all(len(r) == 12 for r in want)
    assert got == want


def test_figure_1_7_transcription_is_self_consistent():
    """转录数据自身的一致性：左上 6×6 = 图 1.4；右下 6×6 全是旋转（翻转 × 翻转 = 旋转）。"""
    t = parse(TABLE_1_7)
    assert [r[:6] for r in t[:6]] == parse(TABLE_1_4)
    assert all(c in ("I", "R1", "R2", "R3", "R4", "R5") for r in t[6:] for c in r[6:])
    assert all(len(set(r)) == 12 for r in t) and all(len({r[j] for r in t}) == 12 for j in range(12))  # 拉丁方


def test_text_line_76_literal_formulas_do_not_give_six_distinct_flips():
    """正文第 76 行 F^f_k(n) ≡ −n−k、F^d_{k,k+1}(n) ≡ −n+k 逐字套用只得到 5 个不同置换——记录这一出入的机械证据。"""
    def perm(c):
        return tuple(3 + (c - n - 3) % 6 for n in SIDE_FACES)
    literal = {perm(-k) for k in (3, 4, 5)} | {perm(k) for k in (3, 4, 5)}
    assert len(literal) == 5
    assert perm(-3) == perm(3)  # F^f_3 与 F^d_{4,5} 重合


def test_latex_names():
    assert [latex(g) for g in elements()] == ["$I$", "$R^c_1$", "$R^c_2$", "$R^c_3$", "$R^c_4$", "$R^c_5$",
                                              "$F^f_3$", "$F^d_{3,4}$", "$F^f_4$", "$F^d_{4,5}$", "$F^f_5$",
                                              "$F^d_{5,6}$"]


@pytest.mark.parametrize("bad", [0, 9, -1])
def test_apply_rejects_bad_face(bad):
    with pytest.raises(ValueError):
        apply(R(1), bad)


def test_apply_acts_on_pyramid_bands_like_the_prism_band():
    """锥面 13+i / 23+i 与侧面 3+i 同方位角：同一元素作用后环带内序号一致；底面不动；环带之间不串。"""
    for g in elements():
        for i in range(6):
            j = apply(g, 3 + i) - 3
            assert apply(g, 13 + i) == 13 + j and apply(g, 23 + i) == 23 + j
        assert apply(g, 1) == 1 and apply(g, 2) == 2
    assert apply(R(1), 18) == 13 and apply(R(5), 23) == 28
    for bad in (12, 19, 22, 29):
        with pytest.raises(ValueError):
            apply(R(1), bad)
