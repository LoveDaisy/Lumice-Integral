"""``lumice_integral.symmetry.signature``（迁移自写作仓 ``halo_notes/math/tests/test_signature.py``，只改 import）。"""

import itertools

import numpy as np
import pytest

from lumice_integral.geometry import Face, HexPrism, Polyhedron, Pyramid, fold_matrix
from lumice_integral.geometry.enumerate import enumerate_raypaths
from lumice_integral.geometry.feasibility import LatLonGrid, external_directions
from lumice_integral.symmetry import reflection_group as rg
from lumice_integral.symmetry.signature import (A_MAX_DEG, D6H, PhiClass, canonical_signature, face_of, family,
                                       fold_fingerprint, phi, phi_class, phi_class_table, phi_fingerprint,
                                       wedge_angle)

PLATE = HexPrism.from_ratio(0.2)
FACES = [f.number for f in PLATE.faces]
GRID = LatLonGrid(90, 180)


def _seq(label: str) -> tuple[int, ...]:
    return tuple(int(x) for x in label.split("-"))


def _conjugate(crystal, g, M, a, b):
    """D6h 元素 g 作用后的 (M', a', b') = (g M g^T, g·a, g·b)。"""
    return g @ M @ g.T, face_of(crystal, g @ crystal.normal(crystal.face(a))), face_of(crystal, g @ crystal.normal(crystal.face(b)))


# ---- 基本量 ----------------------------------------------------------------------------------


def test_d6h_has_24_distinct_elements_and_is_a_symmetry_of_the_prism():
    assert len({rg.key(g) for g in D6H}) == 24
    for g in D6H:
        for f in PLATE.faces:
            face_of(PLATE, g @ PLATE.normal(f))      # 不抛 ValueError ⇔ g 把面法向集合映到自身


def test_wedge_angles_and_families():
    assert wedge_angle(PLATE, np.eye(3), 3, 5) == pytest.approx(60.0)
    assert wedge_angle(PLATE, np.eye(3), 1, 3) == pytest.approx(90.0)
    assert wedge_angle(PLATE, np.eye(3), 1, 2) == pytest.approx(0.0)
    assert wedge_angle(PLATE, np.eye(3), 3, 4) == pytest.approx(120.0)
    assert wedge_angle(PLATE, np.eye(3), 1, 1) == pytest.approx(180.0)
    assert family(0.0) == "平行" and family(60.0) == "22° 族" and family(90.0) == "46° 族"
    assert family(120.0) is None and family(180.0) is None and family(28.0) is None
    assert 99.0 < A_MAX_DEG < 100.0


# ---- 规范型对 D6h 作用不变（性质测试） ------------------------------------------------------------


@pytest.mark.parametrize("number", [e.number for e in rg.ELEMENTS])
def test_canonical_signature_is_d6h_invariant(number):
    M = rg.MATRICES[number]
    for a, b in [(1, 2), (1, 3), (3, 1), (3, 5), (3, 4), (4, 8)]:
        want = canonical_signature(PLATE, M, a, b)
        for g in D6H:
            assert canonical_signature(PLATE, *_conjugate(PLATE, g, M, a, b)) == want


def test_canonical_signature_is_a_fixed_point():
    """规范代表自身再规范化不变，且它的 (a, b) 是所在轨道里字典序最小的。"""
    for M in rg.GROUP:
        for a, b in itertools.product(FACES, FACES):
            k, ca, cb = canonical_signature(PLATE, M, a, b)
            assert canonical_signature(PLATE, np.array(k).reshape(3, 3), ca, cb) == (k, ca, cb)


# ---- 已知光路的 (楔角, 折叠类) --------------------------------------------------------------------


KNOWN = {"3-5": (60.0, 12), "3-1-5": (60.0, 11), "1-3": (90.0, 12), "3-1-5-7-4": (0.0, 3)}


@pytest.mark.parametrize("label,expected", KNOWN.items())
def test_known_paths_wedge_and_fold(label, expected):
    faces = _seq(label)
    M = fold_matrix(PLATE, faces)
    a, b = faces[0], faces[-1]
    wedge, fold = expected
    assert wedge_angle(PLATE, M, a, b) == pytest.approx(wedge)
    assert rg.identify(M).number == fold
    cls = phi_class(PLATE, M, a, b)
    assert cls.wedge_deg == wedge
    row = next(r for r in phi_class_table(PLATE) if r.phi_class == cls)
    assert row.folds == tuple(rg.class_of(fold))                 # 该 Φ 类的折叠 = M 所在共轭类（0° 时 #3 与 #5 同类）
    assert row.family == family(wedge)


# ---- phi_class 的确定性与共轭不变 ---------------------------------------------------------------


def test_phi_class_is_deterministic_and_conjugation_invariant():
    for label in KNOWN:
        faces = _seq(label)
        M, a, b = fold_matrix(PLATE, faces), faces[0], faces[-1]
        first = phi_class(PLATE, M, a, b)
        assert phi_class(PLATE, M, a, b) == first and hash(phi_class(PLATE, M, a, b)) == hash(first)
        for g in D6H[::5]:
            assert phi_class(PLATE, *_conjugate(PLATE, g, M, a, b)) == first


def test_time_reversal_is_a_different_class_at_60_degrees():
    """定义域是 Φ 身份的一部分：3-6-1-4（M = #1，楔角 60°）与它的时间反演 4-1-6-3（M^T = M）落在不同的 Φ 类
    （A60-01 / A60-02）；漏掉入射侧判据会把它们并掉——那正是 60° 从 12 错并成 6 的机制。"""
    M = fold_matrix(PLATE, (3, 6, 1, 4))
    assert rg.identify(M).number == 1 and np.allclose(M, M.T)
    assert wedge_angle(PLATE, M, 3, 4) == pytest.approx(60.0)
    assert phi_class(PLATE, M, 3, 4) != phi_class(PLATE, M.T, 4, 3)
    assert phi_fingerprint(PLATE, M, 3, 4) != phi_fingerprint(PLATE, M.T, 4, 3)


def test_parallel_family_uses_fold_conjugacy_not_domain():
    """0° 的类按 M 的共轭类取（framework 定理 5′：14 → 6），从哪个面进不区分；带定义域的指纹则会把 1-2 与 3-6 分开。"""
    e = np.eye(3)
    assert phi_class(PLATE, e, 1, 2) == phi_class(PLATE, e, 3, 6)
    assert phi_class(PLATE, e, 1, 2).fingerprint == fold_fingerprint(e)
    assert phi_fingerprint(PLATE, e, 1, 2) != phi_fingerprint(PLATE, e, 3, 6)


# ---- 全类表：34 = 6 + 12 + 16 --------------------------------------------------------------------


def test_phi_class_table_counts_and_ids():
    rows = phi_class_table(PLATE)
    live = [r for r in rows if r.transmittable]
    assert len(live) == 34
    by_wedge = {}
    for r in live:
        by_wedge[r.wedge_deg] = by_wedge.get(r.wedge_deg, 0) + 1
    assert by_wedge == {0.0: 6, 60.0: 12, 90.0: 16}
    assert sum(len(r.signatures) for r in live) == 42
    ids = [r.id for r in live]
    assert ids == [f"A0-{k:02d}" for k in range(1, 7)] + [f"A60-{k:02d}" for k in range(1, 13)] + [f"A90-{k:02d}" for k in range(1, 17)]
    assert len({r.phi_class for r in rows}) == len(rows)
    # 0° 的 6 行 = ch7 发表的 6 类，按发表类型顺序
    assert [r.folds for r in live[:6]] == [tuple(c) for c in rg.published_classes()]
    # 60°/90° 一对一：每行恰一个 signature；死角只剩 120°/180° 各一行
    assert all(len(r.signatures) == 1 for r in live[6:])
    assert [(r.wedge_deg, r.transmittable) for r in rows if not r.transmittable] == [(120.0, False), (180.0, False)]


def test_phi_class_table_is_reproducible_and_ratio_independent():
    a, b = phi_class_table(PLATE), phi_class_table(HexPrism.from_ratio(5.0))
    assert [(r.id, r.signatures, r.folds) for r in a] == [(r.id, r.signatures, r.folds) for r in b]


# ---- 面集合参数化：函数读的是传入的 crystal，不是别的面表（round 3 Minor (a)） ---------------------


def test_functions_read_the_given_crystal_not_a_module_table():
    """把六棱柱面 3 / 4 的编号互换成一个「错面表」：同一 (M, a, b) 喂真六棱柱与错面表，三个函数的结果都必须不同。
    若相同即说明函数内部绕过了 crystal、读了模块级面表。"""
    swap = {3: 4, 4: 3}
    wrong = Polyhedron(PLATE.vertices, [Face(swap.get(f.number, f.number), f.vertex_ids) for f in PLATE.faces])
    e = np.eye(3)
    assert wedge_angle(wrong, e, 3, 5) != pytest.approx(wedge_angle(PLATE, e, 3, 5))
    assert canonical_signature(wrong, e, 3, 5) != canonical_signature(PLATE, e, 3, 5)
    assert phi_fingerprint(wrong, e, 3, 5) != phi_fingerprint(PLATE, e, 3, 5)
    assert phi_class(wrong, e, 3, 5) != phi_class(PLATE, e, 3, 5)


def test_face_of_rejects_non_face_normals():
    with pytest.raises(ValueError):
        face_of(PLATE, np.array([1.0, 1.0, 1.0]) / np.sqrt(3))


# ---- Φ 与可行性判定同一份定义域（风险 4 交叉检查） ---------------------------------------------------


def test_phi_is_defined_on_admissible_directions_of_enumerated_paths():
    """枚举出的真实光路：可行方向集（晶体内）经逆 Snell 变成外部方向后喂 phi，必须都有输出；平行光路的输出恰为 M d。"""
    recs = enumerate_raypaths(PLATE, 3, GRID, entry_faces=(1, 3), symmetry_orbit=rg.pbd_orbit)
    checked = 0
    for r in recs:
        if r.length < 2:
            continue
        M = fold_matrix(PLATE, r.faces)
        idx = np.flatnonzero(r.admissible_mask)
        idx = idx[np.linspace(0, len(idx) - 1, min(5, len(idx))).astype(int)]
        ext = external_directions(PLATE, r.faces, GRID.directions_at(idx))
        for d in ext:
            out = phi(PLATE, M, r.faces[0], r.faces[-1], d)
            assert out is not None, r.label
            assert np.linalg.norm(out) == pytest.approx(1.0)
            if rg.refraction_cancels(r.faces):
                assert np.allclose(out, M @ d, atol=1e-9), r.label
            checked += 1
    assert checked > 50


def test_phi_returns_none_outside_domain():
    n3 = PLATE.normal(PLATE.face(3))
    assert phi(PLATE, np.eye(3), 3, 5, n3) is None                   # 从背面入射
    assert phi(PLATE, np.eye(3), 3, 4, -n3) is None                  # 120° 死角：正入射也出不去


# ---- 锥晶 smoke：管线吃更一般的 Polyhedron 子类 ----------------------------------------------------


def test_pyramid_smoke_pipeline_runs_and_classes_grow():
    pyramid = Pyramid()
    recs = enumerate_raypaths(pyramid, 3, LatLonGrid(45, 90), entry_faces=(1, 3, 13))
    classes = set()
    for r in recs:
        if r.length < 2:
            continue
        M = fold_matrix(pyramid, r.faces)
        cls = phi_class(pyramid, M, r.faces[0], r.faces[-1])
        assert isinstance(cls, PhiClass)
        classes.add(cls)
    assert len(classes) > 34                                          # 非平凡增长：比六棱柱全部可透射类还多
    assert any(cls.family is None and cls.wedge_deg < A_MAX_DEG for cls in classes)   # 出现六棱柱之外的可透射楔角
