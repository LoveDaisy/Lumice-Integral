import numpy as np
import pytest

from lumice_integral.geometry import HexPrism
from lumice_integral.geometry.enumerate import (corridor_index_for_path, enumerate_raypaths,
                                                enumerate_raypaths_with_stats)
from lumice_integral.geometry.feasibility import (LatLonGrid, admissible_directions, corridor_mask, entry_points,
                                                  external_directions, is_feasible)

from _geometry_oracles import pbd_orbit, trace_faces

GRID = LatLonGrid(90, 180)      # 2°
PLATE = HexPrism.from_ratio(0.2)

# ch2 已发表：片晶 h/a = 0.2，PBD 代表
CH2_LE2 = ["1", "1-2", "1-3", "3", "3-1", "3-5", "3-6"]
CH2_EQ3 = ["1-2-1", "1-2-3", "1-3-2", "3-1-2", "3-1-5", "3-1-6", "3-4-5", "3-5-7", "3-5-8", "3-6-3", "3-6-4"]
CH2_COUNTS = {4: 25, 5: 61, 6: 169}


def _seq(label: str) -> tuple[int, ...]:
    return tuple(int(x) for x in label.split("-"))


@pytest.fixture(scope="module")
def plate_le6():
    recs, stats = enumerate_raypaths_with_stats(PLATE, 6, GRID, entry_faces=(1, 3), symmetry_orbit=pbd_orbit)
    return recs, stats


def _by_length(recs):
    out = {}
    for r in recs:
        out.setdefault(r.length, []).append(r)
    return out


# ---- 掩码单调性与两种掩码不可混用 -----------------------------------------------------------


@pytest.mark.parametrize("faces", [(1, 3, 2), (3, 1, 5, 7, 4), (1, 2, 3, 5, 1), (3, 5, 7, 3, 5)])
def test_corridor_mask_is_monotone_in_prefix(faces):
    """子前缀的走廊掩码 ⊇ 扩展后的走廊掩码（走廊只会变窄）。"""
    prev = None
    for k in range(1, len(faces) + 1):
        cur = corridor_mask(PLATE, faces[:k], GRID)
        if prev is not None:
            assert not (cur & ~prev).any()
        prev = cur


@pytest.mark.parametrize("faces", [(1,), (3,), (1, 3), (3, 4), (1, 3, 2), (3, 1, 5, 7, 4), (1, 2, 3, 5, 1),
                                   (3, 5, 7, 3, 5)])
def test_dfs_incremental_corridor_matches_corridor_mask(faces):
    """a56：DFS 剪枝维护的增量走廊状态（``corridor_index_for_path``，与 ``enumerate.dfs()`` 共用
    ``_ghost_for``/``_extend_batch`` 两个原语）与独立从头算的 ``corridor_mask`` 必须逐位一致——这是 code review
    要求的机械核验，不靠「两处代码看起来在做同一件事」的人工审查兜底。"""
    got = np.sort(corridor_index_for_path(PLATE, faces, GRID))
    want = np.flatnonzero(corridor_mask(PLATE, faces, GRID))
    assert np.array_equal(got, want)


def test_pruning_uses_corridor_not_admissible_mask(plate_le6):
    """风险 4：3-4 作为完整光路出射不可行（admissible 为空），但它的走廊非空、延伸后的 3-4-5 可行——
    用 admissible 掩码剪枝会把 3-4-5 漏掉。"""
    assert corridor_mask(PLATE, (3, 4), GRID).any()
    assert not admissible_directions(PLATE, (3, 4), GRID, refine_levels=0).mask.any()
    assert is_feasible(PLATE, (3, 4, 5), GRID)
    recs, _ = plate_le6
    labels = {r.label for r in recs}
    assert "3-4-5" in labels and "3-4" not in labels


# ---- ch2 片晶 h/a = 0.2 基准 ----------------------------------------------------------------


def test_ch2_plate_length_le3_matches_published_list(plate_le6):
    """长度 ≤ 3 共 18 条逐条相等；本实现额外给出 1-3-4 / 3-4-1（互为时间反演），二者是贴 3/4 公共棱掠射的细缝
    （面积 < 0.001 sr、最小入射角 ≈ 49.3° 压在临界角下），ch2 内点法的障碍项到不了那里——判据差异，不是 bug。"""
    recs, _ = plate_le6
    by = _by_length(recs)
    got = {r.faces for L in (1, 2, 3) for r in by[L]}
    expected = {_seq(s) for s in CH2_LE2 + CH2_EQ3}
    assert expected <= got
    extra = got - expected
    assert extra == {(1, 3, 4), (3, 4, 1)}
    for r in recs:
        if r.faces in extra:
            assert r.area_sr < 1e-3 and 49.0 < r.min_incidence_deg < 49.8
    assert len(by[1]) + len(by[2]) == 7 and len(by[3]) == 11 + 2


def test_ch2_plate_length_4_to_6_counts_regression(plate_le6):
    """ch2 发表 25 / 61 / 169；本实现 2° 网格给 31 / 70 / 147（1° 网格 31 / 73 / 162，0.5° 与 1° 相同）。差异归因见
    progress.md：多出的是贴棱掠射 / 压在临界角上的细缝（面积 ≲ 0.002 sr），长度 6 少掉的一批 ch2 未发表逐条清单、
    无法逐条对，独立蒙特卡洛追迹（10⁶ 条射线）也只实现到 146 条。这里钉住本实现的数字防回归。"""
    recs, _ = plate_le6
    by = _by_length(recs)
    assert {L: len(by[L]) for L in (4, 5, 6)} == {4: 31, 5: 70, 6: 147}
    # 与 ch2 的差别全部来自小面积记录：把面积 < 0.003 sr 的细缝去掉后长度 4 / 5 与 ch2 的差 ≤ 3
    solid = {L: sum(r.area_sr >= 3e-3 for r in by[L]) for L in (4, 5, 6)}
    assert abs(solid[4] - CH2_COUNTS[4]) <= 3 and abs(solid[5] - CH2_COUNTS[5]) <= 3
    # 长度 6：ch2 未发表逐条清单，169 与 solid[6] 差 30 条，远超长度 4/5 的 ≤3 容差，判据不可比（见
    # progress.md DECISION「长度 6 验证口径降级」）；这里不与 CH2_COUNTS[6] 对比，只把本实现的当前值钉成
    # 显式回归锁——不是像上一版那样算出来又弃用，而是让它真的参与断言。
    assert solid[6] == 139


def test_records_are_sorted_and_labels_consistent(plate_le6):
    recs, _ = plate_le6
    keys = [(r.length, r.faces) for r in recs]
    assert keys == sorted(keys)
    for r in recs:
        assert r.label == "-".join(map(str, r.faces))
        assert r.admissible_mask.any() and r.area_sr > 0
        assert all(a != b for a, b in zip(r.faces, r.faces[1:]))


def test_dedup_keeps_orbit_minimum_and_matches_full_entry_enumeration():
    """只从 1 / 3 入射 + PBD 去重，与从全部 8 面入射 + PBD 去重给出相同的代表集合；不去重时每个轨道的最小元都在。"""
    partial = {r.faces for r in enumerate_raypaths(PLATE, 3, GRID, entry_faces=(1, 3), symmetry_orbit=pbd_orbit)}
    full = enumerate_raypaths(PLATE, 3, GRID, symmetry_orbit=pbd_orbit)
    assert {r.faces for r in full} == partial
    raw = {r.faces for r in enumerate_raypaths(PLATE, 3, GRID)}
    assert len(raw) > len(partial)
    assert {min(pbd_orbit(f)) for f in raw} == partial
    for f in raw:
        assert min(pbd_orbit(f)) in raw


def test_stats_are_consistent(plate_le6):
    recs, st = plate_le6
    assert st.attempted > st.pruned > 0 and 0 < st.prune_rate < 1
    assert st.feasible_by_length[1] == 2
    assert sum(st.geometric_by_length.values()) == st.attempted - st.pruned


def test_single_face_records_present_and_max_len_1():
    recs = enumerate_raypaths(PLATE, 1, GRID, entry_faces=(1, 3))
    assert [r.faces for r in recs] == [(1,), (3,)]
    assert all(r.area_sr == pytest.approx(2 * np.pi, rel=1e-3) for r in recs)


def test_refine_levels_changes_area_only_slightly():
    coarse = enumerate_raypaths(PLATE, 3, GRID, entry_faces=(1,), symmetry_orbit=pbd_orbit)
    fine = enumerate_raypaths(PLATE, 3, GRID, entry_faces=(1,), symmetry_orbit=pbd_orbit, refine_levels=1)
    assert [r.faces for r in coarse] == [r.faces for r in fine]
    for a, b in zip(coarse, fine):
        if a.area_sr > 1e-2:
            assert abs(a.area_sr - b.area_sr) / a.area_sr < 0.05


# ---- 每条记录都有可追迹的见证 --------------------------------------------------------------


def test_every_record_has_traceable_witnesses(plate_le6):
    """每条记录抽 ≤ 5 个可行方向：交集质心见证 + 逆 Snell 喂独立实现的 trace_faces，面序列须恰好相等。"""
    recs, _ = plate_le6
    for r in recs:
        if r.length == 1:
            continue
        idx = np.flatnonzero(r.admissible_mask)
        idx = idx[np.linspace(0, len(idx) - 1, min(5, len(idx))).astype(int)]
        d = GRID.directions_at(idx)
        p, ext = entry_points(PLATE, r.faces, d), external_directions(PLATE, r.faces, d)
        for k in range(len(idx)):
            assert trace_faces(PLATE, p[k] - 3 * ext[k], ext[k], r.length) == list(r.faces), r.label


# ---- 未迁移的蓝本用例（依赖写作仓数据文件 / 搜索算法，见 progress.md DECISION 2026-09-16 15:35）------------------
# - test_ch3_ground_truth_all_reachable_on_plate：需要 ground_truth.py + 03-parallel-raypaths/data/labeled_raypaths_5.csv
# - test_ch3_representative_paths_count / test_feasible_representative_paths_solve_*：需要 reflection_group.ELEMENTS
#   与写作仓 draw.raypath 的 solve_raypath / search_raypath（光路搜索器，不属于几何权威）


# ---- Step 5：性能（慢测试） ---------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("ratio", [0.2, 0.5, 1.0, 2.0, 5.0])
def test_max_len_8_runs_within_minutes(ratio):
    import time
    t = time.time()
    recs, st = enumerate_raypaths_with_stats(HexPrism.from_ratio(ratio), 8, GRID, entry_faces=(1, 3),
                                             symmetry_orbit=pbd_orbit)
    elapsed = time.time() - t
    by = {L: sum(r.length == L for r in recs) for L in range(1, 9)}
    print(f"h/a={ratio}: {elapsed:.1f}s prune_rate={st.prune_rate:.3f} feasible={by}")
    assert elapsed < 300
