"""ch3 光路 ground truth 文件 ``03-parallel-raypaths/data/raypath_ground_truth.txt`` 的解析（唯一实现）：按晕名分组的
面序列清单（93 条，长度 ≤ 5，`render_raypaths.py` 当年用片晶 h/a = 0.2 渲染）。文件格式：晕名单独一行开一组，随后每行
``序号,标签,面序列…``（逗号分隔；``标签`` 是 ch3 当年按渲染结果给的 1–14 类标签，与 ``labeled_raypaths_5.csv`` 首列同义），
空行分隔。

写作仓 ``08-wedge-times-fold/code/check_ground_truth_signature.py``（Φ 类分组一致性）从这里取数，文件格式若变只改这一处。

Provenance：迁移自写作仓 ``halo_notes.math.ground_truth``（2026-09-24，task-symmetry-authority），公开名称 1:1 保留；
数据文件随模块进包（``lumice_integral/symmetry/data/``），默认路径不依赖写作仓或本仓 ``tests/`` 目录。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

GROUND_TRUTH_PATH = Path(__file__).resolve().parent / "data" / "raypath_ground_truth.txt"
"""包内数据：写作仓 ``03-parallel-raypaths/data/raypath_ground_truth.txt`` 的逐字节副本（2026-09-24 拷入）。"""


@dataclass(frozen=True)
class GroundTruthRow:
    """文件里的一行：序号、ch3 标签、面序列、所属晕名（标题行）。"""

    index: int
    label: int
    faces: tuple[int, ...]
    group: str


def ground_truth_rows(path: Path = GROUND_TRUTH_PATH) -> list[GroundTruthRow]:
    """按文件顺序的全部行。"""
    out: list[GroundTruthRow] = []
    group = None
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if not line.replace(",", "").isdigit():
            group = line
            continue
        nums = [int(x) for x in line.split(",")]
        out.append(GroundTruthRow(nums[0], nums[1], tuple(nums[2:]), group))
    return out


def ground_truth_sequences(path: Path = GROUND_TRUTH_PATH) -> dict[tuple[int, ...], str]:
    """``{面序列: 晕名}``，按文件顺序（``dict`` 保序）。"""
    return {r.faces: r.group for r in ground_truth_rows(path)}


def ground_truth_groups(path: Path = GROUND_TRUTH_PATH) -> dict[str, list[GroundTruthRow]]:
    """``{晕名: [行, …]}``，按文件顺序。"""
    groups: dict[str, list[GroundTruthRow]] = {}
    for r in ground_truth_rows(path):
        groups.setdefault(r.group, []).append(r)
    return groups


__all__ = ["GROUND_TRUTH_PATH", "GroundTruthRow", "ground_truth_groups", "ground_truth_rows", "ground_truth_sequences"]
