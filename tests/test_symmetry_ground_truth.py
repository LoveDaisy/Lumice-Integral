"""``lumice_integral.symmetry.ground_truth``: the packaged ch3 ground-truth list parses, and is the writing repository's file byte for byte."""

from __future__ import annotations

from _writing_repo import writing_file

from lumice_integral.symmetry.ground_truth import (
    GROUND_TRUTH_PATH,
    ground_truth_groups,
    ground_truth_rows,
    ground_truth_sequences,
)


def test_packaged_file_parses_into_93_rows_of_16_halos():
    assert GROUND_TRUTH_PATH.is_file()
    rows = ground_truth_rows()
    assert len(rows) == 93 and max(len(r.faces) for r in rows) == 5
    assert len({r.index for r in rows}) == 93  # indices are unique (file order groups them by halo, not by index)
    groups = ground_truth_groups()
    assert len(groups) == 16 and sum(len(v) for v in groups.values()) == 93
    assert {"幻日环", "环天顶弧", "映日"} <= set(groups)
    assert len(ground_truth_sequences()) == 93  # no face sequence is listed twice


def test_packaged_file_is_the_writing_repository_file():
    source = writing_file("03-parallel-raypaths/data/raypath_ground_truth.txt")
    assert GROUND_TRUTH_PATH.read_bytes() == source.read_bytes()
