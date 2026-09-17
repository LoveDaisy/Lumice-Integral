"""Scene-level prescan table: determinism, batch/prefix invariants, exact queries, cache."""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import numpy as np
import pytest

from lumice_integral.canonical_scene import (
    CANONICAL_REFRACTIVE_INDEX,
    CANONICAL_RENDER,
    canonical_incident_direction,
)
from lumice_integral.camera import linear_pixel_outgoing_direction
from lumice_integral.optics import DOMAIN_MARGIN_NAMES, path_3_5_domain
from lumice_integral.prescan import (
    PROVENANCE_SUFFIX,
    PrescanTable,
    build_or_load_prescan_table,
    build_prescan_table,
    chord_radius,
    provenance_path_of,
)

SEED = 20260916
SMALL = 50_000


def _build(sample_count: int = SMALL, **kwargs) -> PrescanTable:
    return build_prescan_table(
        canonical_incident_direction(), CANONICAL_REFRACTIVE_INDEX, sample_count=sample_count, rng_seed=SEED, **kwargs
    )


def _assert_same_table(left: PrescanTable, right: PrescanTable) -> None:
    assert left.build_parameters() == right.build_parameters()
    assert np.array_equal(left.rotations, right.rotations)
    assert np.array_equal(left.directions, right.directions)
    assert np.array_equal(left.sample_indices, right.sample_indices)
    for name in DOMAIN_MARGIN_NAMES:
        assert np.array_equal(left.margins[name], right.margins[name])


@pytest.fixture(scope="module")
def table() -> PrescanTable:
    return _build()


def test_build_is_deterministic_and_keeps_only_domain_valid_samples(table: PrescanTable):
    _assert_same_table(table, _build())
    assert table.path_id == "3-5"
    assert table.sample_count == SMALL and 0 < table.valid_count < SMALL
    assert table.rotations.shape == (table.valid_count, 3, 3)
    assert table.directions.shape == (table.valid_count, 3)
    assert np.all(np.diff(table.sample_indices) > 0)  # sampling order, no duplicates
    for name in DOMAIN_MARGIN_NAMES:
        assert np.all(table.margins[name] > 0)
    assert np.allclose(np.linalg.norm(table.directions, axis=1), 1.0, atol=1e-12)
    # Every stored row passes the scalar gate; a spot-check of rows that were
    # dropped shows they fail it (the table is the valid subset, nothing else).
    for i in (0, table.valid_count // 2, table.valid_count - 1):
        assert path_3_5_domain(table.rotations[i], table.incident_direction, table.refractive_index).valid
    dropped = sorted(set(range(200)) - set(table.sample_indices[table.sample_indices < 200].tolist()))
    assert dropped, "expected some of the first 200 samples to be outside the domain"


def test_batch_size_bounds_memory_but_does_not_change_the_table(table: PrescanTable):
    # Chunk boundaries that divide, do not divide, and exceed the sample count.
    for batch_size in (7_919, 12_500, SMALL + 1):
        _assert_same_table(table, _build(batch_size=batch_size))


def test_prefix_of_a_larger_table_equals_the_smaller_build(table: PrescanTable):
    # Prefix invariant (plan review-01 Minor 2): the N-sample stream is the
    # first N draws of the 2N-sample stream, so the density survey may take
    # ``prefix(N)`` of one 2N build instead of building twice.
    larger = _build(2 * SMALL, batch_size=12_500)
    _assert_same_table(table, larger.prefix(SMALL))
    assert larger.valid_count > table.valid_count
    assert np.array_equal(larger.rotations[: table.valid_count], table.rotations)
    with pytest.raises(ValueError):
        larger.prefix(0)
    with pytest.raises(ValueError):
        larger.prefix(2 * SMALL + 1)


def test_candidates_equal_the_brute_force_alignment_filter(table: PrescanTable):
    rng = np.random.default_rng(1)
    targets = [linear_pixel_outgoing_direction(row, column, **CANONICAL_RENDER) for row, column in ((150, 150), (40, 150), (780, 150))]
    # Random targets inside the reachable cone (a stored direction plus
    # noise) so the balls are populated; plus a few arbitrary ones.
    for _ in range(20):
        vector = table.directions[rng.integers(table.valid_count)] + 0.05 * rng.standard_normal(3)
        targets.append(vector / np.linalg.norm(vector))
    for _ in range(5):
        vector = rng.standard_normal(3)
        targets.append(vector / np.linalg.norm(vector))
    nonempty = 0
    for target in targets:
        for tolerance in (0.5, 2.0, 10.0):
            expected = np.nonzero(table.directions @ target >= np.cos(np.radians(tolerance)))[0]
            found = table.candidates(target, tolerance)
            assert np.array_equal(found, expected)
            nonempty += found.size > 0
    assert nonempty >= 2 * 20  # the check exercised non-trivial balls
    # The chord radius is the exact image of the angular cap on the unit sphere.
    assert chord_radius(60.0) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        table.candidates(np.zeros(2), 2.0)
    with pytest.raises(ValueError):
        table.candidates(np.array([0.0, 0.0, 1.0]), 0.0)


def test_query_latency_is_well_under_the_per_pixel_budget(table: PrescanTable):
    target = linear_pixel_outgoing_direction(150, 150, **CANONICAL_RENDER)
    table.candidates(target, 2.0)
    start = time.perf_counter()
    for _ in range(20):
        table.candidates(target, 2.0)
    per_query_ms = (time.perf_counter() - start) / 20 * 1e3
    assert per_query_ms < 50.0  # loose CI bound; benchmarks/benchmark_prescan_table.py pins <= 5 ms


def test_pickle_round_trip_rebuilds_the_index(table: PrescanTable):
    copy = pickle.loads(pickle.dumps(table))
    _assert_same_table(table, copy)
    target = linear_pixel_outgoing_direction(150, 150, **CANONICAL_RENDER)
    assert np.array_equal(copy.candidates(target, 2.0), table.candidates(target, 2.0))


def test_save_and_load_round_trip_with_provenance(table: PrescanTable, tmp_path: Path):
    files = table.save(tmp_path / "cache" / "prescan.npz")
    assert files["provenance"] == provenance_path_of(files["arrays"])
    assert files["provenance"].name == "prescan.npz" + PROVENANCE_SUFFIX
    provenance = json.loads(files["provenance"].read_text())
    assert provenance["build"] == table.build_parameters()
    assert provenance["build"]["sample_count"] == SMALL and provenance["build"]["rng_seed"] == SEED
    assert provenance["valid_count"] == table.valid_count
    assert "git_commit" in provenance and len(provenance["arrays"]["sha256"]) == 64
    loaded = PrescanTable.load(files["arrays"])
    _assert_same_table(table, loaded)
    # Tampering with the arrays is detected on load.
    payload = bytearray(files["arrays"].read_bytes())
    payload[-1] ^= 0xFF
    files["arrays"].write_bytes(bytes(payload))
    with pytest.raises(ValueError, match="sha256"):
        PrescanTable.load(files["arrays"])


def test_build_or_load_reuses_a_matching_cache_and_rebuilds_a_mismatched_one(table: PrescanTable, tmp_path: Path):
    cache = tmp_path / "prescan.npz"
    messages: list[str] = []
    kwargs = dict(sample_count=SMALL, rng_seed=SEED, log=messages.append)
    first = build_or_load_prescan_table(cache, canonical_incident_direction(), CANONICAL_REFRACTIVE_INDEX, **kwargs)
    _assert_same_table(table, first)
    assert cache.exists() and provenance_path_of(cache).exists()
    assert messages[0].endswith("absent; building") and "cached at" in messages[-1]

    messages.clear()
    second = build_or_load_prescan_table(cache, canonical_incident_direction(), CANONICAL_REFRACTIVE_INDEX, **kwargs)
    _assert_same_table(table, second)
    assert len(messages) == 1 and messages[0].startswith("prescan table loaded from")

    # A different seed must not reuse the cache; the cache is overwritten.
    messages.clear()
    other = build_or_load_prescan_table(
        cache, canonical_incident_direction(), CANONICAL_REFRACTIVE_INDEX, sample_count=SMALL, rng_seed=SEED + 1, log=messages.append
    )
    assert other.rng_seed == SEED + 1 and not np.array_equal(other.rotations[:10], table.rotations[:10])
    assert "different parameters; rebuilding" in messages[0]
    assert json.loads(provenance_path_of(cache).read_text())["build"]["rng_seed"] == SEED + 1

    # In-memory only when no cache path is given.
    memory = build_or_load_prescan_table(None, canonical_incident_direction(), CANONICAL_REFRACTIVE_INDEX, sample_count=SMALL, rng_seed=SEED)
    _assert_same_table(table, memory)


def test_build_rejects_unsupported_paths_and_bad_counts():
    with pytest.raises(ValueError, match="path_id"):
        _build(path_id="4-6")
    with pytest.raises(ValueError):
        _build(sample_count=0)
    with pytest.raises(ValueError):
        _build(batch_size=0)
