"""``s2_store``: building, the disk cache and its refusals, band slices, ``Phi`` groups, task 13 regression."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from lumice_integral import geometry, optics, s2_store
from lumice_integral.canonical_scene import CANONICAL_REFRACTIVE_INDEX, canonical_crystal, canonical_incident_direction
from lumice_integral.s2_store import (
    RandomSphereSampler,
    S2EventStore,
    S2StoreSpec,
    align_rotations,
    build_event_store,
    build_or_load,
    fibonacci_sphere,
)

TASK13_ARTIFACTS = Path(__file__).resolve().parents[1] / "scratchpad/task-band-sum-quadrature-probe/artifacts"
SMALL_N = 20_000


def _build(members=((3, 5),), n: int = SMALL_N, crystal=None, **kwargs) -> S2EventStore:
    kwargs.setdefault("run_checks", False)
    return build_event_store(
        canonical_crystal() if crystal is None else crystal,
        CANONICAL_REFRACTIVE_INDEX,
        members,
        n,
        incident_direction=canonical_incident_direction(),
        **kwargs,
    )


@pytest.fixture(scope="module")
def small_store() -> S2EventStore:
    return _build()


def test_events_match_the_batch_evaluators_directly(small_store) -> None:
    """Oracle: the three production evaluators called here, not through ``evaluate_fields``."""
    s, crystal, index = canonical_incident_direction(), canonical_crystal(), CANONICAL_REFRACTIVE_INDEX
    u = fibonacci_sphere(SMALL_N)
    rotations = align_rotations(u, s)
    w = optics.fresnel_transmission_path_batch(rotations, (3, 5), s, index) * geometry.entry_measure_batch(
        rotations, (3, 5), s, crystal, n_ice=index
    )
    keep = w > 0.0
    events = small_store.events
    assert len(events) == int(np.count_nonzero(keep)) > 0
    assert np.all(np.diff(events.D) >= 0.0)
    ours, theirs = np.lexsort(events.u.T), np.lexsort(u[keep].T)  # both sides in the same u order
    assert np.array_equal(events.u[ours], u[keep][theirs])
    assert np.array_equal(events.w[ours], w[keep][theirs])
    assert events.iw is None
    assert all(a.flags["C_CONTIGUOUS"] for a in events.arrays().values())
    assert small_store.diagnostics["kept_events"] == len(events)


def test_self_checks_pass() -> None:
    store = _build(n=5_000, run_checks=True)
    checks = store.diagnostics["self_checks"]
    assert checks["psi_invariance"]["passed"]
    assert checks["haar_mean"]["passed"]
    assert checks["gate_coverage"]["ok"] > 0 and len(checks["gate_coverage"]) > 1


def test_deviation_window_equals_filtering_the_full_store(small_store) -> None:
    lo, hi = np.radians(22.5), np.radians(25.0)
    windowed = _build(deviation_window=(lo, hi))
    keep = (small_store.events.D >= lo) & (small_store.events.D <= hi)
    for name, array in windowed.events.arrays().items():
        assert np.array_equal(array, small_store.events.arrays()[name][keep]), name


def test_band_slice(small_store) -> None:
    D = small_store.events.D
    empty = small_store.band_slice(0.5, 0.4)
    assert len(empty) == 0 and empty.u.shape == (0, 3)
    everything = small_store.band_slice(D[0], np.nextafter(D[-1], np.inf))
    assert len(everything) == len(D)
    lo_d, hi_d = np.radians(23.0), np.radians(23.5)
    band = small_store.band_slice(lo_d, hi_d)
    lo, hi = np.searchsorted(D, [lo_d, hi_d])
    assert np.array_equal(band.D, D[lo:hi]) and np.array_equal(band.u, small_store.events.u[lo:hi])
    assert np.shares_memory(band.u, small_store.events.u)  # a view, not a copy
    assert np.all((band.D >= lo_d) & (band.D < hi_d))


def test_float32_store_is_the_float64_store_cast(small_store) -> None:
    single = _build(dtype="float32")
    for name, array in single.events.arrays().items():
        assert array.dtype == np.float32
        assert np.array_equal(array, small_store.events.arrays()[name].astype(np.float32)), name


def test_random_sampler_is_reproducible_and_needs_a_description() -> None:
    sampler = RandomSphereSampler()
    first = _build(n=5_000, sampler=sampler, sampling=sampler.description)
    second = _build(n=5_000, sampler=RandomSphereSampler(), sampling=sampler.description)
    assert np.array_equal(first.events.u, second.events.u)
    assert first.spec.cache_key() != _build(n=5_000).spec.cache_key()
    with pytest.raises(ValueError, match="sampling description"):
        _build(n=5_000, sampler=sampler)


def test_phi_group_sums_member_weights_on_the_same_points() -> None:
    """``w_Phi = w_(3-5) + w_(3-1-2-5)`` event by event; kept where either member has ``w > 0``.

    On a plate (``h / a = 0.3``): the canonical column has no ``3-1-2-5`` events.
    """
    members = ((3, 5), (3, 1, 2, 5))
    plate = geometry.HexPrism.from_ratio(0.3)
    group = _build(members=members, crystal=plate)
    singles = [_build(members=(m,), crystal=plate) for m in members]
    assert group.spec.path_id == "3-5+3-1-2-5"
    by_point = {row.tobytes(): w for row, w in zip(group.events.u, group.events.w)}
    total: dict[bytes, float] = {}
    for single in singles:
        assert len(single.events) > 0
        for row, w in zip(single.events.u, single.events.w):
            total[row.tobytes()] = total.get(row.tobytes(), 0.0) + w
    assert by_point.keys() == total.keys()
    assert max(abs(by_point[k] - total[k]) for k in total) <= 1e-15
    assert len(group.events) < sum(len(s.events) for s in singles)  # the members overlap on S^2
    with pytest.raises(ValueError, match="phi_key"):
        _build(members=((3, 5), (3, 7)))


# ------------------------------------------------------------------ cache
def _cached(tmp_path: Path, **kwargs) -> S2EventStore:
    return build_or_load(
        canonical_crystal(),
        CANONICAL_REFRACTIVE_INDEX,
        [(3, 5)],
        kwargs.pop("n", 5_000),
        base_dir=tmp_path,
        incident_direction=canonical_incident_direction(),
        run_checks=False,
        **kwargs,
    )


def test_cache_writes_then_hits(tmp_path, monkeypatch) -> None:
    built = _cached(tmp_path)
    directory = tmp_path / built.spec.cache_key()
    provenance = json.loads((directory / "provenance.json").read_text())
    assert provenance["build"] == built.spec.build_parameters()
    assert provenance["build"]["schema_version"] == s2_store.SCHEMA_VERSION
    assert set(provenance) >= {"git_commit", "arrays", "diagnostics", "cache_key"}
    assert provenance["diagnostics"]["kept_events"] == len(built.events)

    def no_rebuild(*args, **kwargs):
        raise AssertionError("a cache hit must not rebuild")

    monkeypatch.setattr(s2_store, "build_event_store", no_rebuild)
    loaded = _cached(tmp_path)
    assert loaded.spec == built.spec
    for name, array in built.events.arrays().items():
        assert np.array_equal(loaded.events.arrays()[name], array)


def test_cache_key_separates_parameters(tmp_path) -> None:
    a = _cached(tmp_path, n=5_000)
    b = _cached(tmp_path, n=6_000)
    assert a.spec.cache_key() != b.spec.cache_key()
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted([a.spec.cache_key(), b.spec.cache_key()])
    spec = a.spec
    for changed in (
        S2StoreSpec(spec.members, spec.crystal, 1.3110129, spec.incident_direction, spec.n),
        S2StoreSpec(((3, 7),), spec.crystal, spec.refractive_index, spec.incident_direction, spec.n),
        S2StoreSpec(spec.members, {"type": "HexPrism", "a": 1.0, "h": 0.1}, spec.refractive_index, spec.incident_direction, spec.n),
        S2StoreSpec(spec.members, spec.crystal, spec.refractive_index, spec.incident_direction, spec.n, dtype="float32"),
        S2StoreSpec(spec.members, spec.crystal, spec.refractive_index, spec.incident_direction, spec.n, deviation_window=(0.4, 0.5)),
    ):
        assert changed.cache_key() != spec.cache_key()
    assert S2StoreSpec.from_build_parameters(spec.build_parameters()) == spec


def test_cache_refuses_a_modified_events_file(tmp_path) -> None:
    built = _cached(tmp_path)
    events_path = tmp_path / built.spec.cache_key() / "events.npz"
    arrays = built.events.arrays()
    np.savez(events_path, **{**arrays, "w": arrays["w"][:-1]})
    with pytest.raises(ValueError, match="SHA-256"):
        _cached(tmp_path)


def test_cache_refuses_mismatched_recorded_parameters(tmp_path) -> None:
    built = _cached(tmp_path)
    provenance_path = tmp_path / built.spec.cache_key() / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    provenance["build"]["refractive_index"] = 1.32
    provenance_path.write_text(json.dumps(provenance))
    with pytest.raises(ValueError, match="refractive_index"):
        _cached(tmp_path)


def test_cache_refuses_an_interrupted_save_and_never_overwrites(tmp_path) -> None:
    built = _cached(tmp_path)
    with pytest.raises(FileExistsError):
        built.save(tmp_path)
    (tmp_path / built.spec.cache_key() / "provenance.json").unlink()
    with pytest.raises(ValueError, match="provenance"):
        _cached(tmp_path)


def test_crystal_must_be_an_untransformed_hexprism() -> None:
    crystal = canonical_crystal()
    assert s2_store.crystal_from_description(s2_store.crystal_description(crystal)).h == crystal.h
    moved = crystal.transformed(np.eye(3), np.array([0.1, 0.0, 0.0]))
    with pytest.raises(ValueError, match="untransformed"):
        s2_store.crystal_description(moved)


# ------------------------------------------------------- task 13 regression
@pytest.mark.parametrize("n", [1_000_000, pytest.param(10_000_000, marks=pytest.mark.slow)])
def test_rebuilds_task13_store_bit_for_bit(n: int) -> None:
    """``scripts/probe_band_sum.py --stage precompute`` output of task ``band-sum-quadrature-probe``."""
    path = TASK13_ARTIFACTS / f"events_N{n}.npz"
    if not path.exists():
        pytest.skip(f"{path} not present (scratchpad artifact of task 13)")
    store = _build(n=n)
    with np.load(path) as data:
        assert sorted(store.events.arrays()) == sorted(data.files)
        for name in data.files:
            assert np.array_equal(store.events.arrays()[name], data[name]), name
