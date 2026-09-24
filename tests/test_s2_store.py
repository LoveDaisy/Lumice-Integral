"""``s2_store``: building, the disk cache and its refusals, band slices, ``Phi`` groups, task 13 regression."""

from __future__ import annotations

import json
import dataclasses
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pytest

from lumice_integral import geometry, optics, s2_store
from lumice_integral.camera import sun_direction
from lumice_integral.canonical_scene import (
    CANONICAL_REFRACTIVE_INDEX,
    canonical_crystal,
    canonical_incident_direction,
    canonical_sun_direction,
)
from lumice_integral.s2_store import (
    RandomSphereSampler,
    S2EventStore,
    S2StoreSpec,
    align_rotations,
    build_event_store,
    build_or_load,
    events_from_schema1,
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
        **kwargs,
    )


@pytest.fixture(scope="module")
def small_store() -> S2EventStore:
    return _build()


def test_events_match_the_batch_evaluators_directly(small_store) -> None:
    """Oracle: the three production evaluators called here, not through ``evaluate_fields``.

    The store's ``u`` is ``R^-1 s_hat`` with ``s_hat`` toward the sun (above the
    horizon), on the antipodal Fibonacci lattice; the evaluators take the
    propagation direction ``s = -s_hat``.
    """
    sun, crystal, index = canonical_sun_direction(), canonical_crystal(), CANONICAL_REFRACTIVE_INDEX
    s = canonical_incident_direction()
    assert sun[2] > 0.0 and np.array_equal(s, -sun)
    u = -fibonacci_sphere(SMALL_N)
    rotations = align_rotations(u, sun)
    assert np.max(np.abs(np.einsum("nij,nj->ni", rotations, u) - sun)) <= 1e-15  # R u = s_hat
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
    # D is the angle between the body-frame incoming ray -u and the outgoing phi
    cosine = np.sum(events.phi * -events.u, axis=1)
    assert np.max(np.abs(np.cos(events.D) - cosine)) <= 1e-12
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


def _outgoing_offsets(rotations: np.ndarray, faces, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Validity and angle to ``target`` of the production outgoing direction (the oracle of the seed poses)."""
    domain = optics.path_domain_batch(rotations, faces, canonical_incident_direction(), CANONICAL_REFRACTIVE_INDEX)
    return domain.valid, np.arccos(np.clip(domain.direction @ target, -1.0, 1.0))


@pytest.mark.parametrize("g_name", ["identity", "proper", "improper"])
def test_store_seeds_pose_the_band_onto_the_target(small_store, g_name: str) -> None:
    """Every candidate is a valid pose of its path whose outgoing direction is ``|D_i - delta|`` from the target.

    ``proper``: the rotation ``Rz(60)`` serves member ``4-6`` of the ``3-5``
    class; ``improper``: the basal mirror maps ``3-5`` onto itself.
    """
    from lumice_integral.path_class import hexprism_symmetry_matrices

    elements = hexprism_symmetry_matrices()
    g, faces = {
        "identity": (None, (3, 5)),
        "proper": (elements[2], (4, 6)),
        "improper": (np.diag([1.0, 1.0, -1.0]), (3, 5)),
    }[g_name]
    sun = canonical_sun_direction()
    seeds = s2_store.StoreSeeds(small_store, faces, sun, g)
    target = optics.path_domain_batch(
        align_rotations(small_store.events.u[500:501], sun), (3, 5), canonical_incident_direction(), CANONICAL_REFRACTIVE_INDEX
    ).direction[0]
    delta = np.arccos(target @ canonical_incident_direction())
    half_width = np.radians(0.5)
    rotations, offsets = seeds.candidates(target, half_width)
    expected = np.count_nonzero(np.abs(small_store.events.D - delta) <= half_width)
    assert len(rotations) == expected > 50
    assert np.all(offsets <= half_width)
    assert np.allclose(np.linalg.det(rotations), 1.0) and np.allclose(rotations @ rotations.transpose(0, 2, 1), np.eye(3))
    valid, angle = _outgoing_offsets(rotations, faces, target)
    assert valid.all()
    np.testing.assert_allclose(angle, offsets, atol=1e-9)
    if g is None:
        lo, hi = np.searchsorted(small_store.events.D, [delta - half_width, np.nextafter(delta + half_width, np.inf)])
        np.testing.assert_allclose(np.einsum("nij,nj->ni", rotations, small_store.events.u[lo:hi]), np.broadcast_to(sun, (hi - lo, 3)), atol=1e-12)


def test_store_seeds_refuse_a_path_the_store_does_not_serve(small_store) -> None:
    sun = canonical_sun_direction()
    with pytest.raises(ValueError, match="not a member"):
        s2_store.StoreSeeds(small_store, (3, 7), sun)
    with pytest.raises(ValueError, match="image under g"):
        s2_store.StoreSeeds(small_store, (3, 7), sun, np.diag([1.0, 1.0, -1.0]))
    with pytest.raises(ValueError, match="half_width"):
        s2_store.StoreSeeds(small_store, (3, 5), sun).candidates(canonical_incident_direction(), 0.0)


def test_float32_store_is_the_float64_store_cast(small_store) -> None:
    single = _build(dtype="float32")
    for name, array in single.events.arrays().items():
        assert array.dtype == np.float32
        assert np.array_equal(array, small_store.events.arrays()[name].astype(np.float32)), name


def _sorted_all_at_once(members, n: int, **kwargs) -> dict[str, np.ndarray]:
    """The schema 2 build: every kept row of ``evaluate_fields`` concatenated, one stable argsort on ``D``."""
    sun, crystal, index = s2_store._REFERENCE_SUN_DIRECTION, canonical_crystal(), CANONICAL_REFRACTIVE_INDEX
    kept: dict[str, list[np.ndarray]] = {"D": [], "u": [], "phi": [], "w": []}
    for first in range(0, n, kwargs.get("chunk", s2_store.CHUNK)):
        u = s2_store.store_lattice(n, first, min(first + kwargs.get("chunk", s2_store.CHUNK), n))
        fields = s2_store.evaluate_fields(align_rotations(u, sun), sun, crystal, index, members)
        keep = fields["w"] > 0.0
        for name in kept:
            kept[name].append((u if name == "u" else fields[name])[keep])
    arrays = {name: np.concatenate(parts) for name, parts in kept.items()}
    order = np.argsort(arrays["D"], kind="stable")
    return {name: array[order] for name, array in arrays.items()}


@pytest.mark.parametrize("bucket_count", [1, 7, s2_store.DEFAULT_BUCKET_COUNT, 100_000])
def test_bucketed_build_is_the_one_stable_sort(bucket_count: int, monkeypatch) -> None:
    """Bucket sort in ``D`` = one global stable argsort, bit for bit, for any bucket count (several chunks)."""
    monkeypatch.setattr(s2_store, "CHUNK", 3_000)  # 7 chunks at SMALL_N: rows of one bucket from many chunks
    reference = _sorted_all_at_once(((3, 5),), SMALL_N, chunk=3_000)
    store = _build(bucket_count=bucket_count)
    assert store.diagnostics["bucket_count"] == bucket_count
    assert store.diagnostics["largest_bucket_events"] <= len(store.events)
    assert sorted(store.events.arrays()) == sorted(reference)
    for name, array in reference.items():
        assert np.array_equal(store.events.arrays()[name], array), name
    assert store.spec.cache_key() == _build(n=SMALL_N, bucket_count=1).spec.cache_key()  # not a build parameter
    assert "bucket_count" not in store.spec.build_parameters()


def test_bucket_edges_cover_the_deviation_domain() -> None:
    edges = s2_store._bucket_edges(None, 16)
    assert edges[0] == 0.0 and edges[-1] == np.pi and np.all(np.diff(edges) > 0)
    window = s2_store._bucket_edges((0.4, 0.5), 4)
    assert window[0] == 0.4 and window[-1] == 0.5 and len(window) == 5


def test_inverse_weights_are_stored_and_cached(tmp_path) -> None:
    """A non-uniform sampler's ``iw`` rides along the sort as its own array and its own ``.npy``."""
    n = 5_000

    def sampler(first: int, stop: int) -> tuple[np.ndarray, np.ndarray]:
        return s2_store.store_lattice(n, first, stop), np.linspace(0.5, 1.5, n)[first:stop]

    kwargs = {"sampler": sampler, "sampling": "the store lattice with made-up inverse weights"}
    weighted, uniform = _build(n=n, bucket_count=5, **kwargs), _build(n=n)
    for name, array in uniform.events.arrays().items():
        assert np.array_equal(weighted.events.arrays()[name], array), name
    expected = dict(zip(map(bytes, s2_store.store_lattice(n)), np.linspace(0.5, 1.5, n)))
    assert np.array_equal(weighted.events.iw, [expected[bytes(row)] for row in weighted.events.u])
    cached = _cached(tmp_path, n=n, **kwargs)
    assert (tmp_path / cached.spec.cache_key() / "iw.npy").exists()
    assert np.array_equal(cached.events.iw, weighted.events.iw)
    assert np.array_equal(S2EventStore.load(tmp_path / cached.spec.cache_key(), mmap_mode="r").events.iw, weighted.events.iw)


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


# Sun directions of the owner probe (docs/phase2.md section 1.1): the canonical one, a high sun, one below the horizon.
REFERENCE_SUNS = ((15.0, 0.0), (60.0, 37.0), (-30.0, 200.0))


@pytest.mark.parametrize("members", [((3, 5),), ((1, 3, 2),), ((1, 3, 5, 2),)], ids=["3-5", "1-3-2", "1-3-5-2"])
def test_events_are_independent_of_the_reference_direction(members) -> None:
    """Section 4.1(a): the events depend on the pose only through ``u``, whatever ``s_hat`` the build aligns to.

    ``build_event_store`` aligns to one fixed reference direction, so the
    claim is checked on the two general functions it uses, point by point on
    the lattice: the same kept points (hence equal event count and ``u`` bit
    for bit), ``phi`` / ``D`` / ``w`` to ``1e-10``.
    """
    crystal, index = canonical_crystal(), CANONICAL_REFRACTIVE_INDEX
    u = s2_store.store_lattice(200_000)
    results = []
    for altitude, azimuth in REFERENCE_SUNS:
        sun = sun_direction(altitude, azimuth)
        fields = s2_store.evaluate_fields(align_rotations(u, sun), sun, crystal, index, members)
        assert np.max(np.abs(fields["u"] - u)) <= 1e-12  # R^-1 s_hat is the lattice point
        results.append(fields)
    reference = results[0]
    keep = reference["w"] > 0.0
    assert np.count_nonzero(keep) > 0
    for other in results[1:]:
        assert np.array_equal(other["w"] > 0.0, keep)
        for name in ("phi", "D", "w"):
            assert np.max(np.abs(other[name][keep] - reference[name][keep])) <= 1e-10, name


def test_the_build_reference_direction_is_the_canonical_sun_numerically() -> None:
    """The one fixed ``s_hat`` of every build; equal to the canonical sun only so schema 2 builds stay bit for bit."""
    assert np.array_equal(s2_store._REFERENCE_SUN_DIRECTION, canonical_sun_direction())


# ------------------------------------------------------------------ cache
def _cached(tmp_path: Path, **kwargs) -> S2EventStore:
    return build_or_load(
        canonical_crystal(),
        CANONICAL_REFRACTIVE_INDEX,
        [(3, 5)],
        kwargs.pop("n", 5_000),
        base_dir=tmp_path,
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
    assert list(provenance["arrays"]) == ["D", "u", "phi", "w"]
    assert sorted(p.name for p in directory.iterdir()) == ["D.npy", "phi.npy", "provenance.json", "u.npy", "w.npy"]
    for name, record in provenance["arrays"].items():  # one plain .npy per array, as np.save writes it
        assert record["file"] == f"{name}.npy" and record["bytes"] == (directory / record["file"]).stat().st_size
        assert np.array_equal(np.load(directory / record["file"]), built.events.arrays()[name])
    assert not any(p.name.startswith(".") for p in tmp_path.iterdir())  # no staging directory left behind
    in_memory = _build(n=5_000)
    for name, array in in_memory.events.arrays().items():
        assert np.array_equal(built.events.arrays()[name], array), name

    def no_rebuild(*args, **kwargs):
        raise AssertionError("a cache hit must not rebuild")

    monkeypatch.setattr(s2_store, "_build_into", no_rebuild)
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
        dataclasses.replace(spec, refractive_index=1.3110129),
        dataclasses.replace(spec, members=((3, 7),)),
        dataclasses.replace(spec, crystal={"type": "HexPrism", "a": 1.0, "h": 0.1}),
        dataclasses.replace(spec, n=spec.n + 1),
        dataclasses.replace(spec, sampling=RandomSphereSampler.description),
        dataclasses.replace(spec, dtype="float32"),
        dataclasses.replace(spec, deviation_window=(0.4, 0.5)),
    ):
        assert changed.cache_key() != spec.cache_key()
    assert "sun_direction" not in spec.build_parameters()  # schema 3: a store serves every sun
    assert S2StoreSpec.from_build_parameters(spec.build_parameters()) == spec


def _tamper(directory: Path, name: str, *, same_size: bool) -> None:
    """Rewrite ``<name>.npy`` with one value changed (same file size) or one row dropped."""
    array = np.load(directory / f"{name}.npy")
    if same_size:
        array = array.copy()
        array[len(array) // 2] *= 1.5
    else:
        array = array[:-1]
    np.save(directory / f"{name}.npy", array)


@pytest.mark.parametrize("same_size", [True, False])
def test_cache_refuses_a_modified_array(tmp_path, same_size: bool) -> None:
    built = _cached(tmp_path)
    directory = tmp_path / built.spec.cache_key()
    _tamper(directory, "w", same_size=same_size)
    with pytest.raises(ValueError, match="SHA-256 of array 'w'"):
        _cached(tmp_path)
    with pytest.raises(ValueError, match="SHA-256 of array 'w'"):
        S2EventStore.verify(directory)


def test_mmap_load_maps_the_arrays_and_band_slices_stay_views(tmp_path) -> None:
    built = _cached(tmp_path)
    directory = tmp_path / built.spec.cache_key()
    mapped = S2EventStore.load(directory, mmap_mode="r")
    assert mapped.spec == built.spec
    for name, array in mapped.events.arrays().items():
        assert isinstance(array, np.memmap) and not array.flags.writeable, name
        assert np.array_equal(array, built.events.arrays()[name]), name
    lo, hi = np.radians(23.0), np.radians(25.0)
    band, reference = mapped.band_slice(lo, hi), built.band_slice(lo, hi)
    assert len(band) == len(reference) > 0
    for name, array in band.arrays().items():
        assert np.array_equal(array, reference.arrays()[name]), name
        assert np.shares_memory(array, mapped.events.arrays()[name]), name  # a view of the mapping
    with pytest.raises(ValueError, match="mmap_mode"):
        S2EventStore.load(directory, mmap_mode="r+")


def test_mmap_load_checks_sizes_only_and_verify_hashes(tmp_path) -> None:
    """The documented trade-off: a same-size change passes ``mmap_mode="r"`` and is caught by ``verify``."""
    built = _cached(tmp_path)
    directory = tmp_path / built.spec.cache_key()
    S2EventStore.verify(directory)  # intact
    _tamper(directory, "phi", same_size=True)
    S2EventStore.load(directory, mmap_mode="r")
    with pytest.raises(ValueError, match="SHA-256 of array 'phi'"):
        S2EventStore.verify(directory)
    _tamper(directory, "phi", same_size=False)
    with pytest.raises(ValueError, match="bytes"):
        S2EventStore.load(directory, mmap_mode="r")
    with pytest.raises(ValueError, match="bytes"):
        _cached(tmp_path, mmap_mode="r")


def test_sweep_stale_staging_removes_old_but_keeps_fresh(tmp_path) -> None:
    """code-review-01.md Major 1: a killed ``build_or_load`` leaks its staging dir; age reclaims it."""
    stale = tmp_path / ".deadkey.building-abc123"
    stale.mkdir()
    (stale / "junk.npy").write_bytes(b"x")
    old = time.time() - s2_store._STAGING_STALE_SECONDS - 60
    os.utime(stale, (old, old))

    fresh = tmp_path / ".deadkey.building-def456"
    fresh.mkdir()

    s2_store._sweep_stale_staging(tmp_path)

    assert not stale.exists()  # abandoned past the staleness threshold: reclaimed
    assert fresh.exists()  # within the threshold: left alone, a concurrent builder may still own it


def test_build_or_load_sweeps_a_stale_staging_directory_from_a_prior_run(tmp_path) -> None:
    """The sweep runs on every ``build_or_load`` call, not just when a build is needed."""
    orphan = tmp_path / ".some-other-key.building-xyz789"
    orphan.mkdir()
    old = time.time() - s2_store._STAGING_STALE_SECONDS - 60
    os.utime(orphan, (old, old))

    _cached(tmp_path)  # any call sweeps base_dir first, regardless of this spec's own cache key

    assert not orphan.exists()


def test_rename_race_with_an_existing_target_loads_the_winners_store(tmp_path, monkeypatch) -> None:
    """code-review-01.md Minor 2: a ``staging.rename`` failing with ``OSError`` while the target already
    exists is the code's one documented case for "someone else won the race"; pin that it actually loads
    that store instead of merely swallowing the exception."""
    scratch = tmp_path / "scratch"
    winner_store = _cached(scratch)
    winner_dir = scratch / winner_store.spec.cache_key()
    target_dir = tmp_path / winner_store.spec.cache_key()

    real_rename = Path.rename

    def fake_rename(self, target):
        target = Path(target)
        if target == target_dir:
            shutil.copytree(winner_dir, target_dir)  # the "other builder" lands first
            raise OSError("simulated: another builder already renamed into place")
        return real_rename(self, target)

    monkeypatch.setattr(Path, "rename", fake_rename)

    loaded = _cached(tmp_path)  # target_dir does not exist yet: build_or_load must attempt to build + rename

    assert loaded.spec == winner_store.spec
    for name, array in winner_store.events.arrays().items():
        assert np.array_equal(loaded.events.arrays()[name], array), name
    assert not any(p.name.startswith(".") for p in tmp_path.iterdir())  # the losing staging dir was cleaned up


def test_cache_refuses_mismatched_recorded_parameters(tmp_path) -> None:
    built = _cached(tmp_path)
    provenance_path = tmp_path / built.spec.cache_key() / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    provenance["build"]["refractive_index"] = 1.32
    provenance_path.write_text(json.dumps(provenance))
    with pytest.raises(ValueError, match="refractive_index"):
        _cached(tmp_path)


@pytest.mark.parametrize("schema", [1, 2])
def test_cache_refuses_an_older_schema(tmp_path, schema: int) -> None:
    """Schema 1 (``u = R^-1 s``, propagation) and schema 2 (sun direction in the key) are refused, never converted."""
    built = _cached(tmp_path)
    assert s2_store.SCHEMA_VERSION == 3
    directory = tmp_path / built.spec.cache_key()
    provenance_path = directory / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    provenance["build"]["schema_version"] = schema
    if schema == 2:
        provenance["build"]["sun_direction"] = canonical_sun_direction().tolist()
    provenance_path.write_text(json.dumps(provenance))
    with pytest.raises(ValueError, match=f"schema_version {schema}"):
        S2EventStore.load(directory)
    with pytest.raises(ValueError, match=f"schema_version {schema}"):
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
    """``scripts/probe_band_sum.py --stage precompute`` output of task ``band-sum-quadrature-probe``.

    That store is schema 1 (``u = R^-1 s``, propagation): after the named
    conversion (``u`` negated) every array is equal bit for bit, i.e. the
    schema 2 build changed the sign convention of ``u`` and nothing else.
    """
    path = TASK13_ARTIFACTS / f"events_N{n}.npz"
    if not path.exists():
        pytest.skip(f"{path} not present (scratchpad artifact of task 13)")
    store = _build(n=n)
    with np.load(path) as data:
        legacy = events_from_schema1({name: data[name] for name in data.files})
    assert sorted(store.events.arrays()) == sorted(legacy)
    for name, array in legacy.items():
        assert np.array_equal(store.events.arrays()[name], array), name
