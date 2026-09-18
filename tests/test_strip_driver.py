"""Strip orchestration and on-disk format: synthetic results, round trip, small windows."""

from __future__ import annotations

import json
import multiprocessing
import pickle
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from lumice_integral.strip_driver import (
    WORKER_MALLOC_ENV,
    DriverOptions,
    PrescanBuildOptions,
    _merge_subpixels,
    load_checkpoints,
    render_column,
    render_window,
)
from lumice_integral.strip_io import (
    FILE_NAMES,
    FORMAT_VERSION,
    PIXEL_CSV_COLUMNS,
    STATUS_BITS,
    Window,
    assemble_arrays,
    read_strip,
    write_strip,
)
from lumice_integral.strip_pixel import (
    EVENT_NAMES,
    STATUS_HAS_ARC,
    STATUS_HAS_COMPONENT,
    STATUS_RENDERED,
    STATUS_UNKNOWN_COMPLETENESS,
    ComponentRecord,
    PixelOptions,
    PixelResult,
    canonical_strip_scene,
    render_pixel,
)


def _component(value: float, arclength: float = 4.7, kind: str = "closed") -> ComponentRecord:
    arc = kind == "arc"
    return ComponentRecord(
        seed=np.eye(3),
        kind=kind,
        arclength=arclength,
        pose_count=120,
        status="event_terminated" if arc else "closed",
        reason="tir_boundary" if arc else "closed_loop",
        start_reason="visibility_boundary" if arc else "",
        quadrature_status="available",
        value=value,
        error_estimate=1e-7,
        start_truncation_estimate=1e-5 if arc else float("nan"),
        end_truncation_estimate=2e-5 if arc else float("nan"),
        node_count=257,
        refinement_rounds=1,
        node_count_exhausted=False,
        non_finite_node_count=0,
    )


def _pixel(row: int, column: int, value: float, *, completeness: str = "complete", kind: str = "closed", events=None) -> PixelResult:
    components = (_component(value, kind=kind),) if value else ()
    return PixelResult(
        row=row,
        column=column,
        value=value,
        error_estimate=1e-7 * len(components),
        components=components,
        completeness=completeness,
        incomplete_count=0,
        pool_count=0,
        extra_seed_count=0,
        raw_cluster_count=0,
        admissible_count=len(components),
        events={name: 0 for name in EVENT_NAMES} | (events or {}),
        timings={"discovery_s": 0.2, "trace_s": 0.1, "quadrature_s": 1.0, "total_s": 1.3},
    )


# --- pure format tests (no solver) --------------------------------------------


def test_window_validates_and_counts():
    window = Window((10, 12), (3, 5))
    assert window.pixel_count == 4
    assert list(window.row_range) == [10, 11]
    with pytest.raises(ValueError):
        Window((5, 5), (0, 1))
    strided = Window((0, 2), (0, 251), column_step=25)
    assert list(strided.column_range) == list(range(0, 251, 25))
    assert strided.pixel_count == 22 and strided.as_json()["column_step"] == 25
    with pytest.raises(ValueError):
        Window((0, 1), (0, 1), column_step=0)


def test_assemble_arrays_keeps_unknown_pixels_distinguishable_from_complete_zeros():
    results = [
        _pixel(0, 0, 1.5),
        _pixel(0, 1, 0.0, completeness="unknown", events={"incomplete_candidate": 3}),
        _pixel(1, 0, 0.0),  # dark, complete
    ]
    arrays = assemble_arrays(results, height=2, width=2)
    assert arrays.values.tolist() == [[1.5, 0.0], [0.0, 0.0]]
    assert arrays.status[0, 0] == STATUS_RENDERED | STATUS_HAS_COMPONENT
    assert arrays.status[0, 1] == STATUS_RENDERED | STATUS_UNKNOWN_COMPLETENESS
    assert arrays.status[1, 0] == STATUS_RENDERED
    assert arrays.status[1, 1] == 0  # not rendered
    assert arrays.rendered.tolist() == [[True, True], [True, False]]
    assert arrays.component_count.tolist() == [[1, 0], [0, 0]]
    # The three zero-valued pixels are three different things in the status layer.
    assert len({int(arrays.status[0, 1]), int(arrays.status[1, 0]), int(arrays.status[1, 1])}) == 3


def test_write_and_read_strip_round_trip_with_verified_hashes(tmp_path: Path):
    results = [_pixel(r, c, float(r * 10 + c) + 0.25, kind="arc" if (r, c) == (1, 2) else "closed") for r in range(2) for c in range(3)]
    window = Window((0, 2), (0, 3))
    files = write_strip(
        tmp_path,
        results,
        options=PixelOptions(),
        window=window,
        height=4,
        width=3,
        pixel_model={"model": "point", "epsilon": 1e-6},
        execution={"wall_clock_s": 1.0, "workers": 1},
        prescan=PrescanBuildOptions(sample_count=123, rng_seed=7).as_json(),
    )
    assert set(files) == set(FILE_NAMES)
    assert all(path.exists() for path in files.values())
    raw64 = np.fromfile(files["float64"], dtype="<f8").reshape(4, 3)
    raw32 = np.fromfile(files["float32"], dtype="<f4").reshape(4, 3)
    assert raw64[1, 2] == 12.25 and raw64[3, 0] == 0.0
    assert np.array_equal(raw32, raw64.astype(np.float32))
    assert files["float32"].stat().st_size == 4 * 3 * 4

    arrays, provenance = read_strip(tmp_path)
    assert np.array_equal(arrays.values, raw64)
    assert arrays.rendered.sum() == 6
    assert provenance["format"] == FORMAT_VERSION == "lumice-integral.strip/v2"
    assert provenance["arrays"]["shape"] == [4, 3]
    assert provenance["arrays"]["status_bits"] == STATUS_BITS
    assert provenance["window"] == {"rows": [0, 2], "columns": [0, 3], "column_step": 1, "pixel_count": 6}
    assert provenance["options"]["quadrature"]["relative_tolerance"] == 1e-4
    assert provenance["options"]["quadrature"]["initial_node_count"] == 129
    assert provenance["options"]["quadrature"]["method"].startswith("composite Simpson on a uniform grid")
    assert provenance["options"]["discovery"]["distance_threshold"] == PixelOptions().distance_threshold
    assert provenance["options"]["discovery"]["prescan"] == {"sample_count": 123, "rng_seed": 7, "cache_path": None}
    assert "prescan table" in provenance["options"]["discovery"]["strategy"]
    assert "open-arc" in provenance["options"]["discovery"]["strategy"]
    assert "traced once" in provenance["options"]["discovery"]["strategy"]
    assert provenance["options"]["continuation"]["maximum_accepted_steps"] == 4000
    assert set(provenance["options"]["discovery"]) == {"prescan", "angle_tolerance_deg", "cluster_radius_rad", "distance_threshold", "strategy"}
    assert provenance["scene"]["refractive_index"] == {"value": 1.31, "provenance": "canonical-new"}
    assert provenance["scene"]["sun"]["provenance"] == "historical-inferred"
    assert provenance["summary"]["rendered_pixels"] == 6
    assert provenance["summary"]["pixels_with_components"] == 6
    assert provenance["summary"]["pixels_with_arcs"] == 1
    assert provenance["summary"]["event_totals"] == {name: 0 for name in EVENT_NAMES}
    assert arrays.status[1, 2] == STATUS_RENDERED | STATUS_HAS_COMPONENT | STATUS_HAS_ARC
    assert set(provenance["arrays"]["status_bits"]) == {"rendered", "unknown_completeness", "has_component", "has_arc", "quadrature_unavailable", "node_count_exhausted"}
    assert provenance["pixel_model"]["model"] == "point"
    assert "Lumice" in provenance["generator"]["lumice_dependency"]
    assert provenance["environment"]["jax"]
    assert set(WORKER_MALLOC_ENV) <= set(provenance["environment"]["env"])
    json.dumps(provenance)  # serialisable without solver objects

    with files["pixels"].open() as handle:
        header = handle.readline().strip().split(",")
        rows = handle.read().strip().splitlines()
    assert tuple(header) == PIXEL_CSV_COLUMNS
    assert len(rows) == 6
    arc_row = next(r for r in rows if r.startswith("1,2,"))
    cells = dict(zip(header, arc_row.split(",")))
    assert cells["component_kinds"] == "arc" and cells["arc_count"] == "1"
    assert cells["component_end_reasons"] == "visibility_boundary|tir_boundary"
    assert cells["component_start_truncations"] == "1e-05" and cells["component_end_truncations"] == "2e-05"
    closed_row = dict(zip(header, next(r for r in rows if r.startswith("0,1,")).split(",")))
    assert closed_row["component_end_reasons"] == "closed_loop" and closed_row["component_start_truncations"] == "nan"

    # Tampering is detected on read.
    files["float64"].write_bytes(b"\x00" * files["float64"].stat().st_size)
    with pytest.raises(ValueError, match="sha256"):
        read_strip(tmp_path)


def _checkpoint_without_pixel_fields(path: Path, *fields: str) -> None:
    """Write a checkpoint whose pickled ``PixelOptions`` lacks ``fields``.

    Simulates a checkpoint written before those fields existed: an unpickled
    frozen dataclass gets its state through ``__dict__.update`` and never sees
    a key that was not pickled.
    """
    options = DriverOptions()
    pixel_state = {k: v for k, v in options.pixel.__dict__.items() if k not in fields}
    old_pixel = object.__new__(PixelOptions)
    old_pixel.__dict__.update(pixel_state)
    old_options = object.__new__(DriverOptions)
    old_options.__dict__.update({**options.__dict__, "pixel": old_pixel})
    payload = {"format": FORMAT_VERSION, "rows": [0, 1], "results": [_pixel(0, 0, 1.0)], "options": old_options}
    path.write_bytes(pickle.dumps(payload))


def test_load_checkpoints_reuses_a_checkpoint_that_predates_a_literal_default_field(tmp_path: Path):
    # A field with a plain literal default (``distance_threshold``): a
    # checkpoint pickled before it must still match the current defaults (the
    # multi-day full-image resume must not restart).
    _checkpoint_without_pixel_fields(tmp_path / "column_0000.pkl", "distance_threshold")
    payload = pickle.loads((tmp_path / "column_0000.pkl").read_bytes())
    assert "distance_threshold" not in payload["options"].pixel.__dict__
    window = Window((0, 1), (0, 1))
    messages: list[str] = []
    found = load_checkpoints(tmp_path, window, DriverOptions(), log=messages.append)
    assert sorted(found) == [0] and messages == []
    # ... and the field does take part in the fingerprint once it differs.
    changed = DriverOptions(pixel=PixelOptions(distance_threshold=0.5))
    assert load_checkpoints(tmp_path, window, changed, log=messages.append) == {}
    assert messages == ["column 0 checkpoint options differ from this run's options; recomputing"]


def test_load_checkpoints_recomputes_instead_of_crashing_on_a_factory_default_field(tmp_path: Path):
    # A field with ``default_factory`` has no class attribute to fall back on;
    # comparing raises AttributeError, which must read as "predates".
    _checkpoint_without_pixel_fields(tmp_path / "column_0000.pkl", "quadrature")
    window = Window((0, 1), (0, 1))
    messages: list[str] = []
    assert load_checkpoints(tmp_path, window, DriverOptions(), log=messages.append) == {}
    assert messages == ["column 0 checkpoint options predate a current option field; recomputing"]


def test_load_checkpoints_recomputes_a_checkpoint_that_predates_the_scene_prescan(tmp_path: Path):
    # task-scene-prescan-table replaced the per-pixel 400k prescan by the
    # scene-level table: a checkpoint without ``DriverOptions.prescan`` was
    # rendered under the old discovery policy and must not be reused.
    options = DriverOptions()
    old_options = object.__new__(DriverOptions)
    old_options.__dict__.update({k: v for k, v in options.__dict__.items() if k != "prescan"})
    (tmp_path / "column_0000.pkl").write_bytes(
        pickle.dumps({"format": FORMAT_VERSION, "rows": [0, 1], "results": [_pixel(0, 0, 1.0)], "options": old_options})
    )
    messages: list[str] = []
    assert load_checkpoints(tmp_path, Window((0, 1), (0, 1)), options, log=messages.append) == {}
    assert messages == ["column 0 checkpoint options predate a current option field; recomputing"]


def test_load_checkpoints_recomputes_a_checkpoint_of_another_format_version(tmp_path: Path):
    # task-pixel-pipeline-v2 changed the pickled result types: a v1 checkpoint
    # (or one without a format tag) is recomputed even when its options match.
    for tag in ("lumice-integral.strip/v1", None):
        payload = {"rows": [0, 1], "results": [_pixel(0, 0, 1.0)], "options": DriverOptions()}
        if tag is not None:
            payload["format"] = tag
        (tmp_path / "column_0000.pkl").write_bytes(pickle.dumps(payload))
        messages: list[str] = []
        assert load_checkpoints(tmp_path, Window((0, 1), (0, 1)), DriverOptions(), log=messages.append) == {}
        assert messages == [f"column 0 checkpoint format {tag!r} != {FORMAT_VERSION!r}; recomputing"]


def test_prescan_build_options_fingerprint_ignores_the_cache_path(tmp_path: Path):
    base = DriverOptions(prescan=PrescanBuildOptions(sample_count=1_000, rng_seed=1))
    (tmp_path / "column_0000.pkl").write_bytes(
        pickle.dumps({"format": FORMAT_VERSION, "rows": [0, 1], "results": [_pixel(0, 0, 1.0)], "options": base})
    )
    window = Window((0, 1), (0, 1))
    moved = DriverOptions(prescan=PrescanBuildOptions(sample_count=1_000, rng_seed=1, cache_path=tmp_path / "t.npz"))
    assert sorted(load_checkpoints(tmp_path, window, moved)) == [0]
    denser = DriverOptions(prescan=PrescanBuildOptions(sample_count=2_000, rng_seed=1))
    assert load_checkpoints(tmp_path, window, denser) == {}
    reseeded = DriverOptions(prescan=PrescanBuildOptions(sample_count=1_000, rng_seed=2))
    assert load_checkpoints(tmp_path, window, reseeded) == {}
    with pytest.raises(ValueError):
        PrescanBuildOptions(sample_count=0)
    assert moved.prescan.as_json()["cache_path"] == str(tmp_path / "t.npz")


def test_merge_subpixels_averages_and_propagates_unknown():
    parts = [_pixel(5, 5, 1.0), _pixel(5, 5, 3.0, completeness="unknown"), _pixel(5, 5, 2.0)]
    merged = _merge_subpixels(5, 5, parts)
    assert merged.value == 2.0
    assert merged.completeness == "unknown"
    assert merged.components == parts[1].components
    assert merged.timings["quadrature_s"] == pytest.approx(3.0)


def test_driver_options_validate():
    with pytest.raises(ValueError):
        DriverOptions(pixel_model="average")
    assert not hasattr(DriverOptions(), "cold_check_interval")
    with pytest.raises(ValueError):
        DriverOptions(subpixel_grid=0)
    options = DriverOptions(pixel_model="subpixel", subpixel_rows=(50, 80))
    assert options.uses_subpixel(60) and not options.uses_subpixel(80)
    assert options.pixel_model_block()["subpixel_rows"] == [50, 80]
    assert DriverOptions().pixel_model_block()["subpixel_grid"] is None


# --- solver-backed window tests ---------------------------------------------------


# The survey's prescan (400k samples); see tests/test_strip_pixel.py.
TEST_PRESCAN = PrescanBuildOptions(sample_count=400_000)


@pytest.fixture(scope="module")
def scene():
    return canonical_strip_scene(prescan_sample_count=TEST_PRESCAN.sample_count, prescan_rng_seed=TEST_PRESCAN.rng_seed)


def test_render_column_chain_matches_independent_pixels(scene):
    options = DriverOptions()
    results = render_column(scene, 150, range(150, 153), options)
    assert [r.row for r in results] == [150, 151, 152]
    assert [r.extra_seed_count for r in results] == [0, 1, 1]
    for result in results:
        independent = render_pixel(scene, result.row, 150, options.pixel)
        assert result.value == pytest.approx(independent.value, rel=1e-6)
        assert result.component_count == independent.component_count == 1


def test_subpixel_model_on_a_row_band_averages_the_pipeline(scene):
    options = DriverOptions(pixel_model="subpixel", subpixel_grid=2, subpixel_rows=(151, 152))
    results = render_column(scene, 150, range(150, 153), options)
    point = render_pixel(scene, 151, 150, options.pixel)
    assert results[0].extra_seed_count == 0 and results[1].extra_seed_count == 4  # one warm seed per sub-pixel
    assert results[1].completeness == "complete"
    # The 2x2 mean sits within the range of the point values of the neighbours.
    assert min(results[0].value, results[2].value) < results[1].value < max(results[0].value, results[2].value)
    assert abs(results[1].value - point.value) / point.value < 1e-2
    assert results[1].timings["quadrature_s"] > 2 * results[0].timings["quadrature_s"] * 0.5
    assert results[2].extra_seed_count == 1  # chained from the centre-most sub-pixel


def test_render_window_serial_writes_checkpoints_and_resumes(scene, tmp_path: Path):
    window = Window((150, 152), (150, 152))
    options = DriverOptions()
    results, execution = render_window(window, options, workers=1, checkpoint_dir=tmp_path, scene=scene)
    assert len(results) == 4
    assert execution["columns_rendered_now"] == 2 and execution["columns_resumed"] == 0
    assert execution["prescan"]["source"] == "scene" and execution["prescan"]["valid_count"] == 64427
    assert sorted(load_checkpoints(tmp_path, window, options)) == [150, 151]
    assert load_checkpoints(tmp_path, Window((150, 153), (150, 152)), options) == {}

    resumed, execution = render_window(window, options, workers=1, checkpoint_dir=tmp_path, resume=True, scene=scene)
    assert execution["columns_rendered_now"] == 0 and execution["columns_resumed"] == 2
    assert execution["prescan"]["source"] == "not-needed"
    assert [(r.row, r.column, r.value) for r in resumed] == [(r.row, r.column, r.value) for r in results]


def test_render_window_parallel_resume_with_nothing_pending_never_touches_the_pool(scene, tmp_path: Path):
    """``workers > 1`` with every column already checkpointed must render nothing
    and never attempt to build a ``Pool`` (round 2 code review dispute: a disputed
    claim that this path could read an unbound ``table`` before any ``Pool`` is
    ever created; no process is spawned here, so this needs no ``spawn`` support)."""
    window = Window((150, 152), (150, 152))
    options = DriverOptions()
    render_window(window, options, workers=1, checkpoint_dir=tmp_path, scene=scene)
    assert sorted(load_checkpoints(tmp_path, window, options)) == [150, 151]

    resumed, execution = render_window(window, options, workers=2, checkpoint_dir=tmp_path, resume=True)
    assert execution["columns_rendered_now"] == 0 and execution["columns_resumed"] == 2
    assert execution["prescan"]["source"] == "not-needed"
    assert len(resumed) == 4


def test_render_window_without_a_scene_builds_the_table_once_from_the_options(scene, tmp_path: Path):
    window = Window((150, 151), (150, 151))
    cache = tmp_path / "prescan.npz"
    options = DriverOptions(prescan=PrescanBuildOptions(sample_count=400_000, cache_path=cache))
    messages: list[str] = []
    results, execution = render_window(window, options, workers=1, log=messages.append)
    assert execution["prescan"]["source"] == str(cache)
    assert execution["prescan"]["sample_count"] == 400_000 and execution["prescan"]["valid_count"] == 64427
    assert cache.exists() and any("prescan table built" in m for m in messages)
    reference = render_pixel(scene, 150, 150, options.pixel)
    assert results[0].value == pytest.approx(reference.value, rel=1e-12)
    # A second run loads the cache instead of sampling again.
    messages.clear()
    render_window(window, options, workers=1, log=messages.append)
    assert any(m.startswith("prescan table loaded from") for m in messages)
    assert not any("prescan table built" in m for m in messages)


def test_render_window_resume_recomputes_columns_whose_options_changed(scene, tmp_path: Path):
    window = Window((150, 152), (150, 152))
    options = DriverOptions()
    render_window(window, options, workers=1, checkpoint_dir=tmp_path, scene=scene)
    assert sorted(load_checkpoints(tmp_path, window, options)) == [150, 151]

    changed = DriverOptions(pixel=PixelOptions(distance_threshold=0.05))
    assert load_checkpoints(tmp_path, window, changed) == {}
    resumed, execution = render_window(window, changed, workers=1, checkpoint_dir=tmp_path, resume=True, scene=scene)
    assert execution["columns_rendered_now"] == 2 and execution["columns_resumed"] == 0
    assert sorted(load_checkpoints(tmp_path, window, changed)) == [150, 151]


def test_render_window_resume_recomputes_legacy_checkpoints_without_an_options_fingerprint(
    scene, tmp_path: Path
):
    window = Window((150, 152), (150, 152))
    options = DriverOptions()
    results, _ = render_window(window, options, workers=1, checkpoint_dir=tmp_path, scene=scene)
    legacy_path = tmp_path / "column_0150.pkl"
    payload = pickle.loads(legacy_path.read_bytes())
    del payload["options"]
    legacy_path.write_bytes(pickle.dumps(payload))

    assert sorted(load_checkpoints(tmp_path, window, options)) == [151]
    resumed, execution = render_window(window, options, workers=1, checkpoint_dir=tmp_path, resume=True, scene=scene)
    assert execution["columns_rendered_now"] == 1 and execution["columns_resumed"] == 1
    assert [(r.row, r.column, r.value) for r in resumed] == [(r.row, r.column, r.value) for r in results]


@pytest.mark.slow
def test_render_window_parallel_matches_serial(scene, tmp_path: Path):
    if multiprocessing.get_start_method(allow_none=True) not in (None, "spawn", "fork", "forkserver"):
        pytest.skip("unsupported start method")
    window = Window((150, 152), (150, 152))
    options = DriverOptions(prescan=TEST_PRESCAN)
    serial, _ = render_window(window, options, workers=1, scene=scene)
    parallel, execution = render_window(window, options, workers=2)
    assert execution["workers"] == 2
    assert execution["prescan"]["source"] == "in-memory" and execution["prescan"]["valid_count"] == 64427
    assert [(r.row, r.column) for r in parallel] == [(r.row, r.column) for r in serial]
    for a, b in zip(serial, parallel):
        assert a.value == pytest.approx(b.value, rel=1e-12)
        assert a.status_bits == b.status_bits
