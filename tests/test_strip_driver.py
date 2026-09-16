"""Strip orchestration and on-disk format: synthetic results, round trip, small windows."""

from __future__ import annotations

import json
import multiprocessing
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from lumice_integral.strip_driver import (
    DriverOptions,
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
    STATUS_COLD_DISCOVERY,
    STATUS_HAS_COMPONENT,
    STATUS_RENDERED,
    STATUS_UNKNOWN_COMPLETENESS,
    ComponentRecord,
    PixelOptions,
    PixelResult,
    canonical_strip_scene,
    render_pixel,
)


def _component(value: float, arclength: float = 4.7) -> ComponentRecord:
    return ComponentRecord(
        seed=np.eye(3),
        discovery_arclength=arclength,
        production_status="closed",
        production_reason="closed_loop",
        production_arclength=arclength,
        production_pose_count=120,
        quadrature_status="available",
        value=value,
        error_estimate=1e-7,
        refinements=10,
        node_count=250,
        maximum_depth_reached=6,
        depth_exhausted_edge_count=0,
    )


def _pixel(row: int, column: int, value: float, *, completeness: str = "complete", seed_source: str = "hot", events=None) -> PixelResult:
    components = (_component(value),) if value else ()
    return PixelResult(
        row=row,
        column=column,
        value=value,
        error_estimate=1e-7 * len(components),
        components=components,
        completeness=completeness,
        discovery_completeness="not-run" if seed_source == "hot" else completeness,
        seed_source=seed_source,
        incomplete_count=0,
        pool_count=0,
        raw_cluster_count=0,
        admissible_count=len(components),
        events={name: 0 for name in EVENT_NAMES} | (events or {}),
        timings={"hot_start_s": 0.1, "production_s": 0.2, "quadrature_s": 1.0, "total_s": 1.3},
    )


# --- pure format tests (no solver) --------------------------------------------


def test_window_validates_and_counts():
    window = Window((10, 12), (3, 5))
    assert window.pixel_count == 4
    assert list(window.row_range) == [10, 11]
    with pytest.raises(ValueError):
        Window((5, 5), (0, 1))


def test_assemble_arrays_keeps_unknown_pixels_distinguishable_from_complete_zeros():
    results = [
        _pixel(0, 0, 1.5, seed_source="cold"),
        _pixel(0, 1, 0.0, completeness="unknown", seed_source="cold", events={"incomplete_candidate": 3}),
        _pixel(1, 0, 0.0, seed_source="cold"),  # dark, complete
    ]
    arrays = assemble_arrays(results, height=2, width=2)
    assert arrays.values.tolist() == [[1.5, 0.0], [0.0, 0.0]]
    assert arrays.status[0, 0] == STATUS_RENDERED | STATUS_HAS_COMPONENT | STATUS_COLD_DISCOVERY
    assert arrays.status[0, 1] == STATUS_RENDERED | STATUS_UNKNOWN_COMPLETENESS | STATUS_COLD_DISCOVERY
    assert arrays.status[1, 0] == STATUS_RENDERED | STATUS_COLD_DISCOVERY
    assert arrays.status[1, 1] == 0  # not rendered
    assert arrays.rendered.tolist() == [[True, True], [True, False]]
    assert arrays.component_count.tolist() == [[1, 0], [0, 0]]
    # The three zero-valued pixels are three different things in the status layer.
    assert len({int(arrays.status[0, 1]), int(arrays.status[1, 0]), int(arrays.status[1, 1])}) == 3


def test_write_and_read_strip_round_trip_with_verified_hashes(tmp_path: Path):
    results = [_pixel(r, c, float(r * 10 + c) + 0.25, seed_source="cold" if r == 0 else "hot") for r in range(2) for c in range(3)]
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
    assert provenance["format"] == FORMAT_VERSION
    assert provenance["arrays"]["shape"] == [4, 3]
    assert provenance["arrays"]["status_bits"] == STATUS_BITS
    assert provenance["window"] == {"rows": [0, 2], "columns": [0, 3], "pixel_count": 6}
    assert provenance["options"]["quadrature"]["relative_tolerance"] == 1e-6
    assert provenance["options"]["quadrature"]["convergence_order_levels"] == 0
    assert provenance["options"]["discovery"]["retry_step_budget"] == 4000
    assert provenance["scene"]["refractive_index"] == {"value": 1.31, "provenance": "canonical-new"}
    assert provenance["scene"]["sun"]["provenance"] == "historical-inferred"
    assert provenance["summary"]["rendered_pixels"] == 6
    assert provenance["summary"]["cold_discovery_pixels"] == 3
    assert provenance["pixel_model"]["model"] == "point"
    assert "Lumice" in provenance["generator"]["lumice_dependency"]
    assert provenance["environment"]["jax"]
    json.dumps(provenance)  # serialisable without solver objects

    with files["pixels"].open() as handle:
        header = handle.readline().strip().split(",")
        rows = handle.read().strip().splitlines()
    assert tuple(header) == PIXEL_CSV_COLUMNS
    assert len(rows) == 6

    # Tampering is detected on read.
    files["float64"].write_bytes(b"\x00" * files["float64"].stat().st_size)
    with pytest.raises(ValueError, match="sha256"):
        read_strip(tmp_path)


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
    with pytest.raises(ValueError):
        DriverOptions(subpixel_grid=0)
    options = DriverOptions(pixel_model="subpixel", subpixel_rows=(50, 80))
    assert options.uses_subpixel(60) and not options.uses_subpixel(80)
    assert options.pixel_model_block()["subpixel_rows"] == [50, 80]
    assert DriverOptions().pixel_model_block()["subpixel_grid"] is None


# --- solver-backed window tests ---------------------------------------------------


@pytest.fixture(scope="module")
def scene():
    return canonical_strip_scene()


def test_render_column_chain_matches_independent_pixels(scene):
    options = DriverOptions(cold_check_interval=2)
    results = render_column(scene, 150, range(150, 153), options)
    assert [r.row for r in results] == [150, 151, 152]
    assert [r.seed_source for r in results] == ["cold", "hot", "hot"]
    assert results[2].events["cold_check"] == 1 and results[2].events["cold_check_mismatch"] == 0
    assert results[2].status_bits & STATUS_COLD_DISCOVERY
    for result in results:
        independent = render_pixel(scene, result.row, 150, options.pixel)
        assert result.value == pytest.approx(independent.value, rel=1e-6)
        assert result.component_count == independent.component_count == 1


def test_subpixel_model_on_a_row_band_averages_the_pipeline(scene):
    options = DriverOptions(pixel_model="subpixel", subpixel_grid=2, subpixel_rows=(151, 152), cold_check_interval=0)
    results = render_column(scene, 150, range(150, 153), options)
    point = render_pixel(scene, 151, 150, options.pixel)
    assert results[0].seed_source == "cold" and results[1].seed_source == "hot"
    assert results[1].completeness == "complete"
    # The 2x2 mean sits within the range of the point values of the neighbours.
    assert min(results[0].value, results[2].value) < results[1].value < max(results[0].value, results[2].value)
    assert abs(results[1].value - point.value) / point.value < 1e-2
    assert results[1].timings["quadrature_s"] > 2 * results[0].timings["quadrature_s"] * 0.5
    assert results[2].seed_source == "hot"  # chained from the centre-most sub-pixel


def test_render_window_serial_writes_checkpoints_and_resumes(scene, tmp_path: Path):
    window = Window((150, 152), (150, 152))
    options = DriverOptions(cold_check_interval=0)
    results, execution = render_window(window, options, workers=1, checkpoint_dir=tmp_path, scene=scene)
    assert len(results) == 4
    assert execution["columns_rendered_now"] == 2 and execution["columns_resumed"] == 0
    assert sorted(load_checkpoints(tmp_path, window)) == [150, 151]
    assert load_checkpoints(tmp_path, Window((150, 153), (150, 152))) == {}

    resumed, execution = render_window(window, options, workers=1, checkpoint_dir=tmp_path, resume=True, scene=scene)
    assert execution["columns_rendered_now"] == 0 and execution["columns_resumed"] == 2
    assert [(r.row, r.column, r.value) for r in resumed] == [(r.row, r.column, r.value) for r in results]


@pytest.mark.slow
def test_render_window_parallel_matches_serial(scene, tmp_path: Path):
    if multiprocessing.get_start_method(allow_none=True) not in (None, "spawn", "fork", "forkserver"):
        pytest.skip("unsupported start method")
    window = Window((150, 152), (150, 152))
    options = DriverOptions(cold_check_interval=0)
    serial, _ = render_window(window, options, workers=1, scene=scene)
    parallel, execution = render_window(window, options, workers=2)
    assert execution["workers"] == 2
    assert [(r.row, r.column) for r in parallel] == [(r.row, r.column) for r in serial]
    for a, b in zip(serial, parallel):
        assert a.value == pytest.approx(b.value, rel=1e-12)
        assert a.status_bits == b.status_bits
