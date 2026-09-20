"""Aggregation logic of ``benchmarks/benchmark_column_steady_state.py`` (no rendering)."""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "benchmarks" / "benchmark_column_steady_state.py"


@pytest.fixture(scope="module")
def bench():
    spec = importlib.util.spec_from_file_location("benchmark_column_steady_state", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_compile_name_reads_the_jax_compile_message(bench):
    message = (
        "Compiling jit(rotation_distance) with global shapes and types "
        "(ShapedArray(float64[3,3]), ShapedArray(float64[2252,3,3])). Argument mapping: (x, y)."
    )
    assert bench.compile_name(message) == "jit(rotation_distance)"
    assert bench.compile_name("Finished XLA compilation of jit(stage) in 0.004 sec") is None
    assert bench.compile_name("Compiling module jit_f with FDO profile of length 3") is None


def test_compile_counter_counts_only_compile_records_by_name(bench):
    counter = bench.CompileCounter()
    logger = logging.getLogger("jax._src.interpreters.pxla")
    logging.getLogger(bench.JAX_LOGGER_NAME).addHandler(counter)
    try:
        logger.warning("Compiling %s with global shapes and types %s. Argument mapping: %s.", "jit(stage)", "()", "()")
        logger.warning("Compiling %s with global shapes and types %s. Argument mapping: %s.", "jit(stage)", "()", "()")
        logger.warning("Compiling %s with global shapes and types %s. Argument mapping: %s.", "jit(add)", "()", "()")
        logger.warning("Finished XLA compilation of jit(add) in 0.01 sec")
    finally:
        logging.getLogger(bench.JAX_LOGGER_NAME).removeHandler(counter)
    assert counter.total == 3
    assert counter.names == {"jit(stage)": 2, "jit(add)": 1}


def test_summarize_drops_the_warmup_from_the_steady_state_but_not_the_totals(bench):
    seconds = [1.0, 0.9, 0.3, 0.2, 0.4]
    compiles = [50, 20, 0, 3, 0]
    report = bench.summarize(seconds, compiles, warmup=2)
    assert report["pixel_count"] == 5
    assert report["warmup_pixels"] == 2
    assert report["total_seconds"] == pytest.approx(2.8)
    assert report["total_compiles"] == 73
    assert report["steady_pixel_count"] == 3
    assert report["steady_median_seconds_per_pixel"] == pytest.approx(0.3)
    assert report["steady_mean_seconds_per_pixel"] == pytest.approx(0.3)
    assert report["steady_max_seconds_per_pixel"] == pytest.approx(0.4)
    assert report["steady_compiles"] == 3
    assert report["steady_pixels_with_compiles"] == 1


def test_summarize_reports_no_steady_state_for_a_window_inside_the_warmup(bench):
    report = bench.summarize([1.0, 0.5], [10, 2], warmup=10)
    assert report["warmup_pixels"] == 2
    assert report["total_compiles"] == 12
    assert report["steady_pixel_count"] == 0
    assert report["steady_median_seconds_per_pixel"] is None
    assert report["steady_compiles"] is None


def test_summarize_rejects_mismatched_inputs(bench):
    with pytest.raises(ValueError):
        bench.summarize([1.0], [1, 2], warmup=0)
    with pytest.raises(ValueError):
        bench.summarize([1.0], [1], warmup=-1)
