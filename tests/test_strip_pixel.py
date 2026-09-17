"""Per-pixel strip pipeline: canonical value, six-pixel regressions, hot-start chain."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace

import numpy as np
import pytest

import lumice_integral.strip_pixel as strip_pixel_module
from lumice_integral.canonical_scene import (
    CANONICAL_PIXEL_COLUMN,
    CANONICAL_PIXEL_ROW,
    canonical_target_direction,
)
from lumice_integral.continuation import FiberStatus, TerminationReason
from lumice_integral.discovery import ComponentDiscoveryResult, DiscoveredComponent
from lumice_integral.prescan import DEFAULT_SAMPLE_COUNT, build_prescan_table
from lumice_integral.strip_pixel import (
    EVENT_NAMES,
    STATUS_ARCLENGTH_JUMP,
    STATUS_COLD_DISCOVERY,
    STATUS_HAS_COMPONENT,
    STATUS_RENDERED,
    STATUS_UNKNOWN_COMPLETENESS,
    HotSeed,
    PixelOptions,
    PixelResult,
    canonical_strip_scene,
    pixel_target,
    render_pixel,
    subpixel_targets,
)

# docs/ch06-reference-fixture.md section 4.1 (rtol 1e-8, Haar-converted, partial).
# Half of the 4.728847630 recorded before task-continuation-gates-and-fixtures:
# the canonical loop (2.379 < pi) was integrated over two traversals.
CANONICAL_PIXEL_VALUE = 2.364423815
# tests/test_discovery.py baselines.
CANONICAL_ARCLENGTH = 2.379121
ROW_225_ARCLENGTH = 3.121867
ROW_226_ARCLENGTH = 3.130201
# The survey's prescan (400k samples, seed 20260916): every count baseline
# below (``pool_count == 210`` etc.) is pinned to this table, not to the
# production default of ``prescan.DEFAULT_SAMPLE_COUNT``.
TEST_PRESCAN_SAMPLES = 400_000


@pytest.fixture(scope="module")
def scene():
    return canonical_strip_scene(prescan_sample_count=TEST_PRESCAN_SAMPLES)


@pytest.fixture(scope="module")
def options():
    return PixelOptions()


@pytest.fixture(scope="module")
def canonical(scene, options) -> PixelResult:
    return render_pixel(scene, CANONICAL_PIXEL_ROW, CANONICAL_PIXEL_COLUMN, options)


def test_pixel_options_defaults_are_the_image_policy(options):
    assert options.quadrature.relative_tolerance == 1e-6
    assert options.quadrature.convergence_order_levels == 0
    assert options.continuation.maximum_accepted_steps == 4000
    assert options.discovery_step_budget == 250
    # task-discovery-stall-early-exit Step 0: inside the [0, 141] separation
    # band between legit slow-closers (never at the floor) and stalls.
    assert options.stall_floor_window == 100
    assert set(options.discovery_kwargs()) == {
        "discovery_step_budget", "angle_tolerance_deg", "cluster_radius_rad", "arclength_rtol"
    }
    # The prescan sampling is a scene policy (task-scene-prescan-table), not a pixel one.
    assert not hasattr(options, "rng_seed") and not hasattr(options, "prescan_samples")


def test_canonical_scene_carries_a_matching_prescan_table(scene):
    table = scene.prescan_table
    assert table.sample_count == TEST_PRESCAN_SAMPLES and table.valid_count == 64427
    assert np.array_equal(table.incident_direction, scene.incident_direction)
    assert table.refractive_index == scene.refractive_index
    # A supplied table is used as is; one for another scene is rejected.
    reused = canonical_strip_scene(prescan_table=table)
    assert reused.prescan_table is table
    foreign = build_prescan_table(scene.incident_direction, 1.33, sample_count=1_000)
    with pytest.raises(ValueError, match="refractive index"):
        canonical_strip_scene(prescan_table=foreign)
    assert DEFAULT_SAMPLE_COUNT >= 4_000_000


def test_pixel_target_matches_the_canonical_scene(scene):
    assert np.allclose(
        pixel_target(scene.render, CANONICAL_PIXEL_ROW, CANONICAL_PIXEL_COLUMN),
        canonical_target_direction(),
    )
    assert scene.width == 251 and scene.height == 801


def test_subpixel_targets_tile_the_pixel_and_average_to_its_centre(scene):
    targets = subpixel_targets(scene.render, 150, 150, 3)
    assert len(targets) == 9
    centre = pixel_target(scene.render, 150, 150)
    mean = np.mean(targets, axis=0)
    assert np.dot(mean / np.linalg.norm(mean), centre) > 1.0 - 1e-9
    offsets_deg = [np.degrees(np.arccos(np.clip(t @ centre, -1, 1))) for t in targets]
    # Odd grid: the middle sub-pixel is the pixel centre; every other one lies
    # inside the pixel (edge 6/251 deg ~ 0.024 deg).
    assert offsets_deg[4] == 0.0
    assert all(0.0 < offset < 0.02 for i, offset in enumerate(offsets_deg) if i != 4)
    assert len({round(offset, 6) for offset in offsets_deg}) == 5  # 4-fold symmetric tiling
    with pytest.raises(ValueError):
        subpixel_targets(scene.render, 150, 150, 0)


def test_canonical_pixel_single_component_reproduces_the_fixture_value(canonical):
    assert canonical.component_count == 1
    assert canonical.incomplete_count == 0
    assert canonical.completeness == "complete"
    assert canonical.discovery_completeness == "complete"
    assert canonical.seed_source == "cold"
    component = canonical.components[0]
    assert component.production_status == "closed"
    assert component.quadrature_status == "available"
    assert component.discovery_arclength == pytest.approx(CANONICAL_ARCLENGTH, rel=1e-3)
    assert component.production_arclength == pytest.approx(component.discovery_arclength, rel=1e-6)
    assert canonical.value == component.value
    assert abs(canonical.value - CANONICAL_PIXEL_VALUE) <= canonical.error_estimate + 1e-9
    assert canonical.error_estimate < 1e-5
    assert set(canonical.events) == set(EVENT_NAMES)
    assert not any(canonical.events.values())
    assert canonical.status_bits == STATUS_RENDERED | STATUS_HAS_COMPONENT | STATUS_COLD_DISCOVERY
    assert canonical.timings["quadrature_s"] > 0 and canonical.timings["total_s"] > 0


def test_hot_start_to_the_next_row_matches_cold_discovery(scene, options, canonical):
    hot = render_pixel(scene, 151, 150, options, previous=canonical.hot_seeds)
    cold = render_pixel(scene, 151, 150, options)
    assert hot.seed_source == "hot" and cold.seed_source == "cold"
    assert hot.discovery_completeness == "not-run"
    assert hot.completeness == cold.completeness == "complete"
    assert hot.component_count == cold.component_count == 1
    assert hot.value == pytest.approx(cold.value, rel=1e-6)
    assert hot.status_bits == STATUS_RENDERED | STATUS_HAS_COMPONENT
    assert hot.timings["hot_start_s"] > 0 and hot.timings["cold_discovery_s"] == 0.0


def test_cold_check_agreeing_with_the_hot_start_keeps_the_hot_result(scene, options, canonical):
    checked = render_pixel(scene, 151, 150, options, previous=canonical.hot_seeds, cold_check=True)
    assert checked.seed_source == "hot"
    assert checked.events["cold_check"] == 1
    assert checked.events["cold_check_mismatch"] == 0
    assert checked.timings["cold_discovery_s"] > 0


def test_row_226_hot_starts_from_row_225_and_a_doubled_arclength_falls_back_to_cold(
    scene, options
):
    # Rows 225/226 were the one real arclength jump (6.24 -> 3.13) until
    # task-continuation-gates-and-fixtures showed the 6.24 to be row 225's
    # 3.12 loop traversed twice under the absolute closure gate.  The hot
    # start now continues; the jump fallback path is exercised by handing row
    # 226 the retired doubled arclength as its neighbour's.
    upper = render_pixel(scene, 225, 150, options)
    assert upper.components[0].discovery_arclength == pytest.approx(ROW_225_ARCLENGTH, rel=1e-3)
    continued = render_pixel(scene, 226, 150, options, previous=upper.hot_seeds)
    assert continued.seed_source == "hot"
    assert continued.events["arclength_jump"] == 0
    assert continued.components[0].discovery_arclength == pytest.approx(ROW_226_ARCLENGTH, rel=1e-3)
    assert not continued.status_bits & STATUS_ARCLENGTH_JUMP
    doubled = tuple(HotSeed(seed.seed, 2.0 * seed.arclength) for seed in upper.hot_seeds)
    lower = render_pixel(scene, 226, 150, options, previous=doubled)
    assert lower.components[0].discovery_arclength == pytest.approx(ROW_226_ARCLENGTH, rel=1e-3)
    assert lower.seed_source == "cold-fallback"
    assert lower.events["arclength_jump"] == 1
    assert lower.completeness == "complete"
    assert lower.status_bits & STATUS_ARCLENGTH_JUMP
    assert lower.status_bits & STATUS_COLD_DISCOVERY
    assert not lower.status_bits & STATUS_UNKNOWN_COMPLETENESS
    assert lower.value == pytest.approx(render_pixel(scene, 226, 150, options).value, rel=1e-6)


@pytest.mark.parametrize("row, arclength", [(700, 5.408495), (780, 5.635867)])
def test_boundary_hugging_pixel_is_complete_with_one_component(scene, options, row, arclength):
    # Until task-continuation-gates-and-fixtures every candidate here (12/13)
    # was floor-locked after its 250 discovery steps and the pixel stayed
    # ``unknown`` with value 0 (the strip's rows 655-800); the rate-based event
    # slowdown closes them all on one loop.
    result = render_pixel(scene, row, 150, options)
    assert result.component_count == 1
    assert result.incomplete_count == 0
    assert result.completeness == "complete"
    assert result.discovery_completeness == "complete"
    assert result.components[0].discovery_arclength == pytest.approx(arclength, rel=1e-3)
    assert result.events["incomplete_candidate"] == 0
    assert result.events["incomplete_stall_skip"] == 0
    assert result.events["incomplete_retry"] == 0
    assert result.value > 0.0 and np.isfinite(result.value)
    assert len(result.hot_seeds) == 1
    assert not result.status_bits & STATUS_UNKNOWN_COMPLETENESS
    assert result.status_bits & STATUS_HAS_COMPONENT


def test_early_exit_window_setting_does_not_change_a_pixel_without_stalled_candidates(scene):
    # A window above the discovery budget can never be met (the pre-early-exit
    # behaviour).  With no ``step_budget`` candidate left on this pixel both
    # settings must produce the same classification and value.
    disabled = replace(PixelOptions(), stall_floor_window=PixelOptions().discovery_step_budget + 1)
    result = render_pixel(scene, 700, 150, disabled)
    reference = render_pixel(scene, 700, 150, PixelOptions())
    assert result.events["incomplete_stall_skip"] == 0
    assert result.events["incomplete_retry"] == 0
    assert result.events["incomplete_candidate"] == 0
    assert result.component_count == 1 and result.incomplete_count == 0
    assert result.completeness == "complete"
    assert result.value == pytest.approx(reference.value, rel=1e-9)
    assert result.status_bits == reference.status_bits


def test_event_terminated_candidates_are_still_retraced_not_stall_skipped(scene, options):
    # Row 49, column 0 (home-wsl-preview-step9 pixels.csv: retry 4, recovered
    # 2, candidate 2, one component of arclength 3.228596 closing after 1290
    # production steps).  That component is a 0.19 loop that the absolute
    # closure gate traversed 17 times (task-continuation-gates-and-fixtures);
    # it now closes inside the discovery budget, so only the two
    # event-terminated candidates remain.  They are outside the criterion's
    # reason gate and go through the (free) retrace as before.
    events = Counter()
    discovered = strip_pixel_module._cold_discovery(scene, pixel_target(scene.render, 49, 0), options, events)
    assert events["incomplete_stall_skip"] == 0
    assert events["incomplete_retry"] == 2
    assert events["incomplete_recovered"] == 0
    assert discovered.component_count == 1 and discovered.incomplete_count == 2
    assert discovered.components[0].arclength == pytest.approx(0.189896, rel=1e-4)
    assert len(discovered.components[0].result.poses) < 250
    assert {c.reason for c in discovered.incomplete} == {
        TerminationReason.PATH_INFEASIBLE,
        TerminationReason.TIR_BOUNDARY,
    }


def test_dark_pixel_is_complete_with_zero_value(scene, options):
    result = render_pixel(scene, 40, 150, options)
    assert result.component_count == 0 and result.incomplete_count == 0
    assert result.admissible_count == 0 and result.pool_count == 210
    assert result.completeness == "complete"
    assert result.value == 0.0
    assert result.status_bits == STATUS_RENDERED | STATUS_COLD_DISCOVERY


def test_bad_hot_seed_falls_back_to_cold_discovery(scene, options):
    # The identity pose is nowhere near the canonical fiber: the hot start is
    # either inadmissible, incomplete, or lands with a jumped arclength; in
    # every case the pixel must fall back to the cold prescan, not be poisoned.
    result = render_pixel(
        scene, 150, 150, options, previous=(HotSeed(np.eye(3), 100.0),)
    )
    assert result.seed_source == "cold-fallback"
    assert (
        result.events["hot_start_inadmissible"]
        + result.events["hot_start_incomplete"]
        + result.events["arclength_jump"]
    ) == 1
    assert result.component_count == 1
    assert result.completeness == "complete"


def test_hot_start_chain_cannot_discover_a_component_absent_from_the_previous_seeds(
    monkeypatch, scene, options
):
    """Known limitation (code-review round 1 Major 2, see the module docstring):
    ``_hot_start_all`` only revisits the neighbour's own seeds one by one, so
    its result always has exactly ``len(previous)`` components. A component
    that first becomes admissible between two rows (a caustic/topology branch)
    has no matching seed, so the hot chain has no opportunity to find it and
    fires no event; ``render_pixel`` then reports ``completeness="complete"``
    even though a real component was missed. Only the periodic ``cold_check``
    (``DriverOptions.cold_check_interval``, default 8 rows) recovers it, and
    only for the rows from there on -- this test pins that bounded, not
    eliminated, blind spot rather than leaving it un-exercised.
    """
    existing = DiscoveredComponent(
        seed=np.eye(3), result=None, arclength=1.0,
        status=FiberStatus.CLOSED, reason=TerminationReason.CLOSED_LOOP,
    )
    new_component = DiscoveredComponent(
        seed=np.eye(3) * 2.0, result=None, arclength=2.0,
        status=FiberStatus.CLOSED, reason=TerminationReason.CLOSED_LOOP,
    )

    def fake_hot_start_component(seed, target, incident_direction, refractive_index, crystal, *, discovery_step_budget, template):
        return existing

    def fake_discover_components(target, crystal, table, *, template, **kwargs):
        return ComponentDiscoveryResult(
            components=(existing, new_component),
            incomplete=(),
            completeness="complete",
            pool_count=2,
            raw_cluster_count=2,
            admissible_count=2,
        )

    monkeypatch.setattr(strip_pixel_module, "hot_start_component", fake_hot_start_component)
    monkeypatch.setattr(strip_pixel_module, "discover_components", fake_discover_components)

    target = np.array([0.0, 0.0, 1.0])
    events: Counter = Counter()
    hot = strip_pixel_module._hot_start_all(scene, target, (HotSeed(existing.seed, 1.0),), options, events)
    assert hot is not None and hot.component_count == 1
    assert not events["hot_start_inadmissible"]
    assert not events["hot_start_incomplete"]
    assert not events["arclength_jump"]

    cold = strip_pixel_module._cold_discovery(scene, target, options, events)
    assert cold.component_count == 2  # the periodic cold_check is what actually recovers it


def test_tiny_production_budget_is_reported_not_hidden(scene, canonical):
    starved = replace(
        PixelOptions(),
        continuation=replace(PixelOptions().continuation, maximum_accepted_steps=20),
    )
    result = render_pixel(scene, 150, 150, starved)
    assert result.component_count == 1
    component = result.components[0]
    assert component.production_status == "budget_exhausted"
    assert result.events["production_not_closed"] == 1
    assert result.completeness == "unknown"
    # The truncated trace is still integrated as a partial value, visibly.
    assert component.quadrature_status == "available"
    assert 0.0 < result.value < canonical.value
