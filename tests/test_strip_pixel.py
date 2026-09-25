"""Per-pixel strip pipeline: canonical value, pixel regressions, warm seeds, arcs."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import lumice_integral.strip_pixel as strip_pixel_module
from lumice_integral.canonical_scene import (
    CANONICAL_PIXEL_COLUMN,
    CANONICAL_PIXEL_ROW,
    canonical_crystal,
    canonical_pose_density,
    canonical_sun_direction,
    canonical_target_direction,
)
from lumice_integral.geometry import HexPrism
from lumice_integral.continuation import ContinuationOptions, FiberStatus, TerminationReason, trace_fiber
from lumice_integral.discovery import DISCOVERY_EVENT_NAMES, ComponentDiscoveryResult, DiscoveredComponent, retarget_problem
from lumice_integral.pose_density import build_pose_density
from lumice_integral.s2_store import DEFAULT_SEED_STORE_N, StoreSeeds, build_event_store
from lumice_integral.resample import stitch_open_arc
from lumice_integral.strip_pixel import (
    EVENT_NAMES,
    STAGE_NAMES,
    STATUS_HAS_ARC,
    STATUS_HAS_COMPONENT,
    STATUS_QUADRATURE_UNAVAILABLE,
    STATUS_RENDERED,
    STATUS_UNKNOWN_COMPLETENESS,
    PixelOptions,
    PixelResult,
    canonical_strip_scene,
    pixel_target,
    render_pixel,
    subpixel_targets,
)

# docs/ch06-reference-fixture.md section 4.1: the pipeline's resampled
# quadrature (rtol 1e-4) from the *discovered* seed on the canonical ``h/a = 2``
# crystal, 257 nodes (61 poses).  History: ``h/a = 1`` gave 2.364412980 (task
# defect2-crystal-height-convention, 2026-09-20, re-pinned 2.364412980 ->
# 6.581419934 when the canonical crystal became ``h/a = 2``; the fiber, its
# 61 poses and 257 nodes are unchanged, only the ``entry_measure`` weight
# moved), itself half of the 4.728847630 recorded before
# task-continuation-gates-and-fixtures (the loop, 2.379 < pi, had been
# integrated over two traversals).  The retired adaptive integrator's
# rtol=1e-8 reference (2.364423815) exists only for ``h/a = 1``; the alignment
# against it lives in tests/test_resample_quadrature.py on that crystal.
# 6.581419934 -> 6.581365570 (task phase1-seeds-from-store, 2026-09-25): the
# seed now comes from the S^2 store instead of the Haar prescan; the same loop
# (61 poses, 257 nodes) resampled from another seed moves by 8e-6 relative,
# inside the quadrature's own error estimate (1.2e-4).
# 6.581365570 -> 6.581510519 (task phase1-quadrature-start-and-speed, 2026-09-25):
# the arclength speed gains its -nu' . delta term and the error estimate is taken
# panel by panel, which refines this loop to 513 nodes (was 257).  The value driven
# to rtol = 1e-9 at the same epsilon is 6.581453685: the old pin was 8.8e-5 below it
# (estimate 1.2e-4), the new one is 5.7e-5 above it (estimate 3.3e-4).
CANONICAL_PIXEL_RESAMPLED_VALUE = 6.581510519
# tests/test_discovery.py baselines.
CANONICAL_ARCLENGTH = 2.379121
ROW_225_ARCLENGTH = 3.121867
ROW_226_ARCLENGTH = 3.130201
# Every count baseline below is pinned to the production seed store
# (``s2_store.DEFAULT_SEED_STORE_N`` points, the default 0.2 deg band).
KEPT_EVENTS = 160216


@pytest.fixture(scope="module")
def scene():
    return canonical_strip_scene()


@pytest.fixture(scope="module")
def options():
    return PixelOptions()


@pytest.fixture(scope="module")
def canonical(scene, options) -> PixelResult:
    return render_pixel(scene, CANONICAL_PIXEL_ROW, CANONICAL_PIXEL_COLUMN, options)


def test_pixel_options_defaults_are_the_image_policy(options):
    assert options.quadrature.relative_tolerance == 1e-4
    assert options.quadrature.initial_node_count == 129
    assert options.quadrature.maximum_node_count == 1025
    assert options.continuation.maximum_accepted_steps == 4000
    assert options.distance_threshold == options.continuation.closure_distance == 0.08
    assert set(options.discovery_kwargs()) == {
        "continuation", "band_half_width_deg", "cluster_radius_rad", "distance_threshold"
    }
    # One budget: the retired discovery/retry budgets and stall window are gone.
    for retired in ("discovery_step_budget", "retry_step_budget", "stall_floor_window", "jump_relative_threshold"):
        assert not hasattr(options, retired)
    # The seed store's size is a scene policy (task phase1-seeds-from-store), not a pixel one.
    assert not hasattr(options, "seed_store_n") and not hasattr(options, "prescan_samples")
    assert set(EVENT_NAMES) >= set(DISCOVERY_EVENT_NAMES)
    assert STAGE_NAMES == ("discovery_s", "trace_s", "quadrature_s", "total_s")


def test_canonical_scene_carries_matching_seeds(scene):
    seeds = scene.seeds
    assert seeds.n == DEFAULT_SEED_STORE_N == 1_000_000 and len(seeds.store.events) == KEPT_EVENTS
    assert seeds.faces == scene.faces == (3, 5) and seeds.g is None
    assert np.array_equal(seeds.incident_direction, scene.incident_direction)
    assert np.array_equal(seeds.sun_direction, canonical_sun_direction())
    assert seeds.refractive_index == scene.refractive_index
    # Supplied seeds are used as is; seeds of another index or crystal are rejected.
    reused = canonical_strip_scene(seeds=seeds)
    assert reused.seeds is seeds
    other_index = build_event_store(canonical_crystal(), 1.33, [(3, 5)], 1_000, run_checks=False)
    with pytest.raises(ValueError, match="refractive index"):
        canonical_strip_scene(seeds=StoreSeeds(other_index, (3, 5), canonical_sun_direction()))
    with pytest.raises(ValueError, match="crystal"):
        canonical_strip_scene(seeds=seeds, crystal=HexPrism(1.0, 1.0))


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
    # Odd grid: the middle sub-pixel is the pixel centre (arccos of a dot product one
    # ulp below 1 is ~1e-6 deg on the GPU backend); every other one lies inside the
    # pixel (edge 6/251 deg ~ 0.024 deg).
    assert offsets_deg[4] < 1e-5
    assert all(0.0 < offset < 0.02 for i, offset in enumerate(offsets_deg) if i != 4)
    assert len({round(offset, 6) for offset in offsets_deg}) == 5  # 4-fold symmetric tiling
    with pytest.raises(ValueError):
        subpixel_targets(scene.render, 150, 150, 0)


def test_canonical_pixel_single_component_reproduces_the_fixture_value(canonical):
    assert canonical.component_count == 1 and canonical.arc_count == 0
    assert canonical.incomplete_count == 0
    assert canonical.completeness == "complete"
    assert canonical.pool_count == 5024 and canonical.extra_seed_count == 0
    assert canonical.raw_cluster_count == 6 and canonical.admissible_count == 6
    component = canonical.components[0]
    assert component.kind == "closed"
    assert component.status == "closed" and component.reason == "closed_loop" and component.start_reason == ""
    assert component.quadrature_status == "available"
    assert component.arclength == pytest.approx(CANONICAL_ARCLENGTH, rel=1e-3)
    assert component.pose_count == 61
    assert np.isnan(component.start_truncation_estimate) and np.isnan(component.end_truncation_estimate)
    assert canonical.value == component.value
    assert canonical.value == pytest.approx(CANONICAL_PIXEL_RESAMPLED_VALUE, abs=5e-9)
    assert component.node_count == 257 and component.refinement_rounds == 1
    assert not component.node_count_exhausted and component.non_finite_node_count == 0
    # |I_257 - I_129| / I at rtol 1e-4: a conservative estimate (order ~2 grid),
    # not the 1e-9 of the retired adaptive integrator.
    assert canonical.error_estimate < 1e-4 * canonical.value
    assert set(canonical.events) == set(EVENT_NAMES)
    assert canonical.events == {**{name: 0 for name in EVENT_NAMES}, "dedup_merged": 5}
    assert canonical.status_bits == STATUS_RENDERED | STATUS_HAS_COMPONENT
    assert set(canonical.timings) == set(STAGE_NAMES)
    assert 0.0 < canonical.timings["trace_s"] < canonical.timings["discovery_s"] < canonical.timings["total_s"]
    assert canonical.timings["quadrature_s"] > 0
    assert len(canonical.warm_seeds) == 1 and canonical.warm_seeds[0].shape == (3, 3)


def test_warm_seeds_from_the_row_above_give_the_cold_value(scene, options, canonical):
    warm = render_pixel(scene, 151, 150, options, warm_seeds=canonical.warm_seeds)
    cold = render_pixel(scene, 151, 150, options)
    assert warm.extra_seed_count == 1 and cold.extra_seed_count == 0
    assert warm.pool_count == cold.pool_count
    assert warm.completeness == cold.completeness == "complete"
    assert warm.component_count == cold.component_count == 1
    # The same loop from another seed: the resampled quadrature's grid starts at the seed, and its value
    # moves by up to ~5e-5 relative along a loop (8 seeds on rows 150/151/700, task
    # phase1-seeds-from-store), i.e. within the quadrature's relative tolerance.
    assert warm.value == pytest.approx(cold.value, rel=options.quadrature.relative_tolerance)
    assert warm.status_bits == cold.status_bits == STATUS_RENDERED | STATUS_HAS_COMPONENT
    # The warm seed's cluster is traced first and every store cluster on the
    # same loop is folded without a trace.
    assert warm.events["dedup_merged"] == warm.admissible_count - 1


def test_rows_225_and_226_chain_without_a_jump_gate(scene, options):
    upper = render_pixel(scene, 225, 150, options)
    assert upper.components[0].arclength == pytest.approx(ROW_225_ARCLENGTH, rel=1e-3)
    lower = render_pixel(scene, 226, 150, options, warm_seeds=upper.warm_seeds)
    assert lower.components[0].arclength == pytest.approx(ROW_226_ARCLENGTH, rel=1e-3)
    assert lower.completeness == "complete" and lower.component_count == 1
    assert lower.value == pytest.approx(render_pixel(scene, 226, 150, options).value, rel=options.quadrature.relative_tolerance)


@pytest.mark.parametrize("row, arclength", [(700, 5.408495), (780, 5.635867)])
def test_boundary_hugging_pixel_is_complete_with_one_component(scene, options, row, arclength):
    # Until task-continuation-gates-and-fixtures every candidate here (12/13)
    # was floor-locked after its 250 discovery steps and the pixel stayed
    # ``unknown`` with value 0 (the strip's rows 655-800); the rate-based event
    # slowdown closes the first one and the rest are folded by distance.
    result = render_pixel(scene, row, 150, options)
    assert result.component_count == 1 and result.arc_count == 0
    assert result.incomplete_count == 0
    assert result.completeness == "complete"
    assert result.components[0].kind == "closed"
    assert result.components[0].arclength == pytest.approx(arclength, rel=1e-3)
    assert result.events["incomplete_candidate"] == 0
    assert result.events["dedup_merged"] == result.admissible_count - 1
    assert result.value > 0.0 and np.isfinite(result.value)
    assert len(result.warm_seeds) == 1
    assert not result.status_bits & STATUS_UNKNOWN_COMPLETENESS
    assert result.status_bits & STATUS_HAS_COMPONENT


def test_caustic_neighbourhood_pixel_60_126_is_one_short_closed_loop(scene, options):
    result = render_pixel(scene, 60, 126, options)
    assert result.component_count == 1 and result.completeness == "complete"
    assert result.components[0].kind == "closed"
    assert result.components[0].arclength == pytest.approx(0.466117, rel=1e-4)  # 0.466397 from the prescan seed
    # h/a = 1: 19.0379 -> h/a = 2: 39.3657 (2026-09-20); the loop itself is crystal-independent.
    assert result.value == pytest.approx(39.3657, rel=1e-3)


def test_dark_pixel_is_complete_with_zero_value(scene, options):
    result = render_pixel(scene, 40, 150, options)
    assert result.component_count == 0 and result.incomplete_count == 0
    assert result.admissible_count == 0 and result.pool_count == 0
    assert result.completeness == "complete"
    assert result.value == 0.0
    assert result.status_bits == STATUS_RENDERED
    assert result.warm_seeds == ()


def test_bad_warm_seed_neither_poisons_nor_adds_a_component(scene, options, canonical):
    # The identity pose is nowhere near the canonical fiber: it is either
    # inadmissible or folded; the store pool still finds the loop.
    result = render_pixel(scene, 150, 150, options, warm_seeds=(np.eye(3),))
    assert result.extra_seed_count == 1
    assert result.component_count == 1
    assert result.completeness == "complete"
    assert result.value == pytest.approx(canonical.value, rel=1e-9)


def test_starved_production_budget_is_reported_not_hidden(scene, canonical):
    starved = replace(
        PixelOptions(),
        continuation=replace(PixelOptions().continuation, maximum_accepted_steps=20),
    )
    result = render_pixel(scene, 150, 150, starved)
    # No candidate converges, so nothing is integrated; the evidence is the
    # incomplete count, not a silently truncated partial value.
    assert result.component_count == 0
    assert result.incomplete_count >= 1
    assert result.events["incomplete_candidate"] == result.incomplete_count
    assert result.events["incomplete_not_converged"] == result.incomplete_count
    assert result.completeness == "unknown"
    assert result.value == 0.0
    assert result.status_bits == STATUS_RENDERED | STATUS_UNKNOWN_COMPLETENESS


def test_arc_component_is_integrated_and_flagged(monkeypatch, scene, options):
    """No real open arc exists in the current ch06 picture (Step 0 scan of
    2106 pixels, every 10th row and column: 1954 lit, all closed), so the arc
    path of ``render_pixel`` is exercised with a synthetic discovery result:
    the canonical loop cut into a two-sided arc by an arclength window."""
    target = pixel_target(scene.render, 150, 150)
    from lumice_integral.discovery import retarget_problem

    problem = retarget_problem(scene.discovery_template, target, canonical_seed_of(scene, options))
    forward = trace_fiber(problem, ContinuationOptions(maximum_arclength=0.9))
    backward = trace_fiber(problem, ContinuationOptions(maximum_arclength=0.6, initial_tangent_sign=-1))
    assert forward.reason == backward.reason == TerminationReason.ARCLENGTH_BUDGET
    arc = stitch_open_arc(forward, backward)
    # `reason` is deliberately TIR_BOUNDARY, not the real trace's ARCLENGTH_BUDGET
    # (asserted above): `_integrate_component` must read `component.reason`, and this
    # mismatch is what would expose it silently re-deriving the reason from `arc` instead.
    component = DiscoveredComponent(
        seed=np.asarray(problem.seed), kind="arc", result=arc, arclength=arc.arclength,
        status=FiberStatus.EVENT_TERMINATED, reason=TerminationReason.TIR_BOUNDARY,
    )

    def fake_discover_components(target, seeds, *, template, extra_seeds=(), **kwargs):
        return ComponentDiscoveryResult(
            components=(component,), incomplete=(), completeness="complete",
            pool_count=1, extra_seed_count=len(extra_seeds), raw_cluster_count=1, admissible_count=1,
            events={**{name: 0 for name in DISCOVERY_EVENT_NAMES}, "arc_stitched": 1}, trace_seconds=0.01,
        )

    monkeypatch.setattr(strip_pixel_module, "discover_components", fake_discover_components)
    result = render_pixel(scene, 150, 150, options)
    assert result.component_count == 1 and result.arc_count == 1
    assert result.completeness == "complete"
    assert result.status_bits == STATUS_RENDERED | STATUS_HAS_COMPONENT | STATUS_HAS_ARC
    assert result.events["arc_stitched"] == 1
    record = result.components[0]
    assert record.kind == "arc" and record.status == "event_terminated"
    assert record.reason == "tir_boundary" and record.start_reason == "arclength_budget"
    assert record.pose_count == len(arc.poses)
    assert record.arclength == pytest.approx(1.5, abs=0.15)
    assert record.quadrature_status == "available"
    # A 1.5 / 2.379 piece of the canonical loop: a partial value, reported as such.
    assert 0.0 < record.value < CANONICAL_PIXEL_RESAMPLED_VALUE
    assert result.value == record.value
    # Budget-cut ends carry no margin rate, so both truncation estimates are nan
    # (they exist as fields either way and are never added to the value).
    assert np.isnan(record.start_truncation_estimate) and np.isnan(record.end_truncation_estimate)
    assert result.warm_seeds == (record.seed,)


def canonical_seed_of(scene, options) -> np.ndarray:
    return render_pixel(scene, 150, 150, options).components[0].seed


def test_quadrature_unavailable_component_makes_the_pixel_unknown(monkeypatch, scene, options):
    """A component whose curve has no edge (a single accepted pose) cannot be
    integrated: it contributes 0, is counted, and flips completeness."""
    from lumice_integral.discovery import retarget_problem

    problem = retarget_problem(scene.discovery_template, pixel_target(scene.render, 150, 150), canonical_seed_of(scene, options))
    single = trace_fiber(problem, ContinuationOptions(maximum_accepted_steps=1, maximum_evaluations=1))
    assert len(single.poses) == 1
    component = DiscoveredComponent(
        seed=np.asarray(problem.seed), kind="arc", result=stitch_open_arc(single, single),
        arclength=0.0, status=FiberStatus.EVENT_TERMINATED, reason=single.reason,
    )

    def fake_discover_components(target, seeds, *, template, extra_seeds=(), **kwargs):
        return ComponentDiscoveryResult(
            components=(component,), incomplete=(), completeness="complete",
            pool_count=1, extra_seed_count=0, raw_cluster_count=1, admissible_count=1,
            events={name: 0 for name in DISCOVERY_EVENT_NAMES}, trace_seconds=0.0,
        )

    monkeypatch.setattr(strip_pixel_module, "discover_components", fake_discover_components)
    result = render_pixel(scene, 150, 150, options)
    assert result.components[0].quadrature_status == "unavailable_no_edges"
    assert result.events["quadrature_unavailable"] == 1
    assert result.completeness == "unknown" and result.value == 0.0
    assert result.status_bits == STATUS_RENDERED | STATUS_UNKNOWN_COMPLETENESS | STATUS_QUADRATURE_UNAVAILABLE
    assert result.warm_seeds == ()  # not integrated -> not a warm seed



# ---------------------------------------------------------------------------
# pose density families (task-pose-density-families): the same fibers, another rho_pose

# Diagnostic pixels shared with scripts/probe_defect2_factors.py and the ch11
# family comparison (docs/ch11-pose-density-families.md).
FAMILY_PROBE_COLUMN = 126
FAMILY_PROBE_ROWS = (150, 300, 450, 600)
FAMILY_DENSITIES = {
    "random": build_pose_density("random"),
    "plate": build_pose_density("plate", zenith_std_deg=0.5),
    "column": build_pose_density("column", zenith_std_deg=0.5),
    "parry": build_pose_density("parry", zenith_std_deg=1.0, roll_std_deg=1.0),
    "lowitz": build_pose_density("lowitz", zenith_std_deg=40.0, roll_std_deg=1.0),
}


def test_canonical_strip_scene_defaults_to_the_canonical_column_density(scene):
    assert scene.pose_density == canonical_pose_density()
    assert scene.production_template.weight_evaluators["rho_pose"].evaluate is scene.pose_density


@pytest.mark.parametrize("row", FAMILY_PROBE_ROWS)
def test_every_family_renders_the_diagnostic_pixels_on_the_same_fibers(scene, options, row):
    """Discovery/trace never read rho_pose: the five families share the pose arrays
    (same component count, pose count, arclength) and differ only in the weight; every
    family's rho_pose is finite at every accepted pose (parry/lowitz roll extraction
    stays off the gimbal poles on these fibers) and every integral is finite."""
    target = pixel_target(scene.render, row, FAMILY_PROBE_COLUMN)
    results = {}
    for family, density in FAMILY_DENSITIES.items():
        family_scene = canonical_strip_scene(seeds=scene.seeds, pose_density=density)
        assert family_scene.pose_density is density
        result = render_pixel(family_scene, row, FAMILY_PROBE_COLUMN, options)
        results[family] = result
        for record in result.components:
            fiber = trace_fiber(retarget_problem(family_scene.production_template, target, record.seed))
            rho = np.asarray(fiber.weight_observables["rho_pose"].values, dtype=np.float64)
            assert rho.shape == (record.pose_count,) and np.all(np.isfinite(rho)) and np.all(rho >= 0.0), family
            np.testing.assert_allclose(rho, density.evaluate_batch(np.asarray(fiber.poses)), rtol=1e-12)
    reference = results["column"]
    assert reference.component_count >= 1 and reference.completeness == "complete"
    for family, result in results.items():
        assert result.component_count == reference.component_count, family
        assert result.completeness == "complete", family
        assert np.isfinite(result.value) and result.value >= 0.0, family
        for ours, theirs in zip(result.components, reference.components):
            assert ours.kind == theirs.kind and ours.pose_count == theirs.pose_count, family
            assert ours.arclength == theirs.arclength, family
            assert ours.quadrature_status == "available" and ours.non_finite_node_count == 0, family
            assert np.isfinite(ours.value) and np.isfinite(ours.error_estimate), family
    # On this labelled path (3 -> 5) and column the fibers have c-axis zenith 88-92 deg
    # and roll 90-155 deg (face 3 on the side): only column and random carry mass.
    # plate (zenith about 0 deg), parry and lowitz (roll locked at 0 deg, face 3 on top)
    # underflow to exactly zero here; their 3 -> 5 light lands outside the strip
    # (parhelia / Parry / Lowitz arcs), see docs/ch11-pose-density-families.md.
    assert results["column"].value > 0.0 and results["random"].value > 0.0
    assert results["plate"].value == 0.0 and results["parry"].value == 0.0 and results["lowitz"].value == 0.0
