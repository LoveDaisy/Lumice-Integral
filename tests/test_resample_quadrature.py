"""Resampled fixed-grid quadrature: analytic circle, doubling logic, open arcs, alignment.

The external alignment values below are the only correctness evidence of
:func:`integrate_fiber_resampled`; its internal ``|I_N - I_(N+1)/2|`` estimate
is self-consistency, not proof (task-resample-and-integrate plan D2 / a02).
"""

from __future__ import annotations

import dataclasses
import time

import jax.numpy as jnp
import numpy as np
import pytest

from lumice_integral.analytic import BODY_AXIS, direction_map, tangent_basis
from lumice_integral.canonical_scene import canonical_pixel_problem
from lumice_integral.continuation import (
    ContinuationOptions,
    DomainEvaluation,
    EventCandidate,
    FiberProblem,
    FiberStatus,
    TargetChart,
    TerminationReason,
    trace_fiber,
)
from lumice_integral.discovery import retarget_problem
from lumice_integral.geometry import HexPrism
from lumice_integral.quadrature import (
    HAAR_TO_DVOL_G_FACTOR,
    INTEGRAND_FACTOR_NAMES,
    RESAMPLED_QUADRATURE_METHOD,
    ResampledQuadratureResult,
    ResampleOptions,
    integrate_fiber_resampled,
)
from lumice_integral.resample import OpenArc, TraceLike, fiber_spline, resample_spline, stitch_open_arc, uniform_parameters
from lumice_integral.so3 import exp
from lumice_integral.strip_pixel import PixelOptions, canonical_strip_scene, pixel_target
from lumice_integral.weights import WeightEvaluator

EPSILON = 1e-6

# Reference values of the adaptive chord-parametrised integrator
# (``integrate_fiber``, ``relative_tolerance=1e-8``, error estimates 2.5e-9 to
# 4.1e-9 relative, 693-941 nodes) recorded by task-resample-and-integrate
# Step 5 (progress.md, 2026-09-17) before that integrator was removed; the
# canonical value was docs/ch06-reference-fixture.md section 4.1 until
# 2026-09-20.  Frozen here so the alignment no longer needs the old
# integrator at run time.  They were recorded on the ``h/a = 1`` crystal of
# that date (task defect2-crystal-height-convention moved the canonical
# scene to ``h/a = 2``, which changes the ``entry_measure`` weight and hence
# these integrals); the alignment cases therefore bind that crystal
# explicitly (``REFERENCE_CRYSTAL``) -- the integrator check does not depend
# on which crystal it runs on, and the retired integrator cannot re-record.
REFERENCE_CRYSTAL = HexPrism.from_ratio(1.0)
ADAPTIVE_REFERENCE = {
    "canonical (150,150)": 2.36442381498,
    "row 100 col 126": 5.71594639898,
    "row 300 col 126": 1.82223453481,
    "row 500 col 126": 0.456662771746,
}
ALIGNMENT_RTOL = 1e-4


def _constant(value: float) -> WeightEvaluator:
    return WeightEvaluator(lambda _: value, "dimensionless", "test constant")


def _circle_angle(rotation: np.ndarray) -> float:
    return float(np.arctan2(rotation[1, 0], rotation[0, 0]))


def _circle_problem(density, domain=None) -> FiberProblem:
    """Analytic circle ``F(R) = R e3`` with ``rho_pose = density`` and unit other factors."""
    return FiberProblem(
        path="analytic-body-axis",
        incident_direction=jnp.array([1.0, 0.0, 0.0], dtype=jnp.float64),
        target_chart=TargetChart(BODY_AXIS, tangent_basis(BODY_AXIS)),
        direction_evaluator=direction_map,
        seed=jnp.eye(3, dtype=jnp.float64),
        domain_and_event_evaluator=domain,
        weight_evaluators={
            "rho_pose": WeightEvaluator(density, "dimensionless", "relative to Haar"),
            **{name: _constant(1.0) for name in INTEGRAND_FACTOR_NAMES},
        },
    )


@pytest.fixture(scope="module")
def canonical():
    problem = canonical_pixel_problem(crystal=REFERENCE_CRYSTAL)
    return problem, trace_fiber(problem)


# The seeds the adaptive references were recorded from: discovery on ``REFERENCE_CRYSTAL`` with the retired
# 400k-sample prescan (exponential coordinates, recomputed with that code on 2026-09-25).  Frozen because the
# quadrature's value depends on where the trace starts on the loop (the grid phase against the ``entry_measure``
# kinks, up to ~1e-4 relative at the default tolerance): from the seed-store seeds (task phase1-seeds-from-store)
# row 100 landed 2.45e-6 from its reference, 1.33x the global ``|I_N - I_(N+1)/2|`` estimate of that date.  The
# panel-wise estimate (task phase1-quadrature-start-and-speed) bounds every start point checked; the seeds stay
# frozen so the pinned node counts and values do not move with the seed store.
STRIP_PIXEL_SEEDS = {
    100: (-1.506657821759383, 0.3966428317572424, 0.0890374620046359),
    300: (-1.2178814168477645, 0.9704917826539213, 0.014459273801443785),
    500: (-1.6380970762050457, -0.32828079328859805, 1.1261906950875498),
}


@pytest.fixture(scope="module")
def strip_pixels():
    """Production traces of rows 100/300/500 (col 126) of the ch06 strip, on ``REFERENCE_CRYSTAL``."""
    scene = canonical_strip_scene(crystal=REFERENCE_CRYSTAL, seed_store_n=1_000)
    options = PixelOptions()
    traces = {}
    for row, coordinates in STRIP_PIXEL_SEEDS.items():
        target = pixel_target(scene.render, row, 126)
        seed = np.asarray(exp(jnp.asarray(coordinates)))
        problem = retarget_problem(scene.production_template, target, seed)
        traces[f"row {row} col 126"] = (problem, trace_fiber(problem, options.continuation))
    return traces


# --- options ------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"epsilon": 0.0},
        {"relative_tolerance": 0.0},
        {"initial_node_count": 128},
        {"initial_node_count": 3},
        {"initial_node_count": 129, "maximum_node_count": 65},
        {"retraction_iterations": 0},
    ],
)
def test_resample_options_reject_invalid_policy(kwargs):
    with pytest.raises(ValueError):
        ResampleOptions(**kwargs)


def test_resample_options_defaults_are_the_step_5_calibration():
    options = ResampleOptions()
    assert options.relative_tolerance == 1e-4
    assert options.initial_node_count == 129 and options.maximum_node_count == 1025
    assert options.retraction_iterations == 2 and options.epsilon == 1e-6


# --- analytic circle -----------------------------------------------------------


def test_constant_weight_recovers_the_haar_identity_on_the_circle():
    problem = _circle_problem(lambda _: 1.0)
    result = trace_fiber(problem)

    quadrature = integrate_fiber_resampled(problem, result)

    assert quadrature.status == "available" and quadrature.fiber_status == "closed"
    assert quadrature.method == RESAMPLED_QUADRATURE_METHOD
    expected_raw = 2.0 * np.pi / (1.0 + EPSILON)  # J_perp == 1 on the circle
    assert abs(quadrature.raw_value - expected_raw) < 1e-8 * expected_raw  # far below 1e-3
    assert abs(quadrature.raw_value - expected_raw) <= quadrature.raw_error_estimate
    assert quadrature.value == quadrature.raw_value * HAAR_TO_DVOL_G_FACTOR
    assert quadrature.haar_to_dvol_g_factor == HAAR_TO_DVOL_G_FACTOR
    assert quadrature.node_count == 129 and quadrature.refinement_rounds == 0
    assert not quadrature.node_count_exhausted
    assert quadrature.node_count_history == ((65, pytest.approx(quadrature.raw_value, rel=1e-8)), (129, quadrature.raw_value))
    # The spline of a geodesic circle is on the fiber up to round-off.
    assert quadrature.residual_before_max < 1e-12 and quadrature.residual_after_max < 1e-14
    assert quadrature.non_finite_node_count == 0
    assert np.isnan(quadrature.endpoint_truncation_estimate)
    assert quadrature.endpoint_truncation_note.startswith("closed loop")
    assert set(quadrature.factor_seconds) == {"rho_pose", *INTEGRAND_FACTOR_NAMES}
    assert quadrature.component_completeness == "unknown" and quadrature.coverage.startswith("partial")


def test_trigonometric_weight_matches_the_analytic_integral_on_every_grid():
    problem = _circle_problem(lambda rotation: 1.0 + 0.5 * float(rotation[0, 0]))
    result = trace_fiber(problem)
    expected_raw = 2.0 * np.pi / (1.0 + EPSILON)  # int_0^{2pi} (1 + cos(theta)/2) dtheta

    for node_count in (17, 33, 65, 129):
        quadrature = integrate_fiber_resampled(
            problem, result,
            ResampleOptions(initial_node_count=node_count, maximum_node_count=node_count, relative_tolerance=1e-14),
        )
        assert abs(quadrature.raw_value - expected_raw) < 1e-8 * expected_raw
        assert quadrature.node_count == node_count and quadrature.node_count_exhausted


def test_reversed_seed_orientation_keeps_the_integral():
    problem = _circle_problem(lambda rotation: 1.0 + 0.5 * float(rotation[0, 0]))
    forward = integrate_fiber_resampled(problem, trace_fiber(problem, ContinuationOptions()))
    reverse = integrate_fiber_resampled(
        problem, trace_fiber(problem, ContinuationOptions(initial_tangent_sign=-1))
    )
    assert abs(forward.value - reverse.value) <= forward.error_estimate + reverse.error_estimate + 1e-14


# --- doubling and its honest exhaustion ----------------------------------------


# ``1 / (1.2 + cos theta)``: periodic and analytic, so the uniform grid converges
# geometrically at rate ~0.54^N -- slow enough from 5 nodes that reaching 1e-7
# takes three doublings (5 -> 9 -> 17 -> 33), fast enough to stay small.
_PEAKED_RAW_INTEGRAL = 2.0 * np.pi / np.sqrt(1.2**2 - 1.0) / (1.0 + EPSILON)


def _peaked(rotation):
    return 1.0 / (1.2 + float(rotation[0, 0]))


def test_node_count_doubles_until_the_estimate_meets_the_tolerance():
    problem = _circle_problem(_peaked)
    result = trace_fiber(problem)

    quadrature = integrate_fiber_resampled(
        problem, result, ResampleOptions(initial_node_count=5, relative_tolerance=1e-7)
    )

    counts = [count for count, _ in quadrature.node_count_history]
    assert counts[:2] == [3, 5] and all(b == 2 * a - 1 for a, b in zip(counts[1:], counts[2:]))
    assert quadrature.refinement_rounds >= 2 and quadrature.node_count == counts[-1]
    assert not quadrature.node_count_exhausted
    assert quadrature.raw_error_estimate <= 1e-7 * abs(quadrature.raw_value)
    values = [value for _, value in quadrature.node_count_history]
    assert abs(values[-1] - _PEAKED_RAW_INTEGRAL) < 1e-7 * _PEAKED_RAW_INTEGRAL
    assert abs(values[0] - _PEAKED_RAW_INTEGRAL) > 1e-3 * _PEAKED_RAW_INTEGRAL


def test_hitting_maximum_node_count_is_reported_not_passed_off_as_converged():
    problem = _circle_problem(_peaked)
    result = trace_fiber(problem)

    quadrature = integrate_fiber_resampled(
        problem, result,
        ResampleOptions(initial_node_count=5, maximum_node_count=17, relative_tolerance=1e-15),
    )

    assert quadrature.node_count == 17 and quadrature.refinement_rounds == 2
    assert quadrature.node_count_exhausted
    assert quadrature.raw_error_estimate > 1e-15 * abs(quadrature.raw_value)
    assert abs(quadrature.raw_value - _PEAKED_RAW_INTEGRAL) > 1e-9 * _PEAKED_RAW_INTEGRAL
    assert quadrature.status == "available"


# --- open arcs ----------------------------------------------------------------


def test_event_terminated_arc_integrates_to_its_extent_and_estimates_the_truncation():
    theta_max = 2.0

    def cap(rotation):
        margin = theta_max - _circle_angle(np.asarray(rotation))
        if margin < 0.0:
            return DomainEvaluation(
                False, {"cap": margin}, EventCandidate(TerminationReason.VISIBILITY_BOUNDARY, margin)
            )
        return DomainEvaluation(True, {"cap": margin})

    problem = _circle_problem(lambda _: 1.0, cap)
    result = trace_fiber(problem)
    assert result.status == FiberStatus.EVENT_TERMINATED
    theta_end = _circle_angle(np.asarray(result.poses[-1]))
    assert 0.0 < theta_max - theta_end < 0.2

    quadrature = integrate_fiber_resampled(problem, result)

    assert quadrature.fiber_status == "event_terminated"
    assert quadrature.raw_value == pytest.approx(theta_end / (1.0 + EPSILON), rel=1e-9)
    # Constant integrand, linear margin: the linear-rate distance is exact.
    expected_truncation = (theta_max - theta_end) / (1.0 + EPSILON) * HAAR_TO_DVOL_G_FACTOR
    assert quadrature.endpoint_truncation_estimate == pytest.approx(expected_truncation, rel=1e-9)
    assert "visibility_boundary" in quadrature.endpoint_truncation_note
    assert "seed end is not a boundary" in quadrature.endpoint_truncation_note


def _two_sided_cap(theta_min: float, theta_max: float):
    def cap(rotation):
        theta = _circle_angle(np.asarray(rotation))
        margins = {"cap_hi": theta_max - theta, "cap_lo": theta - theta_min}
        if margins["cap_hi"] < 0.0:
            return DomainEvaluation(False, margins, EventCandidate(TerminationReason.VISIBILITY_BOUNDARY, margins["cap_hi"]))
        if margins["cap_lo"] < 0.0:
            return DomainEvaluation(False, margins, EventCandidate(TerminationReason.TIR_BOUNDARY, margins["cap_lo"]))
        return DomainEvaluation(True, margins)

    return cap


def test_stitched_two_sided_arc_integrates_like_its_halves_and_estimates_both_truncations():
    """task-pixel-pipeline-v2 Step 0 probe, frozen: a forward and a backward
    trace of one seed on the circle cut at both ends, stitched into an
    :class:`OpenArc`, resample to a C^1 predictor with ``ds/dt > 0`` and
    integrate to the analytic value (= the sum of the one-sided integrals),
    with the terminal truncation equal to the one-sided estimate and the
    start truncation the mirrored linear-rate estimate at the backward end."""
    theta_min, theta_max = -1.3, 2.0
    problem = _circle_problem(lambda r: 1.0 + 0.5 * np.cos(_circle_angle(np.asarray(r))), _two_sided_cap(theta_min, theta_max))
    forward = trace_fiber(problem)
    backward = trace_fiber(problem, ContinuationOptions(initial_tangent_sign=-1))
    assert forward.reason == TerminationReason.VISIBILITY_BOUNDARY
    assert backward.reason == TerminationReason.TIR_BOUNDARY

    arc = stitch_open_arc(forward, backward)
    assert isinstance(arc, OpenArc) and isinstance(arc, TraceLike) and isinstance(forward, TraceLike)
    thetas = np.array([_circle_angle(pose) for pose in arc.poses])
    assert np.all(np.diff(thetas) > 0.0) and arc.seed_index == len(backward.poses) - 1
    spline = fiber_spline(arc)
    assert not spline.closed
    predictors = resample_spline(spline, uniform_parameters(spline, 2001))
    speeds = np.linalg.norm(predictors.body_velocities, axis=1)
    assert np.all(speeds > 0.0) and np.allclose(speeds, 1.0, atol=1e-9)
    predictor_thetas = np.array([_circle_angle(r) for r in predictors.rotations])
    assert np.abs(np.diff(predictor_thetas, 2)).max() < 1e-9  # C^1 through the seed knot

    quadrature = integrate_fiber_resampled(problem, arc)
    lo, hi = thetas[0], thetas[-1]
    analytic = ((hi - lo) + 0.5 * (np.sin(hi) - np.sin(lo))) / (1.0 + EPSILON)
    halves = integrate_fiber_resampled(problem, forward), integrate_fiber_resampled(problem, backward)
    assert quadrature.fiber_status == "event_terminated"
    assert quadrature.raw_value == pytest.approx(analytic, rel=1e-8)
    assert quadrature.raw_value == pytest.approx(halves[0].raw_value + halves[1].raw_value, rel=1e-8)
    assert quadrature.endpoint_truncation_estimate == pytest.approx(halves[0].endpoint_truncation_estimate, rel=1e-9)
    assert "visibility_boundary" in quadrature.endpoint_truncation_note
    assert "seed end" not in quadrature.endpoint_truncation_note
    # Start end: constant-rate margin, so the linear-rate distance is exact.
    start_integrand = (1.0 + 0.5 * np.cos(lo)) / (1.0 + EPSILON)
    expected_start = start_integrand * (lo - theta_min) * HAAR_TO_DVOL_G_FACTOR
    assert quadrature.start_endpoint_truncation_estimate == pytest.approx(expected_start, rel=1e-6)
    assert "tir_boundary" in quadrature.start_endpoint_truncation_note
    assert "start" in quadrature.start_endpoint_truncation_note
    # One-sided inputs keep the start field inert (regression guard for the
    # frozen ADAPTIVE_REFERENCE path).
    assert np.isnan(halves[0].start_endpoint_truncation_estimate)
    assert "seed" in halves[0].start_endpoint_truncation_note


def test_stitch_open_arc_rejects_traces_of_different_seeds():
    problem = _circle_problem(lambda _: 1.0, _two_sided_cap(-1.0, 1.0))
    forward = trace_fiber(problem)
    other = trace_fiber(
        FiberProblem(**{**problem.__dict__, "seed": jnp.asarray(forward.poses[2])}),
        ContinuationOptions(initial_tangent_sign=-1),
    )
    with pytest.raises(ValueError, match="same seed"):
        stitch_open_arc(forward, other)


def test_budget_truncated_arc_is_integrated_but_gets_no_truncation_estimate(canonical):
    problem, closed = canonical
    result = trace_fiber(problem, ContinuationOptions(maximum_accepted_steps=25))
    assert result.status == FiberStatus.BUDGET_EXHAUSTED

    partial = integrate_fiber_resampled(problem, result)
    full = integrate_fiber_resampled(problem, closed)

    assert partial.status == "available" and partial.fiber_status == "budget_exhausted"
    assert 0.0 < partial.value < full.value
    assert np.isnan(partial.endpoint_truncation_estimate)
    assert "step_budget" in partial.endpoint_truncation_note


def test_missing_factor_is_unavailable_and_short_fibers_have_no_edges(canonical):
    problem, result = canonical
    geometry_only = canonical_pixel_problem(with_weights=False)
    unavailable = integrate_fiber_resampled(geometry_only, result)
    assert unavailable.status == "unavailable_missing_rho_pose"
    assert np.isnan(unavailable.value) and unavailable.node_count == 0
    assert isinstance(unavailable, ResampledQuadratureResult)

    short = trace_fiber(problem, ContinuationOptions(maximum_accepted_steps=1))
    assert len(short.poses) == 2
    assert integrate_fiber_resampled(problem, short).status == "available"
    single = trace_fiber(problem, ContinuationOptions(maximum_accepted_steps=1, maximum_evaluations=1))
    assert len(single.poses) < 2
    assert integrate_fiber_resampled(problem, single).status == "unavailable_no_edges"


# --- alignment with the adaptive reference (Step 5 / Step 7 item 2) ------------


def _alignment_cases(canonical, strip_pixels):
    yield "canonical (150,150)", canonical
    yield from strip_pixels.items()


def test_default_options_align_with_the_adaptive_reference_within_1e_4(canonical, strip_pixels):
    for name, (problem, result) in _alignment_cases(canonical, strip_pixels):
        assert result.status == FiberStatus.CLOSED, name
        quadrature = integrate_fiber_resampled(problem, result)
        reference = ADAPTIVE_REFERENCE[name]

        assert quadrature.status == "available", name
        assert not quadrature.node_count_exhausted, name
        assert quadrature.non_finite_node_count == 0, name
        assert quadrature.residual_after_max < 1e-13, name
        deviation = abs(quadrature.value - reference) / reference
        assert deviation <= ALIGNMENT_RTOL, (name, deviation, quadrature.node_count)
        # The internal estimate must not be optimistic against the external reference.
        assert deviation <= quadrature.error_estimate / reference + 1e-12, (name, deviation)
        # 129-1025 with the panel-wise estimate (row 300 needs 1025; 129-513 with the global one it replaced).
        assert 129 <= quadrature.node_count <= 1025, (name, quadrature.node_count)


def test_deviation_from_the_reference_shrinks_with_the_grid(canonical):
    problem, result = canonical
    reference = ADAPTIVE_REFERENCE["canonical (150,150)"]
    deviations = []
    for node_count in (65, 129, 257, 513):
        quadrature = integrate_fiber_resampled(
            problem, result,
            ResampleOptions(initial_node_count=node_count, maximum_node_count=node_count, relative_tolerance=1e-14),
        )
        deviations.append(abs(quadrature.value - reference) / reference)
    # Order ~2 at the entry_measure slope jumps (Step 5 evidence: 7e-5, 3e-5, 1e-5, 3e-6).
    assert deviations[0] > deviations[1] > deviations[2] > deviations[3]
    assert deviations[1] < 1e-4 and deviations[3] < 1e-5


def test_converged_canonical_value_matches_the_adaptive_reference_within_1e_8(canonical):
    """Driven to ``rtol = 1e-9`` the resampled grid meets the retired adaptive integrator.

    The two share the trace but not the parametrisation or the speed (the adaptive
    one retracted every node on its own and differentiated no phase condition), and
    the reference's own estimate is ~2.5e-9 relative.  Before the ``-nu' . delta``
    term of ``_parametric_speed`` (task phase1-quadrature-start-and-speed) the
    converged value sat 5.0e-6 low: an ``O(delta)`` bias no grid refinement removes.
    After it: 1.6e-11.
    """
    problem, result = canonical
    reference = ADAPTIVE_REFERENCE["canonical (150,150)"]
    quadrature = integrate_fiber_resampled(
        problem, result, ResampleOptions(relative_tolerance=1e-9, maximum_node_count=262145)
    )
    assert not quadrature.node_count_exhausted
    assert abs(quadrature.value - reference) / reference < 1e-8


def _with_origin_at_knot(result, knot: int):
    """The same closed trace with its parameter origin moved to accepted pose ``knot`` (no re-trace)."""
    distinct = len(result.poses) - 1
    poses = np.roll(np.asarray(result.poses[:distinct]), -knot, axis=0)
    tangents = np.roll(np.asarray(result.tangents[:distinct]), -knot, axis=0)
    return dataclasses.replace(
        result,
        poses=np.concatenate((poses, poses[:1])),
        tangents=np.concatenate((tangents, tangents[:1])),
        arclength_increments=np.roll(np.asarray(result.arclength_increments), -knot),
    )


def test_error_estimate_is_conservative_at_every_grid_origin(canonical):
    """Moving the grid origin along the same loop moves the grid over the ``entry_measure`` kinks.

    Only the grid phase changes (same spline segments), yet the value moves by up to ~6e-5
    relative: the start-point dependence of task phase1-quadrature-start-and-speed.  The
    global ``|I_N - I_(N+1)/2|`` it replaced was optimistic against the reference on some
    origins (the 16-start probe: up to 5x); the panel-wise estimate must bound every one.
    """
    problem, result = canonical
    reference = ADAPTIVE_REFERENCE["canonical (150,150)"]
    for knot in range(0, len(result.poses) - 1, 4):
        quadrature = integrate_fiber_resampled(problem, _with_origin_at_knot(result, knot))
        assert not quadrature.node_count_exhausted, knot
        assert abs(quadrature.value - reference) <= quadrature.error_estimate, knot


@pytest.mark.parametrize("initial_step", [0.03, 0.08])
def test_canonical_pixel_integral_is_invariant_under_initial_step(canonical, initial_step):
    problem, _ = canonical
    reference = integrate_fiber_resampled(problem, canonical[1])
    result = trace_fiber(problem, ContinuationOptions(initial_step=initial_step))
    assert result.status == FiberStatus.CLOSED

    quadrature = integrate_fiber_resampled(problem, result)

    # Invariants only: the sample layout is a controller observation, not a criterion.
    assert not quadrature.node_count_exhausted and quadrature.non_finite_node_count == 0
    assert abs(quadrature.value - reference.value) <= quadrature.error_estimate + reference.error_estimate


@pytest.mark.parametrize("relative_tolerance", [1e-3, 1e-5])
def test_canonical_pixel_integral_is_invariant_under_tolerance(canonical, relative_tolerance):
    problem, result = canonical
    reference = integrate_fiber_resampled(problem, result)

    quadrature = integrate_fiber_resampled(problem, result, ResampleOptions(relative_tolerance=relative_tolerance))

    assert not quadrature.node_count_exhausted
    assert abs(quadrature.value - reference.value) <= quadrature.error_estimate + reference.error_estimate
    if relative_tolerance < reference.relative_tolerance:
        assert quadrature.node_count > reference.node_count
        assert quadrature.error_estimate < reference.error_estimate


def test_canonical_pixel_integral_is_invariant_under_reversed_orientation(canonical):
    problem, forward = canonical
    reverse = trace_fiber(problem, ContinuationOptions(initial_tangent_sign=-1))
    assert reverse.status == FiberStatus.CLOSED
    assert float(np.dot(forward.tangents[0], reverse.tangents[0])) < 0.0

    forward_quadrature = integrate_fiber_resampled(problem, forward)
    reverse_quadrature = integrate_fiber_resampled(problem, reverse)

    assert abs(forward_quadrature.value - reverse_quadrature.value) <= (
        forward_quadrature.error_estimate + reverse_quadrature.error_estimate
    )


def test_performance_evidence_is_printed_not_asserted(canonical, strip_pixels, capsys):
    """Wall clock per fiber with the default options (median of 5 after one warm-up).

    Recorded in progress.md / SUMMARY.md; there is deliberately no ``assert
    wall_time < X`` (environment jitter would make it a flaky gate).
    """
    lines = []
    for name, (problem, result) in _alignment_cases(canonical, strip_pixels):
        integrate_fiber_resampled(problem, result)  # warm-up: jit compilation per grid shape
        samples = []
        for _ in range(5):
            start = time.perf_counter()
            quadrature = integrate_fiber_resampled(problem, result)
            samples.append(time.perf_counter() - start)
        lines.append(
            f"{name}: N={quadrature.node_count} rounds={quadrature.refinement_rounds} "
            f"median={np.median(samples) * 1e3:.1f} ms min={min(samples) * 1e3:.1f} ms "
            f"entry_measure={quadrature.factor_seconds['entry_measure'] * 1e3:.2f} ms"
        )
    with capsys.disabled():
        print("\n[resampled quadrature performance]\n  " + "\n  ".join(lines))
