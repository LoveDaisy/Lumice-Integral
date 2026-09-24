"""``contour_quadrature``: line integrals on the level sets of ``D_P``, against Phase I and an independent dense rule."""

from __future__ import annotations

import numpy as np
import pytest

from lumice_integral import contour_quadrature as cq
from lumice_integral.band_sum import pixel_band
from lumice_integral.canonical_scene import (
    CANONICAL_REFRACTIVE_INDEX,
    canonical_crystal,
    canonical_pixel_problem,
    canonical_pose_density,
    canonical_sun_direction,
    canonical_target_direction,
)
from lumice_integral.camera import incident_direction_from_sun
from lumice_integral.continuation import trace_fiber
from lumice_integral.contour import extract_level_sets
from lumice_integral.dp_field import DPField
from lumice_integral.dp_field import field as F
from lumice_integral.pose_density import build_pose_density
from lumice_integral.quadrature import ResampleOptions, integrate_fiber_resampled
from lumice_integral.s2_store import align_rotations, build_event_store, evaluate_fields, event_rotations

INDEX = CANONICAL_REFRACTIVE_INDEX
TIGHT = cq.QuadratureOptions(relative_tolerance=1e-11)
STORE_N = 200_000


@pytest.fixture(scope="module")
def field() -> DPField:
    return DPField.build(canonical_crystal(), (3, 5), INDEX)


@pytest.fixture(scope="module")
def store():
    return build_event_store(canonical_crystal(), INDEX, [(3, 5)], STORE_N, run_checks=False)


@pytest.fixture(scope="module")
def canonical(field, store):
    """The canonical pixel (150, 150): its level set and geometry."""
    sun = canonical_sun_direction()
    centre, delta, _, _ = pixel_band(150, 150, sun)
    (level_set,) = extract_level_sets(field, [delta], store)
    return sun, centre, level_set, cq.LevelSetGeometry.build(field, [level_set], TIGHT)


def _centre_at(sun: np.ndarray, delta: float, azimuth: float) -> np.ndarray:
    """An outgoing direction at deviation ``delta`` from the propagation ``-sun``, at ``azimuth`` about it."""
    s = incident_direction_from_sun(sun)
    e1 = np.cross(s, [0.0, 0.0, 1.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(s, e1)
    return np.cos(delta) * s + np.sin(delta) * (np.cos(azimuth) * e1 + np.sin(azimuth) * e2)


def test_probe_sun_constant_matches_dp_field() -> None:
    np.testing.assert_array_equal(cq._PROBE_SUN, F._PROBE_SUN)


def test_pointwise_identity_with_phase1_normal_jacobian(field) -> None:
    """``J_perp = |grad D_P(u)| sin(delta) / |xi x u|`` at every node of the canonical Phase I fiber (module docstring)."""
    trace = trace_fiber(canonical_pixel_problem(with_weights=False))
    sun = canonical_sun_direction()
    delta = float(np.arccos(np.clip(canonical_target_direction() @ incident_direction_from_sun(sun), -1.0, 1.0)))
    poses, xi = np.asarray(trace.poses), np.asarray(trace.tangents)
    u = np.einsum("nji,j->ni", poses, sun)
    j_perp = np.array([d.normal_jacobian for d in trace.jacobian_diagnostics])
    predicted = np.linalg.norm(field.gradient_batch(u), axis=1) * np.sin(delta) / np.linalg.norm(np.cross(xi, u), axis=1)
    assert len(poses) > 30
    np.testing.assert_allclose(predicted, j_perp, rtol=1e-12, atol=0.0)


def test_points_on_level_set_with_own_deviation(canonical) -> None:
    """Every quadrature point is on ``D_P = delta`` (its own deviation, which the pose uses) and its pose is a rotation."""
    sun, centre, level_set, geometry = canonical
    assert geometry.max_residual < 1e-12
    q = geometry.panels.q.reshape(-1, 3)
    d = geometry.panels.d.reshape(-1)
    np.testing.assert_allclose(d, level_set.delta, rtol=0.0, atol=1e-12)
    rotations = event_rotations(q, geometry.panels.phi.reshape(-1, 3), d, sun, centre)
    np.testing.assert_allclose(np.einsum("nji,njk->nik", rotations, rotations), np.broadcast_to(np.eye(3), rotations.shape), atol=1e-13)
    np.testing.assert_allclose(np.einsum("nij,nj->ni", rotations, q), np.broadcast_to(sun, q.shape), atol=1e-13)


def test_geometry_weights_are_the_store_weights(field, canonical) -> None:
    """The geometric integrand's ``w`` is :func:`.s2_store.evaluate_fields`' (the event store's weight), not a second implementation."""
    _, _, level_set, geometry = canonical
    panels = geometry.panels
    rows = cq._evaluate_points(field, panels.delta, panels.a, panels.b, panels.t[:, 2], 8)
    fields = evaluate_fields(align_rotations(rows["q"], cq._PROBE_SUN), cq._PROBE_SUN, field.crystal, INDEX, [(3, 5)])
    np.testing.assert_array_equal(rows["phi"], fields["phi"])
    np.testing.assert_array_equal(rows["d"], fields["D"])
    np.testing.assert_allclose(rows["g"], geometry.panels.g[:, 2], rtol=1e-13, atol=0.0)  # batch size changes XLA rounding


def test_canonical_pixel_matches_dense_chord_rule(field, canonical) -> None:
    """An independent rule sharing only the level-set points: trapezoid on geodesic chords, ``M`` points per panel, ``O(M^-2)``."""
    sun, centre, level_set, geometry = canonical
    value = geometry.integrate(sun, [centre], canonical_pose_density())[0].value
    density = canonical_pose_density()
    points = level_set.components[0].points
    estimates = []
    for m in (32, 64):
        t = np.tile(np.arange(m) / m, len(points))
        rows = cq._evaluate_points(field, np.full(len(t), level_set.delta), np.repeat(points, m, axis=0), np.repeat(np.roll(points, -1, axis=0), m, axis=0), t, 8)
        q = rows["q"]
        rho = density.evaluate_batch(event_rotations(q, rows["phi"], rows["d"], sun, centre))
        w = evaluate_fields(align_rotations(q, cq._PROBE_SUN), cq._PROBE_SUN, field.crystal, INDEX, [(3, 5)])["w"]
        f = w * rho / np.linalg.norm(field.gradient_batch(q), axis=1)
        nxt = np.roll(q, -1, axis=0)
        chord = np.arctan2(np.linalg.norm(np.cross(q, nxt), axis=1), np.sum(q * nxt, axis=1))
        estimates.append(float(np.sum(0.5 * (f + np.roll(f, -1)) * chord)) * cq._scale(level_set.delta))
    extrapolated = estimates[1] + (estimates[1] - estimates[0]) / 3.0
    assert abs(estimates[1] - estimates[0]) > 1e-9 * value  # the dense rule still moves: it is not the same computation
    assert abs(extrapolated / value - 1.0) < 1e-9


def test_canonical_pixel_against_phase1(canonical) -> None:
    """Production Phase I (``eps = 1e-12``, ``rtol = 1e-9``) agrees to ``1e-5``, not better.

    The remaining ``5.6e-6`` is Phase I's: its arclength speed drops the
    ``nu' . delta`` term of the differentiated phase condition
    (``quadrature._parametric_speed``), which ``scripts/compare_contour_quadrature_phase1.py``
    restores (then ``<= 1e-7``); ``docs/phase2.md`` section 4.
    """
    sun, centre, _, geometry = canonical
    value = geometry.integrate(sun, [centre], canonical_pose_density())[0].value
    problem = canonical_pixel_problem(with_weights=True)
    phase1 = integrate_fiber_resampled(
        problem, trace_fiber(problem), ResampleOptions(epsilon=1e-12, relative_tolerance=1e-9, maximum_node_count=262145)
    )
    assert abs(phase1.error_estimate) < 1e-8 * value
    assert 1e-6 < abs(phase1.value / value - 1.0) < 1e-5


def test_tighter_tolerance_stays_within_the_error_estimate(field, canonical) -> None:
    sun, centre, level_set, geometry = canonical
    density = canonical_pose_density()
    loose = cq.LevelSetGeometry.build(field, [level_set], cq.QuadratureOptions(relative_tolerance=1e-6)).integrate(sun, [centre], density)[0]
    tight = geometry.integrate(sun, [centre], density)[0]
    assert loose.error_estimate < 1e-6 * loose.value
    assert abs(loose.value - tight.value) < 20.0 * loose.error_estimate + 1e-12 * tight.value
    assert tight.error_estimate < 1e-9 * tight.value
    assert tight.exhausted_panels == 0 and tight.non_finite_points == 0


def test_kinks_are_reported_and_depth_limit_is_not_hidden(field, canonical) -> None:
    """``A_P`` kinks on the canonical loop lower the local order (``low_order_splits``); ``max_depth = 0`` reports exhaustion."""
    sun, centre, level_set, geometry = canonical
    assert geometry.low_order_splits[0] > 0
    shallow = cq.LevelSetGeometry.build(field, [level_set], cq.QuadratureOptions(max_depth=0))
    result = shallow.integrate(sun, [centre], canonical_pose_density())[0]
    assert result.exhausted_panels > 0 and np.isfinite(result.value)


def test_random_density_value_depends_on_delta_only(canonical) -> None:
    """AC7: with ``rho = 1`` the pixel value is a function of ``delta`` alone (any azimuth, bit for bit)."""
    sun, _, level_set, geometry = canonical
    random = build_pose_density("random")
    centres = [_centre_at(sun, level_set.delta, a) for a in (0.1, 1.3, 2.9, -2.0)]
    values = {r.value for r in geometry.integrate(sun, centres, random, [0, 0, 0, 0])}
    assert len(values) == 1
    (value,) = values
    assert value > 0.0


def test_open_arcs_integrate(field, store) -> None:
    """``delta = 45 deg`` on 3-5: two open arcs ending on ``dU_P`` (exit TIR / entry grazing); finite, converged."""
    sun = canonical_sun_direction()
    (level_set,) = extract_level_sets(field, [np.radians(45.0)], store)
    assert (level_set.n_closed, level_set.n_open) == (0, 2)
    geometry = cq.LevelSetGeometry.build(field, [level_set])
    (result,) = geometry.integrate(sun, [_centre_at(sun, level_set.delta, 0.4)], build_pose_density("random"))
    assert result.value > 0.0 and all(v > 0.0 for v in result.component_values)
    assert result.non_finite_points == 0 and result.exhausted_panels == 0
    assert result.error_estimate < 1e-8 * result.value


def test_critical_delta_is_flagged(field) -> None:
    minimum = float(field.interval_partition()[0].lower)
    assert cq.critical_delta(field, minimum + 1e-9)
    assert not cq.critical_delta(field, minimum + 1e-3)


def test_level_sets_and_pixels_batched_together_match_one_at_a_time(field, store) -> None:
    """Units are independent: several level sets and pixels in one adaptive run give each one's own value."""
    sun = canonical_sun_direction()
    density = canonical_pose_density()
    bands = [pixel_band(row, 150, sun) for row in (120, 150, 400)]
    level_sets = extract_level_sets(field, [b[1] for b in bands], store)
    together = cq.LevelSetGeometry.build(field, level_sets).integrate(sun, [b[0] for b in bands], density)
    for level_set, band, batched in zip(level_sets, bands, together):
        (alone,) = cq.LevelSetGeometry.build(field, [level_set]).integrate(sun, [band[0]], density)
        assert batched.panels == alone.panels and batched.evaluations == alone.evaluations
        assert abs(batched.value / alone.value - 1.0) < 1e-12


def test_band_average_pixel_model(field, store) -> None:
    """``band_nodes``: Gauss-Legendre over the pixel's deviation band converges (2 vs 4 nodes) and differs from the point value by O(band^2)."""
    from lumice_integral.canonical_scene import CANONICAL_RENDER

    def scene(nodes: int) -> cq.ContourQuadratureScene:
        return cq.ContourQuadratureScene((3, 5), canonical_crystal(), INDEX, canonical_sun_direction(), canonical_pose_density(),
                                         CANONICAL_RENDER, band_nodes=nodes)

    rows = (150, 300)
    point, _ = cq.render_column(scene(0), field, store, 150, rows)
    two, _ = cq.render_column(scene(2), field, store, 150, rows)
    four, timing = cq.render_column(scene(4), field, store, 150, rows)
    assert set(timing) == {"extract_s", "geometry_s", "integrate_s"}
    for p, b2, b4 in zip(point, two, four):
        assert b2.level_sets == 2 and b4.level_sets == 4 and p.level_sets == 1
        assert abs(b2.value / b4.value - 1.0) < 1e-7
        assert 1e-9 < abs(b4.value / p.value - 1.0) < 1e-4  # smooth pixels: the band average is near, not equal to, the point value
    lo, hi = pixel_band(150, 150, canonical_sun_direction())[2:]
    deviations, weights = cq.band_deviations(field, lo, hi, 4)
    assert len(deviations) == 4 and abs(weights.sum() - (hi - lo)) < 1e-15
    minimum = float(field.interval_partition()[0].lower)
    deviations, weights = cq.band_deviations(field, minimum - 1e-4, minimum + 1e-4, 3)
    assert len(deviations) == 6 and np.all(np.abs(deviations - minimum) > 1e-6)  # split at the critical value
