"""Tests for ``lumice_integral.geometry.entry_measure`` (original to this repository).

Coverage map (plan Step 6):

1. normal-incidence analytic case (does not exercise the cosine conversion branch);
2. oblique symmetric minimum-deviation 3-5 case with a mechanical "un-clipped" guard -- the only
   closed-form case where ``cos_i != cos_t``;
2b. clipped configuration against an independent brute-force footprint oracle;
3. optically feasible but geometrically empty corridor -> ``value == 0.0``, ``status == "corridor_empty"``;
4. canonical 3-5 fixture: every pose of the traced fiber evaluates to a finite non-negative value;
5. refraction cross-consistency with ``optics.refract_smooth`` / ``optics.path_3_5`` (the geometry
   package cannot import JAX, so the Snell step is duplicated in numpy and pinned here).

The geometry package itself is pure numpy; this test file may import JAX-backed modules.
"""

import math

import jax.numpy as jnp
import numpy as np
import pytest

from lumice_integral.continuation import trace_fiber
from lumice_integral.geometry import (COS_CRITICAL, HexPrism, LatLonGrid, corridor_intersection, corridor_polygons,
                                      entry_measure, entry_ok, exit_ok, external_directions)
from lumice_integral.geometry.entry_measure import refract_into_crystal
from lumice_integral.optics import (ICE_REFRACTIVE_INDEX, minimum_deviation_incident, path_3_5, path_3_5_problem,
                                    refract_smooth)
from lumice_integral.so3 import exp

N_ICE = float(ICE_REFRACTIVE_INDEX)
CRYSTAL = HexPrism(a=1.0, h=1.0)
EYE = np.eye(3)
MIN_DEV_INCIDENT = np.asarray(minimum_deviation_incident(ICE_REFRACTIVE_INDEX), dtype=np.float64)
COS_THETA_I = 0.755628877161269          # cos(arcsin(1.31 sin 30 deg)), from docs/ch06-reference-fixture.md section 4


# ---- 1. normal incidence -----------------------------------------------------------------------------


def test_normal_incidence_straight_through_equals_face_area():
    """(3, 6) at normal incidence: s_body = d_in = -n_3, the whole face maps onto the opposite face, so the
    cross-section is the face area a*h.  cos_i == cos_t here, so this case does not test the conversion."""
    res = entry_measure(EYE, (3, 6), [-1.0, 0.0, 0.0], CRYSTAL)
    assert res.status == "ok"
    assert res.value == pytest.approx(CRYSTAL.a * CRYSTAL.h, abs=1e-10)
    assert res.cosine_incident == pytest.approx(1.0) and res.cosine_internal == pytest.approx(1.0)
    np.testing.assert_allclose(res.internal_direction, [-1.0, 0.0, 0.0], atol=1e-15)


# ---- 2. oblique symmetric minimum-deviation 3-5 ------------------------------------------------------


def test_min_deviation_3_5_corridor_is_unclipped_guard():
    """Mechanical guard for the analytic case below: at the symmetric minimum-deviation direction the
    internal ray is horizontal and parallel to face 4, so the corridor footprint is the *entire* face 3
    (A_perp = a*h*cos 30 deg, a 4-vertex intersection).  Any tangential or axial deviation would clip."""
    d_in, cos_i, cos_t = refract_into_crystal(MIN_DEV_INCIDENT, CRYSTAL.normal(CRYSTAL.face(3)), N_ICE)
    np.testing.assert_allclose(d_in, [-math.sqrt(3) / 2, 0.5, 0.0], atol=1e-12)
    polys, _ = corridor_polygons(CRYSTAL, (3, 5))
    batch = corridor_intersection(polys, d_in[None, :])
    assert batch.count[0] == 4
    assert batch.area()[0] == pytest.approx(CRYSTAL.a * CRYSTAL.h * math.cos(math.pi / 6), abs=1e-12)


def test_min_deviation_3_5_value_is_face_area_times_cos_theta_i():
    """value = A_perp(d_in) * cos_i / cos_t = (a*h*cos_t) * cos_i / cos_t = a*h*cos_i.  cos_t (30 deg) appears in
    A_perp and is cancelled by the conversion; cos_i (40.92 deg) != cos_t, so the discriminating branch runs."""
    res = entry_measure(EYE, (3, 5), MIN_DEV_INCIDENT, CRYSTAL)
    assert res.status == "ok"
    assert res.value == pytest.approx(CRYSTAL.a * CRYSTAL.h * COS_THETA_I, abs=1e-10)
    assert res.cosine_incident == pytest.approx(COS_THETA_I, abs=1e-12)
    assert res.cosine_internal == pytest.approx(math.cos(math.pi / 6), abs=1e-12)
    assert res.area_perp_internal == pytest.approx(math.cos(math.pi / 6), abs=1e-12)


# ---- 2b. clipped configuration vs brute-force oracle -------------------------------------------------


def _brute_force_footprint_fraction(crystal: HexPrism, entry: int, exit_face: int, s: np.ndarray,
                                    n_samples: int, seed: int) -> float:
    """Fraction of uniformly sampled entry-face points whose refracted internal line leaves the convex body
    through ``exit_face`` first.  Uses only face planes and Snell's law -- no unfolding, no corridor clipping."""
    rng = np.random.default_rng(seed)
    face = crystal.face(entry)
    v = crystal.face_vertices(face)                      # prism side face: a rectangle
    p = v[0] + rng.random((n_samples, 1)) * (v[1] - v[0]) + rng.random((n_samples, 1)) * (v[3] - v[0])
    n = crystal.normal(face)
    eta = 1.0 / N_ICE
    cos_i = -(n @ s)
    d = eta * s + (eta * cos_i - math.sqrt(1.0 - eta * eta * (1.0 - cos_i * cos_i))) * n
    normals = np.stack([crystal.normal(f) for f in crystal.faces])
    anchors = np.stack([crystal.face_vertices(f)[0] for f in crystal.faces])
    denom = normals @ d
    leaving = denom > 1e-12
    num = np.einsum("fj,nfj->nf", normals, anchors[None, :, :] - p[:, None, :])
    t = np.where(leaving[None, :], num / np.where(leaving, denom, 1.0)[None, :], np.inf)
    numbers = np.array([f.number for f in crystal.faces])
    return float((numbers[t.argmin(axis=1)] == exit_face).mean())


def test_clipped_3_5_matches_brute_force_footprint():
    """Add a 0.2 rad axial tilt to the minimum-deviation incident direction: the internal ray now drifts in z
    and part of face 3 no longer reaches face 5.  Expected value = f * a*h * cos_i with f from an independent
    Monte Carlo footprint estimate; tolerance is 5 sigma of that estimate."""
    tilt = 0.2
    s = MIN_DEV_INCIDENT * math.cos(tilt) + np.array([0.0, 0.0, math.sin(tilt)])
    n_samples, seed = 200_000, 0
    frac = _brute_force_footprint_fraction(CRYSTAL, 3, 5, s, n_samples, seed)
    assert 0.05 < frac < 0.95, "configuration must actually be clipped for this test to mean anything"
    cos_i = -(CRYSTAL.normal(CRYSTAL.face(3)) @ s)
    # the oracle ignores the exit critical angle; confirm it is not active here so the comparison is fair
    d_in, _, _ = refract_into_crystal(s, CRYSTAL.normal(CRYSTAL.face(3)), N_ICE)
    assert CRYSTAL.normal(CRYSTAL.face(5)) @ d_in >= COS_CRITICAL
    expected = frac * CRYSTAL.a * CRYSTAL.h * cos_i
    sigma = math.sqrt(frac * (1.0 - frac) / n_samples) * CRYSTAL.a * CRYSTAL.h * cos_i
    res = entry_measure(EYE, (3, 5), s, CRYSTAL)
    assert res.status == "ok"
    assert abs(res.value - expected) < 5.0 * sigma
    assert res.value < CRYSTAL.a * CRYSTAL.h * cos_i          # strictly less than the un-clipped bound


# ---- 3. optically feasible, geometrically empty ------------------------------------------------------


def _find_empty_corridor_direction(crystal: HexPrism, path: tuple[int, ...]) -> np.ndarray | None:
    """Search grid directions that pass both optical gates but whose corridor intersection is *exactly* empty
    (fewer than 3 vertices), so the verdict cannot flip on floating-point noise near ``eps``."""
    n_a = crystal.normal(crystal.face(path[0]))
    polys, n_tilde_b = corridor_polygons(crystal, path)
    for grid in (LatLonGrid(24, 36), LatLonGrid(48, 72)):
        d = grid.directions
        ok = entry_ok(n_a, d) & exit_ok(n_tilde_b, d)
        if not ok.any():
            continue
        batch = corridor_intersection(polys, d[ok])
        empty = np.flatnonzero(batch.count < 3)
        if len(empty):
            return d[ok][empty[0]]
    return None


def test_optically_feasible_but_empty_corridor_gives_zero_with_status():
    for path in ((3, 4, 5), (3, 4, 6), (3, 1, 5)):
        d_in = _find_empty_corridor_direction(CRYSTAL, path)
        if d_in is not None:
            break
    else:
        pytest.fail("no optically feasible, geometrically empty direction found on any candidate path")
    s = external_directions(CRYSTAL, path, d_in[None, :])[0]    # rotation = I: world frame == body frame
    assert np.isfinite(s).all()
    res = entry_measure(EYE, path, s, CRYSTAL)
    assert res.value == 0.0
    assert res.status == "corridor_empty"
    np.testing.assert_allclose(res.internal_direction, d_in, atol=1e-12)
    assert res.area_perp_internal == 0.0


def test_gate_statuses_are_reported_not_silent():
    # ray arriving from behind face 3
    back = entry_measure(EYE, (3, 5), [1.0, 0.0, 0.0], CRYSTAL)
    assert back.value == 0.0 and back.status == "entry_backface" and back.internal_direction is None
    # (3, 4): adjacent faces, internal direction can never satisfy both critical-angle cones (60 deg apart)
    crit = entry_measure(EYE, (3, 4), MIN_DEV_INCIDENT, CRYSTAL)
    assert crit.value == 0.0 and crit.status == "exit_critical_angle"
    with pytest.raises(ValueError):
        entry_measure(EYE, (3,), MIN_DEV_INCIDENT, CRYSTAL)


# ---- 4. canonical 3-5 fixture, pose by pose -----------------------------------------------------------


@pytest.fixture(scope="module")
def canonical_fiber():
    """docs/ch06-reference-fixture.md section 4: seed Exp([0.15, 0.08, -0.05]), min-deviation incident, path 3-5.
    Same construction as scripts/inspect_path_3_5.py."""
    seed = exp(jnp.array([0.15, 0.08, -0.05], dtype=jnp.float64))
    incident = minimum_deviation_incident()
    target = path_3_5(seed, incident).direction
    return trace_fiber(path_3_5_problem(seed, incident, target_direction=target))


def test_canonical_fiber_entry_measure_is_finite_and_nonnegative(canonical_fiber):
    result = canonical_fiber
    assert result.status.value == "closed"
    poses = np.asarray(result.poses, dtype=np.float64)
    assert len(poses) > 20           # the fixture stores ~49 poses (one traversal of its 0.964 loop); do not freeze the incidental count
    values = np.array([entry_measure(R, (3, 5), MIN_DEV_INCIDENT, CRYSTAL).value for R in poses])
    assert np.isfinite(values).all()
    assert (values >= 0.0).all()
    assert values.max() <= CRYSTAL.a * CRYSTAL.h + 1e-12     # never exceeds the entry-face area
    assert (values > 0.0).any()      # the fiber passes through poses where the 3-5 corridor is open


# ---- 5. refraction cross-consistency with optics.py -------------------------------------------------


@pytest.mark.parametrize("theta_deg", [0.0, 10.0, 30.0, 45.0, 60.0, 80.0, 89.0])
def test_numpy_refraction_matches_optics_refract_smooth(theta_deg):
    th = math.radians(theta_deg)
    s = np.array([-math.cos(th), math.sin(th) * 0.6, math.sin(th) * 0.8])
    n = np.array([1.0, 0.0, 0.0])
    d_np, cos_i, cos_t = refract_into_crystal(s, n, N_ICE)
    ref = refract_smooth(jnp.asarray(s), jnp.asarray(n), 1.0 / ICE_REFRACTIVE_INDEX)
    np.testing.assert_allclose(d_np, np.asarray(ref.direction), atol=1e-14)
    assert cos_i == pytest.approx(float(ref.incidence_cosine), abs=1e-14)
    assert cos_t == pytest.approx(-float(n @ np.asarray(ref.direction)), abs=1e-14)


def test_entry_measure_internal_direction_matches_optics_path_3_5_in_world_frame():
    """Body/world convention cross-check: ``optics.path_3_5`` refracts in the world frame with
    ``rotation @ FACE_3_NORMAL``; ``entry_measure`` refracts in the body frame.  ``R @ d_in`` must agree."""
    rng = np.random.default_rng(1)
    checked = 0
    for _ in range(20):
        R = np.asarray(exp(jnp.asarray(rng.normal(size=3) * 0.4)), dtype=np.float64)
        res = entry_measure(R, (3, 5), MIN_DEV_INCIDENT, CRYSTAL)
        if res.internal_direction is None:
            continue
        world = np.asarray(path_3_5(jnp.asarray(R), jnp.asarray(MIN_DEV_INCIDENT)).entry.direction)
        np.testing.assert_allclose(R @ res.internal_direction, world, atol=1e-13)
        checked += 1
    assert checked >= 10


# ---- 6. batch form (task-resample-and-integrate Step 3) --------------------------------------------


def test_entry_measure_batch_matches_the_scalar_form_pose_by_pose(canonical_fiber):
    """Elementwise agreement of ``entry_measure_batch`` with ``entry_measure``: fiber poses (open
    corridor), Haar-like random poses (every gate verdict) and a batch that is empty."""
    from lumice_integral.geometry import entry_measure_batch

    rng = np.random.default_rng(11)
    random_poses = np.asarray([np.asarray(exp(jnp.asarray(rng.normal(size=3)))) for _ in range(500)])
    poses = np.concatenate([np.asarray(canonical_fiber.poses, dtype=np.float64), random_poses])

    batch = entry_measure_batch(poses, (3, 5), MIN_DEV_INCIDENT, CRYSTAL, n_ice=N_ICE)
    scalar = np.array([entry_measure(R, (3, 5), MIN_DEV_INCIDENT, CRYSTAL, n_ice=N_ICE).value for R in poses])
    statuses = {entry_measure(R, (3, 5), MIN_DEV_INCIDENT, CRYSTAL, n_ice=N_ICE).status for R in random_poses}

    assert batch.shape == (len(poses),) and batch.dtype == np.float64
    np.testing.assert_allclose(batch, scalar, rtol=0.0, atol=1e-15)
    assert {"ok", "entry_backface"} <= statuses      # both the open corridor and a failed gate are exercised
    assert entry_measure_batch(np.zeros((0, 3, 3)), (3, 5), MIN_DEV_INCIDENT, CRYSTAL).shape == (0,)
    with pytest.raises(ValueError):
        entry_measure_batch(np.eye(3), (3, 5), MIN_DEV_INCIDENT, CRYSTAL)
