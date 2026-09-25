"""``optics.path_direction`` / ``path_domain`` on arbitrary face sequences vs the ray-cast oracle.

The oracle side is ``tests/_geometry_oracles.py`` (closed-form normals, vector
Snell ``refract`` and mirror ``reflect``; it shares no code with ``optics``).
Three paths are checked pose by pose on Haar samples that pass the smooth
domain: ``3-5`` (the pre-generalisation baseline), ``3-7`` (the other face
pair of the same class) and ``3-1-2-5`` (two basal internal reflections,
the ``M = I`` 60-deg wedge of the issue).  The event classification of
``path_domain`` is checked on analytically constructed boundary poses.  An
internal reflection that is not total stays on the smooth branch (a weight,
not an event); its reflectance is checked against Lumice's ``GetReflectRatio``
form, written out here independently of ``optics``.
"""

from __future__ import annotations

from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest

from lumice_integral.geometry import HexPrism, wedge_angle_deg
from lumice_integral.optics import (
    DOMAIN_MARGIN_NAMES,
    ICE_REFRACTIVE_INDEX,
    PATH_3_5_FACES,
    domain_margin_names,
    faces_of_path_id,
    fresnel_transmission_3_5,
    fresnel_transmission_3_5_batch,
    fresnel_transmission_path,
    fresnel_transmission_path_batch,
    internal_reflectance,
    minimum_deviation_incident,
    normalize_faces,
    path_3_5,
    path_3_5_domain,
    path_3_5_domain_batch,
    path_3_5_problem,
    path_direction,
    path_domain,
    path_domain_batch,
    path_id_of,
    path_problem,
    problem_path_label,
    SNELL_EVENT_TOLERANCE,
    validity_margin_names,
)
from lumice_integral.continuation import ContinuationOptions, TerminationReason, trace_fiber
from lumice_integral.discovery import ARC_EVENTS
from lumice_integral.so3 import exp, haar_rotations, log

from _geometry_oracles import NORMALS, reflect, refract

INDEX = float(ICE_REFRACTIVE_INDEX)
INCIDENT = np.asarray(minimum_deviation_incident(), dtype=np.float64)
PATHS = [(3, 5), (3, 7), (3, 1, 2, 5), (3, 2, 1, 5), (1, 3, 2)]


def oracle_direction(rotation: np.ndarray, faces: tuple[int, ...], incident: np.ndarray) -> np.ndarray:
    """World-frame outgoing direction by the oracle's Snell/mirror primitives; raises off the smooth branch."""
    normals = [rotation @ NORMALS[face] for face in faces]
    if -(normals[0] @ incident) <= 0.0:
        raise ValueError("back-face entry")
    direction = refract(incident, normals[0], 1.0, INDEX)
    for normal in normals[1:-1]:
        cosine = direction @ normal
        if cosine <= 0.0:
            raise ValueError("internal ray does not reach the face")
        direction = reflect(direction, normal)
    if direction @ normals[-1] <= 0.0:
        raise ValueError("back-face exit")
    return refract(direction, -normals[-1], INDEX, 1.0)


def _valid_samples(faces: tuple[int, ...], count: int, seed: int = 7) -> np.ndarray:
    rotations = haar_rotations(count, np.random.default_rng(seed))
    check = path_domain_batch(rotations, faces, INCIDENT, INDEX)
    return rotations[check.valid]


@pytest.mark.parametrize("faces", PATHS)
def test_path_direction_matches_the_oracle_on_domain_valid_poses(faces):
    rotations = _valid_samples(faces, 20_000)
    assert len(rotations) >= 50, f"{faces}: too few domain-valid samples to test"
    for rotation in rotations[:200]:
        expected = oracle_direction(rotation, faces, INCIDENT)
        actual = np.asarray(path_direction(jnp.asarray(rotation), faces, jnp.asarray(INCIDENT), jnp.asarray(INDEX)).direction)
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)
        assert path_domain(rotation, faces, INCIDENT, INDEX).valid


@pytest.mark.parametrize("faces", PATHS)
def test_domain_verdict_agrees_with_the_oracle_on_haar_samples(faces):
    """Valid iff the oracle traces the branch without raising; invalid poses name a face in the message."""
    rotations = haar_rotations(2_000, np.random.default_rng(11))
    batch = path_domain_batch(rotations, faces, INCIDENT, INDEX)
    assert tuple(batch.margins) == domain_margin_names(faces)
    for rotation, valid in zip(rotations, batch.valid):
        scalar = path_domain(rotation, faces, INCIDENT, INDEX)
        assert scalar.valid == bool(valid)
        try:
            oracle_direction(rotation, faces, INCIDENT)
        except ValueError:
            assert not scalar.valid
            assert scalar.event_kind in ("path_infeasible", "tir_boundary")
        else:
            assert scalar.valid
            assert set(scalar.margins) == set(domain_margin_names(faces))


def test_3_1_2_5_and_3_5_map_to_the_same_direction_where_both_are_valid():
    """Same Phi (M = I, 60-deg wedge): the outgoing direction of ``3-1-2-5`` equals that of ``3-5``
    at the same pose, with different weights (a different footprint), which is what a class sums."""
    crystal = HexPrism()
    assert wedge_angle_deg(crystal, (3, 1, 2, 5)) == pytest.approx(wedge_angle_deg(crystal, (3, 5)))
    rotations = _valid_samples((3, 1, 2, 5), 50_000)
    assert len(rotations) >= 20
    for rotation in rotations[:100]:
        assert path_domain(rotation, PATH_3_5_FACES, INCIDENT, INDEX).valid
        long_way = path_direction(jnp.asarray(rotation), (3, 1, 2, 5), jnp.asarray(INCIDENT)).direction
        short_way = path_3_5(jnp.asarray(rotation), jnp.asarray(INCIDENT)).direction
        np.testing.assert_allclose(np.asarray(long_way), np.asarray(short_way), rtol=0.0, atol=1e-12)


def test_internal_reflection_events_are_classified_in_ray_order():
    """A pose whose internal ray reaches face 1 but is not totally reflected there stays on the branch
    (its TIR discriminant is a diagnostic margin, not a gate); one whose internal ray misses face 1
    (goes to face 2 instead) stops at that face's incidence cosine."""
    faces = (3, 1, 2, 5)
    rotations = haar_rotations(20_000, np.random.default_rng(3))
    batch = path_domain_batch(rotations, faces, INCIDENT, INDEX)
    margins = batch.margins
    entry_ok = (margins["entry_incidence_cosine"] > 0) & (margins["entry_snell_discriminant"] > 0)
    not_total = entry_ok & (margins["internal_1_incidence_cosine"] > 0) & (margins["internal_1_tir_discriminant"] <= 0)
    misses = entry_ok & (margins["internal_1_incidence_cosine"] <= 0)
    assert not_total.any() and misses.any()
    for index in np.flatnonzero(not_total)[:50]:
        check = path_domain(rotations[index], faces, INCIDENT, INDEX)
        assert check.valid == bool(batch.valid[index])
        assert "internal_2_incidence_cosine" in check.margins
        assert check.event_kind != "tir_boundary" or check.message.startswith("exit")
    partial_valid = not_total & batch.valid
    assert partial_valid.any()
    check = path_domain(rotations[np.flatnonzero(partial_valid)[0]], faces, INCIDENT, INDEX)
    assert check.valid and check.event_kind is None and check.margins["internal_1_tir_discriminant"] <= 0
    check = path_domain(rotations[np.flatnonzero(misses)[0]], faces, INCIDENT, INDEX)
    assert check.event_kind == "path_infeasible" and "face 1" in check.message
    assert "internal_1_tir_discriminant" in check.margins and "internal_2_incidence_cosine" not in check.margins


def test_validity_margins_leave_out_only_the_internal_tir_discriminants():
    for faces in PATHS + [(3, 5, 6, 7), (3, 4, 5, 7)]:
        names = domain_margin_names(faces)
        gates = validity_margin_names(faces)
        assert set(names) - set(gates) == {f"internal_{k}_tir_discriminant" for k in range(1, len(faces) - 1)}
        assert tuple(name for name in names if name in gates) == gates
    assert validity_margin_names(PATH_3_5_FACES) == DOMAIN_MARGIN_NAMES


def lumice_reflect_ratio(delta: float, rr: float) -> float:
    """``lm_optics::GetReflectRatio`` (Ice Halo ``src/core/shared/optics_shared.h``), transcribed."""
    d_sqrt = np.sqrt(delta)
    r_s = ((rr - d_sqrt) / (rr + d_sqrt)) ** 2
    r_p = ((1.0 - rr * d_sqrt) / (1.0 + rr * d_sqrt)) ** 2
    return 0.5 * (r_s + r_p)


def lumice_internal_reflectance(cos_theta: float, n: float) -> float:
    """``HitSurface`` from inside (``cos_theta > 0`` so ``rr = n``), ``delta`` clamped at 0 as Lumice does."""
    rr = n
    delta = (1.0 - rr * rr) / (cos_theta * cos_theta) + rr * rr
    return lumice_reflect_ratio(max(delta, 0.0), rr)


def test_internal_reflectance_matches_lumice_at_known_angles():
    # 30 deg internal incidence (the 142-deg parhelion's face-5 reflection at minimum deviation):
    # R ~ 2.23 %; the critical angle and beyond give R = 1.
    critical = np.degrees(np.arcsin(1.0 / INDEX))
    for angle_deg in (0.0, 10.0, 30.0, 45.0, critical - 1e-6, critical + 1e-6, 60.0, 89.0):
        cosine = np.cos(np.radians(angle_deg))
        discriminant = INDEX**2 * (1.0 - cosine**2) - 1.0
        expected = lumice_internal_reflectance(cosine, INDEX)
        actual = float(internal_reflectance(INDEX, cosine, discriminant))
        assert actual == pytest.approx(expected, rel=1e-12, abs=1e-15)
    cosine = np.cos(np.radians(30.0))
    assert float(internal_reflectance(INDEX, cosine, INDEX**2 * (1 - cosine**2) - 1)) == pytest.approx(0.02231, abs=5e-5)
    assert float(internal_reflectance(INDEX, 0.3, INDEX**2 * (1 - 0.09) - 1)) == 1.0
    # Continuous across the critical angle: R -> 1 from the partial side.
    below = np.cos(np.radians(critical - 1e-7))
    assert float(internal_reflectance(INDEX, below, INDEX**2 * (1 - below**2) - 1)) == pytest.approx(1.0, abs=1e-3)


def test_the_3_5_wrappers_are_the_generic_functions_at_faces_3_5():
    rotations = haar_rotations(500, np.random.default_rng(5))
    generic = path_domain_batch(rotations, PATH_3_5_FACES, INCIDENT, INDEX)
    legacy = path_3_5_domain_batch(rotations, INCIDENT, INDEX)
    assert np.array_equal(generic.valid, legacy.valid)
    assert np.array_equal(generic.direction, legacy.direction, equal_nan=True)
    for name in DOMAIN_MARGIN_NAMES:
        assert np.array_equal(generic.margins[name], legacy.margins[name], equal_nan=True)
    assert np.array_equal(
        fresnel_transmission_path_batch(rotations, PATH_3_5_FACES, INCIDENT, INDEX),
        fresnel_transmission_3_5_batch(rotations, INCIDENT, INDEX),
    )
    for rotation in rotations[:20]:
        assert fresnel_transmission_path(rotation, PATH_3_5_FACES, INCIDENT, INDEX) == fresnel_transmission_3_5(
            rotation, INCIDENT, INDEX
        )
        assert path_3_5_domain(rotation, INCIDENT, INDEX) == path_domain(rotation, PATH_3_5_FACES, INCIDENT, INDEX)
    assert DOMAIN_MARGIN_NAMES == domain_margin_names(PATH_3_5_FACES) == (
        "entry_incidence_cosine",
        "entry_snell_discriminant",
        "exit_incidence_cosine",
        "exit_snell_discriminant",
    )


def test_fresnel_of_a_reflecting_path_multiplies_every_internal_reflectance():
    """``3-1-2-5`` at a pose = ``3-5`` at the same pose (same entry and exit cosines, M = I) times the two
    basal reflectances, each recomputed from the oracle's own ray (Lumice's formula, not ``optics``)."""
    rotations = _valid_samples((3, 1, 2, 5), 50_000)[:200]
    values = fresnel_transmission_path_batch(rotations, (3, 1, 2, 5), INCIDENT, INDEX)
    same_pose_3_5 = fresnel_transmission_path_batch(rotations, (3, 5), INCIDENT, INDEX)
    partial_poses = 0
    for rotation, value, base in zip(rotations, values, same_pose_3_5):
        direction = refract(INCIDENT, rotation @ NORMALS[3], 1.0, INDEX)
        reflectance = 1.0
        for face in (1, 2):
            normal = rotation @ NORMALS[face]
            reflectance *= lumice_internal_reflectance(float(direction @ normal), INDEX)
            direction = reflect(direction, normal)
        partial_poses += reflectance < 1.0
        assert value == pytest.approx(base * reflectance, rel=1e-12, abs=1e-15)
    assert partial_poses > 0
    assert np.all((values > 0.0) & (values < 1.0))
    for rotation, value in zip(rotations[:10], values):
        assert fresnel_transmission_path(rotation, (3, 1, 2, 5), INCIDENT, INDEX) == pytest.approx(value, abs=1e-12)


def test_path_problem_labels_and_wrapper_equivalence():
    rotation = _valid_samples((3, 1, 2, 5), 50_000)[0]
    problem = path_problem(jnp.asarray(rotation), (3, 1, 2, 5), jnp.asarray(INCIDENT), refractive_index=jnp.asarray(INDEX))
    assert problem.path == problem_path_label((3, 1, 2, 5), INDEX) == "3-1-2-5:n=1.31"
    legacy = path_3_5_problem(jnp.asarray(rotation), jnp.asarray(INCIDENT), refractive_index=jnp.asarray(INDEX))
    assert legacy.path == "3-5:n=1.31"
    # Same Phi: the two seeds' targets coincide up to the two arithmetic routes.
    np.testing.assert_allclose(np.asarray(legacy.target_chart.direction), np.asarray(problem.target_chart.direction), rtol=0.0, atol=1e-12)
    evaluation = problem.domain_and_event_evaluator(jnp.asarray(rotation))
    # Continuation steers by every margin it is given, so it gets the event margins only.
    assert evaluation.valid and tuple(evaluation.margins) == validity_margin_names((3, 1, 2, 5))


def test_fiber_crosses_the_internal_critical_angle_at_full_step():
    """A 3-5-6-7-3 fiber runs through ``internal_k_tir_discriminant = 0`` as through any interior point.

    The discriminant gates nothing, so it is not an event margin of the
    continuation problem: before it was dropped from them, the event-approach
    step limit read it as a boundary, its negative linear-rate distance
    pinned every step past the critical angle at ``minimum_step`` and the
    trace ran out of steps (A60-10 pixels came out 0, task
    phase1-partial-reflection-domain).
    """
    faces = (3, 5, 6, 7, 3)
    names = [f"internal_{k}_tir_discriminant" for k in range(1, len(faces) - 1)]
    rotations = haar_rotations(20_000, np.random.default_rng(3))
    check = path_domain_batch(rotations, faces, INCIDENT, INDEX)
    near = check.valid & np.any(np.stack([np.abs(check.margins[name]) < 0.05 for name in names]), axis=0)
    options = ContinuationOptions()
    crossed = 0
    for index in np.flatnonzero(near)[:3]:
        problem = path_problem(jnp.asarray(rotations[index]), faces, jnp.asarray(INCIDENT), refractive_index=jnp.asarray(INDEX))
        result = trace_fiber(problem, options)
        assert result.reason in ARC_EVENTS or result.reason == TerminationReason.CLOSED_LOOP
        assert all(tuple(margins) == validity_margin_names(faces) for margins in result.branch_diagnostics.accepted_margins)
        margins = path_domain_batch(np.asarray(result.poses), faces, INCIDENT, INDEX).margins
        crossed += any(margins[name].min() < 0.0 < margins[name].max() for name in names)
        pinned = sum(d.accepted and d.proposed_step <= options.minimum_step for d in result.step_diagnostics)
        assert pinned <= 2, pinned
    assert crossed >= 2


def test_snell_discriminant_within_tolerance_is_the_tir_event_of_the_continuation_problem():
    """Between a valid pose and one past the exit critical angle, bisect to ``0 < exit_snell <= tolerance``:
    :func:`path_domain` still calls it valid, the continuation problem's evaluator already the ``tir_boundary`` event."""
    faces = (3, 5, 6, 7)
    rotations = haar_rotations(40_000, np.random.default_rng(5))
    batch = path_domain_batch(rotations, faces, INCIDENT, INDEX)
    past = (
        (batch.margins["entry_incidence_cosine"] > 0)
        & (batch.margins["internal_1_incidence_cosine"] > 0)
        & (batch.margins["internal_2_incidence_cosine"] > 0)
        & (batch.margins["exit_incidence_cosine"] > 0)
        & (batch.margins["exit_snell_discriminant"] <= 0)
    )
    inside = rotations[np.flatnonzero(batch.valid)[0]]
    outside = rotations[np.flatnonzero(past)[0]]
    problem = path_problem(jnp.asarray(inside), faces, jnp.asarray(INCIDENT), refractive_index=jnp.asarray(INDEX))
    step = np.asarray(log(jnp.asarray(outside @ inside.T)))

    def along(t: float) -> np.ndarray:
        return np.asarray(exp(jnp.asarray(t * step))) @ inside

    low, high = 0.0, 1.0
    for _ in range(200):
        middle = 0.5 * (low + high)
        check = path_domain(along(middle), faces, INCIDENT, INDEX)
        if check.valid and check.margins["exit_snell_discriminant"] <= SNELL_EVENT_TOLERANCE:
            break
        low, high = (middle, high) if check.valid else (low, middle)
    else:
        pytest.fail("no pose within the tolerance on the segment")
    pose = along(middle)
    evaluation = problem.domain_and_event_evaluator(jnp.asarray(pose))
    assert not evaluation.valid and evaluation.event.kind == TerminationReason.TIR_BOUNDARY
    assert 0.0 < evaluation.event.margin <= SNELL_EVENT_TOLERANCE
    assert problem.domain_and_event_evaluator(jnp.asarray(inside)).valid


def test_a60_10_member_arcs_end_on_named_events_at_both_ends():
    """``3-5-6-7`` lives where the face-5 reflection is partial (its exit cosine is face 5's), so every
    fiber is an arc whose ends meet the exit critical angle tangentially; both directions end on an event."""
    faces = (3, 5, 6, 7)
    rotations = _valid_samples(faces, 20_000, seed=3)
    options = ContinuationOptions()
    for rotation in rotations[:4]:
        problem = path_problem(jnp.asarray(rotation), faces, jnp.asarray(INCIDENT), refractive_index=jnp.asarray(INDEX))
        for sign in (1, -1):
            result = trace_fiber(problem, replace(options, initial_tangent_sign=sign))
            assert result.reason in ARC_EVENTS, result.reason


def test_face_sequence_helpers():
    assert normalize_faces([3, 1, 2, 5]) == (3, 1, 2, 5)
    assert path_id_of((3, 1, 2, 5)) == "3-1-2-5" and faces_of_path_id("3-1-2-5") == (3, 1, 2, 5)
    assert faces_of_path_id(path_id_of(PATH_3_5_FACES)) == PATH_3_5_FACES
    for bad in [(3,), (3, 9), (0, 5)]:
        with pytest.raises(ValueError):
            normalize_faces(bad)
    with pytest.raises(ValueError, match="path_id"):
        faces_of_path_id("3-x")
    with pytest.raises(ValueError):
        faces_of_path_id("3")
    with pytest.raises(ValueError):
        path_domain(np.eye(3), (3,), INCIDENT, INDEX)
