"""``optics.path_direction`` / ``path_domain`` on arbitrary face sequences vs the ray-cast oracle.

The oracle side is ``tests/_geometry_oracles.py`` (closed-form normals, vector
Snell ``refract`` and mirror ``reflect``; it shares no code with ``optics``).
Three paths are checked pose by pose on Haar samples that pass the smooth
domain: ``3-5`` (the pre-generalisation baseline), ``3-7`` (the other face
pair of the same class) and ``3-1-2-5`` (two basal total internal reflections,
the ``M = I`` 60-deg wedge of the issue).  The event classification of
``path_domain`` is checked on analytically constructed boundary poses.
"""

from __future__ import annotations

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
)
from lumice_integral.so3 import haar_rotations

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
        if INDEX * INDEX * (1.0 - cosine * cosine) <= 1.0:
            raise ValueError("internal reflection is not total")
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
    """A pose whose internal ray reaches face 1 but is not totally reflected there, and one whose
    internal ray misses face 1 (goes to face 2 instead): each stops at its own margin."""
    faces = (3, 1, 2, 5)
    rotations = haar_rotations(20_000, np.random.default_rng(3))
    batch = path_domain_batch(rotations, faces, INCIDENT, INDEX)
    margins = batch.margins
    entry_ok = (margins["entry_incidence_cosine"] > 0) & (margins["entry_snell_discriminant"] > 0)
    not_total = entry_ok & (margins["internal_1_incidence_cosine"] > 0) & (margins["internal_1_tir_discriminant"] <= 0)
    misses = entry_ok & (margins["internal_1_incidence_cosine"] <= 0)
    assert not_total.any() and misses.any()
    check = path_domain(rotations[np.flatnonzero(not_total)[0]], faces, INCIDENT, INDEX)
    assert check.event_kind == "tir_boundary" and "reflection 1 at face 1" in check.message
    assert "exit_incidence_cosine" not in check.margins
    check = path_domain(rotations[np.flatnonzero(misses)[0]], faces, INCIDENT, INDEX)
    assert check.event_kind == "path_infeasible" and "face 1" in check.message
    assert "internal_1_tir_discriminant" in check.margins and "internal_2_incidence_cosine" not in check.margins


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


def test_fresnel_of_a_reflecting_path_only_counts_the_two_refracting_interfaces():
    rotations = _valid_samples((3, 1, 2, 5), 50_000)[:50]
    values = fresnel_transmission_path_batch(rotations, (3, 1, 2, 5), INCIDENT, INDEX)
    same_pose_3_5 = fresnel_transmission_path_batch(rotations, (3, 5), INCIDENT, INDEX)
    # Same entry and exit incidence cosines at the same pose (M = I), so the products coincide.
    np.testing.assert_allclose(values, same_pose_3_5, rtol=0.0, atol=1e-12)
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
    assert evaluation.valid and set(evaluation.margins) == set(domain_margin_names((3, 1, 2, 5)))


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
