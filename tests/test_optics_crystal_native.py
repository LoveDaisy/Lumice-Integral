"""The single-path layer reads its face normals from the crystal (task ``optics-reads-crystal``).

Every consumer that holds a crystal must hand it down to the normal lookup: a crystal whose face 4 is
twisted by 10 degrees about the c axis (explore ``ch12-anchor-probes`` experiment #3, where the output
used to stay bit-identical) must change every quantity that depends on face 4.  On the regular prism, of
any aspect ratio, the crystal path agrees with an independent literal normal table to 1e-12.  The
reduced cluster (``path_class``) fails fast on a crystal whose ``G_true`` is smaller than ``D6h``.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from lumice_integral import focusing, optics, path_class, weights
from lumice_integral.dp_field import DPField
from lumice_integral.dp_field import field as F
from lumice_integral.dp_field.boundary import identical_margins, walk_boundary
from lumice_integral.geometry import HexPrism, Polyhedron, Pyramid
from lumice_integral.pose_density import build_pose_density
from lumice_integral.s2_store import fibonacci_sphere
from lumice_integral.symmetry.crystal_group import true_symmetry_group
from lumice_integral.symmetry.reflection_group import key as rg_key

from _geometry_oracles import NORMALS, reflect, refract

N = 1.31
INCIDENT = np.array([-0.3, 0.55, -0.78]) / np.linalg.norm([-0.3, 0.55, -0.78])
LATTICE_N = 5000


def twisted_face4(twist_deg: float = 10.0) -> Polyhedron:
    """``HexPrism()`` with the four corners of face 4 turned ``twist_deg`` about the c axis (ch12 probe H7).

    Face 4's normal moves from 60 to 70 degrees azimuth; faces 3 and 5, which share those corners, tilt too.
    """
    base = HexPrism()
    v = base.vertices.copy()
    t = np.deg2rad(twist_deg)
    turn = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    for i in (1, 2, 7, 8):
        v[i, :2] = turn @ v[i, :2]
    return Polyhedron(v, base.faces)


_H = np.sqrt(3.0) / 2.0
NORMALS_EXACT = {1: [0.0, 0.0, 1.0], 2: [0.0, 0.0, -1.0], 3: [1.0, 0.0, 0.0], 4: [0.5, _H, 0.0],
                 5: [-0.5, _H, 0.0], 6: [-1.0, 0.0, 0.0], 7: [-0.5, -_H, 0.0], 8: [0.5, -_H, 0.0]}
IDEAL = HexPrism()
TWISTED = twisted_face4()


def test_the_twisted_fixture_moves_face_4():
    azimuth = np.degrees(np.arctan2(*TWISTED.normal(TWISTED.face(4))[1::-1]))
    assert azimuth == pytest.approx(70.0, abs=1e-12)


# ---- optics ---------------------------------------------------------------------------------------------


def _oracle_direction(rotation: np.ndarray, faces: tuple[int, ...], incident: np.ndarray) -> np.ndarray:
    normals = [rotation @ NORMALS[face] for face in faces]
    direction = refract(incident, normals[0], 1.0, N)
    for normal in normals[1:-1]:
        direction = reflect(direction, normal)
    return refract(direction, -normals[-1], N, 1.0)


@pytest.mark.parametrize("crystal", [HexPrism(), HexPrism(1.0, 2.7), HexPrism.from_ratio(0.3)])
@pytest.mark.parametrize("faces", [(3, 5), (4, 6), (3, 1, 2, 5), (1, 3, 2)])
def test_regular_prism_of_any_aspect_matches_the_literal_normal_table(crystal, faces):
    """The literal ``1, 1/2, sqrt(3)/2`` table (``_geometry_oracles``) is independent of the code under test."""
    rotations = np.asarray([np.eye(3)] + [np.asarray(r) for r in _haar(4000)])
    check = optics.path_domain_batch(rotations, faces, INCIDENT, N, crystal=crystal)
    rows = np.flatnonzero(check.valid)[:40]
    assert len(rows) >= 10
    for rotation in rotations[rows]:
        expected = _oracle_direction(rotation, faces, INCIDENT)
        actual = optics.path_direction(jnp.asarray(rotation), faces, jnp.asarray(INCIDENT), N, crystal=crystal)
        np.testing.assert_allclose(np.asarray(actual.direction), expected, rtol=0.0, atol=1e-12)
        scalar = optics.path_domain(rotation, faces, INCIDENT, N, crystal=crystal)
        default = optics.path_domain(rotation, faces, INCIDENT, N)
        assert scalar.valid
        for name, value in scalar.margins.items():
            assert value == pytest.approx(default.margins[name], abs=1e-12)
    np.testing.assert_allclose(check.direction[rows], optics.path_domain_batch(rotations, faces, INCIDENT, N).direction[rows],
                               rtol=0.0, atol=1e-12)


def _haar(count: int) -> np.ndarray:
    from lumice_integral.so3 import haar_rotations

    return haar_rotations(count, np.random.default_rng(5))


def _valid_on_both(faces: tuple[int, ...]) -> np.ndarray:
    rotations = _haar(4000)
    both = (optics.path_domain_batch(rotations, faces, INCIDENT, N, crystal=IDEAL).valid
            & optics.path_domain_batch(rotations, faces, INCIDENT, N, crystal=TWISTED).valid)
    return rotations[np.flatnonzero(both)[0]]


def test_star_faces_read_the_exact_closed_form_direction():
    """Faces on their star direction give the literal table bit for bit (the ch06 / strip / S^2 store pins)."""
    for crystal in (HexPrism(), HexPrism(1.0, 2.7), HexPrism(1.0, 1.0, (1.0, 1.3, 0.7, 1.9, 1.1, 0.4)),
                    Pyramid(a=1, h=1, tip_ratio=0.5)):
        faces = tuple(f for f in range(1, 9) if f in {g.number for g in crystal.faces})
        normals = optics.face_normals(crystal, faces)
        for face, normal in zip(faces, normals):
            np.testing.assert_array_equal(normal, NORMALS_EXACT[face])
            np.testing.assert_allclose(normal, crystal.normal(crystal.face(face)), rtol=0.0, atol=1e-15)
    # off the star by the twist: face 4 keeps its own normal, faces 6-8 (untouched) are exact
    twisted = optics.face_normals(TWISTED, (4, 6, 7, 8))
    np.testing.assert_array_equal(twisted[0], TWISTED.normal(TWISTED.face(4)))
    for normal, face in zip(twisted[1:], (6, 7, 8)):
        np.testing.assert_array_equal(normal, NORMALS_EXACT[face])


def test_every_single_path_function_reads_the_crystal():
    faces = (4, 6)
    rotation = _valid_on_both(faces)
    ideal = optics.path_direction(jnp.asarray(rotation), faces, jnp.asarray(INCIDENT), N, crystal=IDEAL)
    twisted = optics.path_direction(jnp.asarray(rotation), faces, jnp.asarray(INCIDENT), N, crystal=TWISTED)
    assert not np.allclose(np.asarray(ideal.direction), np.asarray(twisted.direction), atol=1e-6)

    ideal_domain = optics.path_domain(rotation, faces, INCIDENT, N, crystal=IDEAL)
    twisted_domain = optics.path_domain(rotation, faces, INCIDENT, N, crystal=TWISTED)
    assert ideal_domain.valid and twisted_domain.valid
    assert abs(ideal_domain.margins["entry_incidence_cosine"] - twisted_domain.margins["entry_incidence_cosine"]) > 1e-3

    rotations = rotation[None]
    ideal_batch = optics.path_domain_batch(rotations, faces, INCIDENT, N, crystal=IDEAL)
    twisted_batch = optics.path_domain_batch(rotations, faces, INCIDENT, N, crystal=TWISTED)
    assert not np.allclose(ideal_batch.direction, twisted_batch.direction, atol=1e-6)
    np.testing.assert_allclose(twisted_batch.direction[0], np.asarray(twisted.direction), rtol=0.0, atol=1e-15)

    ideal_t = optics.fresnel_transmission_path(rotation, faces, INCIDENT, N, crystal=IDEAL)
    twisted_t = optics.fresnel_transmission_path(rotation, faces, INCIDENT, N, crystal=TWISTED)
    assert abs(ideal_t - twisted_t) > 1e-6
    assert optics.fresnel_transmission_path_batch(rotations, faces, INCIDENT, N, crystal=TWISTED)[0] == pytest.approx(twisted_t, abs=1e-15)

    problem = optics.path_problem(
        jnp.asarray(rotation), faces, jnp.asarray(INCIDENT), refractive_index=jnp.asarray(N), crystal=TWISTED
    )
    np.testing.assert_allclose(np.asarray(problem.direction_evaluator(jnp.asarray(rotation))), np.asarray(twisted.direction), atol=1e-15)
    assert problem.domain_and_event_evaluator(jnp.asarray(rotation)).margins["entry_incidence_cosine"] == pytest.approx(
        twisted_domain.margins["entry_incidence_cosine"], abs=1e-15
    )


def test_a_face_the_crystal_does_not_have_is_rejected():
    triangle = HexPrism(1.0, 1.0, (1, 2, 1, 2, 1, 2))  # side faces 3, 5, 7 only
    assert 4 not in {f.number for f in triangle.faces}
    optics.normalize_faces((3, 5), triangle)
    with pytest.raises(ValueError, match=r"unknown hexagonal-prism face numbers \[4\]"):
        optics.normalize_faces((4, 5), triangle)
    with pytest.raises(ValueError, match="unknown"):
        optics.path_direction(jnp.eye(3), (3, 4), jnp.asarray(INCIDENT), N, crystal=triangle)
    with pytest.raises(ValueError, match="unknown"):
        optics.path_domain(np.eye(3), (3, 4), INCIDENT, N, crystal=triangle)
    with pytest.raises(ValueError, match="unknown"):
        DPField.build(triangle, (4, 6), N)
    with pytest.raises(ValueError, match="unknown Pyramid face numbers"):
        optics.normalize_faces((3, 9), Pyramid(a=1, h=1, tip_ratio=0.5))


def test_weights_read_the_crystal():
    faces = (4, 6)
    rotation = _valid_on_both(faces)
    kwargs = dict(faces=faces, incident_direction=INCIDENT, refractive_index=N)
    ideal = weights.fresnel_transmission_weight(rotation, crystal=IDEAL, **kwargs)
    assert weights.fresnel_transmission_weight(rotation, **kwargs) == ideal
    assert abs(weights.fresnel_transmission_weight(rotation, crystal=TWISTED, **kwargs) - ideal) > 1e-6
    # path_validity: a pose valid on the ideal prism, invalid once face 4 turned away (or the reverse)
    rotations = _haar(4000)
    ideal_valid = optics.path_domain_batch(rotations, faces, INCIDENT, N, crystal=IDEAL).valid
    twisted_valid = optics.path_domain_batch(rotations, faces, INCIDENT, N, crystal=TWISTED).valid
    differ = np.flatnonzero(ideal_valid != twisted_valid)
    assert len(differ) > 0
    pick = rotations[differ[0]]
    # the evaluators need a HexPrism (they report its edge a): a fresh one carrying the twisted vertices
    twisted_prism = HexPrism()
    twisted_prism.vertices = TWISTED.vertices.copy()
    for crystal, expected in ((IDEAL, ideal_valid[differ[0]]), (twisted_prism, twisted_valid[differ[0]])):
        evaluators = weights.build_path_weight_evaluators(
            faces=faces, incident_direction=INCIDENT, refractive_index=N, crystal=crystal,
            pose_density=build_pose_density("random"),
        )
        batch = evaluators["path_validity"].evaluate_batch(pick[None])[0]
        # entry_measure > 0 also gates; the domain part must follow the crystal
        assert batch <= float(expected)
        assert (evaluators["fresnel_transmission"].evaluate_batch(pick[None])[0] > 0.0) == bool(expected)


# ---- D_P field ------------------------------------------------------------------------------------------


def test_field_kernels_read_the_crystal():
    faces = (4, 6)
    u = fibonacci_sphere(LATTICE_N)
    inside = F.valid_batch(u, faces, N, crystal=IDEAL) & F.valid_batch(u, faces, N, crystal=TWISTED)
    points = u[inside][:64]
    assert len(points) == 64
    for fn in (F.d_p_batch, F.gradient_batch, F.margins_batch, F.validity_margins_batch):
        assert not np.allclose(fn(points, faces, N, crystal=IDEAL), fn(points, faces, N, crystal=TWISTED), atol=1e-6), fn
        np.testing.assert_array_equal(fn(points, faces, N, crystal=IDEAL), fn(points, faces, N))
    h_ideal, _ = F.hessian_tangent_batch(points, faces, N, crystal=IDEAL)
    h_twisted, _ = F.hessian_tangent_batch(points, faces, N, crystal=TWISTED)
    assert not np.allclose(h_ideal, h_twisted, atol=1e-6)
    # the jitted kernels agree with the eager optics on the crystal (no stale compiled normals)
    for crystal in (IDEAL, TWISTED):
        eager = np.array([
            np.arctan2(np.linalg.norm(np.cross(d, -p)), d @ -p)
            for p, d in zip(points, (np.asarray(optics.path_direction(jnp.eye(3), faces, jnp.asarray(-p), N, crystal=crystal).direction)
                                     for p in points))
        ])
        np.testing.assert_allclose(F.d_p_batch(points, faces, N, crystal=crystal), eager, rtol=0.0, atol=1e-12)


def test_location_and_fold_set_read_the_crystal(monkeypatch):
    faces = (4, 6)
    u = fibonacci_sphere(LATTICE_N)
    differ = np.flatnonzero(F.valid_batch(u, faces, N, crystal=IDEAL) != F.valid_batch(u, faces, N, crystal=TWISTED))
    point = u[differ[0]]
    assert F.location(point, faces, N, crystal=IDEAL) != F.location(point, faces, N, crystal=TWISTED)
    # the slab path 1-4-2 (both basal faces, one reflection off face 4): the fold set is probed on the crystal
    seen: list[object] = []
    original = F.validity_margins_batch

    def spy(u, faces, index, *, crystal=None):
        seen.append(crystal)
        return original(u, faces, index, crystal=crystal)

    monkeypatch.setattr(F, "validity_margins_batch", spy)
    screen = F.fold_screen(TWISTED, (1, 4, 2))
    assert screen.degenerate
    F.degenerate_fold_set(screen, (1, 4, 2), N, crystal=TWISTED)
    assert seen and all(c is TWISTED for c in seen)
    np.testing.assert_allclose(screen.axis, TWISTED.normal(TWISTED.face(4)), atol=1e-12)


def test_critical_points_read_the_crystal():
    faces = (4, 6)
    ideal = F.lattice_newton_critical_points(faces, N, lattice_n=LATTICE_N, crystal=IDEAL)
    twisted = F.lattice_newton_critical_points(faces, N, lattice_n=LATTICE_N, crystal=TWISTED)
    assert [p.kind for p in ideal] == [p.kind for p in twisted] == ["minimum"]
    assert abs(ideal[0].value - twisted[0].value) > 1e-3
    screen = F.fold_screen(TWISTED, faces)
    points, fold = F.interior_critical_points(screen, faces, N, lattice_n=LATTICE_N, crystal=TWISTED)
    assert fold is None and points[0].value == pytest.approx(twisted[0].value, abs=1e-12)


def test_dp_field_reads_its_crystal_end_to_end():
    ideal = DPField.build(IDEAL, (4, 6), N, lattice_n=LATTICE_N)
    twisted = DPField.build(TWISTED, (4, 6), N, lattice_n=LATTICE_N)
    np.testing.assert_allclose(np.asarray(twisted.normals[0]), TWISTED.normal(TWISTED.face(4)), atol=0.0)
    # the interior chain (_interior -> interior_critical_points -> lattice Newton)
    assert abs(ideal.interior_critical_points[0].value - twisted.interior_critical_points[0].value) > 1e-3
    # the certificate chain (domain_topology, boundary walk, interval_partition)
    ideal_partition = ideal.interval_partition()
    twisted_partition = twisted.interval_partition()
    assert twisted.domain_topology.is_disk
    assert len(ideal_partition) != len(twisted_partition) or not np.allclose(
        [iv.upper for iv in ideal_partition], [iv.upper for iv in twisted_partition], atol=1e-6
    )
    assert twisted_partition[0].lower == pytest.approx(twisted.interior_critical_points[0].value, abs=1e-9)


def test_boundary_walk_reads_the_crystal():
    ideal = walk_boundary(IDEAL, (4, 6), N, lattice_n=LATTICE_N)
    twisted = walk_boundary(TWISTED, (4, 6), N, lattice_n=LATTICE_N)
    assert not np.allclose(
        sorted(c.value for c in ideal.corners), sorted(c.value for c in twisted.corners), atol=1e-6
    ) or len(ideal.corners) != len(twisted.corners)


def test_margin_identities_are_checked_on_the_crystal():
    # 3-5-6-7-3: reflections 5, 6, 7 step by +60 deg on the prism, so the third incidence cosine is the first
    assert identical_margins((3, 5, 6, 7, 3), IDEAL) == identical_margins((3, 5, 6, 7, 3))
    assert identical_margins((3, 5, 6, 7, 3), IDEAL)
    # the twist moves face 5 to 125 deg: steps 55 and 60, no identity
    assert identical_margins((3, 5, 6, 7, 3), TWISTED) == {}
    # 3-4-5-6-3 on the twisted crystal: faces 4, 5, 6 at 70, 125, 180 deg, equal steps keep the identity
    assert identical_margins((3, 4, 5, 6, 3), TWISTED) == identical_margins((3, 4, 5, 6, 3), IDEAL)


# ---- pyramid ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("faces", [(13, 15, 26, 28), (13, 5, 26, 28), (13, 24, 26)])
def test_pyramid_paths_classify(faces):
    """The pyramidal faces 13-28 reach the focusing label (its reading is the next task's)."""
    label = focusing.classify(Pyramid(a=1, h=1, tip_ratio=0.5), faces, build_pose_density("random"), N)
    assert isinstance(label, focusing.FocusingClassification)
    assert label.path == "-".join(map(str, faces))


ASYMMETRIC_PYRAMID = Pyramid.from_lumice(0.5, 0.25, 0.6, (1, 0, 1), (2, 0, 3), face_distance=(1, 1.1, 0.9, 1, 1.2, 0.95))


def test_asymmetric_pyramid_cone_normals_are_read_from_the_crystal():
    """Faces 13-28 of an up/down-asymmetric, irregular-section cone: ``optics.face_normals`` is the crystal's own normal."""
    faces = tuple(n for n in (*range(13, 19), *range(23, 29)) if n in {f.number for f in ASYMMETRIC_PYRAMID.faces})
    assert len(faces) == 12
    for face, normal in zip(faces, optics.face_normals(ASYMMETRIC_PYRAMID, faces)):
        np.testing.assert_allclose(normal, ASYMMETRIC_PYRAMID.normal(ASYMMETRIC_PYRAMID.face(face)), rtol=0.0, atol=1e-15)
    upper, lower = optics.face_normals(ASYMMETRIC_PYRAMID, (13, 23))
    assert upper[2] > 0 > lower[2] and not np.isclose(upper[2], -lower[2])   # the two cones really differ


def test_an_offfamily_path_classifies_on_an_asymmetric_pyramid():
    """Runs through and returns finite values; the physical reading is not asserted here."""
    label = focusing.classify(ASYMMETRIC_PYRAMID, (13, 15, 26, 28), build_pose_density("random"), N)
    assert isinstance(label, focusing.FocusingClassification) and label.path == "13-15-26-28"
    assert all(np.isfinite(o.value) for o in label.onsets)


# explore-ch9-offfamily-focusing (29.3): the three off-family paths classified and checked against the writing
# series' own MC renders (each onset within 1 deg); probes/classify_output.json there, recomputed bit for bit
# at 6a82024. Values in degrees at abs=1e-4, measure_limit at abs=1e-3: 4-6 orders above the Newton / bisection
# tolerances inside dp_field (~1e-10 / ~1e-12 rad), so a platform's float64 tail cannot trip them, a moved
# critical point can. Per path: mechanism, the interior onset (source, profile, value, measure_limit) or None,
# the number of boundary onsets, and the boundary onsets themselves where they are the whole story (13-24-26).
PYRAMID_OFFFAMILY = [
    pytest.param((13, 15, 26, 28), "none", ("interior_maximum", "finite_jump", 177.29401194396095,
                                            0.6828958423685803), 2, None, id="13-15-26-28"),
    pytest.param((13, 5, 26, 28), "jacobian", ("interior_saddle", "log_divergence", 136.35814215526733, None),
                 9, None, id="13-5-26-28"),
    pytest.param((13, 24, 26), "none", None, 2,
                 [("corner", 6.99592141362214e-07), ("boundary_extremum", 131.30216743582776)], id="13-24-26"),
]


@pytest.mark.parametrize("faces, mechanism, interior, boundary_count, boundary", PYRAMID_OFFFAMILY)
def test_pyramid_offfamily_focusing_mechanism_regression(faces, mechanism, interior, boundary_count, boundary):
    """Interior maximum -> finite_jump edge, interior saddle -> log_divergence, no interior point -> diffuse."""
    label = focusing.classify(Pyramid(a=1, h=1, tip_ratio=0.5), faces, build_pose_density("random"), N)
    assert label.mechanism == mechanism
    assert label.jacobian_focusing is (mechanism == "jacobian")
    inner = [o for o in label.onsets if o.location == "interior"]
    outer = [o for o in label.onsets if o.location == "boundary"]
    if interior is None:
        assert inner == []
    else:
        source, profile, value_deg, measure_limit = interior
        (onset,) = inner
        assert (onset.source, onset.profile) == (source, profile)
        assert onset.jacobian_focusing is (profile == "log_divergence")
        assert np.degrees(onset.value) == pytest.approx(value_deg, abs=1e-4)
        if measure_limit is None:
            assert onset.measure_limit is None
        else:
            assert onset.measure_limit == pytest.approx(measure_limit, abs=1e-3)
    assert len(outer) == boundary_count
    assert all(o.profile == "boundary_onset" and not o.jacobian_focusing for o in outer)
    if boundary is not None:
        assert [o.source for o in outer] == [source for source, _ in boundary]
        assert [np.degrees(o.value) for o in outer] == [pytest.approx(v, abs=1e-4) for _, v in boundary]


# ---- reduced cluster on G_true --------------------------------------------------------------------------

# design.md 4: on HexPrism(1, 1, (2, 1, 1, 2, 1, 1)), |G_true| = 8, the D6h orbit of a path has more members than
# its G_true orbit, and only the G_true images are the same physics; the reduced cluster uses G_true by default.
# face_distance 2 cuts faces 3 and 6 off that prism, so its fixture path is 4-8.
LOW_SYMMETRY = [HexPrism(1.0, 1.0, (2, 1, 1, 2, 1, 1)), HexPrism(1.0, 1.0, (1.0, 1.3, 0.7, 1.9, 1.1, 0.4))]
LOW_SYMMETRY_ORBITS = [
    ((4, 8), {(4, 8), (5, 7), (7, 5), (8, 4)}),
    ((3, 5), {(3, 5)}),
]


def test_the_d6h_orbit_is_not_the_crystal_orbit_on_a_low_symmetry_prism():
    """design.md 4 / explore H4: ``3-5`` has 12 ``D6h`` images but one ``G_true`` image on the ``|G| = 2`` prism."""
    assert [len(true_symmetry_group(c)) for c in LOW_SYMMETRY] == [8, 2]
    generic = LOW_SYMMETRY[1]
    assert len(path_class.g_true_orbit((3, 5))) == 12
    normals = {f.number: generic.normal(f) for f in generic.faces}
    own_orbit = {path_class._symmetry_image_of_faces(g, (3, 5), normals) for g in true_symmetry_group(generic)}
    assert own_orbit == {(3, 5)}


@pytest.mark.parametrize("crystal, orbit", list(zip(LOW_SYMMETRY, LOW_SYMMETRY_ORBITS)))
def test_reduced_cluster_uses_g_true_by_default_below_d6h(crystal, orbit):
    """The explore H4 counterexample turned into agreement: the default orbit is the ``G_true`` orbit (the
    ``|G| = 2`` prism's ``3-5`` class has one member, not the 12 of ``D6h``), and the class transports stay in it."""
    faces, members = orbit
    assert path_class.g_true_orbit(faces, crystal) == frozenset(members)
    key = path_class.phi_key(crystal, faces)
    assert isinstance(key, tuple) and len(key) == 3 and all(isinstance(k, int) for k in key)
    built = path_class.build_path_class(crystal, faces)
    assert set(built.members) == members
    own_keys = {rg_key(g) for g in true_symmetry_group(crystal)}
    transports = path_class.path_class_symmetry(built, crystal)
    assert set(transports) == members
    assert all(rg_key(g) in own_keys for g in transports.values())
    # a class built on the regular prism is not one G_true orbit of this crystal
    ideal_class = path_class.build_path_class(HexPrism(), (4, 8))
    with pytest.raises(RuntimeError, match="not a D6h image"):
        path_class.path_class_symmetry(ideal_class, crystal)
    # explicit candidates must lie in the crystal's own group
    own = true_symmetry_group(crystal)
    representative_only = path_class.PathClass(
        ideal_class.representative, (ideal_class.representative,), ideal_class.wedge_deg, ideal_class.halo_map_rank
    )
    assert set(path_class.path_class_symmetry(representative_only, crystal, symmetry_elements=own)) == {(4, 8)}
    foreign = [g for g in path_class.hexprism_symmetry_matrices() if not any(np.allclose(g, h) for h in own)]
    with pytest.raises(ValueError, match="G_true"):
        path_class.path_class_symmetry(representative_only, crystal, symmetry_elements=foreign[:1])


def test_the_reduced_cluster_rejects_a_face_the_crystal_does_not_have():
    crystal = LOW_SYMMETRY[0]
    assert 3 not in {f.number for f in crystal.faces}
    with pytest.raises(ValueError, match="do not exist"):
        path_class.g_true_orbit((3, 5), crystal)
    with pytest.raises(ValueError, match="do not exist"):
        path_class.phi_key(crystal, (3, 5))
