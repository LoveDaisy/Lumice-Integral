"""``focusing``: Jacobian focusing vs dimension collapse as an explicit label (``docs/phase2.md`` section 10)."""

from __future__ import annotations

import numpy as np
import pytest

from lumice_integral import focusing
from lumice_integral.canonical_scene import CANONICAL_REFRACTIVE_INDEX, canonical_crystal
from lumice_integral.dp_field import DPField
from lumice_integral.geometry import halo_map_rank
from lumice_integral.pose_density import ZenithGaussianPoseDensity, build_pose_density
from lumice_integral.symmetry import reflection_group

INDEX = CANONICAL_REFRACTIVE_INDEX
RANDOM = build_pose_density("random")
COLUMN = build_pose_density("column", zenith_std_deg=0.5)
PLATE = build_pose_density("plate", zenith_std_deg=0.5)
PARRY = build_pose_density("parry", zenith_std_deg=0.5, roll_std_deg=1.0)
LOWITZ = build_pose_density("lowitz", zenith_std_deg=0.5, roll_std_deg=1.0)
SLABS = ((1, 3, 2), (3, 5, 6, 7, 3), (3, 1, 6), (1, 3, 5, 2), (1, 2, 1))


@pytest.fixture(scope="module")
def labels() -> dict[tuple[int, ...], focusing.FocusingClassification]:
    return {faces: focusing.classify(canonical_crystal(), faces, RANDOM, INDEX) for faces in ((3, 5), *SLABS)}


def test_confined_dimensions_follow_the_density_type() -> None:
    assert focusing.confined_dimensions(RANDOM) == (0, ())
    assert focusing.confined_dimensions(COLUMN) == (1, (pytest.approx(np.radians(0.5)),))
    assert focusing.confined_dimensions(PLATE)[0] == 1
    assert focusing.confined_dimensions(PARRY) == (2, (pytest.approx(np.radians(0.5)), pytest.approx(np.radians(1.0))))
    assert focusing.confined_dimensions(LOWITZ)[0] == 2
    # a wide factor still confines its dimension: no width threshold
    assert focusing.confined_dimensions(build_pose_density("column", zenith_std_deg=30.0))[0] == 1
    with pytest.raises(TypeError):
        focusing.confined_dimensions(object())


def test_rank_zero_path_is_a_point_mass_consistent_with_halo_map_rank() -> None:
    """``3-6`` (fold matrix I, wedge 0): no field, label ``point_mass`` whatever the density (review, a56)."""
    crystal = canonical_crystal()
    assert halo_map_rank(crystal, (3, 6)) == 0
    with pytest.raises(ValueError):
        DPField.build(crystal, (3, 6), INDEX)
    for density in (RANDOM, COLUMN, PARRY):
        label = focusing.classify(crystal, (3, 6), density, INDEX)
        assert label.mechanism == "point_mass"
        assert label.onsets == () and label.gradient_norm_range is None
        assert not label.jacobian_focusing and not label.dimension_collapse
    for faces in ((3, 5), *SLABS):
        assert halo_map_rank(crystal, faces) == 2
        assert focusing.classify(crystal, faces, RANDOM, INDEX).mechanism != "point_mass"


def test_interior_onset_profiles() -> None:
    minimum = focusing.interior_onset(0.4, "minimum", np.array([0.25, 1.0]), 0.0)
    assert minimum.profile == "finite_jump" and not minimum.jacobian_focusing
    assert minimum.measure_limit == pytest.approx(2.0 * np.pi / np.sqrt(0.25))
    assert focusing.interior_onset(0.4, "maximum", np.array([-0.25, -1.0]), 0.0).measure_limit == pytest.approx(4.0 * np.pi)
    saddle = focusing.interior_onset(0.4, "saddle", np.array([-0.25, 1.0]), 0.0)
    assert saddle.profile == "log_divergence" and saddle.jacobian_focusing
    degenerate = focusing.interior_onset(0.4, "degenerate", np.array([0.0, 1.0]), 0.0)
    assert degenerate.profile == "degenerate" and degenerate.jacobian_focusing


def test_mechanism_covers_the_four_states() -> None:
    saddle = focusing.interior_onset(0.4, "saddle", np.array([-1.0, 1.0]), 0.0)
    jump = focusing.interior_onset(0.4, "minimum", np.array([1.0, 1.0]), 0.0)
    states = {
        (False, 0): "none",
        (True, 0): "jacobian",
        (False, 1): "dimension_collapse",
        (True, 2): "jacobian+dimension_collapse",
    }
    for (jacobian, dims), expected in states.items():
        label = focusing.FocusingClassification("x", 2, (saddle if jacobian else jump,), (0.0, 1.0), dims, (0.01,) * dims)
        assert label.mechanism == expected
        assert label.as_json()["mechanism"] == expected


def test_3_5_minimum_is_a_finite_jump(labels) -> None:
    """The 22 deg minimum: non-degenerate, ``2 pi / sqrt(det H)`` finite; no critical value of 3-5 focuses."""
    label = labels[(3, 5)]
    (minimum,) = [o for o in label.onsets if o.source == "interior_minimum"]
    assert minimum.profile == "finite_jump"
    assert np.degrees(minimum.value) == pytest.approx(21.8393, abs=1e-4)
    assert minimum.measure_limit == pytest.approx(11.011045, rel=1e-6)
    assert not label.jacobian_focusing and label.mechanism == "none"
    assert focusing.classify(canonical_crystal(), (3, 5), COLUMN, INDEX).mechanism == "dimension_collapse"
    assert focusing.classify(canonical_crystal(), (3, 5), PARRY, INDEX).confined_dimensions == 2


@pytest.mark.parametrize("faces", [faces for faces in SLABS if faces != (1, 3, 5, 2)])
def test_parallel_face_slabs_have_no_jacobian_focusing(labels, faces) -> None:
    """Wedge 0, ``M != I``: cone points, creases and boundary cusps only; their sharp images are ``rho``'s."""
    label = labels[faces]
    assert not label.jacobian_focusing
    assert {o.profile for o in label.onsets} <= {"cone_point", "crease", "boundary_onset"}
    lower, upper = label.gradient_norm_range
    assert lower > 0.5
    if np.linalg.det(DPField.build(canonical_crystal(), faces, INDEX).fold.fold_matrix) < 0.0:
        # a mirror slab: D = 2 arcsin |u . n_M|, |grad D| = 2 everywhere
        assert lower == pytest.approx(2.0, abs=1e-12) and upper == pytest.approx(2.0, abs=1e-12)
    assert focusing.classify(canonical_crystal(), faces, PLATE, INDEX).mechanism == "dimension_collapse"


def test_rotation_slab_focuses_on_its_fold_circle_at_the_boundary(labels) -> None:
    """``1-3-5-2`` (120 deg rotation): the fold circle ``D = 120`` deg (``|grad D| -> 0``) is an arc of ``dU_P``.

    With partial internal reflections ``U_P`` is the triangle of the entry and
    the two internal grazing great circles, and the fold circle is the entry
    one: a one-sided curve of maxima, ``inverse_sqrt_divergence`` in ``delta``
    -- Jacobian focusing at 120 deg under random orientation (the
    ``dp-field-partial-reflection-boundaries`` label; before it the circle
    missed the closure of ``U_P``).
    """
    label = labels[(1, 3, 5, 2)]
    (circle,) = [o for o in label.onsets if o.source == "slab_circle"]
    assert circle.location == "boundary" and circle.profile == "inverse_sqrt_divergence"
    assert np.degrees(circle.value) == pytest.approx(120.0, abs=1e-12)
    assert label.jacobian_focusing and label.mechanism == "jacobian"
    lower, upper = label.gradient_norm_range
    assert lower < 1e-3 and upper == pytest.approx(np.sqrt(3.0), abs=1e-3)
    assert DPField.build(canonical_crystal(), (1, 3, 5, 2), INDEX).degenerate_fold.circle_interior_fraction == 0.0
    assert focusing.classify(canonical_crystal(), (1, 3, 5, 2), PLATE, INDEX).mechanism == "jacobian+dimension_collapse"


def test_slab_axis_and_circle_labels(labels) -> None:
    liljequist = labels[(3, 5, 6, 7, 3)]
    (cone,) = [o for o in liljequist.onsets if o.source == "slab_axis"]
    assert cone.location == "interior" and cone.profile == "cone_point"
    assert np.degrees(cone.value) == pytest.approx(180.0, abs=1e-9) and cone.gradient_norm == pytest.approx(2.0, abs=1e-6)
    (crease,) = [o for o in liljequist.onsets if o.source == "slab_circle"]
    assert crease.profile == "crease" and crease.value == 0.0
    # rotation slab (det +1, 120 deg): cone slope 2 sin 60 deg at the axis; its fold circle (D = 120 deg) is an
    # arc of dU_P (test_rotation_slab_focuses_on_its_fold_circle_at_the_boundary)
    rotation = labels[(1, 3, 5, 2)]
    (axis,) = [o for o in rotation.onsets if o.source == "slab_axis"]
    assert axis.gradient_norm == pytest.approx(np.sqrt(3.0), abs=1e-6)
    assert [o.location for o in rotation.onsets if o.source == "slab_circle"] == ["boundary"]


def test_boundary_records_are_merged(labels) -> None:
    """Mirror-image extrema and multi-margin corners are one record each, with their multiplicity."""
    for label in labels.values():
        keys = [(round(o.value, 6), o.location, o.source, o.profile) for o in label.onsets]
        assert len(keys) == len(set(keys))
    (peak,) = [o for o in labels[(3, 5, 6, 7, 3)].onsets if o.source == "boundary_extremum"]
    assert np.degrees(peak.value) == pytest.approx(98.1607, abs=1e-4) and peak.multiplicity == 2


def test_wavelength_critical_table_regresses_the_explore_authority() -> None:
    """3-5 on a 0.2 plate under a 1 deg plate density: blue-red shifts of explore-spectral-conventions #4 (H4)."""
    from lumice_integral.geometry import HexPrism
    from lumice_integral.spectrum.dispersion import refractive_index

    indices = {f"{nm}nm": refractive_index(float(nm)) for nm in (450, 550, 650)}
    table = focusing.wavelength_critical_table(
        HexPrism.from_ratio(0.2), (3, 5), build_pose_density("plate", zenith_std_deg=1.0), indices)
    assert table.path == "3-5" and table.labels == ("450nm", "550nm", "650nm")
    assert [row.location for row in table.onsets] == ["interior", "boundary", "boundary", "boundary"]
    shifts = [row.displacement_deg for row in table.onsets]
    assert shifts == pytest.approx([0.581, 0.631, 0.604, 0.748], abs=2e-3)
    assert all(0.58 <= s <= 0.75 for s in shifts)
    for row in table.onsets:
        # higher n, larger deviation: the shift is blue minus red
        assert row.values_deg["450nm"] > row.values_deg["550nm"] > row.values_deg["650nm"]
        assert row.displacement_deg == pytest.approx(row.values_deg["450nm"] - row.values_deg["650nm"], abs=1e-12)
    assert table.as_json()["onsets"][0]["values_deg"] == table.onsets[0].values_deg


def _classification(*onsets: focusing.CriticalOnset) -> focusing.FocusingClassification:
    return focusing.FocusingClassification("x", 2, onsets, (0.0, 1.0), 0, ())


def test_wavelength_critical_table_raises_on_onset_count_mismatch() -> None:
    jump = focusing.interior_onset(0.38, "minimum", np.array([1.0, 1.0]), 0.0)
    edge = focusing.CriticalOnset(0.5, "boundary", "corner", "boundary_onset", 1.0)
    with pytest.raises(ValueError, match="onset counts differ.*'blue': 2.*'red': 1"):
        focusing._align_onsets({"blue": _classification(jump, edge), "red": _classification(jump)})


def test_wavelength_critical_table_raises_on_shape_mismatch() -> None:
    jump = focusing.interior_onset(0.38, "minimum", np.array([1.0, 1.0]), 0.0)
    corner = focusing.CriticalOnset(0.5, "boundary", "corner", "boundary_onset", 1.0)
    extremum = focusing.CriticalOnset(0.51, "boundary", "boundary_extremum", "boundary_onset", 1.0)
    with pytest.raises(ValueError, match="onset 1 differs.*'blue'.*'corner'.*'red'.*'boundary_extremum'"):
        focusing._align_onsets({"blue": _classification(jump, corner), "red": _classification(jump, extremum)})
    rows = focusing._align_onsets({"blue": _classification(jump, corner), "red": _classification(jump, corner)})
    assert [row.displacement_deg for row in rows] == [0.0, 0.0]


# ---- family pinned: the sigma -> 0 family inside one level set -------------------------------------
PINNED = ((3, 6, 4, 8), (1, 2, 1), (1, 3, 4, 2))
# 3-5 and 1-3: fold matrix I (commutes with R_z), wedge 60 / 90 deg -- the wedge guard's counterexamples
NOT_PINNED = ((3, 5), (1, 3))


def test_family_pinned_truth_table() -> None:
    crystal = canonical_crystal()
    for faces in PINNED:
        for density in (PLATE, LOWITZ):
            assert focusing.family_pinned(crystal, faces, density)
        for density in (COLUMN, PARRY, RANDOM):
            assert not focusing.family_pinned(crystal, faces, density)
    for faces in NOT_PINNED:
        for density in (PLATE, LOWITZ, COLUMN, PARRY, RANDOM):
            assert not focusing.family_pinned(crystal, faces, density)
    # the c axis held at the other pole is the same family; off a pole the support is no circle about c
    assert focusing.family_pinned(crystal, (3, 6, 4, 8), ZenithGaussianPoseDensity(np.pi, np.radians(0.5)))
    assert not focusing.family_pinned(crystal, (3, 6, 4, 8), build_pose_density("plate", zenith_mean_deg=20.0, zenith_std_deg=0.5))
    # rank 0: M = I and wedge 0 commute trivially, but there is no field; the label is point_mass alone
    assert not focusing.family_pinned(crystal, (3, 6), PLATE)
    label = focusing.classify(crystal, (3, 6), PLATE, INDEX)
    assert label.mechanism == "point_mass" and not label.family_pinned


def test_family_pinned_matches_the_element_table() -> None:
    """Every representative path of G (rank > 0): pinned under a plate density iff its element commutes with R_z."""
    crystal = canonical_crystal()
    checked = 0
    for element in reflection_group.ELEMENTS:
        for faces in element.representative_paths:
            if len(faces) == 1 or halo_map_rank(crystal, faces) == 0:
                continue
            assert focusing.family_pinned(crystal, faces, PLATE) == element.commutes_with_rz, (element.number, faces)
            checked += 1
    assert checked == 24


def test_classify_carries_family_pinned() -> None:
    crystal = canonical_crystal()
    pinned = focusing.classify(crystal, (3, 6, 4, 8), PLATE, INDEX)
    assert pinned.family_pinned and pinned.mechanism == "jacobian+dimension_collapse"
    assert pinned.as_json()["family_pinned"] is True
    ring = focusing.classify(crystal, (3, 5), PLATE, INDEX)
    assert ring.mechanism == "dimension_collapse" and not ring.family_pinned
    assert ring.as_json()["family_pinned"] is False
    assert not focusing.classify(crystal, (3, 6, 4, 8), RANDOM, INDEX).family_pinned


@pytest.mark.parametrize("faces", PINNED + NOT_PINNED)
def test_family_pinned_against_d_p_on_latitude_circles(faces) -> None:
    """Independent of the criterion: ``D_P`` round the latitude circles about body ``z`` that a plate family's ``u`` runs.

    ``u`` at sun elevations 5, 25, 60 deg (``z = +- sin h``, 721 azimuths),
    points outside ``U_P`` dropped; a pinned path's ``D_P`` is constant on
    every circle (explore degenerate-path-family-coverage #1/#5: <= 5.1e-14
    deg), the others vary by degrees.  At least two circles with 50 valid
    points each, so an empty ``U_P`` cannot pass.
    """
    field = DPField.build(canonical_crystal(), faces, INDEX)
    phi = np.linspace(0.0, 2.0 * np.pi, 721, endpoint=False)
    spreads = []
    for z in np.sin(np.radians([5.0, 25.0, 60.0, -5.0, -25.0, -60.0])):
        r = np.sqrt(1.0 - z * z)
        u = np.stack([r * np.cos(phi), r * np.sin(phi), np.full_like(phi, z)], axis=1)
        inside = u[field.valid_batch(u)]
        if len(inside) >= 50:
            d = np.degrees(field.d_p_batch(inside))
            spreads.append(float(d.max() - d.min()))
    assert len(spreads) >= 2
    if faces in PINNED:
        assert max(spreads) <= 1e-9
    else:
        assert min(spreads) >= 1.0


def test_family_pinned_covers_parry() -> None:
    """Task family-pinned-parry-axis (Lumice corpus C13): the Parry family pins the ``S_x``-fold wedge-0 paths.

    Parry's ``sigma -> 0`` support runs ``u`` round a circle about body ``x``
    (conventions #3), so the pinned paths are those whose fold matrix commutes
    with the rotations about ``x``: the vertical-face mirror ``S_x``, e.g.
    ``1-6-2`` (``D = 2h``, the subsun).  ``1-4-2`` (the mirror of a tilted
    face) and ``3-5`` (wedge 60 deg, the guard's counterexample again) stay
    unpinned.  ``1-3-2``, the roll-180 label-swap partner with the same fold,
    carries the label too: the criterion is algebraic and does not ask which
    poses are valid (that split is
    ``test_conventions.py::test_parry_roll_zero_pins_the_face_6_mirror_path_not_1_3_2``).
    """
    crystal = canonical_crystal()
    assert focusing.family_pinned(crystal, (1, 6, 2), PARRY)
    assert focusing.family_pinned(crystal, (1, 3, 2), PARRY)
    assert not focusing.family_pinned(crystal, (1, 4, 2), PARRY)
    assert not focusing.family_pinned(crystal, (3, 5), PARRY)
    label = focusing.classify(crystal, (1, 6, 2), PARRY, INDEX)
    assert label.family_pinned and label.as_json()["family_pinned"] is True
    assert "dimension_collapse" in label.mechanism


@pytest.mark.parametrize("faces", ((1, 6, 2), (1, 4, 2), (3, 5)))
def test_family_pinned_against_d_p_on_circles_about_body_x(faces) -> None:
    """Independent of the criterion: ``D_P`` round the circles about body ``x`` that a Parry (roll 0) family's ``u`` runs.

    ``u`` at 5, 25, 60, 85 deg off body ``x`` (721 azimuths), points outside
    ``U_P`` dropped; a pinned path's ``D_P`` is constant on every circle
    (wedge 0: ``D_P = angle(u, S_x u)`` depends on ``u . x`` alone), the
    others vary by degrees.  ``1-4-2``'s ``U_P`` misses the two tightest
    circles entirely, hence the fourth level.  At least two circles with 50
    valid points each, so an empty ``U_P`` cannot pass.
    """
    field = DPField.build(canonical_crystal(), faces, INDEX)
    phi = np.linspace(0.0, 2.0 * np.pi, 721, endpoint=False)
    spreads = []
    for eta in np.radians([5.0, 25.0, 60.0, 85.0]):
        c, s = np.cos(eta), np.sin(eta)
        u = np.stack([c * np.ones_like(phi), s * np.cos(phi), s * np.sin(phi)], axis=1)
        inside = u[field.valid_batch(u)]
        if len(inside) >= 50:
            d = np.degrees(field.d_p_batch(inside))
            spreads.append(float(d.max() - d.min()))
    assert len(spreads) >= 2
    if faces == (1, 6, 2):
        assert max(spreads) <= 1e-9
    else:
        assert min(spreads) >= 1.0


def test_mirror_slab_1_2_1_labels_its_creuse_not_a_boundary_extremum(labels) -> None:
    """``1-2-1``: the boundary loop is the crease circle of the mirror fold, a constant ``D_P = 0``.

    ``exit_snell_discriminant = entry_incidence_cosine^2`` on the whole
    domain, so ``dU_P`` is the single entry great circle (corner-free,
    constant): the crease enters as the ``slab_circle`` onset and the axis
    cone point as the ``slab_axis`` one -- no ``boundary_extremum`` is
    invented for a loop that has no isolated extremum
    (``BoundaryLoop.plateau_value``; task ``boundary-corner-1-2-1``).
    """
    label = labels[(1, 2, 1)]
    assert label.mechanism == "none" and not label.jacobian_focusing
    assert [o.source for o in label.onsets] == ["slab_circle", "slab_axis"]
    (crease,) = [o for o in label.onsets if o.source == "slab_circle"]
    assert crease.location == "boundary" and crease.profile == "crease" and crease.value == pytest.approx(0.0, abs=1e-9)
    (axis,) = [o for o in label.onsets if o.source == "slab_axis"]
    assert axis.location == "interior" and axis.profile == "cone_point"
    assert np.degrees(axis.value) == pytest.approx(180.0, abs=1e-9)
    assert not [o for o in label.onsets if o.source == "boundary_extremum" or o.source == "corner"]
