"""``focusing``: Jacobian focusing vs dimension collapse as an explicit label (``docs/phase2.md`` section 10)."""

from __future__ import annotations

import numpy as np
import pytest

from lumice_integral import focusing
from lumice_integral.canonical_scene import CANONICAL_REFRACTIVE_INDEX, canonical_crystal
from lumice_integral.dp_field import DPField
from lumice_integral.geometry import halo_map_rank
from lumice_integral.pose_density import build_pose_density

INDEX = CANONICAL_REFRACTIVE_INDEX
RANDOM = build_pose_density("random")
COLUMN = build_pose_density("column", zenith_std_deg=0.5)
PLATE = build_pose_density("plate", zenith_std_deg=0.5)
PARRY = build_pose_density("parry", zenith_std_deg=0.5, roll_std_deg=1.0)
LOWITZ = build_pose_density("lowitz", zenith_std_deg=0.5, roll_std_deg=1.0)
SLABS = ((1, 3, 2), (3, 5, 6, 7, 3), (3, 1, 6), (1, 3, 5, 2))


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


@pytest.mark.parametrize("faces", SLABS)
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


def test_slab_axis_and_circle_labels(labels) -> None:
    liljequist = labels[(3, 5, 6, 7, 3)]
    (cone,) = [o for o in liljequist.onsets if o.source == "slab_axis"]
    assert cone.location == "interior" and cone.profile == "cone_point"
    assert np.degrees(cone.value) == pytest.approx(180.0, abs=1e-9) and cone.gradient_norm == pytest.approx(2.0, abs=1e-6)
    (crease,) = [o for o in liljequist.onsets if o.source == "slab_circle"]
    assert crease.profile == "crease" and crease.value == 0.0
    # rotation slab (det +1, 120 deg): cone slope 2 sin 60 deg at the axis; its fold circle (D = 120 deg) misses U_P
    rotation = labels[(1, 3, 5, 2)]
    (axis,) = [o for o in rotation.onsets if o.source == "slab_axis"]
    assert axis.gradient_norm == pytest.approx(np.sqrt(3.0), abs=1e-6)
    assert not [o for o in rotation.onsets if o.source == "slab_circle"]


def test_boundary_records_are_merged(labels) -> None:
    """Mirror-image extrema and multi-margin corners are one record each, with their multiplicity."""
    for label in labels.values():
        keys = [(round(o.value, 6), o.location, o.source, o.profile) for o in label.onsets]
        assert len(keys) == len(set(keys))
    (peak,) = [o for o in labels[(3, 5, 6, 7, 3)].onsets if o.source == "boundary_extremum"]
    assert np.degrees(peak.value) == pytest.approx(153.0697, abs=1e-4) and peak.multiplicity == 2
