"""Path classes: PBD orbit vs the independent oracle, invariants, the rank-0 point mass, the class pixel.

Structure (task-path-class-rendering-unit plan Steps 4-6):

- orbit expansion of ``path_class.pbd_orbit_hexprism`` (group matrices acting
  on face normals) against ``_geometry_oracles.pbd_orbit`` (index arithmetic)
  on every face sequence up to three faces plus the issue's four-face member;
- the canonical class ``[3, 5]`` is the 12 rotations of ``3-5`` and ``3-7``;
- the rank-0 estimator of ``1-2`` under a Haar-uniform density against an
  independent ray-cast Monte Carlo (``trace_faces`` + closed-form Fresnel,
  its own generator), its ``1/sqrt(N)`` error decay and its valid fraction;
- the class pixel: a pseudo single-member class reproduces the canonical
  ``render_pixel`` bit for bit (AC5), and the rank-0 class contributes the
  point mass only on the sun pixel and nothing (without sampling) when the
  window does not contain the sun (AC3).
"""

from __future__ import annotations

import itertools
import json

import numpy as np
import pytest

import lumice_integral.path_class as path_class_module
from lumice_integral.canonical_scene import (
    CANONICAL_PIXEL_COLUMN,
    CANONICAL_PIXEL_ROW,
    CANONICAL_REFRACTIVE_INDEX,
    CANONICAL_RENDER,
    canonical_crystal,
    canonical_incident_direction,
    canonical_pose_density,
    canonical_target_direction,
)
from lumice_integral.geometry import HexPrism, halo_map_rank, wedge_angle_deg
from lumice_integral.path_class import (
    ClassPixelResult,
    ClassScene,
    PathClass,
    Rank0Estimate,
    build_class_scene,
    build_path_class,
    canonical_class_scene,
    discover_class_components,
    estimate_rank0_contribution,
    pbd_orbit_hexprism,
    pixel_solid_angle,
    render_class_pixel,
    sun_pixel,
)
from lumice_integral.pose_density import build_pose_density
from lumice_integral.prescan import build_prescan_table
from lumice_integral.quadrature import HAAR_TO_DVOL_G_FACTOR
from lumice_integral.strip_pixel import PixelOptions, canonical_strip_scene, render_pixel

from _geometry_oracles import pbd_orbit, trace_faces

CRYSTAL = HexPrism(1.0, 0.8)
TEST_PRESCAN_SAMPLES = 400_000  # tests/test_strip_pixel.py's table; the canonical value is pinned to it


# ---- Step 4: orbit and class ------------------------------------------------------


@pytest.mark.parametrize("faces", [(3, 5), (3, 7), (1, 2), (3, 6), (1, 3), (3, 1, 2, 5), (1, 3, 2), (3, 1, 4), (1, 3, 5, 2)])
def test_orbit_matches_the_oracle_on_named_paths(faces):
    assert pbd_orbit_hexprism(faces) == frozenset(pbd_orbit(faces))


def test_orbit_matches_the_oracle_on_every_sequence_up_to_three_faces():
    """Exhaustive over 8 * 8 + 8 * 8 * 8 sequences (repeated faces included: the group action does not care)."""
    for length in (2, 3):
        for faces in itertools.product(range(1, 9), repeat=length):
            assert pbd_orbit_hexprism(faces) == frozenset(pbd_orbit(faces)), faces


def test_orbit_does_not_depend_on_the_aspect_ratio():
    assert pbd_orbit_hexprism((3, 1, 2, 5), HexPrism(1.0, 0.2)) == pbd_orbit_hexprism((3, 1, 2, 5), HexPrism(1.0, 5.0))


def test_canonical_class_is_the_twelve_rotations_of_3_5_and_3_7():
    path_class = build_path_class(canonical_crystal(), (3, 5))
    rotations_3_5 = {(3 + k, 3 + (k + 2) % 6) for k in range(6)}
    rotations_3_7 = {(3 + k, 3 + (k + 4) % 6) for k in range(6)}
    assert set(path_class.members) == rotations_3_5 | rotations_3_7
    assert path_class.size == 12 and path_class.representative == (3, 5)
    assert path_class.halo_map_rank == 2 and path_class.wedge_deg == pytest.approx(60.0, abs=1e-9)
    assert path_class.path_ids[:2] == ("3-5", "3-7")
    assert json.loads(json.dumps(path_class.provenance()))["members"][0] == [3, 5]


@pytest.mark.parametrize("representative, size, rank, wedge", [((1, 2), 2, 0, 0.0), ((3, 6), 6, 0, 0.0), ((3, 1, 2, 5), 24, 2, 60.0)])
def test_class_invariants_hold_on_every_member(representative, size, rank, wedge):
    path_class = build_path_class(CRYSTAL, representative)
    assert path_class.size == size and path_class.halo_map_rank == rank
    assert path_class.wedge_deg == pytest.approx(wedge, abs=1e-9)
    for member in path_class.members:
        assert halo_map_rank(CRYSTAL, member) == rank
        assert wedge_angle_deg(CRYSTAL, member) == pytest.approx(wedge, abs=1e-9)
    assert path_class.members == tuple(sorted(path_class.members))


def test_path_class_validation():
    with pytest.raises(ValueError, match="representative"):
        PathClass((3, 5), ((3, 7),), 60.0, 2)
    with pytest.raises(ValueError, match="distinct"):
        PathClass((3, 5), ((3, 5), (3, 5)), 60.0, 2)
    with pytest.raises(ValueError, match="halo_map_rank"):
        PathClass((3, 5), ((3, 5),), 60.0, 1)
    with pytest.raises(TypeError):
        build_path_class(object(), (3, 5))  # type: ignore[arg-type]


# ---- Step 5: rank-0 point mass -----------------------------------------------------


def _oracle_rank0_mass_1_2(crystal: HexPrism, n_ice: float, *, directions: int, rays: int, seed: int) -> tuple[float, float]:
    """``E_Haar[A T]`` of path 1-2 by ray casting: Haar-uniform poses are uniform body-frame incident
    directions; the footprint area is the hit fraction of a disk perpendicular to the ray that covers
    the crystal, the transmittance the closed-form s/p average at the two basal interfaces."""
    rng = np.random.default_rng(seed)
    radius = np.sqrt(crystal.a**2 + (crystal.h / 2.0) ** 2)
    n1 = crystal.normal(crystal.face(1))

    def unpolarized(n_a: float, cos_a: float, n_b: float, cos_b: float) -> float:
        r_s = (n_a * cos_a - n_b * cos_b) / (n_a * cos_a + n_b * cos_b)
        r_p = (n_b * cos_a - n_a * cos_b) / (n_b * cos_a + n_a * cos_b)
        return 1.0 - 0.5 * (r_s * r_s + r_p * r_p)

    values = []
    for _ in range(directions):
        s = rng.standard_normal(3)
        s /= np.linalg.norm(s)
        cos_i = -float(s @ n1)
        if cos_i <= 0.0:
            values.append(0.0)
            continue
        u = np.cross(s, [1.0, 0.0, 0.0] if abs(s[0]) < 0.9 else [0.0, 1.0, 0.0])
        u /= np.linalg.norm(u)
        w = np.cross(s, u)
        hits = 0
        for _ in range(rays):
            r = radius * np.sqrt(rng.random())
            phi = 2.0 * np.pi * rng.random()
            origin = r * np.cos(phi) * u + r * np.sin(phi) * w - 10.0 * s
            try:
                if trace_faces(crystal, origin, s, 2, n_ice=n_ice) == [1, 2]:
                    hits += 1
            except (ValueError, RuntimeError):
                continue
        area = np.pi * radius * radius * hits / rays
        cos_t = np.sqrt(1.0 - (1.0 / n_ice) ** 2 * (1.0 - cos_i * cos_i))
        values.append(area * unpolarized(1.0, cos_i, n_ice, cos_t) * unpolarized(n_ice, cos_t, 1.0, cos_i))
    values = np.asarray(values)
    return float(values.mean()), float(values.std() / np.sqrt(directions))


def test_rank0_mass_of_1_2_agrees_with_an_independent_ray_cast_monte_carlo():
    crystal = canonical_crystal()
    estimate = estimate_rank0_contribution(
        crystal, (1, 2), canonical_incident_direction(), CANONICAL_REFRACTIVE_INDEX, build_pose_density("random"),
        rng_seed=1, sample_count=400_000,
    )
    oracle, oracle_error = _oracle_rank0_mass_1_2(crystal, CANONICAL_REFRACTIVE_INDEX, directions=600, rays=200, seed=99)
    assert estimate.path_id == "1-2" and estimate.sample_count == 400_000 and estimate.rng_seed == 1
    assert abs(estimate.value - oracle) <= 3.0 * np.hypot(estimate.error_estimate, oracle_error)
    assert estimate.error_estimate < 0.01 * estimate.value
    # Under a Haar-uniform density the incident direction is irrelevant (the mass is a body-frame average).
    other = estimate_rank0_contribution(
        crystal, (1, 2), -canonical_incident_direction(), CANONICAL_REFRACTIVE_INDEX, build_pose_density("random"),
        rng_seed=1, sample_count=400_000,
    )
    assert abs(other.value - estimate.value) <= 3.0 * np.hypot(other.error_estimate, estimate.error_estimate)
    assert estimate.value_dvol_g == estimate.value / HAAR_TO_DVOL_G_FACTOR


def test_rank0_error_decays_as_one_over_sqrt_n_and_the_domain_is_half_of_haar():
    """Face 1 faces the sun for exactly half of Haar; no other gate exists before the footprint."""
    kwargs = dict(rng_seed=3)
    small = estimate_rank0_contribution(
        CRYSTAL, (1, 2), canonical_incident_direction(), 1.31, canonical_pose_density(), sample_count=100_000, **kwargs
    )
    large = estimate_rank0_contribution(
        CRYSTAL, (1, 2), canonical_incident_direction(), 1.31, canonical_pose_density(), sample_count=400_000, **kwargs
    )
    assert small.error_estimate / large.error_estimate == pytest.approx(2.0, rel=0.15)
    assert abs(small.value - large.value) <= 3.0 * np.hypot(small.error_estimate, large.error_estimate)
    assert small.valid_fraction == pytest.approx(0.5, abs=0.01) and large.valid_fraction == pytest.approx(0.5, abs=0.005)
    # Same Haar stream as the prescan: the first batch of a prescan of the same seed sees the same poses.
    table = build_prescan_table(canonical_incident_direction(), 1.31, sample_count=100_000, rng_seed=3, path_id="1-2")
    assert table.valid_count == round(small.valid_fraction * 100_000)
    # A rank-2 path is far more selective than the rank-0 one.
    table_3_5 = build_prescan_table(canonical_incident_direction(), 1.31, sample_count=100_000, rng_seed=3)
    assert table_3_5.valid_count / 100_000 < 0.5 * small.valid_fraction
    with pytest.raises(ValueError, match="rank-0"):
        estimate_rank0_contribution(CRYSTAL, (3, 5), canonical_incident_direction(), 1.31, canonical_pose_density())


# ---- Step 6: class scene and class pixel -------------------------------------------


@pytest.fixture(scope="module")
def options() -> PixelOptions:
    return PixelOptions()


def test_pseudo_single_member_class_reproduces_the_canonical_pixel_bit_for_bit(options):
    """AC5: the class driver over ``{3-5}`` is the same code path as ``canonical_strip_scene`` + ``render_pixel``."""
    single = PathClass((3, 5), ((3, 5),), wedge_angle_deg(canonical_crystal(), (3, 5)), 2)
    scene = build_class_scene(
        single,
        incident_direction=canonical_incident_direction(),
        refractive_index=CANONICAL_REFRACTIVE_INDEX,
        crystal=canonical_crystal(),
        pose_density=canonical_pose_density(),
        render=CANONICAL_RENDER,
        prescan_sample_count=TEST_PRESCAN_SAMPLES,
    )
    baseline_scene = canonical_strip_scene(prescan_sample_count=TEST_PRESCAN_SAMPLES)
    baseline = render_pixel(baseline_scene, CANONICAL_PIXEL_ROW, CANONICAL_PIXEL_COLUMN, options)
    result = render_class_pixel(scene, CANONICAL_PIXEL_ROW, CANONICAL_PIXEL_COLUMN, options)

    assert isinstance(result, ClassPixelResult) and tuple(result.members) == ((3, 5),)
    member = result.members[(3, 5)]
    assert result.value == baseline.value and result.error_estimate == baseline.error_estimate
    assert member.value == baseline.value and member.completeness == baseline.completeness == result.completeness
    assert len(member.components) == len(baseline.components) == 1
    for ours, theirs in zip(member.components, baseline.components):
        assert np.array_equal(ours.seed, theirs.seed)
        for name, value in vars(ours).items():
            if name == "seed":
                continue
            other = getattr(theirs, name)
            assert value == other or (isinstance(value, float) and np.isnan(value) and np.isnan(other)), name
    assert result.components[(3, 5)] == member.components
    assert result.events == baseline.events and result.rank0_estimate is None
    provenance = json.loads(json.dumps(result.provenance))
    assert provenance["representative"] == [3, 5] and provenance["members"] == [[3, 5]]
    assert provenance["halo_map_rank"] == 2 and provenance["wedge_deg"] == pytest.approx(60.0)
    assert provenance["per_member"]["3-5"]["value"] == baseline.value
    assert provenance["per_member"]["3-5"]["completeness"] == baseline.completeness
    assert provenance["prescan"] == {"sample_count": TEST_PRESCAN_SAMPLES, "rng_seed": baseline_scene.prescan_table.rng_seed}
    assert provenance["rank0"] is None
    # The discovery-level entry finds the same single closed component.
    discovered = discover_class_components(canonical_target_direction(), scene, options)
    assert tuple(discovered.per_member) == ((3, 5),) and discovered.rank0_estimate is None
    assert discovered.component_count == 1 and discovered.completeness == baseline.completeness
    assert np.array_equal(discovered.per_member[(3, 5)].components[0].seed, member.components[0].seed)


def test_class_scene_rejects_mismatched_member_scenes():
    path_class = build_path_class(canonical_crystal(), (3, 5))
    kwargs = dict(
        incident_direction=canonical_incident_direction(),
        refractive_index=CANONICAL_REFRACTIVE_INDEX,
        crystal=canonical_crystal(),
        pose_density=canonical_pose_density(),
        render=CANONICAL_RENDER,
    )
    only_3_5 = {(3, 5): canonical_strip_scene(prescan_sample_count=1_000)}
    with pytest.raises(ValueError, match="member_scenes"):
        ClassScene(path_class=path_class, prescan_sample_count=1_000, prescan_rng_seed=20260916, member_scenes=only_3_5, **kwargs)
    with pytest.raises(ValueError, match="Haar stream"):
        ClassScene(
            path_class=PathClass((3, 5), ((3, 5),), 60.0, 2), prescan_sample_count=2_000, prescan_rng_seed=20260916,
            member_scenes=only_3_5, **kwargs,
        )
    # A rank-0 class has no member scenes at all.
    scene = build_class_scene(build_path_class(canonical_crystal(), (1, 2)), prescan_sample_count=1_000, **kwargs)
    assert scene.member_scenes == {} and not scene.sun_in_field_of_view


SUN_WINDOW = {"width": 21, "height": 21, "fov_deg": 6.0, "view": {"azimuth": 0.0, "elevation": 15.0}}


def test_sun_pixel_and_pixel_solid_angle():
    incident = canonical_incident_direction()
    assert sun_pixel(CANONICAL_RENDER, incident) is None  # the strip looks 30 deg below the sun
    assert sun_pixel(SUN_WINDOW, incident) == (10, 10)
    assert sun_pixel(SUN_WINDOW, -incident) is None  # behind the camera
    # Pixel size on the tangent plane: 6 deg over 21 pixels; the centre pixel is on the axis.
    centre = pixel_solid_angle(SUN_WINDOW, 10, 10)
    assert centre == pytest.approx(np.radians(6.0 / 21) ** 2, rel=0.02)
    assert pixel_solid_angle(SUN_WINDOW, 0, 0) < centre


def test_rank0_class_contributes_the_point_mass_only_on_the_sun_pixel(options, monkeypatch):
    """AC3: the point mass lands on the sun pixel (pixel-averaged), every other pixel gets 0."""
    scene = canonical_class_scene((1, 2), prescan_sample_count=50_000, render=SUN_WINDOW)
    assert scene.path_class.halo_map_rank == 0 and scene.sun_pixel == (10, 10)
    on_sun = render_class_pixel(scene, 10, 10, options)
    assert isinstance(on_sun.rank0_estimate, Rank0Estimate)
    assert on_sun.rank0_estimate.sample_count == 50_000 and on_sun.rank0_estimate.rng_seed == scene.prescan_rng_seed
    assert on_sun.value == on_sun.rank0_estimate.value / pixel_solid_angle(SUN_WINDOW, 10, 10) > 0.0
    assert on_sun.error_estimate == on_sun.rank0_estimate.error_estimate / pixel_solid_angle(SUN_WINDOW, 10, 10)
    assert on_sun.members == {} and on_sun.completeness == "complete"
    provenance = json.loads(json.dumps(on_sun.provenance))
    assert provenance["halo_map_rank"] == 0 and provenance["rank0"]["sun_in_field_of_view"] is True
    assert provenance["rank0"]["sun_pixel"] == [10, 10] and provenance["rank0"]["this_pixel_contains_sun"] is True
    assert provenance["rank0"]["estimate"]["value"] == on_sun.rank0_estimate.value
    assert provenance["per_member"] == {}

    beside = render_class_pixel(scene, 10, 11, options)
    assert beside.value == 0.0 and beside.rank0_estimate is not None
    assert beside.provenance["rank0"]["this_pixel_contains_sun"] is False and "another pixel" in beside.provenance["rank0"]["reason"]

    discovered = discover_class_components(np.array([0.0, 0.0, 1.0]), scene, options)
    assert discovered.per_member == {} and discovered.rank0_estimate == on_sun.rank0_estimate


def test_rank0_class_outside_the_field_of_view_is_zero_without_sampling(options, monkeypatch):
    """AC3: the strip window does not contain the sun; the estimator must not even be called."""
    scene = canonical_class_scene((3, 6), prescan_sample_count=50_000)
    assert scene.path_class.halo_map_rank == 0 and not scene.sun_in_field_of_view

    def forbidden(*args, **kwargs):
        raise AssertionError("estimate_rank0_contribution must not run when the sun is outside the window")

    monkeypatch.setattr(path_class_module, "estimate_rank0_contribution", forbidden)
    result = render_class_pixel(scene, CANONICAL_PIXEL_ROW, CANONICAL_PIXEL_COLUMN, options)
    assert result.value == 0.0 and result.error_estimate == 0.0 and result.rank0_estimate is None
    assert result.completeness == "complete" and result.members == {}
    block = json.loads(json.dumps(result.provenance))["rank0"]
    assert block["sun_in_field_of_view"] is False and block["sun_pixel"] is None
    assert "does not contain the sun" in block["reason"]
    assert discover_class_components(np.array([0.0, 0.0, 1.0]), scene, options).rank0_estimate is None
