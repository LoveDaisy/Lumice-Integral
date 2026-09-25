"""``ch10_verdicts``: the chapter-10 verdicts on reduced grids (the recorded run is ``scripts/ch10_numerical_verdicts.py``)."""

from __future__ import annotations

import json
from hashlib import sha256

import numpy as np
import pytest

from lumice_integral import ch10_verdicts as V
from lumice_integral import contour_quadrature as cq
from lumice_integral.canonical_scene import CANONICAL_HEIGHT_RATIO, CANONICAL_REFRACTIVE_INDEX, canonical_crystal
from lumice_integral.camera import sun_direction
from lumice_integral.dp_field import DPField
from lumice_integral.figure_data import VERDICT_SCHEMA_VERSION, export_verdict_figure_data
from lumice_integral.geometry import HexPrism
from lumice_integral.pose_density import build_pose_density
from lumice_integral.s2_store import fibonacci_sphere

INDEX = CANONICAL_REFRACTIVE_INDEX


@pytest.fixture(scope="module")
def inner_edge() -> V.Verdict:
    return V.inner_edge(V.InnerEdgeOptions(
        eps_random=tuple(10.0 ** np.arange(-2.0, -6.01, -0.5)),
        eps_column=tuple(10.0 ** np.arange(-1.0, -6.51, -0.25)),
        column_widths_deg=(0.5, 0.1),
    ))


def test_inner_edge_random_orientation_is_a_finite_jump(inner_edge) -> None:
    """``int w / |grad D| dl -> w* 2 pi / sqrt(det H)`` with a ``sqrt(eps)`` approach the kinks of ``w`` predict."""
    n = inner_edge.numbers
    assert n["hessian_max_relative_difference"] < 1e-6  # finite differences of the value against the AD Hessian
    r = n["random"]
    assert 0.997 < r["ratio_to_analytic_at_smallest_eps"] < 1.0
    assert r["extrapolated_ratio_c"] == pytest.approx(1.0, abs=1e-4)
    assert r["sqrt_eps_coefficient_fitted"] == pytest.approx(r["sqrt_eps_coefficient_predicted_from_kinks"], rel=0.03)
    assert abs(r["local_slope_at_smallest_eps"]) < 0.01  # not a power law
    ratio = inner_edge.arrays["raw_random"] / n["raw_limit_analytic"]
    assert np.all(np.diff(ratio) > 0.0)
    assert n["pixel_value_limit_analytic"] == pytest.approx(
        n["raw_limit_analytic"] * cq.HAAR_TO_DVOL_G_FACTOR / np.sin(np.radians(n["D_min_deg"])), rel=1e-12)


def test_inner_edge_column_family_is_inverse_sqrt_between_its_cap_and_1e_2(inner_edge) -> None:
    """At the tangent-arc contact the column profile is ``eps^(-1/2)`` down to ``eps_c ~ sigma^2``, finite below."""
    column = inner_edge.numbers["column"]
    assert column["ring_azimuth_deg"] == pytest.approx(90.0, abs=0.2)
    assert column["ring_top_sky_elevation_deg"] == pytest.approx(15.0 + inner_edge.numbers["D_min_deg"], abs=1e-6)
    wide, narrow = column["widths"]
    lo, hi = narrow["power_law_window_eps"]
    assert hi / lo > 100.0  # sigma = 0.1 deg: more than two decades of -1/2
    assert wide["cap_crossover_eps"] / narrow["cap_crossover_eps"] == pytest.approx(25.0, rel=0.3)  # (0.5 / 0.1)^2
    assert column["cap_crossover_exponent_in_sigma"] == pytest.approx(2.0, abs=0.1)
    caps = list(column["cap_times_sigma_rad"].values())
    assert caps[0] == pytest.approx(caps[1], rel=0.1)
    slopes = inner_edge.arrays["local_slope_column"]
    assert np.all(np.abs(slopes[:, -1]) < 0.1)  # capped at the smallest eps


def test_a60_10_is_reachable_once_internal_reflections_may_be_partial() -> None:
    """Liljequist (i): since task optics-partial-reflection the internal TIR discriminant gates nothing, so the
    A60-10 domain is exactly the set passing every other gate (it was empty while internal TIR was a gate; the
    verdict itself is rerun by task ch10-liljequist-unblock-and-docs)."""
    status = V.blocked_class_status(((3, 5, 6, 7), (3, 4, 5, 7)), INDEX, lattice_n=20000)
    for member in ("3-5-6-7", "3-4-5-7"):
        assert status[member]["valid_points"] == status[member]["points_passing_all_but_internal_tir"] > 500


def test_liljequist_paths_share_one_mirror_field_and_shape_free_critical_values() -> None:
    """``1-3-2`` and ``3-5-6-7-3``: ``D = 2 arcsin |u . n_3|`` on both, ``|grad D| = 2``, critical values independent of ``h / a``."""
    lattice = fibonacci_sphere(20000)
    a, b = (DPField.build(canonical_crystal(), faces, INDEX) for faces in ((1, 3, 2), (3, 5, 6, 7, 3)))
    inside = a.valid_batch(lattice) | b.valid_batch(lattice)
    closed_form = 2.0 * np.arcsin(np.abs(lattice[inside] @ np.array([1.0, 0.0, 0.0])))
    np.testing.assert_allclose(a.d_p_batch(lattice[inside]), closed_form, atol=1e-12)
    np.testing.assert_allclose(b.d_p_batch(lattice[inside]), closed_form, atol=1e-12)
    for faces in ((1, 3, 2), (3, 5, 6, 7, 3)):
        values = [DPField.build(HexPrism.from_ratio(r), faces, INDEX).critical_values for r in (0.2, 2.0)]
        np.testing.assert_allclose(values[0], values[1], atol=1e-12)


def test_liljequist_peak_is_a_one_sided_sqrt_cusp_at_the_boundary_critical_value() -> None:
    crystal = HexPrism.from_ratio(2.0)
    field = DPField.build(crystal, (3, 5, 6, 7, 3), INDEX)
    store = V.seed_store(crystal, INDEX, (3, 5, 6, 7, 3))
    (critical,) = [v for v in field.critical_values if np.radians(150.0) < v < np.radians(160.0)]
    assert np.degrees(critical) == pytest.approx(153.0697, abs=1e-4)
    eps = np.array([1e-3, 1e-4, 1e-5])
    options = cq.QuadratureOptions(relative_tolerance=1e-8)
    _, above, _ = V._random_profile(field, critical + eps, store, options)
    _, below, _ = V._random_profile(field, critical - eps, store, options)
    gap = above[-1] - below
    assert np.all(gap > 0.0)
    assert np.polyfit(np.log(eps), np.log(gap), 1)[0] == pytest.approx(0.5, abs=0.05)
    assert np.all(np.abs(np.diff(above)) / above[-1] < 1e-2)  # finite from above


def test_parhelic_circle_window_geometry() -> None:
    """Vertical plates on ``1-3-2``: image at the sun's elevation, ``d theta / d phi = 2``; six members, one window shifted."""
    crystal, sun = HexPrism.from_ratio(0.2), sun_direction(15.0, 0.0)
    phi = np.linspace(0.0, 2.0 * np.pi, 7200, endpoint=False)
    window = V.ring_window(crystal, INDEX, sun, (1, 3, 2), phi)
    lit = window["w"] > 0.0
    interior = lit & np.roll(lit, 1) & np.roll(lit, -1)
    assert lit.any()
    np.testing.assert_allclose(window["elevation"][lit], np.radians(15.0), atol=1e-12)
    np.testing.assert_allclose(window["dtheta_dphi"][interior], 2.0, atol=1e-6)
    for k in range(4, 9):
        other = V.ring_window(crystal, INDEX, sun, (1, k, 2), phi)["w"]
        np.testing.assert_allclose(other, np.roll(window["w"], -(k - 3) * 1200), atol=1e-12)


def test_parhelic_circle_ring_equals_the_window_prediction() -> None:
    """At one ring azimuth away from window jumps, the elevation-integrated contour value is the window-only prediction."""
    options = V.ParhelicCircleOptions(plate_widths_deg=(0.5,), thetas_deg=(110.0,), cross_nodes=32)
    crystal, sun = HexPrism.from_ratio(options.height_ratio), sun_direction(options.sun_altitude_deg, 0.0)
    phi = np.linspace(0.0, 2.0 * np.pi, options.phi_samples, endpoint=False)
    prediction = V.window_prediction(V.ring_window(crystal, INDEX, sun, (1, 3, 2), phi), phi, np.radians([110.0]))[0]
    field = DPField.build(crystal, (1, 3, 2), INDEX)
    store = V.seed_store(crystal, INDEX, (1, 3, 2))
    sigma = np.radians(0.5)
    density = build_pose_density("plate", zenith_std_deg=0.5)
    quadrature = cq.QuadratureOptions(relative_tolerance=1e-8, initial_panel_rad=sigma / 4.0)

    def values_at(deltas, centres):
        geometry = V.level_set_geometry(field, deltas, store, quadrature)
        return np.array([r.value for r in geometry.integrate(sun, centres, density)])

    ring, elevations, values = V.ring_cross_integral(values_at, sun, np.radians(110.0), np.radians(15.0), sigma, options)
    assert prediction > 0.0
    assert ring == pytest.approx(prediction, rel=5e-3)
    assert max(values[0], values[-1]) < 1e-3 * values.max()  # the cross-section is inside the integration window


def test_verdict_figure_data_round_trip(tmp_path) -> None:
    verdict = V.Verdict(
        "demo", "measured", "a statement", {"x": np.float64(1.5), "nested": {"y": [1, 2]}}, {"p": (1, 2)},
        {"a": np.arange(4.0), "b": np.eye(2)}, {"a": "rad; test", "b": "dimensionless; test"},
    )
    files = export_verdict_figure_data(verdict, tmp_path / "demo", provenance={"commit": "abc"})
    metadata = json.loads(files.metadata.read_text(encoding="utf-8"))
    assert metadata["schema"] == VERDICT_SCHEMA_VERSION
    assert (metadata["verdict"], metadata["status"], metadata["statement"]) == ("demo", "measured", "a statement")
    assert metadata["numbers"] == {"x": 1.5, "nested": {"y": [1, 2]}} and metadata["provenance"] == {"commit": "abc"}
    assert metadata["payload"]["sha256"] == sha256(files.arrays.read_bytes()).hexdigest()
    assert metadata["payload"]["arrays"]["b"] == {"shape": [2, 2], "dtype": "float64", "note": "dimensionless; test"}
    with np.load(files.arrays) as arrays:
        np.testing.assert_array_equal(arrays["a"], np.arange(4.0))
        assert set(arrays.files) == {"a", "b"}
    with pytest.raises(ValueError, match="without a note"):
        export_verdict_figure_data(V.Verdict("bad", "measured", "", {}, {}, {"c": np.zeros(1)}, {}), tmp_path / "bad")


@pytest.fixture(scope="module")
def parallel_face() -> V.Verdict:
    return V.parallel_face(V.ParallelFaceOptions(
        paths=((3, 5), (1, 3, 2)),
        families=(("random", {}), ("column", {"zenith_std_deg": 0.5})),
        demo_widths_deg=(0.5, 0.25),
        cross_nodes=24,
        seed_store_n=50_000,
    ))


def test_parallel_face_table_has_no_jacobian_focusing_on_these_fixtures(parallel_face) -> None:
    """3-5 (finite jump) and 1-3-2 (wedge-0 slab) never focus by Jacobian; only rho confines a dimension."""
    n = parallel_face.numbers
    table = n["table"]
    assert len(table) == 4  # 2 paths x 2 families
    by_path_family = {(row["path"], row["family"]): row["mechanism"] for row in table}
    assert by_path_family == {
        ("3-5", "random"): "none",
        ("3-5", "column"): "dimension_collapse",
        ("1-3-2", "random"): "none",
        ("1-3-2", "column"): "dimension_collapse",
    }
    assert n["paths_with_jacobian_focusing"] == []
    assert n["mechanisms"] == sorted(list(m) for m in (
        ("1-3-2", "column", "dimension_collapse"), ("1-3-2", "random", "none"),
        ("3-5", "column", "dimension_collapse"), ("3-5", "random", "none")))


def test_parallel_face_collapse_demo_narrows_the_peak_at_fixed_integral(parallel_face) -> None:
    """1-3-2 under plates: halving sigma roughly doubles the peak while the cross integral stays fixed."""
    demo = parallel_face.numbers["collapse_demo"]["widths"]
    wide, narrow = demo
    assert wide["plate_zenith_std_deg"] > narrow["plate_zenith_std_deg"]
    gain = narrow["peak_value"] / wide["peak_value"]
    assert gain > 1.0
    narrowing = wide["plate_zenith_std_deg"] / narrow["plate_zenith_std_deg"]
    assert gain == pytest.approx(narrowing, rel=0.1)
    assert narrow["cross_integral"] == pytest.approx(wide["cross_integral"], rel=1e-2)
    # random orientation at the same pixels stays smooth (no collapse signature)
    assert max(narrow["random_value_range"]) < 0.05 * narrow["peak_value"]


def test_parallel_face_provenance_reuses_the_canonical_height_ratio(parallel_face) -> None:
    """The table's crystal is canonical_crystal(); its provenance must not fork a second literal (a56)."""
    assert parallel_face.parameters["crystal"]["height_ratio_table"] == CANONICAL_HEIGHT_RATIO


def test_ring_direction_refuses_a_zenith_sun() -> None:
    with pytest.raises(ValueError):
        V.ring_direction(np.array([0.0, 0.0, 1.0]), 0.4, 0.0)
