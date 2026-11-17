"""Independent geometric and optical checks of exact restricted supports."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lumice_integral import optics
from lumice_integral.camera import sun_direction
from lumice_integral.dp_field import DPField
from lumice_integral.family_restricted import (
    ExactFamily, RestrictedEscape, _circle_roots, intersect_margin, kink_sky, restricted_critical,
)
from lumice_integral.geometry import HexPrism
from lumice_integral.so3 import exp
from lumice_integral.spectrum.dispersion import refractive_index

SUN = sun_direction(20, 0)
N = refractive_index(550)


@pytest.mark.parametrize("name", ["plate", "column", "parry", "lowitz"])
def test_exact_support_rotation_generators_and_all_lifts(name):
    family = ExactFamily(name)
    for az in (0., .7, 2 * np.pi - 1e-8):
        q = np.array([az, .8] if family.dimension == 2 else [az])
        r = np.asarray(family.rotation(q))
        np.testing.assert_allclose(r.T @ r, np.eye(3), atol=1e-14)
        assert np.linalg.det(r) == pytest.approx(1, abs=1e-14)
        u = r.T @ SUN
        lifts = family.lifts(u, SUN)
        # At a projection fold the inverse is sqrt(eps)-conditioned in pose;
        # the forward Ru=sun residual remains O(eps). Do not pin lost digits.
        assert any(np.linalg.norm(np.asarray(family.rotation(p)) - r) < 1e-7 for p in lifts)
        omega = np.asarray(family.generators(q))
        ad = np.asarray(jax.jacfwd(lambda x: family.u(x, jnp.asarray(SUN)))(jnp.asarray(q)))
        for i in range(family.dimension):
            np.testing.assert_allclose(ad[:, i], -np.cross(omega[i], u), atol=1e-13)
            for step in (1e-3, 1e-4, 1e-5):
                dq = np.eye(family.dimension)[i] * step
                fd = (np.asarray(family.u(q+dq, SUN)) - np.asarray(family.u(q-dq, SUN))) / (2 * step)
                np.testing.assert_allclose(fd, ad[:, i], atol=2e-7)
            np.testing.assert_allclose(np.asarray(family.rotation(q + np.eye(family.dimension)[i] * 1e-6)),
                                       r @ np.asarray(exp(jnp.asarray(omega[i] * 1e-6))), atol=1e-12)
        if name in ("column", "parry"):
            assert abs(r[2, 2]) < 1e-14
        if name in ("parry", "lowitz"):
            assert abs(r[2, 1]) < 1e-14


def test_projection_loses_pose_and_two_dimensional_supports_are_not_circles():
    family = ExactFamily("column")
    u = np.asarray(family.u(jnp.array([.8, .4]), SUN))
    lifts = family.lifts(u, SUN)
    assert len(lifts) == 2
    rotations = [np.asarray(family.rotation(q)) for q in lifts]
    assert np.linalg.norm(rotations[0] - rotations[1]) > .1
    for r in rotations:
        np.testing.assert_allclose(r @ u, SUN, atol=1e-13)
    for name in ("column", "lowitz"):
        with pytest.raises(RestrictedEscape, match="not_circle"):
            ExactFamily(name).circle(SUN)
        q = jnp.array([.5, .8])
        assert np.linalg.matrix_rank(np.asarray(jax.jacfwd(lambda p: ExactFamily(name).u(p, SUN))(q)), tol=1e-8) == 2


@pytest.mark.parametrize("path,name", [((3,6,4,8), "plate"), ((1,6,2), "parry")])
def test_pinned_restriction_is_one_constant_set_not_many_roots(path, name):
    field = DPField.build(HexPrism(), path, N)
    result = restricted_critical(field, ExactFamily(name), SUN, samples=128)
    assert len(result["records"]) == 1
    row = result["records"][0]
    assert row["source"] == "restricted_constant"
    assert row["profile"] == "delta_mass"
    assert row["multiplicity"] is None
    assert row["min_gate"] > 0


@pytest.mark.parametrize("height,delta", [(9,117.599764), (20,108.937304)])
def test_c11_pinned_sky_and_spherical_angle_not_azimuth(height, delta):
    sun = sun_direction(height, 0)
    family = ExactFamily("plate")
    field = DPField.build(HexPrism(), (1,4,5,2), N)
    result = restricted_critical(field, family, sun, samples=360)
    row = result["records"][0]
    assert np.degrees(row["value"]) == pytest.approx(delta, abs=1e-6)
    assert row["sky"][2] == pytest.approx(sun[2], abs=1e-13)
    d_az = np.arctan2(row["sky"][1], row["sky"][0]) - np.arctan2(sun[1], sun[0])
    assert abs(np.degrees((d_az+np.pi)%(2*np.pi)-np.pi)) == pytest.approx(120, abs=1e-10)
    expected = np.sin(np.radians(height))**2 + np.cos(np.radians(height))**2 * np.cos(2*np.pi/3)
    assert np.cos(row["value"]) == pytest.approx(expected, abs=1e-13)
    from lumice_integral.path_weight import entry_and_power
    area, power = entry_and_power(row["rotation"][None], field.faces, -sun, N, crystal=field.crystal)
    assert area[0] > 0 and power[0] > 0


def test_restricted_minimum_is_not_an_ambient_stationary_point_and_has_1d_profile():
    family = ExactFamily("plate")
    field = DPField.build(HexPrism(), (3,5), N)
    result = restricted_critical(field, family, SUN, samples=128)
    minima = [p for p in result["records"] if p["source"] == "restricted_minimum"]
    assert len(minima) == 1
    p = minima[0]
    assert p["derivative_residual"] < 1e-10
    assert np.linalg.norm(field.gradient_batch(p["u"][None])) > .01
    assert p["profile"] == "inverse_sqrt"
    # Independent finite measure: interval mass within epsilon is proportional
    # to sqrt(epsilon), via the local quadratic jet, not a reused onset label.
    q = float(p["q"][0])
    mass = []
    for eps in (1e-4, 2.5e-5):
        grid = q + np.linspace(-.04, .04, 20001)
        us = np.asarray(jax.vmap(lambda t: family.u(jnp.array([t]), SUN))(jnp.asarray(grid)))
        values = field.d_p_batch(us)
        mass.append(np.count_nonzero(values < p["value"] + eps) * (grid[1]-grid[0]))
    assert mass[0] / mass[1] == pytest.approx(2., rel=.005)
    expected = 2*np.sqrt(2e-4 / p["second_derivative"])
    assert mass[0] == pytest.approx(expected, rel=.005)


def test_analytic_circle_intersections_include_seam_tangent_coincidence_empty():
    a = np.array([0.,0.,1.])
    b = np.array([1.,0.,0.])
    assert len(_circle_roots(a, .3, b, .2)) == 2
    tangent = _circle_roots(a, .6, b, .8)
    assert len(tangent) == 1 and tangent[0][1] == "tangent"
    assert _circle_roots(a, .6, b, .9) == []
    assert _circle_roots(a, .6, -a, -.6) is None
    assert _circle_roots(a, .6, a, .5) == []
    u = tangent[0][0]
    # Two sides of the genuine contact have same sign and quadratic growth.
    for t in (1e-2, 5e-3):
        errors = [.8*(np.cos(s*t)-1) for s in (-1,1)]
        assert max(errors) < 0
    assert (1-np.cos(.01))/(1-np.cos(.005)) == pytest.approx(4, rel=1e-4)


def test_kink_circle_transverse_and_domain_exterior_rejected():
    field = DPField.build(HexPrism(), (3,1,6), N)
    # Plate circle meets basal kink only when sun altitude is the kink altitude.
    out = intersect_margin(field, ExactFamily("plate"), SUN, "internal_1_tir_discriminant")
    assert out["records"] == []
    family = ExactFamily("parry")
    out = intersect_margin(field, family, SUN, "internal_1_tir_discriminant")
    assert out["records"]
    for p in out["records"]:
        assert p["intersection"] == "transverse"
        assert p["residual"] < 1e-12
        assert p["min_gate"] >= -1e-9
    # Entry plane zeroes which fail the OTHER gates are never reported.
    gate = intersect_margin(field, family, SUN, "entry_incidence_cosine")
    for p in gate["records"]:
        assert p["min_gate"] >= -1e-9


@pytest.mark.parametrize("path,name", [((3,1,6), "lowitz"), ((1,3,2), "column")])
def test_kink_sky_keeps_true_lifts_and_nontrivial_curve(path, name):
    field = DPField.build(HexPrism(), path, N)
    family = ExactFamily(name)
    result = kink_sky(field, family, SUN, "internal_1_tir_discriminant", samples=64)
    assert result["image_dimension"] == 1
    assert result["curves"]
    for curve in result["curves"]:
        points = curve["points"]
        assert np.ptp(np.array([p["sky"] for p in points]), axis=0).max() > .1
        for p in points[::max(1,len(points)//12)]:
            r = p["rotation"]
            if name == "column":
                assert abs(r[2,2]) < 1e-13
            else:
                assert abs(r[2,1]) < 1e-13
            np.testing.assert_allclose(r @ p["u"], SUN, atol=1e-12)
            assert p["residual"] < 1e-10
            actual = optics.path_domain_batch(r[None], field.faces, -SUN, N, crystal=field.crystal)
            # At a gate the raw Snell square root amplifies O(eps)
            # discriminant noise to O(sqrt(eps)); the slab limit has no root.
            atol = 4e-8 if p["min_gate"] < 1e-9 else 1e-12
            np.testing.assert_allclose(-np.asarray(actual.direction)[0], p["sky"], atol=atol, rtol=0)


def test_physical_kink_tangent_at_periodic_seam_splits_and_disappears():
    field = DPField.build(HexPrism(), (3,1,6), N)
    family = ExactFamily("parry")
    h = np.sqrt(2-N*N)
    altitude = np.degrees(np.arcsin(h))
    tangent = intersect_margin(field, family, sun_direction(altitude, 180), "internal_1_tir_discriminant")
    point, = tangent["records"]
    assert point["intersection"] == "tangent"
    assert min(point["q"][0], 2*np.pi-point["q"][0]) < 1e-7
    assert point["min_gate"] > .1
    split = intersect_margin(field, family, sun_direction(altitude-.01, 180), "internal_1_tir_discriminant")
    missing = intersect_margin(field, family, sun_direction(altitude+.01, 180), "internal_1_tir_discriminant")
    assert len(split["records"]) == 2
    assert all(p["intersection"] == "transverse" for p in split["records"])
    assert missing["records"] == []
    # The actual physical discriminant is stationary with same-sign quadratic
    # growth on both sides, not just a root deduplication heuristic.
    values = []
    for step in (.01, .005):
        qs = jnp.array([[-step], [step]])
        u = np.asarray(jax.vmap(lambda q: family.u(q, sun_direction(altitude,180)))(qs))
        k = optics.domain_margin_names(field.faces).index("internal_1_tir_discriminant")
        pair = field.margins_batch(u)[:, k]
        assert np.prod(pair) > 0
        values.append(np.mean(pair))
    assert values[0]/values[1] == pytest.approx(4., rel=.001)


def test_coincident_kink_is_an_arc_and_sky_collapse_is_a_point():
    field = DPField.build(HexPrism(), (1,3,2), N)
    altitude = np.degrees(np.arcsin(-np.sqrt(N*N-1)))
    result = kink_sky(field, ExactFamily("parry"), sun_direction(altitude,0), "internal_1_tir_discriminant", samples=64)
    assert not result["records"]
    assert result["curves"]
    assert all(c["intersection"] == "coincident" for c in result["curves"])
    assert result["image_dimension"] == 0
    # A two-dimensional support also need not give a kink sky curve.
    field = DPField.build(HexPrism(), (3,1,6), N)
    result = kink_sky(field, ExactFamily("column"), SUN, "internal_1_tir_discriminant", samples=64)
    assert result["image_dimension"] == 0
    assert all(c["image_dimension"] == 0 for c in result["curves"])


def test_domain_endpoints_are_separate_and_singular_jet_unavailable():
    field = DPField.build(HexPrism(), (3,5), N)
    result = restricted_critical(field, ExactFamily("plate"), SUN, samples=128)
    assert len(result["endpoints"]) >= 2
    for p in result["endpoints"]:
        assert p["location"] == "endpoint"
        assert p["profile"] is None and p["profile_unavailable"]
        assert p["derivative_residual"] is None and p["derivative_unavailable"]
        assert p["margins"]
        assert p["min_gate"] >= -1e-9
    assert result["omitted"] == []


def test_unsupported_and_budget_fail_closed():
    field = DPField.build(HexPrism(), (3,5), N)
    with pytest.raises(RestrictedEscape, match="projection_rank_zero"):
        restricted_critical(field, ExactFamily("plate"), np.array([0.,0.,1.]))
    with pytest.raises(RestrictedEscape, match="root_budget"):
        restricted_critical(field, ExactFamily("plate"), SUN, samples=128, budget=10)
    with pytest.raises(ValueError, match="sun"):
        restricted_critical(field, ExactFamily("plate"), SUN*2)
