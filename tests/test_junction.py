"""S6 witnesses: raw optical equations and analytic models, not self-replay."""
from dataclasses import replace

import numpy as np
import pytest

from lumice_integral import junction as J, optics
from lumice_integral.canonical_scene import canonical_crystal
from lumice_integral.dp_field import DPField
from lumice_integral.dp_field.boundary import Walker
from lumice_integral.family_restricted import ExactFamily
from lumice_integral.spectrum.dispersion import refractive_index

N = float(refractive_index(550.))
GATE = "kind_2:entry_incidence_cosine"
KINK = "kind_3:internal_1_tir_discriminant"


@pytest.fixture(scope="module")
def fields():
    return {p: DPField.build(canonical_crystal(), p, N) for p in
            ((3, 5), (3, 1, 6), (3, 5, 6, 7), (3, 5, 6, 7, 3), (4, 8, 1, 7, 5))}


def _source(field, id):
    return next(s for s in J.structural_sources(field) if s.id == id)


def _pair(field, ids, **kwargs):
    return J.enumerate_junctions(field, pairs=[ids], **kwargs)["pairs"][0]


def _witness(field, result):
    names = optics.domain_margin_names(field.faces)
    for point in result["records"]:
        u = point["u"]
        assert np.linalg.norm(u) == pytest.approx(1., abs=1e-12)
        margins = field.margins_batch(u[None])[0]
        assert min(field.validity_margins_batch(u[None])[0]) >= -1e-13
        for source in point["sources"]:
            if source.startswith(("kind_2:", "kind_3:")):
                assert abs(margins[names.index(source.split(":", 1)[1])]) < 1e-11
        assert np.isfinite(point["value"])


def test_existing_corner_points_match_without_moving_them(fields):
    field = fields[(3, 5)]
    result = _pair(field, (GATE, "kind_2:exit_snell_discriminant"))
    assert len(result["records"]) == 2
    _witness(field, result)
    for corner in field.corners:
        assert min(np.linalg.norm(corner.position-p["u"]) for p in result["records"]) < 1e-10


def test_gate_kink_uses_physical_reflecting_branch(fields):
    field = fields[(3, 1, 6)]
    result = _pair(field, (GATE, KINK))
    assert result["status"] == "computed"
    assert len(result["records"]) == 2
    _witness(field, result)
    for p in result["records"]:
        # Analytic basal TIR circle z=-sqrt(n^2-1), not the wrong sign branch.
        assert p["u"][2] == pytest.approx(-np.sqrt(N*N-1), abs=1e-12)
        assert p["intersection"] == "transverse"


def test_distinct_internal_events_really_cross(fields):
    field = fields[(4, 8, 1, 7, 5)]
    result = _pair(field, (KINK, "kind_3:internal_2_tir_discriminant"))
    assert len(result["records"]) == 1
    _witness(field, result)
    p = result["records"][0]
    # Independent M1 scipy 3-equation root, not a stored Newton seed.
    np.testing.assert_allclose(p["u"], [-.2824757039268523, .44883472231481775, -.8477941193094051], atol=1e-10)
    assert p["differential"]["available"]
    assert min(p["differential"]["singular_values"]) > .2
    assert p["min_gate"] > .02


def test_coincident_exit_and_kink_are_one_dimensional_not_points(fields):
    field = fields[(3, 5, 6, 7)]
    result = _pair(field, ("kind_2:exit_snell_discriminant", KINK))
    assert result["status"] == "continuous"
    assert not result["records"]
    assert result["curves"]
    for c in result["curves"]:
        assert c["dimension"] == 1
        assert c["evidence"] == "analytic_equal_squared_incidence"
        raw = field.margins_batch(c["points"])
        names = optics.domain_margin_names(field.faces)
        assert abs(raw[:, names.index("internal_1_tir_discriminant")]).max() < 1e-11


def test_margin_identity_keeps_both_source_events(fields):
    result = _pair(fields[(3, 5, 6, 7, 3)], (KINK, "kind_3:internal_3_tir_discriminant"))
    assert result["status"] == "continuous"
    assert not result["records"]
    assert result["sources"] == [KINK, "kind_3:internal_3_tir_discriminant"]


def test_budget_exhaustion_and_failure_are_not_computed_empty(fields):
    field = fields[(4, 8, 1, 7, 5)]
    ids = (KINK, "kind_3:internal_2_tir_discriminant")
    for options, reason in ((dict(budget=0), "budget_exhausted"), (dict(max_iterations=0), "iteration_limit"),
                            (dict(max_step=1e-15), "step_limit")):
        result = _pair(field, ids, **options)
        assert result["status"] == "unresolved"
        assert not result["records"]
        assert reason in {d["reason"] for d in result["diagnostics"]}


def test_reordered_and_duplicated_pairs_are_deduplicated(fields):
    field = fields[(3, 1, 6)]
    result = J.enumerate_junctions(field, pairs=[(GATE,KINK),(KINK,GATE),(GATE,KINK)])
    assert len(result["pairs"]) == 1
    assert len(result["pairs"][0]["records"]) == 2
    assert result["exhaustive"] is False
    assert "corridor_A_zero" in result["omitted"]


def test_upstream_failure_cannot_disappear_behind_success(fields):
    field = fields[(3, 1, 6)]
    a = replace(_source(field, GATE), failed_seeds=1)
    result = J.intersect_sources(field, a, _source(field, KINK))
    assert result["records"]
    assert any(d["reason"] == "upstream_failed_seeds" for d in result["diagnostics"])
    assert result["status"] == "partial"


def test_missing_upstream_arc_is_not_a_global_no_intersection(fields):
    field = fields[(3, 5)]
    a = replace(_source(field, GATE), arcs=(), plane=None)
    b = _source(field, "kind_2:exit_snell_discriminant")
    result = J.intersect_sources(field, a, b)
    assert result["status"] != "disjoint"
    assert not result["records"]


def test_family_adapter_keeps_every_lift_and_continuous_image(fields):
    field = fields[(3, 1, 6)]
    sun = np.array([np.cos(.3),0.,np.sin(.3)])
    family = ExactFamily("parry")
    result = _pair(field, (KINK, "family:parry"), family=family, sun=sun)
    assert result["records"]
    for p in result["records"]:
        np.testing.assert_allclose(p["rotation"] @ p["u"], sun, atol=1e-9)
    family = ExactFamily("column")
    result = _pair(field, (KINK, "family:column"), family=family, sun=sun)
    assert result["status"] == "continuous"
    assert result["curves"] and not result["records"]
    assert {c["sheet"] for c in result["curves"]} == {0,1}
    for c in result["curves"]:
        assert c["dimension"] == 1
        for p in c["points"]:
            np.testing.assert_allclose(p["rotation"] @ p["u"], sun, atol=1e-9)


def test_analytic_family_circles_disjoint_and_sun_pole_refusal(fields):
    field = fields[(3, 1, 6)]
    family = ExactFamily("plate")
    # Fixed positive z cannot meet the negative-z basal onset circle.
    sun = np.array([np.cos(.3),0.,np.sin(.3)])
    result = _pair(field, (KINK,"family:plate"), family=family, sun=sun)
    assert not result["records"] and not result["curves"]
    # The adapter distinguishes an algebraic empty model from numerical search.
    assert result["status"] == "disjoint"
    result = _pair(field, (KINK,"family:plate"), family=family, sun=np.array([0.,0.,1.]))
    assert result["status"] == "unsupported"
    assert result["diagnostics"][0]["reason"] == "projection_rank_zero"


def test_kind1_point_family_member_and_nonmember(fields):
    field = fields[(3,5)]
    sources = [s for s in J.structural_sources(field) if s.kind == "kind_1" and s.point is not None]
    assert sources
    source = sources[0]
    family = ExactFamily("plate")
    sun = np.asarray(family.rotation(np.array([.72]))) @ source.point
    result = J.intersect_family(field, source, family, sun)
    assert result["status"] == "computed" and result["records"]
    np.testing.assert_allclose(result["records"][0]["rotation"] @ source.point, sun, atol=1e-9)
    other = np.array([np.cos(.2),0.,np.sin(.2)])
    assert not J.intersect_family(field, source, family, other)["records"]


def test_known_critical_circle_adapter_is_not_gradient_squared(fields):
    field = fields[(3,1,6)]
    source = _source(field, "kind_1:crease")
    assert source.margin is None
    assert source.plane is not None
    family = ExactFamily("column")
    sun = np.array([np.cos(.3),0.,np.sin(.3)])
    result = J.intersect_family(field, source, family, sun)
    assert result["status"] == "unsupported"
    assert result["diagnostics"][0]["reason"] == "critical_stratum_adapter"


def _circle_source(id, a, h):
    a = np.asarray(a,float)
    e = np.cross(a, np.eye(3)[np.argmin(abs(a))])
    e /= np.linalg.norm(e)
    t = np.linspace(0,2*np.pi,3601)
    points = h*a + np.sqrt(1-h*h)*(np.cos(t)[:,None]*e+np.sin(t)[:,None]*np.cross(a,e))
    return J.Source(id,"test",1,(points,), ("analytic_test",), plane=(a,h))


def test_analytic_disjoint_and_near_double_root_are_distinct(fields):
    field = fields[(3,1,6)]
    a = _circle_source("x", [1,0,0],.9)
    b = _circle_source("z", [0,0,1],-.9)
    assert J.intersect_sources(field,a,b)["status"] == "disjoint"
    # x=.6, z~-0.8 yields two extremely close but distinct y roots. Both
    # lie strictly inside this optical domain and must survive analytic dedup.
    a = _circle_source("x",[1,0,0],.6)
    b = _circle_source("z",[0,0,1],-np.sqrt(.64-1e-12))
    result = J.intersect_sources(field,a,b)
    assert len(result["records"]) == 2
    assert {np.sign(p["u"][1]) for p in result["records"]} == {-1,1}


def test_nonlocal_or_wrong_arc_analytic_root_is_rejected(fields):
    field = fields[(3,1,6)]
    a = _circle_source("x",[1,0,0],.6)
    b = _circle_source("z",[0,0,1],-.7)
    a = replace(a,arcs=(a.points[:3],))
    result = J.intersect_sources(field,a,b)
    assert not result["records"]
    assert result["status"] == "unresolved"
    assert all(d["reason"] == "outside_source_arc_or_raw_residual" for d in result["diagnostics"])


def test_invalid_configuration_is_not_normal_empty(fields):
    field = fields[(3,1,6)]
    with pytest.raises(ValueError,match="budget"):
        _pair(field,(GATE,KINK),budget=-1)
    with pytest.raises(KeyError):
        _pair(field,(GATE,"unknown"))
