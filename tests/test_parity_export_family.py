"""Exact-family parity serialization, replay and independent witness checks.

Only the new kinds are exported here. The full legacy/new matrix is intentionally
left to the serial owner run; the subprocess determinism probe uses one wavelength.
"""
from __future__ import annotations

import ast
import copy
from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import types

import numpy as np
import pytest

from lumice_integral import parity_export as pe
from lumice_integral.family_restricted import RestrictedEscape
from lumice_integral.spectrum.dispersion import refractive_index

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "export_analytic_parity.py"


def matrix_module():
    spec = importlib.util.spec_from_file_location("family_parity_matrix", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def matrix(kind=None):
    module = matrix_module()
    kinds = (kind,) if kind is not None else pe.MC_FAMILY_KINDS
    return [cell for cell in module.MODULE_C_CELLS if set(cell.kinds) & set(kinds)]


def cell(label):
    return next(c for c in [*matrix(), *matrix(pe.MC_FAMILY_TARGET_KIND)] if c.label == label)


@pytest.fixture(scope="module")
def fixtures():
    # All semantic cases, one wavelength: not a second full-matrix export.
    return {
        c.label: pe.build_mc_family_fixture(replace(c, indices=(("550nm", refractive_index(550)),)), {}, c.kinds[0])
        for c in matrix()
    }


def row(fixture):
    result = fixture["expected"]["550nm"]["result"]
    return next(pe._mc_family_points(result))


def test_matrix_is_additive_and_bounded():
    cells = matrix()
    assert len(cells) == 9 <= 10
    assert len({c.name for c in cells}) == 9
    assert sum(len(c.kinds) for c in cells) == 9
    assert set(pe.MC_FAMILY_KINDS) <= set(pe.MC_KINDS)
    for c in cells:
        assert c.indices == tuple((f"{nm}nm", refractive_index(nm)) for nm in (400, 550, 700))
        assert c.restricted_samples in (32, 64)
        assert c.kinds[0] in pe._MC_BUILDERS and c.kinds[0] in pe.VERIFIERS
        assert c.detects[c.kinds[0]]
    with pytest.raises(ValueError, match="needs at least one"):
        pe.MCCell("missing", pe.prism_crystal(.5), (3, 5), ("family_critical",))


def test_written_fixtures_read_back_and_have_no_unexplained_nulls(fixtures, tmp_path):
    for label, fixture in fixtures.items():
        path = tmp_path / f"{label}.json"
        pe.write_json(path, fixture)
        text = path.read_text()
        assert "NaN" not in text and "Infinity" not in text and "null" not in text
        check = pe.verify_fixture(path)
        assert not check.failures, (label, check.failures)
        assert check.compared > 10


def test_semantics_include_nonempty_gate_true_sky_empty_and_named_refusal(fixtures):
    for altitude in (9, 20):
        point = row(fixtures[f"mc_c11_plate_sun{altitude}"])
        h = np.radians(altitude)
        assert np.cos(point["value"]) == pytest.approx(np.sin(h)**2 + np.cos(h)**2 * np.cos(2*np.pi/3), abs=1e-12)
        assert point["source"] == "restricted_constant"
        assert point["profile"] == "delta_mass"
        assert point["multiplicity"]["available"] is False
        assert point["multiplicity"]["reason"]
    assert row(fixtures["mc_family_plate_morse"])["profile"] == "inverse_sqrt"
    assert row(fixtures["mc_family_parry_constant"])["source"] == "restricted_constant"
    gate = fixtures["mc_family_gate"]["expected"]["550nm"]["result"]
    assert gate["kind"] == "kind_2" and gate["records"]
    kink = fixtures["mc_family_kink_intersection"]["expected"]["550nm"]["result"]
    assert kink["kind"] == "kind_3" and len(kink["records"]) == 2
    assert all(p["intersection"] == "transverse" for p in kink["records"])
    sky = fixtures["mc_family_kink_sky"]["expected"]["550nm"]["result"]
    assert sky["image_dimension"] == 1 and len(sky["curves"]) == 2
    for curve in sky["curves"]:
        assert np.ptp(np.array([p["sky"] for p in curve["points"]]), axis=0).max() > .1
    empty = fixtures["mc_family_empty_sky"]["expected"]["550nm"]
    assert empty["available"] is True and empty["status"] == "computed"
    assert empty["result"]["records"] == empty["result"]["curves"] == []
    assert empty["result"]["image_dimension"] == {"available": False, "reason": "empty sky image has no dimension"}
    refusal = fixtures["mc_family_sunpole_refusal"]["expected"]["550nm"]
    assert refusal["status"] == "refused" and refusal["available"] is False
    assert refusal["regime"] == "projection_rank_zero" and refusal["reason"]


@pytest.mark.parametrize("mutation", ["value", "roll", "tangent", "sky", "rotation", "u", "intersection", "refusal", "status", "availability", "nested_availability", "coverage", "missing_lambda", "extra_key", "type", "curve_sheet", "curve_count", "curve_closed", "reason", "tolerance"])
def test_tampering_is_rejected(fixtures, mutation):
    label = "mc_family_kink_sky" if mutation.startswith("curve_") else "mc_family_kink_intersection"
    if mutation in ("refusal", "status", "reason"):
        label = "mc_family_sunpole_refusal"
    if mutation == "nested_availability":
        label = "mc_family_empty_sky"
    fixture = copy.deepcopy(fixtures[label])
    result = fixture["expected"]["550nm"]
    if mutation == "roll":
        fixture["input"]["exact_family"]["roll_rad"] += .1
    elif mutation == "value":
        row(fixture)["value"] += .01
    elif mutation in ("sky", "u"):
        row(fixture)[mutation][0] += .01
    elif mutation in ("tangent", "rotation"):
        row(fixture)[mutation][0][0] += .01
    elif mutation == "intersection":
        row(fixture)["intersection"] = "tangent"
    elif mutation == "refusal":
        result["regime"] = "root_budget"
    elif mutation == "reason":
        result["reason"] = "pretend refusal"
    elif mutation == "status":
        result["status"] = "computed"
    elif mutation == "availability":
        result["available"] = False
    elif mutation == "nested_availability":
        result["result"]["image_dimension"]["available"] = True
    elif mutation == "coverage":
        result["result"]["coverage"] = "certified_global_exhaustion"
    elif mutation == "missing_lambda":
        del fixture["expected"]["550nm"]
    elif mutation == "extra_key":
        row(fixture)["invented"] = 1
    elif mutation == "type":
        row(fixture)["value"] = str(row(fixture)["value"])
    elif mutation == "curve_sheet":
        result["result"]["curves"][0]["sheet"] += 1
    elif mutation == "curve_count":
        result["result"]["curves"].pop()
    elif mutation == "curve_closed":
        result["result"]["curves"][0]["closed"] = not result["result"]["curves"][0]["closed"]
    elif mutation == "tolerance":
        fixture["tolerance"]["numeric"]["value"] = 1.0
    assert pe.verify_mc_family(fixture, mutation).failures


@pytest.mark.parametrize("observable", ["value", "sky", "u", "rotation", "tangent"])
def test_forward_witness_checks_do_not_depend_on_replay_agreement(fixtures, monkeypatch, observable):
    fixture = copy.deepcopy(fixtures["mc_family_kink_intersection"])
    point = row(fixture)
    if observable == "value":
        point[observable] += .01
    elif observable in ("sky", "u"):
        point[observable][0] += .01
    else:
        point[observable][0][0] += .01
    # Simulate a shared failure of the exporter and recompute path. Forward optics,
    # Ru=sun and central differences must still reject the corrupted witness.
    monkeypatch.setattr(pe, "_mc_family_result", lambda *args: copy.deepcopy(fixture["expected"]["550nm"]))
    check = pe.verify_mc_family(fixture, "independent-witness")
    assert check.failures
    assert any("forward optics" in failure or "Ru=sun" in failure or "finite difference tangent" in failure for failure in check.failures)


def test_per_wavelength_refusal_does_not_hide_computed_siblings(monkeypatch):
    import lumice_integral.family_restricted as restricted

    original = restricted.restricted_critical

    def mixed(field, *args, **kwargs):
        if field.index == refractive_index(550):
            raise RestrictedEscape("root_budget", "explicit test budget refusal")
        return original(field, *args, **kwargs)

    monkeypatch.setattr(restricted, "restricted_critical", mixed)
    c = cell("mc_family_plate_morse")
    fixture = pe.build_mc_family_critical_fixture(c, {})
    assert [fixture["expected"][label]["status"] for label in ("400nm", "550nm", "700nm")] == ["computed", "refused", "computed"]
    assert set(fixture["expected"]) == set(fixture["input"]["indices"])
    assert not pe.verify_mc_family(fixture).failures
    broken = copy.deepcopy(fixture)
    del broken["expected"]["550nm"]
    assert pe.verify_mc_family(broken).failures


def test_algorithm_errors_are_not_refusals_or_manifest_skips(monkeypatch, tmp_path):
    import lumice_integral.family_restricted as restricted
    from lumice_integral.dp_field import TopologyEscape

    c = replace(cell("mc_family_plate_morse"), indices=(("550nm", refractive_index(550)),))
    for error in (ValueError("algorithm bug"), TopologyEscape("upstream algorithm failure")):
        def fail(*args, **kwargs):
            raise error
        monkeypatch.setattr(restricted, "restricted_critical", fail)
        with pytest.raises(type(error), match=str(error)):
            pe.export_module_c_cell(c, tmp_path, {})
    with pytest.raises(ValueError, match="unexplained unavailable"):
        pe._mc_family_json({"unknown": None})
    with pytest.raises(ValueError, match="non-finite"):
        pe._mc_family_json({"value": float("nan")})


def test_new_kinds_byte_deterministic_in_two_fresh_interpreters(tmp_path):
    # All four new kinds and a refusal, only n(550): no repeated full matrix.
    program = '''
import importlib.util
from dataclasses import replace
from pathlib import Path
import sys
from lumice_integral import parity_export as p
from lumice_integral.spectrum.dispersion import refractive_index
spec = importlib.util.spec_from_file_location("matrix", sys.argv[1])
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
labels = {"mc_family_plate_morse", "mc_family_kink_intersection", "mc_family_kink_sky", "mc_family_sunpole_refusal", "mc_c16_column_transverse"}
for c in m.MODULE_C_CELLS:
    if c.label in labels:
        p.export_module_c_cell(replace(c, indices=(("550nm", refractive_index(550)),)), Path(sys.argv[2]), p.fixture_provenance())
'''
    env = {**os.environ, "JAX_PLATFORMS": "cpu", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"}
    directories = [tmp_path / "one", tmp_path / "two"]
    for directory in directories:
        directory.mkdir()
        subprocess.run([sys.executable, "-c", program, str(SCRIPT), str(directory)], cwd=ROOT, env=env, check=True, timeout=90)
    names = sorted(p.name for p in directories[0].glob("*.json"))
    assert len(names) == 5
    assert names == sorted(p.name for p in directories[1].glob("*.json"))
    assert {json.loads((directories[0] / name).read_text())["fixture_kind"] for name in names} == {
        *pe.MC_FAMILY_KINDS, pe.MC_FAMILY_TARGET_KIND,
    }
    for name in names:
        assert (directories[0] / name).read_bytes() == (directories[1] / name).read_bytes()


def test_old_builder_implementations_and_representative_fixture_unchanged():
    # Compare against this worktree's HEAD, not another checkout/editable install.
    # This pins every old builder's implementation without re-running the 120-file matrix.
    old_source = subprocess.check_output(["git", "show", "HEAD:src/lumice_integral/parity_export.py"], cwd=ROOT, text=True)
    new_source = (ROOT / "src" / "lumice_integral" / "parity_export.py").read_text()
    old_functions = {node.name: ast.dump(node, include_attributes=False) for node in ast.parse(old_source).body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    new_functions = {node.name: ast.dump(node, include_attributes=False) for node in ast.parse(new_source).body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    for name, body in old_functions.items():
        if name != "export_module_c_cell":
            assert new_functions[name] == body, name
    old_script = subprocess.check_output(["git", "show", "HEAD:scripts/export_analytic_parity.py"], cwd=ROOT, text=True)
    old_matrix = types.ModuleType("parity_matrix_before_family")
    old_matrix.__file__ = str(SCRIPT)
    exec(compile(old_script, str(SCRIPT), "exec"), old_matrix.__dict__)
    new_matrix = matrix_module()
    assert old_matrix.DETECTS == new_matrix.DETECTS
    assert repr(old_matrix.matrix()) == repr(new_matrix.matrix())
    for key in ("EDGE_CELLS", "BAND_SUM_CELLS", "MODULE_C_CELLS"):
        old_cells, new_cells = getattr(old_matrix, key), getattr(new_matrix, key)
        assert repr(old_cells) == repr(new_cells[:len(old_cells)]), key
    legacy = types.ModuleType("lumice_integral._parity_before_family")
    legacy.__file__ = str(ROOT / "src" / "lumice_integral" / "parity_export.py")
    legacy.__package__ = "lumice_integral"
    sys.modules[legacy.__name__] = legacy
    try:
        exec(compile(old_source, legacy.__file__, "exec"), legacy.__dict__)
        old_cell = next(c for c in old_matrix.MODULE_C_CELLS if c.faces == (3, 1, 6) and c.kinds == ("dp_field_kinks",))
        new_cell = next(c for c in new_matrix.MODULE_C_CELLS if c.name == old_cell.name)
        before = legacy.build_mc_field_kinks_fixture(old_cell, {})
        after = pe.build_mc_field_kinks_fixture(new_cell, {})
        assert pe.dumps(after) == legacy.dumps(before)
    finally:
        del sys.modules[legacy.__name__]


@pytest.fixture(scope="module")
def target_fixtures():
    return {
        c.label: pe.build_mc_family_target_fixture(replace(c, indices=(("550nm", refractive_index(550)),)), {})
        for c in matrix(pe.MC_FAMILY_TARGET_KIND) if c.label != "mc_c16_column_coincident"
    }


def test_target_matrix_and_real_local_topology(target_fixtures, tmp_path):
    cells = matrix(pe.MC_FAMILY_TARGET_KIND)
    assert len(cells) == 4
    for c in cells:
        assert c.indices == tuple((f"{nm}nm", refractive_index(nm)) for nm in (400, 550, 700))
        assert c.detects[pe.MC_FAMILY_TARGET_KIND]
    for label, fixture in target_fixtures.items():
        path = tmp_path / f"{label}.json"
        pe.write_json(path, fixture)
        assert "null" not in path.read_text()
        assert not pe.verify_fixture(path).failures
        outputs = fixture["expected"]["550nm"]["results"]
        assert outputs["intersection"]["coverage"] == "local_numerical_not_certified"
        assert outputs["intersection"]["completeness"] == "unknown"
    transverse = target_fixtures["mc_c16_column_transverse"]["expected"]["550nm"]["results"]["intersection"]
    assert len(transverse["records"]) == 2
    assert all(p["intersection"] == "transverse" for p in transverse["records"])
    fold = target_fixtures["mc_c16_column_fold"]["expected"]["550nm"]["results"]
    assert fold["locator"]["records"][0]["intersection"] == "tangent"
    assert fold["intersection"]["records"][0]["intersection"] == "tangent"
    assert len(fold["locator"]["records"][0]["corrected_offsets"]) == 4
    parry = target_fixtures["mc_c16_parry_lowitz"]["expected"]["550nm"]["results"]
    assert parry["intersection"]["records"][0]["intersection"] == "isolated"
    assert parry["lowitz"]["records"][0]["constraint"] == "roll_lock"
    false_hit = parry["projection_false_positive"]
    assert false_hit["family"] == "lowitz"
    assert false_hit["witness"]["q"][1] == pytest.approx(np.pi / 2)
    assert false_hit["witness"]["status"] == "rejected"
    assert false_hit["witness"]["reason"] == "full_sky_direction_mismatch"
    np.testing.assert_allclose(np.dot(false_hit["target"], false_hit["sun"]),
                               np.dot(false_hit["witness"]["sky"], false_hit["sun"]), atol=1e-12)


@pytest.mark.parametrize("mutation", ["type", "jet", "tangent_sign", "offset", "rank", "weight", "reason", "coverage", "availability", "root_count", "roll"])
def test_target_tampering_rejected(target_fixtures, monkeypatch, mutation):
    source = "mc_c16_parry_lowitz" if mutation == "reason" else "mc_c16_column_fold"
    fixture = copy.deepcopy(target_fixtures[source])
    if mutation != "roll":
        # Pure comparison mutations use the already independently replayed real
        # result, avoiding 11 repeated JAX solves of an identical target.
        baseline = copy.deepcopy(fixture["expected"]["550nm"])
        monkeypatch.setattr(pe, "_mc_target_result", lambda *args: copy.deepcopy(baseline))
    outputs = fixture["expected"]["550nm"]["results"]
    point = outputs["intersection"]["records"][0]
    if mutation == "type":
        point["so3_rank"] = float(point["so3_rank"])
    elif mutation == "jet":
        point["second_derivative"] += .01
    elif mutation == "tangent_sign":
        point["fiber_tangent"] = [-x for x in point["fiber_tangent"]]
    elif mutation == "offset":
        point["corrected_offsets"][0]["constraint"] += .01
    elif mutation == "rank":
        point["so3_rank"] = 1
    elif mutation == "weight":
        point["A"] *= 1.01
    elif mutation == "reason":
        outputs["projection_false_positive"]["witness"]["reason"] = "near_enough"
    elif mutation == "coverage":
        outputs["intersection"]["completeness"] = "complete"
    elif mutation == "availability":
        fixture["expected"]["550nm"]["available"] = False
    elif mutation == "root_count":
        outputs["intersection"]["records"].clear()
    elif mutation == "roll":
        # Column has no locked roll; change the exact support to a different
        # valid family/roll instead of making ExactFamily validation throw.
        fixture["input"]["exact_family"] = {"name": "lowitz", "roll_rad": .1}
    assert pe.verify_mc_family_target(fixture).failures


def test_target_forward_check_survives_shared_replay_corruption(target_fixtures, monkeypatch):
    fixture = copy.deepcopy(target_fixtures["mc_c16_parry_lowitz"])
    point = fixture["expected"]["550nm"]["results"]["intersection"]["records"][0]
    point["sky"][0] += .01
    point["A"] *= 1.01
    monkeypatch.setattr(pe, "_mc_target_result", lambda *args: copy.deepcopy(fixture["expected"]["550nm"]))
    failures = pe.verify_mc_family_target(fixture).failures
    assert any("witness" in failure and ("sky" in failure or ".A" in failure) for failure in failures)


def test_target_cross_wavelength_keeps_its_own_locator_and_topology():
    c = replace(cell("mc_c16_column_fold"), indices=(("400nm", refractive_index(400)), ("700nm", refractive_index(700))))
    fixture = pe.build_mc_family_target_fixture(c, {})
    assert not pe.verify_mc_family_target(fixture).failures
    results = fixture["expected"]
    for label in ("400nm", "700nm"):
        assert results[label]["results"]["locator"]["refractive_index"] == fixture["input"]["indices"][label]["refractive_index"]
    assert np.linalg.norm(np.asarray(results["400nm"]["results"]["locator"]["target"]) - results["700nm"]["results"]["locator"]["target"]) > .001
    swapped = copy.deepcopy(fixture)
    swapped["expected"]["400nm"], swapped["expected"]["700nm"] = swapped["expected"]["700nm"], swapped["expected"]["400nm"]
    assert pe.verify_mc_family_target(swapped).failures


# slow: one closed SO(3) trace plus check_target_pose on every curve node takes ~52 s.
@pytest.mark.slow
def test_target_coincidence_curve_is_replayed_point_by_point(tmp_path):
    c = replace(cell("mc_c16_column_coincident"), indices=(("550nm", refractive_index(550)),))
    fixture = pe.build_mc_family_target_fixture(c, {})
    curve, = fixture["expected"]["550nm"]["results"]["intersection"]["curves"]
    assert curve["closed"] and curve["intersection"] == "coincident"
    assert curve["dimension"] == 1 and curve["component_completeness"] == "unknown"
    assert len(curve["points"]) > 20
    path = tmp_path / "coincident.json"
    pe.write_json(path, fixture)
    assert not pe.verify_fixture(path).failures
