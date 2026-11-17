"""S6 additive parity, exact status replay, and independent original-function witnesses."""
from __future__ import annotations

import copy
from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from lumice_integral import parity_export as pe
from lumice_integral.spectrum.dispersion import refractive_index

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "export_analytic_parity.py"


def cells():
    spec = importlib.util.spec_from_file_location("s6_matrix", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return [c for c in module.MODULE_C_CELLS if pe.MC_JUNCTION_KIND in c.kinds]


def cell(label):
    return next(c for c in cells() if c.label == label)


@pytest.fixture(scope="module")
def fixtures():
    # One wavelength per semantic case; the owner performs the complete serial export.
    return {c.label: pe.build_mc_junction_fixture(replace(c, indices=(("550nm", refractive_index(550)),)), {})
            for c in cells()}


def result(fixture):
    return fixture["expected"]["550nm"]


def pair(fixture):
    return result(fixture)["pairs"][0]


def point(fixture):
    return pair(fixture)["records"][0]


def test_additive_registry_and_bounded_matrix():
    matrix = cells()
    assert len(matrix) == 7
    assert len({c.name for c in matrix}) == 7
    assert pe.MC_JUNCTION_KIND in pe.MC_EXACT_KINDS
    assert pe.MC_JUNCTION_KIND in pe._MC_BUILDERS and pe.MC_JUNCTION_KIND in pe.VERIFIERS
    assert pe.SCHEMA_VERSION == 1
    for c in matrix:
        assert c.indices == tuple((f"{nm}nm", refractive_index(nm)) for nm in (400, 550, 700))
        assert c.junction_pairs and c.detects["junction"]
    with pytest.raises(ValueError, match="needs at least one"):
        pe.MCCell("missing", pe.prism_crystal(.5), (3, 5), ("junction",))


def test_real_semantics_and_nonexhaustive_coverage(fixtures):
    for fixture in fixtures.values():
        value = result(fixture)
        assert value["exhaustive"] is False
        assert value["coverage"] == "opportunistic_existing_strata_only"
        assert value["omitted"]
        assert value["options"] == fixture["input"]["options"]
    assert len(pair(fixtures["mc_s6_corner"])["records"]) == 2
    assert len(pair(fixtures["mc_s6_gate_kink"])["records"]) == 2
    kink = pair(fixtures["mc_s6_kink_pair"])
    assert kink["records"]
    assert set(kink["sources"]) == {"kind_3:internal_1_tir_discriminant", "kind_3:internal_2_tir_discriminant"}
    assert all(p["intersection"] == "transverse" for p in kink["records"])
    continuous = pair(fixtures["mc_s6_continuous"])
    assert continuous["status"] == "continuous" and continuous["curves"] and not continuous["records"]
    assert all(c["intersection"] == "coincident" and c["dimension"] == 1 for c in continuous["curves"])
    column = pair(fixtures["mc_s6_column"])
    assert column["status"] == "continuous" and column["curves"] and not column["records"]
    assert {curve["sheet"] for curve in column["curves"]} == {0, 1}
    empty = pair(fixtures["mc_s6_plate_disjoint"])
    assert empty["status"] == "disjoint" and empty["coverage"] == "analytic_circles_disjoint"
    assert not empty["records"] and not empty["curves"]
    failed = pair(fixtures["mc_s6_iteration_failure"])
    assert failed["status"] == "unresolved" and failed["diagnostics"] and not failed["records"]
    assert failed["attempted"] > 0 and failed["rejected"] == failed["attempted"]


def test_point_sets_do_not_depend_on_seed_order(fixtures):
    fixture = copy.deepcopy(fixtures["mc_s6_corner"])
    pair(fixture)["records"].reverse()
    assert not pe.verify_mc_junction(fixture).failures


def test_json_roundtrip_and_original_witnesses(fixtures, tmp_path):
    for label, fixture in fixtures.items():
        path = tmp_path / f"{label}.json"
        pe.write_json(path, fixture)
        assert "NaN" not in path.read_text() and "Infinity" not in path.read_text()
        check = pe.verify_fixture(path)
        assert not check.failures, (label, check.failures)
        assert check.compared > 20


@pytest.mark.parametrize("mutation", [
    "position", "source", "interaction", "coverage", "global_coverage", "diagnostics", "budget",
    "iteration_budget", "availability", "index", "wavelength", "missing_index", "empty_indices", "tolerance",
    "tolerance_basis", "exhaustive", "extra_field", "numeric_type", "curve_position", "curve_rotation",
])
def test_tampering_rejected(fixtures, mutation):
    label = "mc_s6_iteration_failure" if mutation == "diagnostics" else "mc_s6_kink_pair"
    if mutation.startswith("curve_"):
        label = "mc_s6_column"
    fixture = copy.deepcopy(fixtures[label])
    value, first = result(fixture), pair(fixture)
    if mutation == "position":
        point(fixture)["u"][0] += .01
    elif mutation == "source":
        first["sources"][0] = "kind_3:internal_3_tir_discriminant"
    elif mutation == "interaction":
        point(fixture)["intersection"] = "tangent"
    elif mutation == "coverage":
        first["coverage"] = "complete"
    elif mutation == "global_coverage":
        value["coverage"] = "complete"
    elif mutation == "diagnostics":
        first["diagnostics"] = []
    elif mutation == "budget":
        fixture["input"]["options"]["budget"] += 1
    elif mutation == "iteration_budget":
        fixture["input"]["options"]["max_iterations"] = True
    elif mutation == "availability":
        point(fixture)["differential"]["available"] = False
    elif mutation == "index":
        fixture["input"]["indices"]["550nm"]["refractive_index"] = 1.31
    elif mutation == "wavelength":
        fixture["input"]["indices"]["550nm"]["wavelength_nm"] = 551.0
    elif mutation == "missing_index":
        del fixture["expected"]["550nm"]
    elif mutation == "empty_indices":
        fixture["input"]["indices"] = {}
        fixture["expected"] = {}
    elif mutation == "tolerance":
        fixture["tolerance"]["geometry"]["value"] += 1e-12  # must compare exactly, not with its own tolerance
    elif mutation == "tolerance_basis":
        fixture["tolerance"]["geometry"]["basis"] = "invented"
    elif mutation == "exhaustive":
        value["exhaustive"] = True
    elif mutation == "extra_field":
        first["invented"] = 1
    elif mutation == "numeric_type":
        point(fixture)["u"][0] = str(point(fixture)["u"][0])
    elif mutation == "curve_position":
        first["curves"][0]["points"][0]["u"][0] += .01
    elif mutation == "curve_rotation":
        first["curves"][0]["points"][0]["rotation"][0][0] += .01
    assert pe.verify_mc_junction(fixture, mutation).failures


@pytest.mark.parametrize("mutation", ["margin", "sphere", "rotation", "pose", "coverage"])
def test_original_witness_rejects_shared_wrong_export_and_replay(fixtures, monkeypatch, mutation):
    label = "mc_s6_column" if mutation in ("rotation", "pose") else "mc_s6_kink_pair"
    fixture = copy.deepcopy(fixtures[label])
    p = pair(fixture)["curves"][0]["points"][0] if label == "mc_s6_column" else point(fixture)
    if mutation == "margin":
        u = np.asarray(p["u"]) + np.array([.01, .02, -.01])
        p["u"] = (u/np.linalg.norm(u)).tolist()
    elif mutation == "sphere":
        p["u"] = (np.asarray(p["u"])*1.01).tolist()
    elif mutation == "rotation":
        p["rotation"][0][0] += .01
    elif mutation == "pose":
        p["q"][0] += .01
    else:
        result(fixture)["coverage"] = "complete"
    monkeypatch.setattr(pe, "_mc_junction_result", lambda *args: copy.deepcopy(result(fixture)))
    check = pe.verify_mc_junction(fixture, "independent-original-witness")
    assert check.failures
    assert any("raw" in failure or "coverage" in failure for failure in check.failures)


def test_defined_family_refusal_is_data_and_algorithm_errors_propagate(monkeypatch, tmp_path):
    import lumice_integral.junction as junction
    from lumice_integral.dp_field import TopologyEscape
    from lumice_integral.family_restricted import RestrictedEscape

    c = replace(cell("mc_s6_column"), indices=(("550nm", refractive_index(550)),))
    def restricted(*args, **kwargs):
        raise RestrictedEscape("root_budget", "explicit bounded test refusal")
    monkeypatch.setattr(junction, "intersect_margin", restricted)
    fixture = pe.build_mc_junction_fixture(c, {})
    refusal = pair(fixture)
    assert refusal["status"] == "unsupported"
    assert refusal["diagnostics"] == [{"reason": "root_budget", "detail": "root_budget: explicit bounded test refusal"}]
    assert not pe.verify_mc_junction(fixture).failures
    for error in (ValueError("algorithm bug"), TopologyEscape("upstream algorithm failure")):
        def fail(*args, **kwargs):
            raise error
        monkeypatch.setattr(junction, "intersect_margin", fail)
        with pytest.raises(type(error), match=str(error)):
            pe.export_module_c_cell(c, tmp_path, {})


def test_nonfinite_is_not_unavailable():
    for value in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="non-finite junction"):
            pe._mc_junction_json({"u": [value, 0., 1.]})
        for key in ("tangent", "singular_values"):
            with pytest.raises(ValueError, match="non-finite junction"):
                pe._mc_junction_json({key: [value]})
    with pytest.raises(ValueError, match="unexplained unavailable"):
        pe._mc_junction_json({"invented": None})


@pytest.mark.slow  # slow: real three-wavelength field construction and independent readback
@pytest.mark.parametrize("label", ["mc_s6_kink_pair", "mc_s6_iteration_failure"])
def test_complete_dispersion_recomputed_independently(label):
    fixture = pe.build_mc_junction_fixture(cell(label), {})
    assert set(fixture["expected"]) == {"400nm", "550nm", "700nm"}
    for name, spectral in fixture["input"]["indices"].items():
        assert fixture["expected"][name]["index"] == spectral["refractive_index"] == refractive_index(spectral["wavelength_nm"])
        first = fixture["expected"][name]["pairs"][0]
        assert first["records"] if label == "mc_s6_kink_pair" else first["diagnostics"]
    assert not pe.verify_mc_junction(fixture).failures


@pytest.mark.slow  # slow: two serial fresh JAX interpreters; owner runs without concurrent JAX work
def test_same_revision_bytes_in_two_fresh_interpreters(tmp_path):
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
for c in m.MODULE_C_CELLS:
    if "junction" in c.kinds:
        p.export_module_c_cell(replace(c, indices=(("550nm", refractive_index(550)),)), Path(sys.argv[2]), p.fixture_provenance())
'''
    env = {**os.environ, "JAX_PLATFORMS": "cpu", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"}
    directories = [tmp_path / "one", tmp_path / "two"]
    for directory in directories:
        directory.mkdir()
        subprocess.run([sys.executable, "-c", program, str(SCRIPT), str(directory)], cwd=ROOT, env=env, check=True, timeout=180)
    names = sorted(p.name for p in directories[0].glob("*.json"))
    assert len(names) == 7
    assert names == sorted(p.name for p in directories[1].glob("*.json"))
    for name in names:
        assert json.loads((directories[0] / name).read_text())["fixture_kind"] == "junction"
        assert (directories[0] / name).read_bytes() == (directories[1] / name).read_bytes()
