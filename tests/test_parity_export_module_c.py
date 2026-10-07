"""Module C parity fixtures (``docs/analytic-parity-fixtures.md`` section 6.3).

The fast tier: canonical strict JSON, the read-back of a written fixture, the cheap in-memory anchor
pins (the kink sweeps and the chromatic verdicts) and the mutation self-check that every numeric
anchor pin discriminates.  The slow tier: two full module C exports in independent interpreters byte
for byte, the read-back of all of them and the anchor table on the exported files (AC2: the 52
SUMMARY's prose anchors as regression assertions, at the full Lumice n(lambda) caliber of
``progress.md``'s DECISION 2026-10-07 19:44 table).
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from lumice_integral import chromatic as C
from lumice_integral import parity_export as pe
from lumice_integral.spectrum.dispersion import refractive_index

REPO = Path(__file__).resolve().parent.parent
N400, N450, N550, N650, N700 = (refractive_index(nm) for nm in (400.0, 450.0, 550.0, 650.0, 700.0))
BETA = pe.prism_crystal(3.0, (2, 1, 1, 2, 1, 1))
PRISM_H1 = pe.prism_crystal(0.5)
RHOMBIC = pe.prism_crystal(0.5, (1.5, 1, 1, 1.5, 1, 1))


def cell_3_1_6_kinks() -> pe.MCCell:
    return pe.MCCell("mc_kinks", pe.prism_crystal(1.0), (3, 1, 6), ("dp_field_kinks",), (("550nm", N550),))


def cell_3_1_5_chromatic() -> pe.MCCell:
    return pe.MCCell("mc_chromatic", PRISM_H1, (3, 1, 5), ("chromatic_diagnose",), ())


# ------------------------------------------------------------------ fast: format, read-back, cheap anchors
def test_module_c_cell_naming_and_validation() -> None:
    cell = pe.MCCell("mc_field", pe.prism_crystal(1.0), (3, 5), ("dp_field_sample",), (("550nm", N550),))
    assert cell.name == "3-5__mc_field" and cell.path_id == "3-5"
    with pytest.raises(ValueError, match="unknown module C fixture kinds"):
        pe.MCCell("x", pe.prism_crystal(1.0), (3, 5), ("nope",), (("550nm", N550),))
    with pytest.raises(ValueError, match="at least one"):
        pe.MCCell("x", pe.prism_crystal(1.0), (3, 5), ("dp_field_sample",), ())


def test_module_c_json_is_canonical_strict_and_reads_back(tmp_path: Path) -> None:
    fixture = pe.build_mc_field_kinks_fixture(cell_3_1_6_kinks(), pe.fixture_provenance())
    path = tmp_path / "3-1-6__mc_kinks__dp_field_kinks.json"
    pe.write_json(path, fixture)
    text = path.read_text()
    assert json.dumps(json.loads(text), indent=1, sort_keys=True, allow_nan=False) + "\n" == text
    assert "NaN" not in text and "Infinity" not in text
    check = pe.verify_fixture(path)
    assert not check.failures, check.failures


def test_thresholds_snapshot_matches_the_chromatic_module() -> None:
    snapshot = pe.mc_thresholds_snapshot()
    assert snapshot["n_red"] == C.N_RED and snapshot["n_blue"] == C.N_BLUE
    assert snapshot["edge_min_shift_rad"] == C.EDGE_MIN_SHIFT_RAD
    assert snapshot["tint_ratio_min"] == C.TINT_RATIO_MIN
    assert "authority" in snapshot


def _rim(index: float) -> float:
    """The dark-hole rim: ``D = 2 asin sqrt(n^2 - 1)`` on the constant kink (chromatic-module-c.md section 3)."""
    return 2.0 * math.asin(math.sqrt(index * index - 1.0))


def test_constant_kink_of_3_1_6_is_the_closed_form() -> None:
    curves = pe.build_mc_field_kinks_fixture(cell_3_1_6_kinks(), pe.fixture_provenance())["expected"]["kinks"]["550nm"]
    (curve,) = curves
    assert curve["method"] == "great_circle" and curve["spread"] < 1e-12
    value = math.degrees(curve["value_min"])
    assert abs(value - math.degrees(_rim(N550))) <= 1e-9  # the closed form at the fixture's n, to the printed digits
    assert abs(value - 115.9451002) <= 5e-7  # the 52 SUMMARY caliber, full n(550) (its 115.945094 was n = 1.3110129)
    assert abs(math.degrees(_rim(1.3110129)) - 115.945094) <= 1e-6  # provenance: the truncated-n record (a truncation, not a rounding)


def test_chromatic_verdicts_of_the_two_diagnose_cells() -> None:
    verdict_316 = pe.build_mc_chromatic_fixture(pe.MCCell("mc_chromatic", PRISM_H1, (3, 1, 6), ("chromatic_diagnose",), ()), pe.fixture_provenance())["expected"]
    assert (verdict_316["kind"], verdict_316["color"], verdict_316["visible"]) == ("edge", "blue", True)
    (edge,) = verdict_316["features"]
    assert edge["source"] == "internal_1_tir_discriminant" and edge["spread"] < 1e-12
    assert abs(edge["delta_red"] - _rim(C.N_RED)) <= 1e-12 and abs(edge["delta_blue"] - _rim(C.N_BLUE)) <= 1e-12
    assert abs(math.degrees(edge["shift"]) - 3.354) <= 5e-4  # chromatic-module-c.md section 3, printed digits
    verdict_315 = pe.build_mc_chromatic_fixture(cell_3_1_5_chromatic(), pe.fixture_provenance())["expected"]
    assert (verdict_315["kind"], verdict_315["color"], verdict_315["visible"]) == ("edge", "blue", True)
    kink, gate = verdict_315["features"]
    assert kink["kind"] == "edge" and (kink["color"], kink["visible"]) == ("blue", True)
    assert abs(math.degrees(kink["shift"]) - 7.325) <= 5e-4 and abs(math.degrees(kink["spread"]) - 4.878) <= 5e-4
    assert gate["kind"] == "gate_edge" and (gate["color"], gate["visible"]) == ("red", False)
    assert math.degrees(gate["spread"]) > 90.0 and abs(math.degrees(gate["shift"])) < 1.0  # sigma 108.7 vs Delta -0.33


# ------------------------------------------------------------------ the anchor table (values, degrees)
def _constant_kink(fixture: dict, label: str) -> float:
    return math.degrees(next(curve["value_min"] for curve in fixture["expected"]["kinks"][label] if curve["spread"] is not None and curve["spread"] < 1e-12))


def _kink_span(fixture: dict, label: str) -> tuple[float, float]:
    (curve,) = [curve for curve in fixture["expected"]["kinks"][label] if curve["spread"] is not None]
    return math.degrees(curve["value_min"]), math.degrees(curve["value_max"])


# (label, getter on the export dir, expected, tolerance); the mutation test perturbs expected by 2 x tolerance
DEGREE_ANCHORS: tuple[tuple[str, str, object, float, float], ...] = (
    ("C05 partition lower break", "4-8-7-5__mc_field__dp_field_topology.json", lambda d: math.degrees(d["expected"]["interval_partition"][1][0]), 50.1617417, 5e-7),
    ("C05 blade critical value", "4-8-7-5__mc_field__dp_field_topology.json", lambda d: next(math.degrees(v) for v in d["expected"]["critical_values"] if abs(math.degrees(v) - 120.0) < 1e-6), 120.0, 1e-8),
    ("C06 constant kink 550nm", "4-8-1-7-5__mc_kinks__dp_field_kinks.json", lambda d: _constant_kink(d, "550nm"), 149.246753, 5e-7),
    ("C06 constant kink 400nm", "4-8-1-7-5__mc_kinks__dp_field_kinks.json", lambda d: _constant_kink(d, "400nm"), 150.495980, 5e-7),
    ("C06 constant kink 700nm", "4-8-1-7-5__mc_kinks__dp_field_kinks.json", lambda d: _constant_kink(d, "700nm"), 148.645351, 5e-7),
    ("3-1-4-5 partition split", "3-1-4-5__mc_field__dp_field_topology.json", lambda d: math.degrees(d["expected"]["interval_partition"][1][0] if isinstance(d["expected"]["interval_partition"][1], list) else d["expected"]["interval_partition"][1]["lower"]), 120.0, 5e-7),
    ("3-4-1-5 partition split", "3-4-1-5__mc_field__dp_field_topology.json", lambda d: math.degrees(d["expected"]["interval_partition"][1][0] if isinstance(d["expected"]["interval_partition"][1], list) else d["expected"]["interval_partition"][1]["lower"]), 120.0, 5e-7),
    ("3-1-5 boundary D_min (closed form)", "3-1-5__mc_field__dp_field_topology.json", lambda d: math.degrees(min(p["value"] for p in d["expected"]["boundary"]["critical_points"])), math.degrees(2.0 * math.asin(N550 * math.sin(math.radians(30.0))) - math.radians(60.0)), 1e-8),
    ("3-5-6-7 D limit", "3-5-6-7__mc_field__dp_field_topology.json", lambda d: math.degrees(d["expected"]["interval_partition"][0][0]), 50.1617417, 5e-7),
    ("wavelength table inner edge 450nm", "3-5__mc_field__wavelength_critical_table.json", lambda d: next(row["values_deg"]["450nm"] for row in d["expected"]["onsets"] if row["source"] == "interior_minimum"), 22.271639, 5e-7),
    ("wavelength table inner edge 550nm", "3-5__mc_field__wavelength_critical_table.json", lambda d: next(row["values_deg"]["550nm"] for row in d["expected"]["onsets"] if row["source"] == "interior_minimum"), 21.916127, 5e-7),
    ("wavelength table inner edge 650nm", "3-5__mc_field__wavelength_critical_table.json", lambda d: next(row["values_deg"]["650nm"] for row in d["expected"]["onsets"] if row["source"] == "interior_minimum"), 21.690677, 5e-7),
    ("wavelength table inner-edge displacement (task 42.4: 0.581)", "3-5__mc_field__wavelength_critical_table.json", lambda d: next(row["displacement_deg"] for row in d["expected"]["onsets"] if row["source"] == "interior_minimum"), 0.580962, 5e-7),
    ("C02 sweep 700nm low", "3-1-5__mc_kinks__dp_field_kinks.json", lambda d: _kink_span(d, "700nm")[0], 129.364361, 5e-7),
    ("C02 sweep 700nm high", "3-1-5__mc_kinks__dp_field_kinks.json", lambda d: _kink_span(d, "700nm")[1], 134.255299, 5e-7),
    ("C02 sweep 550nm low", "3-1-5__mc_kinks__dp_field_kinks.json", lambda d: _kink_span(d, "550nm")[0], 132.458136, 5e-7),
    ("C02 sweep 550nm high", "3-1-5__mc_kinks__dp_field_kinks.json", lambda d: _kink_span(d, "550nm")[1], 136.842378, 5e-7),
    ("C02 sweep 400nm low", "3-1-5__mc_kinks__dp_field_kinks.json", lambda d: _kink_span(d, "400nm")[0], 141.003938, 5e-7),
    ("C02 sweep 400nm high", "3-1-5__mc_kinks__dp_field_kinks.json", lambda d: _kink_span(d, "400nm")[1], 143.652221, 5e-7),
)


def _pin(label: str, got: float, expected: float, tolerance: float) -> None:
    assert abs(got - expected) <= tolerance, f"{label}: {got!r} vs {expected!r} (tolerance {tolerance:g})"


def test_anchor_pins_discriminate() -> None:
    """Mutation self-check: a wrong value beyond the tolerance never passes a pin (learnings: verification-rigor)."""
    for label, _, getter, expected, tolerance in DEGREE_ANCHORS:
        got = expected + 2.0 * tolerance  # a drift one tolerance-band beyond the pin
        with pytest.raises(AssertionError, match=re.escape(label)):
            _pin(label, got, expected, tolerance)


# ------------------------------------------------------------------ slow: full export, determinism, anchors
pytestmark_slow = pytest.mark.slow

MC_CELL_NAMES = (
    "3-5__mc_field", "3-1-5__mc_field", "3-1-5__mc_kinks", "3-1-6__mc_field", "3-1-6__mc_kinks",
    "3-1-4-5__mc_field", "3-4-1-5__mc_field", "3-5-6-7__mc_field", "4-8-7-5__mc_field",
    "4-8-1-7-5__mc_kinks", "3-1-6__mc_chromatic", "3-1-5__mc_chromatic",
    "1-3-5-2__mc_chromatic_class", "1-3-4-2__mc_chromatic_class", "3-5-6-8__mc_chromatic_class",
)


def _export_module_c(output: Path) -> subprocess.CompletedProcess:
    env = {**os.environ, "JAX_PLATFORMS": "cpu"}
    return subprocess.run(
        [sys.executable, str(REPO / "scripts" / "export_analytic_parity.py"), "--output-dir", str(output), "--cells", *MC_CELL_NAMES],
        cwd=REPO, env=env, capture_output=True, text=True, check=True,
    )


# slow: two full module C exports (~80 s each on an M2 Max; the three class cells dominate) plus their
# byte comparison, the read-back of all 26 fixtures and the anchor table on the exported files
@pytest.fixture(scope="module")
def module_c_export(tmp_path_factory) -> Path:
    output = tmp_path_factory.mktemp("module-c") / "first"
    _export_module_c(output)
    return output


@pytest.mark.slow
def test_module_c_export_is_byte_deterministic_and_reads_back(tmp_path: Path, module_c_export: Path) -> None:
    second = tmp_path / "second"
    _export_module_c(second)
    names = sorted(path.name for path in module_c_export.iterdir())
    assert names == sorted(path.name for path in second.iterdir())
    for name in names:
        assert (module_c_export / name).read_bytes() == (second / name).read_bytes(), name
    checks = pe.verify_directory(module_c_export)
    assert len(checks) == len(names) - 1  # every fixture, manifest excluded
    assert all(not check.failures for check in checks), [(check.fixture, check.failures) for check in checks if check.failures]


@pytest.mark.slow
def test_module_c_manifest_lists_every_cell_with_detects_and_no_orphans(module_c_export: Path) -> None:
    manifest = pe.read_json(module_c_export / pe.MANIFEST)
    cells = manifest["module_c_cells"]
    assert [cell["name"] for cell in cells] == list(MC_CELL_NAMES)
    listed = {name for cell in cells for name in cell["files"]}
    on_disk = {path.name for path in module_c_export.glob("*.json")} - {pe.MANIFEST}
    assert listed == on_disk
    for cell in cells:
        assert cell["serves"] and cell["rationale"], cell["name"]
        assert all(cell["detects"][name.split("__")[-1].removesuffix(".json")] for name in cell["files"]), cell["name"]


@pytest.mark.slow
def test_module_c_anchor_table_on_the_exported_fixtures(module_c_export: Path) -> None:
    for label, filename, getter, expected, tolerance in DEGREE_ANCHORS:
        _pin(label, getter(pe.read_json(module_c_export / filename)), expected, tolerance)
    # dispersion of the C06 circle (the corpus anchor +1.87 deg; the 52 record +1.851)
    kinks = pe.read_json(module_c_export / "4-8-1-7-5__mc_kinks__dp_field_kinks.json")
    dispersion = _constant_kink(kinks, "400nm") - _constant_kink(kinks, "700nm")
    _pin("C06 dispersion 400-700", dispersion, 1.850629, 5e-7)
    assert abs(dispersion - 1.851) <= 5e-4  # the 52 SUMMARY record, printed digits
    # the corpus C02 anchor 131.030 deg lies inside the 700 nm (red-end) sweep span
    low, high = _kink_span(kinks := pe.read_json(module_c_export / "3-1-5__mc_kinks__dp_field_kinks.json"), "700nm")
    assert low + 1e-3 <= 131.030 <= high - 1e-3
    # C05: both intervals (2, 0, 2), the classify jacobian@blade pair, the degenerate fold
    topology = pe.read_json(module_c_export / "4-8-7-5__mc_field__dp_field_topology.json")
    assert [interval[2:] for interval in topology["expected"]["interval_partition"]] == [[2, 0, 2], [2, 0, 2]]
    fold = topology["expected"]["degenerate_fold"]
    assert fold is not None and abs(fold["circle_interior_fraction"] - 0.22764) <= 1e-4 and fold["crease_interior_arcs"] >= 1
    classification = pe.read_json(module_c_export / "4-8-7-5__mc_field__focusing_classify.json")["expected"]
    assert classification["mechanism"] == "jacobian"
    near_blade = [onset for onset in classification["onsets"] if abs(onset["value_deg"] - 120.0) < 1e-6]
    assert {(onset["source"], onset["profile"], onset["jacobian_focusing"]) for onset in near_blade} == {
        ("boundary_extremum", "degenerate", True), ("slab_circle", "inverse_sqrt_divergence", True),
    }
    # 3-1-6: the whole-range partition and 3-5-6-7: the machinery regression cell
    slab = pe.read_json(module_c_export / "3-1-6__mc_field__dp_field_topology.json")["expected"]
    assert len(slab["interval_partition"]) == 1
    assert math.degrees(slab["interval_partition"][0][0]) <= 1e-6 and math.degrees(slab["interval_partition"][0][1]) >= 180.0 - 1e-6
    regression = pe.read_json(module_c_export / "3-5-6-7__mc_field__dp_field_topology.json")["expected"]
    assert [interval[2:] for interval in regression["interval_partition"]] == [[2, 0, 2], [1, 0, 1]]
    assert regression["domain_topology"]["domain_components"] == 1 and regression["domain_topology"]["is_disk"]
    assert regression["domain_topology"]["grid_audit"]["verdict"] == "corrected"
    regression_kinks = pe.read_json(module_c_export / "3-5-6-7__mc_field__dp_field_kinks.json")["expected"]["kinks"]["550nm"]
    assert all(curve["nonfinite_values_dropped"] == 0 for curve in regression_kinks)
    # the plate class cells: the printed tint digits and the lit sets (chromatic-module-c.md section 3)
    for representative, ratio, kind, color in ((("1-3-5-2"), 1.492, "tint", "blue"), (("1-3-4-2"), 0.965, "none", "white"), (("3-5-6-8"), 1.029, "none", "white")):
        verdict = pe.read_json(module_c_export / f"{representative}__mc_chromatic_class__chromatic_class.json")["expected"]
        assert (verdict["verdict"]["kind"], verdict["verdict"]["color"]) == (kind, color), representative
        assert abs(verdict["verdict"]["tint"]["ratio"] - ratio) <= 6e-4, representative
    assert len(pe.read_json(module_c_export / "1-3-5-2__mc_chromatic_class__chromatic_class.json")["expected"]["lit_members"]["red"]) == 4
