"""Red-state evidence for the output-level conformance checks (``docs/phase1-math-contract.md`` section 11.1).

Each check that certifies another backend by its outputs alone is shown to fail on the defect it exists
to catch: a backend output (``got``) is built from LI's own, correct output and then damaged the way a
faulty backend would damage it, and the fixture's recipe must reject it.  Implementation-level injections
(the same defects planted in LI's solver, then reverted) are recorded in the task's progress log; the
section 11.1 table names both for every rewritten row.

The last test keeps that table honest: every row C01-C21 appears once, and every test name it cites exists.
"""

from __future__ import annotations

import copy
import importlib.util
import re
from pathlib import Path

import numpy as np
import pytest

from lumice_integral import parity_export as pe
from lumice_integral.so3 import exp, rotation_distances

REPO = Path(__file__).resolve().parent.parent
CONTRACT = REPO / "docs" / "phase1-math-contract.md"


def _export_script():
    spec = importlib.util.spec_from_file_location("export_analytic_parity", REPO / "scripts" / "export_analytic_parity.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# The edge cells without a seed search (no store to build): the solver-only rows of section 11.1.
FAST_EDGE_CELLS = ("3-5__short_loop", "3-5__limits", "3-5__rank_loss_extremum", "3-5__antipodal_target")


@pytest.fixture(scope="module")
def exported(tmp_path_factory) -> Path:
    script = _export_script()
    directory = tmp_path_factory.mktemp("edge")
    cells = [cell for cell in script.EDGE_CELLS if cell.name in FAST_EDGE_CELLS]
    matrix = [cell for cell in script.matrix() if cell.name == "3-5-6-7__near_boundary"]
    matrix = [pe.Cell(cell.scene, cell.category, cell.rationale, store_n=None, seed_search_skip="not needed here") for cell in matrix]
    pe.export_matrix(matrix, directory, cells)
    return directory


def _fixture(directory: Path, name: str) -> dict:
    return copy.deepcopy(pe.read_json(directory / name))


def _traces_check(fixture: dict, got: list[dict]) -> pe.Check:
    check = pe.Check("tampered")
    pe._compare_traces(check, got, fixture)
    return check


def _fails(check: pe.Check, message: str) -> None:
    assert any(message in failure for failure in check.failures), check.failures


def test_the_edge_fixtures_read_back_unchanged(exported) -> None:
    checks = pe.verify_directory(exported)
    assert all(not check.failures for check in checks), [(check.fixture, check.failures) for check in checks if check.failures]


def test_edge_traces_end_the_way_their_rationale_says(exported) -> None:
    def ends(name: str) -> list[tuple[str, str]]:
        return sorted((t["status"], t["reason"]) for t in pe.read_json(exported / name)["expected"]["traces"])

    assert ends("3-5__short_loop__trace_fiber.json") == [("closed", "closed_loop")]
    assert ends("3-5__rank_loss_extremum__trace_fiber.json") == [("event_terminated", "rank_loss")] * 2
    assert ends("3-5__antipodal_target__trace_fiber.json") == [("event_terminated", "chart_boundary")] * 2
    for reason, status in (
        ("step_budget", "budget_exhausted"),
        ("arclength_budget", "budget_exhausted"),
        ("evaluation_budget", "budget_exhausted"),
        ("corrector_failure", "numerical_failure"),
        ("step_underflow", "numerical_failure"),
    ):
        assert ends(f"3-5__limits__trace_fiber__{reason}.json") == [(status, reason)] * 2
    # The short loop is traced once: its length is the loop's, not a multiple (the 0.01 deg level of section 6.1)
    short = pe.read_json(exported / "3-5__short_loop__trace_fiber.json")["expected"]["traces"][0]
    assert 0.16 < short["arclength"] < 0.17
    # The rank-deficient seed: J_perp at rounding level, no accepted pose
    point = pe.read_json(exported / "3-5__rank_loss_extremum__evaluate_path__point.json")["expected"]
    assert point["jacobian_available"] and point["normal_jacobian"] < 1e-12 and point["singular_values"][1] < 1e-12


# --- EvaluatePath: J_perp, singular values, branch margins, failed gate (C05, C07, C08, C14) ------------------------


@pytest.mark.parametrize(
    "name, edit, message",
    [
        ("3-5__short_loop__evaluate_path__curve_min_jacobian.json", lambda e: e.__setitem__("normal_jacobian", 0.5 * e["normal_jacobian"]), "normal_jacobian"),
        ("3-5__short_loop__evaluate_path__curve_min_jacobian.json", lambda e: e["singular_values"].__setitem__(1, e["singular_values"][1] * (1 + 1e-9)), "singular_values"),
        ("3-5__short_loop__evaluate_path__point.json", lambda e: e.__setitem__("jacobian_available", False), "jacobian_available"),
        (
            "3-5-6-7__near_boundary__evaluate_path__point.json",
            lambda e: e["branch_margins"].__setitem__("internal_2_incidence_cosine", -e["branch_margins"]["internal_2_incidence_cosine"]),
            "branch_margins",
        ),
        ("3-5-6-7__near_boundary__evaluate_path__outside.json", lambda e: e["failed_gate"].__setitem__("name", "exit_incidence_cosine"), "failed_gate"),
        ("3-5-6-7__near_boundary__evaluate_path__outside.json", lambda e: e["failed_gate"].__setitem__("value", 1e-9 - e["failed_gate"]["value"]), "failed_gate"),
    ],
)
def test_evaluate_path_rejects_a_wrong_pointwise_observable(exported, name, edit, message) -> None:
    fixture = _fixture(exported, name)
    edit(fixture["expected"])
    _fails(pe.verify_evaluate_path(fixture, name), message)


# --- TraceFiber per-pose arrays: self-consistency and regularity (C05, C14) -----------------------------------------


def _roll_jacobian(trace: dict) -> None:
    trace["normal_jacobian"] = list(np.roll(trace["normal_jacobian"], 1))  # the Jacobian of the previous pose


def _halve_jacobian(trace: dict) -> None:
    trace["normal_jacobian"] = [0.5 * value for value in trace["normal_jacobian"]]


def _flip_margin(trace: dict) -> None:
    margins = np.asarray(trace["branch_margins"])
    margins[len(margins) // 2, 0] *= -1.0
    trace["branch_margins"] = margins.tolist()


def _drop_availability(trace: dict) -> None:
    trace["jacobian_available"] = [False] + trace["jacobian_available"][1:]


def _short_array(trace: dict) -> None:
    trace["normal_jacobian"] = trace["normal_jacobian"][:-1]


@pytest.mark.parametrize(
    "edit, message",
    [
        (_roll_jacobian, "normal_jacobian"),
        (_halve_jacobian, "normal_jacobian"),
        (_flip_margin, "non-positive margin"),
        (_drop_availability, "unavailable"),
        (_short_array, "per-pose arrays"),
    ],
)
def test_trace_arrays_must_match_the_backends_own_evaluate_path(exported, edit, message) -> None:
    fixture = _fixture(exported, "3-5__short_loop__trace_fiber.json")
    got = copy.deepcopy(fixture["expected"]["traces"])
    assert not _traces_check(fixture, got).failures
    edit(got[0])
    _fails(_traces_check(fixture, got), message)


# --- closure on the first traversal, and controller perturbations (C06) --------------------------------------------


def _twice_around(trace: dict) -> None:
    """A backend that passed its seed once and closed on the second return (the retired absolute gate)."""
    poses = np.asarray(trace["poses"])
    trace["poses"] = np.concatenate((poses, poses[1:])).tolist()
    trace["arclength"] *= 2.0


@pytest.mark.parametrize("name", ["3-5__short_loop__trace_fiber.json", "3-5__short_loop__trace_fiber__initial_step_0.08.json"])
def test_a_short_loop_traversed_twice_is_rejected(exported, name) -> None:
    fixture = _fixture(exported, name)
    got = copy.deepcopy(fixture["expected"]["traces"])
    _twice_around(got[0])
    for trace_key in ("normal_jacobian", "singular_values", "branch_margins", "jacobian_available"):
        got[0][trace_key] = list(got[0][trace_key]) + list(got[0][trace_key])[1:]
    _fails(_traces_check(fixture, got), "arclength")


def test_a_short_loop_closed_at_its_far_side_is_rejected(exported) -> None:
    """C12 on a real path: the far side of the 0.165 rad loop lies within closure_distance of the seed and
    crosses its section with the reversed tangent; a backend without the tangent gate closes there."""
    fixture = _fixture(exported, "3-5__short_loop__trace_fiber.json")
    got = copy.deepcopy(fixture["expected"]["traces"])
    half = len(got[0]["poses"]) // 2
    for key in ("poses", "crystal_frame_sun_directions", "residual_norms", "tangents", "normal_jacobian", "singular_values", "branch_margins", "jacobian_available"):
        got[0][key] = list(got[0][key])[: half + 1]
    got[0]["arclength_increments"] = got[0]["arclength_increments"][:half]
    got[0]["arclength"] = float(np.sum(got[0]["arclength_increments"]))
    seed = np.asarray(fixture["input"]["seed_pose"]).reshape(3, 3)
    assert min(rotation_distances(seed, np.asarray(got[0]["poses"]).reshape(-1, 3, 3)[-3:])) < fixture["input"]["continuation"]["closure_distance"]
    check = _traces_check(fixture, got)
    _fails(check, "arclength")
    _fails(check, "curve distance")


def test_a_perturbed_trace_that_leaves_the_curve_is_rejected(exported) -> None:
    fixture = _fixture(exported, "3-5__short_loop__trace_fiber__controller_thresholds.json")
    got = copy.deepcopy(fixture["expected"]["traces"])
    turn = np.asarray(exp(np.array([0.0, 0.02, 0.0])))
    got[0]["poses"] = (np.asarray(got[0]["poses"]).reshape(-1, 3, 3) @ turn).reshape(-1, 9).tolist()
    _fails(_traces_check(fixture, got), "curve distance")


# --- budgets, corrector failure and step underflow (C09, C11) ------------------------------------------------------


def _limits(exported, reason: str) -> tuple[dict, list[dict]]:
    fixture = _fixture(exported, f"3-5__limits__trace_fiber__{reason}.json")
    return fixture, copy.deepcopy(fixture["expected"]["traces"])


def _drop_last_pose(trace: dict) -> None:
    for key in ("poses", "crystal_frame_sun_directions", "residual_norms", "tangents", "normal_jacobian", "singular_values", "branch_margins", "jacobian_available"):
        trace[key] = list(trace[key])[:-1]
    trace["arclength"] -= trace["arclength_increments"][-1]
    trace["arclength_increments"] = trace["arclength_increments"][:-1]


def test_the_limit_fixtures_read_back_as_exported(exported) -> None:
    for reason in ("step_budget", "arclength_budget", "evaluation_budget", "corrector_failure", "step_underflow"):
        fixture, got = _limits(exported, reason)
        assert not _traces_check(fixture, got).failures, reason


def test_step_budget_counts_the_seed_plus_one_pose_per_accepted_step(exported) -> None:
    fixture, got = _limits(exported, "step_budget")
    assert len(got[0]["poses"]) == fixture["input"]["continuation"]["maximum_accepted_steps"] + 1
    _drop_last_pose(got[0])  # a backend that stops one step early (or counts the seed as a step)
    _fails(_traces_check(fixture, got), "step_budget with")


def test_arclength_budget_stops_where_the_next_edge_would_cross_it(exported) -> None:
    fixture, got = _limits(exported, "arclength_budget")
    options = fixture["input"]["continuation"]
    while got[1]["arclength"] > options["maximum_arclength"] - options["maximum_advance"]:
        _drop_last_pose(got[1])  # stopped more than one maximum advance short of the budget
    _fails(_traces_check(fixture, got), "arclength_budget at")
    fixture, got = _limits(exported, "arclength_budget")
    got[0]["arclength"] = options["maximum_arclength"] * (1 + 1e-9)  # an edge past the budget was accepted
    _fails(_traces_check(fixture, got), "arclength_budget at")


def test_budget_geometry_must_lie_on_the_unbudgeted_curve(exported) -> None:
    fixture, got = _limits(exported, "evaluation_budget")
    turn = np.asarray(exp(np.array([0.03, 0.0, 0.0])))
    got[0]["poses"] = (np.asarray(got[0]["poses"]).reshape(-1, 3, 3) @ turn).reshape(-1, 9).tolist()
    _fails(_traces_check(fixture, got), "from the unbudgeted curve")


@pytest.mark.parametrize(
    "fixture_reason, reported",
    [
        ("step_underflow", ("budget_exhausted", "step_budget")),  # underflow reported as a budget
        ("step_underflow", ("numerical_failure", "corrector_failure")),  # the retry that underflowed, not the underflow
        ("corrector_failure", ("numerical_failure", "step_underflow")),
        ("evaluation_budget", ("budget_exhausted", "step_budget")),
    ],
)
def test_terminal_reasons_are_distinguished(exported, fixture_reason, reported) -> None:
    fixture, got = _limits(exported, fixture_reason)
    got[0]["status"], got[0]["reason"] = reported
    _fails(_traces_check(fixture, got), "status/reason")


# --- rank loss and the antipodal target (C04, C07) -----------------------------------------------------------------


def _accept_the_seed(fixture: dict, got: list[dict]) -> None:
    """A backend that accepted the seed and closed a degenerate loop on it."""
    seed = fixture["input"]["seed_pose"]
    got[1:] = []
    got[0].update(
        status="closed",
        reason="closed_loop",
        poses=[seed],
        crystal_frame_sun_directions=[[0.0, 0.0, 1.0]],
        arclength_increments=[],
        residual_norms=[0.0],
        tangents=[[1.0, 0.0, 0.0]],
        arclength=0.0,
        branch_margins=[[1.0] * len(got[0]["branch_margin_names"])],
        jacobian_available=[True],
        normal_jacobian=[1.0],
        singular_values=[[1.0, 1.0]],
    )


@pytest.mark.parametrize("name", ["3-5__rank_loss_extremum__trace_fiber.json", "3-5__antipodal_target__trace_fiber.json"])
def test_a_seed_that_must_be_rejected_cannot_be_traced(exported, name) -> None:
    fixture = _fixture(exported, name)
    got = copy.deepcopy(fixture["expected"]["traces"])
    assert all(len(trace["poses"]) == 0 for trace in got)
    _accept_the_seed(fixture, got)
    check = _traces_check(fixture, got)
    _fails(check, "status/reason")
    _fails(check, "curve distance")


# --- the section 11.1 table --------------------------------------------------------------------------------------


def _section_11_1() -> str:
    text = CONTRACT.read_text()
    start = text.index("### 11.1 ")
    return text[start : text.index("\n## ", start)]


def test_section_11_1_has_every_row_once_and_cites_existing_tests() -> None:
    section = _section_11_1()
    rows = re.findall(r"^\| (C\d\d) \|", section, flags=re.MULTILINE)
    assert rows == [f"C{i:02d}" for i in range(1, 22)]
    names = set(re.findall(r"`(test_[a-z0-9_]+)`", section))
    assert names
    defined = set()
    for path in (REPO / "tests").glob("test_*.py"):
        defined |= set(re.findall(r"^def (test_[a-z0-9_]+)", path.read_text(), flags=re.MULTILINE))
    assert names <= defined, sorted(names - defined)
    script = _export_script()
    cells = {cell.name for cell in script.EDGE_CELLS} | {cell.name for cell in script.matrix()}
    cited = set(re.findall(r"`((?:\d+-)+\d+__[a-z0-9_.]+)`", section))
    assert cited and {name.split("__")[0] + "__" + name.split("__")[1] for name in cited} <= cells, sorted(cited)
