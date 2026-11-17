"""Analytic sphere intersections and controlled legacy operation-order checks."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from lumice_integral.spherical_newton import PairRefinement, refine_pair


def _unit(u: np.ndarray) -> np.ndarray:
    return u / np.linalg.norm(u)


def _planes(normals: np.ndarray, offsets: np.ndarray):
    def evaluate(u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return normals @ u - offsets, normals

    def residual(u: np.ndarray) -> float:
        # An independent scalar evaluation, not evaluate(u)[0].
        return max(abs(float(n @ u - offset)) for n, offset in zip(normals, offsets, strict=True))

    return evaluate, residual


def _legacy_refine(
    u: np.ndarray,
    evaluate: Callable,
    residual: Callable,
    *,
    max_iterations: int = 30,
    max_step: float = 1e-3,
) -> np.ndarray:
    """Frozen pre-extraction loop; test-only reference for callback and operation order."""
    best = u
    best_residual = residual(u)
    for _ in range(max_iterations):
        m, j = evaluate(u)
        axis = np.eye(3)[int(np.argmin(np.abs(u)))]
        e1 = _unit(np.cross(u, axis))
        e2 = np.cross(u, e1)
        jac = np.array([[j[0] @ e1, j[0] @ e2], [j[1] @ e1, j[1] @ e2]])
        if abs(np.linalg.det(jac)) < 1e-14:
            break
        step = np.linalg.solve(jac, -np.array([m[0], m[1]]))
        if np.linalg.norm(step) > max_step:
            break
        u = _unit(u + step[0] * e1 + step[1] * e2)
        error = residual(u)
        if error < best_residual:
            best, best_residual = u, error
        if error <= 1e-16:
            break
    return best


@pytest.mark.parametrize("offsets", [np.zeros(2), np.array([2e-4, -3e-4])])
def test_transverse_planes_meet_at_the_analytic_sphere_point(offsets) -> None:
    evaluate, residual = _planes(np.eye(3)[:2], offsets)
    start = _unit(np.array([1e-4, -2e-4, 1.0]))
    result = refine_pair(start, evaluate, residual)
    expected = np.array([*offsets, np.sqrt(1.0 - offsets @ offsets)])
    assert isinstance(result, PairRefinement)
    assert result.termination == "converged"
    assert 1 <= result.iterations <= 30
    assert result.residual == residual(result.position) <= 1e-16
    np.testing.assert_allclose(result.position, expected, rtol=0.0, atol=2e-16)
    assert abs(np.linalg.norm(result.position) - 1.0) <= 2e-16


def test_large_step_is_rejected_not_clipped() -> None:
    evaluate, residual = _planes(np.eye(3)[:2], np.zeros(2))
    start = _unit(np.array([0.01, 0.0, 1.0]))
    result = refine_pair(start, evaluate, residual)
    assert result.termination == "step_limit"
    assert result.iterations == 1
    assert result.position is start
    assert result.residual == residual(start)
    # The same analytic root is reached if the caller explicitly expands its budget.
    assert refine_pair(start, evaluate, residual, max_step=0.02).termination == "converged"


@pytest.mark.parametrize("offsets", [np.zeros(2), np.array([0.0, 0.001])])
def test_coincident_or_disjoint_parallel_planes_are_singular_not_success(offsets) -> None:
    evaluate, residual = _planes(np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]), offsets)
    start = np.array([0.0, 0.0, 1.0])
    result = refine_pair(start, evaluate, residual)
    assert result.termination == "singular"  # even a zero initial residual is not pre-accepted
    assert result.iterations == 1
    assert result.position is start
    assert result.residual == residual(start)


@pytest.mark.parametrize(("epsilon", "termination"), [(1e-10, "converged"), (1e-15, "singular")])
def test_near_coincident_great_circles_respect_the_original_determinant_guard(epsilon, termination) -> None:
    evaluate, residual = _planes(np.array([[1.0, 0.0, 0.0], [1.0, epsilon, 0.0]]), np.zeros(2))
    start = _unit(np.array([1e-4, -2e-4, 1.0]))
    result = refine_pair(start, evaluate, residual)
    assert result.termination == termination
    if termination == "converged":
        assert result.residual <= 1e-16
    else:
        assert result.position is start


@pytest.mark.parametrize("budget", [0, 1])
def test_iteration_budget_does_not_report_success(budget) -> None:
    evaluate, residual = _planes(np.eye(3)[:2], np.array([0.3, 0.2]))
    start = _unit(np.array([0.3003, 0.2004, np.sqrt(0.87)]))
    result = refine_pair(start, evaluate, residual, max_iterations=budget)
    assert result.termination == "iteration_limit"
    assert result.iterations == budget
    assert result.residual == residual(result.position) > 1e-16
    if budget == 0:
        assert result.position is start
    else:
        assert result.residual < residual(start)


def test_residual_tolerance_changes_only_the_post_update_stop() -> None:
    evaluate, residual = _planes(np.eye(3)[:2], np.array([0.3, 0.2]))
    start = _unit(np.array([0.3003, 0.2004, np.sqrt(0.87)]))
    strict = refine_pair(start, evaluate, residual, max_iterations=1)
    relaxed = refine_pair(start, evaluate, residual, max_iterations=1, residual_tolerance=1e-6)
    assert strict.termination == "iteration_limit"
    assert relaxed.termination == "converged"
    assert strict.position.tobytes() == relaxed.position.tobytes()
    assert strict.residual == relaxed.residual
    assert 1e-16 < relaxed.residual <= 1e-6
    assert strict.iterations == relaxed.iterations == 1
    # Even an initial residual well inside the chosen tolerance goes through
    # evaluate, the determinant/step guards, and the post-update residual.
    calls = []

    def ev(u):
        calls.append("evaluate")
        return evaluate(u)

    def res(u):
        calls.append("residual")
        return residual(u)

    initial = refine_pair(start, ev, res, residual_tolerance=1.0)
    assert initial.iterations == 1
    assert calls == ["residual", "evaluate", "residual"]
    singular, zero = _planes(np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]), np.zeros(2))
    assert refine_pair(np.array([0.0, 0.0, 1.0]), singular, zero, residual_tolerance=1.0).termination == "singular"
    assert refine_pair(start, evaluate, residual, max_step=1e-5, residual_tolerance=1.0).termination == "step_limit"


@pytest.mark.parametrize("location", ["position", "values", "jacobian", "residual"])
@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_nonfinite_is_explicit_and_preserves_legacy_best_point_and_stops(location, bad) -> None:
    evaluate, residual = _planes(np.eye(3)[:2], np.zeros(2))
    start = _unit(np.array([1e-4, -2e-4, 1.0]))
    if location == "position":
        start[0] = bad

    def faulty_evaluate(u):
        m, j = evaluate(u)
        m, j = m.copy(), j.copy()
        if location == "values":
            m[0] = bad
        if location == "jacobian":
            j[0, 0] = bad
        return m, j

    def faulty_residual(u):
        return float(bad) if location == "residual" else residual(u)

    traces = []
    for run in (refine_pair, _legacy_refine):
        trace = []

        def ev(u):
            trace.append(("evaluate", u.copy()))
            return faulty_evaluate(u)

        def res(u):
            trace.append(("residual", u.copy()))
            return faulty_residual(u)

        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            answer = run(start, ev, res, max_iterations=3)
        traces.append(trace)
        if run is refine_pair:
            result = answer
        else:
            expected = answer
    assert result.termination == "nonfinite"
    assert result.position.tobytes() == expected.tobytes()
    assert len(traces[0]) == len(traces[1])
    for (kind, u), (old_kind, old_u) in zip(*traces, strict=True):
        assert kind == old_kind
        assert u.tobytes() == old_u.tobytes()


def test_nonfinite_jacobian_cannot_be_hidden_by_a_small_independent_residual() -> None:
    def evaluate(u):
        return np.zeros(2), np.full((2, 3), np.nan)

    with np.errstate(invalid="ignore"):
        result = refine_pair(np.array([0.0, 0.0, 1.0]), evaluate, lambda u: 0.0)
    assert result.termination == "nonfinite"
    assert result.iterations == 1  # retains the old post-update residual stop


@pytest.mark.parametrize("errors", [(0.5, 0.25, 0.375), (0.5, 0.5, 0.5)])
def test_independent_residual_selects_best_not_last_and_only_on_strict_improvement(errors) -> None:
    evaluate, _ = _planes(np.eye(3)[:2], np.zeros(2))
    start = _unit(np.array([1e-4, -2e-4, 1.0]))
    visited = []
    pending = iter(errors)

    def residual(u):
        visited.append(u)
        return next(pending)

    result = refine_pair(start, evaluate, residual, max_iterations=2)
    assert result.termination == "iteration_limit"
    assert result.iterations == 2
    expected_index = 1 if errors[1] < errors[0] else 0
    assert result.position is visited[expected_index]
    assert result.residual == errors[expected_index]
    assert len(visited) == 3  # no extra residual call when constructing the result


@pytest.mark.parametrize("seed", range(10))
def test_finite_callback_order_and_iterates_are_bitwise_legacy(seed) -> None:
    rng = np.random.default_rng(seed)
    normals = rng.normal(size=(2, 3))
    root = _unit(rng.normal(size=3))
    evaluate, residual = _planes(normals, normals @ root)
    start = _unit(root + 1e-4 * rng.normal(size=3))
    traces = []
    for run in (refine_pair, _legacy_refine):
        trace = []

        def ev(u):
            trace.append(("evaluate", u.tobytes()))
            return evaluate(u)

        def res(u):
            trace.append(("residual", u.tobytes()))
            return residual(u)

        answer = run(start, ev, res)
        traces.append(trace)
        if run is refine_pair:
            result = answer
        else:
            expected = answer
    assert result.position.tobytes() == expected.tobytes()
    assert result.residual == residual(expected)
    assert traces[0] == traces[1]


def test_callback_errors_are_not_converted_to_numerical_failure() -> None:
    def broken(u):
        raise RuntimeError("broken evaluator")

    with pytest.raises(RuntimeError, match="broken evaluator"):
        refine_pair(np.array([0.0, 0.0, 1.0]), broken, lambda u: 1.0)
