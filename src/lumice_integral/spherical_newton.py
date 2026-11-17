"""Local two-constraint Newton refinement on the unit sphere.

This is the mechanical kernel of the boundary walk, not an intersection
certificate. Consumers must independently check rank, domain and locality;
even a converged result only describes the local iteration's stopping rule.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np

PairTermination = Literal["converged", "singular", "step_limit", "iteration_limit", "nonfinite"]


@dataclass(frozen=True)
class PairRefinement:
    """Best point visited, its independently evaluated residual, and loop diagnostics.

    ``iterations`` counts attempted Newton iterations (including an attempt
    stopped by the determinant or step guard). ``termination`` describes
    the loop, not an acceptance certificate for ``position``. Observing any
    nonfinite input, evaluation or iterate overrides the other reasons with
    ``nonfinite`` without adding an early exit to the legacy iteration.
    """

    position: np.ndarray
    residual: float
    termination: PairTermination
    iterations: int


def refine_pair(
    u: np.ndarray,
    evaluate: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    residual: Callable[[np.ndarray], float],
    *,
    max_iterations: int = 30,
    max_step: float = 1e-3,
    residual_tolerance: float = 1e-16,
) -> PairRefinement:
    """Refine two scalar zero sets using their ambient ``(2, 3)`` Jacobian.

    ``evaluate(u)`` returns the two values and Jacobian rows. ``residual(u)``
    independently evaluates their maximum absolute value, as in the original
    boundary walker (not from the jointly evaluated values). The determinant
    guard is ``1e-14``; the residual stop defaults to ``1e-16`` after an
    update. ``residual_tolerance`` changes only that post-update stop. The
    initial point is not pre-accepted, and only strictly improving residuals
    replace the best point. No normalization of the initial point is added.

    Floating-point operations, callback order and stops deliberately follow
    ``Walker.refine_corner``. In particular a large step is rejected, not
    clipped, and singular/unsuccessful iterations still return the best point.
    Callback errors and linear-algebra exceptions are not converted to data.
    """
    best = u
    best_residual = residual(u)
    nonfinite = not (np.isfinite(u).all() and np.isfinite(best_residual))
    termination: PairTermination = "iteration_limit"
    iterations = 0
    for iteration in range(max_iterations):
        iterations = iteration + 1
        m, j = evaluate(u)
        nonfinite = nonfinite or not (np.isfinite(m).all() and np.isfinite(j).all())
        axis = np.eye(3)[int(np.argmin(np.abs(u)))]
        e1 = np.cross(u, axis)
        e1 = e1 / np.linalg.norm(e1)
        e2 = np.cross(u, e1)
        jac = np.array([[j[0] @ e1, j[0] @ e2], [j[1] @ e1, j[1] @ e2]])
        if abs(np.linalg.det(jac)) < 1e-14:
            termination = "singular"
            break
        step = np.linalg.solve(jac, -np.array([m[0], m[1]]))
        nonfinite = nonfinite or not np.isfinite(step).all()
        if np.linalg.norm(step) > max_step:
            termination = "step_limit"
            break
        u = u + step[0] * e1 + step[1] * e2
        u = u / np.linalg.norm(u)
        current_residual = residual(u)
        nonfinite = nonfinite or not (np.isfinite(u).all() and np.isfinite(current_residual))
        if current_residual < best_residual:
            best, best_residual = u, current_residual
        if current_residual <= residual_tolerance:
            termination = "converged"
            break
    return PairRefinement(best, best_residual, "nonfinite" if nonfinite else termination, iterations)
