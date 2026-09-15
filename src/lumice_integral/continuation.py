"""Reference predictor-corrector continuation on SO(3)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Callable, Mapping

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from .analytic import residual, residual_after_update
from .so3 import exp, rotation_distance


class FiberStatus(StrEnum):
    """Closed set of terminal status classes from the Phase I contract."""

    CLOSED = "closed"
    EVENT_TERMINATED = "event_terminated"
    NUMERICAL_FAILURE = "numerical_failure"
    BUDGET_EXHAUSTED = "budget_exhausted"


class TerminationReason(StrEnum):
    """Reference-core reason codes grouped by :class:`FiberStatus`."""

    CLOSED_LOOP = "closed_loop"
    TIR_BOUNDARY = "tir_boundary"
    BRANCH_BOUNDARY = "branch_boundary"
    PATH_INFEASIBLE = "path_infeasible"
    VISIBILITY_BOUNDARY = "visibility_boundary"
    CHART_BOUNDARY = "chart_boundary"
    RANK_LOSS = "rank_loss"
    TOPOLOGY_AMBIGUITY = "topology_ambiguity"
    CORRECTOR_FAILURE = "corrector_failure"
    LINEAR_SOLVE_FAILURE = "linear_solve_failure"
    NON_FINITE = "non_finite"
    STEP_UNDERFLOW = "step_underflow"
    INVALID_NUMERICAL_INPUT = "invalid_numerical_input"
    STEP_BUDGET = "step_budget"
    ARCLENGTH_BUDGET = "arclength_budget"
    EVALUATION_BUDGET = "evaluation_budget"


@dataclass(frozen=True)
class EventCandidate:
    """A named boundary or invalid-domain observation made outside AD."""

    kind: TerminationReason
    margin: float
    message: str = ""
    details: Mapping[str, float | str] = field(default_factory=dict)


@dataclass(frozen=True)
class DomainEvaluation:
    """Discrete validity result evaluated before a smooth direction map."""

    valid: bool
    margins: Mapping[str, float] = field(default_factory=dict)
    event: EventCandidate | None = None


@dataclass(frozen=True)
class TargetChart:
    """Orthogonal target-local chart and its declared neighborhood."""

    direction: Array
    basis: Array
    minimum_dot: float = 0.0

    def __post_init__(self) -> None:
        direction = np.asarray(self.direction)
        basis = np.asarray(self.basis)
        if direction.shape != (3,) or basis.shape != (3, 2):
            raise ValueError("target chart requires direction (3,) and basis (3, 2)")
        if direction.dtype != np.float64 or basis.dtype != np.float64:
            raise ValueError("reference target chart must use float64")
        if not np.all(np.isfinite(direction)) or not np.all(np.isfinite(basis)):
            raise ValueError("target chart values must be finite")
        if not np.isclose(np.linalg.norm(direction), 1.0, rtol=0.0, atol=1e-10):
            raise ValueError("target direction must be unit length")
        if not np.allclose(basis.T @ basis, np.eye(2), rtol=0.0, atol=1e-10):
            raise ValueError("target basis must have orthonormal columns")
        if not np.allclose(basis.T @ direction, 0.0, rtol=0.0, atol=1e-10):
            raise ValueError("target basis must be tangent to the target direction")
        if not -1.0 < self.minimum_dot < 1.0:
            raise ValueError("target chart minimum_dot must lie strictly inside (-1, 1)")


DirectionEvaluator = Callable[[Array], Array]
DomainAndEventEvaluator = Callable[[Array], DomainEvaluation]


@dataclass(frozen=True)
class FiberProblem:
    """One fixed-path, one-seed continuation problem on ``SO(3)``."""

    path: str
    incident_direction: Array
    target_chart: TargetChart
    direction_evaluator: DirectionEvaluator
    seed: Array
    domain_and_event_evaluator: DomainAndEventEvaluator | None = None
    pose_metric_and_measure: str = "right-invariant-so3/dvol_g"
    convention_version: str = "phase1-v1"
    weight_evaluators: Mapping[str, Callable[[Array], Array]] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        incident = np.asarray(self.incident_direction)
        seed = np.asarray(self.seed)
        if not self.path:
            raise ValueError("path identifier must be non-empty")
        if incident.shape != (3,) or seed.shape != (3, 3):
            raise ValueError("incident direction must be (3,) and seed must be (3, 3)")
        if incident.dtype != np.float64 or seed.dtype != np.float64:
            raise ValueError("reference problem inputs must use float64")
        if not np.all(np.isfinite(incident)) or not np.all(np.isfinite(seed)):
            raise ValueError("reference problem inputs must be finite")
        if not np.isclose(np.linalg.norm(incident), 1.0, rtol=0.0, atol=1e-10):
            raise ValueError("incident direction must be unit length")
        if not np.allclose(seed.T @ seed, np.eye(3), rtol=0.0, atol=1e-10):
            raise ValueError("seed must be orthogonal")
        if not np.isclose(np.linalg.det(seed), 1.0, rtol=0.0, atol=1e-10):
            raise ValueError("seed must have determinant +1")
        if not callable(self.direction_evaluator):
            raise ValueError("direction_evaluator must be callable")
        if self.domain_and_event_evaluator is not None and not callable(
            self.domain_and_event_evaluator
        ):
            raise ValueError("domain_and_event_evaluator must be callable")


@dataclass(frozen=True)
class ContinuationOptions:
    """Observable numerical policy for the float64 reference solver."""

    dtype: str = "float64"
    unit_tolerance: float = 1e-10
    rotation_tolerance: float = 1e-10
    residual_tolerance: float = 1e-11
    relative_residual_tolerance: float = 0.0
    singular_value_tolerance: float = 1e-8
    condition_limit: float = 1e8
    initial_step: float = 0.04
    minimum_step: float = 1e-5
    maximum_step: float = 0.12
    shrink_factor: float = 0.5
    growth_factor: float = 1.25
    maximum_retries: int = 8
    corrector_maximum_iterations: int = 10
    corrector_update_tolerance: float = 1e-12
    maximum_correction: float = 0.2
    maximum_advance: float = 0.2
    minimum_tangent_dot: float = 0.8
    event_slowdown_margin: float = 0.02
    maximum_accepted_steps: int = 4000
    maximum_evaluations: int = 100_000
    maximum_arclength: float = 20.0
    closure_minimum_steps: int = 40
    closure_minimum_arclength: float = np.pi
    closure_distance: float = 0.08
    closure_tangent_dot: float = 0.8
    diagnostic_level: str = "full"
    sample_retention: str = "all"

    def __post_init__(self) -> None:
        if self.dtype != "float64":
            raise ValueError("the reference solver only accepts dtype='float64'")
        if not 0.0 < self.minimum_step <= self.initial_step <= self.maximum_step:
            raise ValueError("step sizes must satisfy 0 < minimum <= initial <= maximum")
        if not 0.0 < self.shrink_factor < 1.0 < self.growth_factor:
            raise ValueError("step factors must satisfy 0 < shrink < 1 < growth")
        if self.maximum_retries < 0 or self.corrector_maximum_iterations < 1:
            raise ValueError("retry and corrector budgets must be nonnegative")
        if self.maximum_accepted_steps < 1 or self.maximum_evaluations < 1:
            raise ValueError("step and evaluation budgets must be positive")
        if self.maximum_arclength <= 0.0:
            raise ValueError("arclength budget must be positive")
        if not -1.0 <= self.minimum_tangent_dot <= 1.0:
            raise ValueError("minimum_tangent_dot must lie in [-1, 1]")
        if not -1.0 <= self.closure_tangent_dot <= 1.0:
            raise ValueError("closure_tangent_dot must lie in [-1, 1]")


@dataclass(frozen=True)
class JacobianDiagnostic:
    singular_values: tuple[float, float]
    normal_jacobian: float
    rank: int
    condition: float
    available: bool = True


@dataclass(frozen=True)
class StepDiagnostic:
    accepted_index: int
    trial_index: int
    proposed_step: float
    accepted: bool
    corrector_iterations: int
    residual_norm: float
    update_norm: float
    correction_norm: float
    advance: float
    tangent_dot: float
    reason: str
    event_margins: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class BranchDiagnostic:
    path: str
    accepted_margins: tuple[Mapping[str, float], ...]
    terminal_margins: Mapping[str, float]


@dataclass(frozen=True)
class ClosureDiagnostic:
    accumulated_arclength: float
    seed_distance: float
    previous_section_value: float
    section_value: float
    section_crossed: bool
    crossing_direction: int
    tangent_dot: float
    final_correction_attempted: bool
    final_correction_accepted: bool


@dataclass(frozen=True)
class TerminalPayload:
    last_accepted_pose: np.ndarray
    rejected_pose: np.ndarray | None
    event: EventCandidate | None
    accepted_steps: int
    evaluations: int
    retries: int
    message: str


@dataclass(frozen=True)
class FiberResult:
    """Quadrature-ready geometry for one component reached from one seed."""

    status: FiberStatus
    reason: TerminationReason
    poses: np.ndarray
    arclength_increments: np.ndarray
    residual_norms: np.ndarray
    tangents: np.ndarray
    jacobian_diagnostics: tuple[JacobianDiagnostic, ...]
    step_diagnostics: tuple[StepDiagnostic, ...]
    branch_diagnostics: BranchDiagnostic
    closure_diagnostics: ClosureDiagnostic
    terminal_payload: TerminalPayload
    conventions: Mapping[str, str]
    weight_observables: Mapping[str, str]
    component_scope: str = "one component reached from one seed"
    component_completeness: str = "unknown"


@dataclass(frozen=True)
class TraceResult:
    rotations: np.ndarray
    residual_norms: np.ndarray
    closure_error: float
    length: float
    steps: int
    detected_gap: float


ResidualFunction = Callable[[Array], Array]


def residual_jacobian(rotation: Array, residual_function: ResidualFunction) -> Array:
    """Differentiate an arbitrary two-component residual on SO(3)."""
    zero = jnp.zeros(3, dtype=rotation.dtype)
    return jax.jacfwd(lambda delta: residual_function(rotation @ exp(delta)))(zero)


def constraint_jacobian(rotation: Array, target: Array, basis: Array) -> Array:
    """Differentiate the local residual in right-trivialized coordinates."""
    zero = jnp.zeros(3, dtype=rotation.dtype)
    return jax.jacfwd(residual_after_update, argnums=0)(
        zero, rotation, target, basis
    )


def null_tangent(jacobian: Array, previous: Array | None = None) -> Array:
    """Return an oriented unit vector spanning a rank-two Jacobian's nullspace."""
    _, _, vh = jnp.linalg.svd(jacobian, full_matrices=True)
    tangent = vh[-1]
    if previous is not None:
        tangent = jnp.where(jnp.dot(tangent, previous) < 0.0, -tangent, tangent)
    return tangent / jnp.linalg.norm(tangent)


def correct(
    predicted: Array,
    target: Array,
    basis: Array,
    tangent: Array,
    *,
    tolerance: float = 1e-13,
    max_iterations: int = 8,
) -> Array:
    """Correct normal to the predicted fiber tangent using bordered Newton steps."""
    rotation = predicted
    for _ in range(max_iterations):
        value = residual(rotation, target, basis)
        if float(jnp.linalg.norm(value)) <= tolerance:
            break
        jacobian = constraint_jacobian(rotation, target, basis)
        bordered = jnp.concatenate((jacobian, tangent[jnp.newaxis, :]), axis=0)
        step = jnp.linalg.solve(bordered, jnp.concatenate((-value, jnp.zeros(1))))
        rotation = rotation @ exp(step)
    return rotation


def trace_closed_fiber(
    initial: Array,
    target: Array,
    basis: Array,
    *,
    steps: int = 128,
) -> TraceResult:
    """Trace one known full turn of the analytic fiber."""
    step_size = 2.0 * jnp.pi / steps
    rotation = initial
    tangent = null_tangent(constraint_jacobian(rotation, target, basis))
    rotations = [np.asarray(rotation)]
    residual_norms = [float(jnp.linalg.norm(residual(rotation, target, basis)))]

    for _ in range(steps):
        predicted = rotation @ exp(step_size * tangent)
        rotation = correct(predicted, target, basis, tangent)
        tangent = null_tangent(
            constraint_jacobian(rotation, target, basis), previous=tangent
        )
        rotations.append(np.asarray(rotation))
        residual_norms.append(
            float(jnp.linalg.norm(residual(rotation, target, basis)))
        )

    return TraceResult(
        rotations=np.stack(rotations),
        residual_norms=np.asarray(residual_norms),
        closure_error=float(rotation_distance(initial, rotation)),
        length=float(steps * step_size),
        steps=steps,
        detected_gap=float(rotation_distance(initial, rotation)),
    )


def _section_coordinate(initial: Array, rotation: Array, tangent: Array) -> Array:
    relative = initial.T @ rotation
    skew_vector = jnp.array(
        [
            relative[2, 1] - relative[1, 2],
            relative[0, 2] - relative[2, 0],
            relative[1, 0] - relative[0, 1],
        ]
    ) / 2.0
    return jnp.dot(tangent, skew_vector)


def trace_implicit_fiber(
    initial: Array,
    residual_function: ResidualFunction,
    *,
    step_size: float = 0.02,
    max_steps: int = 2000,
    min_steps: int = 100,
    closure_search_radius: float = 0.25,
    tolerance: float = 1e-11,
) -> TraceResult:
    """Trace a regular implicit fiber until it returns through the start section."""
    evaluate = jax.jit(residual_function)
    differentiate = jax.jit(lambda rotation: residual_jacobian(rotation, residual_function))
    rotation = initial
    initial_tangent = null_tangent(differentiate(rotation))
    tangent = initial_tangent
    rotations = [np.asarray(rotation)]
    residual_norms = [float(jnp.linalg.norm(evaluate(rotation)))]
    previous_section = float(_section_coordinate(initial, rotation, initial_tangent))
    minimum_gap = float("inf")
    section_crossings = 0

    for step_index in range(1, max_steps + 1):
        predicted = rotation @ exp(
            jnp.asarray(step_size, dtype=rotation.dtype) * tangent
        )
        rotation = predicted
        for _ in range(8):
            value = evaluate(rotation)
            if float(jnp.linalg.norm(value)) <= tolerance:
                break
            jacobian = differentiate(rotation)
            bordered = jnp.concatenate(
                (jacobian, tangent[jnp.newaxis, :]), axis=0
            )
            update = jnp.linalg.solve(
                bordered,
                jnp.concatenate((-value, jnp.zeros(1, dtype=rotation.dtype))),
            )
            rotation = rotation @ exp(update)

        residual_norm = float(jnp.linalg.norm(evaluate(rotation)))
        if not np.isfinite(residual_norm):
            raise RuntimeError(f"non-finite residual at step {step_index}")
        if residual_norm > tolerance:
            raise RuntimeError(
                f"corrector failed at step {step_index}: residual={residual_norm}"
            )
        tangent = null_tangent(differentiate(rotation), previous=tangent)
        rotations.append(np.asarray(rotation))
        residual_norms.append(residual_norm)

        section = float(_section_coordinate(initial, rotation, initial_tangent))
        gap = float(rotation_distance(initial, rotation))
        crossed = previous_section != 0.0 and previous_section * section <= 0.0
        if crossed:
            section_crossings += 1
        if step_index >= min_steps:
            minimum_gap = min(minimum_gap, gap)
        if (
            step_index >= min_steps
            and crossed
            and gap <= closure_search_radius
            and float(jnp.dot(tangent, initial_tangent)) > 0.0
        ):
            detected_gap = gap

            def closing_system(delta):
                candidate = rotation @ exp(delta)
                return jnp.concatenate(
                    (
                        residual_function(candidate),
                        jnp.atleast_1d(
                            _section_coordinate(initial, candidate, initial_tangent)
                        ),
                    )
                )

            delta = jnp.zeros(3, dtype=rotation.dtype)
            for _ in range(8):
                value = closing_system(delta)
                if float(jnp.linalg.norm(value)) <= tolerance:
                    break
                jacobian = jax.jacfwd(closing_system)(delta)
                delta = delta + jnp.linalg.solve(jacobian, -value)
            rotation = rotation @ exp(delta)
            rotations.append(np.asarray(rotation))
            residual_norms.append(float(jnp.linalg.norm(evaluate(rotation))))
            return TraceResult(
                rotations=np.stack(rotations),
                residual_norms=np.asarray(residual_norms),
                closure_error=float(rotation_distance(initial, rotation)),
                length=float(step_index * step_size + detected_gap),
                steps=step_index,
                detected_gap=detected_gap,
            )
        previous_section = section

    raise RuntimeError(
        "fiber did not close within "
        f"{max_steps} steps: min_gap={minimum_gap}, "
        f"section_crossings={section_crossings}"
    )
