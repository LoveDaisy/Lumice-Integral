"""Reference predictor-corrector continuation on SO(3)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Callable, Mapping

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

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
    corrector_phase_tolerance: float = 1e-12
    corrector_update_tolerance: float = 1e-12
    maximum_correction: float = 0.2
    maximum_advance: float = 0.2
    minimum_tangent_dot: float = 0.8
    event_slowdown_margin: float = 0.02
    maximum_accepted_steps: int = 4000
    # One evaluation unit reserves all gated value and AD work for one pose or
    # corrector iterate; sub-operations cannot consume a partial unit.
    maximum_evaluations: int = 100_000
    maximum_arclength: float = 20.0
    closure_minimum_steps: int = 40
    closure_minimum_arclength: float = np.pi
    closure_distance: float = 0.08
    closure_tangent_dot: float = 0.8
    closure_section_tolerance: float = 1e-11
    closure_maximum_iterations: int = 10
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
        if (
            self.corrector_phase_tolerance <= 0.0
            or self.corrector_update_tolerance <= 0.0
        ):
            raise ValueError("corrector convergence tolerances must be positive")
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
    last_accepted_pose: np.ndarray | None
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
class _StateEvaluation:
    accepted: bool
    rotation: Array
    direction: Array | None
    residual: Array | None
    residual_norm: float
    jacobian: Array | None
    tangent: Array | None
    jacobian_diagnostic: JacobianDiagnostic | None
    domain: DomainEvaluation
    reason: TerminationReason | None
    event: EventCandidate | None
    message: str


@dataclass(frozen=True)
class _CorrectorOutcome:
    accepted: bool
    state: _StateEvaluation | None
    predicted_pose: np.ndarray
    rejected_pose: np.ndarray | None
    iterations: int
    residual_norm: float
    update_norm: float
    correction_norm: float
    advance: float
    tangent_dot: float
    reason: TerminationReason | None
    event: EventCandidate | None
    evaluations: int
    message: str


_EVENT_REASONS = {
    TerminationReason.TIR_BOUNDARY,
    TerminationReason.BRANCH_BOUNDARY,
    TerminationReason.PATH_INFEASIBLE,
    TerminationReason.VISIBILITY_BOUNDARY,
    TerminationReason.CHART_BOUNDARY,
    TerminationReason.RANK_LOSS,
    TerminationReason.TOPOLOGY_AMBIGUITY,
}


def _default_domain_evaluation(_: Array) -> DomainEvaluation:
    return DomainEvaluation(valid=True)


def _evaluate_domain(problem: FiberProblem, rotation: Array) -> DomainEvaluation:
    evaluator = problem.domain_and_event_evaluator or _default_domain_evaluation
    try:
        evaluation = evaluator(rotation)
    except Exception as error:  # A user adapter is an explicit failure boundary.
        return DomainEvaluation(
            valid=False,
            event=EventCandidate(
                TerminationReason.NON_FINITE,
                float("nan"),
                f"domain evaluator raised {type(error).__name__}: {error}",
            ),
        )
    if not isinstance(evaluation, DomainEvaluation):
        return DomainEvaluation(
            valid=False,
            event=EventCandidate(
                TerminationReason.INVALID_NUMERICAL_INPUT,
                float("nan"),
                "domain evaluator must return DomainEvaluation",
            ),
        )
    margins = {name: float(value) for name, value in evaluation.margins.items()}
    if evaluation.event is not None:
        event = EventCandidate(
            evaluation.event.kind,
            float(evaluation.event.margin),
            evaluation.event.message,
            evaluation.event.details,
        )
    else:
        event = None
    if not evaluation.valid and event is None:
        event = EventCandidate(
            TerminationReason.PATH_INFEASIBLE,
            min(margins.values(), default=float("nan")),
            "domain evaluator rejected the pose without a more specific event",
        )
    return DomainEvaluation(evaluation.valid, margins, event)


def _rejected_state(
    rotation: Array,
    domain: DomainEvaluation,
    reason: TerminationReason,
    message: str,
    *,
    direction: Array | None = None,
    residual_value: Array | None = None,
    residual_norm: float = float("inf"),
    event: EventCandidate | None = None,
) -> _StateEvaluation:
    return _StateEvaluation(
        accepted=False,
        rotation=rotation,
        direction=direction,
        residual=residual_value,
        residual_norm=residual_norm,
        jacobian=None,
        tangent=None,
        jacobian_diagnostic=None,
        domain=domain,
        reason=reason,
        event=event,
        message=message,
    )


def _evaluate_regular_state(
    problem: FiberProblem,
    options: ContinuationOptions,
    rotation: Array,
    previous_tangent: Array | None = None,
) -> _StateEvaluation:
    """Evaluate one indivisible budgeted pose unit outside the AD graph."""
    domain = _evaluate_domain(problem, rotation)
    if not domain.valid:
        event = domain.event
        reason = event.kind if event is not None else TerminationReason.PATH_INFEASIBLE
        return _rejected_state(
            rotation,
            domain,
            reason,
            event.message if event is not None else "invalid path domain",
            event=event,
        )

    try:
        direction = problem.direction_evaluator(rotation)
        direction_array = np.asarray(direction)
    except Exception as error:
        return _rejected_state(
            rotation,
            domain,
            TerminationReason.NON_FINITE,
            f"direction evaluator raised {type(error).__name__}: {error}",
        )
    if direction_array.shape != (3,) or direction_array.dtype != np.float64:
        return _rejected_state(
            rotation,
            domain,
            TerminationReason.INVALID_NUMERICAL_INPUT,
            "direction evaluator must return a float64 array with shape (3,)",
            direction=direction,
        )
    if not np.all(np.isfinite(direction_array)):
        return _rejected_state(
            rotation,
            domain,
            TerminationReason.NON_FINITE,
            "direction evaluator returned a non-finite value",
            direction=direction,
        )
    direction_norm = float(np.linalg.norm(direction_array))
    if not np.isclose(
        direction_norm, 1.0, rtol=0.0, atol=options.unit_tolerance
    ):
        return _rejected_state(
            rotation,
            domain,
            TerminationReason.INVALID_NUMERICAL_INPUT,
            f"direction evaluator returned non-unit output: norm={direction_norm}",
            direction=direction,
        )

    chart = problem.target_chart
    chart_dot = float(jnp.dot(direction, chart.direction))
    if chart_dot <= chart.minimum_dot:
        event = EventCandidate(
            TerminationReason.CHART_BOUNDARY,
            chart_dot - chart.minimum_dot,
            "direction left the declared target chart neighborhood",
        )
        return _rejected_state(
            rotation,
            domain,
            event.kind,
            event.message,
            direction=direction,
            event=event,
        )

    residual_value = chart.basis.T @ (direction - chart.direction)
    residual_norm = float(jnp.linalg.norm(residual_value))
    if not np.isfinite(residual_norm):
        return _rejected_state(
            rotation,
            domain,
            TerminationReason.NON_FINITE,
            "target residual is non-finite",
            direction=direction,
            residual_value=residual_value,
            residual_norm=residual_norm,
        )

    try:
        jacobian = local_residual_jacobian(problem, rotation)
        jacobian_array = np.asarray(jacobian)
    except Exception as error:
        return _rejected_state(
            rotation,
            domain,
            TerminationReason.NON_FINITE,
            f"direction differentiation failed: {type(error).__name__}: {error}",
            direction=direction,
            residual_value=residual_value,
            residual_norm=residual_norm,
        )
    if jacobian_array.shape != (2, 3) or not np.all(np.isfinite(jacobian_array)):
        return _rejected_state(
            rotation,
            domain,
            TerminationReason.NON_FINITE,
            "local residual Jacobian is non-finite or has the wrong shape",
            direction=direction,
            residual_value=residual_value,
            residual_norm=residual_norm,
        )

    _, singular_values, vh = np.linalg.svd(jacobian_array, full_matrices=True)
    sigma_1, sigma_2 = (float(singular_values[0]), float(singular_values[1]))
    normal_jacobian = sigma_1 * sigma_2
    rank = int(np.count_nonzero(singular_values >= options.singular_value_tolerance))
    condition = sigma_1 / sigma_2 if sigma_2 > 0.0 else float("inf")
    diagnostic = JacobianDiagnostic(
        singular_values=(sigma_1, sigma_2),
        normal_jacobian=normal_jacobian,
        rank=rank,
        condition=condition,
    )
    if rank < 2 or condition > options.condition_limit:
        event = EventCandidate(
            TerminationReason.RANK_LOSS,
            sigma_2 - options.singular_value_tolerance,
            "local residual Jacobian failed the regularity gate",
            {"sigma_1": sigma_1, "sigma_2": sigma_2, "condition": condition},
        )
        return _StateEvaluation(
            accepted=False,
            rotation=rotation,
            direction=direction,
            residual=residual_value,
            residual_norm=residual_norm,
            jacobian=jacobian,
            tangent=None,
            jacobian_diagnostic=diagnostic,
            domain=domain,
            reason=event.kind,
            event=event,
            message=event.message,
        )

    tangent_array = vh[-1]
    if previous_tangent is not None and float(
        np.dot(tangent_array, np.asarray(previous_tangent))
    ) < 0.0:
        tangent_array = -tangent_array
    tangent_array = tangent_array / np.linalg.norm(tangent_array)
    tangent = jnp.asarray(tangent_array, dtype=rotation.dtype)
    return _StateEvaluation(
        accepted=True,
        rotation=rotation,
        direction=direction,
        residual=residual_value,
        residual_norm=residual_norm,
        jacobian=jacobian,
        tangent=tangent,
        jacobian_diagnostic=diagnostic,
        domain=domain,
        reason=None,
        event=None,
        message="regular state",
    )


def _correct_trial(
    problem: FiberProblem,
    options: ContinuationOptions,
    current: Array,
    predicted: Array,
    phase_tangent: Array,
    maximum_evaluations: int | None = None,
) -> _CorrectorOutcome:
    """Apply a bounded bordered Newton correction around one predictor.

    Each iteration reserves one indivisible budget unit before its gated value
    evaluation and any bordered-system AD work.
    """
    delta = jnp.zeros(3, dtype=predicted.dtype)
    last_residual = float("inf")
    last_update = float("inf")
    evaluations = 0

    def budget_exhausted(iteration: int, candidate: Array) -> _CorrectorOutcome:
        return _CorrectorOutcome(
            False,
            None,
            np.asarray(predicted),
            np.asarray(candidate),
            iteration,
            last_residual,
            last_update,
            float(jnp.linalg.norm(delta)),
            float(rotation_distance(current, candidate)),
            float("nan"),
            TerminationReason.EVALUATION_BUDGET,
            None,
            evaluations,
            "corrector evaluation budget exhausted",
        )

    def system(correction: Array) -> Array:
        candidate = predicted @ exp(correction)
        direction = problem.direction_evaluator(candidate)
        residual_value = problem.target_chart.basis.T @ (
            direction - problem.target_chart.direction
        )
        return jnp.concatenate(
            (residual_value, jnp.atleast_1d(jnp.dot(phase_tangent, correction)))
        )

    for iteration in range(options.corrector_maximum_iterations + 1):
        candidate = predicted @ exp(delta)
        if maximum_evaluations is not None and evaluations >= maximum_evaluations:
            return budget_exhausted(iteration, candidate)
        domain = _evaluate_domain(problem, candidate)
        evaluations += 1
        if not domain.valid:
            event = domain.event
            reason = event.kind if event is not None else TerminationReason.PATH_INFEASIBLE
            return _CorrectorOutcome(
                False,
                None,
                np.asarray(predicted),
                np.asarray(candidate),
                iteration,
                last_residual,
                last_update,
                float(jnp.linalg.norm(delta)),
                float(rotation_distance(current, candidate)),
                float("nan"),
                reason,
                event,
                evaluations,
                event.message if event is not None else "invalid path domain",
            )
        try:
            value = system(delta)
            value_array = np.asarray(value)
        except Exception as error:
            value_array = np.full(3, np.nan)
            message = f"corrector evaluation failed: {type(error).__name__}: {error}"
        else:
            message = "corrector evaluation returned a non-finite value"
        last_residual = float(np.linalg.norm(value_array[:2]))
        phase_norm = abs(float(value_array[2]))
        if not np.all(np.isfinite(value_array)):
            return _CorrectorOutcome(
                False,
                None,
                np.asarray(predicted),
                np.asarray(candidate),
                iteration,
                last_residual,
                last_update,
                float(jnp.linalg.norm(delta)),
                float(rotation_distance(current, candidate)),
                float("nan"),
                TerminationReason.NON_FINITE,
                None,
                evaluations,
                message,
            )
        if (
            last_residual <= options.residual_tolerance
            and phase_norm <= options.corrector_phase_tolerance
            and (last_update if np.isfinite(last_update) else 0.0)
            <= options.corrector_update_tolerance
        ):
            if maximum_evaluations is not None and evaluations >= maximum_evaluations:
                return budget_exhausted(iteration, candidate)
            state = _evaluate_regular_state(
                problem, options, candidate, previous_tangent=phase_tangent
            )
            evaluations += 1
            correction_norm = float(jnp.linalg.norm(delta))
            advance = float(rotation_distance(current, candidate))
            tangent_dot = (
                float(jnp.dot(phase_tangent, state.tangent))
                if state.tangent is not None
                else float("nan")
            )
            if not state.accepted:
                return _CorrectorOutcome(
                    False,
                    state,
                    np.asarray(predicted),
                    np.asarray(candidate),
                    iteration,
                    state.residual_norm,
                    last_update,
                    correction_norm,
                    advance,
                    tangent_dot,
                    state.reason,
                    state.event,
                    evaluations,
                    state.message,
                )
            if correction_norm > options.maximum_correction:
                reason = TerminationReason.CORRECTOR_FAILURE
                message = "corrector exceeded the correction trust limit"
            elif advance > options.maximum_advance:
                reason = TerminationReason.CORRECTOR_FAILURE
                message = "corrected step exceeded the advance trust limit"
            elif tangent_dot < options.minimum_tangent_dot:
                reason = TerminationReason.CORRECTOR_FAILURE
                message = "corrected state exceeded the tangent-change limit"
            else:
                return _CorrectorOutcome(
                    True,
                    state,
                    np.asarray(predicted),
                    None,
                    iteration,
                    state.residual_norm,
                    last_update if np.isfinite(last_update) else 0.0,
                    correction_norm,
                    advance,
                    tangent_dot,
                    None,
                    None,
                    evaluations,
                    "corrector converged",
                )
            return _CorrectorOutcome(
                False,
                state,
                np.asarray(predicted),
                np.asarray(candidate),
                iteration,
                state.residual_norm,
                last_update,
                correction_norm,
                advance,
                tangent_dot,
                reason,
                None,
                evaluations,
                message,
            )
        if iteration == options.corrector_maximum_iterations:
            break
        if maximum_evaluations is not None and evaluations >= maximum_evaluations:
            return budget_exhausted(iteration, candidate)
        try:
            bordered = jax.jacfwd(system)(delta)
            bordered_array = np.asarray(bordered)
            if not np.all(np.isfinite(bordered_array)):
                raise FloatingPointError("non-finite bordered Jacobian")
            condition = float(np.linalg.cond(bordered_array))
            if not np.isfinite(condition) or condition > options.condition_limit:
                raise np.linalg.LinAlgError(
                    f"bordered system condition {condition} exceeds limit"
                )
            update = jnp.linalg.solve(bordered, -value)
            update_array = np.asarray(update)
            if not np.all(np.isfinite(update_array)):
                raise FloatingPointError("non-finite Newton update")
        except (np.linalg.LinAlgError, FloatingPointError) as error:
            return _CorrectorOutcome(
                False,
                None,
                np.asarray(predicted),
                np.asarray(candidate),
                iteration,
                last_residual,
                last_update,
                float(jnp.linalg.norm(delta)),
                float(rotation_distance(current, candidate)),
                float("nan"),
                TerminationReason.LINEAR_SOLVE_FAILURE,
                None,
                evaluations,
                str(error),
            )
        except Exception as error:
            return _CorrectorOutcome(
                False,
                None,
                np.asarray(predicted),
                np.asarray(candidate),
                iteration,
                last_residual,
                last_update,
                float(jnp.linalg.norm(delta)),
                float(rotation_distance(current, candidate)),
                float("nan"),
                TerminationReason.NON_FINITE,
                None,
                evaluations,
                f"corrector differentiation failed: {type(error).__name__}: {error}",
            )
        last_update = float(jnp.linalg.norm(update))
        delta = delta + update

    candidate = predicted @ exp(delta)
    return _CorrectorOutcome(
        False,
        None,
        np.asarray(predicted),
        np.asarray(candidate),
        options.corrector_maximum_iterations,
        last_residual,
        last_update,
        float(jnp.linalg.norm(delta)),
        float(rotation_distance(current, candidate)),
        float("nan"),
        TerminationReason.CORRECTOR_FAILURE,
        None,
        evaluations,
        "corrector iteration budget exhausted",
    )


def _status_for_reason(reason: TerminationReason) -> FiberStatus:
    if reason == TerminationReason.CLOSED_LOOP:
        return FiberStatus.CLOSED
    if reason in _EVENT_REASONS:
        return FiberStatus.EVENT_TERMINATED
    if reason in {
        TerminationReason.STEP_BUDGET,
        TerminationReason.ARCLENGTH_BUDGET,
        TerminationReason.EVALUATION_BUDGET,
    }:
        return FiberStatus.BUDGET_EXHAUSTED
    return FiberStatus.NUMERICAL_FAILURE


def _empty_closure_diagnostic() -> ClosureDiagnostic:
    return ClosureDiagnostic(
        accumulated_arclength=0.0,
        seed_distance=float("inf"),
        previous_section_value=0.0,
        section_value=0.0,
        section_crossed=False,
        crossing_direction=0,
        tangent_dot=float("nan"),
        final_correction_attempted=False,
        final_correction_accepted=False,
    )


def _make_result(
    problem: FiberProblem,
    options: ContinuationOptions,
    reason: TerminationReason,
    states: list[_StateEvaluation],
    arclength_increments: list[float],
    step_diagnostics: list[StepDiagnostic],
    accepted_margins: list[Mapping[str, float]],
    closure_diagnostic: ClosureDiagnostic,
    *,
    rejected_pose: np.ndarray | None,
    event: EventCandidate | None,
    evaluations: int,
    retries: int,
    message: str,
) -> FiberResult:
    poses = (
        np.stack([np.asarray(state.rotation) for state in states])
        if states
        else np.empty((0, 3, 3), dtype=np.float64)
    )
    residual_norms = np.asarray(
        [state.residual_norm for state in states], dtype=np.float64
    )
    tangents = (
        np.stack([np.asarray(state.tangent) for state in states])
        if states
        else np.empty((0, 3), dtype=np.float64)
    )
    jacobian_diagnostics = tuple(
        state.jacobian_diagnostic
        for state in states
        if state.jacobian_diagnostic is not None
    )
    standard_weights = (
        "rho_pose",
        "entry_measure",
        "visibility",
        "fresnel_transmission",
        "path_validity",
        "source_factor",
        "pixel_factor",
        "other_radiometric",
    )
    weight_observables = {
        name: (
            "not_evaluated_by_geometry_solver"
            if name in problem.weight_evaluators
            else "unavailable"
        )
        for name in standard_weights
    }
    return FiberResult(
        status=_status_for_reason(reason),
        reason=reason,
        poses=poses,
        arclength_increments=np.asarray(arclength_increments, dtype=np.float64),
        residual_norms=residual_norms,
        tangents=tangents,
        jacobian_diagnostics=jacobian_diagnostics,
        step_diagnostics=tuple(step_diagnostics),
        branch_diagnostics=BranchDiagnostic(
            path=problem.path,
            accepted_margins=tuple(accepted_margins),
            terminal_margins=(
                dict(event.details)
                if event is not None
                else (
                    dict(step_diagnostics[-1].event_margins)
                    if step_diagnostics
                    else {}
                )
            ),
        ),
        closure_diagnostics=closure_diagnostic,
        terminal_payload=TerminalPayload(
            last_accepted_pose=np.asarray(states[-1].rotation) if states else None,
            rejected_pose=rejected_pose,
            event=event,
            accepted_steps=max(0, len(states) - 1),
            evaluations=evaluations,
            retries=retries,
            message=message,
        ),
        conventions={
            "coordinate_sign": problem.convention_version,
            "pose_representation": "float64 rotation matrix (3, 3)",
            "metric_measure": problem.pose_metric_and_measure,
            "dtype": options.dtype,
            "arclength_unit": "radian",
            "solver_options_version": "reference-continuation-v1",
            "evaluation_unit": (
                "one gated pose or corrector iterate including smooth value and AD work"
            ),
        },
        weight_observables=weight_observables,
    )


def _step_diagnostic(
    accepted_index: int,
    trial_index: int,
    proposed_step: float,
    outcome: _CorrectorOutcome,
) -> StepDiagnostic:
    margins = (
        dict(outcome.state.domain.margins)
        if outcome.state is not None
        else {}
    )
    if outcome.event is not None:
        margins = {**margins, outcome.event.kind.value: outcome.event.margin}
    return StepDiagnostic(
        accepted_index=accepted_index,
        trial_index=trial_index,
        proposed_step=proposed_step,
        accepted=outcome.accepted,
        corrector_iterations=outcome.iterations,
        residual_norm=outcome.residual_norm,
        update_norm=outcome.update_norm,
        correction_norm=outcome.correction_norm,
        advance=outcome.advance,
        tangent_dot=outcome.tangent_dot,
        reason="accepted" if outcome.accepted else outcome.reason.value,
        event_margins=margins,
    )


def _adapt_accepted_step(
    step: float,
    outcome: _CorrectorOutcome,
    options: ContinuationOptions,
) -> float:
    state = outcome.state
    assert state is not None and state.jacobian_diagnostic is not None
    clear_of_event = all(
        margin > options.event_slowdown_margin
        for margin in state.domain.margins.values()
    )
    residual_ratio = outcome.residual_norm / options.residual_tolerance
    easy = (
        outcome.iterations <= 2
        and residual_ratio <= 0.1
        and outcome.correction_norm <= 0.1 * step
        and outcome.tangent_dot >= 0.98
        and state.jacobian_diagnostic.condition <= 0.1 * options.condition_limit
        and clear_of_event
    )
    difficult = (
        outcome.iterations >= max(3, options.corrector_maximum_iterations // 2)
        or residual_ratio >= 0.5
        or outcome.correction_norm >= 0.5 * step
        or outcome.tangent_dot < 0.95
        or not clear_of_event
    )
    if easy:
        step *= options.growth_factor
    elif difficult:
        step *= options.shrink_factor
    return min(options.maximum_step, max(options.minimum_step, step))


def _crossing_direction(previous: float, current: float) -> int:
    if previous < 0.0 <= current:
        return 1
    if previous > 0.0 >= current:
        return -1
    return 0


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


def target_residual(problem: FiberProblem, rotation: Array) -> Array:
    """Evaluate the declared target-local residual on a smooth-domain pose."""
    direction = problem.direction_evaluator(rotation)
    chart = problem.target_chart
    return chart.basis.T @ (direction - chart.direction)


def local_residual_jacobian(problem: FiberProblem, rotation: Array) -> Array:
    """Differentiate the target residual in right-trivialized coordinates."""
    zero = jnp.zeros(3, dtype=rotation.dtype)
    return jax.jacfwd(
        lambda delta: target_residual(problem, rotation @ exp(delta))
    )(zero)


def _correct_closure(
    problem: FiberProblem,
    options: ContinuationOptions,
    current: Array,
    seed: Array,
    initial_tangent: Array,
    current_tangent: Array,
    maximum_evaluations: int | None = None,
) -> _CorrectorOutcome:
    delta = jnp.zeros(3, dtype=current.dtype)
    last_residual = float("inf")
    last_update = float("inf")
    evaluations = 0

    def budget_exhausted(iteration: int, candidate: Array) -> _CorrectorOutcome:
        return _CorrectorOutcome(
            False,
            None,
            np.asarray(current),
            np.asarray(candidate),
            iteration,
            last_residual,
            last_update,
            float(jnp.linalg.norm(delta)),
            float(rotation_distance(current, candidate)),
            float("nan"),
            TerminationReason.EVALUATION_BUDGET,
            None,
            evaluations,
            "closure evaluation budget exhausted",
        )

    def closing_system(correction: Array) -> Array:
        candidate = current @ exp(correction)
        direction = problem.direction_evaluator(candidate)
        residual_value = problem.target_chart.basis.T @ (
            direction - problem.target_chart.direction
        )
        section = _section_coordinate(seed, candidate, initial_tangent)
        return jnp.concatenate((residual_value, jnp.atleast_1d(section)))

    for iteration in range(options.closure_maximum_iterations + 1):
        candidate = current @ exp(delta)
        if maximum_evaluations is not None and evaluations >= maximum_evaluations:
            return budget_exhausted(iteration, candidate)
        domain = _evaluate_domain(problem, candidate)
        evaluations += 1
        if not domain.valid:
            event = domain.event
            reason = event.kind if event is not None else TerminationReason.PATH_INFEASIBLE
            return _CorrectorOutcome(
                False,
                None,
                np.asarray(current),
                np.asarray(candidate),
                iteration,
                last_residual,
                last_update,
                float(jnp.linalg.norm(delta)),
                float(rotation_distance(current, candidate)),
                float("nan"),
                reason,
                event,
                evaluations,
                event.message if event is not None else "invalid closure domain",
            )
        try:
            value = closing_system(delta)
            value_array = np.asarray(value)
        except Exception as error:
            return _CorrectorOutcome(
                False,
                None,
                np.asarray(current),
                np.asarray(candidate),
                iteration,
                float("inf"),
                last_update,
                float(jnp.linalg.norm(delta)),
                float(rotation_distance(current, candidate)),
                float("nan"),
                TerminationReason.NON_FINITE,
                None,
                evaluations,
                f"closure evaluation failed: {type(error).__name__}: {error}",
            )
        last_residual = float(np.linalg.norm(value_array[:2]))
        section_norm = abs(float(value_array[2]))
        if not np.all(np.isfinite(value_array)):
            return _CorrectorOutcome(
                False,
                None,
                np.asarray(current),
                np.asarray(candidate),
                iteration,
                last_residual,
                last_update,
                float(jnp.linalg.norm(delta)),
                float(rotation_distance(current, candidate)),
                float("nan"),
                TerminationReason.NON_FINITE,
                None,
                evaluations,
                "closure system returned a non-finite value",
            )
        if (
            last_residual <= options.residual_tolerance
            and section_norm <= options.closure_section_tolerance
            and (last_update if np.isfinite(last_update) else 0.0)
            <= options.corrector_update_tolerance
        ):
            if maximum_evaluations is not None and evaluations >= maximum_evaluations:
                return budget_exhausted(iteration, candidate)
            state = _evaluate_regular_state(
                problem, options, candidate, previous_tangent=current_tangent
            )
            evaluations += 1
            tangent_dot = (
                float(jnp.dot(initial_tangent, state.tangent))
                if state.tangent is not None
                else float("nan")
            )
            correction_norm = float(jnp.linalg.norm(delta))
            advance = float(rotation_distance(current, candidate))
            if not state.accepted:
                return _CorrectorOutcome(
                    False,
                    state,
                    np.asarray(current),
                    np.asarray(candidate),
                    iteration,
                    state.residual_norm,
                    last_update if np.isfinite(last_update) else 0.0,
                    correction_norm,
                    advance,
                    tangent_dot,
                    state.reason,
                    state.event,
                    evaluations,
                    state.message,
                )
            accepted = (
                correction_norm <= options.maximum_correction
                and advance <= options.maximum_advance
                and tangent_dot >= options.closure_tangent_dot
                and float(rotation_distance(seed, candidate))
                <= options.closure_distance
            )
            return _CorrectorOutcome(
                accepted,
                state,
                np.asarray(current),
                None if accepted else np.asarray(candidate),
                iteration,
                state.residual_norm,
                last_update if np.isfinite(last_update) else 0.0,
                correction_norm,
                advance,
                tangent_dot,
                None if accepted else TerminationReason.TOPOLOGY_AMBIGUITY,
                None,
                evaluations,
                "closure corrector converged" if accepted else "closure gates failed",
            )
        if iteration == options.closure_maximum_iterations:
            break
        if maximum_evaluations is not None and evaluations >= maximum_evaluations:
            return budget_exhausted(iteration, candidate)
        try:
            jacobian = jax.jacfwd(closing_system)(delta)
            jacobian_array = np.asarray(jacobian)
            condition = float(np.linalg.cond(jacobian_array))
            if not np.all(np.isfinite(jacobian_array)) or not np.isfinite(condition):
                raise FloatingPointError("non-finite closure Jacobian")
            if condition > options.condition_limit:
                raise np.linalg.LinAlgError(
                    f"closure system condition {condition} exceeds limit"
                )
            update = jnp.linalg.solve(jacobian, -value)
            if not np.all(np.isfinite(np.asarray(update))):
                raise FloatingPointError("non-finite closure update")
        except (np.linalg.LinAlgError, FloatingPointError) as error:
            return _CorrectorOutcome(
                False,
                None,
                np.asarray(current),
                np.asarray(candidate),
                iteration,
                last_residual,
                last_update,
                float(jnp.linalg.norm(delta)),
                float(rotation_distance(current, candidate)),
                float("nan"),
                TerminationReason.LINEAR_SOLVE_FAILURE,
                None,
                evaluations,
                str(error),
            )
        last_update = float(jnp.linalg.norm(update))
        delta = delta + update

    candidate = current @ exp(delta)
    return _CorrectorOutcome(
        False,
        None,
        np.asarray(current),
        np.asarray(candidate),
        options.closure_maximum_iterations,
        last_residual,
        last_update,
        float(jnp.linalg.norm(delta)),
        float(rotation_distance(current, candidate)),
        float("nan"),
        TerminationReason.CORRECTOR_FAILURE,
        None,
        evaluations,
        "closure corrector iteration budget exhausted",
    )


def trace_fiber(
    problem: FiberProblem,
    options: ContinuationOptions | None = None,
) -> FiberResult:
    """Trace the regular component reachable from ``problem.seed``.

    The host controls bounded continuation and diagnostics while JAX evaluates
    each smooth local residual and Jacobian.  The result never claims global
    component completeness.
    """
    options = options or ContinuationOptions()
    seed = jnp.asarray(problem.seed)
    initial = _evaluate_regular_state(problem, options, seed)
    evaluations = 1
    if not initial.accepted:
        assert initial.reason is not None
        return _make_result(
            problem,
            options,
            initial.reason,
            [],
            [],
            [],
            [],
            _empty_closure_diagnostic(),
            rejected_pose=np.asarray(seed),
            event=initial.event,
            evaluations=evaluations,
            retries=0,
            message=initial.message,
        )
    seed_tolerance = options.residual_tolerance + options.relative_residual_tolerance
    if initial.residual_norm > seed_tolerance:
        return _make_result(
            problem,
            options,
            TerminationReason.INVALID_NUMERICAL_INPUT,
            [],
            [],
            [],
            [],
            _empty_closure_diagnostic(),
            rejected_pose=np.asarray(seed),
            event=None,
            evaluations=evaluations,
            retries=0,
            message=f"seed residual {initial.residual_norm} exceeds tolerance",
        )

    states = [initial]
    arclength_increments: list[float] = []
    step_diagnostics: list[StepDiagnostic] = []
    accepted_margins: list[Mapping[str, float]] = [dict(initial.domain.margins)]
    closure_diagnostic = _empty_closure_diagnostic()
    step = options.initial_step
    total_arclength = 0.0
    total_retries = 0
    previous_section = 0.0

    while True:
        accepted_steps = len(states) - 1
        if accepted_steps >= options.maximum_accepted_steps:
            return _make_result(
                problem,
                options,
                TerminationReason.STEP_BUDGET,
                states,
                arclength_increments,
                step_diagnostics,
                accepted_margins,
                closure_diagnostic,
                rejected_pose=None,
                event=None,
                evaluations=evaluations,
                retries=total_retries,
                message="accepted-step budget exhausted",
            )
        if evaluations >= options.maximum_evaluations:
            return _make_result(
                problem,
                options,
                TerminationReason.EVALUATION_BUDGET,
                states,
                arclength_increments,
                step_diagnostics,
                accepted_margins,
                closure_diagnostic,
                rejected_pose=None,
                event=None,
                evaluations=evaluations,
                retries=total_retries,
                message="evaluation budget exhausted",
            )

        current_state = states[-1]
        assert current_state.tangent is not None
        current = current_state.rotation
        trial_index = 0
        while True:
            predicted = current @ exp(
                jnp.asarray(step, dtype=current.dtype) * current_state.tangent
            )
            outcome = _correct_trial(
                problem,
                options,
                current,
                predicted,
                current_state.tangent,
                maximum_evaluations=options.maximum_evaluations - evaluations,
            )
            evaluations += outcome.evaluations
            diagnostic = _step_diagnostic(
                accepted_steps + 1, trial_index, step, outcome
            )
            step_diagnostics.append(diagnostic)

            if outcome.reason == TerminationReason.EVALUATION_BUDGET:
                return _make_result(
                    problem,
                    options,
                    TerminationReason.EVALUATION_BUDGET,
                    states,
                    arclength_increments,
                    step_diagnostics,
                    accepted_margins,
                    closure_diagnostic,
                    rejected_pose=outcome.rejected_pose,
                    event=None,
                    evaluations=evaluations,
                    retries=total_retries,
                    message=outcome.message,
                )
            if not outcome.accepted:
                assert outcome.reason is not None
                if outcome.reason in _EVENT_REASONS:
                    return _make_result(
                        problem,
                        options,
                        outcome.reason,
                        states,
                        arclength_increments,
                        step_diagnostics,
                        accepted_margins,
                        closure_diagnostic,
                        rejected_pose=outcome.rejected_pose,
                        event=outcome.event,
                        evaluations=evaluations,
                        retries=total_retries,
                        message=outcome.message,
                    )
                total_retries += 1
                if trial_index >= options.maximum_retries:
                    return _make_result(
                        problem,
                        options,
                        outcome.reason,
                        states,
                        arclength_increments,
                        step_diagnostics,
                        accepted_margins,
                        closure_diagnostic,
                        rejected_pose=outcome.rejected_pose,
                        event=outcome.event,
                        evaluations=evaluations,
                        retries=total_retries,
                        message=outcome.message,
                    )
                reduced_step = step * options.shrink_factor
                if reduced_step < options.minimum_step:
                    terminal_reason = (
                        TerminationReason.NON_FINITE
                        if outcome.reason == TerminationReason.NON_FINITE
                        else TerminationReason.STEP_UNDERFLOW
                    )
                    return _make_result(
                        problem,
                        options,
                        terminal_reason,
                        states,
                        arclength_increments,
                        step_diagnostics,
                        accepted_margins,
                        closure_diagnostic,
                        rejected_pose=outcome.rejected_pose,
                        event=outcome.event,
                        evaluations=evaluations,
                        retries=total_retries,
                        message=(
                            f"step reduction below minimum after {outcome.reason.value}: "
                            f"{reduced_step} < {options.minimum_step}"
                        ),
                    )
                step = reduced_step
                trial_index += 1
                continue

            assert outcome.state is not None
            if evaluations > options.maximum_evaluations:
                return _make_result(
                    problem,
                    options,
                    TerminationReason.EVALUATION_BUDGET,
                    states,
                    arclength_increments,
                    step_diagnostics,
                    accepted_margins,
                    closure_diagnostic,
                    rejected_pose=np.asarray(outcome.state.rotation),
                    event=None,
                    evaluations=evaluations,
                    retries=total_retries,
                    message="evaluation budget exhausted during a trial",
                )
            if total_arclength + outcome.advance > options.maximum_arclength:
                return _make_result(
                    problem,
                    options,
                    TerminationReason.ARCLENGTH_BUDGET,
                    states,
                    arclength_increments,
                    step_diagnostics,
                    accepted_margins,
                    closure_diagnostic,
                    rejected_pose=np.asarray(outcome.state.rotation),
                    event=None,
                    evaluations=evaluations,
                    retries=total_retries,
                    message="next accepted edge would exceed the arclength budget",
                )

            states.append(outcome.state)
            arclength_increments.append(outcome.advance)
            accepted_margins.append(dict(outcome.state.domain.margins))
            total_arclength += outcome.advance
            section = float(
                _section_coordinate(seed, outcome.state.rotation, initial.tangent)
            )
            crossing_direction = _crossing_direction(previous_section, section)
            crossed = previous_section != 0.0 and crossing_direction != 0
            seed_distance = float(rotation_distance(seed, outcome.state.rotation))
            tangent_dot = float(jnp.dot(initial.tangent, outcome.state.tangent))
            extent_gate = (
                len(states) - 1 >= options.closure_minimum_steps
                and total_arclength >= options.closure_minimum_arclength
            )
            closure_diagnostic = ClosureDiagnostic(
                accumulated_arclength=total_arclength,
                seed_distance=seed_distance,
                previous_section_value=previous_section,
                section_value=section,
                section_crossed=crossed,
                crossing_direction=crossing_direction,
                tangent_dot=tangent_dot,
                final_correction_attempted=False,
                final_correction_accepted=False,
            )
            if (
                extent_gate
                and crossed
                and seed_distance <= options.closure_distance
                and tangent_dot >= options.closure_tangent_dot
            ):
                closure = _correct_closure(
                    problem,
                    options,
                    outcome.state.rotation,
                    seed,
                    initial.tangent,
                    outcome.state.tangent,
                    maximum_evaluations=options.maximum_evaluations - evaluations,
                )
                evaluations += closure.evaluations
                closure_diagnostic = ClosureDiagnostic(
                    accumulated_arclength=total_arclength,
                    seed_distance=seed_distance,
                    previous_section_value=previous_section,
                    section_value=section,
                    section_crossed=crossed,
                    crossing_direction=crossing_direction,
                    tangent_dot=tangent_dot,
                    final_correction_attempted=True,
                    final_correction_accepted=closure.accepted,
                )
                if closure.reason == TerminationReason.EVALUATION_BUDGET:
                    return _make_result(
                        problem,
                        options,
                        TerminationReason.EVALUATION_BUDGET,
                        states,
                        arclength_increments,
                        step_diagnostics,
                        accepted_margins,
                        closure_diagnostic,
                        rejected_pose=closure.rejected_pose,
                        event=None,
                        evaluations=evaluations,
                        retries=total_retries,
                        message=closure.message,
                    )
                if closure.accepted:
                    assert closure.state is not None
                    previous_pose = states[-2].rotation
                    closing_advance = float(
                        rotation_distance(previous_pose, closure.state.rotation)
                    )
                    closing_arclength = (
                        total_arclength + closing_advance - outcome.advance
                    )
                    if evaluations > options.maximum_evaluations:
                        return _make_result(
                            problem,
                            options,
                            TerminationReason.EVALUATION_BUDGET,
                            states,
                            arclength_increments,
                            step_diagnostics,
                            accepted_margins,
                            closure_diagnostic,
                            rejected_pose=np.asarray(closure.state.rotation),
                            event=None,
                            evaluations=evaluations,
                            retries=total_retries,
                            message="evaluation budget exhausted during closure",
                        )
                    if closing_arclength > options.maximum_arclength:
                        return _make_result(
                            problem,
                            options,
                            TerminationReason.ARCLENGTH_BUDGET,
                            states,
                            arclength_increments,
                            step_diagnostics,
                            accepted_margins,
                            closure_diagnostic,
                            rejected_pose=np.asarray(closure.state.rotation),
                            event=None,
                            evaluations=evaluations,
                            retries=total_retries,
                            message="closure edge exceeds the arclength budget",
                        )
                    states[-1] = closure.state
                    arclength_increments[-1] = closing_advance
                    accepted_margins[-1] = dict(closure.state.domain.margins)
                    total_arclength = closing_arclength
                    closure_diagnostic = ClosureDiagnostic(
                        accumulated_arclength=total_arclength,
                        seed_distance=float(
                            rotation_distance(seed, closure.state.rotation)
                        ),
                        previous_section_value=previous_section,
                        section_value=float(
                            _section_coordinate(
                                seed, closure.state.rotation, initial.tangent
                            )
                        ),
                        section_crossed=True,
                        crossing_direction=crossing_direction,
                        tangent_dot=closure.tangent_dot,
                        final_correction_attempted=True,
                        final_correction_accepted=True,
                    )
                    return _make_result(
                        problem,
                        options,
                        TerminationReason.CLOSED_LOOP,
                        states,
                        arclength_increments,
                        step_diagnostics,
                        accepted_margins,
                        closure_diagnostic,
                        rejected_pose=None,
                        event=None,
                        evaluations=evaluations,
                        retries=total_retries,
                        message="all closure gates passed",
                    )
                if closure.reason in _EVENT_REASONS:
                    assert closure.reason is not None
                    return _make_result(
                        problem,
                        options,
                        closure.reason,
                        states,
                        arclength_increments,
                        step_diagnostics,
                        accepted_margins,
                        closure_diagnostic,
                        rejected_pose=closure.rejected_pose,
                        event=closure.event,
                        evaluations=evaluations,
                        retries=total_retries,
                        message=closure.message,
                    )

            previous_section = section
            step = _adapt_accepted_step(step, outcome, options)
            break
