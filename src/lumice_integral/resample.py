"""Spline resampling of a traced fiber into uniformly parametrised predictor poses.

The accepted samples of a :class:`.continuation.FiberResult` are sparse and
adaptively spaced.  This module turns them into a smooth *predictor* curve
and evaluates it at ``N`` equally spaced parameter values, so that the
quadrature (:func:`.quadrature.integrate_fiber_resampled`) can retract all
``N`` points onto the fiber in one batch instead of one host-side Newton
solve per refinement node.

Chart and spline.  Every accepted pose is converted to a unit quaternion
(:func:`.so3.quaternion_from_rotation`) with signs propagated along the
sample order (:func:`.so3.continuous_quaternion_signs`), and the curve is a
componentwise cubic Hermite spline between consecutive samples.  The knot
parameter ``t`` is the cumulative chord (``FiberResult.arclength_increments``,
the geodesic distance between consecutive accepted poses), and the knot
derivatives are the *exact* fiber tangents the trace already computed,
``dq/dt = q (0, tau) / 2`` (:func:`.so3.quaternion_derivative`), not finite
differences.  The spline is therefore C^1 through every knot with the true
tangent direction, and it stays local: each segment depends only on its two
end samples, so a closed loop needs no global periodic fit and is immune to
the ``q_end = -q_0`` case of the non-trivial class of ``pi_1(SO(3)) = Z/2``
(the wrapped neighbour is sign-aligned pairwise; task-resample-and-integrate
plan D1'/D4, Step 0 fact 2: all four ch06 fixtures propagate to ``+q_0``).

Closed loops drop the duplicated final sample (the closure corrector lands on
the seed to ~1e-15, Step 0 fact 1) and close the last knot interval back to
the first sample; open arcs keep both end samples exactly (natural boundary:
the end tangents are the trace's own, no neighbour is fabricated).

The spline is only a predictor: ``t`` is *not* arclength (chord and arclength
differ at ``O(h^2)`` on a non-geodesic fiber, ``scratchpad/learnings.md``
code-quality) and the normalised quaternion curve does not lie on the fiber.
Both are absorbed downstream: the retraction moves each point onto the fiber
and the implicit-function speed ``ds/dt`` is evaluated exactly there.

Inputs.  :func:`fiber_spline` reads only the structural attributes declared by
:class:`TraceLike`; both :class:`.continuation.FiberResult` (one trace from a
seed) and :class:`OpenArc` (two traces from the same seed stitched into one
curve, task-pixel-pipeline-v2) satisfy it.  The two are deliberately not
related by inheritance: an ``OpenArc`` is a *pair* of traces with its own
two-ended event bookkeeping, and the Protocol makes the shared contract
checkable instead of a documented convention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import jax
import jax.numpy as jnp
import numpy as np

from .continuation import BranchDiagnostic, FiberResult, FiberStatus, TerminationReason
from .so3 import (
    continuous_quaternion_signs,
    quaternion_derivative,
    quaternion_from_rotation,
    rotation_from_quaternion,
    vee,
)


@runtime_checkable
class TraceLike(Protocol):
    """What the resampling and the quadrature read from a traced curve.

    ``poses`` ``(N, 3, 3)``, ``tangents`` ``(N, 3)`` (unit fiber tangents in
    the body frame, oriented along increasing sample order),
    ``arclength_increments`` ``(N - 1,)`` (geodesic chords between consecutive
    poses, all positive), ``status``/``reason`` of the curve's terminal end,
    and ``branch_diagnostics.accepted_margins`` aligned with ``poses`` (the
    event margins the endpoint truncation estimates read).
    """

    @property
    def poses(self) -> np.ndarray: ...

    @property
    def tangents(self) -> np.ndarray: ...

    @property
    def arclength_increments(self) -> np.ndarray: ...

    @property
    def status(self) -> FiberStatus: ...

    @property
    def reason(self) -> TerminationReason: ...

    @property
    def branch_diagnostics(self) -> BranchDiagnostic: ...


@dataclass(frozen=True)
class OpenArc:
    """Two traces from one seed, stitched into one curve with an event at each end.

    Built by :func:`stitch_open_arc` from a ``forward`` trace and a
    ``backward`` trace (``ContinuationOptions.initial_tangent_sign = -1``) of
    the same corrected seed, both terminated by a named event.  The stitched
    curve runs from the backward end through the seed to the forward end:
    ``poses[seed_index]`` is the seed, the reversed backward tangents are
    negated so every ``tangents[i]`` points along increasing ``i``, and the
    seed knot carries ``forward.tangents[0]``.

    ``status`` is :attr:`.continuation.FiberStatus.EVENT_TERMINATED` and
    ``reason`` is the *forward* end's event (the terminal end of the
    parametrisation, as for a one-sided arc); ``start_reason`` is the backward
    end's.  ``branch_diagnostics.accepted_margins`` is aligned with ``poses``
    so the quadrature can estimate the truncation at both ends.  The one
    continuation-level scope statement still holds: one component reached
    from one seed, completeness unknown.
    """

    forward: FiberResult
    backward: FiberResult
    poses: np.ndarray
    tangents: np.ndarray
    arclength_increments: np.ndarray
    branch_diagnostics: BranchDiagnostic
    seed_index: int
    status: FiberStatus = FiberStatus.EVENT_TERMINATED

    @property
    def reason(self) -> TerminationReason:
        return self.forward.reason

    @property
    def start_reason(self) -> TerminationReason:
        return self.backward.reason

    @property
    def arclength(self) -> float:
        return float(np.sum(self.arclength_increments)) if len(self.poses) > 1 else 0.0


def stitch_open_arc(forward: FiberResult, backward: FiberResult) -> OpenArc:
    """Stitch a forward and a backward trace of the same seed (see :class:`OpenArc`).

    Both traces must start at the same pose (the seed, index 0 of each) and
    have at least one accepted pose; neither is required to have advanced
    (a trace that met its event at the seed contributes only the seed).  The
    tangent sign convention is the one the Step 0 probe of
    task-pixel-pipeline-v2 verified on an analytic two-sided arc: the
    stitched spline is C^1 through the seed, ``ds/dt > 0`` everywhere, and its
    integral equals the sum of the two one-sided integrals.
    """
    if len(forward.poses) < 1 or len(backward.poses) < 1:
        raise ValueError("both traces need at least the seed pose")
    if not np.allclose(forward.poses[0], backward.poses[0], atol=1e-12):
        raise ValueError("forward and backward traces do not start at the same seed")
    back_poses = np.asarray(backward.poses[1:][::-1], dtype=np.float64)
    back_tangents = -np.asarray(backward.tangents[1:][::-1], dtype=np.float64)
    back_increments = np.asarray(backward.arclength_increments[::-1], dtype=np.float64)
    back_margins = tuple(backward.branch_diagnostics.accepted_margins[1:][::-1])
    return OpenArc(
        forward=forward,
        backward=backward,
        poses=np.concatenate((back_poses, np.asarray(forward.poses, dtype=np.float64))),
        tangents=np.concatenate((back_tangents, np.asarray(forward.tangents, dtype=np.float64))),
        arclength_increments=np.concatenate(
            (back_increments, np.asarray(forward.arclength_increments, dtype=np.float64))
        ),
        branch_diagnostics=BranchDiagnostic(
            path=forward.branch_diagnostics.path,
            accepted_margins=back_margins + tuple(forward.branch_diagnostics.accepted_margins),
            terminal_margins=forward.branch_diagnostics.terminal_margins,
        ),
        seed_index=len(back_poses),
    )


@dataclass(frozen=True)
class FiberSpline:
    """Sign-continuous cubic Hermite quaternion spline through the accepted poses.

    ``knots`` (``K + 1`` cumulative chord parameters starting at ``0``),
    ``quaternions`` and ``derivatives`` (``(K + 1, 4)``) describe ``K``
    Hermite segments.  For a closed fiber the last knot is the first sample
    again (sign-aligned with its predecessor) and ``total`` is the full loop
    chord; for an open arc the last knot is the last accepted pose.
    """

    closed: bool
    knots: np.ndarray
    quaternions: np.ndarray
    derivatives: np.ndarray

    @property
    def total(self) -> float:
        return float(self.knots[-1])

    @property
    def segment_count(self) -> int:
        return len(self.knots) - 1


@dataclass(frozen=True)
class ResampledPredictors:
    """``N`` predictor poses at uniform parameters ``t in [0, total]``.

    ``body_velocities`` are ``vee(P^T dP/dt)`` of the normalised spline curve
    (the predictor's velocity in right-trivialized coordinates) and
    ``phase_tangents`` their unit vectors: the direction the retraction keeps
    its correction orthogonal to, and the reference the fiber tangent is
    oriented along.
    """

    parameters: np.ndarray
    rotations: np.ndarray
    body_velocities: np.ndarray
    phase_tangents: np.ndarray
    closed: bool
    total: float


def _quaternion_from_rotation_batch(rotations: np.ndarray) -> np.ndarray:
    return np.asarray(
        jax.vmap(quaternion_from_rotation)(jnp.asarray(rotations, dtype=jnp.float64)),
        dtype=np.float64,
    )


def _quaternion_derivative_batch(quaternions: np.ndarray, tangents: np.ndarray) -> np.ndarray:
    return np.asarray(
        jax.vmap(quaternion_derivative)(
            jnp.asarray(quaternions, dtype=jnp.float64), jnp.asarray(tangents, dtype=jnp.float64)
        ),
        dtype=np.float64,
    )


def fiber_spline(result: TraceLike) -> FiberSpline:
    """Build the predictor spline of a traced curve (closed loop or open arc).

    ``result`` is any :class:`TraceLike` (a :class:`.continuation.FiberResult`
    or a stitched :class:`OpenArc`).  Requires at least two accepted poses
    (three for a closed loop, whose duplicated closing sample is dropped) and
    aligned ``tangents`` / ``arclength_increments``.
    """
    poses = np.asarray(result.poses, dtype=np.float64)
    tangents = np.asarray(result.tangents, dtype=np.float64)
    increments = np.asarray(result.arclength_increments, dtype=np.float64)
    sample_count = len(poses)
    if tangents.shape != (sample_count, 3) or increments.shape != (sample_count - 1,):
        raise ValueError("fiber tangents and arclength increments do not align with poses")
    closed = result.status == FiberStatus.CLOSED
    if sample_count < (3 if closed else 2):
        raise ValueError("resampling needs at least two distinct accepted poses")
    if not np.all(increments > 0.0):
        raise ValueError("arclength increments must be positive")

    if closed:
        # The closing sample is the seed again (Step 0 fact 1); the last
        # increment is the chord from the last distinct pose back to it.
        distinct = sample_count - 1
        quaternions = continuous_quaternion_signs(_quaternion_from_rotation_batch(poses[:distinct]))
        wrapped = quaternions[0]
        if float(np.dot(wrapped, quaternions[-1])) < 0.0:
            wrapped = -wrapped
        quaternions = np.vstack((quaternions, wrapped[None, :]))
        knot_tangents = np.vstack((tangents[:distinct], tangents[0][None, :]))
    else:
        quaternions = continuous_quaternion_signs(_quaternion_from_rotation_batch(poses))
        knot_tangents = tangents
    knots = np.concatenate(([0.0], np.cumsum(increments)))
    # dq/dt with ds/dt taken as 1 at the knots (t is the cumulative chord).
    derivatives = _quaternion_derivative_batch(quaternions, knot_tangents)
    return FiberSpline(closed, knots, quaternions, derivatives)


def evaluate_spline(spline: FiberSpline, parameters: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unnormalised quaternion curve and its ``t`` derivative at ``parameters``."""
    t = np.asarray(parameters, dtype=np.float64)
    if np.any(t < 0.0) or np.any(t > spline.total):
        raise ValueError("spline parameters must lie in [0, total]")
    segment = np.clip(np.searchsorted(spline.knots, t, side="right") - 1, 0, spline.segment_count - 1)
    left, right = spline.knots[segment], spline.knots[segment + 1]
    width = right - left
    u = (t - left) / width
    u2, u3 = u * u, u * u * u
    h00, h10 = 2.0 * u3 - 3.0 * u2 + 1.0, u3 - 2.0 * u2 + u
    h01, h11 = -2.0 * u3 + 3.0 * u2, u3 - u2
    d00, d10 = 6.0 * u2 - 6.0 * u, 3.0 * u2 - 4.0 * u + 1.0
    d01, d11 = -6.0 * u2 + 6.0 * u, 3.0 * u2 - 2.0 * u
    q_left, q_right = spline.quaternions[segment], spline.quaternions[segment + 1]
    m_left = width[:, None] * spline.derivatives[segment]
    m_right = width[:, None] * spline.derivatives[segment + 1]
    position = (
        h00[:, None] * q_left + h10[:, None] * m_left + h01[:, None] * q_right + h11[:, None] * m_right
    )
    derivative = (
        d00[:, None] * q_left + d10[:, None] * m_left + d01[:, None] * q_right + d11[:, None] * m_right
    ) / width[:, None]
    return position, derivative


@jax.jit
def _rotation_and_body_velocity_kernel(quaternions: jnp.ndarray, derivatives: jnp.ndarray):
    def one(quaternion, derivative):
        rotation, rotation_rate = jax.jvp(rotation_from_quaternion, (quaternion,), (derivative,))
        return rotation, vee(rotation.T @ rotation_rate)

    return jax.vmap(one)(quaternions, derivatives)


def resample_fiber(result: TraceLike, node_count: int) -> ResampledPredictors:
    """``node_count`` predictor poses at uniform parameters over the whole fiber."""
    spline = fiber_spline(result)
    return resample_spline(spline, uniform_parameters(spline, node_count))


def uniform_parameters(spline: FiberSpline, node_count: int) -> np.ndarray:
    if node_count < 2:
        raise ValueError("node_count must be at least 2")
    return np.linspace(0.0, spline.total, node_count)


def resample_spline(spline: FiberSpline, parameters: np.ndarray) -> ResampledPredictors:
    """Predictor poses, body velocities and phase tangents at ``parameters``."""
    parameters = np.asarray(parameters, dtype=np.float64)
    quaternions, derivatives = evaluate_spline(spline, parameters)
    rotations, velocities = _rotation_and_body_velocity_kernel(
        jnp.asarray(quaternions), jnp.asarray(derivatives)
    )
    rotations = np.asarray(rotations, dtype=np.float64)
    velocities = np.asarray(velocities, dtype=np.float64)
    speeds = np.linalg.norm(velocities, axis=1)
    if not np.all(speeds > 0.0):
        raise ValueError("predictor curve has a stationary point; cannot define a phase tangent")
    return ResampledPredictors(
        parameters=parameters,
        rotations=rotations,
        body_velocities=velocities,
        phase_tangents=velocities / speeds[:, None],
        closed=spline.closed,
        total=spline.total,
    )


__all__ = [
    "FiberSpline",
    "OpenArc",
    "ResampledPredictors",
    "TraceLike",
    "evaluate_spline",
    "fiber_spline",
    "resample_fiber",
    "resample_spline",
    "stitch_open_arc",
    "uniform_parameters",
]
