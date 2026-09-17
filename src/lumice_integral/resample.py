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
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from .continuation import FiberResult, FiberStatus
from .so3 import (
    continuous_quaternion_signs,
    quaternion_derivative,
    quaternion_from_rotation,
    rotation_from_quaternion,
    vee,
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


def fiber_spline(result: FiberResult) -> FiberSpline:
    """Build the predictor spline of a traced fiber (closed loop or open arc).

    Requires at least two accepted poses (three for a closed loop, whose
    duplicated closing sample is dropped) and aligned ``tangents`` /
    ``arclength_increments``.
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


def resample_fiber(result: FiberResult, node_count: int) -> ResampledPredictors:
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
    "ResampledPredictors",
    "evaluate_spline",
    "fiber_spline",
    "resample_fiber",
    "resample_spline",
    "uniform_parameters",
]
