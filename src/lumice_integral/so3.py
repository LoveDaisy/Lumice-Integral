"""Small SO(3) helpers using right-trivialized tangent coordinates.

Unit quaternions appear only as an interpolation chart (``(w, x, y, z)``,
scalar first): :func:`quaternion_from_rotation` / :func:`rotation_from_quaternion`
convert to and from the ``(3, 3)`` matrices everything else works with, and
:func:`continuous_quaternion_signs` fixes the ``q ~ -q`` ambiguity along a
sequence so that componentwise splines see a continuous curve.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jax import Array, lax


def _rodrigues_taylor(value: Array) -> tuple[Array, Array]:
    value_squared = value * value
    a = 1.0 - value / 6.0 + value_squared / 120.0
    b = 0.5 - value / 24.0 + value_squared / 720.0
    return a, b


def _rodrigues_regular(value: Array) -> tuple[Array, Array]:
    theta = jnp.sqrt(value)
    return jnp.sin(theta) / theta, (1.0 - jnp.cos(theta)) / value


def hat(vector: Array) -> Array:
    """Return the skew matrix whose action is ``vector x operand``."""
    x, y, z = vector
    return jnp.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=vector.dtype,
    )


def exp(rotation_vector: Array) -> Array:
    """Map a rotation vector to SO(3) with a zero-safe Rodrigues formula."""
    theta_squared = jnp.dot(rotation_vector, rotation_vector)
    generator = hat(rotation_vector)

    a, b = lax.cond(
        theta_squared < 1e-8,
        _rodrigues_taylor,
        _rodrigues_regular,
        theta_squared,
    )
    return jnp.eye(3, dtype=rotation_vector.dtype) + a * generator + b * (
        generator @ generator
    )


def _log_taylor(theta_squared: Array) -> Array:
    return 1.0 + theta_squared / 6.0 + 7.0 * theta_squared * theta_squared / 360.0


def _log_regular(theta_squared: Array) -> Array:
    theta = jnp.sqrt(theta_squared)
    return theta / jnp.sin(theta)


def log(rotation: Array) -> Array:
    """Map a rotation to its rotation vector on the injectivity domain.

    Zero-safe like :func:`exp`; the angle is recovered from ``arctan2`` as in
    :func:`rotation_distance`.  Rotations with angle close to ``pi`` are not
    supported (the axis becomes ill-conditioned there).
    """
    cosine = jnp.clip((jnp.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    skew_vector = jnp.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ]
    ) / 2.0
    sine_squared = jnp.dot(skew_vector, skew_vector)
    # Zero-safe norm: keeps the AD graph finite at the identity.
    positive = sine_squared > 0.0
    sine = jnp.where(positive, jnp.sqrt(jnp.where(positive, sine_squared, 1.0)), 0.0)
    theta = jnp.arctan2(sine, cosine)
    scale = lax.cond(theta * theta < 1e-8, _log_taylor, _log_regular, theta * theta)
    return scale * skew_vector


def rotation_distance(left: Array, right: Array) -> Array:
    """Return the geodesic angle between two rotation matrices.

    The angle of ``left.T @ right`` from ``arctan2(sine, cosine)``: the cosine
    from the trace, the sine from the norm of the skew part, which stays
    accurate near the identity where ``arccos`` of the trace does not.
    Differentiable (the continuation's step control differentiates through
    it); :func:`rotation_distances` is the host-side batch of the same
    formula and must stay numerically equivalent (``tests/test_so3.py``).
    """
    relative = left.T @ right
    cosine = jnp.clip((jnp.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    skew_vector = jnp.array(
        [
            relative[2, 1] - relative[1, 2],
            relative[0, 2] - relative[2, 0],
            relative[1, 0] - relative[0, 1],
        ]
    ) / 2.0
    sine = jnp.linalg.norm(skew_vector)
    return jnp.arctan2(sine, cosine)


def rotation_distances(left: np.ndarray, rights: np.ndarray) -> np.ndarray:
    """Geodesic angles from ``left`` ``(3, 3)`` to each of ``rights`` ``(N, 3, 3)``.

    Host-side numpy batch of :func:`rotation_distance` (same trace / skew
    vector / ``arctan2`` derivation, evaluated once per member) for callers
    that only compare thousands of poses against one and never
    differentiate: candidate-pool clustering and curve deduplication in
    ``discovery.py``.  A ``jax.vmap`` of :func:`rotation_distance` would be
    recompiled for every distinct ``N``, and ``N`` changes with every pixel
    (task-pixel-cost-shape-stable-kernels); numpy has no shape
    specialisation at all.  Equivalence with :func:`rotation_distance` is
    locked by ``tests/test_so3.py``.
    """
    left = np.asarray(left, dtype=np.float64)
    rights = np.asarray(rights, dtype=np.float64)
    if left.shape != (3, 3) or rights.ndim != 3 or rights.shape[1:] != (3, 3):
        raise ValueError("left must have shape (3, 3) and rights shape (N, 3, 3)")
    relative = left.T @ rights
    cosine = np.clip((np.trace(relative, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    skew_vectors = (
        np.stack(
            [
                relative[:, 2, 1] - relative[:, 1, 2],
                relative[:, 0, 2] - relative[:, 2, 0],
                relative[:, 1, 0] - relative[:, 0, 1],
            ],
            axis=1,
        )
        / 2.0
    )
    sine = np.linalg.norm(skew_vectors, axis=1)
    return np.arctan2(sine, cosine)


def quaternion_from_rotation(rotation: Array) -> Array:
    """Unit quaternion ``(w, x, y, z)`` of a rotation matrix (Shepperd's method).

    Branch-free (all four candidate pivots are formed and the largest one is
    selected), so it is ``jax.vmap``-safe and well conditioned for every
    rotation including angles near ``pi`` where :func:`log` is not.  The sign
    is the pivot's own; use :func:`continuous_quaternion_signs` on sequences.
    """
    m = rotation
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    # Pivot k = 0 on the trace, k = 1..3 on the diagonal entries; each 4-vector
    # is (w, x, y, z) times 4 * pivot component, a positive multiple of q.
    candidates = jnp.stack(
        [
            jnp.array([1.0 + trace, m[2, 1] - m[1, 2], m[0, 2] - m[2, 0], m[1, 0] - m[0, 1]]),
            jnp.array([m[2, 1] - m[1, 2], 1.0 + m[0, 0] - m[1, 1] - m[2, 2], m[1, 0] + m[0, 1], m[0, 2] + m[2, 0]]),
            jnp.array([m[0, 2] - m[2, 0], m[1, 0] + m[0, 1], 1.0 - m[0, 0] + m[1, 1] - m[2, 2], m[2, 1] + m[1, 2]]),
            jnp.array([m[1, 0] - m[0, 1], m[0, 2] + m[2, 0], m[2, 1] + m[1, 2], 1.0 - m[0, 0] - m[1, 1] + m[2, 2]]),
        ]
    )
    pivots = jnp.array([1.0 + trace, 1.0 + m[0, 0] - m[1, 1] - m[2, 2], 1.0 - m[0, 0] + m[1, 1] - m[2, 2], 1.0 - m[0, 0] - m[1, 1] + m[2, 2]])
    best = jnp.argmax(pivots)
    quaternion = candidates[best]
    return quaternion / jnp.linalg.norm(quaternion)


def rotation_from_quaternion(quaternion: Array) -> Array:
    """Rotation matrix of a quaternion ``(w, x, y, z)``, normalised first.

    Differentiable in the quaternion, so ``jax.jvp`` through it turns the
    derivative of a componentwise quaternion spline into a body angular
    velocity (:func:`body_velocity`).
    """
    q = quaternion / jnp.linalg.norm(quaternion)
    w, x, y, z = q
    return jnp.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=quaternion.dtype,
    )


def quaternion_derivative(quaternion: Array, body_velocity: Array) -> Array:
    """``dq/ds = q * (0, omega) / 2`` for the body angular velocity ``omega``.

    Consistent with :func:`rotation_from_quaternion` and the right-trivialized
    convention ``dR/ds = R hat(omega)`` used by :func:`exp`; the caller's unit
    fiber tangent is such an ``omega``.
    """
    w, x, y, z = quaternion
    ox, oy, oz = body_velocity
    return 0.5 * jnp.array(
        [
            -x * ox - y * oy - z * oz,
            w * ox + y * oz - z * oy,
            w * oy + z * ox - x * oz,
            w * oz + x * oy - y * ox,
        ]
    )


def vee(matrix: Array) -> Array:
    """Inverse of :func:`hat` on a skew-symmetric matrix."""
    return jnp.array([matrix[2, 1], matrix[0, 2], matrix[1, 0]])


def continuous_quaternion_signs(quaternions: np.ndarray) -> np.ndarray:
    """Flip signs along a ``(M, 4)`` sequence so consecutive dot products are positive.

    Host-side and sequential (each sign depends on the previous one).  A loop
    is *not* made periodic: for a closed fiber in the non-trivial class of
    ``pi_1(SO(3)) = Z/2`` the propagated last sign is ``-q_0``, which a caller
    handles locally by aligning wrapped neighbours pairwise (see
    ``resample.py``; task-resample-and-integrate Step 0 fact 2).
    """
    signed = np.array(quaternions, dtype=np.float64, copy=True)
    if signed.ndim != 2 or signed.shape[1] != 4:
        raise ValueError("quaternions must have shape (M, 4)")
    for index in range(1, len(signed)):
        if float(np.dot(signed[index], signed[index - 1])) < 0.0:
            signed[index] = -signed[index]
    return signed
