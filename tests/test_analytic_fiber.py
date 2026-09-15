from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from lumice_integral.analytic import BODY_AXIS, residual_after_update, tangent_basis
from lumice_integral.continuation import constraint_jacobian, trace_closed_fiber


def central_difference_jacobian(function, point, step=1e-5):
    columns = []
    for axis in np.eye(point.size):
        columns.append((function(point + step * axis) - function(point - step * axis)) / (2 * step))
    return np.stack(columns, axis=1)


def test_ad_jacobian_matches_central_difference():
    rotation = jnp.eye(3, dtype=jnp.float64)
    target = BODY_AXIS
    basis = tangent_basis(target)

    actual = np.asarray(constraint_jacobian(rotation, target, basis))
    expected = central_difference_jacobian(
        lambda delta: np.asarray(
            residual_after_update(jnp.asarray(delta), rotation, target, basis)
        ),
        np.zeros(3),
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)
    assert np.linalg.matrix_rank(actual) == 2


def test_analytic_fiber_closes_with_small_residual():
    initial = jnp.eye(3, dtype=jnp.float64)
    target = BODY_AXIS
    result = trace_closed_fiber(initial, target, tangent_basis(target))

    assert result.residual_norms.max() <= 1e-10
    assert result.closure_error <= 1e-8
    np.testing.assert_allclose(result.length, 2 * np.pi, rtol=0, atol=1e-14)
