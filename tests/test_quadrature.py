"""Pointwise integrand and factor availability of the line quadrature.

The integrator itself (resampled fixed grid) is covered by
``tests/test_resample_quadrature.py``; this file keeps the parts of the
quadrature module that do not depend on the integration method.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from lumice_integral.canonical_scene import canonical_pixel_problem
from lumice_integral.continuation import ContinuationOptions, trace_fiber
from lumice_integral.quadrature import (
    DENSITY_FACTOR_NAME,
    HAAR_TO_DVOL_G_FACTOR,
    INTEGRAND_FACTOR_NAMES,
    integrand_availability,
    integrand_expression,
    integrate_fiber_resampled,
    pointwise_integrand,
)

EPSILON = 1e-6


@pytest.fixture(scope="module")
def canonical():
    problem = canonical_pixel_problem()
    return problem, trace_fiber(problem)


def test_integrand_schema_is_explicit():
    assert DENSITY_FACTOR_NAME == "rho_pose"
    assert INTEGRAND_FACTOR_NAMES == ("entry_measure", "fresnel_transmission", "path_validity")
    assert HAAR_TO_DVOL_G_FACTOR == 1.0 / (8.0 * np.pi**2)
    np.testing.assert_allclose(
        integrand_expression(np.array([2.0, 3.0]), np.array([0.5, 1.0]), np.array([1.0, 0.0]), EPSILON),
        [1.0 / (1.0 + EPSILON), 3.0 / EPSILON],
    )


def test_pointwise_integrand_combines_named_factors_over_regularised_j_perp(canonical):
    problem, result = canonical
    values = pointwise_integrand(result, epsilon=EPSILON)

    assert values.shape == (len(result.poses),)
    assert values.dtype == np.float64
    assert np.all(np.isfinite(values)) and np.all(values > 0.0)
    observables = result.weight_observables
    for index in (0, len(values) // 3, len(values) - 1):
        expected = observables["rho_pose"].values[index]
        for name in INTEGRAND_FACTOR_NAMES:
            expected *= observables[name].values[index]
        expected /= result.jacobian_diagnostics[index].normal_jacobian + EPSILON
        assert values[index] == pytest.approx(expected, rel=0.0, abs=1e-12 * expected)
    # The gate factor is one everywhere on the canonical fiber; the raw
    # J_perp array in the result is untouched by epsilon.
    np.testing.assert_array_equal(observables["path_validity"].values, 1.0)
    assert min(d.normal_jacobian for d in result.jacobian_diagnostics) > 0.08
    with pytest.raises(ValueError):
        pointwise_integrand(result, epsilon=0.0)


def test_missing_density_factor_is_unavailable_not_silently_one(canonical):
    problem, result = canonical
    without_density = replace(
        problem,
        weight_evaluators={
            name: evaluator
            for name, evaluator in problem.weight_evaluators.items()
            if name != "rho_pose"
        },
    )
    assert integrand_availability(without_density) == "unavailable_missing_rho_pose"
    with pytest.raises(ValueError, match="unavailable_missing_rho_pose"):
        pointwise_integrand(
            trace_fiber(without_density, ContinuationOptions(maximum_accepted_steps=2)),
            epsilon=EPSILON,
        )

    quadrature = integrate_fiber_resampled(without_density, result)
    assert quadrature.status == "unavailable_missing_rho_pose"
    assert np.isnan(quadrature.value) and quadrature.node_count == 0

    geometry_only = canonical_pixel_problem(with_weights=False)
    assert integrand_availability(geometry_only) == "unavailable_missing_rho_pose"
