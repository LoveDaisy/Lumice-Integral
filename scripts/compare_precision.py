"""Compare float32 and float64 on one complete 3-5 fiber."""

from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import numpy as np

from lumice_integral.analytic import tangent_basis
from lumice_integral.continuation import trace_implicit_fiber
from lumice_integral.optics import minimum_deviation_incident, path_3_5
from lumice_integral.so3 import exp


def evaluate(dtype, tolerance):
    rotation = exp(jnp.asarray([0.15, 0.08, -0.05], dtype=dtype))
    incident = minimum_deviation_incident(jnp.asarray(1.31, dtype=dtype))
    target = path_3_5(rotation, incident).direction
    basis = tangent_basis(target)
    zero = jnp.zeros(3, dtype=dtype)

    def local_residual(delta):
        direction = path_3_5(rotation @ exp(delta), incident).direction
        return basis.T @ (direction - target)

    jacobian = jax.jacfwd(local_residual)(zero)
    trace = trace_implicit_fiber(
        rotation,
        lambda candidate: basis.T
        @ (path_3_5(candidate, incident).direction - target),
        tolerance=tolerance,
    )
    return np.asarray(target), np.asarray(jacobian), trace


def main() -> None:
    direction64, jacobian64, trace64 = evaluate(jnp.float64, 1e-11)
    direction32, jacobian32, trace32 = evaluate(jnp.float32, 1e-6)
    report = {
        "direction_max_absolute_error": float(
            np.max(np.abs(direction32 - direction64))
        ),
        "jacobian_relative_error": float(
            np.linalg.norm(jacobian32 - jacobian64) / np.linalg.norm(jacobian64)
        ),
        "float64": {
            "steps": trace64.steps,
            "max_residual": float(trace64.residual_norms.max()),
            "closure_error_rad": trace64.closure_error,
        },
        "float32": {
            "steps": trace32.steps,
            "max_residual": float(trace32.residual_norms.max()),
            "closure_error_rad": trace32.closure_error,
        },
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
