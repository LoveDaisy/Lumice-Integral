"""Compare float32 and float64 on one complete 3-5 fiber."""

from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import numpy as np

from lumice_integral.analytic import tangent_basis
from lumice_integral.continuation import trace_fiber
from lumice_integral.optics import (
    minimum_deviation_incident,
    path_3_5,
    path_3_5_problem,
)
from lumice_integral.so3 import exp


def evaluate(dtype):
    rotation = exp(jnp.asarray([0.15, 0.08, -0.05], dtype=dtype))
    incident = minimum_deviation_incident(jnp.asarray(1.31, dtype=dtype))
    target = path_3_5(rotation, incident).direction
    basis = tangent_basis(target)
    zero = jnp.zeros(3, dtype=dtype)

    def local_residual(delta):
        direction = path_3_5(rotation @ exp(delta), incident).direction
        return basis.T @ (direction - target)

    jacobian = jax.jacfwd(local_residual)(zero)
    return rotation, incident, np.asarray(target), np.asarray(jacobian)


def main() -> None:
    rotation64, incident64, direction64, jacobian64 = evaluate(jnp.float64)
    rotation32, incident32, direction32, jacobian32 = evaluate(jnp.float32)
    trace64 = trace_fiber(path_3_5_problem(rotation64, incident64))
    try:
        path_3_5_problem(rotation32, incident32)
    except ValueError as error:
        float32_trace = {"rejected": str(error)}
    else:
        raise AssertionError("float32 unexpectedly entered the reference solver")
    report = {
        "direction_max_absolute_error": float(
            np.max(np.abs(direction32 - direction64))
        ),
        "jacobian_relative_error": float(
            np.linalg.norm(jacobian32 - jacobian64) / np.linalg.norm(jacobian64)
        ),
        "float64": {
            "status": trace64.status.value,
            "steps": trace64.terminal_payload.accepted_steps,
            "max_residual": float(trace64.residual_norms.max()),
            "closure_error_rad": trace64.closure_diagnostics.seed_distance,
        },
        "float32": float32_trace,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
