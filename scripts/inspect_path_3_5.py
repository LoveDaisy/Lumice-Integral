"""Inspect one regular point of the minimal 3-5 direction map."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from lumice_integral.analytic import tangent_basis
from lumice_integral.continuation import trace_implicit_fiber
from lumice_integral.optics import minimum_deviation_incident, path_3_5
from lumice_integral.so3 import exp


def main() -> None:
    rotation = exp(jnp.array([0.15, 0.08, -0.05], dtype=jnp.float64))
    incident = minimum_deviation_incident()
    evaluation = path_3_5(rotation, incident)
    target = evaluation.direction
    basis = tangent_basis(target)

    def local_residual(delta):
        direction = path_3_5(rotation @ exp(delta), incident).direction
        return basis.T @ (direction - target)

    jacobian = jax.jacfwd(local_residual)(jnp.zeros(3, dtype=jnp.float64))
    singular_values = jnp.linalg.svd(jacobian, compute_uv=False)

    def pose_residual(candidate):
        direction = path_3_5(candidate, incident).direction
        return basis.T @ (direction - target)

    trace = trace_implicit_fiber(rotation, pose_residual)
    report = {
        "incident": np.asarray(incident).tolist(),
        "outgoing": np.asarray(target).tolist(),
        "outgoing_norm": float(jnp.linalg.norm(target)),
        "entry_discriminant": float(evaluation.entry.discriminant),
        "exit_discriminant": float(evaluation.exit.discriminant),
        "entry_incidence_cosine": float(evaluation.entry.incidence_cosine),
        "exit_incidence_cosine": float(evaluation.exit.incidence_cosine),
        "jacobian": np.asarray(jacobian).tolist(),
        "singular_values": np.asarray(singular_values).tolist(),
        "trace_steps": trace.steps,
        "trace_max_residual": float(trace.residual_norms.max()),
        "trace_detected_gap": trace.detected_gap,
        "trace_closure_error": trace.closure_error,
    }
    for name, value in report.items():
        print(f"{name}={value}")


if __name__ == "__main__":
    main()
