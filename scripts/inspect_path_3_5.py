"""Inspect one regular point of the minimal 3-5 direction map."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from lumice_integral.continuation import local_residual_jacobian, trace_fiber
from lumice_integral.optics import (
    minimum_deviation_incident,
    path_3_5,
    path_3_5_problem,
)
from lumice_integral.so3 import exp


def main() -> None:
    rotation = exp(jnp.array([0.15, 0.08, -0.05], dtype=jnp.float64))
    incident = minimum_deviation_incident()
    evaluation = path_3_5(rotation, incident)
    target = evaluation.direction
    problem = path_3_5_problem(rotation, incident, target_direction=target)
    jacobian = local_residual_jacobian(problem, rotation)
    singular_values = jnp.linalg.svd(jacobian, compute_uv=False)
    trace = trace_fiber(problem)
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
        "trace_status": trace.status.value,
        "trace_reason": trace.reason.value,
        "trace_steps": trace.terminal_payload.accepted_steps,
        "trace_max_residual": float(trace.residual_norms.max()),
        "trace_length": float(trace.arclength_increments.sum()),
        "trace_detected_gap": trace.closure_diagnostics.seed_distance,
        "trace_closure_error": trace.closure_diagnostics.seed_distance,
    }
    for name, value in report.items():
        print(f"{name}={value}")


if __name__ == "__main__":
    main()
