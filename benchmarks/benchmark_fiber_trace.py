"""Measure warm single-fiber continuation against the compile pass.

The same ``FiberProblem`` instance is traced twice per seed: the first call
absorbs kernel compilation, the second is the steady-state figure.  Reusing
one instance matters because the compiled kernels are cached on the identity
of ``problem.direction_evaluator``; a fresh problem recompiles.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time

import jax
import jax.numpy as jnp

from lumice_integral.continuation import ContinuationOptions, trace_fiber
from lumice_integral.optics import (
    minimum_deviation_incident,
    path_3_5,
    path_3_5_problem,
)
from lumice_integral.so3 import exp

SEEDS = {
    "canonical": (0.15, 0.08, -0.05),
    "run2": (0.16, 0.07, -0.04),
}


def timed_trace(problem, options):
    started = time.perf_counter()
    result = trace_fiber(problem, options)
    return time.perf_counter() - started, result


def benchmark_seed(name: str, seed_vector, options: ContinuationOptions, repeats: int):
    incident = minimum_deviation_incident()
    rotation = exp(jnp.asarray(seed_vector, dtype=jnp.float64))
    target = path_3_5(rotation, incident).direction
    problem = path_3_5_problem(rotation, incident, target_direction=target)

    compile_seconds, first = timed_trace(problem, options)
    warm_samples = []
    for _ in range(repeats):
        seconds, result = timed_trace(problem, options)
        warm_samples.append(seconds)
    warm_seconds = statistics.median(warm_samples)
    payload = result.terminal_payload
    steps = payload.accepted_steps
    evaluations = payload.evaluations
    return {
        "seed": name,
        "seed_vector": list(seed_vector),
        "status": result.status.value,
        "reason": result.reason.value,
        "accepted_steps": steps,
        "evaluations": evaluations,
        "max_residual": float(result.residual_norms.max()),
        "arclength": float(result.arclength_increments.sum()),
        "closure_gap": result.closure_diagnostics.seed_distance,
        "compile_seconds": compile_seconds,
        "warm_seconds_samples": warm_samples,
        "warm_seconds": warm_seconds,
        "compile_to_warm_ratio": compile_seconds / warm_seconds,
        "warm_us_per_step": warm_seconds / steps * 1e6,
        "warm_us_per_evaluation": warm_seconds / evaluations * 1e6,
        "first_call_matches_warm_result": (
            first.terminal_payload.accepted_steps == steps
            and first.terminal_payload.evaluations == evaluations
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--seeds", default=",".join(SEEDS), help="comma-separated subset of seeds"
    )
    args = parser.parse_args()

    options = ContinuationOptions()
    results = [
        benchmark_seed(name, SEEDS[name], options, args.repeats)
        for name in args.seeds.split(",")
    ]

    device = jax.devices()[0]
    report = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "jax": jax.__version__,
        "device": str(device),
        "backend": device.platform,
        "x64_enabled": bool(jax.config.x64_enabled),
        "dtype": options.dtype,
        "repeats": args.repeats,
        "results": results,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
