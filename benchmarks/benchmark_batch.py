"""Benchmark batched 3-5 direction and Jacobian evaluation."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time

import jax
import jax.numpy as jnp

from lumice_integral.optics import minimum_deviation_incident, path_3_5
from lumice_integral.so3 import exp


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", default="1,8,64,512,4096")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()

    dtype = jnp.float64 if args.dtype == "float64" else jnp.float32
    incident = minimum_deviation_incident(jnp.asarray(1.31, dtype=dtype))
    zero = jnp.zeros(3, dtype=dtype)

    def evaluate_one(rotation):
        def direction_after_update(delta):
            return path_3_5(rotation @ exp(delta), incident).direction

        direction = direction_after_update(zero)
        jacobian = jax.jacfwd(direction_after_update)(zero)
        return direction, jacobian

    evaluate_batch = jax.jit(jax.vmap(evaluate_one))
    results = []
    for batch_size in (int(value) for value in args.batch_sizes.split(",")):
        parameter = jnp.linspace(0.0, 2.0 * jnp.pi, batch_size, dtype=dtype)
        perturbations = jnp.stack(
            (
                0.02 * jnp.sin(parameter),
                0.02 * jnp.cos(parameter),
                0.01 * jnp.sin(2.0 * parameter),
            ),
            axis=1,
        )
        base = jnp.asarray([0.15, 0.08, -0.05], dtype=dtype)
        rotations = jax.vmap(exp)(base + perturbations)

        started = time.perf_counter()
        evaluate_batch(rotations)[0].block_until_ready()
        compile_seconds = time.perf_counter() - started

        samples = []
        for _ in range(args.repeats):
            started = time.perf_counter()
            evaluate_batch(rotations)[0].block_until_ready()
            samples.append(time.perf_counter() - started)
        median_seconds = statistics.median(samples)
        results.append(
            {
                "batch_size": batch_size,
                "compile_seconds": compile_seconds,
                "sample_seconds": samples,
                "median_seconds": median_seconds,
                "evaluations_per_second": batch_size / median_seconds,
                "us_per_evaluation": median_seconds / batch_size * 1e6,
            }
        )

    device = jax.devices()[0]
    print(
        json.dumps(
            {
                "platform": platform.platform(),
                "machine": platform.machine(),
                "python": platform.python_version(),
                "jax": jax.__version__,
                "device": str(device),
                "backend": device.platform,
                "dtype": args.dtype,
                "x64_enabled": bool(jax.config.x64_enabled),
                "repeats": args.repeats,
                "results": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
