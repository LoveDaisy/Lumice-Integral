"""Measure host-controlled continuation against a compiled scan."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time

import jax
import jax.numpy as jnp

from lumice_integral.analytic import BODY_AXIS, tangent_basis
from lumice_integral.continuation import constraint_jacobian, null_tangent
from lumice_integral.so3 import exp


def continuation_step(state, inputs):
    rotation, tangent = state
    target, basis, step_size = inputs
    predicted = rotation @ exp(step_size * tangent)
    jacobian = constraint_jacobian(predicted, target, basis)
    next_tangent = null_tangent(jacobian, previous=tangent)
    return predicted, next_tangent


def elapsed_seconds(function):
    start = time.perf_counter()
    result = function()
    jax.block_until_ready(result)
    return time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    rotation = jnp.eye(3, dtype=jnp.float64)
    target = BODY_AXIS
    basis = tangent_basis(target)
    tangent = null_tangent(constraint_jacobian(rotation, target, basis))
    step_size = jnp.asarray(2.0 * jnp.pi / args.steps, dtype=jnp.float64)
    inputs = (target, basis, step_size)
    compiled_step = jax.jit(continuation_step)

    compile_start = time.perf_counter()
    compiled_step((rotation, tangent), inputs)[0].block_until_ready()
    step_compile_seconds = time.perf_counter() - compile_start

    def host_loop(*, synchronize_each_step: bool):
        state = (rotation, tangent)
        for _ in range(args.steps):
            state = compiled_step(state, inputs)
            if synchronize_each_step:
                state[0].block_until_ready()
        return state[0]

    def scan_loop(state):
        def body(current, _):
            return continuation_step(current, inputs), None

        return jax.lax.scan(body, state, xs=None, length=args.steps)[0][0]

    compiled_scan = jax.jit(scan_loop)
    compile_start = time.perf_counter()
    compiled_scan((rotation, tangent)).block_until_ready()
    scan_compile_seconds = time.perf_counter() - compile_start

    host_async_times = [
        elapsed_seconds(lambda: host_loop(synchronize_each_step=False))
        for _ in range(args.repeats)
    ]
    host_sync_times = [
        elapsed_seconds(lambda: host_loop(synchronize_each_step=True))
        for _ in range(args.repeats)
    ]
    scan_times = [
        elapsed_seconds(lambda: compiled_scan((rotation, tangent)))
        for _ in range(args.repeats)
    ]
    host_async_median = statistics.median(host_async_times)
    host_sync_median = statistics.median(host_sync_times)
    scan_median = statistics.median(scan_times)

    device = jax.devices()[0]
    report = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "jax": jax.__version__,
        "device": str(device),
        "backend": device.platform,
        "x64_enabled": bool(jax.config.x64_enabled),
        "steps": args.steps,
        "repeats": args.repeats,
        "step_compile_seconds": step_compile_seconds,
        "scan_compile_seconds": scan_compile_seconds,
        "host_async_seconds": host_async_times,
        "host_sync_seconds": host_sync_times,
        "scan_seconds": scan_times,
        "host_async_median_us_per_step": host_async_median / args.steps * 1e6,
        "host_sync_median_us_per_step": host_sync_median / args.steps * 1e6,
        "scan_median_us_per_step": scan_median / args.steps * 1e6,
        "host_async_to_scan_ratio": host_async_median / scan_median,
        "host_sync_to_scan_ratio": host_sync_median / scan_median,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
