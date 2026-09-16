# ADR 0001: Python and JAX for the Phase I Reference Solver

- Status: accepted
- Date: 2026-09-16

## Context

Phase I needs automatic derivatives of fixed-path maps from SO(3) to S2,
predictor-corrector continuation of one-dimensional inverse-image fibers, and
eventually one weighted integral per image direction. The original prototype
showed that a CPU-only image can be expensive, but adaptive continuation also
contains irregular control flow that is a poor scalar GPU workload.

The first probe implemented two maps:

- the analytic fiber `F(R) = R e3`, whose inverse image is a known circle;
- independent Snell refraction through hexagonal-prism faces 3 and 5, without
  obstruction or Fresnel weights.

Both use right-trivialized three-dimensional tangent updates and forward-mode
JAX Jacobians. Lumice code is not linked, imported, copied, or invoked.

## Decision

Use Python 3.12 and JAX for the Phase I reference solver. Manage the environment
with `uv` and commit `uv.lock`. Use NumPy and SciPy for reference calculations
and supporting algorithms, and pytest for regression tests.

The reference continuation and final integral use float64. Float32 may be used
only by explicitly approximate, separately validated candidate or prefilter
stages. It must not silently enter the reference path.

Keep adaptive orchestration in Python initially, but make expensive value,
Jacobian, and weight evaluators pure JAX functions. GPU calls must operate on
large batches of roots, active fibers, or pixels, or on compiled multi-step
blocks. A single fiber step is not an acceptable accelerator boundary.

Do not build a C++ backend in Phase I until a representative implementation has
been profiled. If one becomes necessary, its Python boundary must be a coarse
batch rather than a continuation step.

## Evidence

All steady-state timings exclude JIT compilation and synchronize the measured
result. Tests used Python `3.12.11` and JAX `0.11.1`.

| Workload | M2 Max CPU | Ryzen 9 9950X / WSL CPU | RTX 5090 D / WSL GPU |
|---|---:|---:|---:|
| float64, 1,048,576 batched 3-5 direction/Jacobian evaluations | 33.16 M/s | 9.33 M/s | 298.47 M/s |
| float32, 1,048,576 batched evaluations | 33.81 M/s | 17.03 M/s | 803.81 M/s |
| float64, one 145-step 3-5 fiber | 2.82 s | not measured | 11.27 s |

At batch 32,768 the float64 GPU was still slightly slower than the M2 Max,
showing that the GPU crossover is workload- and batch-dependent. At batch
1,048,576 the GPU was about `9x` faster for float64 and `24x` faster for
float32. Approximate measured memory at that batch was 1.59 GiB Mac process RSS,
1.37 GiB WSL CPU process RSS, and 856/656 MiB peak GPU allocator use for
float64/float32.

The 3-5 float64 fiber closed after 145 fixed steps with maximum residual
`9.99e-12` and rotation error `1.06e-10 rad`. Float32 preserved the step count
but produced maximum residual `9.98e-7` and closure error `1.86e-5 rad`.

### Compiled step kernels inside the Python control loop (2026-09-16)

The adaptive reference solver keeps its control flow, event gates and failure
classification on the host, but every per-iteration numerical unit is a
module-level `jax.jit` kernel keyed on the identity of
`problem.direction_evaluator`: direction plus target-chart residual, the
(2, 3) local Jacobian, one bordered Newton iterate (primal value, Jacobian,
condition number and update from a single `jacfwd(has_aux=True)` pass), the
predictor, the geodesic distance and the section coordinate. Measured with
`benchmarks/benchmark_fiber_trace.py` on the M2 Max CPU (Python `3.12.11`,
JAX `0.11.1`, float64), same `FiberProblem` instance traced twice so the
second call hits the compile cache:

| Workload (adaptive 3-5 fiber) | Before | After | Steps / evaluations |
|---|---:|---:|---|
| canonical seed `[0.15, 0.08, -0.05]`, warm | 7.86 s | 0.141 s | 192 / 1161 |
| seed `[0.16, 0.07, -0.04]`, warm | 14.30 s | 0.243 s | 364 / 1975 |
| per evaluation unit, warm | ~7.5 ms | ~122 µs | — |
| first call including kernel compilation | — | 0.6–0.7 s | — |

Steps, evaluation counts and reason classification are unchanged; residual,
arclength and closure gap moved at the `1e-14`–`1e-16` level from XLA
reordering. This is a different fiber from the 145-step fixed-step row above
and is not comparable to it step for step.

The remaining ~120 µs per evaluation is dispatch granularity, not kernel
arithmetic: about 25 µs kernel dispatch, 30 µs for the device-to-host copies
the gates read, 12 µs host-side domain gate and 20 µs host gate and
classification bookkeeping. Closing the gap to the compiled `lax.scan` figure
would require moving the corrector loop and its gates into the graph, which is
a separate decision. Two boundaries follow from the cache design: kernels are
cached per `direction_evaluator` object, so a workflow that constructs one
`FiberProblem` per pixel recompiles the four evaluator-bound kernels
(~0.4 s) for every pixel unless it shares or hashes the evaluator; and the
per-step host synchronization remains the reason a single fiber step is not
an accelerator boundary.

## Operational Constraints

On `home-wsl`, set:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false
```

JAX otherwise attempted a roughly 24 GiB initial allocation and emitted a
series of OOM fallbacks despite more than 31 GiB reported free. Benchmarks must
record the backend, device, dtype, batch size, compile time, synchronized
steady-state time, and memory independently.

The optional CUDA environment is installed with:

```bash
uv sync --extra cuda13 --dev
```

## Consequences

- The Python implementation is a durable numerical reference, not disposable
  prototype code.
- SO(3) primitives and branch-local optics must be written for AD at singular
  coordinate limits; value stability alone is insufficient.
- The next performance experiment concerns batched adaptive continuation, not
  a language rewrite.
- C++20/Eigen and custom CUDA remain contingency options triggered by profiling,
  not committed dependencies.
