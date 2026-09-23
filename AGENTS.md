# AGENTS.md

## Project Overview

Lumice Integral is a deterministic numerical renderer for ice halos. For a
fixed light source, outgoing direction, and ray path, it traces the inverse-image
pose fibers in SO(3) and evaluates their weighted coarea integrals. It is a
sibling of Lumice's forward Monte Carlo simulator and is currently in the design
and prototype-reconstruction stage. Its differentiable optics implementation is
independent of Lumice by design.

## Common Commands

Phase I uses Python 3.12, JAX, uv, and pytest. The CUDA extra is intended for
the Linux/NVIDIA reference machine.

```bash
uv sync --dev
uv run pytest -q
uv run python scripts/inspect_path_3_5.py
uv run python benchmarks/benchmark_batch.py --dtype float64
uv run python benchmarks/benchmark_fiber_trace.py
# the strip's per-pixel ruler: one column single-process, steady-state s/px and XLA compile count
uv run python benchmarks/benchmark_column_steady_state.py --column 126 --rows 100:160
# ch06 251 x 801 direct strip: a sub-window smoke on a laptop ...
uv run python scripts/render_ch06_strip.py --rows 140:160 --columns 145:155 --workers 4 --output-dir /tmp/strip-smoke
# ... and the full image on the many-core reference machine (home-wsl, <= 30 workers,
# CPU JAX: one small kernel per pixel does not pay for a shared GPU; resumable per column;
# spawned workers get glibc malloc trimming by default, see strip_driver.WORKER_MALLOC_ENV).
# 251 x 801 takes 35 min wall clock with 30 workers (2026-09-20, after task pixel-cost-shape-stable-kernels;
# 0.29 s per pixel per worker on the 16C/32T box, 0.17 s single-process there, 0.11 s single-process on the M2 Max;
# throughput is flat from 30 to 32 workers, the machine's logical CPU count, so 30 is the recommended count).
# It took 1.86 h on 2026-09-17 before the per-pixel XLA recompilations were removed.
JAX_PLATFORMS=cpu XLA_FLAGS="--xla_cpu_multi_thread_eigen=false" OMP_NUM_THREADS=1 \
  uv run python scripts/render_ch06_strip.py --workers 30 --output-dir artifacts/strip-full --resume
# log-domain comparison with the historical raw and the Lumice remake (matplotlib is not a dependency)
uv run --with matplotlib python scripts/compare_strip_v2.py --strip-dir artifacts/strip-full --output-dir /tmp/strip-compare
# ... plus the radiometric three-way check against a Lumice float export (`Lumice render --format npy`, run
# externally; img_01.npy with its img_01.json sidecar; a second seed is summed in and gives the noise floor)
uv run --with matplotlib python scripts/compare_strip_v2.py --strip-dir artifacts/strip-full --output-dir /tmp/strip-compare \
  --lumice-float <run1>/img_01.npy --lumice-float-run2 <run2>/img_01.npy
# ... and the absolute scale raw/E = K_p V: the probe predicts Lumice per pixel with nothing fitted (~30 s; add
# --refractive-index 1.3110129 to match Lumice's own n(550)), its output dir feeds compare_strip_v2's absolute_scale block
uv run python scripts/probe_absolute_scale.py --lumice-run <run1> --lumice-run <run2> --output-dir /tmp/abs-scale
uv run --with matplotlib python scripts/compare_strip_v2.py --strip-dir artifacts/strip-full --output-dir /tmp/strip-compare \
  --lumice-float <run1>/img_01.npy --lumice-float-run2 <run2>/img_01.npy --absolute-scale-probe /tmp/abs-scale
# a non-canonical pose-density family (recorded in provenance.json and the resume fingerprint)
uv run python scripts/render_ch06_strip.py --rows 300:302 --columns 126:127 --output-dir /tmp/strip-parry \
  --pose-density-family parry --pose-density-zenith-std-deg 1 --pose-density-roll-std-deg 1
# Linux/NVIDIA environment
uv sync --extra cuda13 --dev
XLA_PYTHON_CLIENT_PREALLOCATE=false uv run python benchmarks/benchmark_batch.py
```

## Architecture and Design

The authoritative staged design is `docs/roadmap.md`.

```text
.
├── README.md              # Concise project identity and navigation
├── pyproject.toml         # Python package, dependencies, and test config
├── src/lumice_integral/   # Differentiable numerical building blocks
│   └── geometry/          # Finite-crystal geometry: polyhedra, unfolding, corridor
│                          # intersection, path enumeration, entry_measure (pure numpy)
├── tests/                 # Analytic and optical regression fixtures
├── benchmarks/            # Reproducible CPU/GPU probes
├── docs/
│   ├── roadmap.md         # Mathematical model, phased scope, validation
│   └── decisions/         # Accepted architecture decisions
└── scratchpad/            # Local task management, ignored by git
```

Phase I directly traces implicit one-dimensional fibers in SO(3) using root
finding, automatic differentiation, predictor-corrector continuation, and line
quadrature. Phase II independently implements the newer reduction to level-set
contours on S2 for cross-validation and possible acceleration.

The production implementation owns its complete differentiable computation
graph, from pose and fixed ray path through direction and named physical
weights. Lumice is outside that graph and is used only by explicit validation
workflows as a black-box Monte Carlo oracle or source of analysis artifacts.

## Important Constraints

### DO NOT

- Do not replace the Phase I SO(3) continuation path with the Phase II fiber
  reduction before the original formulation has been reconstructed and
  validated.
- Do not call numerical results exact; report convergence and residuals.
- Do not treat one successfully closed loop as proof that every connected
  component of a fiber was found.
- Do not link, import, embed, or invoke Lumice from the production solver. A
  user must be able to build and run Lumice Integral without Lumice source,
  libraries, or binaries.
- Do not design or extract a Lumice API for Lumice Integral. The projects have
  independent implementations and meet only at validation boundaries.
- Do not let the independent optical equations silently drift from shared
  physical and coordinate conventions.

### DO

- Keep physical weight factors independently observable before multiplying
  them into the final integrand.
- Own the differentiable geometry, optics, and event-boundary handling required
  by continuation; fixed-path smooth branches may use AD, while TIR, face
  changes, and obstruction boundaries must remain explicit events.
- Validate with analytic fixtures, historical direct-integration data, and
  independently converged Lumice Monte Carlo results. Validation tooling may
  run Lumice explicitly and ingest its files, but this must not become a product
  dependency or a shared computational kernel.
- Use local three-dimensional Lie algebra coordinates for SO(3) derivatives and
  updates, even if rotations are stored as unit quaternions.
- Use `scratchpad/` to manage development tasks; see `scratchpad/common.md`.

## Key Files

- `README.md`: project entry point.
- `docs/roadmap.md`: current design authority and phase boundary.
- `scratchpad/tasks.md`: local task index.
- `scratchpad/backlog.md`: unstructured ideas awaiting task selection.

## Code Style

Use Python 3.12 type syntax, four-space indentation, and small pure functions
that remain compatible with JAX transformations where differentiation or
batching is required. Use English for code identifiers and comments.
Documentation may be written in English or Chinese; durable mathematical
notation and terminology should stay consistent with `docs/roadmap.md`.
