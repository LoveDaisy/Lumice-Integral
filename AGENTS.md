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
