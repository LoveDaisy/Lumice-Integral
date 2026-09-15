# AGENTS.md

## Project Overview

Lumice Integral is a deterministic numerical renderer for ice halos. For a
fixed light source, outgoing direction, and ray path, it traces the inverse-image
pose fibers in SO(3) and evaluates their weighted coarea integrals. It is a
sibling of Lumice's forward Monte Carlo simulator and is currently in the design
and prototype-reconstruction stage.

## Common Commands

No implementation language or build system has been selected yet.

```bash
git status --short
git diff
```

Add build, test, and formatting commands here when the first implementation
task selects the technical stack.

## Architecture and Design

The authoritative staged design is `docs/roadmap.md`.

```text
.
├── README.md              # Concise project identity and navigation
├── docs/
│   └── roadmap.md         # Mathematical model, phased scope, validation
└── scratchpad/            # Local task management, ignored by git
```

Phase I directly traces implicit one-dimensional fibers in SO(3) using root
finding, automatic differentiation, predictor-corrector continuation, and line
quadrature. Phase II independently implements the newer reduction to level-set
contours on S2 for cross-validation and possible acceleration.

## Important Constraints

### DO NOT

- Do not replace the Phase I SO(3) continuation path with the Phase II fiber
  reduction before the original formulation has been reconstructed and
  validated.
- Do not call numerical results exact; report convergence and residuals.
- Do not treat one successfully closed loop as proof that every connected
  component of a fiber was found.
- Do not prematurely extract or freeze a public Lumice engine API before a
  working prototype identifies the real consumer boundary.
- Do not let duplicated optical equations silently diverge from Lumice's
  conventions.

### DO

- Keep physical weight factors independently observable before multiplying
  them into the final integrand.
- Validate with analytic fixtures, historical direct-integration data, and
  independently converged Lumice Monte Carlo results.
- Use local three-dimensional Lie algebra coordinates for SO(3) derivatives and
  updates, even if rotations are stored as unit quaternions.
- Use `scratchpad/` to manage development tasks; see `scratchpad/common.md`.

## Key Files

- `README.md`: project entry point.
- `docs/roadmap.md`: current design authority and phase boundary.
- `scratchpad/tasks.md`: local task index.
- `scratchpad/backlog.md`: unstructured ideas awaiting task selection.

## Code Style

The language-specific style is pending selection of the implementation stack.
Use English for code identifiers and comments. Documentation may be written in
English or Chinese; durable mathematical notation and terminology should stay
consistent with `docs/roadmap.md`.
