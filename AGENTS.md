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
# S^2 event store regressions at N = 1e7 (task 13 bit-for-bit rebuild, task 14 class symmetry transport; ~3 min,
# need the task 13/14 scratchpad artifacts, skipped otherwise)
uv run pytest -m slow tests/test_s2_store.py tests/test_s2_store_symmetry.py
# writing series ch8 / ch9 signature tables recomputed with lumice_integral.symmetry, byte-compared with the
# published CSVs (reads the writing repo's data read-only, LUMICE_INTEGRAL_WRITING_ROOT; skipped if absent; ~14 min on an M2 Max)
uv run pytest -m slow tests/test_symmetry_signature_table_regression.py
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
# S^2 band-sum renderer (roadmap 4.2): the same strip layout in minutes on a Mac. N = 1e8 full image:
# 2.8 min on 4 workers with the store cached under artifacts/s2-store (first build ~80 s); N = 1e7: 35 s
JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 \
  uv run python scripts/render_band_sum.py --store-n 100000000 --workers 4 --output-dir artifacts/band-sum-full
# ... a path class with a narrow family and its own camera (one store per class, D6h transports incl. mirrors)
uv run python scripts/render_band_sum.py --store-n 10000000 --path 3 5 --path-class --workers 3 \
  --pose-density-family plate --pose-density-zenith-std-deg 1 \
  --width 321 --height 161 --fov-deg 32 --view-elevation 15 --output-dir /tmp/band-sum-plate
# ... and its regressions (against strip-full; against task 14's profiles) and log-scale figures
uv run python scripts/regress_band_sum.py --stage full --band-dir artifacts/band-sum-full --output /tmp/regression_full.json
# ... the K_eff ruler: two i.i.d. stores of class [3,5] on task 14's profiles (z of their difference, ~1 min)
uv run python scripts/regress_band_sum.py --stage k-eff --random-n 10000000 --output /tmp/regression_k_eff.json
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
│   ├── geometry/          # Finite-crystal geometry: polyhedra, unfolding, corridor
│   │                      # intersection, path enumeration, entry_measure (pure numpy)
│   └── symmetry/          # D6h / G tables, signature and Phi classes, ch3 ground truth,
│                          # attitude construction (pure numpy, depends on geometry only)
├── tests/                 # Analytic and optical regression fixtures
├── benchmarks/            # Reproducible CPU/GPU probes
├── docs/
│   ├── roadmap.md         # Mathematical model, phased scope, validation
│   ├── conventions.md     # Every coordinate / sign / symbol convention and its check
│   ├── s2-precomputation.md # Design note: the S^2 event store, its consumers and costs
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
  physical and coordinate conventions. Conventions follow fixed authorities
  (owner ruling 2026-09-23): coordinates, face numbering, pose chain,
  azimuth, light source, camera and pixels follow the Lumice documentation
  (`doc/coordinate-convention.md` first); mathematical notation Lumice does
  not cover (fiber coordinates, `Phi`, `D`, `M`, `W`, signatures) follows the
  writing series' `docs/framework.md`; the rest is decided in
  `docs/conventions.md`, the single table of every convention and its check.
  Settle any disagreement by these rules instead of case by case. The public
  sun direction is `s_hat`, toward the sun; the solver's `incident_direction`
  is the propagation `-s_hat`, converted only by
  `camera.incident_direction_from_sun`.

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
- `docs/conventions.md`: convention authority (Lumice / writing series / this project, with checks).
- `docs/s2-precomputation.md`: design note for the $S^2$ event store (source independence, its consumers, cost structure, band sum organised by deviation, divergent light).
- `scratchpad/tasks.md`: local task index.
- `scratchpad/backlog.md`: unstructured ideas awaiting task selection.

## Code Style

Use Python 3.12 type syntax, four-space indentation, and small pure functions
that remain compatible with JAX transformations where differentiation or
batching is required. Use English for code identifiers and comments.
Documentation may be written in English or Chinese; durable mathematical
notation and terminology should stay consistent with `docs/roadmap.md`.
