# Lumice Integral

Lumice Integral is a deterministic numerical renderer for ice halos. Given a
light source, an outgoing sky direction, and a ray path, it traces the
corresponding one-dimensional pose fibers in SO(3) and evaluates their weighted
integrals without Monte Carlo sampling.

The project is a sibling of [Lumice](https://github.com/LoveDaisy/ice_halo_sim),
not a replacement for it. Lumice performs forward Monte Carlo ray tracing;
Lumice Integral studies and evaluates the inverse-image integral behind a
specified image direction. Lumice Integral owns an independent differentiable
optics implementation and does not use Lumice as a library or runtime engine;
Lumice serves only as an external validation oracle and analysis-data source.

The project is currently in the design and reconstruction stage. The first
implementation will reproduce the original SO(3) continuation prototype before
considering the newer fiber-reduction formulation.

The Phase I reference API is `trace_fiber(FiberProblem, ContinuationOptions)`.
It traces only the connected component reachable from the supplied regular
seed and returns structured closure, event, numerical-failure, or budget
diagnostics. Its geometry-only result deliberately leaves physical weight
factors unavailable; component discovery and coarea integration remain
separate stages.

See [docs/roadmap.md](docs/roadmap.md) for the mathematical model, scope, phased
plan, and validation strategy. The accepted Phase I stack and its measured
CPU/GPU boundaries are recorded in
[ADR 0001](docs/decisions/0001-phase-i-python-jax.md).
