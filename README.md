# Lumice Integral

Lumice Integral is a deterministic numerical renderer for ice halos. Given a
light source, an outgoing sky direction, and a ray path, it traces the
corresponding one-dimensional pose fibers in SO(3) and evaluates their weighted
integrals without Monte Carlo sampling.

The project is a sibling of [Lumice](https://github.com/LoveDaisy/ice_halo_sim),
not a replacement for it. Lumice performs forward Monte Carlo ray tracing;
Lumice Integral studies and evaluates the inverse-image integral behind a
specified image direction.

The project is currently in the design and reconstruction stage. The first
implementation will reproduce the original SO(3) continuation prototype before
considering the newer fiber-reduction formulation.

See [docs/roadmap.md](docs/roadmap.md) for the mathematical model, scope, phased
plan, and validation strategy.
