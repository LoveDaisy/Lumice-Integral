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

Status: Phase I (SO(3) fibre continuation) is closed and agrees with Lumice
in shape and absolute scale; Phase II's band-sum renderer over precomputed
$S^2$ events is in production; the Phase II contour method (milestone M2) is
next.

Start with [docs/overview.md](docs/overview.md) ([中文](docs/overview_zh.md)):
how direct integration differs from Monte Carlo rendering, what Phases I and
II are, and the plan. Design details: [docs/phase1.md](docs/phase1.md),
[docs/phase2.md](docs/phase2.md). Status, queue and decisions:
[docs/roadmap.md](docs/roadmap.md). The accepted Phase I stack is
[ADR 0001](docs/decisions/0001-phase-i-python-jax.md).
