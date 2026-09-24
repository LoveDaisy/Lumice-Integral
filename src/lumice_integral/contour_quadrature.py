"""Line quadrature on the level sets of ``D_P``: deterministic Phase II pixel values (``docs/phase2.md`` section 4).

For a pixel at deviation ``delta`` and azimuth ``alpha`` (its centre direction)

    I(delta, alpha) sin(delta) = (1 / 8 pi^2) sum_components int_{D_P = delta} rho(R(u, alpha)) w(u) / |grad_{S^2} D_P(u)| dl

with ``w = A_P T_P`` (:func:`.s2_store.evaluate_fields`, the store's weight),
``R(u, alpha)`` the pose with ``R u = s_hat`` whose outgoing direction lies at
the pixel's azimuth (:func:`.s2_store.event_rotations` at the point's *own*
deviation, the band-sum construction) and ``1 / 8 pi^2`` the Haar conversion
of Phase I (:data:`.quadrature.HAAR_TO_DVOL_G_FACTOR`).  The components are
those of :func:`.contour.extract_level_sets` (certified complete against the
interval partition); every node of theirs lies on the level set.

Pointwise identity with Phase I.  ``SO(3)`` with the rotation-angle metric is
``dA(u) dpsi`` (``psi`` the twist about ``s_hat``), the sky map is
``(delta, alpha) = (D_P(u), psi + beta(u))`` and the sky area ``sin(delta)
d delta d alpha``, so the fiber measures agree, ``ds / J_perp = dl_u /
(|grad D_P| sin delta)``; along a Phase I fiber ``R' = R hat(xi)`` with
``|xi| = 1`` moves ``u = R^T s_hat`` at ``|xi x u|``, hence

    J_perp F_P(R) = |grad_{S^2} D_P(u)| sin(delta) / |xi x u|

(checked on the canonical fiber to ``3e-15``, ``tests/test_contour_quadrature.py``).
``J_perp`` vanishes only where ``grad D_P`` does, i.e. at a critical value of
``delta``; there the level set changes topology and
:func:`.contour.extract_level_sets` refuses the ``delta``.  The value here is
the point value in the ``epsilon -> 0`` limit of Phase I's ``J_perp + eps``:
no regularisation, and a pixel whose ``delta`` is within
``dp_field.boundary.EXTREMUM_ATOL`` of a critical value is not integrated
(reported, value ``0``).

Parametrisation (exact, not interpolated).  A panel is the stretch of a
component between two consecutive nodes ``a``, ``b`` (a loop's last node is
joined to its first).  Its points are ``q(t) = normalize(cos s c(t) + sin s n)``
with ``c(t)`` the great-circle arc from ``a`` to ``b``, ``n`` the arc's pole and
``s(t)`` solved by Newton so that ``D_P(q) = delta``; the arclength speed
``|dq/dt|`` follows from the implicit function theorem (``ds/dt = -D_t /
D_s``, both by ``jax.jvp``).  So every quadrature point is on the level set to
round-off and the chord never stands in for the arc
(``scratchpad/learnings``: a chord parametrisation lowers Simpson's order).

Quadrature.  Simpson on five points of ``t`` per panel (``N``) against its
three-point subset (``N / 2``): ``|S_N - S_{N/2}| / 15`` is the panel's error
estimate.  A panel above its share of the tolerance (``max(rtol |I|, atol)``
by chord length) is halved, its children reusing the parent's points (four
new points per split), until every panel passes or reaches ``max_depth``
(reported as exhausted, never hidden).  A smooth split cuts the estimate by
``~16``; ``low_order_splits`` counts splits that cut it by less than ``8``,
the local order loss of a kink of ``A_P`` (a vertex of the entry polygon
crossing an edge) or of the ``sqrt`` end of an arc on an exit-TIR piece.

Two stages.  The geometry of a level set (points, ``w``, ``|grad D|``, speed)
does not depend on the pixel's azimuth or on the density:
:meth:`LevelSetGeometry.build` refines on ``w / |grad D|`` alone, once per
``delta``.  :meth:`LevelSetGeometry.integrate` then takes one pixel (``s_hat``,
centre direction, density): ``rho`` at the existing points, and new points
only where ``rho`` needs them (a narrow density's peak).  For the random
family ``rho = 1`` and the second stage adds nothing: the value depends on
``delta`` alone.

Batches: every point evaluation is one ``jax.vmap`` over all points of a
round (rows padded to a power of two, one compilation per size) and one
batch of the production field evaluators; the Python loops run over
refinement rounds and pixels, never over points.

Nothing here imports or calls Lumice.
"""

from __future__ import annotations

import dataclasses
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .contour import LevelSet
from .dp_field import DPField
from .dp_field.boundary import EXTREMUM_ATOL
from .dp_field.field import d_value
from .quadrature import HAAR_TO_DVOL_G_FACTOR
from .s2_store import align_rotations, evaluate_fields, event_rotations

# The body-frame alignment of the field evaluators: R u = this, any twist (the fields depend on u only).
# Same value as dp_field.field._PROBE_SUN, declared here rather than imported across the package's
# private boundary (pinned equal by tests/test_contour_quadrature.py).
_PROBE_SUN = np.array([0.0, 0.0, 1.0])
# Simpson points of a panel, as fractions of its parameter interval; a split adds the odd eighths.
PANEL_FRACTIONS = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
SMOOTH_SPLIT_RATIO = 16.0
LOW_ORDER_SPLIT_RATIO = 8.0
# A split's error ratio is only meaningful above round-off of the panel value.
ROUNDOFF_ULPS = 1e3
METHOD = (
    "contour quadrature (docs/phase2.md section 4): I sin(delta) = (1/8 pi^2) sum_components "
    "int_{D_P=delta} rho(R) w / |grad D_P| dl; points on the level set by Newton across the chord, "
    "arclength speed by the implicit function theorem, adaptive Simpson (5 vs 3 points per panel)"
)


@dataclasses.dataclass(frozen=True)
class QuadratureOptions:
    """Numerical policy of the level-set quadrature.

    ``relative_tolerance`` is on the pixel value, ``absolute_tolerance`` on
    the raw integral ``sum int rho w / |grad D| dl`` (before ``1 / 8 pi^2 sin
    delta``); ``max_depth`` halvings of a node-to-node panel at most;
    ``newton_iterations`` of the cross-chord Newton (quadratic, from ``s = 0``).
    """

    relative_tolerance: float = 1e-10
    absolute_tolerance: float = 1e-16
    max_depth: int = 24
    newton_iterations: int = 8

    def __post_init__(self) -> None:
        if not (self.relative_tolerance > 0.0 and self.absolute_tolerance > 0.0):
            raise ValueError("tolerances must be positive")
        if self.max_depth < 0 or self.newton_iterations < 1:
            raise ValueError("max_depth must be >= 0 and newton_iterations >= 1")

    def as_json(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


# ---- JAX kernel ---------------------------------------------------------------------------------------


def _panel_point(a: jax.Array, b: jax.Array, t: jax.Array, delta: jax.Array, faces, index, slab, iterations: int):
    """``(q, |dq/dt|, |grad D(q)|, D(q) - delta)`` of the level-set point at chord parameter ``t`` of the panel ``a -> b``."""
    axis = jnp.cross(a, b)
    sine = jnp.linalg.norm(axis)
    theta = jnp.arctan2(sine, jnp.dot(a, b))
    n = axis / jnp.maximum(sine, 1e-300)
    e = jnp.cross(n, a)

    def point(tt, ss):
        c = jnp.cos(theta * tt) * a + jnp.sin(theta * tt) * e
        v = jnp.cos(ss) * c + jnp.sin(ss) * n
        return v / jnp.linalg.norm(v)

    def residual(tt, ss):
        return d_value(point(tt, ss), faces, index, slab) - delta

    reach = jnp.maximum(theta, 1e-12)

    def newton(_, s):
        r, r_s = jax.jvp(lambda ss: residual(t, ss), (s,), (jnp.ones_like(s),))
        step = -r / jnp.where(r_s == 0.0, 1e-300, r_s)
        return s + jnp.clip(step, -reach, reach)

    s = jax.lax.fori_loop(0, iterations, newton, jnp.zeros_like(t))
    r, r_t = jax.jvp(lambda tt: residual(tt, s), (t,), (jnp.ones_like(t),))
    _, r_s = jax.jvp(lambda ss: residual(t, ss), (s,), (jnp.ones_like(s),))
    s_t = -r_t / r_s
    q, q_t = jax.jvp(point, (t, s), (jnp.ones_like(t), s_t))
    g = jax.grad(d_value)(q, faces, index, slab)
    g = g - jnp.dot(g, q) * q
    return q, jnp.linalg.norm(q_t), jnp.linalg.norm(g), r


@partial(jax.jit, static_argnums=(4, 7))
def _panel_points(a, b, t, delta, faces, index, slab, iterations: int):
    return jax.vmap(_panel_point, in_axes=(0, 0, 0, None, None, None, None, None))(a, b, t, delta, faces, index, slab, iterations)


def _bucket(n: int) -> int:
    """Batch rows padded to a power of two (at least 8): one XLA compilation per size, not per count."""
    return max(8, 1 << (n - 1).bit_length())


# ---- panels -------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, eq=False)
class _Panels:
    """Panels of one level set and the pixel-independent data at their five points (arrays aligned on axis 0).

    ``t`` ``(P, 5)`` chord parameters of the five Simpson points in the node
    panel ``a -> b`` (``theta`` its angle); ``q``/``phi`` ``(P, 5, 3)`` the
    level-set points and ``Phi_P(-q)``; ``d`` ``(P, 5)`` their own deviation;
    ``g`` ``(P, 5)`` the geometric integrand ``w / |grad D| |dq/dt|`` (``0``
    outside the weight's support); ``bad`` non-finite points with ``w > 0``
    (integrand taken as ``0`` and counted); ``depth`` halvings so far,
    ``parent_error`` the parent's estimate (``inf`` for a node panel).
    """

    component: np.ndarray
    a: np.ndarray
    b: np.ndarray
    theta: np.ndarray
    t: np.ndarray
    q: np.ndarray
    phi: np.ndarray
    d: np.ndarray
    g: np.ndarray
    bad: np.ndarray
    depth: np.ndarray
    parent_error: np.ndarray

    def __len__(self) -> int:
        return len(self.component)

    @property
    def length(self) -> np.ndarray:
        """Chord angle of each panel (its share of the tolerance)."""
        return self.theta * (self.t[:, -1] - self.t[:, 0])

    def take(self, index: np.ndarray) -> "_Panels":
        return _Panels(*(getattr(self, f.name)[index] for f in dataclasses.fields(self)))

    @staticmethod
    def concatenate(parts: list["_Panels"]) -> "_Panels":
        return _Panels(*(np.concatenate([getattr(p, f.name) for p in parts]) for f in dataclasses.fields(_Panels)))


def _evaluate_points(field: DPField, delta: float, a: np.ndarray, b: np.ndarray, t: np.ndarray, iterations: int) -> dict[str, np.ndarray]:
    """Level-set points at ``(a, b, t)`` rows and their pixel-independent data (one JAX batch, one field batch)."""
    n = len(t)
    extra = _bucket(n) - n
    pad = lambda x: np.concatenate([x, np.repeat(x[:1], extra, axis=0)])  # noqa: E731
    slab = None if field.slab is None else jnp.asarray(field.slab)
    q, speed, gradient, residual = (
        np.asarray(x)[:n]
        for x in _panel_points(pad(a), pad(b), pad(t), jnp.float64(delta), field.faces, jnp.float64(field.index), slab, iterations)
    )
    finite = np.all(np.isfinite(q), axis=1) & np.isfinite(speed) & np.isfinite(gradient)
    q_safe = np.where(finite[:, None], q, a)
    fields = evaluate_fields(align_rotations(q_safe, _PROBE_SUN), _PROBE_SUN, field.crystal, field.index, (field.faces,))
    w = fields["w"]
    good = finite & (w > 0.0) & (gradient > 0.0)
    g = np.where(good, w / np.where(good, gradient, 1.0) * speed, 0.0)
    return {
        "q": q_safe,
        "phi": fields["phi"],
        "d": fields["D"],
        "g": g,
        "bad": ~finite & (w > 0.0),
        "residual": np.where(finite, residual, np.nan),
    }


def _new_panels(field, delta, component, a, b, theta, t, depth, parent_error, iterations) -> _Panels:
    p = len(component)
    rows = _evaluate_points(field, delta, np.repeat(a, 5, axis=0), np.repeat(b, 5, axis=0), t.reshape(-1), iterations)
    shape = lambda x: x.reshape(p, 5, *x.shape[1:])  # noqa: E731
    return _Panels(component, a, b, theta, t, shape(rows["q"]), shape(rows["phi"]), shape(rows["d"]), shape(rows["g"]),
                   shape(rows["bad"]), depth, parent_error)


def _node_panels(field: DPField, level_set: LevelSet, iterations: int) -> _Panels:
    """One panel per pair of consecutive nodes of every component (a loop closes back on its first node)."""
    a, b, component = [], [], []
    for k, c in enumerate(level_set.components):
        points = np.asarray(c.points, dtype=np.float64)
        ends = np.roll(points, -1, axis=0) if c.closed else points[1:]
        starts = points if c.closed else points[:-1]
        keep = np.any(starts != ends, axis=1)
        a.append(starts[keep])
        b.append(ends[keep])
        component.append(np.full(int(keep.sum()), k))
    if not a or sum(len(x) for x in a) == 0:
        return _empty_panels()
    a, b, component = np.vstack(a), np.vstack(b), np.concatenate(component)
    theta = np.arctan2(np.linalg.norm(np.cross(a, b), axis=1), np.sum(a * b, axis=1))
    t = np.broadcast_to(PANEL_FRACTIONS, (len(a), 5)).copy()
    return _new_panels(field, level_set.delta, component, a, b, theta, t, np.zeros(len(a), dtype=int),
                       np.full(len(a), np.inf), iterations)


def _empty_panels() -> _Panels:
    z = np.zeros
    return _Panels(z(0, dtype=int), z((0, 3)), z((0, 3)), z(0), z((0, 5)), z((0, 5, 3)), z((0, 5, 3)), z((0, 5)), z((0, 5)),
                   z((0, 5), dtype=bool), z(0, dtype=int), z(0))


def _split(field: DPField, delta: float, panels: _Panels, error: np.ndarray, iterations: int) -> _Panels:
    """Halve every panel: children on ``[t0, t2]`` and ``[t2, t4]``, the parent's points reused, four new points each."""
    m = len(panels)
    t0, t4 = panels.t[:, 0], panels.t[:, -1]
    new_t = t0[:, None] + (t4 - t0)[:, None] * np.array([0.125, 0.375, 0.625, 0.875])
    rows = _evaluate_points(field, delta, np.repeat(panels.a, 4, axis=0), np.repeat(panels.b, 4, axis=0), new_t.reshape(-1), iterations)
    new = {k: v.reshape(m, 4, *v.shape[1:]) for k, v in rows.items()}

    def children(name: str, old: np.ndarray) -> np.ndarray:
        fresh = new[name] if name != "t" else new_t
        left = np.stack([old[:, 0], fresh[:, 0], old[:, 1], fresh[:, 1], old[:, 2]], axis=1)
        right = np.stack([old[:, 2], fresh[:, 2], old[:, 3], fresh[:, 3], old[:, 4]], axis=1)
        return np.concatenate([left, right])

    twice = lambda x: np.concatenate([x, x])  # noqa: E731
    return _Panels(
        twice(panels.component), twice(panels.a), twice(panels.b), twice(panels.theta),
        children("t", panels.t), children("q", panels.q), children("phi", panels.phi), children("d", panels.d),
        children("g", panels.g), children("bad", panels.bad), twice(panels.depth + 1), twice(error),
    )


def _simpson(panels: _Panels, f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Five-point (``N``) and three-point (``N / 2``) Simpson sums of ``f`` ``(P, 5)`` over each panel's parameter interval."""
    h = panels.t[:, -1] - panels.t[:, 0]
    fine = h / 12.0 * (f[:, 0] + 4.0 * f[:, 1] + 2.0 * f[:, 2] + 4.0 * f[:, 3] + f[:, 4])
    coarse = h / 6.0 * (f[:, 0] + 4.0 * f[:, 2] + f[:, 4])
    return fine, coarse


@dataclasses.dataclass
class _Tally:
    value: np.ndarray
    error: float = 0.0
    evaluations: int = 0
    splits: int = 0
    low_order_splits: int = 0
    exhausted: int = 0
    max_depth: int = 0


def _adapt(field: DPField, delta: float, panels: _Panels, integrand, n_components: int, options: QuadratureOptions,
           keep_leaves: bool) -> tuple[_Tally, _Panels | None]:
    """Adaptive Simpson over ``panels``: ``integrand(panels) -> (P, 5)``; per-component values, accepted leaves if asked."""
    tally = _Tally(np.zeros(n_components))
    total_length = float(np.sum(panels.length)) if len(panels) else 0.0
    leaves: list[_Panels] = []
    # only splits made here pair up (i, i + m); the leaves of an earlier stage are no one's children
    active = dataclasses.replace(panels, parent_error=np.full(len(panels), np.inf))
    while len(active):
        f = integrand(active)
        fine, coarse = _simpson(active, f)
        error = np.abs(fine - coarse) / 15.0
        estimate = float(np.sum(tally.value) + np.sum(fine))
        tolerance = max(options.relative_tolerance * abs(estimate), options.absolute_tolerance)
        share = tolerance * active.length / max(total_length, 1e-300)
        # children come in pairs (i, i + m) from one split; the ratio of the parent's estimate to theirs is the local order
        has_parent = np.isfinite(active.parent_error)
        if np.any(has_parent):
            m = int(np.sum(has_parent)) // 2
            idx = np.flatnonzero(has_parent)
            pair_error = error[idx[:m]] + error[idx[m:]]
            parent = active.parent_error[idx[:m]]
            above_roundoff = parent > ROUNDOFF_ULPS * np.finfo(float).eps * np.abs(fine[idx[:m]] + fine[idx[m:]])
            tally.low_order_splits += int(np.sum(above_roundoff & (parent < LOW_ORDER_SPLIT_RATIO * pair_error)))
        exhausted = active.depth >= options.max_depth
        done = (error <= share) | exhausted
        tally.exhausted += int(np.sum(done & exhausted & (error > share)))
        np.add.at(tally.value, active.component[done], fine[done])
        tally.error += float(np.sum(error[done]))
        tally.max_depth = max(tally.max_depth, int(active.depth.max()))
        if keep_leaves:
            leaves.append(active.take(np.flatnonzero(done)))
        rest = np.flatnonzero(~done)
        if len(rest) == 0:
            break
        tally.splits += len(rest)
        tally.evaluations += 4 * len(rest)
        active = _split(field, delta, active.take(rest), error[rest], options.newton_iterations)
    return tally, (_Panels.concatenate(leaves) if keep_leaves and leaves else (_empty_panels() if keep_leaves else None))


# ---- results ------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ContourQuadratureResult:
    """One pixel (or one ``(delta, centre)``): the value, its error estimate and the quadrature diagnostics.

    ``value`` and ``error_estimate`` carry ``1 / (8 pi^2 sin delta)``; the
    ``raw_*`` fields are the line integrals alone.  ``component_values`` are
    per component, in the order of the level set.  ``evaluations`` counts
    the level-set points this pixel added beyond the level set's geometry
    (``geometry_evaluations``); ``exhausted_panels`` passed ``max_depth``
    above their tolerance (the value is reported as is, not as converged);
    ``low_order_splits`` see the module docstring; ``non_finite_points`` had
    ``w > 0`` but a non-finite point or speed (integrand ``0``, counted).
    ``status`` is ``"integrated"``, ``"empty"`` (no component) or
    ``"critical_delta"`` (``delta`` at a critical value; not integrated).
    """

    delta: float
    value: float
    error_estimate: float
    raw_value: float
    raw_error_estimate: float
    component_values: tuple[float, ...]
    n_closed: int
    n_open: int
    panels: int
    evaluations: int
    geometry_evaluations: int
    max_depth: int
    exhausted_panels: int
    low_order_splits: int
    non_finite_points: int
    status: str = "integrated"


def _scale(delta: float) -> float:
    return HAAR_TO_DVOL_G_FACTOR / np.sin(delta)


@dataclasses.dataclass(frozen=True, eq=False)
class LevelSetGeometry:
    """The pixel-independent quadrature geometry of one level set: panels refined on ``w / |grad D|`` (stage one).

    ``evaluations`` counts every level-set point evaluated (node panels and
    splits); ``tally`` holds the stage-one diagnostics.  Build with
    :meth:`build`, integrate pixels with :meth:`integrate`.
    """

    field: DPField
    level_set: LevelSet
    options: QuadratureOptions
    panels: _Panels
    evaluations: int
    low_order_splits: int
    exhausted_panels: int
    max_residual: float

    @classmethod
    def build(cls, field: DPField, level_set: LevelSet, options: QuadratureOptions = QuadratureOptions()) -> "LevelSetGeometry":
        nodes = _node_panels(field, level_set, options.newton_iterations)
        tally, leaves = _adapt(field, level_set.delta, nodes, lambda p: p.g, len(level_set.components), options, keep_leaves=True)
        assert leaves is not None
        residual = float(np.max(np.abs(field.d_p_batch(leaves.q.reshape(-1, 3)) - level_set.delta))) if len(leaves) else 0.0
        return cls(field, level_set, options, leaves, 5 * len(nodes) + tally.evaluations, tally.low_order_splits,
                   tally.exhausted, residual)

    @property
    def delta(self) -> float:
        return self.level_set.delta

    def integrate(self, sun: np.ndarray, centre: np.ndarray, density) -> ContourQuadratureResult:
        """The pixel whose centre (outgoing) direction is ``centre``, for ``s_hat = sun`` and the pose density ``density``."""
        sun = np.asarray(sun, dtype=np.float64)
        centre = np.asarray(centre, dtype=np.float64)

        def integrand(panels: _Panels) -> np.ndarray:
            g = panels.g.reshape(-1)
            live = g != 0.0
            rho = np.zeros_like(g)
            if np.any(live):
                rotations = event_rotations(panels.q.reshape(-1, 3)[live], panels.phi.reshape(-1, 3)[live],
                                            panels.d.reshape(-1)[live], sun, centre)
                rho[live] = density.evaluate_batch(rotations)
            return (g * rho).reshape(-1, 5)

        tally, _ = _adapt(self.field, self.delta, self.panels, integrand, len(self.level_set.components), self.options,
                          keep_leaves=False)
        raw = float(np.sum(tally.value))
        scale = _scale(self.delta)
        return ContourQuadratureResult(
            delta=self.delta,
            value=scale * raw,
            error_estimate=scale * tally.error,
            raw_value=raw,
            raw_error_estimate=tally.error,
            component_values=tuple(float(scale * v) for v in tally.value),
            n_closed=self.level_set.n_closed,
            n_open=self.level_set.n_open,
            panels=len(self.panels) + tally.splits,
            evaluations=tally.evaluations,
            geometry_evaluations=self.evaluations,
            max_depth=tally.max_depth,
            exhausted_panels=self.exhausted_panels + tally.exhausted,
            low_order_splits=self.low_order_splits + tally.low_order_splits,
            non_finite_points=int(np.sum(self.panels.bad)),
            status="integrated" if self.level_set.components else "empty",
        )


def critical_delta(field: DPField, delta: float) -> bool:
    """``delta`` within ``EXTREMUM_ATOL`` of a critical value of ``D_P`` (the level set is not a manifold there)."""
    values = field.critical_values
    return bool(len(values)) and bool(np.min(np.abs(np.asarray(values) - delta)) <= EXTREMUM_ATOL)


def not_integrated(delta: float, status: str) -> ContourQuadratureResult:
    """The record of a pixel that was not integrated (``status`` says why); value ``0``."""
    return ContourQuadratureResult(delta, 0.0, 0.0, 0.0, 0.0, (), 0, 0, 0, 0, 0, 0, 0, 0, 0, status)


__all__ = [
    "METHOD",
    "ContourQuadratureResult",
    "LevelSetGeometry",
    "QuadratureOptions",
    "critical_delta",
    "not_integrated",
]
