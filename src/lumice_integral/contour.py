"""Level sets ``{D_P = delta} ∩ U_P`` on ``S^2``: every component, with the completeness certificate (``docs/phase2.md`` section 4).

For one face sequence (a :class:`.dp_field.DPField`) and a set of
deviations ``delta`` this module returns every connected component of the
level set inside the valid domain ``U_P``: closed loops, and open arcs whose
two ends lie on ``dU_P``, as node sequences on the level set.  The result
depends on ``delta`` alone (no pixel, no azimuth, no sun direction), so every
pixel of one deviation shares it and a ``Phi`` group / ``D6h`` class member
gets it by ``u -> g u`` (:meth:`LevelSet.transported`) instead of a second
extraction.

Seeds, three sources merged per ``delta``:

- **critical data** of the field layer: the crossings of ``delta`` by
  ``D_P`` along the boundary loop (bisection along the piece, pulled
  ``BOUNDARY_SEED_MARGIN`` inside ``U_P``), which are the ends of every open
  arc, and the first crossing along a geodesic ray out of the interior
  extremum, which lies on the closed loop around it.  These reach components
  that no finite sampling resolves: next to a boundary maximum at
  ``delta = value - 1e-6`` the arc it cuts off is ``~1e-6`` rad deep (``~1e-12``
  on an exit-TIR piece, where ``D_P`` is Hoelder-1/2), next to an interior
  extremum the loop is ``~1e-3`` rad across (``5e-7`` at the cone point
  ``D = pi`` of a slab path);
- **the event store** (:meth:`.s2_store.S2EventStore.band_slice`): events
  with ``|D - delta| < h``;
- **marching on a grid**: edges of an orthographic chart of the entry
  hemisphere (``U_P`` lies in ``n_a . u > 0``, the entry incidence cosine is
  a margin of every path) on which ``D - delta`` changes sign, linearly
  interpolated.

The critical-data seeds are walked first; the store and grid seeds are the
independent check: any of them not lying on an extracted component (point to
polyline distance, :func:`_polyline_distance`) is refined and walked, and the
component it finds is an extra one.  The certificate then compares the
number of closed loops and open arcs of every ``delta`` with the interval of
:meth:`.dp_field.DPField.interval_partition` containing it and raises
:class:`ContourCertificateError` on a mismatch (a missed component, or an
extra one the partition did not predict); a :class:`.dp_field.TopologyEscape`
of the partition (a saddle, several interior critical points, a domain that
is not a disk) is propagated, extraction does not run without a prediction.

Walking (:func:`_walk`) is lockstep over every curve of every ``delta``: one
``jax.vmap``-ed step (geodesic predictor along ``cross(u, grad D)``, Newton
corrector along ``grad D`` onto the level set) scanned ``WALK_CHUNK_STEPS``
at a time, finished curves frozen by ``jnp.where`` and compacted out between
chunks; the Python loop runs over chunks, never over components or pixels.
The step is accepted when the corrected point is in ``U_P`` (every margin
positive), on the level set to ``LEVEL_RESIDUAL_TOL`` (or to the rounding of
``D`` where ``|grad D|`` is large), close to the predictor and the tangent
turned by at most ``MAX_TURN_RAD``; a rejected step halves, an accepted one
grows by 1.5 up to ``WALK_MAX_STEP_RAD``.  Leaving ``U_P`` halves the step
down to ``WALK_MIN_STEP_RAD``, which puts an arc end on ``dU_P`` (its
smallest margin is reported).  A walk closes when its start is within one
step ahead: a threshold relative to the current step, never an absolute arc
length (Phase I defect 1, ``docs/phase1.md``: an absolute closure distance
let short loops be walked twice), the rule of the boundary walk
(:mod:`.dp_field.boundary`).

The walker traces ``D_P`` and the margins inside one ``jax.jit``, so it
takes the JAX kernels ``d_value`` / ``margin_vector`` of
:mod:`.dp_field.field` rather than the batched methods of
:class:`.dp_field.DPField`; every entry point takes a built ``DPField``, so
its rank-0 refusal still applies.  The chart grid is independent of the
grid of ``scripts/verify_dp_field_intervals.py`` on purpose: that script is
the oracle of the field layer and does not share code with production.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import NamedTuple, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial import cKDTree

from . import optics
from .dp_field import DPField
from .dp_field.boundary import EXTREMUM_ATOL
from .dp_field.certificate import DeviationInterval
from .dp_field.field import d_value, margin_vector
from .s2_store import S2EventStore

# ---- walk ---------------------------------------------------------------------------------------------
# Largest and first step along a level curve (rad); the boundary walk uses the same largest step.
WALK_MAX_STEP_RAD = float(np.radians(0.25))
WALK_FIRST_STEP_RAD = 1e-4
# A step halved below this ends the walk: on dU_P if a margin is (nearly) violated there, else it stalled.
WALK_MIN_STEP_RAD = 1e-13
# Largest turn of the tangent over one step: at least 72 nodes on the smallest loop.
MAX_TURN_RAD = float(np.radians(5.0))
# Accepted steps after leaving U_P before the step may grow again (approach dU_P by halving only).
CALM_STEPS = 4
WALK_CHUNK_STEPS = 64
MAX_WALK_STEPS = 40000
CORRECTOR_ITERATIONS = 4
# The corrector moves the predictor by at most this fraction of the step.
CORRECTOR_REACH = 0.25
# ---- level-set residual -------------------------------------------------------------------------------
# |D - delta| of a node (rad); where |grad D| is large (next to an exit-TIR curve, D ~ sqrt(margin)) the
# rounding of u alone moves D by eps |grad D|, and a node is accepted within ROUNDING_ULPS of that.
LEVEL_RESIDUAL_TOL = 1e-12
ROUNDING_ULPS = 64.0
SEED_NEWTON_ITERATIONS = 40
SEED_NEWTON_MAX_STEP_RAD = 0.05
# ---- seeds --------------------------------------------------------------------------------------------
# Boundary seeds sit on {margin = this}: below the depth ~1e-12 of the thinnest arc the certificate
# fixtures reach (an exit-TIR piece at 1e-6 rad from a loop maximum), above the margin's rounding.
BOUNDARY_SEED_MARGIN = 1e-14
# A seed pulled 1e-14 inside can have a coincident margin (a square such as exit_snell_discriminant =
# entry_incidence_cosine^2 on 3-5-6-7-3) at 1e-28, outside by rounding: those are pulled this far instead.
BOUNDARY_SEED_MARGIN_RETRY = 1e-8
BISECTION_ITERATIONS = 64
MARGIN_PROJECTION_ITERATIONS = 8
RAY_SAMPLES = 400
DEFAULT_GRID = 401
# Band half width of the store seeds in units of the mean point spacing sqrt(4 pi / N) (rad of D per rad).
BAND_HALFWIDTH_SPACINGS = 2.0
MAX_BAND_SEEDS = 256
# ---- coverage, arc ends -------------------------------------------------------------------------------
# A point lies on an extracted polyline when within this fraction of the nearest segment's length (the chord
# of a step that turns by MAX_TURN_RAD misses the curve by ~1% of its length) plus COVER_ATOL.
COVER_RELATIVE = 0.05
COVER_ATOL = 1e-10
# The walker's U_P: every margin above this.  path_domain_batch (the authority of the gates, through a pose
# round trip) rounds a margin by a few eps; a node at 1.5e-16 was outside for it.
INSIDE_MARGIN_FLOOR = 1e-15
# A walk that stops with its smallest margin below this is on dU_P.
ENDPOINT_MARGIN_ATOL = 1e-8
MAX_EXTRA_ROUNDS = 6
EXTRA_SEEDS_PER_ROUND = 32

_ACTIVE, _CLOSED, _BOUNDARY, _STALLED, _PADDING = 0, 1, 2, 3, 4


class ContourCertificateError(RuntimeError):
    """The extracted components of some ``delta`` do not match the interval partition's prediction."""


# ---- data ---------------------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class ContourComponent:
    """One connected component of ``{D_P = delta} ∩ U_P``.

    ``kind`` is ``"closed"`` (``points[-1]`` is joined back to
    ``points[0]``, which is not repeated) or ``"open"`` (``points[0]`` and
    ``points[-1]`` are on ``dU_P``).  ``points`` ``(K, 3)`` run along
    ``cross(u, grad D_P)`` (``D_P`` above ``delta`` on the right);
    ``residuals`` is ``D_P(points) - delta``.  An open arc carries the
    smallest margin at each end, its name and value (``None`` for a loop).
    """

    delta: float
    kind: str
    points: np.ndarray
    residuals: np.ndarray
    start_margin: str | None = None
    start_margin_value: float | None = None
    end_margin: str | None = None
    end_margin_value: float | None = None

    @property
    def closed(self) -> bool:
        return self.kind == "closed"

    def transported(self, g: np.ndarray) -> "ContourComponent":
        """The component moved by a ``D6h`` element (``D_{gPg^-1}(g u) = D_P(u)``), orientation kept."""
        g = np.asarray(g, dtype=np.float64)
        points = self.points @ g.T
        if np.linalg.det(g) > 0.0:
            return ContourComponent(
                self.delta, self.kind, points, self.residuals,
                self.start_margin, self.start_margin_value, self.end_margin, self.end_margin_value,
            )
        # an improper g reverses cross(u, grad D): reverse the nodes (a loop keeps its first node)
        order = np.r_[0, np.arange(len(points) - 1, 0, -1)] if self.closed else np.arange(len(points) - 1, -1, -1)
        return ContourComponent(
            self.delta, self.kind, points[order], self.residuals[order],
            self.end_margin, self.end_margin_value, self.start_margin, self.start_margin_value,
        )


@dataclass(frozen=True, eq=False)
class LevelSet:
    """Every component of ``{D_P = delta} ∩ U_P`` and the certified interval (``None`` outside ``[min D_P, max D_P]``)."""

    delta: float
    components: tuple[ContourComponent, ...]
    interval: DeviationInterval | None

    @property
    def n_closed(self) -> int:
        return sum(c.closed for c in self.components)

    @property
    def n_open(self) -> int:
        return sum(not c.closed for c in self.components)

    def transported(self, g: np.ndarray) -> "LevelSet":
        """The level set of a class member ``g P g^-1``: every component moved by ``u -> g u``, not re-extracted."""
        return LevelSet(self.delta, tuple(c.transported(g) for c in self.components), self.interval)


# ---- JAX kernels --------------------------------------------------------------------------------------


def _angle(a: jax.Array, b: jax.Array) -> jax.Array:
    return jnp.arctan2(jnp.linalg.norm(jnp.cross(a, b)), jnp.dot(a, b))


def _residual_and_gradient(u: jax.Array, delta: jax.Array, faces, index, slab) -> tuple[jax.Array, jax.Array]:
    """``D_P(u) - delta`` and the tangent gradient of ``D_P``."""
    value, g = jax.value_and_grad(d_value)(u, faces, index, slab)
    return value - delta, g - jnp.dot(g, u) * u


def _residual_tolerance(gradient_norm: jax.Array) -> jax.Array:
    return jnp.maximum(LEVEL_RESIDUAL_TOL, ROUNDING_ULPS * jnp.finfo(jnp.float64).eps * gradient_norm)


def _project_onto_level(u: jax.Array, delta: jax.Array, faces, index, slab, iterations: int, max_step: jax.Array) -> jax.Array:
    """Damped Newton along ``grad D`` onto ``{D_P = delta}`` (fixed iteration count, branch free)."""

    def body(_, v):
        r, g = _residual_and_gradient(v, delta, faces, index, slab)
        step = -r * g / jnp.maximum(jnp.dot(g, g), 1e-300)
        norm = jnp.linalg.norm(step)
        moved = v + step * jnp.minimum(1.0, max_step / jnp.maximum(norm, 1e-300))
        return moved / jnp.linalg.norm(moved)

    return jax.lax.fori_loop(0, iterations, body, u)


def _inside(u: jax.Array, faces, index) -> tuple[jax.Array, jax.Array]:
    """``u in U_P`` (every margin finite and above ``INSIDE_MARGIN_FLOOR``) and the smallest margin."""
    margins = margin_vector(u, faces, index)
    smallest = jnp.min(margins)
    return jnp.all(jnp.isfinite(margins)) & (smallest > INSIDE_MARGIN_FLOOR), smallest


@partial(jax.jit, static_argnums=(2, 5))
def _project_batch(u, delta, faces, index, slab, iterations: int, max_step):
    return jax.vmap(_project_onto_level, in_axes=(0, 0, None, None, None, None, None))(u, delta, faces, index, slab, iterations, max_step)


@partial(jax.jit, static_argnums=1)
def _inside_batch(u, faces, index) -> jax.Array:
    return jax.vmap(lambda v: _inside(v, faces, index)[0])(u)


@partial(jax.jit, static_argnums=2)
def _residual_batch(u, delta, faces, index, slab) -> tuple[jax.Array, jax.Array]:
    """``D - delta`` and its tolerance at each row."""

    def one(v, d):
        r, g = _residual_and_gradient(v, d, faces, index, slab)
        return r, _residual_tolerance(jnp.linalg.norm(g))

    return jax.vmap(one)(u, delta)


def _margin_projection(u: jax.Array, k: jax.Array, target: jax.Array, faces, index) -> jax.Array:
    """Newton along the gradient of margin ``k`` onto ``{margin_k = target}``."""

    def margin(v):
        return margin_vector(v, faces, index)[k]

    def body(_, v):
        m, g = jax.value_and_grad(margin)(v)
        g = g - jnp.dot(g, v) * v
        moved = v - (m - target) * g / jnp.maximum(jnp.dot(g, g), 1e-300)
        return moved / jnp.linalg.norm(moved)

    return jax.lax.fori_loop(0, MARGIN_PROJECTION_ITERATIONS, body, u)


@partial(jax.jit, static_argnums=5)
def _boundary_bisection(a, b, k, delta, sign_a, faces, index, slab, target) -> jax.Array:
    """The point of ``{margin_k = target}`` between ``a`` and ``b`` where ``D_P = delta`` (bisection).

    ``a``, ``b`` are consecutive samples of a boundary piece of margin ``k``
    with ``D - delta`` of sign ``sign_a`` at ``a`` and the opposite at ``b``.
    """

    def point(lam, a, b, k):
        chord = (1.0 - lam) * a + lam * b
        return _margin_projection(chord / jnp.linalg.norm(chord), k, target, faces, index)

    def one(a, b, k, delta, sign_a):
        def body(_, bracket):
            lo, hi = bracket
            mid = 0.5 * (lo + hi)
            same = sign_a * (d_value(point(mid, a, b, k), faces, index, slab) - delta) > 0.0
            return jnp.where(same, mid, lo), jnp.where(same, hi, mid)

        lo, hi = jax.lax.fori_loop(0, BISECTION_ITERATIONS, body, (0.0, 1.0))
        return point(0.5 * (lo + hi), a, b, k)

    return jax.vmap(one)(a, b, k, delta, sign_a)


class _WalkState(NamedTuple):
    u: jax.Array
    start: jax.Array
    delta: jax.Array
    step: jax.Array
    sign: jax.Array
    status: jax.Array
    accepted: jax.Array
    calm: jax.Array
    left_domain: jax.Array


def _walk_step(s: _WalkState, faces, index, slab) -> tuple[_WalkState, jax.Array]:
    """One predictor-corrector step of one curve (module docstring); returns the new state and whether a node was added."""
    active = s.status == _ACTIVE
    _, g = _residual_and_gradient(s.u, s.delta, faces, index, slab)
    tangent = s.sign * jnp.cross(s.u, g) / jnp.maximum(jnp.linalg.norm(g), 1e-300)
    closes = active & (s.accepted >= 2) & (_angle(s.u, s.start) <= s.step) & (jnp.dot(s.start - s.u, tangent) > 0.0)

    predictor = jnp.cos(s.step) * s.u + jnp.sin(s.step) * tangent
    # next to dU_P (an exit-TIR piece above all, where D ~ sqrt(margin) and the curve runs ~1e-12 deep) the
    # geodesic predictor leaves U_P by the boundary's curvature alone: when the smallest margin, to first order
    # along the tangent, stays above half its value but the predictor loses more than that, the predictor is
    # moved onto that first-order margin (a curve heading out of U_P is left alone and ends by halving)
    margins = margin_vector(s.u, faces, index)
    k = jnp.argmin(margins)
    depth = margins[k]
    slope = jnp.dot(jax.grad(lambda v: margin_vector(v, faces, index)[k])(s.u), tangent)
    target = depth + jnp.sin(s.step) * slope
    sagging = (target > 0.5 * depth) & (margin_vector(predictor, faces, index)[k] < 0.5 * target)
    predictor = jnp.where(sagging, _margin_projection(predictor, k, target, faces, index), predictor)
    candidate = _project_onto_level(predictor, s.delta, faces, index, slab, CORRECTOR_ITERATIONS, CORRECTOR_REACH * s.step)
    residual, g_new = _residual_and_gradient(candidate, s.delta, faces, index, slab)
    g_norm = jnp.linalg.norm(g_new)
    new_tangent = s.sign * jnp.cross(candidate, g_new) / jnp.maximum(g_norm, 1e-300)
    turn = _angle(tangent, new_tangent)
    inside, _ = _inside(candidate, faces, index)
    good = (
        inside
        & (jnp.abs(residual) <= _residual_tolerance(g_norm))
        & (_angle(candidate, predictor) <= CORRECTOR_REACH * s.step)
        & (turn <= MAX_TURN_RAD)
        & (jnp.dot(candidate - s.u, tangent) > 0.0)
    )
    accept = active & ~closes & good
    reject = active & ~closes & ~good
    left = jnp.where(reject, ~inside, s.left_domain)
    calm = jnp.where(accept, s.calm + 1, jnp.where(reject & ~inside, 0, s.calm))
    grown = jnp.where((calm >= CALM_STEPS) & (turn <= 0.5 * MAX_TURN_RAD), jnp.minimum(1.5 * s.step, WALK_MAX_STEP_RAD), s.step)
    step = jnp.where(accept, grown, jnp.where(reject, 0.5 * s.step, s.step))
    smallest = depth
    ends = reject & (step < WALK_MIN_STEP_RAD)
    on_boundary = left | (smallest <= ENDPOINT_MARGIN_ATOL)
    status = jnp.where(closes, _CLOSED, jnp.where(ends, jnp.where(on_boundary, _BOUNDARY, _STALLED), s.status))
    state = _WalkState(
        jnp.where(accept, candidate, s.u), s.start, s.delta, step, s.sign, status,
        s.accepted + accept.astype(s.accepted.dtype), calm, left,
    )
    return state, accept


@partial(jax.jit, static_argnums=(1, 4))
def _walk_chunk(state: _WalkState, faces, index, slab, steps: int):
    step = jax.vmap(_walk_step, in_axes=(0, None, None, None))

    def body(s, _):
        s, accepted = step(s, faces, index, slab)
        return s, (s.u, accepted)

    return jax.lax.scan(body, state, None, length=steps)


# ---- walking (host driver) ----------------------------------------------------------------------------


@dataclass(frozen=True)
class _Walk:
    nodes: np.ndarray  # (K, 3), the seed first
    status: int


def _bucket(n: int) -> int:
    """Batch sizes are powers of two (at least 8): one compilation per bucket, not per active count."""
    return max(8, 1 << (n - 1).bit_length())


def _in_buckets(kernel, *arrays):
    """``kernel(*arrays)`` with every array padded along axis 0 to :func:`_bucket` rows (row 0 repeated), cut back after.

    Every batched call here gets a data-dependent row count; unpadded, each
    new count would be one more XLA compilation.
    """
    n = len(arrays[0])
    extra = _bucket(n) - n
    out = kernel(*(np.concatenate([a, np.repeat(a[:1], extra, axis=0)]) for a in map(np.asarray, arrays)))
    if isinstance(out, tuple):
        return tuple(np.asarray(o)[:n] for o in out)
    return np.asarray(out)[:n]


def _walk(field: DPField, seeds: np.ndarray, deltas: np.ndarray, signs: np.ndarray) -> list[_Walk]:
    """Walk every ``(seed, delta, sign)`` in lockstep until it closes, ends on ``dU_P`` or stalls."""
    k = len(seeds)
    if k == 0:
        return []
    u = np.array(seeds, dtype=np.float64)
    start = u.copy()
    delta = np.asarray(deltas, dtype=np.float64)
    step = np.full(k, WALK_FIRST_STEP_RAD)
    sign = np.asarray(signs, dtype=np.float64)
    status = np.full(k, _ACTIVE, dtype=np.int32)
    accepted = np.zeros(k, dtype=np.int32)
    calm = np.full(k, CALM_STEPS, dtype=np.int32)
    left = np.zeros(k, dtype=bool)
    index = jnp.float64(field.index)
    slab = None if field.slab is None else jnp.asarray(field.slab, dtype=jnp.float64)
    logged_curve: list[np.ndarray] = []
    logged_order: list[np.ndarray] = []
    logged_node: list[np.ndarray] = []
    done_steps = 0
    while np.any(status == _ACTIVE):
        if done_steps >= MAX_WALK_STEPS:
            raise RuntimeError(f"{np.sum(status == _ACTIVE)} level curve walk(s) of {optics.path_id_of(field.faces)} "
                               f"did not finish in {MAX_WALK_STEPS} steps")
        rows = np.flatnonzero(status == _ACTIVE)
        size = _bucket(len(rows))
        pad = size - len(rows)

        def batch(a, fill):
            return jnp.asarray(np.concatenate([a[rows], np.repeat(np.asarray(fill, dtype=a.dtype)[None], pad, axis=0)]))

        state = _WalkState(
            batch(u, [0.0, 0.0, 1.0]), batch(start, [0.0, 0.0, 1.0]), batch(delta, 0.0), batch(step, 1.0),
            batch(sign, 1.0), batch(status, _PADDING), batch(accepted, 0), batch(calm, 0), batch(left, False),
        )
        state, (nodes, added) = _walk_chunk(state, field.faces, index, slab, WALK_CHUNK_STEPS)
        n = len(rows)
        u[rows] = np.asarray(state.u)[:n]
        step[rows] = np.asarray(state.step)[:n]
        status[rows] = np.asarray(state.status)[:n]
        accepted[rows] = np.asarray(state.accepted)[:n]
        calm[rows] = np.asarray(state.calm)[:n]
        left[rows] = np.asarray(state.left_domain)[:n]
        added = np.asarray(added)[:, :n]
        t, col = np.nonzero(added)
        logged_curve.append(rows[col])
        logged_order.append(done_steps + t)
        logged_node.append(np.asarray(nodes)[t, col])
        done_steps += WALK_CHUNK_STEPS
    curve = np.concatenate(logged_curve)
    order = np.lexsort((np.concatenate(logged_order), curve))
    curve = curve[order]
    node = np.concatenate(logged_node)[order]
    bounds = np.searchsorted(curve, np.arange(k + 1))
    return [_Walk(np.vstack([seeds[i][None], node[bounds[i] : bounds[i + 1]]]), int(status[i])) for i in range(k)]


# ---- polylines and coverage ---------------------------------------------------------------------------


def _polyline_distance(
    queries: np.ndarray, query_groups: np.ndarray, polylines: Sequence[tuple[int, np.ndarray, bool]]
) -> tuple[np.ndarray, np.ndarray]:
    """Chord distance of each query to the nearest segment of the polylines of its group, and that segment's length.

    ``polylines`` are ``(group, points, closed)``; a query only sees the
    polylines of its own group (one group per ``delta``).  ``inf`` where the
    group has none.
    """
    q = len(queries)
    if q == 0 or not polylines:
        return np.full(q, np.inf), np.zeros(q)
    points, groups, nxt, prv = [], [], [], []
    offset = 0
    for group, pts, closed in polylines:
        k = len(pts)
        ids = np.arange(k) + offset
        after = np.r_[ids[1:], ids[0] if closed and k > 1 else -1]
        before = np.r_[ids[-1] if closed and k > 1 else -1, ids[:-1]]
        points.append(pts)
        groups.append(np.full(k, group))
        nxt.append(after)
        prv.append(before)
        offset += k
    p = np.vstack(points)
    g = np.concatenate(groups)
    nxt_all, prv_all = np.concatenate(nxt), np.concatenate(prv)
    # groups 10 apart in a fourth coordinate: a neighbour query never crosses into another delta (chords <= 2)
    tree = cKDTree(np.column_stack([p, 10.0 * g]))
    k = min(8, len(p))
    _, cand = tree.query(np.column_stack([queries, 10.0 * query_groups]), k=k)
    cand = cand.reshape(q, k)
    best = np.full(q, np.inf)
    length = np.zeros(q)
    for other in (nxt_all, prv_all):
        a_idx = cand
        b_idx = other[cand]
        ok = (b_idx >= 0) & (g[a_idx] == query_groups[:, None])
        a = p[a_idx]
        b = p[np.where(b_idx >= 0, b_idx, a_idx)]
        ab = b - a
        seg = np.einsum("qkd,qkd->qk", ab, ab)
        t = np.clip(np.einsum("qkd,qkd->qk", queries[:, None, :] - a, ab) / np.maximum(seg, 1e-300), 0.0, 1.0)
        dist = np.linalg.norm(queries[:, None, :] - (a + t[..., None] * ab), axis=2)
        dist = np.where(ok, dist, np.inf)
        j = np.argmin(dist, axis=1)
        d = dist[np.arange(q), j]
        better = d < best
        best = np.where(better, d, best)
        length = np.where(better, np.sqrt(seg[np.arange(q), j]), length)
    # a group whose only polyline is a single point: distance to that point
    single = ~np.isfinite(best)
    if np.any(single):
        same = g[cand[single]] == query_groups[single][:, None]
        d = np.where(same, np.linalg.norm(queries[single][:, None, :] - p[cand[single]], axis=2), np.inf)
        best[single] = d.min(axis=1)
    return best, length


def _covered(queries, query_groups, polylines, slack: np.ndarray | float = 0.0) -> np.ndarray:
    distance, length = _polyline_distance(queries, query_groups, polylines)
    return distance <= COVER_ATOL + COVER_RELATIVE * length + slack


# ---- seeds --------------------------------------------------------------------------------------------


def _pairs_in_ranges(lower: np.ndarray, upper: np.ndarray, deltas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """All ``(item, delta index)`` with ``lower[item] < deltas[j] < upper[item]``; ``deltas`` sorted ascending."""
    first = np.searchsorted(deltas, lower, side="right")
    last = np.searchsorted(deltas, upper, side="left")
    counts = np.maximum(last - first, 0)
    item = np.repeat(np.arange(len(lower)), counts)
    j = np.repeat(first, counts) + (np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts))
    return item, j


def _boundary_seeds(field: DPField, deltas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Crossings of every ``delta`` by ``D_P`` along the boundary loop (module docstring): seeds and their delta index."""
    loop = field.boundary
    names = optics.domain_margin_names(field.faces)
    starts, ends, values_a, values_b, margins = [], [], [], [], []
    for piece in loop.pieces:
        points, values = piece.points, piece.values
        # the loop extrema refined inside this piece become samples: a crossing pair next to an extremum
        # can fall between two samples of the 0.25 degree walk
        for extremum in loop.critical_points:
            if extremum.corner or extremum.margin != piece.margin:
                continue
            angles = np.arctan2(np.linalg.norm(np.cross(points, extremum.position), axis=1), points @ extremum.position)
            i = int(np.argmin(angles))
            if angles[i] > 2.0 * WALK_MAX_STEP_RAD or angles[i] == 0.0:
                continue
            neighbour = [j for j in (i - 1, i + 1) if 0 <= j < len(points)]
            j = min(neighbour, key=lambda j: angles[j])
            at = max(i, j)
            points = np.insert(points, at, extremum.position, axis=0)
            values = np.insert(values, at, extremum.value)
        starts.append(points[:-1])
        ends.append(points[1:])
        values_a.append(values[:-1])
        values_b.append(values[1:])
        margins.append(np.full(len(points) - 1, names.index(piece.margin)))
    a, b = np.vstack(starts), np.vstack(ends)
    va, vb = np.concatenate(values_a), np.concatenate(values_b)
    item, j = _pairs_in_ranges(np.minimum(va, vb), np.maximum(va, vb), deltas)
    if len(item) == 0:
        return np.zeros((0, 3)), np.zeros(0, dtype=int)
    sign_a = np.where(va[item] > deltas[j], 1.0, -1.0)
    slab = None if field.slab is None else jnp.asarray(field.slab)
    k = np.concatenate(margins)[item]
    seeds = np.zeros((len(item), 3))
    todo = np.ones(len(item), dtype=bool)
    for target in (BOUNDARY_SEED_MARGIN, BOUNDARY_SEED_MARGIN_RETRY):
        rows = np.flatnonzero(todo)
        if len(rows) == 0:
            break
        seeds[rows] = _in_buckets(
            lambda a_, b_, k_, d_, s_: _boundary_bisection(a_, b_, k_, d_, s_, field.faces, jnp.float64(field.index), slab, jnp.float64(target)),
            a[item][rows], b[item][rows], k[rows], deltas[j][rows], sign_a[rows],
        )
        todo[rows] = ~_seeds_inside(field, seeds[rows])
    return seeds[~todo], j[~todo]


def _seeds_inside(field: DPField, seeds: np.ndarray) -> np.ndarray:
    """Inside for the walker (every margin above ``INSIDE_MARGIN_FLOOR``) and for :meth:`.dp_field.DPField.valid_batch`."""
    if len(seeds) == 0:
        return np.zeros(0, dtype=bool)
    finite = np.all(np.isfinite(seeds), axis=1)
    out = np.zeros(len(seeds), dtype=bool)
    rows = np.flatnonzero(finite)
    if len(rows):
        inside = _in_buckets(lambda u: _inside_batch(jnp.asarray(u), field.faces, jnp.float64(field.index)), seeds[rows])
        out[rows] = inside & _in_buckets(field.valid_batch, seeds[rows])
    return out


def _extremum_seeds(field: DPField, deltas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The first crossing of each ``delta`` along a geodesic ray out of each interior critical point (bisection)."""
    out_seeds, out_j = [], []
    for point in field.interior_critical_points:
        p = np.asarray(point.position, dtype=np.float64)
        axis = np.eye(3)[int(np.argmin(np.abs(p)))]
        e = np.cross(p, axis)
        e /= np.linalg.norm(e)
        radii = np.r_[0.0, np.geomspace(1e-10, 0.5 * np.pi, RAY_SAMPLES - 1)]

        def ray(r):
            return np.cos(r)[:, None] * p + np.sin(r)[:, None] * e

        samples = ray(radii)
        values = field.d_p_batch(samples)
        inside = field.valid_batch(samples)
        inside[0] = True  # the critical point itself: interior by construction
        # the ray's valid prefix
        prefix = len(radii) if inside.all() else int(np.argmin(inside))
        values = values[:prefix]
        sign = np.sign(values[:, None] - deltas[None, :])  # (samples, deltas)
        flips = sign[1:] != sign[:1]
        has = flips.any(axis=0)
        if not np.any(has):
            continue
        j = np.flatnonzero(has)
        first = np.argmax(flips[:, j], axis=0)
        lo, hi = radii[first], radii[first + 1]
        below = sign[0, j]
        for _ in range(BISECTION_ITERATIONS):
            mid = 0.5 * (lo + hi)
            same = np.sign(_in_buckets(field.d_p_batch, ray(mid)) - deltas[j]) == below
            lo, hi = np.where(same, mid, lo), np.where(same, hi, mid)
        out_seeds.append(ray(0.5 * (lo + hi)))
        out_j.append(j)
    if not out_seeds:
        return np.zeros((0, 3)), np.zeros(0, dtype=int)
    seeds, j = np.vstack(out_seeds), np.concatenate(out_j)
    inside = _seeds_inside(field, seeds)
    return seeds[inside], j[inside]


def _grid_seeds(field: DPField, deltas: np.ndarray, grid: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Linear crossings of ``D - delta`` on the edges of a ``grid x grid`` chart of the entry hemisphere, with the edge lengths."""
    n_a = np.asarray(optics.HEXPRISM_BODY_NORMALS[field.faces[0]], dtype=np.float64)
    e1 = np.cross(n_a, np.eye(3)[int(np.argmin(np.abs(n_a)))])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n_a, e1)
    xs = np.linspace(-1.0, 1.0, grid)
    x, y = np.meshgrid(xs, xs, indexing="ij")
    height2 = 1.0 - x**2 - y**2
    on_chart = height2 > 0.0
    u = x[..., None] * e1 + y[..., None] * e2 + np.sqrt(np.clip(height2, 0.0, None))[..., None] * n_a
    value = np.full((grid, grid), np.nan)
    flat = u[on_chart]
    inside = _in_buckets(field.valid_batch, flat)
    value_flat = np.full(len(flat), np.nan)
    value_flat[inside] = _in_buckets(field.d_p_batch, flat[inside])
    value[on_chart] = value_flat
    seeds, j_all, lengths = [], [], []
    for a, b, va, vb in (
        (u[:-1, :], u[1:, :], value[:-1, :], value[1:, :]),
        (u[:, :-1], u[:, 1:], value[:, :-1], value[:, 1:]),
    ):
        ok = np.isfinite(va) & np.isfinite(vb)
        a, b, va, vb = a[ok], b[ok], va[ok], vb[ok]
        item, j = _pairs_in_ranges(np.minimum(va, vb), np.maximum(va, vb), deltas)
        s = (deltas[j] - va[item]) / (vb[item] - va[item])
        point = (1.0 - s)[:, None] * a[item] + s[:, None] * b[item]
        seeds.append(point / np.linalg.norm(point, axis=1, keepdims=True))
        j_all.append(j)
        lengths.append(np.linalg.norm(b[item] - a[item], axis=1))
    return np.vstack(seeds), np.concatenate(j_all), np.concatenate(lengths)


def _band_seeds(field: DPField, store: S2EventStore, deltas: np.ndarray, halfwidth: float) -> tuple[np.ndarray, np.ndarray]:
    """Store events with ``|D - delta| < halfwidth`` inside this path's ``U_P``, at most ``MAX_BAND_SEEDS`` per delta (strided)."""
    seeds, js = [], []
    for j, delta in enumerate(deltas):
        band = store.band_slice(delta - halfwidth, delta + halfwidth).u
        if len(band) > MAX_BAND_SEEDS:
            band = band[np.linspace(0, len(band) - 1, MAX_BAND_SEEDS).astype(int)]
        seeds.append(np.asarray(band, dtype=np.float64))
        js.append(np.full(len(band), j))
    seeds_all, j_all = np.vstack(seeds), np.concatenate(js)
    if len(seeds_all) == 0:
        return seeds_all, j_all
    keep = _in_buckets(field.valid_batch, seeds_all)
    return seeds_all[keep], j_all[keep]


def _refine(field: DPField, seeds: np.ndarray, deltas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Newton onto the level set; returns the points and a mask of those in ``U_P`` with residual within tolerance."""
    if len(seeds) == 0:
        return seeds, np.zeros(0, dtype=bool)
    slab = None if field.slab is None else jnp.asarray(field.slab)
    points = _in_buckets(
        lambda u, d: _project_batch(u, d, field.faces, jnp.float64(field.index), slab, SEED_NEWTON_ITERATIONS, jnp.float64(SEED_NEWTON_MAX_STEP_RAD)),
        seeds, deltas,
    )
    finite = np.all(np.isfinite(points), axis=1)
    good = finite.copy()
    if np.any(finite):
        residual, tolerance = _residuals(field, points[finite], deltas[finite])
        good[finite] = (np.abs(residual) <= tolerance) & _seeds_inside(field, points[finite])
    return points, good


def _residuals(field: DPField, points: np.ndarray, deltas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    slab = None if field.slab is None else jnp.asarray(field.slab)
    return _in_buckets(lambda u, d: _residual_batch(u, d, field.faces, jnp.float64(field.index), slab), points, deltas)


# ---- components ---------------------------------------------------------------------------------------


def _components_of(field: DPField, seeds: np.ndarray, j: np.ndarray, deltas: np.ndarray) -> list[tuple[int, str, np.ndarray]]:
    """Walk each seed both ways and join the halves: ``(delta index, kind, points)`` per seed."""
    walks = _walk(field, np.vstack([seeds, seeds]), np.r_[deltas[j], deltas[j]], np.r_[np.ones(len(seeds)), -np.ones(len(seeds))])
    out = []
    for i in range(len(seeds)):
        forward, backward = walks[i], walks[i + len(seeds)]
        kinds = {forward.status, backward.status}
        if _STALLED in kinds:
            raise RuntimeError(
                f"level curve walk of {optics.path_id_of(field.faces)} stalled at delta = {deltas[j[i]]!r} "
                f"from seed {seeds[i]} (step below {WALK_MIN_STEP_RAD} away from dU_P)"
            )
        if kinds == {_CLOSED}:
            out.append((int(j[i]), "closed", forward.nodes))
        elif kinds == {_BOUNDARY}:
            out.append((int(j[i]), "open", np.vstack([backward.nodes[::-1], forward.nodes[1:]])))
        else:
            raise RuntimeError(
                f"level curve of {optics.path_id_of(field.faces)} at delta = {deltas[j[i]]!r} closes one way and "
                f"ends on dU_P the other (seed {seeds[i]})"
            )
    return out


def _deduplicate(found: list[tuple[int, str, np.ndarray]], kept: list[tuple[int, str, np.ndarray]]) -> list[tuple[int, str, np.ndarray]]:
    """Components of ``found`` whose middle node is not on any kept component (or on one found before it).

    Not an end: an arc meets an exit-TIR piece of ``dU_P`` tangentially, so
    two walks of it stop ``~sqrt(WALK_MIN_STEP_RAD)`` apart along the curve.
    """
    new: list[tuple[int, str, np.ndarray]] = []
    for group, kind, points in found:
        polylines = [(g, p, k == "closed") for g, k, p in kept + new if g == group]
        if polylines and _covered(points[len(points) // 2][None], np.array([group]), polylines)[0]:
            continue
        new.append((group, kind, points))
    return new


def _extract_unique(field: DPField, deltas: np.ndarray, store: S2EventStore, grid: int, halfwidth: float) -> list[list[tuple[str, np.ndarray]]]:
    """Components per (sorted, unique) ``delta``: critical-data seeds, then every uncovered store / grid seed."""
    seeds_b, j_b = _boundary_seeds(field, deltas)
    seeds_e, j_e = _extremum_seeds(field, deltas)
    first_seeds, first_j = np.vstack([seeds_b, seeds_e]), np.r_[j_b, j_e].astype(int)
    kept = _deduplicate(_components_of(field, first_seeds, first_j, deltas), [])

    grid_points, grid_j, edge = _grid_seeds(field, deltas, grid)
    band_points, band_j = _band_seeds(field, store, deltas, halfwidth)

    def polylines():
        return [(g, p, k == "closed") for g, k, p in kept]

    # a grid crossing is within its edge of the true crossing: only one farther than that from every
    # extracted component can be on another component
    far = ~_covered(grid_points, grid_j, polylines(), slack=edge)
    candidates, candidate_j = np.vstack([grid_points[far], band_points]), np.r_[grid_j[far], band_j].astype(int)
    refined, good = _refine(field, candidates, deltas[candidate_j])
    candidates, candidate_j = refined[good], candidate_j[good]
    for _ in range(MAX_EXTRA_ROUNDS):
        uncovered = ~_covered(candidates, candidate_j, polylines())
        candidates, candidate_j = candidates[uncovered], candidate_j[uncovered]
        if len(candidates) == 0:
            break
        pick = np.linspace(0, len(candidates) - 1, min(len(candidates), EXTRA_SEEDS_PER_ROUND)).astype(int)
        kept += _deduplicate(_components_of(field, candidates[pick], candidate_j[pick], deltas), kept)
    else:
        raise RuntimeError(f"{len(candidates)} seed(s) of {optics.path_id_of(field.faces)} still off every component "
                           f"after {MAX_EXTRA_ROUNDS} rounds")
    out: list[list[tuple[str, np.ndarray]]] = [[] for _ in deltas]
    for group, kind, points in kept:
        out[group].append((kind, points))
    return out


def _interval_of(partition: Sequence[DeviationInterval], delta: float) -> DeviationInterval | None:
    """The interval containing ``delta``; ``None`` outside ``[min D_P, max D_P]``; ``ValueError`` at a critical value."""
    breaks = [partition[0].lower] + [iv.upper for iv in partition] if partition else []
    for value in breaks:
        if abs(delta - value) <= EXTREMUM_ATOL:
            raise ValueError(f"delta = {delta!r} is a critical value of D_P ({value!r}): the level set topology changes there")
    for interval in partition:
        if interval.lower < delta < interval.upper:
            return interval
    return None


def _components(field: DPField, deltas: np.ndarray, found: list[list[tuple[str, np.ndarray]]]) -> list[tuple[ContourComponent, ...]]:
    """:class:`ContourComponent` records of every ``delta``: residuals and end margins in one batch each."""
    flat = [(float(delta), kind, points) for delta, components in zip(deltas, found) for kind, points in components]
    if not flat:
        return [() for _ in deltas]
    lengths = [len(points) for _, _, points in flat]
    residuals, _ = _residuals(field, np.vstack([p for _, _, p in flat]), np.repeat([d for d, _, _ in flat], lengths))
    residuals = np.split(residuals, np.cumsum(lengths)[:-1])
    ends = _in_buckets(field.margins_batch, np.vstack([p[[0, -1]] for _, _, p in flat])).reshape(len(flat), 2, -1)
    names = optics.domain_margin_names(field.faces)
    records = []
    for (delta, kind, points), residual, margins in zip(flat, residuals, ends):
        if kind == "closed":
            records.append(ContourComponent(delta, kind, points, residual))
            continue
        k = np.argmin(margins, axis=1)
        records.append(ContourComponent(
            delta, kind, points, residual, names[k[0]], float(margins[0, k[0]]), names[k[1]], float(margins[1, k[1]]),
        ))
    out, at = [], 0
    for components in found:
        out.append(tuple(records[at : at + len(components)]))
        at += len(components)
    return out


def extract_level_sets(
    field: DPField,
    deltas: Sequence[float] | np.ndarray | float,
    store: S2EventStore,
    *,
    grid: int = DEFAULT_GRID,
    band_halfwidth: float | None = None,
) -> tuple[LevelSet, ...]:
    """Every component of ``{D_P = delta} ∩ U_P`` for each ``delta`` (radians), certified (module docstring).

    ``deltas`` may repeat (one extraction per distinct value, shared by
    every position that asks for it: the level set of a pixel depends on its
    ``delta`` only).  ``store`` is an event store of this path's ``Phi``
    group (its events outside this member's ``U_P`` are ignored);
    ``band_halfwidth`` (rad) defaults to ``BAND_HALFWIDTH_SPACINGS`` mean
    point spacings of the store.  Raises :class:`ContourCertificateError`
    when the counts of some ``delta`` differ from the interval partition,
    :class:`.dp_field.TopologyEscape` when the partition itself escapes, and
    ``ValueError`` for a ``delta`` at a critical value.
    """
    requested = np.atleast_1d(np.asarray(deltas, dtype=np.float64))
    if requested.ndim != 1 or not np.all(np.isfinite(requested)):
        raise ValueError("deltas must be finite scalars")
    partition = field.interval_partition()
    unique, back = np.unique(requested, return_inverse=True)
    intervals = [_interval_of(partition, float(delta)) for delta in unique]
    if band_halfwidth is None:
        band_halfwidth = BAND_HALFWIDTH_SPACINGS * float(np.sqrt(4.0 * np.pi / store.spec.n))
    found = _extract_unique(field, unique, store, grid, band_halfwidth)
    level_sets = []
    mismatches = []
    for delta, interval, components in zip(unique, intervals, _components(field, unique, found)):
        level_set = LevelSet(float(delta), components, interval)
        expected = (0, 0) if interval is None else (int(interval.n_closed), int(interval.n_open))
        if (level_set.n_closed, level_set.n_open) != expected:
            mismatches.append(f"delta = {float(delta)!r}: predicted (closed, open) = {expected}, extracted "
                              f"{(level_set.n_closed, level_set.n_open)}")
        level_sets.append(level_set)
    if mismatches:
        raise ContourCertificateError(f"{optics.path_id_of(field.faces)}: " + "; ".join(mismatches))
    return tuple(level_sets[i] for i in back)
