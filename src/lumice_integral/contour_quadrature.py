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
component between two of its nodes ``a``, ``b``: a first panel merges a run
of the extraction's nodes up to ``QuadratureOptions.initial_panel_rad`` of
chord (a loop's last node is joined to its first), a split halves it.  Its
points are ``q(t) = normalize(cos s c(t) + sin s n)``
with ``c(t)`` the great-circle arc from ``a`` to ``b``, ``n`` the arc's pole and
``s(t)`` solved by Newton so that ``D_P(q) = delta``; the arclength speed
``|dq/dt|`` follows from the implicit function theorem (``ds/dt = -D_t /
D_s``, both by ``jax.jvp``).  So every quadrature point is on the level set to
round-off and the chord never stands in for the arc
(``scratchpad/learnings``: a chord parametrisation lowers Simpson's order).
The Newton residual is not asserted at build time (a non-convergent panel
still yields a finite, if wrong, point); it is instead carried through as
``max_residual`` on :class:`ContourQuadratureResult` / :class:`ContourPixelResult`
and rolled up in ``provenance.json``'s ``summary.max_residual_rad`` /
``summary.residual_exceeded_pixels`` (against ``EXTREMUM_ATOL``), so a render
on a non-canonical crystal or an extreme-kink geometry carries evidence of
this invariant rather than only the canonical fixture's unit test.

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

Pixel models (:class:`ContourQuadratureScene`).  The default is the point
value at the pixel centre, Phase I's pixel model.  ``band_nodes = k > 0`` is
the band-sum's model instead: the average over the pixel's deviation band
``[delta_lo, delta_hi]`` (:func:`.band_sum.pixel_band`, the same corners) at
the centre's azimuth,

    I_band = (1 / (delta_hi - delta_lo) sin delta) int_{delta_lo}^{delta_hi} I(delta', alpha) sin(delta') d delta'

by ``k``-point Gauss-Legendre on each piece of the band between the critical
values of ``D_P`` inside it (the band sum is this integral estimated from the
event store, ``docs/phase2.md`` section 4), so the two renderers are compared
on one pixel model (``scratchpad/learnings``: a band average and a point value
differ at steep pixels by the model, not by error).

Rendering (:func:`render_contour_quadrature_window`): the parent builds or
loads the path's event store (the level-set extraction's independent seed
check) once; each worker builds the field and renders whole columns: one
:func:`.contour.extract_level_sets` call for every ``delta`` of the column
("finding the curves"), one :meth:`LevelSetGeometry.build` per ``delta`` and one
:meth:`LevelSetGeometry.integrate` per pixel; the three are timed apart.
Output (:func:`write_contour_quadrature_strip`) is the :mod:`.strip_io`
layout with a contour-quadrature ``pixels.csv`` and ``provenance.json``.

Worker pool (``--workers > 1``): its own ``spawn`` :class:`concurrent.futures.ProcessPoolExecutor`
(:func:`_worker_init` / :func:`_worker_column`), not a reuse of
:mod:`.strip_driver`'s.  Evaluated and rejected: ``strip_driver``'s worker
state is a Phase I ``DriverOptions`` + seed store (continuation-based
fiber discovery with per-column checkpoint files for a resumable render);
this module's worker state is a Phase II :class:`ContourQuadratureScene` +
:class:`.s2_store.S2EventStore` (event-store based, no checkpointing —
:func:`render_contour_quadrature_window` has no ``--resume``).  The two pools
share only the ``spawn`` + per-column job shape, not the state they carry,
so a forced merge would couple two renderers' worker lifecycles for no
shared behaviour.

Nothing here imports or calls Lumice.
"""

from __future__ import annotations

import concurrent.futures
import csv
import dataclasses
import json
import multiprocessing
import time
from functools import partial
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from .band_sum import pixel_band
from .contour import LevelSet, extract_level_sets
from .dp_field import DPField
from .dp_field.boundary import EXTREMUM_ATOL
# Non-underscore dp_field/dp_field.boundary symbols (d_value, validity_margin_vector, EXTREMUM_ATOL) are the
# package's ordinary public surface and are imported directly; only genuinely private names (leading
# underscore, e.g. dp_field.field._PROBE_SUN above) get an independent declaration instead.
from .dp_field.field import d_value, validity_margin_vector
from .quadrature import HAAR_TO_DVOL_G_FACTOR
from .geometry import HexPrism
from .optics import normalize_faces, path_id_of
from .pose_density import PoseDensity
from .provenance import git_commit, sha256_of
from .s2_store import DEFAULT_CACHE_DIR, S2EventStore, align_rotations, build_or_load, evaluate_fields, event_rotations, max_rss_mb
from .strip_io import FILE_NAMES, Window, environment_block, scene_block, write_binary_arrays
from .strip_pixel import (
    STATUS_HAS_ARC,
    STATUS_HAS_COMPONENT,
    STATUS_NODE_COUNT_EXHAUSTED,
    STATUS_QUADRATURE_UNAVAILABLE,
    STATUS_RENDERED,
)

# The body-frame alignment of the field evaluators: R u = this, any twist (the fields depend on u only).
# Same value as dp_field.field._PROBE_SUN, declared here rather than imported across the package's
# private boundary (pinned equal by tests/test_contour_quadrature.py).
_PROBE_SUN = np.array([0.0, 0.0, 1.0])
# Simpson points of a panel, as fractions of its parameter interval; a split adds the odd eighths.
PANEL_FRACTIONS = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
# Level sets per stage-one batch and pixels per stage-two batch (bounds the panel arrays, ~600 panels a level set).
GEOMETRY_CHUNK = 64
JOB_CHUNK = 64
LOW_ORDER_SPLIT_RATIO = 8.0
# Largest total turn of the node intervals merged into one first panel (the level curve stays a graph over the chord).
INITIAL_PANEL_TURN_RAD = float(np.radians(20.0))
# Nodes whose smallest margin of U_P is below this start and end panels of one interval (no merged chord near dU_P).
MERGE_MARGIN = 1e-3
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
    delta``); ``initial_panel_rad`` the largest chord of a first panel (runs
    of the extraction's nodes are merged up to it, and up to
    ``INITIAL_PANEL_TURN_RAD`` of turn): its five points, ``initial_panel_rad
    / 4`` apart, must resolve the narrowest feature of ``rho`` along the curve
    (``1 deg``: points ``0.25 deg`` apart against the canonical ``0.5 deg``
    zenith width), since a feature falling between them is not seen by the
    ``N / 2`` estimate;
    ``max_depth`` halvings of a first panel at most; ``newton_iterations`` of
    the cross-chord Newton (quadratic, from ``s = 0``).
    """

    relative_tolerance: float = 1e-9
    absolute_tolerance: float = 1e-16
    initial_panel_rad: float = float(np.radians(1.0))
    max_depth: int = 24
    newton_iterations: int = 8

    def __post_init__(self) -> None:
        if not (self.relative_tolerance > 0.0 and self.absolute_tolerance > 0.0 and self.initial_panel_rad > 0.0):
            raise ValueError("tolerances and initial_panel_rad must be positive")
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
    return jax.vmap(_panel_point, in_axes=(0, 0, 0, 0, None, None, None, None))(a, b, t, delta, faces, index, slab, iterations)


def _bucket(n: int) -> int:
    """Batch rows padded to a power of two (at least 8): one XLA compilation per size, not per count."""
    return max(8, 1 << (n - 1).bit_length())


# ---- panels -------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, eq=False)
class _Panels:
    """Panels and the pixel-independent data at their five points (arrays aligned on axis 0).

    ``unit`` is the tolerance unit of a panel (a level set in stage one, a
    pixel's level set in stage two) and ``slot`` the component it adds to;
    ``delta`` its level.  ``t`` ``(P, 5)`` are the chord parameters of the
    five Simpson points in the first panel ``a -> b`` (``theta`` its angle);
    ``q``/``phi`` ``(P, 5, 3)`` the level-set points and ``Phi_P(-q)``; ``d``
    ``(P, 5)`` their own deviation; ``g`` ``(P, 5)`` the geometric integrand
    ``w / |grad D| |dq/dt|`` (``0`` outside the weight's support);
    ``residual`` ``(P, 5)`` the Newton residual ``D_P(q) - delta`` at each
    point (``nan`` where non-finite); ``bad`` non-finite points with
    ``w > 0`` (integrand ``0``, counted); ``depth`` halvings so far,
    ``parent_error`` the parent's estimate (``inf`` for a panel not split in
    the current stage).
    """

    unit: np.ndarray
    slot: np.ndarray
    delta: np.ndarray
    a: np.ndarray
    b: np.ndarray
    theta: np.ndarray
    t: np.ndarray
    q: np.ndarray
    phi: np.ndarray
    d: np.ndarray
    g: np.ndarray
    residual: np.ndarray
    bad: np.ndarray
    depth: np.ndarray
    parent_error: np.ndarray

    def __len__(self) -> int:
        return len(self.unit)

    @property
    def length(self) -> np.ndarray:
        """Chord angle of each panel (its share of the tolerance)."""
        return self.theta * (self.t[:, -1] - self.t[:, 0])

    def take(self, index: np.ndarray) -> "_Panels":
        return _Panels(*(getattr(self, f.name)[index] for f in dataclasses.fields(self)))

    def replace(self, **changes) -> "_Panels":
        return dataclasses.replace(self, **changes)

    @staticmethod
    def concatenate(parts: Sequence["_Panels"]) -> "_Panels":
        return _Panels(*(np.concatenate([getattr(p, f.name) for p in parts]) for f in dataclasses.fields(_Panels)))


def _evaluate_points(field: DPField, delta: np.ndarray, a: np.ndarray, b: np.ndarray, t: np.ndarray, iterations: int) -> dict[str, np.ndarray]:
    """Level-set points at ``(a, b, t, delta)`` rows and their pixel-independent data (one JAX batch, one field batch)."""
    n = len(t)
    extra = _bucket(n) - n
    pad = lambda x: np.concatenate([x, np.repeat(x[:1], extra, axis=0)])  # noqa: E731
    slab = None if field.slab is None else jnp.asarray(field.slab)
    a = pad(a)
    q, speed, gradient, residual = (
        np.asarray(x)
        for x in _panel_points(a, pad(b), pad(t), pad(np.asarray(delta, dtype=np.float64)), field.faces,
                               jnp.float64(field.index), slab, iterations)
    )
    finite = np.all(np.isfinite(q), axis=1) & np.isfinite(speed) & np.isfinite(gradient)
    q_safe = np.where(finite[:, None], q, a)
    # the padded rows too: the evaluators' eager vmap compiles per batch size
    fields = evaluate_fields(align_rotations(q_safe, _PROBE_SUN), _PROBE_SUN, field.crystal, field.index, (field.faces,))
    fields = {name: fields[name][:n] for name in ("w", "phi", "D")}
    q_safe, speed, gradient, residual, finite = q_safe[:n], speed[:n], gradient[:n], residual[:n], finite[:n]
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


def _empty_panels() -> _Panels:
    z = np.zeros
    return _Panels(z(0, dtype=int), z(0, dtype=int), z(0), z((0, 3)), z((0, 3)), z(0), z((0, 5)), z((0, 5, 3)), z((0, 5, 3)),
                   z((0, 5)), z((0, 5)), z((0, 5)), z((0, 5), dtype=bool), z(0, dtype=int), z(0))


@partial(jax.jit, static_argnums=1)
def _smallest_margin_kernel(u, faces, index):
    return jax.vmap(lambda v: jnp.min(validity_margin_vector(v, faces, index)))(u)


def _smallest_margin(field: DPField, points: Sequence[np.ndarray]) -> np.ndarray:
    """The smallest margin of ``U_P`` at every row of the stacked ``points`` (one padded batch)."""
    u = np.vstack(points)
    n = len(u)
    padded = np.concatenate([u, np.repeat(u[:1], _bucket(n) - n, axis=0)])
    return np.asarray(_smallest_margin_kernel(padded, field.faces, jnp.float64(field.index)))[:n]


def _panel_breaks(points: np.ndarray, closed: bool, max_chord: float, near_boundary: np.ndarray) -> np.ndarray:
    """Node indices that start the first panels of one component, and the end (``len(points)`` closes a loop).

    A panel runs over consecutive node intervals while its chord stays within
    ``max_chord`` and the intervals' directions turn by at most
    ``INITIAL_PANEL_TURN_RAD`` in total (so the level curve stays a graph
    over the chord); an interval at a node ``near_boundary`` (smallest margin
    below ``MERGE_MARGIN``) is a panel of its own (a merged chord there can
    leave ``U_P``, where ``D_P`` is undefined); one interval is always a panel.
    """
    ends = np.roll(points, -1, axis=0) if closed else points[1:]
    starts = points if closed else points[:-1]
    steps = ends - starts
    steps /= np.maximum(np.linalg.norm(steps, axis=1, keepdims=True), 1e-300)
    turn = np.r_[0.0, np.arccos(np.clip(np.sum(steps[1:] * steps[:-1], axis=1), -1.0, 1.0))]
    near = near_boundary | np.roll(near_boundary, -1) if closed else near_boundary[:-1] | near_boundary[1:]
    breaks = [0]
    turned = 0.0
    for k in range(1, len(steps)):
        turned += turn[k]
        chord = np.arccos(np.clip(points[breaks[-1]] @ ends[k], -1.0, 1.0))
        if chord > max_chord or turned > INITIAL_PANEL_TURN_RAD or near[k] or near[k - 1]:
            breaks.append(k)
            turned = 0.0
    return np.array(breaks + [len(steps)])


def _node_panels(field: DPField, level_sets: Sequence[LevelSet], options: QuadratureOptions) -> tuple[_Panels, np.ndarray]:
    """First panels of every component: runs of its nodes (:func:`_panel_breaks`; a loop closes on its first node).

    Returns the panels and the slots' units: slots number the components of
    all ``level_sets`` in order; ``unit`` is the level set's index.
    """
    iterations = options.newton_iterations
    components = [c for level_set in level_sets for c in level_set.components]
    near = iter(np.split(_smallest_margin(field, [c.points for c in components]) < MERGE_MARGIN,
                         np.cumsum([len(c.points) for c in components])[:-1]) if components else [])
    a, b, unit, slot, slot_unit = [], [], [], [], []
    for j, level_set in enumerate(level_sets):
        for c in level_set.components:
            points = np.asarray(c.points, dtype=np.float64)
            breaks = _panel_breaks(points, c.closed, options.initial_panel_rad, next(near))
            wrapped = np.vstack([points, points[:1]]) if c.closed else points
            starts, ends = wrapped[breaks[:-1]], wrapped[breaks[1:]]
            keep = np.any(starts != ends, axis=1)
            a.append(starts[keep])
            b.append(ends[keep])
            unit.append(np.full(int(keep.sum()), j))
            slot.append(np.full(int(keep.sum()), len(slot_unit)))
            slot_unit.append(j)
    slot_unit = np.array(slot_unit, dtype=int)
    if not a or sum(len(x) for x in a) == 0:
        return _empty_panels(), slot_unit
    a, b = np.vstack(a), np.vstack(b)
    unit, slot = np.concatenate(unit), np.concatenate(slot)
    delta = np.array([level_sets[j].delta for j in unit])
    theta = np.arctan2(np.linalg.norm(np.cross(a, b), axis=1), np.sum(a * b, axis=1))
    t = np.broadcast_to(PANEL_FRACTIONS, (len(a), 5)).copy()
    p = len(a)
    rows = _evaluate_points(field, np.repeat(delta, 5), np.repeat(a, 5, axis=0), np.repeat(b, 5, axis=0), t.reshape(-1), iterations)
    shape = lambda x: x.reshape(p, 5, *x.shape[1:])  # noqa: E731
    panels = _Panels(unit, slot, delta, a, b, theta, t, shape(rows["q"]), shape(rows["phi"]), shape(rows["d"]), shape(rows["g"]),
                     shape(rows["residual"]), shape(rows["bad"]), np.zeros(p, dtype=int), np.full(p, np.inf))
    return panels, slot_unit


def _split(field: DPField, panels: _Panels, error: np.ndarray, iterations: int) -> _Panels:
    """Halve every panel: children on ``[t0, t2]`` and ``[t2, t4]`` (all lefts, then all rights), the parent's points reused."""
    m = len(panels)
    t0, t4 = panels.t[:, 0], panels.t[:, -1]
    new_t = t0[:, None] + (t4 - t0)[:, None] * np.array([0.125, 0.375, 0.625, 0.875])
    rows = _evaluate_points(field, np.repeat(panels.delta, 4), np.repeat(panels.a, 4, axis=0), np.repeat(panels.b, 4, axis=0),
                            new_t.reshape(-1), iterations)
    new = {k: v.reshape(m, 4, *v.shape[1:]) for k, v in rows.items()}
    new["t"] = new_t

    def children(name: str) -> np.ndarray:
        old, fresh = getattr(panels, name), new[name]
        left = np.stack([old[:, 0], fresh[:, 0], old[:, 1], fresh[:, 1], old[:, 2]], axis=1)
        right = np.stack([old[:, 2], fresh[:, 2], old[:, 3], fresh[:, 3], old[:, 4]], axis=1)
        return np.concatenate([left, right])

    twice = lambda x: np.concatenate([x, x])  # noqa: E731
    return _Panels(
        twice(panels.unit), twice(panels.slot), twice(panels.delta), twice(panels.a), twice(panels.b), twice(panels.theta),
        children("t"), children("q"), children("phi"), children("d"), children("g"), children("residual"), children("bad"),
        twice(panels.depth + 1), twice(error),
    )


def _simpson(panels: _Panels, f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Five-point (``N``) and three-point (``N / 2``) Simpson sums of ``f`` ``(P, 5)`` over each panel's parameter interval."""
    h = panels.t[:, -1] - panels.t[:, 0]
    fine = h / 12.0 * (f[:, 0] + 4.0 * f[:, 1] + 2.0 * f[:, 2] + 4.0 * f[:, 3] + f[:, 4])
    coarse = h / 6.0 * (f[:, 0] + 4.0 * f[:, 2] + f[:, 4])
    return fine, coarse


@dataclasses.dataclass
class _Tally:
    """Per-slot values and per-unit diagnostics of one adaptive run.

    ``max_residual`` is the worst Newton residual (``|D_P(q) - delta|``)
    among every panel point evaluated in this run, stage-one or stage-two,
    accepted leaf or later superseded by a split.
    """

    value: np.ndarray
    error: np.ndarray
    splits: np.ndarray
    low_order_splits: np.ndarray
    exhausted: np.ndarray
    max_depth: np.ndarray
    max_residual: np.ndarray

    @classmethod
    def zeros(cls, n_slots: int, n_units: int) -> "_Tally":
        z = lambda dtype: np.zeros(n_units, dtype=dtype)  # noqa: E731
        return cls(np.zeros(n_slots), z(float), z(int), z(int), z(int), z(int), z(float))


def _adapt(field: DPField, panels: _Panels, integrand: Callable[[_Panels], np.ndarray], slot_unit: np.ndarray, n_units: int,
           options: QuadratureOptions, keep_leaves: bool) -> tuple[_Tally, _Panels | None]:
    """Adaptive Simpson over ``panels`` of every unit at once: ``integrand(panels) -> (P, 5)``; the accepted leaves if asked.

    Each unit's tolerance ``max(rtol |I_unit|, atol)`` is shared among its
    panels by chord length; a panel above its share is split (one batch of
    new points for the whole round) until it passes or reaches ``max_depth``.
    """
    tally = _Tally.zeros(len(slot_unit), n_units)
    unit_length = np.bincount(panels.unit, weights=panels.length, minlength=n_units)
    leaves: list[_Panels] = []
    # only splits made here pair up (i, i + m); panels carried over from an earlier stage are no one's children
    active = panels.replace(parent_error=np.full(len(panels), np.inf))
    while len(active):
        abs_residual = np.where(np.isnan(active.residual), -np.inf, np.abs(active.residual))
        panel_residual = np.max(abs_residual, axis=1)
        panel_residual = np.where(np.isneginf(panel_residual), 0.0, panel_residual)
        np.maximum.at(tally.max_residual, active.unit, panel_residual)
        f = integrand(active)
        fine, coarse = _simpson(active, f)
        error = np.abs(fine - coarse) / 15.0
        estimate = np.bincount(slot_unit, weights=tally.value, minlength=n_units) + np.bincount(active.unit, weights=fine, minlength=n_units)
        tolerance = np.maximum(options.relative_tolerance * np.abs(estimate), options.absolute_tolerance)
        share = tolerance[active.unit] * active.length / np.maximum(unit_length[active.unit], 1e-300)
        split_now = np.isfinite(active.parent_error)
        if np.any(split_now):
            m = int(np.sum(split_now)) // 2
            index = np.flatnonzero(split_now)
            left, right = index[:m], index[m:]
            parent = active.parent_error[left]
            above_roundoff = parent > ROUNDOFF_ULPS * np.finfo(float).eps * np.abs(fine[left] + fine[right])
            low = above_roundoff & (parent < LOW_ORDER_SPLIT_RATIO * (error[left] + error[right]))
            tally.low_order_splits += np.bincount(active.unit[left][low], minlength=n_units)
        at_limit = active.depth >= options.max_depth
        done = (error <= share) | at_limit
        tally.exhausted += np.bincount(active.unit[at_limit & (error > share)], minlength=n_units)
        np.add.at(tally.value, active.slot[done], fine[done])
        tally.error += np.bincount(active.unit[done], weights=error[done], minlength=n_units)
        np.maximum.at(tally.max_depth, active.unit, active.depth)
        if keep_leaves:
            leaves.append(active.take(np.flatnonzero(done)))
        rest = np.flatnonzero(~done)
        if len(rest) == 0:
            break
        tally.splits += np.bincount(active.unit[rest], minlength=n_units)
        active = _split(field, active.take(rest), error[rest], options.newton_iterations)
    if not keep_leaves:
        return tally, None
    return tally, (_Panels.concatenate(leaves) if leaves else _empty_panels())


# ---- results ------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ContourQuadratureResult:
    """One ``(delta, centre)``: the value, its error estimate and the quadrature diagnostics.

    ``value`` and ``error_estimate`` carry ``1 / (8 pi^2 sin delta)``; the
    ``raw_*`` fields are the line integrals alone.  ``component_values`` are
    per component, in the order of the level set.  ``evaluations`` counts
    the level-set points this pixel added beyond the level set's geometry
    (``geometry_evaluations``); ``exhausted_panels`` passed ``max_depth``
    above their tolerance (the value is reported as is, not as converged);
    ``low_order_splits`` see the module docstring; ``non_finite_points`` had
    ``w > 0`` but a non-finite point or speed (integrand ``0``, counted).
    ``max_residual`` is the worst ``|D_P(q) - delta|`` among every panel
    point evaluated for this pixel's level set, its geometry (stage one) and
    its own rho-driven refinement (stage two) alike (should be at round-off;
    surfaced so a future non-canonical render can be checked against
    ``EXTREMUM_ATOL`` without a one-off probe script). ``status`` is
    ``"integrated"``, ``"empty"`` (no component) or ``"critical_delta"``
    (``delta`` at a critical value; not integrated).
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
    max_residual: float = 0.0
    status: str = "integrated"


def _scale(delta: float) -> float:
    return HAAR_TO_DVOL_G_FACTOR / np.sin(delta)


@dataclasses.dataclass(frozen=True, eq=False)
class LevelSetGeometry:
    """The pixel-independent quadrature geometry of some level sets: panels refined on ``w / |grad D|`` (stage one).

    ``panels`` are the accepted leaves of every level set (``unit`` = its
    index in ``level_sets``, ``slot`` = its component numbered across all of
    them, ``slot_offsets[j]`` the first slot of level set ``j``); the arrays
    ``evaluations`` / ``low_order_splits`` / ``exhausted_panels`` /
    ``max_depth`` / ``residual_by_unit`` are per level set; ``max_residual``
    is their overall worst case.  Build with :meth:`build`, integrate pixels
    with :meth:`integrate`.
    """

    field: DPField
    level_sets: tuple[LevelSet, ...]
    options: QuadratureOptions
    panels: _Panels
    slot_offsets: np.ndarray
    evaluations: np.ndarray
    low_order_splits: np.ndarray
    exhausted_panels: np.ndarray
    max_residual: float
    residual_by_unit: np.ndarray

    @classmethod
    def build(cls, field: DPField, level_sets: Sequence[LevelSet], options: QuadratureOptions = QuadratureOptions()) -> "LevelSetGeometry":
        level_sets = tuple(level_sets)
        nodes, slot_unit = _node_panels(field, level_sets, options)
        tally, leaves = _adapt(field, nodes, lambda p: p.g, slot_unit, len(level_sets), options, keep_leaves=True)
        assert leaves is not None
        order = np.argsort(leaves.unit, kind="stable")
        leaves = leaves.take(order)
        residual_by_unit = tally.max_residual
        residual = float(residual_by_unit.max()) if len(residual_by_unit) else 0.0
        offsets = np.r_[0, np.cumsum(np.bincount(slot_unit, minlength=len(level_sets)))]
        evaluations = 5 * np.bincount(nodes.unit, minlength=len(level_sets)) + 4 * tally.splits
        return cls(field, level_sets, options, leaves, offsets, evaluations, tally.low_order_splits, tally.exhausted, residual,
                   residual_by_unit)

    def integrate(
        self, sun: np.ndarray, centres: Sequence[np.ndarray] | np.ndarray, density, level_set_index: Sequence[int] | None = None
    ) -> list[ContourQuadratureResult]:
        """Pixels whose centre (outgoing) directions are ``centres``, on level sets ``level_set_index`` (default: in order).

        ``sun`` is ``s_hat``, ``density`` the pose density.  Pixels are
        refined together (stage two), :data:`JOB_CHUNK` at a time.
        """
        sun = np.asarray(sun, dtype=np.float64)
        centres = np.atleast_2d(np.asarray(centres, dtype=np.float64))
        index = np.arange(len(centres)) if level_set_index is None else np.asarray(level_set_index, dtype=int)
        bounds = np.searchsorted(self.panels.unit, np.arange(len(self.level_sets) + 1))
        out: list[ContourQuadratureResult] = []
        for first in range(0, len(centres), JOB_CHUNK):
            out.extend(self._integrate_jobs(sun, centres[first:first + JOB_CHUNK], density, index[first:first + JOB_CHUNK], bounds))
        return out

    def _integrate_jobs(self, sun, centres, density, groups, bounds) -> list[ContourQuadratureResult]:
        sizes = np.diff(self.slot_offsets)[groups]
        job_offsets = np.r_[0, np.cumsum(sizes)]
        parts, slot_unit = [], np.repeat(np.arange(len(groups)), sizes)
        for job, group in enumerate(groups):
            rows = self.panels.take(np.arange(bounds[group], bounds[group + 1]))
            parts.append(rows.replace(unit=np.full(len(rows), job), slot=rows.slot - self.slot_offsets[group] + job_offsets[job]))
        panels = _Panels.concatenate(parts) if parts else _empty_panels()

        def integrand(p: _Panels) -> np.ndarray:
            rho = np.zeros(p.g.shape)
            live = p.g != 0.0
            for job in np.unique(p.unit):
                mask = live & (p.unit == job)[:, None]
                if np.any(mask):
                    rotations = event_rotations(p.q[mask], p.phi[mask], p.d[mask], sun, centres[job])
                    rho[mask] = density.evaluate_batch(rotations)
            return p.g * rho

        tally, _ = _adapt(self.field, panels, integrand, slot_unit, len(groups), self.options, keep_leaves=False)
        results = []
        for job, group in enumerate(groups):
            level_set = self.level_sets[group]
            values = tally.value[job_offsets[job]:job_offsets[job + 1]]
            raw = float(np.sum(values))
            scale = _scale(level_set.delta)
            results.append(ContourQuadratureResult(
                delta=level_set.delta,
                value=scale * raw,
                error_estimate=scale * float(tally.error[job]),
                raw_value=raw,
                raw_error_estimate=float(tally.error[job]),
                component_values=tuple(float(scale * v) for v in values),
                n_closed=level_set.n_closed,
                n_open=level_set.n_open,
                panels=int(bounds[group + 1] - bounds[group] + tally.splits[job]),
                evaluations=int(4 * tally.splits[job]),
                geometry_evaluations=int(self.evaluations[group]),
                max_depth=int(tally.max_depth[job]),
                exhausted_panels=int(self.exhausted_panels[group] + tally.exhausted[job]),
                low_order_splits=int(self.low_order_splits[group] + tally.low_order_splits[job]),
                non_finite_points=int(np.sum(self.panels.bad[bounds[group]:bounds[group + 1]])),
                max_residual=float(max(self.residual_by_unit[group], tally.max_residual[job])),
                status="integrated" if level_set.components else "empty",
            ))
        return results


def critical_delta(field: DPField, delta: float) -> bool:
    """``delta`` within ``EXTREMUM_ATOL`` of a critical value of ``D_P`` (the level set is not a manifold there)."""
    values = np.asarray(field.critical_values, dtype=float)
    return bool(values.size) and bool(np.min(np.abs(values - delta)) <= EXTREMUM_ATOL)


def not_integrated(delta: float, status: str) -> ContourQuadratureResult:
    """The record of a pixel that was not integrated (``status`` says why); value ``0``."""
    return ContourQuadratureResult(
        delta=delta, value=0.0, error_estimate=0.0, raw_value=0.0, raw_error_estimate=0.0, component_values=(),
        n_closed=0, n_open=0, panels=0, evaluations=0, geometry_evaluations=0, max_depth=0, exhausted_panels=0,
        low_order_splits=0, non_finite_points=0, max_residual=0.0, status=status,
    )


# ---- rendering ----------------------------------------------------------------------------------------

FORMAT_VERSION = "lumice-integral.contour-quadrature/v1"
PIXEL_CSV_COLUMNS = (
    "row", "column", "value", "error_estimate", "delta_deg", "band_width_rad", "status", "n_closed", "n_open",
    "level_sets", "panels", "geometry_evaluations", "pixel_evaluations", "max_depth", "exhausted_panels",
    "low_order_splits", "non_finite_points", "max_residual",
)


@dataclasses.dataclass(frozen=True)
class ContourQuadratureScene:
    """What a contour-quadrature render depends on besides the event store (the seed check of the extraction).

    ``sun_direction`` is ``s_hat`` (toward the sun).  ``band_nodes = 0`` is
    the point pixel (Phase I's model), ``k > 0`` the band average by ``k``-point
    Gauss-Legendre per piece of the band (module docstring).
    """

    faces: tuple[int, ...]
    crystal: HexPrism
    refractive_index: float
    sun_direction: np.ndarray
    pose_density: PoseDensity
    render: Mapping[str, Any]
    options: QuadratureOptions = QuadratureOptions()
    band_nodes: int = 0

    def __post_init__(self) -> None:
        if self.band_nodes < 0:
            raise ValueError("band_nodes must be >= 0")
        object.__setattr__(self, "faces", normalize_faces(self.faces))

    @property
    def pixel_model(self) -> str:
        if self.band_nodes == 0:
            return "point value at the pixel centre (the Phase I pixel model)"
        return (f"band average over the pixel's deviation band [delta_lo, delta_hi] at the centre's azimuth "
                f"(the band-sum pixel model), {self.band_nodes}-point Gauss-Legendre per piece between critical values")


@dataclasses.dataclass(frozen=True)
class ContourPixelResult:
    """One rendered pixel: its value under the scene's pixel model and the summed diagnostics of its level sets.

    ``level_sets`` is ``1`` for a point pixel and the number of quadrature
    deviations of a band pixel; ``n_closed`` / ``n_open`` are the largest
    counts among them; the evaluation counts and flags are summed.
    ``status`` is ``"integrated"``, ``"empty"`` (no component: dark) or
    ``"critical_delta"`` (the centre ``delta`` of a point pixel within
    ``EXTREMUM_ATOL`` of a critical value: not integrated, value ``0``).
    ``max_residual`` is the worst level-set point residual among the pixel's
    deviations (see :class:`ContourQuadratureResult`); a non-canonical render
    should compare it against ``EXTREMUM_ATOL`` (``provenance.json``'s
    ``summary.max_residual_rad`` / ``summary.residual_exceeded_pixels``).
    """

    row: int
    column: int
    value: float
    error_estimate: float
    delta: float
    band_width_rad: float
    status: str
    n_closed: int
    n_open: int
    level_sets: int
    panels: int
    geometry_evaluations: int
    pixel_evaluations: int
    max_depth: int
    exhausted_panels: int
    low_order_splits: int
    non_finite_points: int
    max_residual: float = 0.0

    def csv_row(self) -> dict[str, Any]:
        row = dataclasses.asdict(self)
        row["value"] = repr(float(self.value))
        row["error_estimate"] = repr(float(self.error_estimate))
        row["delta_deg"] = repr(float(np.degrees(row.pop("delta"))))
        row["band_width_rad"] = repr(float(self.band_width_rad))
        row["max_residual"] = repr(float(self.max_residual))
        return row


def band_deviations(field: DPField, lo: float, hi: float, nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre deviations and weights on ``[lo, hi]``, per piece between the critical values of ``D_P`` inside it.

    The weights sum to ``hi - lo``; a node within ``EXTREMUM_ATOL`` of a
    critical value is dropped (its piece is that short; the loss is reported
    by the weights' sum).
    """
    x, w = np.polynomial.legendre.leggauss(nodes)
    cuts = [c for c in np.asarray(field.critical_values, dtype=float) if lo < c < hi]
    edges = np.array([lo, *sorted(cuts), hi])
    deltas = (0.5 * (edges[:-1, None] + edges[1:, None]) + 0.5 * np.diff(edges)[:, None] * x).reshape(-1)
    weights = (0.5 * np.diff(edges)[:, None] * w).reshape(-1)
    keep = np.array([not critical_delta(field, d) for d in deltas], dtype=bool)
    return deltas[keep], weights[keep]


def _combine(row: int, column: int, delta: float, width: float, weights: np.ndarray, results: Sequence[ContourQuadratureResult],
             normalise: float) -> ContourPixelResult:
    """Sum ``weights[j] * raw_j`` of the pixel's level sets times ``normalise`` (``1 / 8 pi^2 sin delta``, over the width for a band)."""
    raw = float(np.dot(weights, [r.raw_value for r in results]))
    raw_error = float(np.dot(np.abs(weights), [r.raw_error_estimate for r in results]))
    statuses = {r.status for r in results}
    status = "critical_delta" if "critical_delta" in statuses else ("integrated" if "integrated" in statuses else "empty")
    return ContourPixelResult(
        row=int(row), column=int(column), value=normalise * raw, error_estimate=normalise * raw_error, delta=float(delta),
        band_width_rad=float(width), status=status,
        n_closed=max((r.n_closed for r in results), default=0), n_open=max((r.n_open for r in results), default=0),
        level_sets=len(results), panels=sum(r.panels for r in results),
        geometry_evaluations=sum(r.geometry_evaluations for r in results), pixel_evaluations=sum(r.evaluations for r in results),
        max_depth=max((r.max_depth for r in results), default=0), exhausted_panels=sum(r.exhausted_panels for r in results),
        low_order_splits=sum(r.low_order_splits for r in results), non_finite_points=sum(r.non_finite_points for r in results),
        max_residual=max((r.max_residual for r in results), default=0.0),
    )


def render_column(
    scene: ContourQuadratureScene, field: DPField, store: S2EventStore, column: int, rows: Sequence[int]
) -> tuple[list[ContourPixelResult], dict[str, float]]:
    """Every pixel of ``rows`` in ``column``; the wall clock of curve finding, geometry and per-pixel integration.

    One :func:`.contour.extract_level_sets` call for every deviation of the
    column, then :meth:`LevelSetGeometry.build` on :data:`GEOMETRY_CHUNK`
    level sets at a time and one :meth:`LevelSetGeometry.integrate` per chunk
    for every pixel that uses its level sets.
    """
    sun = np.asarray(scene.sun_direction, dtype=np.float64)
    timing = {"extract_s": 0.0, "geometry_s": 0.0, "integrate_s": 0.0}
    plans = []  # (row, centre, delta, width, deviations, weights, critical)
    for row in rows:
        centre, delta, lo, hi = pixel_band(row, column, sun, scene.render)
        if scene.band_nodes:
            deviations, weights = band_deviations(field, lo, hi, scene.band_nodes)
            plans.append((row, centre, delta, hi - lo, deviations, weights, False))
        else:
            critical = critical_delta(field, delta)
            plans.append((row, centre, delta, hi - lo, np.array([] if critical else [delta]), np.ones(0 if critical else 1), critical))
    wanted = np.unique(np.concatenate([p[4] for p in plans])) if plans else np.zeros(0)
    start = time.perf_counter()
    level_sets = extract_level_sets(field, wanted, store) if len(wanted) else ()
    timing["extract_s"] = time.perf_counter() - start
    position = {float(d): k for k, d in enumerate(wanted)}
    jobs = [(i, position[float(d)], centre) for i, (_, centre, _, _, deviations, _, _) in enumerate(plans) for d in deviations]
    per_job: dict[tuple[int, int], ContourQuadratureResult] = {}
    for first in range(0, len(level_sets), GEOMETRY_CHUNK):
        start = time.perf_counter()
        geometry = LevelSetGeometry.build(field, level_sets[first:first + GEOMETRY_CHUNK], scene.options)
        timing["geometry_s"] += time.perf_counter() - start
        mine = [(i, k, centre) for i, k, centre in jobs if first <= k < first + GEOMETRY_CHUNK]
        start = time.perf_counter()
        results = geometry.integrate(sun, [c for _, _, c in mine], scene.pose_density, [k - first for _, k, _ in mine])
        timing["integrate_s"] += time.perf_counter() - start
        per_job.update({(i, k): r for (i, k, _), r in zip(mine, results)})
    out = []
    for i, (row, centre, delta, width, deviations, weights, critical) in enumerate(plans):
        results = [per_job[(i, position[float(d)])] for d in deviations]
        if critical:
            results, weights = [not_integrated(delta, "critical_delta")], np.zeros(1)
        normalise = _scale(delta) / (width if scene.band_nodes else 1.0)
        out.append(_combine(row, column, delta, width, weights, results, normalise))
    return out, timing


_WORKER: dict[str, Any] = {}


def _worker_init(scene: ContourQuadratureScene, store_directory: Path) -> None:
    field = DPField.build(scene.crystal, scene.faces, scene.refractive_index)
    field.interval_partition()  # the cached layers, once per worker
    _WORKER.update(scene=scene, field=field, store=S2EventStore.load(store_directory))


def _worker_column(job: tuple[int, Sequence[int]]) -> tuple[list[ContourPixelResult], dict[str, float], float, float]:
    column, rows = job
    start = time.perf_counter()
    results, timing = render_column(_WORKER["scene"], _WORKER["field"], _WORKER["store"], column, rows)
    return results, timing, time.perf_counter() - start, max_rss_mb()


def render_contour_quadrature_window(
    scene: ContourQuadratureScene,
    window: Window,
    store_n: int,
    *,
    workers: int = 1,
    base_dir: Path = DEFAULT_CACHE_DIR,
    run_checks: bool = True,
    log: Callable[[str], None] | None = None,
) -> tuple[list[ContourPixelResult], dict[str, Any]]:
    """Render every pixel of ``window``: the path's store built or loaded once, then columns across ``workers``.

    Returns the pixel results and an execution record: the store, wall
    clocks, and the summed CPU seconds of curve finding (``extract_s``),
    level-set geometry (``geometry_s``) and per-pixel integration
    (``integrate_s``), with the number of distinct deviations.
    """
    if workers < 1:
        raise ValueError("workers must be positive")
    start = time.perf_counter()
    store = build_or_load(scene.crystal, scene.refractive_index, [scene.faces], store_n, base_dir=base_dir, mmap_mode="r",
                          run_checks=run_checks, log=log)
    directory = Path(base_dir) / store.spec.cache_key()
    store_record = {"cache_key": store.spec.cache_key(), "directory": str(directory), "kept_events": int(len(store.events)),
                    "build_parameters": store.spec.build_parameters()}
    del store
    execution: dict[str, Any] = {"workers": workers, "store": store_record, "store_wall_clock_s": time.perf_counter() - start}
    jobs = [(column, list(window.row_range)) for column in window.column_range]
    render_start = time.perf_counter()
    results: list[ContourPixelResult] = []
    totals = {"extract_s": 0.0, "geometry_s": 0.0, "integrate_s": 0.0, "column_s": 0.0}
    worker_rss = 0.0

    def collect(index: int, chunk, timing, seconds, rss) -> None:
        nonlocal worker_rss
        results.extend(chunk)
        for key, value in timing.items():
            totals[key] += value
        totals["column_s"] += seconds
        worker_rss = max(worker_rss, rss)
        if log is not None and (index + 1) % 25 == 0:
            log(f"{index + 1}/{len(jobs)} columns, {time.perf_counter() - render_start:.1f} s")

    if workers == 1:
        _worker_init(scene, directory)
        execution["in_process_init_s"] = time.perf_counter() - render_start
        for index, job in enumerate(jobs):
            collect(index, *_worker_column(job))
        _WORKER.clear()
    else:
        context = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers, mp_context=context, initializer=_worker_init,
                                                    initargs=(scene, directory)) as executor:
            for index, output in enumerate(executor.map(_worker_column, jobs)):
                collect(index, *output)
    pixels = max(len(results), 1)
    execution.update(
        render_wall_clock_s=time.perf_counter() - render_start,
        wall_clock_s=time.perf_counter() - start,
        cpu_seconds=totals,
        distinct_deviations=int(sum(r.level_sets for r in results)),
        per_pixel_mean_s=totals["column_s"] / pixels,
        max_rss_mb_parent=max_rss_mb(),
        max_rss_mb_worker=worker_rss,
    )
    return results, execution


def write_contour_quadrature_strip(
    output_dir: Path,
    results: Sequence[ContourPixelResult],
    *,
    scene: ContourQuadratureScene,
    window: Window,
    pose_density_block: Mapping[str, Any],
    execution: Mapping[str, Any],
    repo: Path | None = None,
) -> dict[str, Path]:
    """The :mod:`.strip_io` layout (``read_strip`` reads it) with a contour-quadrature CSV and provenance."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    height, width = int(scene.render["height"]), int(scene.render["width"])
    values = np.zeros((height, width), dtype=np.float64)
    status = np.zeros((height, width), dtype=np.uint8)
    component_count = np.zeros((height, width), dtype=np.uint8)
    for r in results:
        bits = STATUS_RENDERED
        bits |= STATUS_HAS_COMPONENT if r.n_closed + r.n_open else 0
        bits |= STATUS_HAS_ARC if r.n_open else 0
        bits |= STATUS_NODE_COUNT_EXHAUSTED if r.exhausted_panels else 0
        bits |= STATUS_QUADRATURE_UNAVAILABLE if r.status == "critical_delta" else 0
        values[r.row, r.column] = r.value
        status[r.row, r.column] = bits
        component_count[r.row, r.column] = min(r.n_closed + r.n_open, 255)
    files = write_binary_arrays(output_dir, values, status, component_count)
    files["pixels"] = output_dir / FILE_NAMES["pixels"]
    with files["pixels"].open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PIXEL_CSV_COLUMNS)
        writer.writeheader()
        for r in sorted(results, key=lambda r: (r.column, r.row)):
            writer.writerow(r.csv_row())
    rendered = (status & STATUS_RENDERED) != 0
    rendered_values = values[rendered]
    lit = [r for r in results if r.value > 0.0]
    relative_error = np.array([r.error_estimate / r.value for r in lit])
    scene_json = scene_block(pose_density_block)
    scene_json["path"] = {"value": list(scene.faces), "provenance": "run-option"}
    scene_json["camera"] = {"value": {"lens": "linear", **dict(scene.render)}, "provenance": "run-option"}
    scene_json["image_shape"] = {"value": [height, width], "provenance": "run-option"}
    provenance = {
        "format": FORMAT_VERSION,
        "product": "S^2 contour-quadrature render (deterministic level-set line integrals, point light source)",
        "generator": {
            "package": "lumice_integral",
            "modules": ["contour_quadrature", "contour", "dp_field", "s2_store", "strip_io"],
            "git_commit": git_commit(repo),
            "lumice_dependency": "none (independent implementation; Lumice is neither imported nor invoked)",
        },
        "scene": scene_json,
        "options": {
            "method": METHOD,
            "pixel_model": scene.pixel_model,
            "band_nodes": scene.band_nodes,
            "quadrature": scene.options.as_json(),
            "haar_to_dvol_g_factor": "1/(8*pi**2)",
            "path": path_id_of(scene.faces),
            "store": execution.get("store"),
            "critical_delta_tolerance_rad": EXTREMUM_ATOL,
        },
        "window": window.as_json(),
        "arrays": {
            "shape": [height, width],
            "order": "row-major, row 0 at the top of the image, column 0 at the left",
            "files": {
                "float64": {"name": FILE_NAMES["float64"], "dtype": "<f8", "sha256": sha256_of(files["float64"])},
                "float32": {"name": FILE_NAMES["float32"], "dtype": "<f4", "sha256": sha256_of(files["float32"])},
                "status": {"name": FILE_NAMES["status"], "dtype": "u1", "sha256": sha256_of(files["status"])},
                "component_count": {"name": FILE_NAMES["component_count"], "dtype": "u1", "sha256": sha256_of(files["component_count"])},
                "pixels": {"name": FILE_NAMES["pixels"], "sha256": sha256_of(files["pixels"]), "columns": list(PIXEL_CSV_COLUMNS)},
            },
            "status_bits": {
                "rendered": STATUS_RENDERED, "has_component": STATUS_HAS_COMPONENT, "has_arc": STATUS_HAS_ARC,
                "node_count_exhausted": STATUS_NODE_COUNT_EXHAUSTED, "quadrature_unavailable": STATUS_QUADRATURE_UNAVAILABLE,
            },
            "status_bit_meanings": {
                "rendered": "pixel was computed (0 = outside the rendered window; its value is a placeholder 0)",
                "has_component": "the level set (any band deviation) has a component; component_count is the largest count",
                "has_arc": "an open arc (ends on dU_P) among the components",
                "node_count_exhausted": "a panel reached max_depth above its tolerance (value reported as is)",
                "quadrature_unavailable": "the centre delta is a critical value of D_P: not integrated, value 0",
                "completeness": (
                    "no unknown_completeness bit: every level set is certified against the interval partition of "
                    "dp_field (contour.extract_level_sets raises otherwise)"
                ),
            },
            "value_semantics": scene.pixel_model + "; error_estimate is the summed |S_N - S_N/2| / 15 of the panels",
            "radiometric_normalization": "the Phase I pixel normalisation (1/8 pi^2 Haar conversion); not aligned with Lumice",
        },
        "summary": {
            "rendered_pixels": int(rendered.sum()),
            "pixels_with_light": int((values > 0.0).sum()),
            "critical_delta_pixels": int(sum(r.status == "critical_delta" for r in results)),
            "exhausted_pixels": int(sum(r.exhausted_panels > 0 for r in results)),
            "value_min": float(rendered_values.min()) if rendered_values.size else None,
            "value_max": float(rendered_values.max()) if rendered_values.size else None,
            "value_mean": float(rendered_values.mean()) if rendered_values.size else None,
            "relative_error_estimate_max_lit": float(relative_error.max()) if relative_error.size else None,
            "relative_error_estimate_median_lit": float(np.median(relative_error)) if relative_error.size else None,
            "residual_tolerance_rad": EXTREMUM_ATOL,
            "max_residual_rad": max((r.max_residual for r in results), default=0.0),
            "residual_exceeded_pixels": int(sum(r.max_residual > EXTREMUM_ATOL for r in results)),
        },
        "execution": dict(execution),
        "environment": environment_block(),
    }
    files["provenance"] = output_dir / FILE_NAMES["provenance"]
    files["provenance"].write_text(json.dumps(provenance, indent=2) + "\n")
    return files


__all__ = [
    "FORMAT_VERSION",
    "METHOD",
    "PIXEL_CSV_COLUMNS",
    "ContourPixelResult",
    "ContourQuadratureResult",
    "ContourQuadratureScene",
    "LevelSetGeometry",
    "QuadratureOptions",
    "band_deviations",
    "critical_delta",
    "not_integrated",
    "render_column",
    "render_contour_quadrature_window",
    "write_contour_quadrature_strip",
]
