"""Weight kinks: the TIR onsets ``C_k = {internal_k_tir_discriminant = 0}`` in the closure of ``U_P``.

The third kind of critical line of the field layer, next to the critical
points of ``D_P`` (:mod:`.field`) and the boundary ``dU_P``
(:mod:`.boundary`).  A partial internal reflection keeps the pose in
``U_P``: the internal step ``k``'s Fresnel ``R_k``
(:func:`.optics.internal_reflectance`) is ``1`` on the total side and drops
continuously, with an unbounded normal derivative, on the partial side.
``C_k`` is therefore a kink of the path's weight, not of its support, and
it moves with ``n`` while the gates of a slab path do not: the mechanism of
the blue edge of the random-orientation ``3-1-6`` / ``1-3-2`` dark hole and
of the blue tint of the plate ``1-3-5-2`` class (task
``chromatic-weight-kink-diagnostic``).  Nothing here touches ``dU_P`` or the
topology certificate of :mod:`.certificate`.

Two ways to find ``C_k`` (the same split as :mod:`.boundary`'s curve kinds):

- ``great_circle`` normal: when ``m_k = R_{k-1}^T n_k`` (the step's
  unfolded incidence normal, :func:`.boundary.incidence_normals`) is
  orthogonal to the entry normal ``n_a``, the entry refraction keeps the
  tangential component, ``incidence_cosine = -(m_k . u) / n`` and
  ``disc_k = n^2 - 1 - (m_k . u)^2``: ``C_k`` inside the incidence gate is
  the small circle ``m_k . u = -sqrt(n^2 - 1)``, clipped to ``U_P`` by
  bisection on its angle.  Closed form; it is the authority for every path
  whose ``m_k`` passes the check (the ``1e-12`` normal test of
  :data:`.boundary.GREAT_CIRCLE_ATOL`, plus the residual on the circle).
  On a single-mirror slab (``3-1-6``, ``1-3-2``: ``M`` the mirror of ``m``)
  ``D_P = 2 arcsin |m . u|`` is constant on it, ``2 arcsin sqrt(n^2 - 1)``.
- ``marched``: otherwise, predictor-corrector along the zero set of
  ``disc_k`` with :func:`.boundary.walk_zero_set` (its :class:`.boundary.Walker`
  steppers take any margin name), from lattice seeds near
  the zero set, both ways until a gate of ``U_P`` stops it or the walk
  closes.  Seeds within a few steps of a walked arc are dropped, the rest
  start new arcs; the arcs found are *not* certified to be all of ``C_k``
  (``AGENTS.md``: one closed loop is no proof of completeness).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .. import optics
from ..geometry import Polyhedron, unit
from ..s2_store import fibonacci_sphere
from .boundary import GREAT_CIRCLE_ATOL, WALK_STEP_RAD, Walker, incidence_normals, walk_zero_set
from .field import d_p_batch, margins_batch, validity_margins_batch

Faces = tuple[int, ...]

# Samples of a closed-form circle before its U_P arcs are cut by bisection (0.1 deg apart).
CIRCLE_SAMPLES = 3600
# Bisection on the circle angle stops below this (rad).
ARC_END_ATOL = 1e-13
# The closed form is accepted when the discriminant on the circle is within this of zero.
CIRCLE_RESIDUAL_ATOL = 1e-12
# Marched walk: lattice seeds with |disc_k| below this start a walk (the discriminant is O(1) over S^2).
SEED_BAND = 0.02
# A seed within this many walk steps of an arc already walked starts nothing new.
SEED_COVERED_STEPS = 4.0


@dataclass(frozen=True)
class KinkArc:
    """One arc of ``C_k`` inside the closure of ``U_P``.

    ``points`` ``(N, 3)`` in curve order and ``values`` ``D_P`` there
    (radians); ``closed`` for a whole loop inside ``U_P`` (then ``ends`` is
    ``(None, None)``), otherwise ``ends`` names the gate that stops the arc
    at its first and last point.
    """

    points: np.ndarray
    values: np.ndarray
    closed: bool
    ends: tuple[str | None, str | None]


@dataclass(frozen=True)
class KinkCurve:
    """``C_k``: the TIR onset of internal step ``step`` at refractive index ``index`` (module docstring).

    ``method`` is ``"great_circle"`` or ``"marched"``; ``normal`` is
    ``m_k`` for the closed form (``None`` when marched).  ``arcs`` may be
    empty: the onset misses ``U_P`` (every pose of the path reflects
    totally, or none does, at that step).  ``note`` says why a marched curve
    has no arcs when a walk failed (the error is reported, not hidden).
    """

    step: int
    margin: str
    index: float
    method: str
    normal: np.ndarray | None
    arcs: tuple[KinkArc, ...]
    note: str = ""

    @property
    def points(self) -> np.ndarray:
        return np.concatenate([arc.points for arc in self.arcs]) if self.arcs else np.zeros((0, 3))

    @property
    def values(self) -> np.ndarray:
        return np.concatenate([arc.values for arc in self.arcs]) if self.arcs else np.zeros(0)

    @property
    def spread(self) -> float:
        """``sigma_k``: the width ``max - min`` of ``D_P`` on ``C_k`` (radians; ``nan`` without arcs, ``0`` on a single-mirror slab)."""
        return float(np.ptp(self.values)) if self.arcs else float("nan")


def weight_kinks(
    crystal: Polyhedron, faces: Faces, index: float, *, slab: np.ndarray | None, lattice_n: int = 20000
) -> tuple[KinkCurve, ...]:
    """One :class:`KinkCurve` per internal reflection of ``faces`` at ``index``, in step order (``()`` for ``a-b`` paths)."""
    walker = Walker(crystal, faces, index, slab)
    incidence = incidence_normals(crystal, faces)
    n_a = incidence["entry_incidence_cosine"]
    out = []
    for step in range(1, len(faces) - 1):
        margin = f"internal_{step}_tir_discriminant"
        m = incidence[f"internal_{step}_incidence_cosine"]
        circle = _circle_curve(walker, step, margin, unit(m), index) if abs(m @ n_a) <= GREAT_CIRCLE_ATOL else None
        out.append(circle if circle is not None else _marched_curve(walker, step, margin, index, lattice_n))
    return tuple(out)


# ---- closed form ----------------------------------------------------------------------------------------


def _circle_curve(walker: Walker, step: int, margin: str, m: np.ndarray, index: float) -> KinkCurve | None:
    """``m . u = -sqrt(n^2 - 1)`` clipped to ``U_P``; ``None`` if the discriminant does not vanish on it (fall back to the walk)."""
    height = np.sqrt(index * index - 1.0)
    e1 = unit(np.cross(m, np.eye(3)[int(np.argmin(np.abs(m)))]))
    e2 = np.cross(m, e1)
    radius = np.sqrt(1.0 - height * height)

    def at(t: np.ndarray) -> np.ndarray:
        t = np.atleast_1d(t)
        return -height * m + radius * (np.cos(t)[:, None] * e1 + np.sin(t)[:, None] * e2)

    t = np.linspace(0.0, 2.0 * np.pi, CIRCLE_SAMPLES, endpoint=False)
    u = at(t)
    k = walker.k(margin)
    residual = max(abs(float(walker.margins(p)[k])) for p in u[:: CIRCLE_SAMPLES // 16])
    if residual > CIRCLE_RESIDUAL_ATOL:
        return None
    gates = validity_margins_batch(u, walker.faces, walker.index, crystal=walker.crystal)
    inside = np.all(gates > 0.0, axis=1)
    arcs = []
    if inside.all():
        arcs.append(_arc(walker, np.append(t, 2.0 * np.pi), at, closed=True, ends=(None, None)))
    elif inside.any():
        # rotate the sample order so that it starts outside: every run of inside samples is then one arc
        shift = int(np.flatnonzero(~inside)[0])
        order = np.roll(np.arange(CIRCLE_SAMPLES), -shift)
        runs = np.split(order, np.flatnonzero(np.diff(inside[order].astype(int)) != 0) + 1)
        step_t = 2.0 * np.pi / CIRCLE_SAMPLES
        for run in runs:
            if not inside[run[0]]:
                continue
            t0 = t[run[0]]
            t1 = t0 + step_t * (len(run) - 1)
            start, start_gate = _bisect_end(walker, at, t0, t0 - step_t)
            stop, stop_gate = _bisect_end(walker, at, t1, t1 + step_t)
            grid = np.concatenate([[start], t0 + step_t * np.arange(len(run)), [stop]])
            arcs.append(_arc(walker, grid, at, closed=False, ends=(start_gate, stop_gate)))
    return KinkCurve(step, margin, float(index), "great_circle", m, tuple(arcs))


def _bisect_end(walker: Walker, at, inside_t: float, outside_t: float) -> tuple[float, str]:
    """The angle where the circle leaves ``U_P`` between an inside and an outside sample, and the gate that stops it."""
    names = optics.validity_margin_names(walker.faces)
    lo, hi = inside_t, outside_t
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if np.all(validity_margins_batch(at(mid), walker.faces, walker.index, crystal=walker.crystal)[0] > 0.0):
            lo = mid
        else:
            hi = mid
        if abs(hi - lo) < ARC_END_ATOL:
            break
    gates = validity_margins_batch(at(hi), walker.faces, walker.index, crystal=walker.crystal)[0]
    return lo, names[int(np.argmin(gates))]


def _arc(walker: Walker, t: np.ndarray, at, *, closed: bool, ends: tuple[str | None, str | None]) -> KinkArc:
    points = at(t)
    values = d_p_batch(points, walker.faces, walker.index, walker.slab, crystal=walker.crystal)
    return KinkArc(points, values, closed, ends)


# ---- marched ---------------------------------------------------------------------------------------------


def marched_kink(
    crystal: Polyhedron, faces: Faces, index: float, step: int, *, slab: np.ndarray | None, lattice_n: int = 20000
) -> KinkCurve:
    """``C_k`` of internal step ``step`` by the walk alone, whatever its normal (the closed form's cross-check)."""
    walker = Walker(crystal, faces, index, slab)
    return _marched_curve(walker, step, f"internal_{step}_tir_discriminant", index, lattice_n)


def _marched_curve(walker: Walker, step: int, margin: str, index: float, lattice_n: int) -> KinkCurve:
    k = walker.k(margin)
    lattice = fibonacci_sphere(lattice_n)
    gates = validity_margins_batch(lattice, walker.faces, walker.index, crystal=walker.crystal)
    lattice = lattice[np.all(gates > 0.0, axis=1)]
    if len(lattice) == 0:
        return KinkCurve(step, margin, float(index), "marched", None, ())
    disc = margins_batch(lattice, walker.faces, walker.index, crystal=walker.crystal)[:, k]
    near = np.abs(disc) < SEED_BAND
    seeds = lattice[near][np.argsort(np.abs(disc[near]))]
    arcs: list[KinkArc] = []
    walked: list[np.ndarray] = []
    covered = SEED_COVERED_STEPS * WALK_STEP_RAD
    try:
        for seed in seeds:
            if walked and cKDTree(np.concatenate(walked)).query(seed)[0] < 2.0 * np.sin(covered / 2.0):
                continue
            start = walker.correct(seed, margin)
            if abs(walker.margins(start)[k]) > 1e-12 or walker.violated(start, {margin}):
                continue
            arc = _walk_both_ways(walker, start, margin)
            arcs.append(arc)
            walked.append(arc.points)
    except RuntimeError as error:
        return KinkCurve(step, margin, float(index), "marched", None, tuple(arcs), note=str(error))
    return KinkCurve(step, margin, float(index), "marched", None, tuple(arcs))


def _walk_both_ways(walker: Walker, start: np.ndarray, margin: str) -> KinkArc:
    forward, corner, _ = walk_zero_set(walker, start, margin, stop_at=start, step=WALK_STEP_RAD)
    if corner is None:  # back at the seed: a loop inside U_P
        points = np.asarray(forward)
        return KinkArc(points, _values(walker, points), True, (None, None))
    backward, back_corner, _ = walk_zero_set(walker, start, margin, orientation=-1.0, step=WALK_STEP_RAD)
    points = np.asarray(backward[::-1] + forward[1:])
    ends = (_stopping_gate(walker, back_corner, margin), _stopping_gate(walker, corner, margin))
    return KinkArc(points, _values(walker, points), False, ends)


def _stopping_gate(walker: Walker, corner: np.ndarray, margin: str) -> str:
    """The gate closest to zero at an arc end (the one :func:`.boundary.walk_zero_set` stopped at)."""
    m = walker.margins(corner)
    return min(walker.active, key=lambda name: abs(float(m[walker.k(name)])))


def _values(walker: Walker, points: np.ndarray) -> np.ndarray:
    return d_p_batch(points, walker.faces, walker.index, walker.slab, crystal=walker.crystal)


__all__ = ["KinkArc", "KinkCurve", "marched_kink", "weight_kinks"]
