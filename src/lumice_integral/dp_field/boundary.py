"""``dU_P``: the boundary loop of the valid domain, its corners and the restricted critical points of ``D_P``.

``U_P = {u : every margin > 0}`` (:mod:`.field`), so ``dU_P`` is made of
arcs of the zero sets of single margins meeting at corners.  The boundary is
found by *walking* it: from one boundary point, march along the zero set of
the active margin with ``U_P`` on the left (tangent ``grad m x u``), stop at
the first point where another margin turns negative (bisection, then a
two-margin tangent-plane Newton), pick the one margin through the corner
whose own zero set continues the boundary, and go on until the walk is back
at its first corner.  A closed walk is the certificate that the loop is
complete; that ``U_P`` has a single boundary loop is the disk check of
:mod:`.certificate`.  Walking subsumes the explore-stage corner enumerations
(the entry-great-circle scan of ``dp-field-boundary-corners`` finds only the
corners on that circle; TIR-TIR corners and corners of deeper reflections
are met by the walk in order).

Two curve kinds, by margin (explore ``dp-field-exit-tir-marching-generalize``):

- ``*_incidence_cosine`` of step ``k`` is ``m . d_entry(u)`` with
  ``m = R_{k-1}^T n_k`` (``R_{k-1}`` the fold of the reflections before step
  ``k``; ``M^T n_b`` for the exit) and ``d_entry`` a combination of ``-u`` and
  ``n_a``.  When ``m . n_a = 0`` the margin is linear in ``u`` and its zero set
  is the great circle ``m . u = 0``: walked in closed form (a rotation about
  ``m``).  This is checked per path at run time -- the normal condition and
  the margin on sampled circle points -- and a margin that fails either
  check is marched.  The entry margin is ``n_a . u``, always a great circle.
- every other margin (TIR and Snell discriminants, non-orthogonal
  incidence cosines) is marched: tangent predictor, Newton corrector along
  the tangent gradient back onto the zero set (residual at rounding level).

Margin identities are removed before walking, never assumed absent:

- three consecutive internal reflections off side faces whose azimuths step
  by the same ``+-60`` degrees make the first and third incidence cosines
  (hence TIR discriminants) the same function (explore
  ``dp-field-boundary-deep-internal-faces``: two reflections compose to a
  rotation that carries the third normal onto the first); the third step's
  margins are dropped and reported in ``identical_margins``;
- a margin that vanishes along a whole piece of the walk (for example
  ``exit_snell_discriminant = entry_incidence_cosine^2`` on ``3-5-6-7-3``,
  explore ``dp-field-bigon-other-corner-pair``; mechanism not needed) is
  recorded as coincident with that piece and does not stop the walk there.

Corners carry every margin that vanishes there (algebraic multiplicity can
exceed 2, ``3-5-6-7-3``), which two of them bound ``U_P`` (the incoming and
outgoing pieces of the walk) and which of the others are tangent to a
bounding piece (gradients parallel: they touch the corner without cutting
the domain, explore ``dp-field-bigon-other-corner-pair``) versus transversal.

``D_P`` restricted to the loop is sampled along each piece; its local
extrema are refined by golden-section search on the piece (derivative-free:
``grad D_P`` diverges on the exit-TIR curve, where ``D_P`` itself is finite)
and corner values are kept as they are.  On the exit TIR curve the corrector
leaves points on the ``U_P`` side (margin ``>= 0``) so the exit refraction's
square root is real; the value error there is ``~ sqrt(1e-16) = 1e-8`` rad.
Slab paths are evaluated in closed form (:func:`.field.d_value`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial import cKDTree

from .. import optics
from ..geometry import Polyhedron, fold_matrix
from ..s2_store import fibonacci_sphere
from .field import Faces, d_value, margin_vector, valid_batch

# March step along a boundary piece (rad).
WALK_STEP_RAD = np.radians(0.25)
# Distance from a corner at which the candidate outgoing pieces are probed (rad).
CORNER_PROBE_RAD = 1e-5
# A margin at or below this in size vanishes at a point (corner membership, coincidence with a piece).
ZERO_MARGIN_ATOL = 1e-9
COINCIDENT_ATOL = 1e-12
# A margin violates U_P below -VIOLATION_ATOL: rounding of a margin that only touches zero (a square such as
# exit_snell_discriminant = entry_incidence_cosine^2 on 3-5-6-7-3) is ~2e-16 and must not stop the walk.
VIOLATION_ATOL = 1e-13
# A vanishing margin whose tangent gradient is below this touches zero without changing sign: never an edge.
TOUCHING_GRADIENT_ATOL = 1e-8
# |m . n_a| below this makes an incidence cosine linear in u (great circle).
GREAT_CIRCLE_ATOL = 1e-12
# Two gradients whose unit tangent parts have |cross| below this are parallel (tangent curves).
TANGENT_SINE_ATOL = 1e-6
# The walk is closed when it meets its first corner again within this (rad).
CORNER_CLOSE_RAD = 1e-6
MAX_WALK_STEPS = 40000
# Loop values within this (rad) are one plateau: above the D_P error on the loop (~1e-8 on an exit TIR curve,
# where the exit square root sees a 1e-16 margin), below any extremum the sampling resolves.
EXTREMUM_ATOL = 1e-7


# ---- single-point JAX kernels -------------------------------------------------------------------------


@partial(jax.jit, static_argnums=1)
def _margins_and_jacobian(u: jax.Array, faces: Faces, index: jax.Array) -> tuple[jax.Array, jax.Array]:
    return margin_vector(u, faces, index), jax.jacfwd(margin_vector)(u, faces, index)


@partial(jax.jit, static_argnums=1)
def _margins(u: jax.Array, faces: Faces, index: jax.Array) -> jax.Array:
    return margin_vector(u, faces, index)


@partial(jax.jit, static_argnums=1)
def _d(u: jax.Array, faces: Faces, index: jax.Array, slab: jax.Array | None) -> jax.Array:
    return d_value(u, faces, index, slab)


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def _angle(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.arctan2(np.linalg.norm(np.cross(a, b)), a @ b))


def _tangent(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return v - (v @ u) * u


# ---- data ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class BoundaryPiece:
    """One smooth arc of ``dU_P``: the zero set of ``margin`` from ``points[0]`` to ``points[-1]``, ``U_P`` on the left.

    ``kind`` is ``"great_circle"`` (``circle_normal`` set, walked in closed
    form) or ``"marched"``; ``coincident`` lists the margins that vanish all
    along it.  ``values`` is ``D_P`` at ``points``.
    """

    margin: str
    kind: str
    circle_normal: np.ndarray | None
    points: np.ndarray
    values: np.ndarray
    coincident: tuple[str, ...]


@dataclass(frozen=True)
class Corner:
    """A non-smooth point of ``dU_P``.

    ``margins``: every margin vanishing there (``|m| <= ZERO_MARGIN_ATOL``),
    in :func:`.optics.domain_margin_names` order; ``incoming`` /
    ``outgoing``: the two that bound ``U_P`` there (walk order);
    ``tangent``: the other vanishing margins whose curve is tangent to a
    bounding one (gradients parallel: they touch the corner from outside
    without cutting ``U_P``); ``transversal``: the other vanishing margins
    crossing both bounding curves (necessarily through the exterior);
    ``coincident``: vanishing margins whose zero set contains a bounding
    piece (:class:`BoundaryPiece` ``coincident``).  ``residual`` is the
    largest ``|m|`` over ``margins``.
    """

    position: np.ndarray
    value: float
    margins: tuple[str, ...]
    incoming: str
    outgoing: str
    tangent: tuple[str, ...]
    transversal: tuple[str, ...]
    coincident: tuple[str, ...]
    residual: float


@dataclass(frozen=True)
class BoundaryCriticalPoint:
    """A local extremum of ``D_P`` restricted to ``dU_P``: inside a piece (``corner`` false) or at a corner."""

    position: np.ndarray
    value: float
    kind: str
    margin: str
    corner: bool


@dataclass(frozen=True)
class BoundaryLoop:
    """``dU_P`` as one closed walk: pieces and corners in walk order (``pieces[i]`` runs from ``corners[i - 1]`` to ``corners[i]``)."""

    pieces: tuple[BoundaryPiece, ...]
    corners: tuple[Corner, ...]
    critical_points: tuple[BoundaryCriticalPoint, ...]
    identical_margins: dict[str, str] = field(default_factory=dict)
    great_circle_margins: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def values(self) -> np.ndarray:
        """``D_P`` along the loop in walk order, the closing point not repeated."""
        return np.concatenate([piece.values[:-1] for piece in self.pieces])


# ---- margin identities and great circles --------------------------------------------------------------


def _side_azimuth(face: int) -> int | None:
    return (face - 3) * 60 if 3 <= face <= 8 else None


def identical_margins(faces: Faces) -> dict[str, str]:
    """Margins that equal an earlier one as functions: the ``+-60`` degree side-face triples (module docstring).

    Returns ``{dropped name: kept name}`` for both the incidence cosine and
    the TIR discriminant of the third reflection of each such triple.
    """
    internal = faces[1:-1]
    out: dict[str, str] = {}
    for j in range(len(internal) - 2):
        azimuths = [_side_azimuth(face) for face in internal[j : j + 3]]
        if any(a is None for a in azimuths):
            continue
        first, second = (azimuths[1] - azimuths[0]) % 360, (azimuths[2] - azimuths[1]) % 360
        if first == second and first in (60, 300):
            kept, dropped = j + 1, j + 3
            while f"internal_{kept}_incidence_cosine" in out:  # chains of triples all map to the first
                kept = int(out[f"internal_{kept}_incidence_cosine"].split("_")[1])
            for suffix in ("incidence_cosine", "tir_discriminant"):
                out[f"internal_{dropped}_{suffix}"] = f"internal_{kept}_{suffix}"
    return out


def _incidence_normals(crystal: Polyhedron, faces: Faces) -> dict[str, np.ndarray]:
    """``m`` of every incidence cosine: ``n_a`` (entry), ``R_{k-1}^T n_k`` (step ``k``), ``M^T n_b`` (exit)."""
    normals = {name: np.asarray(n) for name, n in optics.HEXPRISM_BODY_NORMALS.items()}
    out = {"entry_incidence_cosine": normals[faces[0]]}
    for k in range(1, len(faces) - 1):
        # fold_matrix of (a, m_1, ..., m_{k-1}, m_k) folds the reflections m_1..m_{k-1}: R_{k-1}.
        out[f"internal_{k}_incidence_cosine"] = fold_matrix(crystal, faces[: k + 1]).T @ normals[faces[k]]
    out["exit_incidence_cosine"] = fold_matrix(crystal, faces).T @ normals[faces[-1]]
    return out


def great_circle_margins(crystal: Polyhedron, faces: Faces, index: float, *, samples: int = 64) -> dict[str, np.ndarray]:
    """``{margin: unit normal}`` of the incidence cosines whose zero set is a great circle, oriented into ``U_P``.

    Accepted when ``|m . n_a| <= GREAT_CIRCLE_ATOL`` *and* the margin is
    ``<= COINCIDENT_ATOL`` in size at ``samples`` points of ``m . u = 0``
    (run-time check of the derivation in the module docstring).
    """
    names = optics.domain_margin_names(faces)
    n_a = np.asarray(optics.HEXPRISM_BODY_NORMALS[faces[0]])
    t = np.linspace(0.0, 2.0 * np.pi, samples, endpoint=False)
    out: dict[str, np.ndarray] = {}
    for name, m in _incidence_normals(crystal, faces).items():
        if abs(m @ n_a) > GREAT_CIRCLE_ATOL and name != "entry_incidence_cosine":
            continue
        m = _unit(m)
        e1 = _unit(np.cross(m, np.eye(3)[int(np.argmin(np.abs(m)))]))
        e2 = np.cross(m, e1)
        k = names.index(name)
        on_circle = [float(_margins(jnp.asarray(np.cos(s) * e1 + np.sin(s) * e2), faces, jnp.float64(index))[k]) for s in t]
        if max(abs(v) for v in on_circle) > COINCIDENT_ATOL:
            continue
        sign = np.sign(float(_margins(jnp.asarray(m), faces, jnp.float64(index))[k]))
        out[name] = sign * m
    return out


# ---- walker -------------------------------------------------------------------------------------------


class _Walker:
    """Stateful helper holding the path, the margin bookkeeping and the curve steppers."""

    def __init__(self, crystal: Polyhedron, faces: Faces, index: float, slab: np.ndarray | None) -> None:
        self.faces = faces
        self.index = float(index)
        self._index = jnp.float64(index)
        self._slab = None if slab is None else jnp.asarray(slab, dtype=jnp.float64)
        self.names = optics.domain_margin_names(faces)
        self.identical = identical_margins(faces)
        self.circles = great_circle_margins(crystal, faces, index)
        # entry_snell_discriminant = 1 - (1 - c^2) / n^2 > 0 for n > 1: never a boundary; it stays in the
        # validity test (it is a margin of the authority) but is not walked.
        self.active = [name for name in self.names if name not in self.identical]

    # -- evaluation
    def margins(self, u: np.ndarray) -> np.ndarray:
        return np.asarray(_margins(jnp.asarray(u), self.faces, self._index))

    def margins_jacobian(self, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        m, j = _margins_and_jacobian(jnp.asarray(u), self.faces, self._index)
        return np.asarray(m), np.asarray(j)

    def d(self, u: np.ndarray) -> float:
        """``D_P`` at a point of the closure of ``U_P`` (``RuntimeError`` if not finite: the point is outside)."""
        value = float(_d(jnp.asarray(u), self.faces, self._index, self._slab))
        if not np.isfinite(value):
            raise RuntimeError(f"D_P is not finite at {u} on {optics.path_id_of(self.faces)}")
        return value

    def k(self, name: str) -> int:
        return self.names.index(name)

    def tangent_gradient(self, u: np.ndarray, name: str) -> np.ndarray:
        _, j = self.margins_jacobian(u)
        return _tangent(u, j[self.k(name)])

    # -- curve steppers
    def correct(self, u: np.ndarray, name: str) -> np.ndarray:
        """Project ``u`` onto the zero set of ``name``, ending on its non-negative side."""
        if name in self.circles:
            normal = self.circles[name]
            u = _unit(u - (u @ normal) * normal)
            return _unit(u + 1e-16 * normal) if self.margins(u)[self.k(name)] < 0.0 else u
        k = self.k(name)
        for _ in range(30):
            m, j = self.margins_jacobian(u)
            g = _tangent(u, j[k])
            if abs(m[k]) <= 1e-15:
                break
            u = _unit(u - (m[k] / (g @ g)) * g)
        for _ in range(8):
            m, j = self.margins_jacobian(u)
            if m[k] >= 0.0:
                break
            g = _tangent(u, j[k])
            u = _unit(u + ((-m[k]) / (g @ g) + 1e-16 / np.linalg.norm(g)) * g)
        return u

    def direction(self, u: np.ndarray, name: str) -> np.ndarray:
        """Unit tangent of the zero set of ``name`` at ``u`` with ``U_P`` (margin ``> 0``) on the left."""
        if name in self.circles:
            return _unit(np.cross(self.circles[name], u))
        return _unit(np.cross(self.tangent_gradient(u, name), u))

    def advance(self, u: np.ndarray, name: str, tangent: np.ndarray, step: float) -> np.ndarray:
        if name in self.circles:
            return _unit(np.cos(step) * u + np.sin(step) * _unit(np.cross(self.circles[name], u)))
        return self.correct(_unit(u + step * tangent), name)

    # -- corners
    def refine_corner(self, u: np.ndarray, a: str, b: str) -> np.ndarray:
        """Two-margin tangent-plane Newton on ``m_a = m_b = 0`` (left as is if the curves are tangent there)."""
        ka, kb = self.k(a), self.k(b)
        best = u
        best_residual = self._residual(u, (a, b))
        for _ in range(30):
            m, j = self.margins_jacobian(u)
            axis = np.eye(3)[int(np.argmin(np.abs(u)))]
            e1 = _unit(np.cross(u, axis))
            e2 = np.cross(u, e1)
            jac = np.array([[j[ka] @ e1, j[ka] @ e2], [j[kb] @ e1, j[kb] @ e2]])
            if abs(np.linalg.det(jac)) < 1e-14:
                break
            step = np.linalg.solve(jac, -np.array([m[ka], m[kb]]))
            if np.linalg.norm(step) > 1e-3:
                break
            u = _unit(u + step[0] * e1 + step[1] * e2)
            residual = self._residual(u, (a, b))
            if residual < best_residual:
                best, best_residual = u, residual
            if residual <= 1e-16:
                break
        return best

    def _residual(self, u: np.ndarray, names: tuple[str, ...]) -> float:
        m = self.margins(u)
        return max(abs(float(m[self.k(n)])) for n in names)

    def violated(self, u: np.ndarray, excluded: set[str]) -> list[str]:
        m = self.margins(u)
        return [n for n in self.active if n not in excluded and not m[self.k(n)] >= -VIOLATION_ATOL]

    def most_violated(self, u: np.ndarray, names: list[str]) -> str:
        """The margin of ``names`` farthest outside at ``u`` in signed distance ``m / |grad m|``."""
        m, j = self.margins_jacobian(u)
        return min(names, key=lambda n: m[self.k(n)] / np.linalg.norm(_tangent(u, j[self.k(n)])))

    def coincident_with(self, u: np.ndarray, name: str) -> set[str]:
        m = self.margins(u)
        return {n for n in self.active if n != name and abs(m[self.k(n)]) <= COINCIDENT_ATOL}


def _start_point(walker: _Walker, lattice_n: int) -> tuple[np.ndarray, str]:
    """A point of ``dU_P`` and its margin: bisect between a lattice point of ``U_P`` and an outside neighbour."""
    lattice = fibonacci_sphere(lattice_n)
    valid = valid_batch(lattice, walker.faces, walker.index)
    if not valid.any():
        raise ValueError(f"U_P of {optics.path_id_of(walker.faces)} has no point on a {lattice_n}-point lattice")
    if valid.all():
        raise ValueError("U_P covers the whole lattice: no boundary")
    tree = cKDTree(lattice)
    _, neighbours = tree.query(lattice[valid], k=7)
    inside_points = lattice[valid]
    for row, candidates in enumerate(neighbours):
        outside = [c for c in candidates[1:] if not valid[c]]
        if outside:
            inside, out = inside_points[row], lattice[outside[0]]
            break
    for _ in range(200):
        mid = _unit(inside + out)
        if not walker.violated(mid, set()):
            inside = mid
        else:
            out = mid
        if _angle(inside, out) < 1e-12:
            break
    name = walker.most_violated(out, walker.violated(out, set()))
    return walker.correct(inside, name), name


def _walk_piece(
    walker: _Walker, start: np.ndarray, name: str, *, stop_at: np.ndarray | None, step: float
) -> tuple[list[np.ndarray], np.ndarray | None, set[str]]:
    """March along ``name`` from ``start`` (a point inside the piece) to the next corner.

    Returns the points (``start`` first, the corner last when one is met),
    the corner (``None`` if ``stop_at`` -- a point of this piece -- is passed
    first; it is then the last point) and the margins coincident with the
    piece.
    """
    probe = walker.advance(start, name, walker.direction(start, name), min(step, 1e-3))
    coincident = walker.coincident_with(start, name) & walker.coincident_with(probe, name)
    excluded = coincident | {name}
    points = [start]
    u = start
    for _ in range(MAX_WALK_STEPS):
        tangent = walker.direction(u, name)
        if stop_at is not None and len(points) > 1 and _angle(u, stop_at) <= 2.0 * step:
            to_stop = stop_at - u
            if to_stop @ tangent > 0.0 and _angle(u, stop_at) <= step:
                points.append(stop_at)
                return points, None, coincident
        nxt = walker.advance(u, name, tangent, step)
        bad = walker.violated(nxt, excluded)
        if not bad:
            points.append(nxt)
            u = nxt
            continue
        lo, hi = 0.0, step
        hi_point = nxt
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            p = walker.advance(u, name, tangent, mid)
            if walker.violated(p, excluded):
                hi, hi_point = mid, p
            else:
                lo = mid
            if hi - lo < 1e-15:
                break
        crossing = walker.most_violated(hi_point, walker.violated(hi_point, excluded))
        corner = walker.refine_corner(walker.advance(u, name, tangent, lo), name, crossing)
        points.append(corner)
        return points, corner, coincident
    raise RuntimeError(f"boundary walk of {optics.path_id_of(walker.faces)} did not reach a corner in {MAX_WALK_STEPS} steps")


def _outgoing(walker: _Walker, corner: np.ndarray, incoming: str, incoming_coincident: set[str]) -> str:
    """The one vanishing margin at ``corner`` whose zero set continues ``dU_P`` (``U_P`` on its left)."""
    m, j = walker.margins_jacobian(corner)
    zero = [n for n in walker.active if abs(m[walker.k(n)]) <= ZERO_MARGIN_ATOL]
    accepted = []
    for name in zero:
        if name == incoming or name in incoming_coincident:
            continue
        if np.linalg.norm(_tangent(corner, j[walker.k(name)])) < TOUCHING_GRADIENT_ATOL:
            continue
        probe = walker.advance(corner, name, walker.direction(corner, name), CORNER_PROBE_RAD)
        excluded = {name} | walker.coincident_with(probe, name)
        if not walker.violated(probe, excluded):
            accepted.append(name)
    if len(accepted) != 1:
        raise RuntimeError(
            f"corner {corner} of {optics.path_id_of(walker.faces)} continues along {accepted} "
            f"(vanishing margins {zero}): not a simple boundary loop"
        )
    return accepted[0]


def _corner_record(walker: _Walker, u: np.ndarray, incoming: str, outgoing: str, coincident: set[str]) -> Corner:
    m, j = walker.margins_jacobian(u)
    zero = tuple(n for n in walker.active if abs(m[walker.k(n)]) <= ZERO_MARGIN_ATOL or n in (incoming, outgoing))
    edges = [_unit(_tangent(u, j[walker.k(n)])) for n in (incoming, outgoing)]
    tangent, transversal = [], []
    for name in zero:
        if name in (incoming, outgoing) or name in coincident:
            continue
        g = _unit(_tangent(u, j[walker.k(name)]))
        (tangent if min(np.linalg.norm(np.cross(g, e)) for e in edges) < TANGENT_SINE_ATOL else transversal).append(name)
    residual = max(abs(float(m[walker.k(n)])) for n in zero)
    coincident_here = tuple(n for n in zero if n in coincident)
    return Corner(u, walker.d(u), zero, incoming, outgoing, tuple(tangent), tuple(transversal), coincident_here, residual)


def _golden_extremum(walker: _Walker, piece: BoundaryPiece, i: int, kind: str) -> tuple[np.ndarray, float]:
    """Refine the sample extremum ``piece.points[i]`` on the piece between its neighbours (golden section)."""
    # i == 0 only on a loop without corners, whose single piece ends where it starts
    a, b = (piece.points[-2] if i == 0 else piece.points[i - 1]), piece.points[i + 1]
    sign = 1.0 if kind == "minimum" else -1.0

    def point(lam: float) -> np.ndarray:
        return walker.correct(_unit((1.0 - lam) * a + lam * b), piece.margin)

    def f(lam: float) -> float:
        return sign * walker.d(point(lam))

    ratio = (np.sqrt(5.0) - 1.0) / 2.0
    lo, hi = 0.0, 1.0
    x1, x2 = hi - ratio * (hi - lo), lo + ratio * (hi - lo)
    f1, f2 = f(x1), f(x2)
    for _ in range(80):
        if f1 <= f2:
            hi, x2, f2 = x2, x1, f1
            x1 = hi - ratio * (hi - lo)
            f1 = f(x1)
        else:
            lo, x1, f1 = x1, x2, f2
            x2 = lo + ratio * (hi - lo)
            f2 = f(x2)
        if hi - lo < 1e-12:
            break
    best = point(0.5 * (lo + hi))
    return best, walker.d(best)


def _plateau_extrema(values: np.ndarray, atol: float = EXTREMUM_ATOL) -> list[tuple[int, str]]:
    """Indices of local extrema of a cyclic sequence, runs of equal values (plateaus) counted once."""
    n = len(values)
    # compress runs of equal values
    runs: list[tuple[int, float]] = []
    for i in range(n):
        if runs and abs(values[i] - runs[-1][1]) <= atol:
            continue
        runs.append((i, float(values[i])))
    if len(runs) > 1 and abs(runs[0][1] - runs[-1][1]) <= atol:
        runs.pop()
    out = []
    r = len(runs)
    if r < 3:
        return [(runs[0][0], "minimum")] if r == 1 else [(runs[0][0], "minimum" if runs[0][1] < runs[1][1] else "maximum"), (runs[1][0], "maximum" if runs[0][1] < runs[1][1] else "minimum")]
    for j in range(r):
        prev_v, v, next_v = runs[j - 1][1], runs[j][1], runs[(j + 1) % r][1]
        if v < prev_v and v < next_v:
            out.append((runs[j][0], "minimum"))
        elif v > prev_v and v > next_v:
            out.append((runs[j][0], "maximum"))
    return out


def walk_boundary(
    crystal: Polyhedron,
    faces: Faces,
    index: float,
    *,
    slab: np.ndarray | None = None,
    lattice_n: int = 20000,
    step: float = WALK_STEP_RAD,
) -> BoundaryLoop:
    """Walk ``dU_P`` once around (module docstring) and collect pieces, corners and restricted extrema of ``D_P``.

    ``slab`` is the fold matrix of a degenerate-fold path (:func:`.field.d_value`).
    """
    walker = _Walker(crystal, faces, index, slab)
    start, name = _start_point(walker, lattice_n)
    # 1. find the first corner (or come back to the start: a smooth loop)
    points, corner, coincident = _walk_piece(walker, start, name, stop_at=start, step=step)
    pieces: list[BoundaryPiece] = []
    corners: list[Corner] = []
    if corner is None:
        pieces.append(_piece(walker, name, points, coincident))
    else:
        first = corner
        incoming, incoming_coincident = name, coincident
        u = corner
        for _ in range(1000):
            outgoing = _outgoing(walker, u, incoming, incoming_coincident)
            begin = walker.advance(u, outgoing, walker.direction(u, outgoing), CORNER_PROBE_RAD)
            points, nxt, coincident = _walk_piece(walker, begin, outgoing, stop_at=None, step=step)
            pieces.append(_piece(walker, outgoing, [u, *points], coincident))
            incoming, incoming_coincident, u = outgoing, coincident, nxt
            if _angle(nxt, first) <= CORNER_CLOSE_RAD:
                break
        else:
            raise RuntimeError(f"boundary walk of {optics.path_id_of(faces)} did not close")
        # the loop starts and ends at `first`: pieces[i] runs from corner i - 1 to corner i
        pieces[-1] = _replace_last_point(walker, pieces[-1], first)
        for i, piece in enumerate(pieces):
            after = pieces[(i + 1) % len(pieces)]
            corners.append(
                _corner_record(walker, piece.points[-1], piece.margin, after.margin, set(piece.coincident) | set(after.coincident))
            )
    critical = _loop_critical_points(walker, pieces, corners)
    return BoundaryLoop(tuple(pieces), tuple(corners), tuple(critical), dict(walker.identical), dict(walker.circles))


def _piece(walker: _Walker, name: str, points: list[np.ndarray], coincident: set[str]) -> BoundaryPiece:
    array = np.stack(points)
    values = np.array([walker.d(p) for p in array])
    kind = "great_circle" if name in walker.circles else "marched"
    return BoundaryPiece(name, kind, walker.circles.get(name), array, values, tuple(n for n in walker.active if n in coincident))


def _replace_last_point(walker: _Walker, piece: BoundaryPiece, point: np.ndarray) -> BoundaryPiece:
    points = piece.points.copy()
    points[-1] = point
    values = piece.values.copy()
    values[-1] = walker.d(point)
    return BoundaryPiece(piece.margin, piece.kind, piece.circle_normal, points, values, piece.coincident)


def _loop_critical_points(walker: _Walker, pieces: list[BoundaryPiece], corners: list[Corner]) -> list[BoundaryCriticalPoint]:
    """Local extrema of ``D_P`` along the loop: corners as they are, piece samples refined by golden section."""
    # loop samples: each piece without its last point (the next piece starts there); remember owners
    owners: list[tuple[int, int]] = []
    values: list[float] = []
    for p_index, piece in enumerate(pieces):
        for i in range(len(piece.points) - 1):
            owners.append((p_index, i))
            values.append(float(piece.values[i]))
    out = []
    for flat, kind in _plateau_extrema(np.array(values)):
        p_index, i = owners[flat]
        piece = pieces[p_index]
        if i == 0 and corners:
            corner = corners[p_index - 1]
            out.append(BoundaryCriticalPoint(corner.position, corner.value, kind, piece.margin, True))
            continue
        position, value = _golden_extremum(walker, piece, i, kind)
        out.append(BoundaryCriticalPoint(position, value, kind, piece.margin, False))
    return out
