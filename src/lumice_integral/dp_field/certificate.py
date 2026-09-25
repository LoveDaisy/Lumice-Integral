"""The ``delta``-axis partition of ``D_P`` on ``U_P`` and its component counts (the completeness certificate's data).

The level set ``{D_P = delta}`` in ``U_P`` is a union of closed loops and of
open arcs whose two ends lie on ``dU_P``.  Its topology can only change at a
critical value: an interior critical value, a critical value of ``D_P``
restricted to a smooth piece of ``dU_P``, or the value at a corner.  Between
consecutive critical values the counts are constant and follow from the
critical data alone:

- **open arcs.**  Every arc end is a point where ``D_P`` restricted to the
  boundary loop crosses ``delta``, and every crossing is the end of one arc,
  so ``n_open`` is half the number of crossings.  ``D_P`` along the loop is
  monotone between consecutive loop extrema (:class:`.boundary.BoundaryLoop`
  ``critical_points``, in walk order), so a monotone stretch contributes one
  crossing iff ``delta`` lies strictly inside its range.  This holds for any
  topology of ``U_P`` with the given boundary loops.
- **closed loops.**  A closed level loop bounds a disk in ``U_P`` (``U_P`` a
  disk) and that disk contains an interior extremum.  With no interior
  critical point there are none.  With exactly one interior extremum (a
  non-degenerate minimum, or a strict extremum of a slab path at ``+-n_M``,
  for example ``D = pi`` inside ``3-5-6-7-3``) at value ``v``, say a
  minimum: for ``delta`` below the loop minimum ``L`` the sublevel set is
  interior, every component of it holds an interior minimum, so it is one
  disk around ``v`` and there is exactly one closed loop; once the component
  of ``v`` reaches ``dU_P`` there is none (a second loop around ``v`` would
  enclose that component).  It reaches ``dU_P`` at ``L`` when ``D_P``
  decreases into ``U_P`` at a boundary point of value ``L`` (the interior
  points below ``L`` next to it belong to a component whose minimum is
  interior, i.e. ``v``'s); this is checked on a small ring around every
  loop minimum at ``L``.  So ``n_closed = 1`` on ``(v, L)``, else ``0``;
  a maximum is the mirror image.

Everything outside that reasoning is the explicit escape hatch
(:class:`TopologyEscape`, issue ``dp-field-layer``, explore
``dp-field-saddle-search``): ``U_P`` or its complement not connected on the
lattice (not a disk), more than one interior critical point, a saddle or a
degenerate Morse point, a slab crease inside ``U_P``, or a loop extremum at
``L`` with ``D_P`` increasing into ``U_P`` (a sublevel component born on the
boundary).  None of them is resolved silently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from ..s2_store import fibonacci_sphere
from .boundary import EXTREMUM_ATOL, BoundaryLoop
from .field import DegenerateFoldSet, Faces, InteriorCriticalPoint, d_p_batch, tangent_basis, valid_batch, validity_margins_batch

# Radius of the ring that decides on which side of a boundary extremum D_P is lower (rad), and its directions.
SIDE_RING_RAD = 1e-5
SIDE_RING_DIRECTIONS = 64


class TopologyEscape(RuntimeError):
    """The simple disk / single-extremum reasoning of the module docstring does not apply to this path."""


class DeviationInterval(NamedTuple):
    """``(lower, upper)`` in radians with the constant counts of ``{D_P = delta}`` for ``lower < delta < upper``."""

    lower: float
    upper: float
    n_components: int
    n_closed: int
    n_open: int


@dataclass(frozen=True)
class DomainTopology:
    """Connected components of ``U_P`` and of its complement on a Fibonacci lattice (k-NN graph)."""

    lattice_n: int
    domain_components: int
    complement_components: int

    @property
    def is_disk(self) -> bool:
        return self.domain_components == 1 and self.complement_components == 1


def _component_count(points: np.ndarray) -> int:
    """k-NN (k = 8) connected components, edges longer than 3x the median dropped (explore ``dp-field-saddle-search``)."""
    n = len(points)
    if n < 9:
        return int(n > 0)
    distances, neighbours = cKDTree(points).query(points, k=9)
    rows = np.repeat(np.arange(n), 8)
    cols = neighbours[:, 1:].ravel()
    lengths = distances[:, 1:].ravel()
    keep = lengths <= 3.0 * np.median(lengths)
    graph = coo_matrix((np.ones(int(keep.sum())), (rows[keep], cols[keep])), shape=(n, n))
    return int(connected_components(graph, directed=False)[0])


def domain_topology(faces: Faces, index: float, *, lattice_n: int = 20000) -> DomainTopology:
    """Component counts of ``U_P`` and ``S^2 \\ U_P`` on the ``lattice_n``-point Fibonacci lattice."""
    lattice = fibonacci_sphere(lattice_n)
    valid = valid_batch(lattice, faces, index)
    return DomainTopology(lattice_n, _component_count(lattice[valid]), _component_count(lattice[~valid]))


def _side_values(point: np.ndarray, faces: Faces, index: float, slab: np.ndarray | None) -> np.ndarray:
    """``D_P`` on the part of a ``SIDE_RING_RAD`` ring around ``point`` inside ``U_P``."""
    e = np.asarray(tangent_basis(point))
    angles = np.linspace(0.0, 2.0 * np.pi, SIDE_RING_DIRECTIONS, endpoint=False)
    ring = np.cos(SIDE_RING_RAD) * point + np.sin(SIDE_RING_RAD) * (np.cos(angles)[:, None] * e[0] + np.sin(angles)[:, None] * e[1])
    inside = np.all(validity_margins_batch(ring, faces, index) > 0.0, axis=1)
    return d_p_batch(ring[inside], faces, index, slab)


def _interior_extremum(point: InteriorCriticalPoint, faces: Faces, index: float, slab: np.ndarray | None) -> str:
    """``"minimum"`` / ``"maximum"`` of a critical point (a slab point by its ring), else ``TopologyEscape``."""
    if point.kind in ("minimum", "maximum"):
        return point.kind
    if point.kind == "degenerate" and slab is not None:
        ring = _side_values(point.position, faces, index, slab)
        if len(ring) == SIDE_RING_DIRECTIONS and np.all(ring > point.value):
            return "minimum"
        if len(ring) == SIDE_RING_DIRECTIONS and np.all(ring < point.value):
            return "maximum"
    raise TopologyEscape(f"interior critical point of kind {point.kind!r} at {point.position}, D = {point.value}")


def _loop_crossings(extrema_values: np.ndarray, delta: float) -> int:
    k = len(extrema_values)
    return sum(
        min(extrema_values[i], extrema_values[(i + 1) % k]) < delta < max(extrema_values[i], extrema_values[(i + 1) % k])
        for i in range(k)
    )


def critical_values(interior: tuple[InteriorCriticalPoint, ...], loop: BoundaryLoop) -> np.ndarray:
    """Sorted critical values (rad): interior, restricted to pieces, and corners; merged within ``EXTREMUM_ATOL``."""
    raw = sorted([p.value for p in interior] + [c.value for c in loop.critical_points] + [c.value for c in loop.corners])
    merged: list[float] = []
    for value in raw:
        if not merged or value - merged[-1] > EXTREMUM_ATOL:
            merged.append(value)
    return np.array(merged)


def interval_partition(
    faces: Faces,
    index: float,
    interior: tuple[InteriorCriticalPoint, ...],
    fold_set: DegenerateFoldSet | None,
    loop: BoundaryLoop,
    topology: DomainTopology,
    slab: np.ndarray | None,
) -> tuple[DeviationInterval, ...]:
    """The partition of ``[min D_P, max D_P]`` with the counts of every interval (module docstring)."""
    if not topology.is_disk:
        raise TopologyEscape(
            f"U_P is not a disk on the {topology.lattice_n}-point lattice: {topology.domain_components} component(s), "
            f"complement {topology.complement_components}"
        )
    if fold_set is not None and fold_set.circle_interior_fraction > 0.0:
        raise TopologyEscape(f"the slab crease u . n_M = 0 runs through U_P ({fold_set.circle_interior_fraction:.3%} of it)")
    if len(interior) > 1:
        raise TopologyEscape(f"{len(interior)} interior critical points: " + ", ".join(p.kind for p in interior))
    extrema = loop.critical_points
    kinds = [p.kind for p in extrema]
    if any(kinds[i] == kinds[(i + 1) % len(kinds)] for i in range(len(kinds))):
        raise TopologyEscape(f"loop extrema do not alternate: {kinds}")
    values = np.array([p.value for p in extrema])
    closed_range: tuple[float, float] | None = None
    if interior:
        kind = _interior_extremum(interior[0], faces, index, slab)
        v = interior[0].value
        edge = values.min() if kind == "minimum" else values.max()
        touching = [p for p in extrema if p.kind == kind and abs(p.value - edge) <= EXTREMUM_ATOL]
        sign = 1.0 if kind == "minimum" else -1.0
        reaches = any(np.any(sign * (_side_values(p.position, faces, index, slab) - edge) < -1e-12) for p in touching)
        if not reaches or sign * (edge - v) <= 0.0:
            raise TopologyEscape(
                f"interior {kind} D = {v} and loop {kind} {edge}: the sublevel component of the interior extremum "
                "is not shown to reach dU_P first at the loop extremum"
            )
        closed_range = (min(v, edge), max(v, edge))
    breaks = critical_values(interior, loop)
    out = []
    for lower, upper in zip(breaks[:-1], breaks[1:]):
        delta = 0.5 * (lower + upper)
        crossings = _loop_crossings(values, delta)
        if crossings % 2:
            raise TopologyEscape(f"odd number of boundary crossings ({crossings}) at delta = {delta}")
        n_closed = int(closed_range is not None and closed_range[0] < delta < closed_range[1])
        n_open = crossings // 2
        out.append(DeviationInterval(float(lower), float(upper), n_closed + n_open, n_closed, n_open))
    return tuple(out)


# ---- the critical set and its transport by a crystal symmetry -----------------------------------------


@dataclass(frozen=True)
class CriticalSet:
    """Positions and values of the interior critical points, the loop extrema and the corners."""

    interior: tuple[tuple[np.ndarray, float], ...]
    boundary: tuple[tuple[np.ndarray, float], ...]
    corners: tuple[tuple[np.ndarray, float], ...]

    @classmethod
    def of(cls, interior: tuple[InteriorCriticalPoint, ...], loop: BoundaryLoop) -> "CriticalSet":
        return cls(
            tuple((p.position, p.value) for p in interior),
            tuple((p.position, p.value) for p in loop.critical_points if not p.corner),
            tuple((c.position, c.value) for c in loop.corners),
        )

    def transported(self, g: np.ndarray) -> "CriticalSet":
        """The set moved by a ``D6h`` element: ``u -> g u``, values unchanged (``D_{gPg^-1}(g u) = D_P(u)``)."""
        g = np.asarray(g, dtype=np.float64)

        def move(items):
            return tuple((g @ position, value) for position, value in items)

        return CriticalSet(move(self.interior), move(self.boundary), move(self.corners))

    def mismatch(self, other: "CriticalSet") -> tuple[float, float]:
        """Largest ``(position angle, value difference)`` pairing each point of either set with its nearest match."""
        worst_angle = worst_value = 0.0
        for mine, theirs in ((self.interior, other.interior), (self.boundary, other.boundary), (self.corners, other.corners)):
            for a, b in ((mine, theirs), (theirs, mine)):
                if not a:
                    continue
                if not b:
                    return float("inf"), float("inf")
                for position, value in a:
                    angles = [np.arctan2(np.linalg.norm(np.cross(position, q)), position @ q) for q, _ in b]
                    j = int(np.argmin(angles))
                    worst_angle = max(worst_angle, float(angles[j]))
                    worst_value = max(worst_value, abs(value - b[j][1]))
        return worst_angle, worst_value
