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
  a maximum is the mirror image.  A loop of constant ``D_P`` (a slab's
  crease circle as ``dU_P`` itself, :attr:`.boundary.BoundaryLoop.plateau_value`)
  has every point both its minimum and its maximum at that value: ``L`` is
  the constant, the ring check is taken at any one point of the loop, and
  ``D_P`` restricted to the boundary never crosses any ``delta`` off it.

Everything outside that reasoning is the explicit escape hatch
(:class:`TopologyEscape`, issue ``dp-field-layer``, explore
``dp-field-saddle-search``): ``U_P`` or its complement not connected on the
lattice (not a disk; a plural count is audited first, below), more than one
interior critical point, a saddle or a degenerate Morse point, a slab crease
inside ``U_P``, or a loop extremum at ``L`` with ``D_P`` increasing into
``U_P`` (a sublevel component born on the boundary).  None of them is
resolved silently.

The lattice count is resolution-limited, and one regime of that is audited
(task ``dp-thin-neck-topology``, explore ``u-space-dissolution-probe`` #4):
a ``U_P`` with a neck thinner than the lattice spacing (``3-5-6-7``: neck
< 1e-3 rad against 0.016 rad at ``N = 20000``) is split into two k-NN
components by an artefact, non-monotonically in ``N``.  A plural count is
therefore audited (:func:`chart_audit`) on a ladder of orthographic charts
of the entry hemisphere -- a second chain sharing no failure mode with the
k-NN graph: different sampling geometry, 4-connectivity on the chart
instead of a 3D k-NN graph, finer resolution.  Counts every grid agrees on
are the adjudicated counts, evidence rather than proof (two independent
chains agreeing and converging, still short of an arrangement-exact
certificate); disagreement establishes nothing and escapes as
``unconverged``, the fail-closed direction throughout.  This audit and
``scripts/verify_dp_field_intervals.py`` are deliberately two
implementations of the same chart (shared gate authority, unshared code):
their independence is what makes the audit worth anything, it is guarded by
the fixture cross-checks (``test_partition_agrees_with_the_independent_grid``),
and neither side's chart construction (axis choice, rim push,
4-connectivity) may be edited without the other.  Known limitation, left
uninstrumented (no trigger exists without unconditional cost): the mirror
blind spot -- a thin *invalid* gap wider than both chains' resolutions can
merge a truly plural domain into one on both chains and pass a false disk.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from .. import optics
from ..geometry import Polyhedron
from ..s2_store import fibonacci_sphere
from .boundary import EXTREMUM_ATOL, BoundaryLoop
from .field import DegenerateFoldSet, Faces, InteriorCriticalPoint, d_p_batch, tangent_basis, valid_batch, validity_margins_batch

# Radius of the ring that decides on which side of a boundary extremum D_P is lower (rad), and its directions.
SIDE_RING_RAD = 1e-5
SIDE_RING_DIRECTIONS = 64

# The chart resolutions of the component-count audit (task ``dp-thin-neck-topology``), coarse to fine,
# and the chart evaluation chunk.  An empty ladder turns the audit off (the rollback switch).
# (801, 1601) settled 2026-10-07 on 3-5-6-7: both levels adjudicate one component, the full audited
# chain costs ~2.3 s, and 3201 is not needed.
AUDIT_LADDER: tuple[int, ...] = (801, 1601)
CHART_CHUNK = 200_000


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
    """Component counts of ``U_P`` and of its complement, adjudicated (module docstring).

    The counts come from the ``lattice_n``-point Fibonacci lattice's k-NN graph; when a count is
    plural the chart-grid audit (:func:`chart_audit`) runs and, if it converges, its counts
    replace the lattice's here (``grid_audit`` keeps both).  Without an audit the fields are the
    lattice counts, unchanged.
    """

    lattice_n: int
    domain_components: int
    complement_components: int
    grid_audit: ChartAudit | None = None

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


# ---- the chart-grid audit of the lattice component counts (task ``dp-thin-neck-topology``) ------------


@dataclass(frozen=True)
class ChartAudit:
    """The chart-grid audit of the lattice component counts: its evidence, per grid (module docstring).

    ``grids`` are the audit chart resolutions; ``domain_counts`` / ``complement_counts`` hold one
    count per grid (``-1``: not counted, that grid's mask breached the chart premise).  The
    ``lattice_*`` fields keep the audited k-NN counts next to the verdict, so a correction stays
    visible instead of silent.  ``verdict`` rolls the two counts' statuses up worst-first:
    ``"unconverged"`` (the grids disagree, or a mask breached the premise) over ``"corrected"``
    (the grids agree on a count the lattice got wrong) over ``"confirmed"`` (the grids agree with
    the lattice).  It is an audit trail only: what escapes is decided by the audited counts
    through :attr:`DomainTopology.is_disk`.
    """

    grids: tuple[int, ...]
    domain_counts: tuple[int, ...]
    complement_counts: tuple[int, ...]
    lattice_domain_count: int
    lattice_complement_count: int
    verdict: str


def _chart_grid(grid: int, n_a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The ``grid x grid`` orthographic chart of the entry hemisphere: its nodes and the off-chart mask.

    On the chart ``u = x e1 + y e2 + sqrt(1 - x^2 - y^2) n_a``; off-chart nodes (``r >= 1``) are
    pushed radially onto the rim ``u . n_a = 0``.  This is the chart construction of
    ``scripts/verify_dp_field_intervals.py``, deliberately re-implemented here: the production
    audit and the independent verifier must not share code (module docstring).
    """
    e1 = np.cross(n_a, np.eye(3)[int(np.argmin(np.abs(n_a)))])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n_a, e1)
    xs = np.linspace(-1.0, 1.0, grid)
    x, y = np.meshgrid(xs, xs, indexing="ij")
    r = np.sqrt(x**2 + y**2)
    off_chart = r >= 1.0
    scale = np.where(~off_chart, 1.0, 1.0 / np.maximum(r, 1e-300))
    height = np.sqrt(np.clip(1.0 - r**2, 0.0, None))
    u = (x * scale)[..., None] * e1 + (y * scale)[..., None] * e2 + height[..., None] * n_a
    u /= np.linalg.norm(u, axis=-1, keepdims=True)
    return u, off_chart


def _chart_valid(u: np.ndarray, faces: Faces, index: float, crystal: Polyhedron | None) -> np.ndarray:
    """The ``U_P`` mask on the chart nodes, chunked :func:`valid_batch` (the gates' single authority)."""
    flat = u.reshape(-1, 3)
    valid = np.zeros(len(flat), dtype=bool)
    for start in range(0, len(flat), CHART_CHUNK):
        valid[start : start + CHART_CHUNK] = valid_batch(flat[start : start + CHART_CHUNK], faces, index, crystal=crystal)
    return valid.reshape(u.shape[:-1])


def _chart_component_counts(valid: np.ndarray, off_chart: np.ndarray) -> tuple[int, int] | None:
    """4-connected ``(domain, complement)`` counts of a chart mask, or ``None`` on a rim breach.

    The domain count labels the valid mask.  The complement count labels the invalid mask with
    every component holding an off-chart node merged into one: ``U_P`` lies in the open
    hemisphere ``u . n_a > 0`` (the entry incidence gate), so the pushed rim ring is invalid and
    connected, and an invalid region of the chart that reaches it joins the one far-hemisphere
    component of ``S^2 \\ U_P``; an invalid component with no off-chart node is an island of the
    complement inside the chart, counted separately.  A valid node on the pushed rim breaches
    that premise (valid already requires ``u . n_a > 0``) and voids the counts -- the rim would
    merge what it touches -- rather than counting through it (fail closed).
    """
    if valid[off_chart].any():
        return None
    four = ndimage.generate_binary_structure(2, 1)
    _, domain = ndimage.label(valid, structure=four)
    labels, count = ndimage.label(~valid, structure=four)
    rim_labels = set(np.unique(labels[off_chart]).tolist()) - {0}
    complement = 1 + sum(1 for label in range(1, count + 1) if label not in rim_labels)
    return int(domain), int(complement)


def _audit_verdict(
    domain_counts: tuple[int, ...], complement_counts: tuple[int, ...], lattice_domain: int, lattice_complement: int
) -> str:
    """The roll-up verdict of the two counts' audit statuses, worst first (``ChartAudit.verdict``)."""
    statuses = []
    for counts, lattice in ((domain_counts, lattice_domain), (complement_counts, lattice_complement)):
        if -1 in counts or len(set(counts)) > 1:
            statuses.append("unconverged")
        elif counts[0] == lattice:
            statuses.append("confirmed")
        else:
            statuses.append("corrected")
    if "unconverged" in statuses:
        return "unconverged"
    return "corrected" if "corrected" in statuses else "confirmed"


def chart_audit(
    faces: Faces,
    index: float,
    lattice_domain: int,
    lattice_complement: int,
    *,
    crystal: Polyhedron | None = None,
    ladder: tuple[int, ...] | None = None,
) -> ChartAudit:
    """Audit the lattice component counts against a ladder of orthographic charts (module docstring).

    ``ladder`` defaults to the module constant ``AUDIT_LADDER``, read at call time so a runtime
    change of the constant (empty = the rollback switch) takes effect on the next call.  One chart
    per resolution of ``ladder`` (each grid's nodes are the next finer grid's
    even-indexed subset: ``linspace(-1, 1, g)`` sits on half the spacing), its ``U_P`` mask from
    :func:`valid_batch`, its counts from :func:`_chart_component_counts`.  A count every grid
    agrees on is the audited count -- a ``"correction"`` of the lattice or a ``"confirmation"``
    of it; any disagreement or rim breach is ``"unconverged"`` and establishes nothing.  The
    audit is evidence, not a proof: two chains that share no failure mode (sampling geometry,
    connectivity, resolution) agreeing and converging -- an arrangement-exact certificate would
    need more.
    """
    if ladder is None:
        ladder = AUDIT_LADDER
    n_a = optics.face_normals(crystal, faces)[0]
    domain_counts: list[int] = []
    complement_counts: list[int] = []
    for grid in ladder:
        u, off_chart = _chart_grid(grid, n_a)
        counts = _chart_component_counts(_chart_valid(u, faces, index, crystal), off_chart)
        domain_counts.append(-1 if counts is None else counts[0])
        complement_counts.append(-1 if counts is None else counts[1])
    verdict = _audit_verdict(tuple(domain_counts), tuple(complement_counts), lattice_domain, lattice_complement)
    return ChartAudit(tuple(ladder), tuple(domain_counts), tuple(complement_counts), lattice_domain, lattice_complement, verdict)


def domain_topology(
    faces: Faces, index: float, *, lattice_n: int = 20000, crystal: Polyhedron | None = None
) -> DomainTopology:
    """Component counts of ``U_P`` and ``S^2 \\ U_P``, audited when the lattice count is plural (module docstring)."""
    lattice = fibonacci_sphere(lattice_n)
    valid = valid_batch(lattice, faces, index, crystal=crystal)
    domain, complement = _component_count(lattice[valid]), _component_count(lattice[~valid])
    audit = None
    if AUDIT_LADDER and (domain > 1 or complement > 1):  # the trigger gate: the audit's cost is paid only here
        audit = chart_audit(faces, index, domain, complement, crystal=crystal)
        if audit.verdict != "unconverged":
            domain, complement = audit.domain_counts[0], audit.complement_counts[0]
    return DomainTopology(lattice_n, domain, complement, grid_audit=audit)


def _side_values(
    point: np.ndarray, faces: Faces, index: float, slab: np.ndarray | None, crystal: Polyhedron | None
) -> np.ndarray:
    """``D_P`` on the part of a ``SIDE_RING_RAD`` ring around ``point`` inside ``U_P``."""
    e = np.asarray(tangent_basis(point))
    angles = np.linspace(0.0, 2.0 * np.pi, SIDE_RING_DIRECTIONS, endpoint=False)
    ring = np.cos(SIDE_RING_RAD) * point + np.sin(SIDE_RING_RAD) * (np.cos(angles)[:, None] * e[0] + np.sin(angles)[:, None] * e[1])
    inside = np.all(validity_margins_batch(ring, faces, index, crystal=crystal) > 0.0, axis=1)
    return d_p_batch(ring[inside], faces, index, slab, crystal=crystal)


def _interior_extremum(
    point: InteriorCriticalPoint, faces: Faces, index: float, slab: np.ndarray | None, crystal: Polyhedron | None
) -> str:
    """``"minimum"`` / ``"maximum"`` of a critical point (a slab point by its ring), else ``TopologyEscape``."""
    if point.kind in ("minimum", "maximum"):
        return point.kind
    if point.kind == "degenerate" and slab is not None:
        ring = _side_values(point.position, faces, index, slab, crystal)
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
    """Sorted critical values (rad): interior, restricted to pieces, and corners; merged within ``EXTREMUM_ATOL``.

    A loop of constant ``D_P`` (:attr:`.boundary.BoundaryLoop.plateau_value`)
    contributes that value: it is a critical value of the restricted ``D_P``
    (the whole boundary is one level), carried by no isolated extremum.
    """
    raw = sorted(
        [p.value for p in interior]
        + [c.value for c in loop.critical_points]
        + [c.value for c in loop.corners]
        + ([] if loop.plateau_value is None else [loop.plateau_value])
    )
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
    *,
    crystal: Polyhedron | None = None,
) -> tuple[DeviationInterval, ...]:
    """The partition of ``[min D_P, max D_P]`` with the counts of every interval (module docstring).

    ``crystal`` (default: the canonical hexagonal prism) supplies the face normals of the ring probes.
    """
    if not topology.is_disk:
        audit = topology.grid_audit
        if audit is None:  # no audit ran: a hand-built topology, or an empty audit ladder
            raise TopologyEscape(
                f"U_P is not a disk on the {topology.lattice_n}-point lattice: {topology.domain_components} component(s), "
                f"complement {topology.complement_components}"
            )
        if audit.verdict == "unconverged":
            raise TopologyEscape(
                f"U_P is not a disk: lattice {topology.lattice_n} says {audit.lattice_domain_count}/"
                f"{audit.lattice_complement_count}, chart grids {audit.grids} say {audit.domain_counts}/"
                f"{audit.complement_counts}; counts are not resolution-converged, the topology is not established"
            )
        if audit.verdict == "confirmed":
            raise TopologyEscape(
                f"U_P is not a disk: {topology.domain_components} component(s), complement {topology.complement_components} "
                f"(lattice {topology.lattice_n} and chart grids {audit.grids} agree)"
            )
        # corrected: the grids agree on a count the lattice got wrong, and the adjudicated counts still fail is_disk
        assert audit.verdict == "corrected"
        raise TopologyEscape(
            f"U_P is not a disk: {topology.domain_components} component(s), complement {topology.complement_components} "
            f"(chart grids {audit.grids} agree, correcting the lattice {topology.lattice_n} counts "
            f"{audit.lattice_domain_count}/{audit.lattice_complement_count})"
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
        kind = _interior_extremum(interior[0], faces, index, slab, crystal)
        v = interior[0].value
        if loop.plateau_value is not None:
            # a constant loop: every point of dU_P is both its minimum and its maximum, at the plateau
            edge = loop.plateau_value
            touching_positions = [loop.pieces[0].points[0]]
        else:
            edge = values.min() if kind == "minimum" else values.max()
            touching_positions = [p.position for p in extrema if p.kind == kind and abs(p.value - edge) <= EXTREMUM_ATOL]
        sign = 1.0 if kind == "minimum" else -1.0
        reaches = any(
            np.any(sign * (_side_values(position, faces, index, slab, crystal) - edge) < -1e-12)
            for position in touching_positions
        )
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
