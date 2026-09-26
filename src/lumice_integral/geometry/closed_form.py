"""Closed-form hexagonal cross-section: the six side planes of a prism with Lumice ``face_distance``.

Scope: only the fixed six-direction star (side face ``3+i`` has its outward normal at azimuth ``i·60°``,
``geometry.core``'s numbering); this is not a general polygon clipper.  Both :class:`.core.HexPrism` and
:class:`.pyramid.Pyramid` build their hexagonal rings here, so the repository has one cross-section.

Semantics follow Lumice (``doc/configuration.md`` §prism, ``doc/crystal-geometry-representation.md`` §4):

- ``face_distance[i]`` is the distance of side plane ``3+i`` from the c axis as a ratio of the regular
  hexagon's apothem ``a·√3/2`` (``a`` = edge = circumradius, :class:`.core.HexPrism`'s ``a``).  All ones
  is the regular hexagon; a value may be negative (the plane passes beyond the axis).
- A side face is *present* iff its line carries a segment of the cross-section of positive length
  (Lumice: "≥ 2 distinct feasible corners lie on it").  Presence is decided analytically per face: the
  line of face ``i`` is cut by the other five half-planes into one interval, and the face is present iff
  that interval is longer than ``PRESENT_REL_TOL`` times the largest plane offset (scale-relative, as
  Lumice ``doc/numerical-robustness.md`` convention ② asks).  A plane that only touches a corner (for
  example the far faces of ``[1, 2, 1, 2, 1, 2]``, whose distance equals the corner radius of the
  triangle the near faces cut) is absent.
- Fewer than three present faces means the cross-section has no area (empty, a point or a segment):
  :func:`hex_cross_section` raises ``ValueError`` (fail-fast; Lumice drops such a crystal silently).
- The ring is closed form: corner ``k`` is the intersection of the lines of present faces ``k-1`` and
  ``k`` (cyclic, azimuth order), written as the corresponding corner of the regular reference hexagon
  plus the exact linear correction for the distance change.  With all ones the correction is exactly
  zero, so the regular ring is the reference ring bit for bit (the S^2 event store rebuilds its crystal
  and compares vertices with ``np.array_equal``).

No numerical topology discovery (no vertex search, no dedup, no hull): the face set is the analytic
presence mask and the face numbers are the constant table ``3+i``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

HALF_SQRT_3 = float(np.sqrt(3.0)) / 2.0
# outward normals of side faces 3+i, i = 0..5, azimuth i·60°, written with exact ±1/2 and ±√3/2
SIDE_NORMALS_2D = np.array([
    [1.0, 0.0], [0.5, HALF_SQRT_3], [-0.5, HALF_SQRT_3],
    [-1.0, 0.0], [-0.5, -HALF_SQRT_3], [0.5, -HALF_SQRT_3],
])
REGULAR_FACE_DISTANCE = (1.0,) * 6
PRESENT_REL_TOL = 1e-9
"""A side face is present iff its feasible segment is longer than this times the largest plane offset."""

_SIN_60K = (0.0, HALF_SQRT_3, HALF_SQRT_3, 0.0, -HALF_SQRT_3, -HALF_SQRT_3)   # sin(k·60°)
_COS_60K = (1.0, 0.5, -0.5, -1.0, -0.5, 0.5)                                 # cos(k·60°)


def _tangent(i: int) -> np.ndarray:
    """Unit tangent of side line ``i``, counter-clockwise seen from +c: ``(-sin, cos)`` of its azimuth."""
    n = SIDE_NORMALS_2D[i]
    return np.array([-n[1], n[0]])


@dataclass(frozen=True)
class HexCrossSection:
    """The closed-form cross-section: presence mask and counter-clockwise corner ring.

    ``present`` lists the present side indices ``i`` (face ``3+i``) in azimuth order starting from the
    smallest; ``ring[k]`` is the corner where face ``present[k-1]`` ends and face ``present[k]`` begins,
    so face ``3 + present[k]`` runs from ``ring[k]`` to ``ring[k+1]`` (cyclic).  For the regular hexagon
    ``present = (0, ..., 5)`` and ``ring[k]`` sits at azimuth ``-30° + 60°k`` (``HexPrism``'s layout).
    """

    a: float
    face_distance: tuple[float, ...]
    offsets: np.ndarray          # (6,) plane offsets n_i · x <= offsets[i]
    face_present: tuple[bool, ...]
    present: tuple[int, ...]
    ring: np.ndarray             # (len(present), 2)


def _face_interval(i: int, offsets: np.ndarray) -> tuple[float, float]:
    """Feasible parameter interval ``[t_lo, t_hi]`` of ``offsets[i]·n_i + t·tangent_i`` under the other planes."""
    t_lo, t_hi = -np.inf, np.inf
    for j in range(6):
        if j == i:
            continue
        k = (j - i) % 6
        s, c = _SIN_60K[k], _COS_60K[k]   # n_j · tangent_i = sin(θj − θi), n_j · n_i = cos(θj − θi)
        rhs = offsets[j] - offsets[i] * c
        if s == 0.0:                       # the antiparallel plane: feasible everywhere or nowhere
            if rhs < 0.0:
                return np.inf, -np.inf
            continue
        bound = rhs / s
        if s > 0.0:
            t_hi = min(t_hi, bound)
        else:
            t_lo = max(t_lo, bound)
    return t_lo, t_hi


def _reference_corner(a: float, q: int, gap: int) -> np.ndarray:
    """Intersection of the regular hexagon's lines ``q - gap`` and ``q`` (``gap`` 1 or 2).

    ``gap == 1`` is corner ``q`` of the regular hexagon, azimuth ``-30° + 60°q``, evaluated exactly as
    ``HexPrism`` always evaluated its ring (one vectorised ``a·cos`` / ``a·sin`` over the six azimuths,
    so the regular ring stays bit for bit); ``gap == 2`` (face ``q-1`` absent) lies on the bisector
    ``-60° + 60°q`` at radius ``(a·√3/2) / cos 60° = a·√3``.
    """
    if gap == 1:
        azimuths = np.deg2rad(-30.0 + 60.0 * np.arange(6))
        return np.array([(a * np.cos(azimuths))[q], (a * np.sin(azimuths))[q]])
    azimuth = np.deg2rad(-60.0 + 60.0 * q)
    radius = a * np.sqrt(3.0)
    return np.array([radius * np.cos(azimuth), radius * np.sin(azimuth)])


def _corner(a: float, p: int, q: int, excess: np.ndarray) -> np.ndarray:
    """Intersection of lines ``p`` and ``q`` (consecutive present faces) as reference corner + correction.

    Solves ``n_p · x = r + excess[p]``, ``n_q · x = r + excess[q]`` (``r`` the regular apothem): the
    reference corner takes ``r``, the correction is the 2x2 inverse applied to the excess,
    ``x = x_ref + (excess[p]·(-t_q) + excess[q]·t_p) / sin(θq − θp)`` with ``t`` the line tangents.
    """
    gap = (q - p) % 6
    if gap not in (1, 2):
        raise AssertionError(f"consecutive present faces {p}, {q} are {gap * 60} degrees apart")
    x_ref = _reference_corner(a, q, gap)
    correction = (excess[p] * -_tangent(q) + excess[q] * _tangent(p)) / _SIN_60K[gap]
    return x_ref + correction


def hex_cross_section(a: float = 1.0, face_distance: Sequence[float] = REGULAR_FACE_DISTANCE) -> HexCrossSection:
    """The closed-form cross-section of the six side planes (module docstring for the semantics)."""
    a = float(a)
    if not (np.isfinite(a) and a > 0.0):
        raise ValueError(f"a must be a positive finite length, got {a!r}")
    fd = tuple(float(f) for f in face_distance)
    if len(fd) != 6 or not all(np.isfinite(fd)):
        raise ValueError(f"face_distance must be six finite numbers, got {face_distance!r}")
    apothem = a * HALF_SQRT_3
    offsets = apothem * np.asarray(fd)
    # distance change against the regular hexagon, exactly zero for a face at ratio 1
    excess = apothem * (np.asarray(fd) - 1.0)
    tol = PRESENT_REL_TOL * float(np.max(np.abs(offsets)))

    face_present = []
    for i in range(6):
        t_lo, t_hi = _face_interval(i, offsets)
        face_present.append(bool(t_hi - t_lo > tol))
    present = tuple(i for i in range(6) if face_present[i])
    if len(present) < 3:
        raise ValueError(
            f"face_distance {list(fd)} leaves {len(present)} side faces with a segment of positive length: "
            "the cross-section has no area (Lumice's closed-manifold check rejects it)")
    ring = np.stack([_corner(a, present[k - 1], present[k], excess) for k in range(len(present))])
    return HexCrossSection(a, fd, offsets, tuple(face_present), present, ring)
