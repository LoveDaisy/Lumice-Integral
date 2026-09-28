"""The crystal's own symmetry group ``G_true``: the elements of a candidate group that map the crystal to itself.

``G_true`` is the *shape* half of the physical symmetry meaning, L2
(``docs/conventions.md`` #21, Lumice ``doc/raypath-symmetry.md`` §1.1): a
raypath-analysis panel row (Lumice's other meaning) also needs the pose
ensemble (§2b there) to actually realize the symmetry, which this module
does not decide -- that half is the caller's ``pose_density``.  L1, Lumice's
``symmetry: "PBD"`` filter, is a different, unconditional label rewrite
(:func:`.reflection_group.pbd_orbit`) that ignores both halves; the two
meanings coincide only when ``G_true`` is all of ``D6h`` (the regular
hexagonal prism).

A crystal built by :mod:`lumice_integral.geometry` has every face normal in the six-direction star
``{i·60°}`` or on ``±c``; an orthogonal map that permutes that finite set is one of the 24 elements of
:data:`.signature.D6H`, so ``D6H`` is an exhaustive candidate set (read from the one table in the repository,
not rebuilt).  An element ``g`` is kept iff it maps the set of present faces
``{(n_i, d_i)}`` onto itself, where ``n_i`` is the outward normal and ``d_i = n_i · (p_i − c)`` the plane
offset measured from the vertex centroid ``c`` (a symmetry permutes the vertices, so it fixes ``c``; measuring
from ``c`` rather than the origin makes the answer independent of where the crystal sits).  Only the linear
part matters to the consumers (directions, fold matrices, entry cross-sections), so a translation of the
crystal changes nothing.

The input is a crystal in its body frame (c axis along +z, face 3 at azimuth 0°), the frame the candidate
table is written in.  The returned subset is checked to be a group (identity, closure, inverses) before it is
returned; a failure there is a defect of this function, not of the input, and raises ``AssertionError``.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..geometry import Polyhedron
from . import reflection_group as rg
from .signature import D6H

NORMAL_TOL = 1e-6
"""Two unit normals are the same direction iff they differ by at most this (``signature._face_lookup``'s tolerance)."""
OFFSET_REL_TOL = 1e-6
"""Two plane offsets are equal iff they differ by at most this times the crystal's largest offset."""


def face_planes(crystal: Polyhedron) -> tuple[np.ndarray, np.ndarray]:
    """``(normals (F, 3), offsets (F,))`` of the present faces, offsets from the vertex centroid, ``crystal.faces`` order."""
    centroid = crystal.centroid()
    normals = np.stack([crystal.normal(f) for f in crystal.faces])
    offsets = np.array([n @ (crystal.face_vertices(f)[0] - centroid) for n, f in zip(normals, crystal.faces)])
    return normals, offsets


def _maps_onto_itself(g: np.ndarray, normals: np.ndarray, offsets: np.ndarray, offset_tol: float) -> bool:
    images = normals @ g.T
    for image, offset in zip(images, offsets):
        distance = np.linalg.norm(normals - image, axis=1)
        j = int(np.argmin(distance))
        if distance[j] > NORMAL_TOL or abs(offsets[j] - offset) > offset_tol:
            return False
    return True   # face normals are distinct, so the matching is one-to-one and the image set is the set


def check_group(elements: Sequence[np.ndarray]) -> None:
    """Raise ``AssertionError`` unless ``elements`` (orthogonal matrices) contain the identity, are closed
    under products and contain every inverse (the transpose), all compared by :func:`.reflection_group.key`."""
    keys = {rg.key(g) for g in elements}
    if len(keys) != len(elements):
        raise AssertionError("duplicate elements")
    if rg.key(np.eye(3)) not in keys:
        raise AssertionError("the identity is missing")
    for g in elements:
        if rg.key(g.T) not in keys:
            raise AssertionError(f"the inverse of {rg.key(g)} is missing")
        for h in elements:
            if rg.key(g @ h) not in keys:
                raise AssertionError(f"the product of {rg.key(g)} and {rg.key(h)} is missing")


def true_symmetry_group(crystal: Polyhedron,
                        candidates: Sequence[np.ndarray] = D6H) -> tuple[np.ndarray, ...]:
    """``G_true``: the ``candidates`` (default :data:`.signature.D6H`, in its order) that map ``crystal`` onto itself.

    See the module docstring for the test and its frame.  The result is verified by :func:`check_group`.
    This is L2's shape half only (module docstring, ``docs/conventions.md`` #21); combine with the caller's
    pose ensemble for the full physical-equivalence meaning.
    """
    normals, offsets = face_planes(crystal)
    offset_tol = OFFSET_REL_TOL * float(np.max(np.abs(offsets)))
    group = tuple(g for g in candidates if _maps_onto_itself(np.asarray(g, dtype=float), normals, offsets, offset_tol))
    check_group(group)
    return group
