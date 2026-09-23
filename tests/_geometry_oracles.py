"""Independent oracles for the geometry tests.

The blueprint tests in ``halo_notes.geometry.tests`` cross-check the geometry
package against other writing-repository modules (``halo_notes.math.
reflection_group``, ``halo_notes.draw.raypath``, ``docs/checks/check_pyramid``).
Those modules are not part of Lumice Integral, so this file re-implements the
minimal oracle surface the migrated tests need.  Everything here is derived
from closed-form face normals and the convex-body ray intersection only; it
deliberately shares no code with ``unfold`` / ``feasibility`` / ``enumerate``
so that agreement is evidence, not tautology.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from lumice_integral.geometry import N_ICE, Polyhedron, unit

# ---- hexagonal-prism reflection group (replaces halo_notes.math.reflection_group) ----

NORMALS: dict[int, np.ndarray] = {1: np.array([0.0, 0.0, 1.0]), 2: np.array([0.0, 0.0, -1.0])}
for _k in range(6):
    _a = np.radians(60.0 * _k)
    NORMALS[3 + _k] = np.array([np.cos(_a), np.sin(_a), 0.0])


def mirror(n: Sequence[float]) -> np.ndarray:
    """Householder reflection ``I - 2 n n^T`` for a unit normal."""
    v = unit(n)
    return np.eye(3) - 2.0 * np.outer(v, v)


MIRRORS: dict[int, np.ndarray] = {k: mirror(v) for k, v in NORMALS.items()}


def path_matrix(faces: Sequence[int]) -> np.ndarray:
    """Direction matrix of a face sequence: internal mirrors left-multiplied in order of encounter."""
    faces = [int(f) for f in faces]
    if len(faces) == 1:
        return MIRRORS[faces[0]]
    M = np.eye(3)
    for f in faces[1:-1]:
        M = MIRRORS[f] @ M
    return M


def refraction_cancels(faces: Sequence[int]) -> bool:
    """Parallel-path criterion: exit normal equals ``-M n_in``."""
    faces = [int(f) for f in faces]
    if len(faces) == 1:
        return True
    return bool(np.allclose(NORMALS[faces[-1]], -path_matrix(faces) @ NORMALS[faces[0]]))


def _d6_images(face: int) -> list[int]:
    """Images of one face under the 12 elements of D6 (rotations and flips about the c axis)."""
    if face in (1, 2):
        return [face] * 12
    out = []
    for shift in range(6):
        out.append(3 + (face + shift - 3) % 6)
    for shift in range(6):
        out.append(3 + (shift - face - 3) % 6)
    return out


def pbd_orbit(faces: Sequence[int]) -> set[tuple[int, ...]]:
    """Orbit of a face sequence under D6 x (basal swap 1 <-> 2): Lumice ``PBD`` symmetry."""
    faces = tuple(int(f) for f in faces)
    columns = [_d6_images(f) for f in faces]
    out: set[tuple[int, ...]] = set()
    for g in range(12):
        image = tuple(col[g] for col in columns)
        out.add(image)
        out.add(tuple({1: 2, 2: 1}.get(f, f) for f in image))
    return out


# ---- pyramid face normals in closed form (replaces docs/checks/check_pyramid.py) ----


def pyramid_reference_normals(c_over_a: float) -> dict[int, np.ndarray]:
    """Outward normals of all 20 faces: basal +-z, prism azimuth ``i*60``, cone faces tilted by
    ``theta = arctan(2/sqrt(3) * c_over_a)`` from the c axis with the same azimuth as prism face i."""
    theta = np.arctan(2.0 / np.sqrt(3.0) * c_over_a)
    out = {1: np.array([0.0, 0.0, 1.0]), 2: np.array([0.0, 0.0, -1.0])}
    for i in range(6):
        a = np.radians(60.0 * i)
        c, s = np.cos(a), np.sin(a)
        out[3 + i] = np.array([c, s, 0.0])
        out[13 + i] = np.array([np.sin(theta) * c, np.sin(theta) * s, np.cos(theta)])
        out[23 + i] = np.array([np.sin(theta) * c, np.sin(theta) * s, -np.cos(theta)])
    return out


# ---- ray tracing witness (replaces halo_notes.draw.raypath.trace) ----


def reflect(d: np.ndarray, n: np.ndarray) -> np.ndarray:
    return d - 2.0 * (d @ n) * n


def refract(d: np.ndarray, n: np.ndarray, n1: float, n2: float) -> np.ndarray:
    """Vector Snell refraction; ``n`` points toward the incident medium.  Raises on TIR."""
    eta = n1 / n2
    cos_i = -(n @ d)
    sin2_t = eta * eta * (1.0 - cos_i * cos_i)
    if sin2_t > 1.0:
        raise ValueError("total internal reflection: refraction impossible here")
    return eta * d + (eta * cos_i - np.sqrt(1.0 - sin2_t)) * n


def trace_faces(crystal: Polyhedron, origin: Sequence[float], direction: Sequence[float],
                n_events: int, *, n_ice: float = N_ICE) -> list[int]:
    """Trace a ray from outside the crystal and return the face numbers it meets.

    The first hit refracts in; the next ``n_events - 2`` hits reflect internally; the last hit
    refracts out.  ``n_events == 1`` means external reflection (one face).  Only
    :meth:`Polyhedron.intersect_ray` and Snell's law are used -- no unfolding, no corridors.
    Raises ``ValueError`` if the ray misses the crystal or hits TIR at the exit.
    """
    o = np.asarray(origin, dtype=float)
    d = unit(direction)
    hit = crystal.intersect_ray(o, d)
    if hit is None or hit[0] <= 0:
        raise ValueError("ray does not hit the crystal from outside")
    t_in, f_in, _, _ = hit
    p = o + t_in * d
    faces = [f_in.number]
    if n_events == 1:
        return faces
    d = refract(d, crystal.normal(f_in), 1.0, n_ice)
    for k in range(1, n_events):
        hit = crystal.intersect_ray(p + d * 1e-9, d)
        if hit is None:
            raise RuntimeError("internal ray lost the crystal")
        _, _, t_out, f_out = hit
        p = p + d * (t_out + 1e-9)
        faces.append(f_out.number)
        n_out = crystal.normal(f_out)
        if k == n_events - 1:
            refract(d, -n_out, n_ice, 1.0)   # raises on TIR: the path must be able to exit here
        else:
            d = reflect(d, -n_out)
    return faces


# ---- Phi-group key (independent of lumice_integral.path_class.phi_key) ----


def phi_group_key(faces: Sequence[int]) -> tuple:
    """``(M rounded, entry face, face of M^T n_b)`` from the closed-form normals and :func:`path_matrix`.

    Rounding ``M`` to six decimals stands in for ``phi_key``'s match against the
    ``D6h`` table, so the two keys agree only if they induce the same partition.
    """
    faces = [int(f) for f in faces]
    M = path_matrix(faces)
    target = M.T @ NORMALS[faces[-1]]
    exit_face = [k for k, n in NORMALS.items() if np.allclose(n, target, atol=1e-9)]
    assert len(exit_face) == 1, (faces, exit_face)
    return tuple(np.round(M, 6).ravel() + 0.0), faces[0], exit_face[0]
