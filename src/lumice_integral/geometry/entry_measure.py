"""Single-pose effective entry cross-section of a fixed face sequence.

This module is original to Lumice Integral (the blueprint ``halo_notes.geometry``
has no counterpart).  It composes the migrated corridor primitives in
:mod:`.feasibility` into one geometric weight:

    ``entry_measure(rotation, path, incident_direction, crystal)``

For a crystal at world pose ``rotation`` (body -> world), a fixed face sequence
``path = (a, m_1, ..., m_k, b)`` and a world-frame incident propagation
direction ``s``, the set of entry points on face ``a`` whose refracted internal
line follows exactly this face sequence is the *footprint* ``F``.  The measure
returned here is the area of ``F`` projected onto the plane perpendicular to
``s`` -- the cross-section that a parallel incident beam presents to this
path.  It is the geometric factor ``A_P(R)`` that later physical weights
multiply.

Normalisation contract (explicit, no hidden reference area):

- ``value`` is an **absolute area** in the square of the length unit of
  ``crystal.vertices``; it is not divided by any face area, crystal
  cross-section or other reference.
- ``value`` is measured perpendicular to the **world-frame incident
  direction** ``incident_direction`` (not perpendicular to the internal
  direction and not on the face itself).

Algorithm.  Everything is evaluated in the crystal body frame:

1. ``s_body = R^T s``.  Entry gate: ``cos_i = -(n_a . s_body) > 0`` (the
   incident side has no critical angle, so the gate is ``entry_ok`` with
   ``cos_tc = 0``).  Failure -> ``status = "entry_backface"``.
2. Snell refraction into the crystal with relative index ``1 / n_ice`` gives
   the internal direction ``d_in`` and ``cos_t = -(n_a . d_in)``.
3. Exit gate on the unfolded exit normal ``n_tilde_b`` from
   :func:`~.feasibility.corridor_polygons`: ``exit_ok(n_tilde_b, d_in)`` with
   the default ``COS_CRITICAL``.  Failure -> ``status = "exit_critical_angle"``.
4. ``A_perp = corridor_intersection(polys, d_in).area()``: area of the
   corridor footprint projected perpendicular to ``d_in``.  ``A_perp <= eps``
   -> ``status = "corridor_empty"``.
5. ``value = A_perp * cos_i / cos_t``: the same footprint on face ``a`` has
   area ``A_perp / cos_t`` on the face and ``(A_perp / cos_t) * cos_i``
   perpendicular to ``s``.

The refraction step is written out in numpy on purpose: this package must not
import JAX, so it cannot reuse ``lumice_integral.optics.refract_smooth``.  The
formula is the same Snell vector form as ``optics.refract_smooth`` /
``optics.path_3_5_domain``; ``tests/test_geometry_entry_measure.py`` asserts the
two implementations agree numerically.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .core import N_ICE, Polyhedron, Vec3, unit
from .feasibility import (
    area_eps,
    corridor_intersection,
    corridor_polygons,
    entry_ok,
    exit_ok,
)


@dataclass(frozen=True)
class EntryMeasureResult:
    """Result of :func:`entry_measure`.

    - ``value``: effective entry cross-section perpendicular to the world-frame
      incident direction (absolute area, see module docstring).  ``0.0`` when
      any gate fails.
    - ``status``: ``"ok"``, ``"entry_backface"``, ``"exit_critical_angle"`` or
      ``"corridor_empty"``.
    - ``area_perp_internal``: corridor footprint area projected perpendicular
      to the internal direction (``A_perp``), before the cosine conversion.
      ``nan`` when a gate failed before it was computed.
    - ``internal_direction``: body-frame internal direction ``d_in``
      (``None`` if the entry gate failed).
    - ``cosine_incident``: ``cos_i = -(n_a . s_body)``; negative or zero
      means the ray arrives from behind face ``a``.
    - ``cosine_internal``: ``cos_t = -(n_a . d_in)`` (``None`` if the entry
      gate failed).
    """

    value: float
    status: str
    area_perp_internal: float
    internal_direction: np.ndarray | None
    cosine_incident: float | None
    cosine_internal: float | None


def refract_into_crystal(direction: Vec3, outward_normal: Vec3, n_ice: float) -> tuple[Vec3, float, float]:
    """Snell refraction from air into the crystal at a face with outward normal ``outward_normal``.

    Returns ``(d_in, cos_i, cos_t)``.  ``direction`` is the unit propagation direction of the
    incident ray; ``cos_i = -(n . d) > 0`` is required by the caller.  With relative index
    ``eta = 1 / n_ice < 1`` the discriminant is always positive, so no TIR branch exists here.
    Same vector form as ``lumice_integral.optics.refract_smooth`` (kept numpy-only on purpose).
    """
    d = np.asarray(direction, dtype=float)
    n = np.asarray(outward_normal, dtype=float)
    eta = 1.0 / n_ice
    cos_i = -float(n @ d)
    discriminant = 1.0 - eta * eta * (1.0 - cos_i * cos_i)
    d_in = eta * d + (eta * cos_i - np.sqrt(discriminant)) * n
    cos_t = -float(n @ d_in)
    return d_in, cos_i, cos_t


def entry_measure(rotation: np.ndarray, path: Sequence[int], incident_direction: Sequence[float],
                  crystal: Polyhedron, *, eps: float | None = None, n_ice: float = N_ICE) -> EntryMeasureResult:
    """Effective entry cross-section of face sequence ``path`` at one pose.

    ``rotation`` is the body -> world rotation matrix (same convention as ``optics.path_3_5``:
    world face normal = ``rotation @ body_normal``).  ``path`` is the face sequence
    ``(entry, *reflections, exit)`` with at least two faces.  ``incident_direction`` is the
    world-frame propagation direction of the incident light (pointing *toward* the crystal).
    ``eps`` is the corridor-area threshold below which the footprint counts as empty
    (default :func:`~.feasibility.area_eps`); ``n_ice`` the refractive index.

    See the module docstring for the normalisation contract and the gate order.
    """
    faces = [int(f) for f in path]
    if len(faces) < 2:
        raise ValueError("path must be (entry, *reflections, exit) with at least two faces")
    R = np.asarray(rotation, dtype=float)
    if R.shape != (3, 3):
        raise ValueError("rotation must be a 3x3 matrix")
    eps = area_eps(crystal) if eps is None else float(eps)

    s_body = R.T @ unit(incident_direction)
    n_a = crystal.normal(crystal.face(faces[0]))
    cos_i = -float(n_a @ s_body)
    # Entry side has no critical angle: the gate is entry_ok with cos_tc = 0 on the open half-space.
    entered = bool(entry_ok(n_a, s_body[None, :], cos_tc=0.0)[0]) and cos_i > 0.0
    if not entered:
        return EntryMeasureResult(0.0, "entry_backface", float("nan"), None, cos_i, None)

    d_in, cos_i, cos_t = refract_into_crystal(s_body, n_a, n_ice)
    polys, n_tilde_b = corridor_polygons(crystal, faces)
    if not bool(exit_ok(n_tilde_b, d_in[None, :])[0]):
        return EntryMeasureResult(0.0, "exit_critical_angle", float("nan"), d_in, cos_i, cos_t)

    area_perp = float(corridor_intersection(polys, d_in[None, :]).area()[0])
    if area_perp <= eps:
        return EntryMeasureResult(0.0, "corridor_empty", area_perp, d_in, cos_i, cos_t)

    value = area_perp * cos_i / cos_t
    return EntryMeasureResult(value, "ok", area_perp, d_in, cos_i, cos_t)


__all__ = ["EntryMeasureResult", "entry_measure", "refract_into_crystal"]
