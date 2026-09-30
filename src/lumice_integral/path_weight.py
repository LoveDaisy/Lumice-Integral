"""The two-factor kernel of a fixed path: entry measure ``A`` and path power ``T`` at one refractive index.

Only the kernel, no weight schema (units, descriptions and the pose density
are :mod:`.weights`).  The one place that binds the two factors to the same
face sequence (:func:`.optics.normalize_faces` on the crystal) and the same
``n``: ``A`` from :func:`.geometry.entry_measure.entry_measure_batch` with
``n_ice = n`` (its exit gate refracts at that index), ``T`` from
:func:`.optics.fresnel_transmission_path_batch` (every internal ``R_k``
included).  Both are ``0`` outside the path's domain.

The factors stay separately observable (``AGENTS.md``: physical weights are
independently observable before they are multiplied), and callers keep their
own products (``rho A T`` of a rank-0 estimate, the store's ``sum_m A_m T_m``):
:func:`entry_and_power` returns the pair, :func:`weighted_power` is the plain
``A T``.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from . import optics
from .geometry import Polyhedron
from .geometry.entry_measure import entry_measure_batch


def entry_measure(
    rotations: np.ndarray, faces: Sequence[int], incident_direction: np.ndarray, index: float, *, crystal: Polyhedron
) -> np.ndarray:
    """``A`` per pose: the entry footprint of ``faces`` on ``crystal``, its exit gate at refractive index ``index``."""
    faces = optics.normalize_faces(faces, crystal)
    return entry_measure_batch(rotations, faces, incident_direction, crystal, n_ice=float(index))


def path_power(
    rotations: np.ndarray, faces: Sequence[int], incident_direction: np.ndarray, index: float, *, crystal: Polyhedron
) -> np.ndarray:
    """``T`` per pose: entry and exit transmittance times every internal reflectance, at ``index``."""
    faces = optics.normalize_faces(faces, crystal)
    return optics.fresnel_transmission_path_batch(rotations, faces, incident_direction, float(index), crystal=crystal)


def entry_and_power(
    rotations: np.ndarray, faces: Sequence[int], incident_direction: np.ndarray, index: float, *, crystal: Polyhedron
) -> tuple[np.ndarray, np.ndarray]:
    """``(A, T)`` per pose, both at ``index`` (:func:`entry_measure`, :func:`path_power`)."""
    return (
        entry_measure(rotations, faces, incident_direction, index, crystal=crystal),
        path_power(rotations, faces, incident_direction, index, crystal=crystal),
    )


def weighted_power(
    rotations: np.ndarray, faces: Sequence[int], incident_direction: np.ndarray, index: float, *, crystal: Polyhedron
) -> np.ndarray:
    """``A T`` per pose, ``0`` outside the domain."""
    area, power = entry_and_power(rotations, faces, incident_direction, index, crystal=crystal)
    return area * power


__all__ = ["entry_and_power", "entry_measure", "path_power", "weighted_power"]
