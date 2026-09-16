"""Finite-crystal geometry for Lumice Integral: convex polyhedra, unfolding,
corridor projection intersection, direction feasibility, ray-path enumeration,
and the single-pose effective entry cross-section ``entry_measure``.

This subpackage is pure numpy and never imports JAX. It is the authoritative
implementation of the crystal geometry that the writing repository's
``halo_notes.geometry`` used to own; that repository now calls this package.

Migrated from the blueprint (2026-09-16): every public name below is kept 1:1
with ``halo_notes.geometry`` -- only the import path changed, so downstream
code migrates by rewriting ``halo_notes.geometry`` to
``lumice_integral.geometry``:

- ``core``: ``Polyhedron`` / ``Face`` / ``HexPrism``, rigid transforms, ray
  intersection, ``N_ICE``, face-number constants.
- ``unfold``: ``unfold_faces`` / ``fold_matrix``.
- ``pyramid``: ``Pyramid`` and its face-number constants.
- ``feasibility``: ``LatLonGrid``, corridor clipping, ``admissible_directions``,
  ``entry_points``, ``external_directions`` and the optical gates.
- ``enumerate``: DFS ray-path enumeration.

Original to this repository (not in the blueprint):

- ``entry_measure`` / ``EntryMeasureResult``: effective entry cross-section of
  a fixed face sequence at one pose, measured perpendicular to the world-frame
  incident direction (see ``entry_measure.py`` for the normalisation contract).

Known coupling: ``lumice_integral/__init__.py`` enables JAX float64 at import
time, so importing this subpackage still imports JAX through the parent
package even though no module here uses it.
"""

from .core import (
    BASAL_BOTTOM,
    BASAL_TOP,
    N_ICE,
    PRISM_FACES,
    Face,
    HexPrism,
    Polyhedron,
    perp_basis,
    rotation,
    rotation_between,
    rotation_from_frames,
    unit,
)
from .entry_measure import EntryMeasureResult, entry_measure
from .enumerate import (
    EnumerationStats,
    RaypathRecord,
    SymmetryOrbit,
    corridor_index_for_path,
    enumerate_raypaths,
    enumerate_raypaths_with_stats,
)
from .feasibility import (
    COS_CRITICAL,
    EPS_REL,
    AdmissibleMask,
    AdmissibleResult,
    CorridorMask,
    LatLonGrid,
    admissible_directions,
    area_eps,
    corridor_intersection,
    corridor_mask,
    corridor_polygons,
    entry_ok,
    entry_points,
    exit_ok,
    external_directions,
    geometric_ok,
    incidence_objective_deg,
    is_feasible,
    min_edge_length,
    perp_bases,
)
from .pyramid import (
    C_OVER_A_ICE,
    LOWER_PYRAMID_FACES,
    UPPER_PYRAMID_FACES,
    Pyramid,
    pyramid_face_angle,
)
from .unfold import fold_matrix, unfold_faces

__all__ = [
    "AdmissibleMask",
    "AdmissibleResult",
    "BASAL_BOTTOM",
    "BASAL_TOP",
    "COS_CRITICAL",
    "C_OVER_A_ICE",
    "CorridorMask",
    "EPS_REL",
    "EntryMeasureResult",
    "EnumerationStats",
    "Face",
    "HexPrism",
    "LOWER_PYRAMID_FACES",
    "LatLonGrid",
    "N_ICE",
    "PRISM_FACES",
    "Polyhedron",
    "Pyramid",
    "RaypathRecord",
    "SymmetryOrbit",
    "UPPER_PYRAMID_FACES",
    "admissible_directions",
    "area_eps",
    "corridor_index_for_path",
    "corridor_intersection",
    "corridor_mask",
    "corridor_polygons",
    "entry_measure",
    "entry_ok",
    "entry_points",
    "enumerate_raypaths",
    "enumerate_raypaths_with_stats",
    "exit_ok",
    "external_directions",
    "fold_matrix",
    "geometric_ok",
    "incidence_objective_deg",
    "is_feasible",
    "min_edge_length",
    "perp_basis",
    "perp_bases",
    "pyramid_face_angle",
    "rotation",
    "rotation_between",
    "rotation_from_frames",
    "unfold_faces",
    "unit",
]
