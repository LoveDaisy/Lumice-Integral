"""Pixel-to-direction adapter for the canonical ``linear`` (pinhole) camera.

The canonical ch06 scene (``docs/ch06-reference-fixture.md`` section 3.3) is a
Lumice render with ``lens.type == "linear"``.  This module re-implements the
forward projection independently so that a pixel of that render can be bound
to an outgoing direction ``d`` for the fiber solver.  It is a convention
adapter, not a camera model of its own: every formula below was transcribed
from Lumice ``v4.6.0`` source read as evidence (not linked or imported):

- ``src/core/scatter_accum.hpp::MakeCameraRotation``:
  ``R = Rz(az) . Ry(90 - el) . Rz(-90 + roll)`` (active rotations; the
  columns of ``R`` are the camera x / y / z axes in world coordinates and the
  z axis is the line of sight).
- ``src/core/shared/projection_shared.h::ProjectExitToPixel``: the camera-frame
  vector is ``c = R^T (-w)`` where ``w`` is the world direction the light
  travels *after* leaving the crystal, so ``-w`` points toward the sky; the
  linear forward map is ``(x, y) = (c_x / c_z, c_y / c_z)`` with ``c_z > 0``;
  screen handedness then negates ``x`` ("right = +az"); finally
  ``px = floor(x * scale + W / 2 + shift_x)`` and
  ``py = floor(y * scale + H / 2 + shift_y)``.
- ``src/core/lens_proj_build.hpp::ComputeLensScale``:
  ``scale = min(W, H) / 2 / tan(fov / 2)`` for the linear lens.
- ``src/core/simulator.cpp::SampleRayDir`` and ``geo3d.cpp::SampleSphCapPoint``:
  the sun ray *travels* along ``-(cos(alt) cos(az), cos(alt) sin(az), sin(alt))``,
  the negative of the sun position vector :func:`sun_direction` returns.

The same chain (minus the linear branch) is already pinned against real
Lumice renders in the writing project's ``halo_notes/sim/projection.py``
tests; that module is used here only as a second reading of the convention.

Contract boundary (``docs/phase1-math-contract.md`` section 2): the solver's
``d`` is the propagation direction from the crystal toward the observer, i.e.
``d = w = -(sky direction)``.  :func:`linear_pixel_outgoing_direction` performs
that negation explicitly; :func:`linear_pixel_sky_direction` returns the
un-negated camera-side vector for projection round trips.  The same boundary
on the source side: :func:`sun_direction` is the public ``s_hat`` (toward
the sun), :func:`incident_direction_from_sun` the solver's propagation
direction ``s = -s_hat``.  (A pixel's continuous ``(u, v)`` below is a screen
coordinate, unrelated to Phase II's ``u = R^-1 s_hat``; ``docs/conventions.md``.)
"""

from __future__ import annotations

import functools
from collections.abc import Mapping

import numpy as np


def rotation_about_axis(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    """Active Rodrigues rotation about ``axis`` by ``angle_deg`` degrees."""
    unit_axis = np.asarray(axis, dtype=np.float64)
    unit_axis = unit_axis / np.linalg.norm(unit_axis)
    cosine = np.cos(np.radians(angle_deg))
    sine = np.sin(np.radians(angle_deg))
    generator = np.array(
        [
            [0.0, -unit_axis[2], unit_axis[1]],
            [unit_axis[2], 0.0, -unit_axis[0]],
            [-unit_axis[1], unit_axis[0], 0.0],
        ]
    )
    return cosine * np.eye(3) + sine * generator + (1.0 - cosine) * np.outer(
        unit_axis, unit_axis
    )


def camera_rotation(view: Mapping[str, float] | None) -> np.ndarray:
    """Lumice ``render.view`` -> camera rotation (columns = camera x / y / z); read-only, cached per view."""
    settings = dict(view or {})
    return _camera_rotation(
        float(settings.get("azimuth", 0.0)), float(settings.get("elevation", 0.0)), float(settings.get("roll", 0.0))
    )


@functools.lru_cache(maxsize=64)  # far more than the distinct views any single render or process touches
def _camera_rotation(azimuth: float, elevation: float, roll: float) -> np.ndarray:
    # Every pixel direction of a render asks for the same rotation (three Rodrigues matrices, most of a
    # pixel's geometry time); the same arithmetic once per view, so the directions are unchanged bit for bit.
    z_axis = np.array([0.0, 0.0, 1.0])
    y_axis = np.array([0.0, 1.0, 0.0])
    rotation = (
        rotation_about_axis(z_axis, azimuth)
        @ rotation_about_axis(y_axis, 90.0 - elevation)
        @ rotation_about_axis(z_axis, -90.0 + roll)
    )
    rotation.flags.writeable = False
    return rotation


def linear_scale(fov_deg: float, width: int, height: int) -> float:
    """Pixels per unit tangent for the Lumice linear lens."""
    return min(int(width), int(height)) / 2.0 / np.tan(np.radians(fov_deg) / 2.0)


def sun_direction(altitude_deg: float, azimuth_deg: float = 0.0) -> np.ndarray:
    """World unit vector ``s_hat`` *toward* the sun (Lumice ``coordinate-convention.md`` section 4).

    The public sun direction of the project (``docs/conventions.md``); the
    writing series' ``s`` of framework theorem 8, ``u = R^-1 s_hat``.
    """
    altitude = np.radians(altitude_deg)
    azimuth = np.radians(azimuth_deg)
    return np.array(
        [
            np.cos(altitude) * np.cos(azimuth),
            np.cos(altitude) * np.sin(azimuth),
            np.sin(altitude),
        ],
        dtype=np.float64,
    )


def incident_direction_from_sun(sun: np.ndarray) -> np.ndarray:
    """Propagation direction of sunlight (sun -> crystal), ``-s_hat``: the contract's ``s``.

    The one conversion between the public ``s_hat`` and the solver's
    ``incident_direction`` (``docs/phase1-math-contract.md`` section 2).
    Written ``0.0 - sun`` so that a zero component stays ``+0.0``, bit for
    bit the vector every fixture was recorded with.
    """
    return 0.0 - np.asarray(sun, dtype=np.float64)


def project_linear(
    sky_direction: np.ndarray,
    *,
    width: int,
    height: int,
    fov_deg: float,
    view: Mapping[str, float] | None,
    lens_shift: tuple[float, float] = (0.0, 0.0),
) -> tuple[float, float]:
    """Sky direction (unit vector toward the sky) -> continuous pixel ``(u, v)``.

    Pixel ``(column, row)`` covers ``u in [column, column + 1)`` and
    ``v in [row, row + 1)``.  Directions behind the camera raise ``ValueError``.
    """
    sky = np.asarray(sky_direction, dtype=np.float64)
    sky = sky / np.linalg.norm(sky)
    camera = camera_rotation(view).T @ sky
    if camera[2] <= 0.0:
        raise ValueError("direction lies behind the linear camera")
    x = -(camera[0] / camera[2])
    y = camera[1] / camera[2]
    scale = linear_scale(fov_deg, width, height)
    return (
        float(x * scale + width / 2.0 + lens_shift[0]),
        float(y * scale + height / 2.0 + lens_shift[1]),
    )


def linear_pixel_sky_direction(
    row: int,
    column: int,
    *,
    width: int,
    height: int,
    fov_deg: float,
    view: Mapping[str, float] | None,
    lens_shift: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Pixel centre ``(column + 1/2, row + 1/2)`` -> unit sky direction (toward the sky)."""
    scale = linear_scale(fov_deg, width, height)
    x = -((column + 0.5) - width / 2.0 - lens_shift[0]) / scale
    y = ((row + 0.5) - height / 2.0 - lens_shift[1]) / scale
    camera = np.array([x, y, 1.0], dtype=np.float64)
    camera /= np.linalg.norm(camera)
    return camera_rotation(view) @ camera


def linear_pixel_outgoing_direction(
    row: int,
    column: int,
    *,
    width: int,
    height: int,
    fov_deg: float,
    view: Mapping[str, float] | None,
    lens_shift: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Pixel centre -> solver target ``d`` (crystal -> observer), contract section 2.

    The negation of the camera-side sky direction happens here and nowhere else.
    """
    return -linear_pixel_sky_direction(
        row,
        column,
        width=width,
        height=height,
        fov_deg=fov_deg,
        view=view,
        lens_shift=lens_shift,
    )
