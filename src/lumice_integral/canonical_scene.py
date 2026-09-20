"""Canonical ch06 one-pixel scene: frozen constants and single-authority assembly.

Scene binding follows ``docs/ch06-reference-fixture.md`` section 3.3 (all
``canonical-new`` unless noted there).  The pixel and seed below are the
frozen *output* of one recorded run of
``scripts/discover_canonical_pixel_seed.py`` (parameters in
:data:`CANONICAL_SEED_DISCOVERY`); they are fixture constants, not the result
of a component-discovery algorithm, and a single closed loop from this seed
does not establish component completeness.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import jax.numpy as jnp
import numpy as np

from .camera import linear_pixel_outgoing_direction, project_linear, sun_incident_direction
from .continuation import FiberProblem
from .geometry import HexPrism
from .optics import path_3_5_problem
from .pose_density import PoseDensity, build_pose_density
from .pose_density_provenance import pose_density_provenance
from .so3 import exp
from .weights import build_3_5_weight_evaluators

# Lumice's crystal ``"height": 1.0`` means h / (base circumscribed diameter): its
# side faces sit at inradius sqrt(3)/4 * dist and its bases at z = +-h/2
# (``src/core/geo3d_closedform.cpp``, read as evidence, never linked), so the base
# circumscribed diameter is 1 and the hexagon edge is a = 1/2.  ``HexPrism.from_ratio``
# takes h / a with a = hexagon edge length, so the same crystal is 2 x the Lumice number.
LUMICE_HEIGHT_OVER_DIAMETER = 1.0  # the Lumice remake and the ``column1.0`` filenames
CANONICAL_HEIGHT_RATIO = 2.0 * LUMICE_HEIGHT_OVER_DIAMETER  # h / a, hexagonal column
CANONICAL_REFRACTIVE_INDEX = 1.31
CANONICAL_WAVELENGTH_NM = 550.0
CANONICAL_SUN_ALTITUDE_DEG = 15.0
CANONICAL_SUN_AZIMUTH_DEG = 0.0
CANONICAL_ZENITH_MEAN_DEG = 90.0
CANONICAL_ZENITH_STD_DEG = 0.5
CANONICAL_RENDER: dict[str, Any] = {
    "width": 251,
    "height": 801,
    "fov_deg": 6.0,
    "view": {"azimuth": 0.0, "elevation": -15.0},
}
CANONICAL_PIXEL_ROW = 150
CANONICAL_PIXEL_COLUMN = 150
CANONICAL_SEED_EXPONENTIAL_COORDINATES = np.array(
    [-1.6021189246370302, -0.06647507226372225, 0.614105700979059], dtype=np.float64
)
CANONICAL_SEED_DISCOVERY: dict[str, Any] = {
    "script": "scripts/discover_canonical_pixel_seed.py",
    "command": "uv run python scripts/discover_canonical_pixel_seed.py --row 150 --column 150",
    "samples": 400_000,
    "angle_tolerance_deg": 2.0,
    "candidates": 12,
    "rng_seed": 20260916,
    "selection": "admissible survivor whose c-axis zenith is closest to 90 deg",
    "pixel_rationale": (
        "row 150 / column 150 sits on the lit 3-5 band of the historical strip "
        "(raw value ~0.23), about 2.2 deg below the inner-edge caustic and 0.6 deg "
        "right of the sun azimuth"
    ),
}


def canonical_crystal() -> HexPrism:
    return HexPrism.from_ratio(CANONICAL_HEIGHT_RATIO)


def canonical_incident_direction() -> np.ndarray:
    return sun_incident_direction(CANONICAL_SUN_ALTITUDE_DEG, CANONICAL_SUN_AZIMUTH_DEG)


def canonical_target_direction() -> np.ndarray:
    return linear_pixel_outgoing_direction(
        CANONICAL_PIXEL_ROW, CANONICAL_PIXEL_COLUMN, **CANONICAL_RENDER
    )


def canonical_seed() -> np.ndarray:
    return np.asarray(exp(jnp.asarray(CANONICAL_SEED_EXPONENTIAL_COORDINATES)))


CANONICAL_POSE_DENSITY_FAMILY = "column"


def canonical_pose_density() -> PoseDensity:
    return build_pose_density(
        CANONICAL_POSE_DENSITY_FAMILY,
        zenith_mean_deg=CANONICAL_ZENITH_MEAN_DEG,
        zenith_std_deg=CANONICAL_ZENITH_STD_DEG,
    )


def canonical_pixel_problem(*, with_weights: bool = True) -> FiberProblem:
    """The canonical pixel's 3-5 continuation problem, optionally with the four weights."""
    incident = canonical_incident_direction()
    problem = path_3_5_problem(
        jnp.asarray(canonical_seed()),
        jnp.asarray(incident),
        target_direction=jnp.asarray(canonical_target_direction()),
        refractive_index=jnp.asarray(CANONICAL_REFRACTIVE_INDEX, dtype=jnp.float64),
    )
    if not with_weights:
        return problem
    evaluators = build_3_5_weight_evaluators(
        incident_direction=incident,
        refractive_index=CANONICAL_REFRACTIVE_INDEX,
        crystal=canonical_crystal(),
        pose_density=canonical_pose_density(),
    )
    return replace(problem, weight_evaluators=evaluators)


def canonical_fixture_metadata() -> dict[str, Any]:
    """Figure-data ``fixture`` block for the canonical pixel (JSON-serializable)."""
    sky = -canonical_target_direction()
    return {
        "name": "canonical-ch06-pixel-path-3-5",
        "path": [3, 5],
        "crystal": {"type": "hexagonal_column", "height_ratio": CANONICAL_HEIGHT_RATIO},
        "refractive_index": CANONICAL_REFRACTIVE_INDEX,
        "wavelength_nm": CANONICAL_WAVELENGTH_NM,
        "sun": {
            "altitude_deg": CANONICAL_SUN_ALTITUDE_DEG,
            "azimuth_deg": CANONICAL_SUN_AZIMUTH_DEG,
        },
        "pose_density": pose_density_provenance(
            CANONICAL_POSE_DENSITY_FAMILY,
            zenith_mean_deg=CANONICAL_ZENITH_MEAN_DEG,
            zenith_std_deg=CANONICAL_ZENITH_STD_DEG,
        ),
        "camera": {"lens": "linear", **CANONICAL_RENDER},
        "pixel": {
            "row": CANONICAL_PIXEL_ROW,
            "column": CANONICAL_PIXEL_COLUMN,
            "sky_elevation_deg": float(np.degrees(np.arcsin(sky[2]))),
            "sky_azimuth_deg": float(np.degrees(np.arctan2(sky[1], sky[0]))),
            "projected_center": list(project_linear(sky, **CANONICAL_RENDER)),
        },
        "incident_direction": canonical_incident_direction(),
        "target_direction": canonical_target_direction(),
        "seed_exponential_coordinates": CANONICAL_SEED_EXPONENTIAL_COORDINATES,
        "seed_discovery": CANONICAL_SEED_DISCOVERY,
    }
