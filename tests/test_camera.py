from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from lumice_integral.camera import (
    camera_rotation,
    incident_direction_from_sun,
    linear_pixel_outgoing_direction,
    linear_pixel_sky_direction,
    linear_scale,
    project_linear,
    sun_direction,
)
from lumice_integral.canonical_scene import (
    CANONICAL_PIXEL_COLUMN,
    CANONICAL_PIXEL_ROW,
    CANONICAL_RENDER,
    canonical_incident_direction,
    canonical_sun_direction,
    canonical_target_direction,
)
from lumice_integral.optics import minimum_deviation_incident, path_3_5
from lumice_integral.so3 import exp


def sky(elevation_deg: float, azimuth_deg: float) -> np.ndarray:
    elevation = np.radians(elevation_deg)
    azimuth = np.radians(azimuth_deg)
    return np.array(
        [
            np.cos(elevation) * np.cos(azimuth),
            np.cos(elevation) * np.sin(azimuth),
            np.sin(elevation),
        ]
    )


def azimuth_deg(direction: np.ndarray) -> float:
    return float(np.degrees(np.arctan2(direction[1], direction[0])))


def test_camera_rotation_is_proper_and_looks_along_the_view_direction():
    for azimuth, elevation in ((0.0, 0.0), (90.0, 0.0), (180.0, 10.0), (35.0, 60.0), (0.0, -15.0)):
        rotation = camera_rotation({"azimuth": azimuth, "elevation": elevation})
        np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-14)
        assert np.isclose(np.linalg.det(rotation), 1.0)
        np.testing.assert_allclose(rotation[:, 2], sky(elevation, azimuth), atol=1e-14)
        # Camera +y points toward the world nadir: screen v grows downward.
        assert rotation[2, 1] < 0.0


def test_linear_projection_center_scale_and_screen_handedness():
    render = CANONICAL_RENDER
    view = render["view"]
    center = project_linear(sky(view["elevation"], view["azimuth"]), **render)
    assert center == pytest.approx((render["width"] / 2.0, render["height"] / 2.0))
    assert linear_scale(6.0, 251, 801) == pytest.approx(251 / 2.0 / np.tan(np.radians(3.0)))

    # A direction 1 degree above the optical axis lands tan(1 deg) * scale pixels up.
    above = project_linear(sky(view["elevation"] + 1.0, view["azimuth"]), **render)
    assert above[0] == pytest.approx(center[0])
    assert center[1] - above[1] == pytest.approx(np.tan(np.radians(1.0)) * linear_scale(6.0, 251, 801))
    # Screen handedness: larger azimuth projects to the right (larger u).
    right = project_linear(sky(view["elevation"], view["azimuth"] + 1.0), **render)
    assert right[0] > center[0]
    # A constant-elevation step is not exactly level on a tilted pinhole frame.
    assert abs(right[1] - center[1]) < 0.2
    with pytest.raises(ValueError, match="behind"):
        project_linear(-sky(view["elevation"], view["azimuth"]), **render)


@pytest.mark.parametrize("row, column", [(0, 0), (150, 150), (400, 125), (800, 250)])
def test_pixel_center_round_trips_through_the_forward_projection(row, column):
    direction = linear_pixel_sky_direction(row, column, **CANONICAL_RENDER)
    np.testing.assert_allclose(np.linalg.norm(direction), 1.0, atol=1e-15)
    u, v = project_linear(direction, **CANONICAL_RENDER)
    assert (u, v) == pytest.approx((column + 0.5, row + 0.5), abs=1e-9)
    outgoing = linear_pixel_outgoing_direction(row, column, **CANONICAL_RENDER)
    np.testing.assert_array_equal(outgoing, -direction)


def test_sun_direction_points_toward_the_sun_and_light_travels_away_from_it():
    """``s_hat`` is the Lumice sun position vector; the solver's ``s`` is ``-s_hat`` (docs/conventions.md)."""
    sun = sun_direction(15.0, 0.0)
    np.testing.assert_allclose(sun, [np.cos(np.radians(15.0)), 0.0, np.sin(np.radians(15.0))])
    assert sun[2] > 0.0  # the sun is above the horizon
    np.testing.assert_allclose(np.linalg.norm(sun), 1.0, atol=1e-15)
    incident = incident_direction_from_sun(sun)
    assert incident[2] < 0.0  # sunlight travels downward
    for altitude, azimuth in [(0.0, 0.0), (15.0, 0.0), (22.0, 37.0), (90.0, 0.0), (-5.0, 180.0), (45.0, -90.0)]:
        sun = sun_direction(altitude, azimuth)
        incident = incident_direction_from_sun(sun)
        np.testing.assert_array_equal(incident, -sun)
        # 0.0 - x keeps a zero component +0.0, the bits every fixture was recorded with
        assert not np.any(np.signbit(incident[incident == 0.0]))
        np.testing.assert_allclose(np.degrees(np.arcsin(sun[2])), altitude, atol=1e-12)
        if abs(altitude) < 90.0:
            np.testing.assert_allclose(np.degrees(np.arctan2(sun[1], sun[0])), azimuth, atol=1e-12)


def test_canonical_pixel_direction_regression():
    np.testing.assert_allclose(
        canonical_target_direction(),
        [-0.9875255807607193, -0.010382806499944452, 0.1571159593179154],
        rtol=0.0,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        canonical_incident_direction(),
        [-0.9659258262890683, 0.0, -0.25881904510252074],
        rtol=0.0,
        atol=1e-15,
    )
    np.testing.assert_array_equal(canonical_sun_direction(), -canonical_incident_direction())


def test_path_3_5_chirality_is_the_right_hand_side_of_the_sun():
    """3-5 deviates toward +azimuth, which the linear screen maps to the right."""
    # Independent reference: the in-plane minimum-deviation fixture.
    incident = minimum_deviation_incident()
    seed = exp(jnp.array([0.15, 0.08, -0.05], dtype=jnp.float64))
    outgoing = np.asarray(path_3_5(seed, incident).direction)
    reference_offset = azimuth_deg(-outgoing) - azimuth_deg(-np.asarray(incident))
    assert 0.0 < reference_offset < 90.0

    # Canonical pixel: its sky direction must sit on the same side of the sun.
    sun_azimuth = azimuth_deg(canonical_sun_direction())
    pixel_offset = azimuth_deg(-canonical_target_direction()) - sun_azimuth
    assert np.sign(pixel_offset) == np.sign(reference_offset)
    # ... and that side is the right half of the frame.
    assert CANONICAL_PIXEL_COLUMN + 0.5 > CANONICAL_RENDER["width"] / 2.0
    u, _ = project_linear(-canonical_target_direction(), **CANONICAL_RENDER)
    assert u == pytest.approx(CANONICAL_PIXEL_COLUMN + 0.5)
    assert CANONICAL_PIXEL_ROW == 150 and CANONICAL_PIXEL_COLUMN == 150
