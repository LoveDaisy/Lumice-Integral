from __future__ import annotations

import numpy as np
import pytest

from lumice_integral.pose_density import (
    ZenithGaussianPoseDensity,
    c_axis_zenith,
    column_zenith_pose_density,
    zenith_marginal_integral,
)


def haar_rotations(count: int, rng: np.random.Generator) -> np.ndarray:
    """Independent Haar sampler (uniform unit quaternions), not the theta marginal."""
    quaternion = rng.standard_normal((count, 4))
    quaternion /= np.linalg.norm(quaternion, axis=1, keepdims=True)
    w, x, y, z = quaternion.T
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
            np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
            np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
        ],
        axis=1,
    )


def test_c_axis_zenith_reads_the_world_z_component_of_the_body_axis():
    assert c_axis_zenith(np.eye(3)) == pytest.approx(0.0)
    tilt = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])  # 90 deg about x
    assert c_axis_zenith(tilt) == pytest.approx(np.pi / 2.0)
    with pytest.raises(ValueError):
        c_axis_zenith(np.eye(2))


@pytest.mark.parametrize(
    "mean_deg, std_deg, samples, tolerance_sigmas",
    [(90.0, 0.5, 2_000_000, 4.0), (90.0, 5.0, 1_000_000, 4.0), (90.0, 30.0, 500_000, 4.0), (10.0, 5.0, 1_000_000, 4.0)],
)
def test_density_integrates_to_one_against_independent_haar_samples(
    mean_deg, std_deg, samples, tolerance_sigmas
):
    """E_Haar[rho_H] = 1 within the Monte Carlo standard error (declared, not tuned)."""
    density = ZenithGaussianPoseDensity(np.radians(mean_deg), np.radians(std_deg))
    rotations = haar_rotations(samples, np.random.default_rng(20260916))
    values = density.density_at_zenith(np.arccos(np.clip(rotations[:, 2, 2], -1.0, 1.0)))
    standard_error = values.std() / np.sqrt(samples)
    assert abs(values.mean() - 1.0) <= tolerance_sigmas * standard_error
    # The estimate is informative: the error bar itself is at most a few percent.
    assert standard_error < 0.02


def test_narrow_width_matches_the_small_sigma_closed_form():
    """For sigma << 1 rad, I ~ sigma sqrt(2 pi) sin(mean), so rho(mean) ~ 2 / (sigma sqrt(2 pi))."""
    sigma = np.radians(0.5)
    density = ZenithGaussianPoseDensity(np.pi / 2.0, sigma)
    assert density.marginal_integral == pytest.approx(sigma * np.sqrt(2.0 * np.pi), rel=1e-4)
    assert density.density_at_zenith(np.pi / 2.0) == pytest.approx(2.0 / (sigma * np.sqrt(2.0 * np.pi)), rel=1e-4)
    # Two independent quadrature resolutions agree: fine trapezoid vs Gauss-Legendre.
    theta = np.linspace(np.pi / 2.0 - 12.0 * sigma, np.pi / 2.0 + 12.0 * sigma, 200_001)
    trapezoid = np.trapezoid(np.exp(-((theta - np.pi / 2.0) ** 2) / (2.0 * sigma**2)) * np.sin(theta), theta)
    assert zenith_marginal_integral(zenith_mean_rad=np.pi / 2.0, zenith_std_rad=sigma) == pytest.approx(trapezoid, rel=1e-10)


def test_narrowing_the_width_sharpens_the_peak_and_kills_the_tail():
    horizontal = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    tilted = np.array(
        [[1.0, 0.0, 0.0], [0.0, np.cos(np.radians(88.0)), -np.sin(np.radians(88.0))], [0.0, np.sin(np.radians(88.0)), np.cos(np.radians(88.0))]]
    )
    wide = ZenithGaussianPoseDensity(np.pi / 2.0, np.radians(5.0))
    narrow = ZenithGaussianPoseDensity(np.pi / 2.0, np.radians(0.5))
    assert narrow(horizontal) > wide(horizontal) > 1.0
    assert narrow(tilted) < wide(tilted)
    assert narrow(tilted) / narrow(horizontal) == pytest.approx(np.exp(-0.5 * (2.0 / 0.5) ** 2), rel=1e-9)
    assert column_zenith_pose_density(horizontal, zenith_std_rad=np.radians(0.5)) == narrow(horizontal)
    assert narrow.unit == "dimensionless"
    assert "Haar" in narrow.normalization and "8 pi^2" in narrow.normalization


def test_invalid_parameters_are_rejected():
    with pytest.raises(ValueError):
        ZenithGaussianPoseDensity(np.pi / 2.0, 0.0)
    with pytest.raises(ValueError):
        ZenithGaussianPoseDensity(-0.1, 0.1)
