from __future__ import annotations

import numpy as np
import pytest

from lumice_integral.pose_density import (
    HaarUniformPoseDensity,
    ZenithGaussianPoseDensity,
    ZenithRollGaussianPoseDensity,
    c_axis_roll,
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
    [
        (90.0, 0.5, 2_000_000, 4.0),
        (90.0, 5.0, 1_000_000, 4.0),
        (90.0, 30.0, 500_000, 4.0),
        (10.0, 5.0, 1_000_000, 4.0),
        # plate: the zenith window is clipped at the pole (theta >= 0)
        (0.0, 5.0, 1_000_000, 4.0),
    ],
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


def test_batch_density_matches_the_scalar_call_pose_by_pose():
    density = ZenithGaussianPoseDensity(np.pi / 2.0, np.radians(0.5))
    rotations = haar_rotations(3000, np.random.default_rng(5))

    batch = density.evaluate_batch(rotations)
    scalar = np.array([density(rotation) for rotation in rotations])

    assert batch.shape == (len(rotations),) and batch.dtype == np.float64
    np.testing.assert_allclose(batch, scalar, rtol=0.0, atol=1e-12)
    with pytest.raises(ValueError):
        density.evaluate_batch(np.eye(3))


def chain_rotation(az_rad: float, zenith_rad: float, roll_rad: float) -> np.ndarray:
    """``R = Rz(az - pi) . Ry(-zenith) . Rz(roll)`` written out (independent of the module)."""

    def rz(angle: float) -> np.ndarray:
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    def ry(angle: float) -> np.ndarray:
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])

    return rz(az_rad - np.pi) @ ry(-zenith_rad) @ rz(roll_rad)


def wrapped_difference(a: float, b: float) -> float:
    return float((a - b + np.pi) % (2.0 * np.pi) - np.pi)


def test_c_axis_roll_round_trips_the_chain_rotation_off_the_poles():
    """Construct (az, zenith, roll) -> R -> c_axis_roll(R) recovers roll mod 2 pi; zenith 10..170 deg."""
    for az_deg in (0.0, 37.0, 90.0, 181.0, 359.0):
        for zenith_deg in (10.0, 45.0, 90.0, 135.0, 170.0):
            for roll_deg in (-179.0, -90.0, -1.0, 0.0, 1.0, 60.0, 120.0, 179.0):
                rotation = chain_rotation(np.radians(az_deg), np.radians(zenith_deg), np.radians(roll_deg))
                assert c_axis_zenith(rotation) == pytest.approx(np.radians(zenith_deg), abs=1e-12)
                assert wrapped_difference(c_axis_roll(rotation), np.radians(roll_deg)) == pytest.approx(0.0, abs=1e-12)


def test_c_axis_roll_is_the_body_e1_elevation_reference():
    # roll = 0 with a horizontal c axis: body e1 (face-3 normal) points straight up.
    rotation = chain_rotation(np.radians(123.0), np.pi / 2.0, 0.0)
    np.testing.assert_allclose(rotation @ np.array([1.0, 0.0, 0.0]), [0.0, 0.0, 1.0], atol=1e-12)
    # A spin about the c axis by delta changes the extracted roll by delta and nothing else.
    base = chain_rotation(np.radians(20.0), np.radians(70.0), np.radians(15.0))
    axis = base @ np.array([0.0, 0.0, 1.0])
    delta = np.radians(33.0)
    k = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    spun = (np.eye(3) + np.sin(delta) * k + (1.0 - np.cos(delta)) * k @ k) @ base
    assert c_axis_zenith(spun) == pytest.approx(c_axis_zenith(base), abs=1e-12)
    assert wrapped_difference(c_axis_roll(spun), c_axis_roll(base) + delta) == pytest.approx(0.0, abs=1e-12)
    with pytest.raises(ValueError):
        c_axis_roll(np.eye(2))


def test_haar_uniform_density_is_exactly_one_everywhere():
    density = HaarUniformPoseDensity()
    rotations = haar_rotations(10_000, np.random.default_rng(7))
    batch = density.evaluate_batch(rotations)
    assert batch.shape == (10_000,) and batch.dtype == np.float64
    assert batch.mean() == 1.0 and batch.min() == 1.0 and batch.max() == 1.0
    assert all(density(rotation) == 1.0 for rotation in rotations[:50])
    assert density.unit == "dimensionless"
    assert "Haar" in density.normalization and "8 pi^2" in density.normalization
    with pytest.raises(ValueError):
        density(np.eye(2))
    with pytest.raises(ValueError):
        density.evaluate_batch(np.eye(3))


def test_plate_family_is_the_zenith_gaussian_about_the_pole():
    """mean = 0: the window is clipped at theta = 0 and I = int_0 g sin theta d theta still normalizes."""
    plate = ZenithGaussianPoseDensity(0.0, np.radians(5.0))
    theta = np.linspace(0.0, 12.0 * np.radians(5.0), 400_001)
    trapezoid = np.trapezoid(np.exp(-(theta**2) / (2.0 * np.radians(5.0) ** 2)) * np.sin(theta), theta)
    assert plate.marginal_integral == pytest.approx(trapezoid, rel=1e-9)
    assert plate(np.eye(3)) == pytest.approx(2.0 / plate.marginal_integral, rel=1e-12)
    horizontal = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    # 18 sigma from the pole: 2 exp(-162) / I, i.e. ~1e-68 -- negligible but not underflowed
    assert plate(horizontal) == pytest.approx(2.0 * np.exp(-162.0) / plate.marginal_integral, rel=1e-9)


def test_zenith_roll_density_with_a_flat_roll_factor_reduces_to_the_zenith_density():
    """Wiring baseline: the roll factor integrates to one over a flat spin, so the two
    classes agree pose-by-pose *only* when the roll factor is identically one; kept as the
    reference point of the placeholder stage (plan Step 4) and superseded by the real
    roll factor in Step 5."""
    column = ZenithGaussianPoseDensity(np.pi / 2.0, np.radians(1.0))
    parry = ZenithRollGaussianPoseDensity(np.pi / 2.0, np.radians(1.0), 0.0, np.radians(1.0))
    rotations = haar_rotations(2000, np.random.default_rng(11))
    np.testing.assert_allclose(parry.evaluate_batch(rotations), column.evaluate_batch(rotations), rtol=0.0, atol=1e-12)
    assert parry(rotations[0]) == pytest.approx(column(rotations[0]), abs=1e-12)
    assert parry.unit == "dimensionless" and "spin about the c axis Gaussian" in parry.normalization
    with pytest.raises(ValueError):
        ZenithRollGaussianPoseDensity(np.pi / 2.0, np.radians(1.0), 0.0, 0.0)
    with pytest.raises(ValueError):
        ZenithRollGaussianPoseDensity(-0.1, np.radians(1.0), 0.0, np.radians(1.0))
