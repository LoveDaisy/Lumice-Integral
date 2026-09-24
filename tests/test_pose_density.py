from __future__ import annotations

import numpy as np
import pytest

from lumice_integral.canonical_scene import (
    CANONICAL_ZENITH_MEAN_DEG,
    CANONICAL_ZENITH_STD_DEG,
    canonical_fixture_metadata,
    canonical_pose_density,
)
from lumice_integral.pose_density import (
    POSE_DENSITY_FAMILIES,
    HaarUniformPoseDensity,
    ZenithGaussianPoseDensity,
    ZenithRollGaussianPoseDensity,
    build_pose_density,
    c_axis_roll,
    c_axis_zenith,
    column_zenith_pose_density,
    resolve_pose_density_parameters,
    zenith_marginal_integral,
)
from lumice_integral.pose_density_provenance import pose_density_provenance
from lumice_integral.strip_io import scene_block


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


def test_zenith_roll_density_factorizes_into_the_zenith_density_times_a_unit_mean_spin_factor():
    """The zenith factor is the column/plate density verbatim; the spin factor averages to
    one over a flat spin (so the placeholder stage, roll factor == 1, was the right wiring
    baseline: both classes then agreed pose by pose)."""
    column = ZenithGaussianPoseDensity(np.pi / 2.0, np.radians(1.0))
    parry = ZenithRollGaussianPoseDensity(np.pi / 2.0, np.radians(1.0), 0.0, np.radians(1.0))
    rotations = haar_rotations(2000, np.random.default_rng(11))
    psi = np.array([c_axis_roll(rotation) for rotation in rotations])
    expected = column.evaluate_batch(rotations) * parry.density_at_roll(psi)
    np.testing.assert_allclose(parry.evaluate_batch(rotations), expected, rtol=1e-12, atol=0.0)
    assert parry(rotations[0]) == pytest.approx(expected[0], rel=1e-12)
    psi_grid = np.linspace(-np.pi, np.pi, 2_000_001)
    assert np.trapezoid(parry.density_at_roll(psi_grid), psi_grid) / (2.0 * np.pi) == pytest.approx(1.0, rel=1e-9)
    assert parry.unit == "dimensionless" and "spin about the c axis Gaussian" in parry.normalization
    with pytest.raises(ValueError):
        ZenithRollGaussianPoseDensity(np.pi / 2.0, np.radians(1.0), 0.0, 0.0)
    with pytest.raises(ValueError):
        ZenithRollGaussianPoseDensity(-0.1, np.radians(1.0), 0.0, np.radians(1.0))


def haar_expectation(density, samples: int, seed: int, chunk: int = 1_000_000) -> tuple[float, float]:
    """Monte Carlo ``E_Haar[rho]`` and its standard error, sampled in chunks."""
    rng = np.random.default_rng(seed)
    total = 0.0
    total_squares = 0.0
    remaining = samples
    while remaining > 0:
        values = density.evaluate_batch(haar_rotations(min(chunk, remaining), rng))
        total += float(values.sum())
        total_squares += float(np.square(values).sum())
        remaining -= len(values)
    mean = total / samples
    variance = total_squares / samples - mean * mean
    return mean, float(np.sqrt(variance / samples))


@pytest.mark.parametrize(
    "family, zenith_mean_deg, zenith_std_deg, roll_std_deg, samples, standard_error_bound",
    [
        ("parry", 90.0, 1.0, 1.0, 4_000_000, 0.05),  # var ~ sqrt(pi)/sigma_roll . 1/(sqrt(pi) sigma_zen) ~ 3.3e3
        ("lowitz", 0.0, 40.0, 1.0, 1_000_000, 0.02),
    ],
)
def test_roll_locked_density_integrates_to_one_against_independent_haar_samples(
    family, zenith_mean_deg, zenith_std_deg, roll_std_deg, samples, standard_error_bound
):
    density = ZenithRollGaussianPoseDensity(
        np.radians(zenith_mean_deg), np.radians(zenith_std_deg), 0.0, np.radians(roll_std_deg)
    )
    mean, standard_error = haar_expectation(density, samples, 20260920)
    assert abs(mean - 1.0) <= 4.0 * standard_error, family
    assert standard_error < standard_error_bound, family


def haar_integral_by_scipy(density) -> float:
    """``int rho d mu_Haar`` on the ZYZ chart: ``sin(beta) d alpha d beta d gamma / (8 pi^2)``.

    Independent of the classes' own Gauss-Legendre normalisation (adaptive
    QUADPACK through ``scipy.integrate.nquad``); the azimuth integral is the
    factor ``2 pi`` since no family depends on it.
    """
    from scipy.integrate import nquad

    def integrand(gamma: float, beta: float) -> float:
        return density(chain_rotation(0.3, beta, gamma)) * np.sin(beta) / (4.0 * np.pi)

    breakpoints = {"points": [np.radians(d) for d in (0.5, 1.0, 5.0, 45.0, 90.0, 135.0, 175.0, 179.0, 179.5)], "limit": 200}
    value, _ = nquad(integrand, [[-np.pi, np.pi], [0.0, np.pi]], opts=[breakpoints, breakpoints])
    return float(value)


@pytest.mark.parametrize(
    "density",
    [
        HaarUniformPoseDensity(),
        ZenithGaussianPoseDensity(np.pi / 2.0, np.radians(0.5)),  # column
        ZenithGaussianPoseDensity(0.0, np.radians(0.5)),  # plate
        ZenithRollGaussianPoseDensity(np.pi / 2.0, np.radians(1.0), 0.0, np.radians(1.0)),  # parry
        ZenithRollGaussianPoseDensity(0.0, np.radians(40.0), 0.0, np.radians(1.0)),  # lowitz
        ZenithRollGaussianPoseDensity(np.pi / 2.0, np.radians(1.0), np.radians(10.0), np.radians(60.0)),  # wide roll
    ],
    ids=["random", "column", "plate", "parry", "lowitz", "wide-roll"],
)
def test_every_family_integrates_to_one_by_independent_scipy_quadrature(density):
    assert haar_integral_by_scipy(density) == pytest.approx(1.0, abs=1e-6)


def test_narrowing_the_roll_width_sharpens_the_locked_peak_and_kills_the_tail():
    locked = chain_rotation(np.radians(40.0), np.pi / 2.0, 0.0)
    spun = chain_rotation(np.radians(40.0), np.pi / 2.0, np.radians(2.0))
    wide = ZenithRollGaussianPoseDensity(np.pi / 2.0, np.radians(1.0), 0.0, np.radians(5.0))
    narrow = ZenithRollGaussianPoseDensity(np.pi / 2.0, np.radians(1.0), 0.0, np.radians(0.5))
    column = ZenithGaussianPoseDensity(np.pi / 2.0, np.radians(1.0))
    assert narrow(locked) > wide(locked) > column(locked)
    assert narrow(spun) < wide(spun)
    assert narrow(spun) / narrow(locked) == pytest.approx(np.exp(-0.5 * (2.0 / 0.5) ** 2), rel=1e-9)
    # the roll factor is periodic: psi and psi + 2 pi (and the (-pi, pi] representative) agree
    assert narrow.density_at_roll(np.radians(359.0)) == pytest.approx(narrow.density_at_roll(np.radians(-1.0)), rel=1e-12)
    # wide locks (60 deg) are outside the Parry/Lowitz scenario; only "runs and stays normalised" is promised
    ZenithRollGaussianPoseDensity(np.pi / 2.0, np.radians(1.0), 0.0, np.radians(60.0))


def test_roll_locked_batch_density_matches_the_scalar_call_pose_by_pose():
    density = ZenithRollGaussianPoseDensity(np.radians(0.0), np.radians(40.0), 0.0, np.radians(1.0))
    rotations = haar_rotations(3000, np.random.default_rng(5))
    batch = density.evaluate_batch(rotations)
    scalar = np.array([density(rotation) for rotation in rotations])
    assert batch.shape == (len(rotations),) and batch.dtype == np.float64
    np.testing.assert_allclose(batch, scalar, rtol=1e-12, atol=0.0)
    with pytest.raises(ValueError):
        density.evaluate_batch(np.eye(3))


# ---------------------------------------------------------------------------
# axis-zenith interface (task band-sum-scatter-renderer): evaluate_batch is the oracle

AXIS_ZENITH_FAMILIES = {
    "random": {},
    "plate": {"zenith_std_deg": 1.0},
    "column": {"zenith_std_deg": 0.5},
    "parry": {"zenith_std_deg": 1.0, "roll_std_deg": 1.0},
    "lowitz": {"zenith_std_deg": 1.0, "roll_std_deg": 1.0},
}


def near_mode_rotations(density, count: int, rng: np.random.Generator) -> np.ndarray:
    """Poses within a few widths of the family's mode (the narrow densities vanish on most Haar poses)."""
    mean = getattr(density, "zenith_mean_rad", np.pi / 2.0)
    zenith = np.clip(mean + np.radians(3.0) * rng.standard_normal(count), 1e-3, np.pi - 1e-3)
    roll = getattr(density, "roll_mean_rad", 0.0) + np.radians(3.0) * rng.standard_normal(count)
    az = rng.uniform(0.0, 2.0 * np.pi, count)
    return np.stack([chain_rotation(a, z, r) for a, z, r in zip(az, zenith, roll)])


@pytest.mark.parametrize("family", sorted(AXIS_ZENITH_FAMILIES))
def test_axis_zenith_interface_matches_evaluate_batch_pose_by_pose(family):
    density = build_pose_density(family, **AXIS_ZENITH_FAMILIES[family])
    rng = np.random.default_rng(22)
    rotations = np.concatenate([haar_rotations(4000, rng), near_mode_rotations(density, 4000, rng)])
    oracle = density.evaluate_batch(rotations)
    components = {name: rotations[:, 2, index] for index, name in enumerate(("e1", "e2", "e3")) if name in density.axis_zeniths}
    rho = np.broadcast_to(density.evaluate_axis_zeniths(**components), oracle.shape)
    assert (oracle > 0.0).sum() > 1000
    np.testing.assert_allclose(rho, oracle, rtol=1e-13, atol=0.0)
    # (M, K) grids evaluate element by element (the renderer's pixel x event blocks)
    grid = {name: value.reshape(80, 100) for name, value in components.items()}
    np.testing.assert_array_equal(np.broadcast_to(density.evaluate_axis_zeniths(**grid), (80, 100)), rho.reshape(80, 100))


def test_axis_zenith_interface_declares_and_requires_its_components():
    assert build_pose_density("random").axis_zeniths == ()
    assert build_pose_density("column", zenith_std_deg=0.5).axis_zeniths == ("e3",)
    assert build_pose_density("parry", zenith_std_deg=1.0, roll_std_deg=1.0).axis_zeniths == ("e1", "e2", "e3")
    with pytest.raises(ValueError, match="e3"):
        build_pose_density("plate", zenith_std_deg=1.0).evaluate_axis_zeniths()
    with pytest.raises(ValueError, match="e1, e2 and e3"):
        build_pose_density("lowitz", zenith_std_deg=1.0, roll_std_deg=1.0).evaluate_axis_zeniths(e3=np.ones(3))
    # a class attribute, not a dataclass field: construction, equality and repr are unchanged
    assert HaarUniformPoseDensity() == HaarUniformPoseDensity() and repr(HaarUniformPoseDensity()) == "HaarUniformPoseDensity()"


# ---------------------------------------------------------------------------
# family enumeration, factory and provenance (plan Step 6)


def test_the_family_enumeration_is_the_five_ch11_families():
    assert POSE_DENSITY_FAMILIES == ("random", "plate", "column", "parry", "lowitz")


def test_factory_dispatches_each_family_to_its_class_with_the_family_defaults():
    random = build_pose_density("random")
    assert isinstance(random, HaarUniformPoseDensity)

    plate = build_pose_density("plate", zenith_std_deg=0.5)
    assert isinstance(plate, ZenithGaussianPoseDensity)
    assert plate.zenith_mean_rad == 0.0 and plate.zenith_std_rad == np.radians(0.5)

    column = build_pose_density("column", zenith_std_deg=0.5)
    assert isinstance(column, ZenithGaussianPoseDensity)
    assert column.zenith_mean_rad == np.radians(90.0)
    assert column == ZenithGaussianPoseDensity(np.radians(90.0), np.radians(0.5))

    parry = build_pose_density("parry", zenith_std_deg=1.0, roll_std_deg=1.0)
    assert isinstance(parry, ZenithRollGaussianPoseDensity)
    assert parry.zenith_mean_rad == np.radians(90.0) and parry.zenith_std_rad == np.radians(1.0)
    assert parry.roll_mean_rad == 0.0 and parry.roll_std_rad == np.radians(1.0)

    lowitz = build_pose_density("lowitz", zenith_std_deg=40.0, roll_std_deg=1.0)
    assert isinstance(lowitz, ZenithRollGaussianPoseDensity)
    assert lowitz.zenith_mean_rad == 0.0 and lowitz.zenith_std_rad == np.radians(40.0)

    # explicit means override the family defaults
    tilted = build_pose_density("parry", zenith_mean_deg=80.0, zenith_std_deg=1.0, roll_mean_deg=5.0, roll_std_deg=2.0)
    assert tilted.zenith_mean_rad == np.radians(80.0) and tilted.roll_mean_rad == np.radians(5.0)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(family="column"), "requires zenith_std_deg"),
        (dict(family="plate"), "requires zenith_std_deg"),
        (dict(family="parry", zenith_std_deg=1.0), "requires roll_std_deg"),
        (dict(family="lowitz", roll_std_deg=1.0), "requires zenith_std_deg"),
        (dict(family="column", zenith_std_deg=0.5, roll_std_deg=1.0), "takes no roll_std_deg"),
        (dict(family="random", zenith_std_deg=0.5), "takes no zenith_std_deg"),
        (dict(family="custom"), "unknown pose density family"),
    ],
)
def test_factory_names_the_missing_or_unused_parameter(kwargs, message):
    with pytest.raises(ValueError, match=message):
        build_pose_density(**kwargs)
    with pytest.raises(ValueError, match=message):
        pose_density_provenance(**kwargs)


def test_canonical_pose_density_is_the_column_family_at_the_canonical_width():
    density = canonical_pose_density()
    assert density == ZenithGaussianPoseDensity(np.radians(CANONICAL_ZENITH_MEAN_DEG), np.radians(CANONICAL_ZENITH_STD_DEG))


# Golden 1: the flat literal ``canonical_scene.canonical_fixture_metadata`` wrote
# before the families existed (copied from the source at commit 102c904), plus the
# new trailing ``family`` key; the three historical keys keep their order and values.
CANONICAL_SCENE_POSE_DENSITY_GOLDEN = {
    "model": "zenith-gaussian column",
    "zenith_mean_deg": 90.0,
    "zenith_std_deg": 0.5,
}
# Golden 2: the wrapped literal ``strip_io.scene_block`` wrote (same commit).
STRIP_IO_POSE_DENSITY_GOLDEN = {
    "value": {
        "model": "zenith-gaussian column",
        "zenith_mean_deg": 90.0,
        "zenith_std_deg": 0.5,
    },
    "provenance": "canonical-new",
}


def test_column_provenance_keeps_the_canonical_scene_literal_and_appends_the_family():
    block = canonical_fixture_metadata()["pose_density"]
    assert block == {**CANONICAL_SCENE_POSE_DENSITY_GOLDEN, "family": "column"}
    assert list(block) == [*CANONICAL_SCENE_POSE_DENSITY_GOLDEN, "family"]
    assert block == pose_density_provenance("column", zenith_mean_deg=90.0, zenith_std_deg=0.5)


def test_column_provenance_keeps_the_strip_io_literal_and_appends_the_family():
    block = scene_block()["pose_density"]
    assert block == {"value": {**STRIP_IO_POSE_DENSITY_GOLDEN["value"], "family": "column"}, "provenance": "canonical-new"}
    assert list(block) == ["value", "provenance"]
    assert list(block["value"]) == [*STRIP_IO_POSE_DENSITY_GOLDEN["value"], "family"]


def test_provenance_blocks_of_the_other_families_carry_only_their_own_parameters():
    assert pose_density_provenance("random") == {"model": "haar-uniform random", "family": "random"}
    assert pose_density_provenance("plate", zenith_std_deg=0.5) == {
        "model": "zenith-gaussian plate",
        "zenith_mean_deg": 0.0,
        "zenith_std_deg": 0.5,
        "family": "plate",
    }
    parry = pose_density_provenance("parry", zenith_std_deg=1.0, roll_std_deg=1.0)
    assert parry == {
        "model": "zenith-roll-gaussian parry",
        "zenith_mean_deg": 90.0,
        "zenith_std_deg": 1.0,
        "roll_mean_deg": 0.0,
        "roll_std_deg": 1.0,
        "family": "parry",
    }
    assert list(parry) == ["model", "zenith_mean_deg", "zenith_std_deg", "roll_mean_deg", "roll_std_deg", "family"]
    lowitz = pose_density_provenance("lowitz", zenith_std_deg=40.0, roll_std_deg=1.0)
    assert lowitz["model"] == "zenith-roll-gaussian lowitz" and lowitz["zenith_mean_deg"] == 0.0
    # provenance and factory resolve the same parameter set: rebuilding from the block round-trips
    for family, kwargs in [("parry", dict(zenith_std_deg=1.0, roll_std_deg=1.0)), ("lowitz", dict(zenith_std_deg=40.0, roll_std_deg=1.0)), ("plate", dict(zenith_std_deg=0.5))]:
        block = pose_density_provenance(family, **kwargs)
        parameters = {key: value for key, value in block.items() if key not in ("model", "family")}
        assert parameters == resolve_pose_density_parameters(family, **kwargs)
        assert build_pose_density(family, **parameters) == build_pose_density(family, **kwargs)
