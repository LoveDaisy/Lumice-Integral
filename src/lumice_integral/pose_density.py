"""Crystal pose densities relative to Haar probability measure on SO(3).

Three classes cover the five ch11 families (``docs/ch11-pose-density-families.md``,
Lumice ``src/gui/axis_presets.hpp::kAxisPresets`` read as evidence only):

- :class:`HaarUniformPoseDensity` -- ``random`` (``rho_H = 1``);
- :class:`ZenithGaussianPoseDensity` -- ``column`` (zenith mean 90 deg) and
  ``plate`` (zenith mean 0 deg), uniform azimuth and spin;
- :class:`ZenithRollGaussianPoseDensity` -- ``parry`` (zenith mean 90 deg) and
  ``lowitz`` (zenith mean 0 deg), spin about the c axis locked by a narrow
  Gaussian (Lumice ``kRollLockedGauss``).

All three share one duck-typed contract consumed by ``weights.py``: ``unit``,
``normalization``, ``__call__(rotation)`` and ``evaluate_batch(rotations)``.
:data:`PoseDensity` is the type alias naming that contract.

Model (canonical ch06 scene, ``docs/ch06-reference-fixture.md`` section 3.3):
the crystal c axis (body ``+z``, see ``geometry.core.HexPrism``) has world
direction ``n = R e3``; its zenith angle ``theta = arccos(n_z)`` follows a
Gaussian about ``zenith_mean`` with width ``zenith_std`` *as a density on the
sphere*, the azimuth of ``n`` is uniform, and the spin about the c axis is
uniform.  This is the same zonal-band model the Monte Carlo reference uses:
Lumice's sampler builds its inverse-CDF table from ``p_sphere(theta) ~
p(theta) sin(theta)`` (``doc/crystal-orientation-sampling.md`` section 4.1,
read as evidence only), and the writing project's ``halo_notes/sim/config.py::
random_axis`` documents the same area weighting.

Derivation of the Haar density (kept here so the reduction is reviewable):

1. Haar probability ``mu_Haar`` on SO(3) factorizes under
   ``R -> (n = R e3, spin about e3)`` into the uniform area measure
   ``dA(n) / (4 pi)`` on ``S2`` times the uniform spin measure
   ``d psi / (2 pi)``.  Left invariance makes the push-forward of ``mu_Haar``
   to ``S2`` rotation invariant, hence uniform; right multiplication by
   rotations about ``e3`` preserves ``mu_Haar`` (bi-invariance) and shifts the
   spin, so the conditional spin law on every fiber is uniform.
2. The model is ``dP = p(n) dA(n) . d psi / (2 pi)`` with
   ``p(n) = g(theta(n)) / Z``, ``g(theta) = exp(-(theta - mean)^2 / (2 std^2))``
   on ``theta in [0, pi]`` and ``Z = int_S2 g dA = 2 pi int_0^pi g sin theta
   d theta``.  Azimuth and spin are uniform under both laws, so their density
   ratio is one and only the zenith marginal survives.
3. Therefore ``rho_H(R) = dP / d mu_Haar = 4 pi g(theta) / Z = 2 g(theta) / I``
   with ``I = int_0^pi g(theta) sin(theta) d theta``.

``rho_H`` is dimensionless and integrates to one against ``mu_Haar``.  The
contract's ``1 / (8 pi^2)`` conversion from Haar probability to ``dVol_g``
(``docs/phase1-math-contract.md`` section 5.1) is *not* applied here; it stays
an explicit separate convention entry in ``FiberResult.conventions``.

Roll-locked families (Parry, Lowitz) multiply the same zenith factor by a spin
factor: with ``psi = c_axis_roll(R)`` the model is ``dP = p(n) dA(n) . q(psi)
d psi``, ``q = h / Q``, ``h(psi) = exp(-(psi - roll_mean)^2 / (2 roll_std^2))``
on the single period ``[roll_mean - pi, roll_mean + pi]`` and ``Q = int h d
psi`` over that period, so ``rho_H(R) = (2 g(theta) / I) . (2 pi h(psi) / Q)``.
Step 1 above already gives the spin measure ``d psi / (2 pi)`` on every fiber;
the zenith and spin factors are the two marginals of the ZYZ chain
``R = Rz(az - pi) . Ry(-zenith) . Rz(roll)`` whose Haar density is
``sin(zenith) d az d zenith d roll / (8 pi^2)``.  A wrapped (rather than
single-period truncated) Gaussian would differ only for ``roll_std``
approaching the period; the locked widths (about 1 deg) are far from that.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

_GAUSS_LEGENDRE_NODES = 400
_WINDOW_HALF_WIDTH_SIGMAS = 12.0

# Family-level zenith means (degrees), Lumice ``kAxisPresets``: the c axis is
# horizontal for column/parry and vertical for plate/lowitz.
COLUMN_ZENITH_MEAN_DEG = 90.0
PLATE_ZENITH_MEAN_DEG = 0.0
PARRY_ZENITH_MEAN_DEG = 90.0
LOWITZ_ZENITH_MEAN_DEG = 0.0
# Roll lock centre (degrees) shared by parry/lowitz (Lumice ``kRollLockedGauss`` mean).
LOCKED_ROLL_MEAN_DEG = 0.0


def c_axis_zenith(rotation: np.ndarray) -> float:
    """Zenith angle (radians, in ``[0, pi]``) of the crystal c axis ``R e3``."""
    rotation = np.asarray(rotation, dtype=np.float64)
    if rotation.shape != (3, 3):
        raise ValueError("rotation must be a (3, 3) matrix")
    return float(np.arccos(np.clip(rotation[2, 2], -1.0, 1.0)))


def c_axis_roll(rotation: np.ndarray) -> float:
    """Spin angle ``psi`` (radians, in ``(-pi, pi]``) about the crystal c axis.

    ``psi`` is the ``roll`` of the ZYZ chain ``R = Rz(az - pi) . Ry(-zenith) .
    Rz(roll)`` (Lumice ``simulator.cpp::BuildCrystalRotation``, read as
    evidence only): the third row of ``Ry(-zenith) . Rz(roll)`` is
    ``(sin(zenith) cos(roll), -sin(zenith) sin(roll), cos(zenith))`` and the
    outer ``Rz`` leaves it unchanged, so ``roll = atan2(-R[2, 1], R[2, 0])``
    whenever ``sin(zenith) > 0``.  ``roll = 0`` puts the body ``e1`` (the face-3
    outward normal of ``geometry.core.HexPrism``) in the vertical plane through
    the c axis, on the upper side; for a horizontal c axis face 3 is then the
    horizontal top face.  This reference is this renderer's own convention and
    is not tied to Lumice's mesh face numbering.

    At the gimbal-lock poles (``zenith = 0`` or ``pi``) only ``az +- roll`` is
    defined and the value returned is arbitrary; callers must not rely on it
    there (Parry/Lowitz zenith windows stay away from the poles).
    """
    rotation = np.asarray(rotation, dtype=np.float64)
    if rotation.shape != (3, 3):
        raise ValueError("rotation must be a (3, 3) matrix")
    return float(np.arctan2(-rotation[2, 1], rotation[2, 0]))


def zenith_gaussian(theta: np.ndarray, *, zenith_mean_rad: float, zenith_std_rad: float) -> np.ndarray:
    """Unnormalized sphere-density profile ``g(theta)``; zero outside ``[0, pi]``."""
    theta = np.asarray(theta, dtype=np.float64)
    inside = (theta >= 0.0) & (theta <= np.pi)
    profile = np.exp(-((theta - zenith_mean_rad) ** 2) / (2.0 * zenith_std_rad**2))
    return np.where(inside, profile, 0.0)


def zenith_marginal_integral(*, zenith_mean_rad: float, zenith_std_rad: float) -> float:
    """``I = int_0^pi g(theta) sin(theta) d theta`` by Gauss-Legendre quadrature.

    The integrand is confined to a ``+-12 sigma`` window (tail mass below
    ``exp(-72)``) clipped to ``[0, pi]``, so 400 nodes resolve it to round-off
    even for the canonical ``0.5 deg`` width.
    """
    lower = max(0.0, zenith_mean_rad - _WINDOW_HALF_WIDTH_SIGMAS * zenith_std_rad)
    upper = min(np.pi, zenith_mean_rad + _WINDOW_HALF_WIDTH_SIGMAS * zenith_std_rad)
    if not upper > lower:
        raise ValueError("zenith window does not intersect [0, pi]")
    nodes, weights = np.polynomial.legendre.leggauss(_GAUSS_LEGENDRE_NODES)
    theta = 0.5 * (upper - lower) * nodes + 0.5 * (upper + lower)
    values = zenith_gaussian(theta, zenith_mean_rad=zenith_mean_rad, zenith_std_rad=zenith_std_rad)
    return float(0.5 * (upper - lower) * np.sum(weights * values * np.sin(theta)))


@dataclass(frozen=True)
class ZenithGaussianPoseDensity:
    """``rho_H(R) = 2 g(theta(R)) / I`` with the normalization computed once."""

    zenith_mean_rad: float
    zenith_std_rad: float
    marginal_integral: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        if not np.isfinite(self.zenith_mean_rad) or not 0.0 <= self.zenith_mean_rad <= np.pi:
            raise ValueError("zenith_mean_rad must lie in [0, pi]")
        if not np.isfinite(self.zenith_std_rad) or self.zenith_std_rad <= 0.0:
            raise ValueError("zenith_std_rad must be a positive finite width")
        object.__setattr__(
            self,
            "marginal_integral",
            zenith_marginal_integral(
                zenith_mean_rad=self.zenith_mean_rad, zenith_std_rad=self.zenith_std_rad
            ),
        )

    @property
    def unit(self) -> str:
        return "dimensionless"

    @property
    def normalization(self) -> str:
        return (
            "density relative to Haar probability measure d mu_Haar on SO(3); "
            "integrates to 1; zenith-Gaussian sphere density of the c axis "
            f"(mean {np.degrees(self.zenith_mean_rad):.6g} deg, std "
            f"{np.degrees(self.zenith_std_rad):.6g} deg), uniform azimuth and spin; "
            "the 1/(8 pi^2) Haar-to-dVol_g factor is not applied"
        )

    def density_at_zenith(self, theta: np.ndarray) -> np.ndarray:
        return 2.0 * zenith_gaussian(
            theta, zenith_mean_rad=self.zenith_mean_rad, zenith_std_rad=self.zenith_std_rad
        ) / self.marginal_integral

    def __call__(self, rotation: np.ndarray) -> float:
        return float(self.density_at_zenith(c_axis_zenith(rotation)))

    def evaluate_batch(self, rotations: np.ndarray) -> np.ndarray:
        """``rho_H`` of ``(N, 3, 3)`` rotations at once (same zenith formula as ``__call__``)."""
        rotations = np.asarray(rotations, dtype=np.float64)
        if rotations.ndim != 3 or rotations.shape[1:] != (3, 3):
            raise ValueError("rotations must have shape (N, 3, 3)")
        theta = np.arccos(np.clip(rotations[:, 2, 2], -1.0, 1.0))
        return np.asarray(self.density_at_zenith(theta), dtype=np.float64)


@dataclass(frozen=True)
class HaarUniformPoseDensity:
    """``rho_H(R) = 1``: the ``random`` family (Haar-uniform orientations).

    Lumice's ``Random`` preset (zenith, azimuth and roll all uniform over
    360 deg) takes its exact ``kFullSphere`` sampling path, i.e. the uniform
    area measure on the sphere times a uniform spin -- Haar itself.
    """

    @property
    def unit(self) -> str:
        return "dimensionless"

    @property
    def normalization(self) -> str:
        return (
            "density relative to Haar probability measure d mu_Haar on SO(3); "
            "integrates to 1; Haar-uniform (random) orientations, rho_H = 1 everywhere; "
            "the 1/(8 pi^2) Haar-to-dVol_g factor is not applied"
        )

    def __call__(self, rotation: np.ndarray) -> float:
        rotation = np.asarray(rotation, dtype=np.float64)
        if rotation.shape != (3, 3):
            raise ValueError("rotation must be a (3, 3) matrix")
        return 1.0

    def evaluate_batch(self, rotations: np.ndarray) -> np.ndarray:
        rotations = np.asarray(rotations, dtype=np.float64)
        if rotations.ndim != 3 or rotations.shape[1:] != (3, 3):
            raise ValueError("rotations must have shape (N, 3, 3)")
        return np.ones(rotations.shape[0], dtype=np.float64)


def roll_gaussian(psi: np.ndarray, *, roll_mean_rad: float, roll_std_rad: float) -> np.ndarray:
    """Unnormalized spin profile ``h(psi)`` on the period ``[mean - pi, mean + pi]``.

    ``psi`` is reduced into that period first, so any representative of the
    angle (``c_axis_roll`` returns ``(-pi, pi]``) evaluates the same value.
    """
    psi = np.asarray(psi, dtype=np.float64)
    offset = (psi - roll_mean_rad + np.pi) % (2.0 * np.pi) - np.pi
    return np.exp(-(offset**2) / (2.0 * roll_std_rad**2))


def roll_marginal_integral(*, roll_mean_rad: float, roll_std_rad: float) -> float:
    """``Q = int_{mean - pi}^{mean + pi} h(psi) d psi`` by Gauss-Legendre quadrature.

    Same ``+-12 sigma`` window as :func:`zenith_marginal_integral`, clipped to
    the single period (no ``sin`` weight: the spin measure is flat).
    """
    lower = max(roll_mean_rad - np.pi, roll_mean_rad - _WINDOW_HALF_WIDTH_SIGMAS * roll_std_rad)
    upper = min(roll_mean_rad + np.pi, roll_mean_rad + _WINDOW_HALF_WIDTH_SIGMAS * roll_std_rad)
    nodes, weights = np.polynomial.legendre.leggauss(_GAUSS_LEGENDRE_NODES)
    psi = 0.5 * (upper - lower) * nodes + 0.5 * (upper + lower)
    values = roll_gaussian(psi, roll_mean_rad=roll_mean_rad, roll_std_rad=roll_std_rad)
    return float(0.5 * (upper - lower) * np.sum(weights * values))


@dataclass(frozen=True)
class ZenithRollGaussianPoseDensity:
    """``rho_H(R) = (2 g(theta) / I) . (2 pi h(psi) / Q)``: parry / lowitz.

    The zenith factor is exactly :class:`ZenithGaussianPoseDensity`'s; the
    spin factor locks ``psi = c_axis_roll(R)`` to a narrow Gaussian about
    ``roll_mean_rad`` (module docstring, last paragraph).
    """

    zenith_mean_rad: float
    zenith_std_rad: float
    roll_mean_rad: float
    roll_std_rad: float
    zenith: ZenithGaussianPoseDensity = field(init=False, repr=False, compare=False)
    roll_integral: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        object.__setattr__(self, "zenith", ZenithGaussianPoseDensity(self.zenith_mean_rad, self.zenith_std_rad))
        if not np.isfinite(self.roll_mean_rad):
            raise ValueError("roll_mean_rad must be finite")
        if not np.isfinite(self.roll_std_rad) or self.roll_std_rad <= 0.0:
            raise ValueError("roll_std_rad must be a positive finite width")
        object.__setattr__(
            self,
            "roll_integral",
            roll_marginal_integral(roll_mean_rad=self.roll_mean_rad, roll_std_rad=self.roll_std_rad),
        )

    @property
    def unit(self) -> str:
        return "dimensionless"

    @property
    def normalization(self) -> str:
        return (
            "density relative to Haar probability measure d mu_Haar on SO(3); "
            "integrates to 1; zenith-Gaussian sphere density of the c axis "
            f"(mean {np.degrees(self.zenith_mean_rad):.6g} deg, std "
            f"{np.degrees(self.zenith_std_rad):.6g} deg), uniform azimuth, spin about "
            f"the c axis Gaussian (mean {np.degrees(self.roll_mean_rad):.6g} deg, std "
            f"{np.degrees(self.roll_std_rad):.6g} deg, roll = 0 puts the body e1 axis "
            "in the vertical plane through the c axis); "
            "the 1/(8 pi^2) Haar-to-dVol_g factor is not applied"
        )

    def density_at_roll(self, psi: np.ndarray) -> np.ndarray:
        """Spin factor ``2 pi h(psi) / Q``.

        Placeholder stage (plan Step 4): held at ``1`` so that the class reduces
        to :class:`ZenithGaussianPoseDensity` while the wiring is validated.
        """
        return np.ones_like(np.asarray(psi, dtype=np.float64))

    def __call__(self, rotation: np.ndarray) -> float:
        return float(self.zenith.density_at_zenith(c_axis_zenith(rotation)) * self.density_at_roll(c_axis_roll(rotation)))

    def evaluate_batch(self, rotations: np.ndarray) -> np.ndarray:
        """``rho_H`` of ``(N, 3, 3)`` rotations (same zenith and roll formulas as ``__call__``)."""
        rotations = np.asarray(rotations, dtype=np.float64)
        if rotations.ndim != 3 or rotations.shape[1:] != (3, 3):
            raise ValueError("rotations must have shape (N, 3, 3)")
        theta = np.arccos(np.clip(rotations[:, 2, 2], -1.0, 1.0))
        psi = np.arctan2(-rotations[:, 2, 1], rotations[:, 2, 0])
        return np.asarray(self.zenith.density_at_zenith(theta) * self.density_at_roll(psi), dtype=np.float64)


# The duck-typed contract ``weights.py`` consumes (``unit``, ``normalization``,
# ``__call__``, ``evaluate_batch``); a type alias only, no runtime behaviour.
PoseDensity = HaarUniformPoseDensity | ZenithGaussianPoseDensity | ZenithRollGaussianPoseDensity


def column_zenith_pose_density(
    rotation: np.ndarray,
    *,
    zenith_std_rad: float,
    zenith_mean_rad: float = np.pi / 2.0,
) -> float:
    """One-shot ``rho_H(R)``; build :class:`ZenithGaussianPoseDensity` for repeated use."""
    return ZenithGaussianPoseDensity(zenith_mean_rad, zenith_std_rad)(rotation)
