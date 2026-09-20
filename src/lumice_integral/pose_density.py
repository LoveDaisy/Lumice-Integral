"""Column-crystal pose density relative to Haar probability measure on SO(3).

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
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

_GAUSS_LEGENDRE_NODES = 400
_WINDOW_HALF_WIDTH_SIGMAS = 12.0


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


def column_zenith_pose_density(
    rotation: np.ndarray,
    *,
    zenith_std_rad: float,
    zenith_mean_rad: float = np.pi / 2.0,
) -> float:
    """One-shot ``rho_H(R)``; build :class:`ZenithGaussianPoseDensity` for repeated use."""
    return ZenithGaussianPoseDensity(zenith_mean_rad, zenith_std_rad)(rotation)
