"""Caller-owned family fixtures; independent NumPy direction and displacement oracles.

The chain is Rz(az-pi) Ry(-zenith) Rz(roll), all parameters in radians.
No writing-repository imports or data are needed.
"""

import numpy as np
import pytest

from lumice_integral.camera import sun_direction
from lumice_integral.geometry import HexPrism, fold_matrix
from lumice_integral.optics import path_domain_batch
from lumice_integral.symmetry.signature import phi_batch


CRYSTAL = HexPrism()
SUN = sun_direction(15.0)
PATH = (3, 5)


def rz(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def ry(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def chain(parameters):
    az, zenith, roll = parameters
    return rz(az - np.pi) @ ry(-zenith) @ rz(roll)


def generators(parameters, axes):
    """Analytic body angular velocities, independently checked by displacements."""
    az, _, _ = parameters
    r = chain(parameters)
    world = (np.array([0.0, 0.0, 1.0]), rz(az - np.pi) @ np.array([0.0, -1.0, 0.0]))
    all_axes = (r.T @ world[0], r.T @ world[1], np.array([0.0, 0.0, 1.0]))
    return np.array([all_axes[k] for k in axes])


def independent_sky(rotations, sun=SUN):
    r = np.asarray(rotations).reshape(-1, 3, 3)
    incoming = -np.einsum("nji,j->ni", r, sun)
    outgoing = phi_batch(CRYSTAL, fold_matrix(CRYSTAL, PATH), *PATH, incoming)
    return -np.einsum("nij,nj->ni", r, outgoing)


def parameter_fd(parameters, axes, step=1e-5, sun=SUN):
    columns = []
    for axis in axes:
        delta = np.eye(3)[axis] * step
        pair = independent_sky([chain(parameters + delta), chain(parameters - delta)], sun)
        columns.append((pair[0] - pair[1]) / (2 * step))
    return np.array(columns)


def lowitz_valleys():
    """Original 3-deg az / 0.25-deg zenith search, including the special point.

    Keep the original endpoint inset, <= comparisons and all candidates. The
    special symmetry point is separated by coordinates, never by measured T.
    """
    zeniths = np.radians(np.arange(1.0, 179.0, 0.25))
    rows = []
    for az in np.radians(np.arange(0.0, 360.0, 3.0)):
        params = np.array([[az, z, 0.0] for z in zeniths])
        sky = independent_sky([chain(p) for p in params])
        idx = np.flatnonzero(np.isfinite(sky[:, 0]))
        if len(idx) < 20:
            continue
        inner = idx[(idx >= idx[0] + 4) & (idx <= idx[-1] - 4)]
        if len(inner) < 10:
            continue
        deviation = np.arccos(np.clip(sky[inner] @ SUN, -1, 1))
        minima = np.flatnonzero((deviation[1:-1] <= deviation[:-2]) & (deviation[1:-1] <= deviation[2:])) + 1
        for k in minima:
            p = params[inner[k]]
            if np.isfinite(parameter_fd(p, (0, 1), np.radians(0.02))).all():
                rows.append(p)
    return np.array(rows)


def test_analytic_generators_and_independent_direction():
    parameters = np.radians([[270, 90, 0], [90, 90, 115.90], [210, 90, 5.10], [282, 83, 4]])
    rotations = np.array([chain(p) for p in parameters])
    reference = independent_sky(rotations)
    domain = path_domain_batch(rotations, PATH, -SUN)
    assert domain.valid.all()
    np.testing.assert_allclose(reference, -domain.direction, atol=2e-15, rtol=0)
    for p, r in zip(parameters, rotations):
        for k, omega in enumerate(generators(p, (0, 1, 2))):
            delta = np.eye(3)[k] * 1e-5
            dr = (chain(p + delta) - chain(p - delta)) / 2e-5
            # R^T dR acts as omega cross, without using production SO(3).
            np.testing.assert_allclose(r.T @ dr, np.cross(omega, np.eye(3)).T, atol=3e-10, rtol=0)


def test_lowitz_reference_search_and_correct_speed_units():
    candidates = lowitz_valleys()
    special = np.all(np.isclose(candidates, np.radians([270, 90, 0]), atol=1e-12), axis=1)
    assert len(candidates) == 51
    assert special.sum() == 1
    general = candidates[~special]
    derivatives = np.array([parameter_fd(p, (0, 1)) for p in general])
    sky = independent_sky([chain(p) for p in general])
    area = np.abs(np.einsum("ni,ni->n", sky, np.cross(derivatives[:, 0], derivatives[:, 1])))
    assert area.min() > 1e-4
    assert np.median(area) == pytest.approx(0.040, rel=0.2)
    assert area.max() == pytest.approx(0.132, rel=0.2)
    speed = np.linalg.norm(parameter_fd(np.radians([270, 90, 0]), (0, 1)), axis=1)
    np.testing.assert_allclose(speed, [0.2771, 0.4661], atol=5e-5, rtol=0)
