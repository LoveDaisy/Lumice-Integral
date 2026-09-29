"""``contour_quadrature``'s finite-crystal gate: ``gated_out`` separates "every point outside ``w``'s support" from a real zero.

The integrand carries ``w = A_P T_P`` point by point, so a level set whose
every quadrature point lies outside the finite crystal's support integrates to
exactly ``0`` with ``status == "integrated"`` (it has components); before the
flag that was indistinguishable from a cancelling or ``rho = 0`` zero and, in
``value`` alone, from ``"empty"``.  The pyramid fixture is the one of
``explore-level-set-vs-finite-crystal-gate`` (experiments.md #1): path
13-15-26-28 is gated below ``delta ~ 120.3 deg``; ``delta = 120 deg`` sits on
the gate's edge (``w`` has a kink there, panels reach ``max_depth``) and
``134.7 deg`` well inside.  The values below were probed at ``N = 2e5`` and
``1e6`` (bit-identical: the store only seeds the extraction).
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from lumice_integral import contour_quadrature as cq
from lumice_integral.canonical_scene import CANONICAL_REFRACTIVE_INDEX, canonical_crystal, canonical_pose_density, canonical_sun_direction
from lumice_integral.camera import incident_direction_from_sun
from lumice_integral.contour import extract_level_sets
from lumice_integral.dp_field import DPField
from lumice_integral.geometry.pyramid import Pyramid
from lumice_integral.pose_density import build_pose_density
from lumice_integral.s2_store import build_event_store

INDEX = CANONICAL_REFRACTIVE_INDEX
FACES = (13, 15, 26, 28)
STORE_N = 200_000
OPTIONS = cq.QuadratureOptions(relative_tolerance=1e-6)


def _centre_at(sun: np.ndarray, delta: float) -> np.ndarray:
    """An outgoing direction at deviation ``delta`` from the propagation ``-sun`` (``random`` density: azimuth is immaterial)."""
    s = incident_direction_from_sun(sun)
    e1 = np.cross(s, [0.0, 0.0, 1.0])
    e1 /= np.linalg.norm(e1)
    return np.cos(delta) * s + np.sin(delta) * e1


@pytest.fixture(scope="module")
def pyramid():
    crystal = Pyramid.from_lumice(0.5, 0.25, 0.6, (1, 0, 1), (2, 0, 3), face_distance=(1, 1.1, 0.9, 1, 1.2, 0.95))
    field = DPField.build(crystal, FACES, INDEX)
    store = build_event_store(crystal, INDEX, [FACES], STORE_N, run_checks=False)
    return field, store


@pytest.fixture(scope="module")
def pyramid_results(pyramid):
    """``delta -> (level set, geometry, result)`` for the three fixture deviations (one geometry build for all three)."""
    field, store = pyramid
    sun = canonical_sun_direction()
    deltas = np.radians([109.588, 120.0, 134.7])
    level_sets = extract_level_sets(field, deltas, store)
    geometry = cq.LevelSetGeometry.build(field, level_sets, OPTIONS)
    results = geometry.integrate(sun, [_centre_at(sun, d) for d in deltas], build_pose_density("random"))
    return geometry, dict(zip((109.588, 120.0, 134.7), zip(level_sets, results)))


def test_gated_out_below_the_gate(pyramid_results) -> None:
    """``delta = 109.588 deg``: two open arcs, every point at ``w = 0``: value exactly ``0``, integrated, gated out."""
    geometry, by_delta = pyramid_results
    level_set, result = by_delta[109.588]
    assert (level_set.n_closed, level_set.n_open) == (0, 2)
    assert result.status == "integrated" and result.gated_out
    assert result.value == 0.0 and result.raw_value == 0.0 and result.error_estimate == 0.0
    assert result.exhausted_panels == 0 and result.non_finite_points == 0
    assert geometry.gated_out.tolist() == [True, False, False]


def test_gate_edge_is_not_gated_out(pyramid_results) -> None:
    """``delta = 120 deg``: the edge of ``w``'s support; a tiny positive value, the kink exhausts panels (a lower bound only)."""
    _, by_delta = pyramid_results
    level_set, result = by_delta[120.0]
    assert (level_set.n_closed, level_set.n_open) == (0, 2)
    assert result.status == "integrated" and not result.gated_out
    assert 1e-13 < result.value < 1e-10  # probed: 3.1045635446794385e-12
    assert abs(result.value / 3.1045635446794385e-12 - 1.0) < 1e-3
    assert result.exhausted_panels >= 1  # probed: 2


def test_inside_the_gate(pyramid_results) -> None:
    """``delta = 134.7 deg``: one closed loop inside the support; an ordinary lit pixel."""
    _, by_delta = pyramid_results
    level_set, result = by_delta[134.7]
    assert (level_set.n_closed, level_set.n_open) == (1, 0)
    assert result.status == "integrated" and not result.gated_out
    assert abs(result.value / 2.25419178085628e-06 - 1.0) < 1e-3
    assert result.error_estimate < 1e-6 * result.value


def test_empty_level_set_is_not_gated_out(pyramid) -> None:
    """No component: ``"empty"``, never ``gated_out`` (the two are exclusive)."""
    field, store = pyramid
    sun = canonical_sun_direction()
    delta = float(field.interval_partition()[0].lower) - np.radians(1.0)
    (level_set,) = extract_level_sets(field, [delta], store)
    assert not level_set.components
    geometry = cq.LevelSetGeometry.build(field, [level_set], OPTIONS)
    (result,) = geometry.integrate(sun, [_centre_at(sun, delta)], build_pose_density("random"))
    assert result.status == "empty" and not result.gated_out and result.value == 0.0
    assert geometry.gated_out.tolist() == [False]


class _ZeroDensity:
    """A pose density that vanishes everywhere: a zero from ``rho``, not from the crystal."""

    def evaluate_batch(self, rotations):
        return np.zeros(len(rotations))


def test_density_zero_is_not_gated_out() -> None:
    """``rho = 0`` on a lit level set: value ``0`` but the geometry is live, so not ``gated_out``."""
    field = DPField.build(canonical_crystal(), (3, 5), INDEX)
    store = build_event_store(canonical_crystal(), INDEX, [(3, 5)], 20_000, run_checks=False)
    sun = canonical_sun_direction()
    delta = np.radians(30.0)
    (level_set,) = extract_level_sets(field, [delta], store)
    geometry = cq.LevelSetGeometry.build(field, [level_set], OPTIONS)
    centre = _centre_at(sun, delta)
    (zero,) = geometry.integrate(sun, [centre], _ZeroDensity())
    (lit,) = geometry.integrate(sun, [centre], canonical_pose_density())
    assert zero.status == "integrated" and zero.value == 0.0 and not zero.gated_out
    assert not lit.gated_out
    assert geometry.gated_out.tolist() == [False]


def test_not_integrated_is_not_gated_out() -> None:
    assert not cq.not_integrated(1.0, "critical_delta").gated_out


def test_pixel_combines_gated_out_over_its_level_sets() -> None:
    """A pixel is gated out only if every one of its level sets is; ``critical_delta`` or an empty one keeps it ``False``."""
    base = cq.not_integrated(2.0, "integrated")
    gated = dataclasses.replace(base, gated_out=True)
    empty = cq.not_integrated(2.0, "empty")
    critical = cq.not_integrated(2.0, "critical_delta")
    combine = lambda results: cq._combine(1, 2, 2.0, 0.0, np.ones(len(results)), results, 1.0)  # noqa: E731
    assert combine([gated]).gated_out
    assert combine([gated, gated]).gated_out
    assert not combine([gated, base]).gated_out
    assert not combine([gated, empty]).gated_out
    assert not combine([critical]).gated_out
    pixel = combine([gated])
    assert cq.PIXEL_CSV_COLUMNS[-1] == "gated_out"
    assert set(pixel.csv_row()) == set(cq.PIXEL_CSV_COLUMNS)
    assert pixel.csv_row()["gated_out"] is True
