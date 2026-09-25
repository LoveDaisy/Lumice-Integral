"""``dp_field.field``: batched ``D_P`` / gradient / Riemannian Hessian, the fold pre-screen, interior critical points."""

from __future__ import annotations

import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lumice_integral import optics
from lumice_integral.canonical_scene import canonical_crystal
from lumice_integral.dp_field import DPField
from lumice_integral.dp_field import field as F
from lumice_integral.s2_store import align_rotations, evaluate_fields

N = 1.31
FIXTURES = ((3, 5), (1, 3), (3, 1, 6), (1, 3, 2), (3, 5, 6, 7, 3))
SUN = np.array([0.3, -0.2, 0.9]) / np.linalg.norm([0.3, -0.2, 0.9])


@pytest.fixture(scope="module")
def fields() -> dict[tuple[int, ...], DPField]:
    return {faces: DPField.build(canonical_crystal(), faces, N) for faces in FIXTURES}


def _random_valid(faces: tuple[int, ...], count: int, seed: int) -> np.ndarray:
    points = np.random.default_rng(seed).normal(size=(400_000, 3))
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    inside = points[F.valid_batch(points, faces, N)]
    assert len(inside) >= count
    return inside[:count]


@pytest.mark.parametrize("faces", FIXTURES)
def test_d_matches_the_event_store_evaluator(fields, faces) -> None:
    """``D_P`` against :func:`.s2_store.evaluate_fields` (poses ``R u = s_hat``, ``arccos``) on 256 random points of ``U_P``.

    Points within ``1e-3`` rad of ``D = 0`` or ``pi`` are skipped: the
    ``arccos`` of the store loses ``sqrt(eps)`` there (``d_p`` is in ``atan2``
    form, a slab path in closed form).
    """
    u = _random_valid(faces, 256, seed=sum(faces))
    ours = fields[faces].d_p_batch(u)
    store = evaluate_fields(align_rotations(u, SUN), SUN, canonical_crystal(), N, [faces])
    assert store["valid"].all()
    keep = (ours > 1e-3) & (ours < np.pi - 1e-3)
    assert keep.sum() > 200
    assert np.max(np.abs(ours[keep] - store["D"][keep])) <= 1e-12


@pytest.mark.parametrize("faces", [(3, 1, 6), (1, 3, 2), (3, 5, 6, 7, 3)])
def test_slab_closed_form_equals_the_optics_chain_inside(faces) -> None:
    screen = F.fold_screen(canonical_crystal(), faces)
    assert screen.degenerate
    u = _random_valid(faces, 256, seed=7)
    chain = F.d_p_batch(u, faces, N)
    closed = F.d_p_batch(u, faces, N, screen.fold_matrix)
    assert np.max(np.abs(chain - closed)) <= 1e-12


@pytest.mark.parametrize("faces", FIXTURES)
def test_margins_are_the_gates_of_path_domain_batch(faces) -> None:
    """The differentiable margins read the same fields as the authority (:func:`.optics.path_domain_batch`)."""
    u = _random_valid(faces, 64, seed=3)
    ours = F.margins_batch(u, faces, N)
    check = optics.path_domain_batch(align_rotations(u, SUN), faces, -SUN, N)
    theirs = np.stack([check.margins[name] for name in optics.domain_margin_names(faces)], axis=1)
    assert np.max(np.abs(ours - theirs)) <= 1e-12


def test_gradient_against_central_differences_has_a_noise_plateau() -> None:
    """Scan the step: truncation error falls as ``h^2``, rounding grows as ``1/h``; the plateau is at ``<= 1e-9``."""
    faces = (3, 5)
    u0 = _random_valid(faces, 1, seed=11)[0]
    grad = F.gradient_batch(u0[None, :], faces, N)[0]
    basis = np.asarray(F.tangent_basis(jnp.asarray(u0)))
    errors = []
    for h in np.logspace(-2, -8, 13):
        for e in basis:
            plus, minus = u0 * np.cos(h) + e * np.sin(h), u0 * np.cos(h) - e * np.sin(h)
            d_plus, d_minus = F.d_p_batch(np.stack([plus, minus]), faces, N)
            errors.append((h, abs((d_plus - d_minus) / (2.0 * h) - grad @ e)))
    by_step = {}
    for h, err in errors:
        by_step[h] = max(by_step.get(h, 0.0), err)
    steps = sorted(by_step)
    best = min(by_step.values())
    assert best <= 1e-9
    # both walls of the plateau are there: the largest and the smallest step are clearly worse
    assert by_step[steps[-1]] > 100 * best and by_step[steps[0]] > 100 * best


def test_hessian_needs_the_curvature_term_at_the_3_5_minimum(fields) -> None:
    """The Riemannian Hessian does not depend on how ``D_P`` is extended off the sphere; the naive projection does.

    Two ambient extensions of the same field (``atan2`` as in ``d_p``,
    ``arccos`` as in the event store): with the ``-(u . g) I`` term both give
    ``~[0.34, 0.96]`` (a minimum); without it the ``arccos`` extension reads
    a maximum ``~[-5.4, -4.7]`` (explore ``dp-field-topology`` run #1) and the
    ``atan2`` one a saddle.
    """
    point = fields[(3, 5)].interior_critical_points[0]
    u = jnp.asarray(point.position)
    hessian, basis = fields[(3, 5)].hessian_tangent_batch(point.position[None, :])
    np.testing.assert_allclose(np.linalg.eigvalsh(hessian[0]), [0.33757484, 0.96456737], atol=1e-7)

    def d_arccos(v):
        phi = optics.path_direction(jnp.eye(3), (3, 5), -v, jnp.float64(N)).direction
        return jnp.arccos(jnp.dot(phi, -v))

    naive = {}
    for name, fn in (("atan2", lambda v: F.d_p(v, (3, 5), jnp.float64(N))), ("arccos", d_arccos)):
        h, g = np.asarray(jax.hessian(fn)(u)), np.asarray(jax.grad(fn)(u))
        corrected = basis[0] @ (h - (point.position @ g) * np.eye(3)) @ basis[0].T
        np.testing.assert_allclose(np.linalg.eigvalsh(corrected), [0.33757484, 0.96456737], atol=1e-7)
        naive[name] = np.linalg.eigvalsh(basis[0] @ h @ basis[0].T)
    assert np.all(naive["arccos"] < -4.0)
    assert naive["atan2"][0] < 0.0 < naive["atan2"][1]


def test_fold_screen_on_the_fixtures() -> None:
    """``n_a . M^T n_b``: ``-1/2`` for 3-5 and ``0`` for the 90 degree wedge (interior folds), ``-1`` for the three slabs."""
    crystal = canonical_crystal()
    dots = {faces: F.fold_screen(crystal, faces).dot for faces in FIXTURES}
    assert dots[(3, 5)] == pytest.approx(-0.5, abs=1e-15)
    assert dots[(1, 3)] == pytest.approx(0.0, abs=1e-15)
    for faces in ((3, 1, 6), (1, 3, 2), (3, 5, 6, 7, 3)):
        assert dots[faces] == -1.0
        assert F.fold_screen(crystal, faces).degenerate
    assert not F.fold_screen(crystal, (3, 5)).degenerate and not F.fold_screen(crystal, (1, 3)).degenerate


def test_fold_axis_of_a_mirror_is_its_normal() -> None:
    """A mirror's axis is the eigenvalue ``-1`` vector; asking for ``+1`` would return an arbitrary in-plane vector."""
    crystal = canonical_crystal()
    axis = F.fold_screen(crystal, (3, 1, 6)).axis
    assert abs(abs(axis @ np.array([0.0, 0.0, 1.0])) - 1.0) <= 1e-15
    rotation = F.fold_axis(np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]))
    assert abs(abs(rotation[2]) - 1.0) <= 1e-15
    assert F.fold_axis(np.eye(3)) is None


def test_rank_zero_path_is_refused() -> None:
    with pytest.raises(ValueError, match="rank 0"):
        DPField.build(canonical_crystal(), (1, 2), N)


def test_the_field_does_not_take_a_sun_direction() -> None:
    assert set(inspect.signature(DPField.build).parameters) == {"crystal", "faces", "index", "lattice_n"}


def test_3_5_minimum_deviation_is_the_only_interior_critical_point(fields) -> None:
    points = fields[(3, 5)].interior_critical_points
    assert len(points) == 1
    expected = 2.0 * np.arcsin(N * np.sin(np.radians(30.0))) - np.radians(60.0)
    assert points[0].value == pytest.approx(expected, abs=1e-12)
    assert np.degrees(points[0].value) == pytest.approx(21.8393, abs=5e-5)
    assert points[0].kind == "minimum" and points[0].morse_index == 0
    assert points[0].gradient_norm <= F.CRITICAL_GRADIENT_TOL


def test_90_degree_wedge_minimum(fields) -> None:
    points = fields[(1, 3)].interior_critical_points
    assert len(points) == 1
    expected = 2.0 * np.arcsin(N * np.sin(np.radians(45.0))) - np.radians(90.0)
    assert points[0].value == pytest.approx(expected, abs=1e-12)
    assert points[0].kind == "minimum"


@pytest.mark.parametrize("faces", [(3, 5), (1, 3)])
def test_two_lattice_densities_find_the_same_critical_points(faces) -> None:
    coarse = F.lattice_newton_critical_points(faces, N, lattice_n=2000)
    fine = F.lattice_newton_critical_points(faces, N, lattice_n=20000)
    assert len(coarse) == len(fine) == 1
    assert np.linalg.norm(np.cross(coarse[0].position, fine[0].position)) <= 1e-9
    assert coarse[0].value == pytest.approx(fine[0].value, abs=1e-13)


@pytest.mark.parametrize("faces", [(3, 1, 6), (1, 3, 2)])
def test_slab_paths_without_interior_critical_points(fields, faces) -> None:
    """``+-n_M`` sit on the entry great circle but outside the closure of ``U_P`` (an internal TIR fails there)."""
    field = fields[faces]
    assert field.interior_critical_points == ()
    fold_set = field.degenerate_fold
    assert fold_set is not None
    assert [where for _, where in fold_set.axis_points] == ["exterior", "exterior"]
    for point, _ in fold_set.axis_points:
        margins = dict(zip(optics.domain_margin_names(faces), F.margins_batch(point[None, :], faces, N)[0]))
        assert abs(margins["entry_incidence_cosine"]) <= 1e-15
        assert margins["internal_1_tir_discriminant"] < -0.2
    assert fold_set.circle_interior_fraction == 0.0


def test_liljequist_slab_has_its_mirror_axis_inside(fields) -> None:
    """``3-5-6-7-3``: ``M`` is the mirror of face 3 and ``+n_M`` (normal incidence on face 3) is inside ``U_P``: ``D = pi``.

    The only interior critical point, a degenerate (cone) maximum -- the
    non-empty ``+-n_M`` branch of the issue's parallel-face class.
    """
    field = fields[(3, 5, 6, 7, 3)]
    (point,) = field.interior_critical_points
    assert abs(abs(point.position @ np.array([1.0, 0.0, 0.0])) - 1.0) <= 1e-15
    assert point.kind == "degenerate" and point.value == pytest.approx(np.pi, abs=1e-15)
    assert [where for _, where in field.degenerate_fold.axis_points] == ["interior", "exterior"]
    chain = F.d_p_batch(point.position[None, :], (3, 5, 6, 7, 3), N)[0]
    assert chain == pytest.approx(np.pi, abs=1e-7)  # the optics chain's sqrt(eps) at the cone


@pytest.mark.parametrize("faces", [(3, 1, 6), (1, 3, 2), (3, 5, 6, 7, 3)])
def test_lattice_newton_finds_no_smooth_critical_point_on_slab_paths(faces) -> None:
    """Cross-check of skipping the search: ``|grad D| = 2`` wherever a slab field is smooth (explore ``dp-field-topology`` #6)."""
    assert F.lattice_newton_critical_points(faces, N, lattice_n=2000) == ()
    u = _random_valid(faces, 64, seed=5)
    np.testing.assert_allclose(np.linalg.norm(F.gradient_batch(u, faces, N, F.fold_screen(canonical_crystal(), faces).fold_matrix), axis=1), 2.0, atol=1e-9)
