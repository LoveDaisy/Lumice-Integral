"""The deviation field ``D_P`` on ``S^2``: evaluation, fold pre-screen, interior critical points.

``u`` is the body-frame direction toward the sun (``docs/conventions.md``),
the incoming ray is ``-u`` and ``Phi_P(-u)`` is the outgoing direction of
the fixed face sequence ``P`` at the identity pose (``R = I``: the field is
a function on the body-frame sphere and does not see the sun direction).
``D_P(u) = angle(Phi_P(-u), -u)``, the deviation formula of
:func:`.s2_store.evaluate_fields` without the pose round trip.  ``U_P`` is
the open set where every margin of :func:`.optics.domain_margin_names` is
positive; the margins are read off the same :class:`.optics.PathEvaluation`
that :func:`.optics.path_domain_batch` reads (no second derivation of the
gates).  Outside ``U_P`` the smooth branch keeps evaluating (reflections have
no square root, the exit refraction goes ``NaN`` beyond its Snell limit), so
every root finder here re-checks membership after converging.

Riemannian Hessian: the Hessian of ``D_P`` restricted to the unit sphere at
``u`` is ``P (H - (u . g) I) P`` with ``H``, ``g`` the ambient Hessian and
gradient and ``P`` the tangent projector.  Dropping the ``(u . g)`` term
(the sphere's second fundamental form) flips the signs of the eigenvalues at
the 3-5 minimum-deviation point (explore ``dp-field-topology`` run #1:
``[-5.4, -4.7]`` naive, ``[+0.34, +0.96]`` corrected).

Fold pre-screen (explore ``dp-field-topology`` H6): with ``n_a`` the entry
normal and ``n~_b = M^T n_b`` the unfolded exit normal (``M`` the fold
matrix), ``|n_a . n~_b| = 1`` means the entry and exit refractions cancel
(a slab), ``Phi_P(-u) = -M u`` and ``D_P(u) = arccos(u^T M u)``.  Its
critical set is fixed by ``M`` alone: the axis ``n_M`` (eigenvalue ``+1`` of
a rotation, ``-1`` of a mirror) and the great circle ``u . n_M = 0`` (the
double eigenvalue).  Such a path has no interior fold unless that set meets
the interior of ``U_P``; the lattice Newton search is skipped for it.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np

from .. import optics
from ..geometry import Polyhedron, fold_matrix
from ..s2_store import align_rotations, fibonacci_sphere

Faces = tuple[int, ...]

# |n_a . n~_b| within this of 1 is a slab (the dot is exactly -1.0 in float64 on the fixtures; the
# nearest non-slab value on the prism is cos 60 deg = 0.5, so the tolerance is not a tuning knob).
FOLD_DOT_ATOL = 1e-9
# A point is on dU_P rather than inside when its smallest margin is below this.
BOUNDARY_MARGIN_ATOL = 1e-10
# Tangent-gradient norm below which a Newton iterate is a critical point.
CRITICAL_GRADIENT_TOL = 1e-10
# Hessian eigenvalue magnitude below which a critical point is degenerate.
DEGENERATE_EIGENVALUE_TOL = 1e-6
# Two critical points closer than this (rad) are one.
CRITICAL_POINT_MERGE_RAD = 1e-6
# Any fixed direction off the sun axis works for pulling margins through path_domain_batch: margins are
# pose invariant, the rotation only adds rounding.
_PROBE_SUN = np.array([0.0, 0.0, 1.0])


# ---- scalar field and margins (JAX, differentiable) ------------------------------------------------


def _evaluation(u: jax.Array, faces: Faces, index: jax.Array) -> optics.PathEvaluation:
    return optics.path_direction(jnp.eye(3, dtype=u.dtype), faces, -u, index)


def d_p(u: jax.Array, faces: Faces, index: jax.Array) -> jax.Array:
    """``D_P(u)`` in radians for one unit ``u`` (module docstring).

    The angle is taken in ``atan2`` form: ``arccos`` of the dot product (the
    form of :func:`.s2_store.evaluate_fields`, equal to ``1e-15`` elsewhere)
    loses ``sqrt(eps) ~ 1e-8`` rad next to ``D = 0`` and ``D = pi``, which a
    slab path reaches on a whole boundary arc (``1-3-2``) or at an interior
    point (``3-5-6-7-3``).
    """
    phi = _evaluation(u, faces, index).direction
    return jnp.arctan2(jnp.linalg.norm(jnp.cross(phi, -u)), jnp.dot(phi, -u))


def d_slab(u: jax.Array, m: jax.Array) -> jax.Array:
    """``D_P(u) = angle(M u, u)`` of a degenerate-fold (slab) path, ``M`` its fold matrix (module docstring).

    Equal to :func:`d_p` on the closure of ``U_P`` (entry and exit refraction
    cancel exactly there); used for slab paths because it is exact on the
    crease ``u . n_M = 0`` and at ``+-n_M``, where the optics chain loses
    ``sqrt(eps)`` and, on a crease that is also the entry circle, goes
    ``NaN`` through the exit square root of a margin that only touches zero.
    """
    mu = m @ u
    return jnp.arctan2(jnp.linalg.norm(jnp.cross(mu, u)), jnp.dot(mu, u))


def margin_vector(u: jax.Array, faces: Faces, index: jax.Array) -> jax.Array:
    """Every margin of :func:`.optics.domain_margin_names` at ``u``, in that order (positive inside ``U_P``)."""
    evaluation = _evaluation(u, faces, index)
    values = [evaluation.entry.incidence_cosine, evaluation.entry.discriminant]
    for reflection in evaluation.internal:
        values.extend((reflection.incidence_cosine, reflection.tir_discriminant))
    values.extend((evaluation.exit.incidence_cosine, evaluation.exit.discriminant))
    return jnp.stack(values)


def tangent_basis(u: jax.Array) -> jax.Array:
    """An orthonormal tangent basis ``(e1, e2)`` at unit ``u``, shape ``(2, 3)`` (cross with the least aligned axis)."""
    axis = jax.nn.one_hot(jnp.argmin(jnp.abs(u)), 3, dtype=u.dtype)
    e1 = jnp.cross(u, axis)
    e1 = e1 / jnp.linalg.norm(e1)
    return jnp.stack([e1, jnp.cross(u, e1)])


def d_value(u: jax.Array, faces: Faces, index: jax.Array, slab: jax.Array | None) -> jax.Array:
    """The field as evaluated by this package: :func:`d_slab` for a slab path (``slab`` its fold matrix), :func:`d_p` otherwise."""
    return d_p(u, faces, index) if slab is None else d_slab(u, slab)


def _tangent_gradient(u: jax.Array, faces: Faces, index: jax.Array, slab: jax.Array | None) -> jax.Array:
    g = jax.grad(d_value)(u, faces, index, slab)
    return g - jnp.dot(g, u) * u


def _tangent_hessian(u: jax.Array, faces: Faces, index: jax.Array, slab: jax.Array | None) -> tuple[jax.Array, jax.Array]:
    """Riemannian Hessian in :func:`tangent_basis` coordinates and that basis (module docstring)."""
    g = jax.grad(d_value)(u, faces, index, slab)
    h = jax.hessian(d_value)(u, faces, index, slab)
    basis = tangent_basis(u)
    return basis @ (h - jnp.dot(u, g) * jnp.eye(3, dtype=u.dtype)) @ basis.T, basis


@partial(jax.jit, static_argnums=1)
def _d_batch(u: jax.Array, faces: Faces, index: jax.Array, slab: jax.Array | None) -> jax.Array:
    return jax.vmap(d_value, in_axes=(0, None, None, None))(u, faces, index, slab)


@partial(jax.jit, static_argnums=1)
def _gradient_batch(u: jax.Array, faces: Faces, index: jax.Array, slab: jax.Array | None) -> jax.Array:
    return jax.vmap(_tangent_gradient, in_axes=(0, None, None, None))(u, faces, index, slab)


@partial(jax.jit, static_argnums=1)
def _hessian_batch(u: jax.Array, faces: Faces, index: jax.Array, slab: jax.Array | None) -> tuple[jax.Array, jax.Array]:
    return jax.vmap(_tangent_hessian, in_axes=(0, None, None, None))(u, faces, index, slab)


@partial(jax.jit, static_argnums=1)
def _margins_batch(u: jax.Array, faces: Faces, index: jax.Array) -> jax.Array:
    return jax.vmap(margin_vector, in_axes=(0, None, None))(u, faces, index)


def _as_points(u: np.ndarray | jax.Array) -> jax.Array:
    points = jnp.asarray(u, dtype=jnp.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("u must have shape (N, 3)")
    return points


def _as_slab(slab: np.ndarray | None) -> jax.Array | None:
    return None if slab is None else jnp.asarray(slab, dtype=jnp.float64)


def d_p_batch(u: np.ndarray, faces: Faces, index: float, slab: np.ndarray | None = None) -> np.ndarray:
    """``D_P`` at each row of ``u`` (``(N, 3)`` unit vectors), one ``jax.vmap``; meaningful only on the closure of ``U_P``.

    ``slab`` (the fold matrix of a degenerate-fold path) selects :func:`d_slab`.
    """
    return np.asarray(_d_batch(_as_points(u), faces, jnp.float64(index), _as_slab(slab)))


def gradient_batch(u: np.ndarray, faces: Faces, index: float, slab: np.ndarray | None = None) -> np.ndarray:
    """The tangent (``S^2``) gradient of ``D_P`` at each row of ``u``, ``(N, 3)`` ambient vectors orthogonal to ``u``."""
    return np.asarray(_gradient_batch(_as_points(u), faces, jnp.float64(index), _as_slab(slab)))


def hessian_tangent_batch(
    u: np.ndarray, faces: Faces, index: float, slab: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """The Riemannian Hessian of ``D_P`` at each row of ``u``: ``(N, 2, 2)`` in the basis returned alongside, ``(N, 2, 3)``."""
    hessian, basis = _hessian_batch(_as_points(u), faces, jnp.float64(index), _as_slab(slab))
    return np.asarray(hessian), np.asarray(basis)


def margins_batch(u: np.ndarray, faces: Faces, index: float) -> np.ndarray:
    """:func:`margin_vector` at each row of ``u``, ``(N, len(domain_margin_names(faces)))``."""
    return np.asarray(_margins_batch(_as_points(u), faces, jnp.float64(index)))


def valid_batch(u: np.ndarray, faces: Faces, index: float) -> np.ndarray:
    """``u in U_P`` for each row, decided by :func:`.optics.path_domain_batch` (the single authority of the gates)."""
    u = np.asarray(u, dtype=np.float64)
    rotations = align_rotations(u, _PROBE_SUN)
    return optics.path_domain_batch(rotations, faces, -_PROBE_SUN, index).valid


# ---- fold pre-screen ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class FoldScreen:
    """The fold pre-screen of a face sequence (module docstring).

    ``dot`` is ``n_a . n~_b``; ``degenerate`` is ``|dot| = 1`` (a slab,
    ``D_P = arccos(u^T M u)``); ``axis`` is ``n_M`` (``None`` for ``M = I``).
    """

    dot: float
    degenerate: bool
    fold_matrix: np.ndarray
    axis: np.ndarray | None


def fold_screen(crystal: Polyhedron, faces: Faces) -> FoldScreen:
    """``n_a . M^T n_b`` with the body normals of :data:`.optics.HEXPRISM_BODY_NORMALS` and ``M`` from :func:`.geometry.fold_matrix`."""
    m = fold_matrix(crystal, faces)
    n_a = np.asarray(optics.HEXPRISM_BODY_NORMALS[faces[0]])
    n_tilde_b = m.T @ np.asarray(optics.HEXPRISM_BODY_NORMALS[faces[-1]])
    dot = float(n_a @ n_tilde_b)
    return FoldScreen(dot, abs(abs(dot) - 1.0) <= FOLD_DOT_ATOL, m, fold_axis(m))


def fold_axis(m: np.ndarray) -> np.ndarray | None:
    """The fixed axis ``n_M`` of an orthogonal ``M``: eigenvalue ``+1`` if ``det M > 0``, ``-1`` if a mirror.

    A mirror has a double eigenvalue ``+1``; asking it for ``+1`` returns an
    arbitrary vector in the mirror plane (explore ``dp-field-boundary-corners``
    insight 3).  ``None`` for ``M = I`` (no axis).
    """
    if np.allclose(m, np.eye(3), atol=1e-12):
        return None
    target = 1.0 if np.linalg.det(m) > 0.0 else -1.0
    eigenvalues, eigenvectors = np.linalg.eig(m)
    k = int(np.argmin(np.abs(eigenvalues - target)))
    axis = np.real(eigenvectors[:, k])
    return axis / np.linalg.norm(axis)


# ---- interior critical points ------------------------------------------------------------------------


@dataclass(frozen=True)
class InteriorCriticalPoint:
    """A zero of the tangent gradient of ``D_P`` inside ``U_P``.

    ``kind`` is ``"minimum"``, ``"maximum"``, ``"saddle"`` or ``"degenerate"``
    (a Hessian eigenvalue within ``DEGENERATE_EIGENVALUE_TOL`` of 0, or a
    point of the slab critical set, where ``arccos`` is not smooth);
    ``morse_index`` is the number of negative eigenvalues, ``None`` when
    degenerate.  ``value`` is ``D_P`` in radians.
    """

    position: np.ndarray
    value: float
    hessian_eigenvalues: np.ndarray
    kind: str
    morse_index: int | None
    gradient_norm: float


@dataclass(frozen=True)
class DegenerateFoldSet:
    """The slab critical set of a degenerate-fold path and where it lies relative to ``U_P``.

    ``axis_points`` are ``+-n_M`` with their location ``"interior"``,
    ``"boundary"`` or ``"exterior"``; ``circle_interior_fraction`` is the
    fraction of a dense sampling of the great circle ``u . n_M = 0`` inside
    ``U_P`` (0 on every fixture: the circle is a grazing boundary).
    """

    axis: np.ndarray | None
    axis_points: tuple[tuple[np.ndarray, str], ...]
    circle_interior_fraction: float


def location(u: np.ndarray, faces: Faces, index: float) -> str:
    """``"interior"``, ``"boundary"`` (smallest margin ``<= BOUNDARY_MARGIN_ATOL`` in size) or ``"exterior"``."""
    margins = margins_batch(np.asarray(u, dtype=np.float64)[None, :], faces, index)[0]
    smallest = float(np.min(margins))
    if not np.isfinite(smallest):
        return "exterior"
    if abs(smallest) <= BOUNDARY_MARGIN_ATOL:
        return "boundary"
    return "interior" if smallest > 0.0 else "exterior"


def degenerate_fold_set(screen: FoldScreen, faces: Faces, index: float, *, circle_samples: int = 7200) -> DegenerateFoldSet:
    """Locate the slab critical set ``{+-n_M} U {u . n_M = 0}`` relative to ``U_P``."""
    if screen.axis is None:
        return DegenerateFoldSet(None, (), 0.0)
    axis = screen.axis
    points = tuple((sign * axis, location(sign * axis, faces, index)) for sign in (1.0, -1.0))
    e = np.asarray(tangent_basis(jnp.asarray(axis)))
    t = np.linspace(0.0, 2.0 * np.pi, circle_samples, endpoint=False)
    circle = np.cos(t)[:, None] * e[0] + np.sin(t)[:, None] * e[1]
    margins = margins_batch(circle, faces, index)
    inside = np.all(margins > BOUNDARY_MARGIN_ATOL, axis=1)
    return DegenerateFoldSet(axis, points, float(inside.mean()))


@partial(jax.jit, static_argnums=(1, 3))
def _newton_batch(u0: jax.Array, faces: Faces, index: jax.Array, iterations: int, max_step: jax.Array) -> jax.Array:
    """Damped tangent-space Newton on ``grad_{S^2} D_P = 0`` from every row of ``u0`` (fixed iteration count, branch free)."""

    def step(u: jax.Array) -> jax.Array:
        g = jax.grad(d_p)(u, faces, index)
        hessian, basis = _tangent_hessian(u, faces, index, None)
        delta = jnp.linalg.solve(hessian, -(basis @ g))
        norm = jnp.linalg.norm(delta)
        delta = delta * jnp.minimum(1.0, max_step / jnp.maximum(norm, 1e-300))
        moved = u + delta @ basis
        return moved / jnp.linalg.norm(moved)

    def body(_, u):
        return jax.vmap(step)(u)

    return jax.lax.fori_loop(0, iterations, body, u0)


def classify(hessian_eigenvalues: np.ndarray) -> tuple[str, int | None]:
    """Morse kind and index from the two Riemannian Hessian eigenvalues."""
    if np.any(np.abs(hessian_eigenvalues) <= DEGENERATE_EIGENVALUE_TOL):
        return "degenerate", None
    negative = int(np.sum(hessian_eigenvalues < 0.0))
    return ("minimum", "saddle", "maximum")[negative], negative


def lattice_newton_critical_points(
    faces: Faces, index: float, *, lattice_n: int = 20000, iterations: int = 40, max_step_rad: float = 0.05
) -> tuple[InteriorCriticalPoint, ...]:
    """Every interior critical point reached by Newton from the ``U_P`` points of a Fibonacci lattice.

    Seeds are the lattice points inside ``U_P``; converged iterates that are
    back inside ``U_P`` with tangent gradient below ``CRITICAL_GRADIENT_TOL``
    are merged within ``CRITICAL_POINT_MERGE_RAD`` and classified by the
    Riemannian Hessian.  A path with no seeds returns ``()``.
    """
    lattice = fibonacci_sphere(lattice_n)
    seeds = lattice[valid_batch(lattice, faces, index)]
    if len(seeds) == 0:
        return ()
    ends = np.asarray(_newton_batch(jnp.asarray(seeds), faces, jnp.float64(index), iterations, jnp.float64(max_step_rad)))
    finite = np.all(np.isfinite(ends), axis=1)
    ends = ends[finite]
    if len(ends) == 0:
        return ()
    gradient_norm = np.linalg.norm(gradient_batch(ends, faces, index), axis=1)
    converged = ends[(gradient_norm < CRITICAL_GRADIENT_TOL) & valid_batch(ends, faces, index)]
    unique: list[np.ndarray] = []
    for point in converged:
        if all(np.arccos(np.clip(point @ other, -1.0, 1.0)) > CRITICAL_POINT_MERGE_RAD for other in unique):
            unique.append(point)
    if not unique:
        return ()
    points = np.stack(unique)
    values = d_p_batch(points, faces, index)
    hessians, _ = hessian_tangent_batch(points, faces, index)
    norms = np.linalg.norm(gradient_batch(points, faces, index), axis=1)
    out = []
    for point, value, hessian, norm in zip(points, values, hessians, norms):
        eigenvalues = np.linalg.eigvalsh(0.5 * (hessian + hessian.T))
        kind, morse = classify(eigenvalues)
        out.append(InteriorCriticalPoint(point, float(value), eigenvalues, kind, morse, float(norm)))
    return tuple(sorted(out, key=lambda c: c.value))


def interior_critical_points(
    screen: FoldScreen, faces: Faces, index: float, *, lattice_n: int = 20000
) -> tuple[tuple[InteriorCriticalPoint, ...], DegenerateFoldSet | None]:
    """Interior critical points of ``D_P``: the slab set for a degenerate fold, lattice Newton otherwise.

    For a degenerate fold the interior critical points are the members of the
    slab set lying inside ``U_P`` (kind ``"degenerate"``; a slab circle inside
    ``U_P`` is reported by the returned :class:`DegenerateFoldSet` and stops
    the interval partition); the :class:`DegenerateFoldSet` is ``None`` for a
    non-degenerate path.
    """
    if not screen.degenerate:
        return lattice_newton_critical_points(faces, index, lattice_n=lattice_n), None
    fold_set = degenerate_fold_set(screen, faces, index)
    points = []
    for point, where in fold_set.axis_points:
        if where == "interior":
            value = float(d_p_batch(point[None, :], faces, index, screen.fold_matrix)[0])
            points.append(InteriorCriticalPoint(point, value, np.full(2, np.nan), "degenerate", None, 0.0))
    return tuple(points), fold_set


def as_faces(faces: Sequence[int]) -> Faces:
    return optics.normalize_faces(faces)
