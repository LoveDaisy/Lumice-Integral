"""The deviation field ``D_P`` on ``S^2``: evaluation, fold pre-screen, interior critical points.

``u`` is the body-frame direction toward the sun (``docs/conventions.md``),
the incoming ray is ``-u`` and ``Phi_P(-u)`` is the outgoing direction of
the fixed face sequence ``P`` at the identity pose (``R = I``: the field is
a function on the body-frame sphere and does not see the sun direction).
``D_P(u) = angle(Phi_P(-u), -u)``, the deviation formula of
:func:`.s2_store.evaluate_fields` without the pose round trip.  ``U_P`` is
the open set where every margin of :func:`.optics.validity_margin_names` is
positive (:func:`validity_margin_vector`); the internal TIR discriminants of
:func:`.optics.domain_margin_names` stay in :func:`margin_vector` as
diagnostics but bound nothing: a partial internal reflection keeps the pose
in ``U_P`` with Fresnel weight ``R``.  The margins are read off the same
:class:`.optics.PathEvaluation` that :func:`.optics.path_domain_batch` reads
(no second derivation of the gates).  Outside ``U_P`` the smooth branch keeps evaluating (reflections have
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
# The crease-cluster evidence of degenerate_fold_set (task ``dp-slab-partition-completion``): a crease
# sample whose binding gate is within CREASE_CONTACT_MARGIN of zero counts as touching dU_P, and a run of
# such samples hugging dU_P over more than CREASE_TOUCHING_ARC_RAD of arc is a tangency or coincidence
# with it, not a transversal crossing.  Both are pinned on the one positive fixture (``4-8-7-5`` on the
# beta crystal, 7200 samples, 2026-10-07) to its non-triggering side: its interior cluster's contact
# margin is 9.4e-5 and its two transversal crossings hug dU_P for at most 0.031 rad at this tolerance,
# against the grazing fixtures' coincident arcs (``3-1-6``: 3.14 rad, ``1-2-1``: the whole circle); the
# sample spacing is 8.7e-4 rad.  Whether they trigger on a genuine tangency or closed ridge is pinned by
# the synthetic escape tests only -- no fixture exercises the triggering side.
CREASE_CONTACT_MARGIN = 1e-2
CREASE_TOUCHING_ARC_RAD = 0.1
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


def body_normals(crystal: Polyhedron | None, faces: Faces) -> jax.Array:
    """The kernels' ``normals`` argument: :func:`.optics.face_normals` of ``faces`` on ``crystal``.

    ``crystal=None`` is the canonical hexagonal prism.  Resolved once on the
    host and passed into the jitted kernels as a traced array, so ``faces``
    alone keys the compilation and every crystal reuses it.
    """
    return jnp.asarray(optics.face_normals(crystal, faces))


def _evaluation(u: jax.Array, faces: Faces, index: jax.Array, normals: jax.Array | None) -> optics.PathEvaluation:
    if normals is None:
        normals = body_normals(None, faces)
    return optics.trace_path(jnp.eye(3, dtype=u.dtype), normals, -u, index)


def d_p(u: jax.Array, faces: Faces, index: jax.Array, normals: jax.Array | None = None) -> jax.Array:
    """``D_P(u)`` in radians for one unit ``u`` (module docstring).

    The angle is taken in ``atan2`` form: ``arccos`` of the dot product (the
    form of :func:`.s2_store.evaluate_fields`, equal to ``1e-15`` elsewhere)
    loses ``sqrt(eps) ~ 1e-8`` rad next to ``D = 0`` and ``D = pi``, which a
    slab path reaches on a whole boundary arc (``1-3-2``) or at an interior
    point (``3-5-6-7-3``).  ``normals`` is :func:`body_normals` (``None``:
    the canonical prism), likewise for every kernel below.
    """
    return _deviation(_evaluation(u, faces, index, normals).direction, u)


def d_p_grazing(u: jax.Array, faces: Faces, index: jax.Array, normals: jax.Array | None = None) -> jax.Array:
    """:func:`d_p` with the exit refraction's ``sqrt(discriminant)`` set to 0: ``D_P`` on the exit TIR curve.

    The exit direction is ``n d + (n c - sqrt(disc)) (-N)`` (:func:`.optics.refract_smooth`, ``N`` the exit face
    normal), so dropping the root is adding ``-sqrt(disc) N``.  At a point left by a corrector on the ``U_P`` side
    of that curve (``0 <= disc ~ 1e-16``) :func:`d_p` is off the curve's value by ``~ sqrt(disc) ~ 1e-8``, rounding
    noise that differs by BLAS kernel; this is the value there to ``~1e-14``.  Only meaningful where ``disc ~ 0``.
    """
    if normals is None:
        normals = body_normals(None, faces)
    evaluation = _evaluation(u, faces, index, normals)
    root = jnp.sqrt(jnp.maximum(evaluation.exit.discriminant, 0.0))
    return _deviation(evaluation.direction - root * jnp.asarray(normals)[-1], u)


def d_p_exit_limit(u: jax.Array, faces: Faces, index: jax.Array, normals: jax.Array | None = None) -> jax.Array:
    """:func:`d_p` at the ``disc -> 0+`` limit of the exit refraction: the closure value of the exit-Snell convention.

    The exit direction is ``n d_int + (n c - sqrt(disc)) N_t``
    (:func:`.optics.refract_smooth`, ``N_t = -normals[-1]`` toward the incident
    medium, ``d_int`` the last internal direction -- of :attr:`.optics.PathEvaluation`
    ``internal[-1]``, or ``entry`` with no internal reflection -- and ``c`` the exit
    ``incidence_cosine``); this kernel *recomputes* the direction with the root
    dropped, ``n d_int + n c N_t``, instead of subtracting it after the fact.  Same
    mathematics as :func:`d_p_grazing` (both are the transmitted direction's limit
    as ``disc -> 0+``), two float paths: the grazing form starts from
    ``evaluation.direction`` -- already ``NaN`` wherever ``disc < 0``, the exact rounding
    situation of a two-margin Newton corner or of a coincident kink arc -- while this
    one has no square root anywhere, so it stays finite as long as the chain up to the
    exit does (reflections have no root).  It is the value half of the exit-Snell
    closure convention of :mod:`.dp_field.boundary` / :mod:`.dp_field.weight_kink`
    (member decision: every gate ``>= -VIOLATION_ATOL``); use it only there -- at a
    deep interior point (``disc = O(1)``) it is off :func:`d_p` by ``~ sqrt(disc)``,
    it is not a general ``D_P`` replacement.  Not for slab paths (:func:`d_slab`
    has no NaN).  The scale ``n`` cancels in the ``atan2`` deviation.
    """
    if normals is None:
        normals = body_normals(None, faces)
    evaluation = _evaluation(u, faces, index, normals)
    d_int = evaluation.internal[-1].direction if evaluation.internal else evaluation.entry.direction
    normal_t = -jnp.asarray(normals)[-1]
    d_exit = index * d_int + index * evaluation.exit.incidence_cosine * normal_t
    return _deviation(d_exit, u)


def _deviation(phi: jax.Array, u: jax.Array) -> jax.Array:
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


def margin_vector(u: jax.Array, faces: Faces, index: jax.Array, normals: jax.Array | None = None) -> jax.Array:
    """Every margin of :func:`.optics.domain_margin_names` at ``u``, in that order (gates: :func:`validity_margin_vector`)."""
    evaluation = _evaluation(u, faces, index, normals)
    values = [evaluation.entry.incidence_cosine, evaluation.entry.discriminant]
    for reflection in evaluation.internal:
        values.extend((reflection.incidence_cosine, reflection.tir_discriminant))
    values.extend((evaluation.exit.incidence_cosine, evaluation.exit.discriminant))
    return jnp.stack(values)


def validity_margin_vector(u: jax.Array, faces: Faces, index: jax.Array, normals: jax.Array | None = None) -> jax.Array:
    """The gates of ``U_P`` at ``u``: :func:`margin_vector` at :func:`.optics.validity_margin_indices`.

    ``u in U_P`` iff every entry is positive, the test of
    :func:`.optics.path_domain_batch` (:func:`valid_batch`) in a form that
    runs inside a JAX kernel; the two agree by sharing the one list of gates.
    """
    return margin_vector(u, faces, index, normals)[np.asarray(optics.validity_margin_indices(faces))]


def tangent_basis(u: jax.Array) -> jax.Array:
    """An orthonormal tangent basis ``(e1, e2)`` at unit ``u``, shape ``(2, 3)`` (cross with the least aligned axis)."""
    axis = jax.nn.one_hot(jnp.argmin(jnp.abs(u)), 3, dtype=u.dtype)
    e1 = jnp.cross(u, axis)
    e1 = e1 / jnp.linalg.norm(e1)
    return jnp.stack([e1, jnp.cross(u, e1)])


def d_value(
    u: jax.Array, faces: Faces, index: jax.Array, slab: jax.Array | None, normals: jax.Array | None = None
) -> jax.Array:
    """The field as evaluated by this package: :func:`d_slab` for a slab path (``slab`` its fold matrix), :func:`d_p` otherwise."""
    return d_p(u, faces, index, normals) if slab is None else d_slab(u, slab)


def _tangent_gradient(u: jax.Array, faces: Faces, index: jax.Array, slab: jax.Array | None, normals: jax.Array) -> jax.Array:
    g = jax.grad(d_value)(u, faces, index, slab, normals)
    return g - jnp.dot(g, u) * u


def _tangent_hessian(
    u: jax.Array, faces: Faces, index: jax.Array, slab: jax.Array | None, normals: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Riemannian Hessian in :func:`tangent_basis` coordinates and that basis (module docstring)."""
    g = jax.grad(d_value)(u, faces, index, slab, normals)
    h = jax.hessian(d_value)(u, faces, index, slab, normals)
    basis = tangent_basis(u)
    return basis @ (h - jnp.dot(u, g) * jnp.eye(3, dtype=u.dtype)) @ basis.T, basis


@partial(jax.jit, static_argnums=1)
def _d_batch(u: jax.Array, faces: Faces, index: jax.Array, slab: jax.Array | None, normals: jax.Array) -> jax.Array:
    return jax.vmap(d_value, in_axes=(0, None, None, None, None))(u, faces, index, slab, normals)


@partial(jax.jit, static_argnums=1)
def _gradient_batch(u: jax.Array, faces: Faces, index: jax.Array, slab: jax.Array | None, normals: jax.Array) -> jax.Array:
    return jax.vmap(_tangent_gradient, in_axes=(0, None, None, None, None))(u, faces, index, slab, normals)


@partial(jax.jit, static_argnums=1)
def _hessian_batch(
    u: jax.Array, faces: Faces, index: jax.Array, slab: jax.Array | None, normals: jax.Array
) -> tuple[jax.Array, jax.Array]:
    return jax.vmap(_tangent_hessian, in_axes=(0, None, None, None, None))(u, faces, index, slab, normals)


@partial(jax.jit, static_argnums=1)
def _index_derivatives_batch(
    u: jax.Array, faces: Faces, index: jax.Array, slab: jax.Array | None, normals: jax.Array
) -> tuple[jax.Array, jax.Array]:
    d = jax.vmap(jax.grad(d_value, argnums=2), in_axes=(0, None, None, None, None))(u, faces, index, slab, normals)
    m = jax.vmap(jax.jacfwd(margin_vector, argnums=2), in_axes=(0, None, None, None))(u, faces, index, normals)
    return d, m


@partial(jax.jit, static_argnums=1)
def _margins_batch(u: jax.Array, faces: Faces, index: jax.Array, normals: jax.Array) -> jax.Array:
    return jax.vmap(margin_vector, in_axes=(0, None, None, None))(u, faces, index, normals)


@partial(jax.jit, static_argnums=1)
def _validity_margins_batch(u: jax.Array, faces: Faces, index: jax.Array, normals: jax.Array) -> jax.Array:
    return jax.vmap(validity_margin_vector, in_axes=(0, None, None, None))(u, faces, index, normals)


@partial(jax.jit, static_argnums=1)
def _d_exit_limit_batch(u: jax.Array, faces: Faces, index: jax.Array, normals: jax.Array) -> jax.Array:
    return jax.vmap(d_p_exit_limit, in_axes=(0, None, None, None))(u, faces, index, normals)


def _as_points(u: np.ndarray | jax.Array) -> jax.Array:
    points = jnp.asarray(u, dtype=jnp.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("u must have shape (N, 3)")
    return points


def _as_slab(slab: np.ndarray | None) -> jax.Array | None:
    return None if slab is None else jnp.asarray(slab, dtype=jnp.float64)


# The host-side functions below take ``crystal`` (default ``None``: the canonical hexagonal prism) and
# read its face normals through :func:`body_normals`; :class:`.DPField` passes its own crystal.


def d_p_batch(
    u: np.ndarray, faces: Faces, index: float, slab: np.ndarray | None = None, *, crystal: Polyhedron | None = None
) -> np.ndarray:
    """``D_P`` at each row of ``u`` (``(N, 3)`` unit vectors), one ``jax.vmap``; meaningful only on the closure of ``U_P``.

    ``slab`` (the fold matrix of a degenerate-fold path) selects :func:`d_slab`.
    """
    return np.asarray(_d_batch(_as_points(u), faces, jnp.float64(index), _as_slab(slab), body_normals(crystal, faces)))


def gradient_batch(
    u: np.ndarray, faces: Faces, index: float, slab: np.ndarray | None = None, *, crystal: Polyhedron | None = None
) -> np.ndarray:
    """The tangent (``S^2``) gradient of ``D_P`` at each row of ``u``, ``(N, 3)`` ambient vectors orthogonal to ``u``."""
    return np.asarray(
        _gradient_batch(_as_points(u), faces, jnp.float64(index), _as_slab(slab), body_normals(crystal, faces))
    )


def hessian_tangent_batch(
    u: np.ndarray, faces: Faces, index: float, slab: np.ndarray | None = None, *, crystal: Polyhedron | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """The Riemannian Hessian of ``D_P`` at each row of ``u``: ``(N, 2, 2)`` in the basis returned alongside, ``(N, 2, 3)``."""
    hessian, basis = _hessian_batch(
        _as_points(u), faces, jnp.float64(index), _as_slab(slab), body_normals(crystal, faces)
    )
    return np.asarray(hessian), np.asarray(basis)


def index_derivatives_batch(
    u: np.ndarray, faces: Faces, index: float, slab: np.ndarray | None = None, *, crystal: Polyhedron | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """``dD_P/dn`` ``(N,)`` and every ``d margin / dn`` ``(N, len(domain_margin_names(faces)))`` at fixed ``u`` (JAX AD).

    ``dD_P/dn`` is the direction dispersion of the field (``0`` for a slab:
    :func:`d_slab` does not see ``n``); the sign of ``d disc_k / dn`` on a
    TIR onset says which wavelength reflects totally on which side of it.
    """
    d, m = _index_derivatives_batch(_as_points(u), faces, jnp.float64(index), _as_slab(slab), body_normals(crystal, faces))
    return np.asarray(d), np.asarray(m)


def margins_batch(u: np.ndarray, faces: Faces, index: float, *, crystal: Polyhedron | None = None) -> np.ndarray:
    """:func:`margin_vector` at each row of ``u``, ``(N, len(domain_margin_names(faces)))``."""
    return np.asarray(_margins_batch(_as_points(u), faces, jnp.float64(index), body_normals(crystal, faces)))


def validity_margins_batch(
    u: np.ndarray, faces: Faces, index: float, *, crystal: Polyhedron | None = None
) -> np.ndarray:
    """:func:`validity_margin_vector` at each row of ``u``, ``(N, len(validity_margin_names(faces)))``."""
    return np.asarray(_validity_margins_batch(_as_points(u), faces, jnp.float64(index), body_normals(crystal, faces)))


def d_p_exit_limit_batch(
    u: np.ndarray, faces: Faces, index: float, *, crystal: Polyhedron | None = None
) -> np.ndarray:
    """The closure-limit value (:func:`d_p_exit_limit`) at each row of ``u``; the batch half of the exit-Snell convention."""
    return np.asarray(_d_exit_limit_batch(_as_points(u), faces, jnp.float64(index), body_normals(crystal, faces)))


def valid_batch(u: np.ndarray, faces: Faces, index: float, *, crystal: Polyhedron | None = None) -> np.ndarray:
    """``u in U_P`` for each row, decided by :func:`.optics.path_domain_batch` (the single authority of the gates)."""
    u = np.asarray(u, dtype=np.float64)
    rotations = align_rotations(u, _PROBE_SUN)
    return optics.path_domain_batch(rotations, faces, -_PROBE_SUN, index, crystal=crystal).valid


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
    """``n_a . M^T n_b`` with the body normals of ``crystal`` (:func:`.optics.face_normals`) and ``M`` from :func:`.geometry.fold_matrix`."""
    m = fold_matrix(crystal, faces)
    normals = optics.face_normals(crystal, faces)
    n_a = normals[0]
    n_tilde_b = m.T @ normals[-1]
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
    fraction of a dense sampling of the great circle ``u . n_M = 0`` (the
    crease) inside ``U_P`` -- 0 when the circle only grazes it (every
    canonical fixture), 22.76% on the beta crystal's ``4-8-7-5``.  The
    ``crease_*`` fields are that sampling's cluster evidence (task
    ``dp-slab-partition-completion``): ``crease_interior_arcs`` is the number
    of maximal interior runs of the crease (0 exactly when the fraction is 0;
    a fraction claiming otherwise is a contradiction
    :func:`.certificate.interval_partition` refuses), ``crease_closed_ridge``
    says some interior run never comes within ``CREASE_CONTACT_MARGIN`` of
    ``dU_P`` (a closed ridge: level loops around it are not the boundary
    walk's to count), and ``crease_touching_arc`` says the crease hugs ``dU_P``
    (binding gate within ``CREASE_CONTACT_MARGIN`` of zero) over more than
    ``CREASE_TOUCHING_ARC_RAD`` -- a tangency or coincidence, not a
    transversal crossing.
    """

    axis: np.ndarray | None
    axis_points: tuple[tuple[np.ndarray, str], ...]
    circle_interior_fraction: float
    crease_interior_arcs: int = 0
    crease_closed_ridge: bool = False
    crease_touching_arc: bool = False


def location(u: np.ndarray, faces: Faces, index: float, *, crystal: Polyhedron | None = None) -> str:
    """``"interior"``, ``"boundary"`` (smallest gate ``<= BOUNDARY_MARGIN_ATOL`` in size) or ``"exterior"``."""
    margins = validity_margins_batch(np.asarray(u, dtype=np.float64)[None, :], faces, index, crystal=crystal)[0]
    smallest = float(np.min(margins))
    if not np.isfinite(smallest):
        return "exterior"
    if abs(smallest) <= BOUNDARY_MARGIN_ATOL:
        return "boundary"
    return "interior" if smallest > 0.0 else "exterior"


def _circular_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Maximal circular runs of ``True`` as ``(start index, length)``, wrapping runs included."""
    n = len(mask)
    if not mask.any():
        return []
    if mask.all():
        return [(0, n)]
    rotate = int(np.flatnonzero(~mask)[0])  # rotate so index 0 is False: no run crosses it, pairing is linear
    steps = np.diff(np.concatenate([np.roll(mask, -rotate), np.roll(mask, -rotate)[:1]]).astype(np.int8))
    starts = np.flatnonzero(steps == 1) + 1
    ends = np.flatnonzero(steps == -1)
    return sorted(((int(s) + rotate) % n, int(e - s + 1)) for s, e in zip(starts, ends))


def degenerate_fold_set(
    screen: FoldScreen, faces: Faces, index: float, *, circle_samples: int = 7200, crystal: Polyhedron | None = None
) -> DegenerateFoldSet:
    """Locate the slab critical set ``{+-n_M} U {u . n_M = 0}`` relative to ``U_P`` and cluster its crease sampling."""
    if screen.axis is None:
        return DegenerateFoldSet(None, (), 0.0)
    axis = screen.axis
    points = tuple((sign * axis, location(sign * axis, faces, index, crystal=crystal)) for sign in (1.0, -1.0))
    e = np.asarray(tangent_basis(jnp.asarray(axis)))
    t = np.linspace(0.0, 2.0 * np.pi, circle_samples, endpoint=False)
    circle = np.cos(t)[:, None] * e[0] + np.sin(t)[:, None] * e[1]
    margins = validity_margins_batch(circle, faces, index, crystal=crystal)
    binding = margins.min(axis=1)
    inside = binding > BOUNDARY_MARGIN_ATOL
    arcs = _circular_runs(inside)
    closed_ridge = False
    for start, length in arcs:
        contact = binding[(start + np.arange(length)) % circle_samples].min()
        closed_ridge |= contact > CREASE_CONTACT_MARGIN
    spacing = 2.0 * np.pi / circle_samples
    touching_arc = any(run_length * spacing >= CREASE_TOUCHING_ARC_RAD for _, run_length in _circular_runs(np.abs(binding) <= CREASE_CONTACT_MARGIN))
    return DegenerateFoldSet(axis, points, float(inside.mean()), len(arcs), bool(closed_ridge), bool(touching_arc))


@partial(jax.jit, static_argnums=(1, 3))
def _newton_batch(
    u0: jax.Array, faces: Faces, index: jax.Array, iterations: int, max_step: jax.Array, normals: jax.Array
) -> jax.Array:
    """Damped tangent-space Newton on ``grad_{S^2} D_P = 0`` from every row of ``u0`` (fixed iteration count, branch free)."""

    def step(u: jax.Array) -> jax.Array:
        g = jax.grad(d_p)(u, faces, index, normals)
        hessian, basis = _tangent_hessian(u, faces, index, None, normals)
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
    faces: Faces,
    index: float,
    *,
    lattice_n: int = 20000,
    iterations: int = 40,
    max_step_rad: float = 0.05,
    crystal: Polyhedron | None = None,
) -> tuple[InteriorCriticalPoint, ...]:
    """Every interior critical point reached by Newton from the ``U_P`` points of a Fibonacci lattice.

    Seeds are the lattice points inside ``U_P``; converged iterates that are
    back inside ``U_P`` (every gate above ``BOUNDARY_MARGIN_ATOL``, :func:`location` ``"interior"``) with tangent gradient below ``CRITICAL_GRADIENT_TOL``
    are merged within ``CRITICAL_POINT_MERGE_RAD`` and classified by the
    Riemannian Hessian.  A path with no seeds returns ``()``.
    """
    lattice = fibonacci_sphere(lattice_n)
    seeds = lattice[valid_batch(lattice, faces, index, crystal=crystal)]
    if len(seeds) == 0:
        return ()
    ends = np.asarray(
        _newton_batch(
            jnp.asarray(seeds), faces, jnp.float64(index), iterations, jnp.float64(max_step_rad), body_normals(crystal, faces)
        )
    )
    finite = np.all(np.isfinite(ends), axis=1)
    ends = ends[finite]
    if len(ends) == 0:
        return ()
    gradient_norm = np.linalg.norm(gradient_batch(ends, faces, index, crystal=crystal), axis=1)
    # inside the open U_P, not on dU_P: a critical point of the smooth extension that sits on a grazing
    # internal-reflection piece (3-4-5-7, D = 141.84 deg, gate 1e-16) is the walk's loop extremum, not interior
    off_boundary = np.min(validity_margins_batch(ends, faces, index, crystal=crystal), axis=1) > BOUNDARY_MARGIN_ATOL
    converged = ends[(gradient_norm < CRITICAL_GRADIENT_TOL) & valid_batch(ends, faces, index, crystal=crystal) & off_boundary]
    unique: list[np.ndarray] = []
    for point in converged:
        if all(np.arccos(np.clip(point @ other, -1.0, 1.0)) > CRITICAL_POINT_MERGE_RAD for other in unique):
            unique.append(point)
    if not unique:
        return ()
    points = np.stack(unique)
    values = d_p_batch(points, faces, index, crystal=crystal)
    hessians, _ = hessian_tangent_batch(points, faces, index, crystal=crystal)
    norms = np.linalg.norm(gradient_batch(points, faces, index, crystal=crystal), axis=1)
    out = []
    for point, value, hessian, norm in zip(points, values, hessians, norms):
        eigenvalues = np.linalg.eigvalsh(0.5 * (hessian + hessian.T))
        kind, morse = classify(eigenvalues)
        out.append(InteriorCriticalPoint(point, float(value), eigenvalues, kind, morse, float(norm)))
    return tuple(sorted(out, key=lambda c: c.value))


def interior_critical_points(
    screen: FoldScreen, faces: Faces, index: float, *, lattice_n: int = 20000, crystal: Polyhedron | None = None
) -> tuple[tuple[InteriorCriticalPoint, ...], DegenerateFoldSet | None]:
    """Interior critical points of ``D_P``: the slab set for a degenerate fold, lattice Newton otherwise.

    For a degenerate fold the interior critical points are the members of the
    slab set lying inside ``U_P`` (kind ``"degenerate"``; a slab circle inside
    ``U_P`` is reported by the returned :class:`DegenerateFoldSet` and judged
    by :func:`.certificate.interval_partition` on its cluster evidence); the
    :class:`DegenerateFoldSet` is ``None`` for a non-degenerate path.
    """
    if not screen.degenerate:
        return lattice_newton_critical_points(faces, index, lattice_n=lattice_n, crystal=crystal), None
    fold_set = degenerate_fold_set(screen, faces, index, crystal=crystal)
    points = []
    for point, where in fold_set.axis_points:
        if where == "interior":
            value = float(d_p_batch(point[None, :], faces, index, screen.fold_matrix, crystal=crystal)[0])
            points.append(InteriorCriticalPoint(point, value, np.full(2, np.nan), "degenerate", None, 0.0))
    return tuple(points), fold_set
