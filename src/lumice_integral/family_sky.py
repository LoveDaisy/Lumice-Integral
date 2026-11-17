"""First differential of a caller-defined pose family into the unit sky sphere.

The caller supplies body angular velocities Omega, dR/dt = R [Omega]_cross,
not normalized axes. Parameters are in radians: dx has units sky radian per
parameter radian (numerically also degree/degree). A world left generator w
becomes R.T @ w. Family parameterizations are deliberately not defined here.

The sky map is x(R) = -trace_path(R, normals, -sun_direction, n).direction.
Only the open optical domain is differentiated; finite-crystal footprints,
weights and brightness are outside this instrument's contract.

For a unit x, differentiating x.x = 1 gives x.x_i = 0. Thus x_a cross x_b
is parallel to x, and T = x.(x_a cross x_b) satisfies T² = det(A.T A), where
A = (x_a, x_b). Only for a two-parameter sphere-valued map does this single
scalar detect rank loss. For a general R³ map, x=(1,a,b) at a=b=0 has T=1,
whereas x=(a,b,0) at (1,0) has T=0 with two independent derivatives. For S¹
use speed; there is no signed area. Numerical rank loss is a first-order
statement, not a proof of a fold, turning, caustic or increased brightness.

See ``docs/family-sky-differential.md`` for domain, tolerance and usage details.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from .camera import incident_direction_from_sun
from .geometry import Polyhedron
from .optics import face_normals, normalize_faces, path_domain_batch, trace_path
from .so3 import exp


@dataclass(frozen=True, eq=False)
class SkyDifferential:
    """Host float64 arrays: sky (N,3), dx/generators (N,K,3), flags (N,).

    ``valid`` is optical-domain validity. ``available`` additionally requires
    finite AD output; unavailable sky/dx are NaN. Domain evidence is retained
    even on failure. Arrays are returned as observations, not JAX tracers.
    """

    sky: np.ndarray
    dx: np.ndarray
    tangent_generators: np.ndarray
    valid: np.ndarray
    available: np.ndarray
    margins: Mapping[str, np.ndarray]


@dataclass(frozen=True, eq=False)
class SkyDiagnostics:
    """First-order diagnostics for K=1/2; leading dimension is N.

    ``speeds`` (N,K), ``gram`` (N,K,K), ascending ``gram_eigenvalues``
    (N,K), descending ``singular_values`` (N,K), ``null_direction`` (N,K)
    (the least right singular vector; a null line only when rank deficient).
    Its sign is arbitrary; at rank zero no unique null line is implied.
    ``signed_area`` is (N,) for K=2, None for K=1. Unavailable continuous
    values are NaN, integer ranks -1, and mechanism ``invalid`` (off domain)
    or ``unavailable`` (on domain but non-finite differential/diagnostics).
    """

    speeds: np.ndarray
    gram: np.ndarray
    gram_eigenvalues: np.ndarray
    singular_values: np.ndarray
    null_direction: np.ndarray
    signed_area: np.ndarray | None
    rank: np.ndarray
    generator_rank: np.ndarray
    rank_threshold: np.ndarray
    mechanism: np.ndarray


def _finite_array(name: str, value) -> np.ndarray:
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real and finite")
    try:
        array = np.asarray(value, dtype=np.float64)
    except (ValueError, TypeError) as error:
        raise ValueError(f"{name} must be real and finite") from error
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be real and finite")
    return array


@jax.jit
def _differential_batch(rotations, generators, normals, incident, index):
    def one(rotation, omega):
        def sky_at(delta):
            return -trace_path(rotation @ exp(delta), normals, incident, index).direction

        sky = sky_at(jnp.zeros(3, dtype=rotation.dtype))
        jacobian = jax.jacfwd(sky_at)(jnp.zeros(3, dtype=rotation.dtype))
        return sky, omega @ jacobian.T

    return jax.vmap(one)(rotations, generators)


def sky_differential(
    rotations: np.ndarray,
    tangent_generators: np.ndarray,
    faces: Sequence[int],
    sun_direction: np.ndarray,
    *,
    crystal: Polyhedron | None = None,
    refractive_index: float = 1.31,
) -> SkyDifferential:
    """Evaluate sky and dx on an explicit (N,3,3) / (N,K,3) batch, K>=1.

    No broadcasting, axis normalization or input repair. Rotations must be
    SO(3) to 1e-10 absolute residual, the sun unit to 1e-12, n positive finite.
    All inputs must be real finite. Optical gates come exclusively from
    ``optics.path_domain_batch`` (internal partial reflection is allowed).
    Only valid rows enter AD; unavailable rows never enter a decomposition.
    K>2 is supported here but not by :func:`diagnose_sky`. N=0 is supported.
    Kernel normals and n are numerical arguments, not crystal-identity jit
    closures. This host validation/selection API is not itself JAX-transformable.
    """
    r = _finite_array("rotations", rotations)
    omega = _finite_array("tangent_generators", tangent_generators)
    sun = _finite_array("sun_direction", sun_direction)
    index = _finite_array("refractive_index", refractive_index)
    if r.ndim != 3 or r.shape[1:] != (3, 3):
        raise ValueError("rotations must have shape (N, 3, 3)")
    if omega.ndim != 3 or omega.shape[0] != len(r) or omega.shape[2] != 3 or omega.shape[1] < 1:
        raise ValueError("tangent_generators must have shape (N, K, 3), K >= 1")
    if sun.shape != (3,) or not np.isclose(sun @ sun, 1.0, atol=1e-12, rtol=0):
        raise ValueError("sun_direction must be a unit vector of shape (3,)")
    if index.shape != () or index <= 0:
        raise ValueError("refractive_index must be a finite positive scalar")
    if not (np.allclose(r.swapaxes(1, 2) @ r, np.eye(3), atol=1e-10, rtol=0)
            and np.allclose(np.linalg.det(r), 1.0, atol=1e-10, rtol=0)):
        raise ValueError("rotations must be in SO(3)")
    face_array = np.asarray(faces)
    if face_array.ndim != 1 or face_array.dtype.kind not in "iu":
        raise ValueError("faces must be a sequence of integer face numbers")
    faces = normalize_faces(faces, crystal)
    normals = face_normals(crystal, faces)
    incident = incident_direction_from_sun(sun)
    domain = path_domain_batch(r, faces, incident, index, crystal=crystal)
    sky = np.full((len(r), 3), np.nan)
    dx = np.full(omega.shape, np.nan)
    available = np.zeros(len(r), dtype=bool)
    rows = np.flatnonzero(domain.valid)
    if len(rows):
        x, derivative = map(np.asarray, _differential_batch(
            jnp.asarray(r[rows]), jnp.asarray(omega[rows]), jnp.asarray(normals),
            jnp.asarray(incident), jnp.asarray(index),
        ))
        finite = np.isfinite(x).all(axis=1) & np.isfinite(derivative).all(axis=(1, 2))
        available[rows] = finite
        sky[rows[finite]] = x[finite]
        dx[rows[finite]] = derivative[finite]
    return SkyDifferential(sky, dx, omega.copy(), domain.valid, available, domain.margins)


def diagnose_sky(
    differential: SkyDifferential,
    *,
    atol: float = 1e-12,
    rtol: float = 1e-10,
    generator_rtol: float = 1e-12,
) -> SkyDiagnostics:
    """Diagnose K=1/2 with rank threshold atol + rtol*sigma_max.

    Defaults target float64 roundoff, not the 1e-4 residual at historical
    rounded valley coordinates. atol has derivative units, rtol is relative;
    neither changes the raw dx/T. Small-parameter rescalings can change the
    *numerical* rank, as expected for an explicit absolute threshold.

    Input tangent rank uses independently length-normalized nonzero Omega
    columns and threshold generator_rtol*sigma_max (dimensionless), so unit
    disparities alone do not imply dependence. A zero column is degenerate.
    Labels are coordinate-dependent: passing/stationary (S¹), regular,
    axis_stationary/parallel_collapse (rank 1), rank_zero (S²), overridden by
    degenerate_generators for deficient input tangent rank. Invertible changes
    of parameter coordinates preserve exact rank, not these mechanism labels.
    """
    for name, value in (("atol", atol), ("rtol", rtol), ("generator_rtol", generator_rtol)):
        scalar = _finite_array(name, value)
        if scalar.shape != () or scalar < 0:
            raise ValueError(f"{name} must be a finite nonnegative scalar")
    n, k, _ = differential.dx.shape
    if k not in (1, 2):
        raise ValueError("diagnose_sky requires K=1 or K=2")
    speeds = np.full((n, k), np.nan)
    gram = np.full((n, k, k), np.nan)
    eigenvalues = np.full((n, k), np.nan)
    singular = np.full((n, k), np.nan)
    null = np.full((n, k), np.nan)
    area = np.full(n, np.nan) if k == 2 else None
    rank = np.full(n, -1, dtype=int)
    input_rank = np.full(n, -1, dtype=int)
    threshold = np.full(n, np.nan)
    labels = np.where(differential.valid, "unavailable", "invalid").astype("U24")
    rows = np.flatnonzero(differential.valid & differential.available)
    if len(rows):
        derivatives = differential.dx[rows]
        with np.errstate(over="ignore", invalid="ignore"):
            g = derivatives @ derivatives.swapaxes(1, 2)
        # Finite AD can still overflow quadratic diagnostics. Fail explicitly
        # before LAPACK, without poisoning independent rows in the batch.
        finite = np.isfinite(g).all(axis=(1, 2))
        rows, derivatives, g = rows[finite], derivatives[finite], g[finite]
        if len(rows):
            _, s, vh = np.linalg.svd(derivatives.swapaxes(1, 2), full_matrices=False)
            cut = atol + rtol * s[:, 0]
            ranks = (s > cut[:, None]).sum(axis=1)
            speed = np.linalg.norm(derivatives, axis=2)
            omega = differential.tangent_generators[rows]
            # Max-component scaling avoids overflow/underflow in length normalization.
            scale = np.max(np.abs(omega), axis=2, keepdims=True)
            scaled = omega / np.where(scale > 0, scale, 1.0)
            length = np.linalg.norm(scaled, axis=2, keepdims=True)
            normalized = scaled / np.where(length > 0, length, 1.0)
            generator_s = np.linalg.svd(normalized.swapaxes(1, 2), compute_uv=False)
            generator_rank = (generator_s > generator_rtol * generator_s[:, :1]).sum(axis=1)
            speeds[rows], gram[rows] = speed, g
            eigenvalues[rows] = np.linalg.eigvalsh(g)
            singular[rows], null[rows] = s, vh[:, -1]
            rank[rows], input_rank[rows], threshold[rows] = ranks, generator_rank, cut
            if k == 1:
                mechanism = np.where(ranks == 1, "passing", "stationary")
            else:
                area[rows] = np.einsum("ni,ni->n", differential.sky[rows], np.cross(derivatives[:, 0], derivatives[:, 1]))
                mechanism = np.where(ranks == 2, "regular", np.where(
                    ranks == 0, "rank_zero", np.where(
                        (speed <= cut[:, None]).any(axis=1), "axis_stationary", "parallel_collapse",
                    ),
                ))
            labels[rows] = np.where(generator_rank < k, "degenerate_generators", mechanism)
    return SkyDiagnostics(speeds, gram, eigenvalues, singular, null, area, rank, input_rank, threshold, labels)
