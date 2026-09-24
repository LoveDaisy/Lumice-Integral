"""The ray-path *class* as the rendering unit of the hexagonal prism.

A halo is the image of a whole conjugacy class of ray paths under the
crystal's symmetry group, not of one face sequence (writing series ch7/ch8;
Lumice's ``symmetry: PBD`` comparisons are class level).  This module turns
one representative face sequence into its class and drives the single-path
pipeline of :mod:`.strip_pixel` once per member:

1. :func:`pbd_orbit_hexprism` expands the representative under the face
   permutations induced by ``D6h`` acting on the prism (Lumice ``PBD``: the
   six rotations about the c axis, the vertical mirrors and the basal swap).
   It is an independent implementation -- group elements are orthogonal
   matrices acting on the crystal's face normals, faces are matched by normal
   -- and ``tests/test_path_class.py`` cross-checks it against the index
   arithmetic of ``tests/_geometry_oracles.py::pbd_orbit`` (two routes, one
   answer: a01/a02).
2. :func:`build_path_class` attaches the path-level invariants
   :func:`.geometry.wedge_angle_deg` / :func:`.geometry.halo_map_rank` and
   verifies mechanically that every member shares them.
3. A rank-2 class (``halo_map_rank == 2``, one-dimensional fibers) is
   rendered by :func:`render_class_pixel`: one :class:`.strip_pixel.StripScene`
   per member, :func:`.strip_pixel.render_pixel` per member, contributions
   summed.  The members' discovery seeds come from the class's
   :func:`store_plan` (:class:`ClassScene`): one S^2 event store for the
   ``D6h`` orbit, each member's candidates posed through its ``Phi`` group's
   element ``g`` (:class:`.s2_store.StoreSeeds`) -- the plan the band-sum
   renderer uses, not a store per member.  Members are neither merged nor assumed equal: the
   symmetric-density identity "3-7 equals 3-5 pointwise" is a *test* of the
   canonical scene, not an assumption of the code.
4. A rank-0 class (``M = I`` and ``W = 0``: ``1-2``, ``3-6``, ...; ch8 A0-06)
   sends every pose to the sun direction.  Its contribution is a point mass
   at ``-incident_direction`` of total weight

       m = E_Haar[ [R in V_P] rho_H(R) A_P(R) T_P(R) ]
         = (1 / 8 pi^2) integral_(V_P) rho_H A_P T_P dVol_g,

   estimated by :func:`estimate_rank0_contribution` from a raw Haar stream
   (:data:`RANK0_RNG_SEED`, :data:`RANK0_SAMPLE_COUNT`).  ``m``
   is in the same normalisation as the pixel values of a rank-2 path (the
   push-forward of ``rho_H W dmu_Haar``, whose sphere density the fiber
   quadrature reports; ``docs/phase1-math-contract.md`` section 7), so on
   the one pixel that contains the sun the pixel-averaged value is
   ``m / pixel_solid_angle``, every other pixel gets ``0.0``, and a render
   window that does not contain the sun gets ``0.0`` without sampling
   (recorded in ``provenance``).  A rank-0 class never enters the fiber
   pipeline: almost every pose is aligned, the "one-dimensional fiber"
   premise is false there.
5. :func:`phi_key` groups face sequences by their outgoing-direction map
   ``Phi`` (``3-5`` and ``3-1-2-5`` share one), and
   :func:`path_class_symmetry` gives each class member a ``D6h`` element
   (proper or improper) that transports the representative's S^2 event
   store onto it (:mod:`.s2_store`); :func:`store_plan` combines the two
   into the stores of a class (:class:`StoreGroup`, :class:`Transport`).
   All three are pure combinatorics on the prism's face normals and the
   ``D6h`` table; the store itself (building, caching, I/O) lives in
   :mod:`.s2_store`, not here.  Everything in this module is
   specific to the hexagonal prism.

Nothing here imports or calls Lumice.
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from .camera import camera_rotation, incident_direction_from_sun, linear_pixel_sky_direction, linear_scale, project_linear
from .canonical_scene import (
    CANONICAL_REFRACTIVE_INDEX,
    CANONICAL_RENDER,
    canonical_crystal,
    canonical_pose_density,
    canonical_sun_direction,
)
from .discovery import ComponentDiscoveryResult, discover_components
from .geometry import HexPrism, Polyhedron, entry_measure_batch, fold_matrix, halo_map_rank, wedge_angle_deg
from .optics import fresnel_transmission_path_batch, normalize_faces, path_domain_batch, path_id_of
from .pose_density import PoseDensity
from .quadrature import HAAR_TO_DVOL_G_FACTOR
from .s2_store import DEFAULT_SEED_STORE_N, S2EventStore, StoreSeeds
from .so3 import haar_rotations
from .strip_pixel import (
    EVENT_NAMES,
    STAGE_NAMES,
    ComponentRecord,
    PixelOptions,
    PixelResult,
    StripScene,
    build_strip_scene,
    render_pixel,
    seed_store,
)
from .symmetry.signature import D6H

Faces = tuple[int, ...]
Completeness = str  # "complete" | "unknown"
# The Haar stream of the rank-0 point-mass estimate (module docstring item 4): the values the retired Phase I
# prescan table was sampled with, which every recorded rank-0 estimate used.
RANK0_SAMPLE_COUNT = 4_000_000
RANK0_RNG_SEED = 20260916
RANK0_BATCH_SIZE = 200_000

# ---- PBD orbit ---------------------------------------------------------------


def hexprism_symmetry_matrices() -> tuple[np.ndarray, ...]:
    """The 24 orthogonal matrices of ``D6h`` in the body frame (c axis = +z, face 3 normal = +x).

    This is :data:`.symmetry.signature.D6H`, the repository's only ``D6h``
    element table (migrated from the writing series, task
    ``symmetry-authority``): ``Rz(60 k)`` and the vertical mirror
    ``sxy(30 k)`` interleaved for ``k = 0..5``, then the same twelve
    premultiplied by the basal mirror ``B = diag(1, 1, -1)``.  The order is
    that fixed construction order; it is not the published numbering #1-#12
    of the fold group ``G`` (:mod:`.symmetry.reflection_group`).  Across
    projects an element is identified by its matrix
    (``docs/conventions.md`` row 14).
    """
    return D6H


_D6H = hexprism_symmetry_matrices()
NORMAL_MATCH_ATOL = 1e-9


def _hexprism_normals(crystal: HexPrism) -> dict[int, np.ndarray]:
    return {face.number: crystal.normal(face) for face in crystal.faces}


def _face_of_normal(target: np.ndarray, normals: Mapping[int, np.ndarray]) -> int:
    """The one face whose outward normal is ``target`` (``RuntimeError`` unless exactly one matches)."""
    matches = [number for number, normal in normals.items() if np.allclose(normal, target, atol=NORMAL_MATCH_ATOL)]
    if len(matches) != 1:
        raise RuntimeError(f"normal {target} matched faces {matches}")
    return matches[0]


def _symmetry_image_of_faces(element: np.ndarray, faces: Faces, normals: Mapping[int, np.ndarray]) -> Faces:
    """Face sequence whose normals are ``element @ n_f`` for each face ``f`` of ``faces``."""
    return tuple(_face_of_normal(element @ normals[face], normals) for face in faces)


def pbd_orbit_hexprism(faces: Sequence[int], crystal: HexPrism | None = None) -> frozenset[Faces]:
    """Orbit of a face sequence under ``D6h`` (Lumice ``PBD``) as a set of face sequences.

    Each group element ``g`` maps face ``f`` to the face whose outward normal
    is ``g @ n_f`` (normals from ``crystal``, default :class:`.geometry.HexPrism`);
    the face sequence is mapped elementwise.  Only the point group acts, so
    the result does not depend on the prism's aspect ratio.
    """
    faces = normalize_faces(faces)
    normals = _hexprism_normals(HexPrism() if crystal is None else crystal)
    return frozenset(_symmetry_image_of_faces(element, faces, normals) for element in _D6H)


def phi_key(crystal: Polyhedron, faces: Sequence[int]) -> tuple[int, int, int]:
    """Hashable key of the outgoing-direction map ``Phi``: equal keys, equal ``Phi(u)`` (and ``D(u)``).

    ``Phi`` of a face sequence is fixed by the fold matrix ``M``
    (:func:`.geometry.fold_matrix`), the entry normal ``n_a`` and the
    unfolded exit normal ``n_tilde_b = M^T n_b``: refract in through ``n_a``,
    refract out through ``n_tilde_b``, apply ``M``.  On the hexagonal prism
    every mirror is an element of ``D6h`` and ``D6h`` permutes the face
    normals, so ``M`` is one of :func:`hexprism_symmetry_matrices` and
    ``n_a``, ``n_tilde_b`` are face normals: the key is the exact integer
    triple ``(index of M, face of n_a, face of n_tilde_b)``, matched against
    those finite sets (``RuntimeError`` if a match fails -- the closure
    argument would be broken, not a new key).  ``3-5`` and ``3-1-2-5`` share
    a key; ``3-5`` and ``3-7`` do not.  The key is compared for equality
    only: the index of ``M`` is a position in :func:`hexprism_symmetry_matrices`,
    not a published element number (``docs/conventions.md`` row 14).

    Relation to the writing series' classes (:mod:`.symmetry.signature`):
    equal keys mean the same ``Phi`` exactly, no quotient taken.  ``D6h``
    acts on a key as ``(g M g^T, g a, g a~)``, and one orbit of keys is one
    canonical signature class (``canonical_signature``); a
    ``symmetry.signature.phi_class`` is a union of such orbits -- one orbit
    for the 60- and 90-degree wedges, the parallel (0-degree) orbits merged
    by the conjugacy class of ``M`` (framework theorem 5').  Checked by
    ``tests/test_path_class_phi_key.py::test_d6h_orbit_of_the_key_is_one_signature_class_and_refines_phi_class``.
    """
    if not isinstance(crystal, HexPrism):
        raise TypeError("phi_key is implemented for the hexagonal prism only")
    faces = normalize_faces(faces)
    normals = _hexprism_normals(crystal)
    M = fold_matrix(crystal, faces)
    matches = [i for i, element in enumerate(_D6H) if np.allclose(M, element, atol=NORMAL_MATCH_ATOL)]
    if len(matches) != 1:
        raise RuntimeError(f"fold matrix of {path_id_of(faces)} matched D6h elements {matches}")
    return matches[0], faces[0], _face_of_normal(M.T @ normals[faces[-1]], normals)


def path_class_symmetry(
    path_class: PathClass,
    crystal: HexPrism | None = None,
    *,
    symmetry_elements: Sequence[np.ndarray] | None = None,
) -> dict[Faces, np.ndarray]:
    """Per member, a ``D6h`` element ``g`` (proper or improper) mapping the representative's faces onto it.

    ``g`` maps face ``f`` to the face with normal ``g @ n_f`` (the action of
    :func:`pbd_orbit_hexprism`), so the member's ``Phi``, weights and valid
    domain are the representative's transported by ``g`` (:mod:`.s2_store`,
    roadmap section 4.1(d)); on ``S^2`` a mirror transports like a rotation.
    The representative maps to the identity.  ``symmetry_elements`` (default
    all 24 of ``D6h``) restricts the search; proper elements are tried
    first, then improper ones, each in order, and the first match wins.
    Any matching element serves: two of them differ by an element fixing
    the representative's face sequence, which fixes its fields.  A member
    that no element reaches is not in the representative's orbit -- the
    class was built wrong -- and raises ``RuntimeError``.
    """
    normals = _hexprism_normals(HexPrism() if crystal is None else crystal)
    elements = _D6H if symmetry_elements is None else tuple(np.asarray(e, dtype=np.float64) for e in symmetry_elements)
    ordered = [e for e in elements if np.linalg.det(e) > 0.0] + [e for e in elements if np.linalg.det(e) < 0.0]
    images = [(_symmetry_image_of_faces(e, path_class.representative, normals), e) for e in ordered]
    out: dict[Faces, np.ndarray] = {}
    for member in path_class.members:
        found = next((e for image, e in images if image == member), None)
        if found is None:
            raise RuntimeError(
                f"{path_id_of(member)} is not a D6h image of {path_id_of(path_class.representative)}: "
                "the class is not one orbit"
            )
        out[member] = np.array(found)
    return out


@dataclass(frozen=True)
class PathClass:
    """A ray-path class: representative, sorted members, and the shared invariants.

    ``wedge_deg`` is :func:`.geometry.wedge_angle_deg` (degrees) and
    ``halo_map_rank`` is :func:`.geometry.halo_map_rank` of the representative;
    :func:`build_path_class` checks that every member agrees.
    """

    representative: Faces
    members: tuple[Faces, ...]
    wedge_deg: float
    halo_map_rank: int

    def __post_init__(self) -> None:
        representative = normalize_faces(self.representative)
        members = tuple(normalize_faces(member) for member in self.members)
        if representative not in members:
            raise ValueError("the representative must be one of the members")
        if len(set(members)) != len(members):
            raise ValueError("members must be distinct")
        if self.halo_map_rank not in (0, 2):
            raise ValueError("halo_map_rank must be 0 or 2")
        object.__setattr__(self, "representative", representative)
        object.__setattr__(self, "members", tuple(sorted(members)))
        object.__setattr__(self, "wedge_deg", float(self.wedge_deg))
        object.__setattr__(self, "halo_map_rank", int(self.halo_map_rank))

    @property
    def size(self) -> int:
        return len(self.members)

    @property
    def path_ids(self) -> tuple[str, ...]:
        return tuple(path_id_of(member) for member in self.members)

    def provenance(self) -> dict[str, Any]:
        """JSON-serialisable description of the class."""
        return {
            "representative": list(self.representative),
            "members": [list(member) for member in self.members],
            "path_ids": list(self.path_ids),
            "size": self.size,
            "wedge_deg": self.wedge_deg,
            "halo_map_rank": self.halo_map_rank,
        }


def build_path_class(crystal: Polyhedron, representative: Sequence[int]) -> PathClass:
    """The PBD class of ``representative`` on the hexagonal prism, with verified invariants."""
    if not isinstance(crystal, HexPrism):
        raise TypeError("path classes are implemented for the hexagonal prism only")
    representative = normalize_faces(representative)
    members = pbd_orbit_hexprism(representative, crystal)
    wedge = wedge_angle_deg(crystal, representative)
    rank = halo_map_rank(crystal, representative)
    for member in sorted(members):
        member_wedge = wedge_angle_deg(crystal, member)
        member_rank = halo_map_rank(crystal, member)
        if abs(member_wedge - wedge) > 1e-9 or member_rank != rank:
            raise ValueError(
                f"member {path_id_of(member)} has wedge {member_wedge:.9g} deg / rank {member_rank}, "
                f"representative {path_id_of(representative)} has {wedge:.9g} deg / rank {rank}"
            )
    return PathClass(representative, tuple(sorted(members)), wedge, rank)


# ---- store plan ----------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class Transport:
    """A ``Phi`` group served by a store through the ``D6h`` element ``g`` (:func:`.s2_store.transported_rotations`).

    ``g`` may be proper or improper.  ``g is None`` is the identity (the
    store's own group; no multiplication, so the poses are bit-identical to
    the untransported ones).

    ``eq=False`` (identity comparison): ``g`` is an ``np.ndarray``, which
    breaks the dataclass-generated ``__eq__``/``__hash__`` (ambiguous truth
    value / unhashable) if ever compared or hashed.
    """

    members: tuple[Faces, ...]
    g: np.ndarray | None

    def as_json(self) -> dict[str, Any]:
        return {
            "members": [path_id_of(m) for m in self.members],
            "g": None if self.g is None else np.round(self.g, 12).tolist(),
        }


@dataclass(frozen=True, eq=False)
class StoreGroup:
    """One event store (of the ``Phi`` group ``members``) and the groups it serves.

    ``eq=False`` for the same reason as :class:`Transport`: it holds
    ``Transport`` instances (which carry an ``np.ndarray`` field), so
    identity comparison avoids the same ambiguous-truth-value/unhashable trap.
    """

    members: tuple[Faces, ...]
    transports: tuple[Transport, ...]

    @property
    def served_members(self) -> tuple[Faces, ...]:
        return tuple(m for t in self.transports for m in t.members)

    def as_json(self) -> dict[str, Any]:
        return {"store_members": [path_id_of(m) for m in self.members], "transports": [t.as_json() for t in self.transports]}


def single_path_class(crystal: HexPrism, faces: Sequence[int]) -> PathClass:
    """A one-member :class:`.path_class.PathClass` of ``faces`` (the single-path renderer's unit)."""
    faces = normalize_faces(faces)
    return PathClass(faces, (faces,), wedge_angle_deg(crystal, faces), halo_map_rank(crystal, faces))


def store_plan(path_class: PathClass, crystal: HexPrism, *, transport: bool = True) -> tuple[StoreGroup, ...]:
    """Stores of a rank-2 class: ``Phi`` groups, then one store for the ``D6h`` orbit of groups (module docstring).

    Every member is served exactly once (checked).  A class is one ``D6h``
    orbit, so with ``transport`` the plan is a single store (the
    representative's group); a member outside the orbit is a class
    construction error (``RuntimeError`` from
    :func:`.path_class.path_class_symmetry`).  ``transport=False`` gives
    every group its own store.  A rank-0 class has no stores (empty plan).
    """
    if path_class.halo_map_rank == 0:
        return ()
    by_key: dict[Any, list[Faces]] = {}
    for member in path_class.members:
        by_key.setdefault(phi_key(crystal, member), []).append(member)
    groups = [tuple(sorted(g)) for g in by_key.values()]
    groups.sort(key=lambda g: (path_class.representative not in g, g))
    if not transport:
        plan = tuple(StoreGroup(group, (Transport(group, None),)) for group in groups)
    else:
        source = groups[0]
        rooted = PathClass(source[0], path_class.members, path_class.wedge_deg, path_class.halo_map_rank)
        symmetry = path_class_symmetry(rooted, crystal)
        transports = [Transport(source, None)]
        for group in groups[1:]:
            # g maps source[0] onto a member of ``group``; Phi_{g m}(u) = g Phi_m(g^-1 u) for every member m
            # of ``source`` (proper or improper g), so g maps the whole Phi group onto ``group`` (same size).
            # Any member's element serves; a proper one is preferred (no reflection factor, cheaper).
            g = next((symmetry[m] for m in group if np.linalg.det(symmetry[m]) > 0.0), symmetry[group[0]])
            transports.append(Transport(group, g))
        plan = (StoreGroup(source, tuple(transports)),)
    served = sorted(m for s in plan for m in s.served_members)
    if served != sorted(path_class.members):
        raise RuntimeError(f"store plan serves {served}, class has {sorted(path_class.members)}")
    return plan



# ---- rank-0 point mass -------------------------------------------------------


def haar_domain_batches(
    faces: Sequence[int],
    incident_direction: np.ndarray,
    refractive_index: float,
    *,
    sample_count: int = RANK0_SAMPLE_COUNT,
    rng_seed: int = RANK0_RNG_SEED,
    batch_size: int = RANK0_BATCH_SIZE,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """The Haar stream restricted to the smooth domain of ``faces``: per batch, the valid poses and their directions.

    ``sample_count`` poses of :func:`.so3.haar_rotations` from one
    ``numpy`` generator seeded with ``rng_seed``, drawn ``batch_size`` at a
    time (the batch size bounds memory and does not change the stream), gated
    by :func:`.optics.path_domain_batch` on the infinite prism (no crystal).
    The stream of :func:`estimate_rank0_contribution` and of the sky landing
    maps of the diagnostic scripts; Phase I discovery seeds from the S^2
    event store instead (:class:`.s2_store.StoreSeeds`).
    """
    faces = normalize_faces(faces)
    if sample_count < 1 or batch_size < 1:
        raise ValueError("sample_count and batch_size must be positive")
    incident = np.asarray(incident_direction, dtype=np.float64)
    index = float(refractive_index)
    rng = np.random.default_rng(rng_seed)
    for start in range(0, sample_count, batch_size):
        chunk = haar_rotations(min(batch_size, sample_count - start), rng)
        domain = path_domain_batch(chunk, faces, incident, index)
        rows = np.flatnonzero(domain.valid)
        yield chunk[rows], domain.direction[rows]


def haar_domain_samples(
    faces: Sequence[int], incident_direction: np.ndarray, refractive_index: float, **stream: int
) -> tuple[np.ndarray, np.ndarray]:
    """All of :func:`haar_domain_batches` at once: valid poses ``(M, 3, 3)`` and outgoing directions ``(M, 3)``."""
    batches = list(haar_domain_batches(faces, incident_direction, refractive_index, **stream))
    return (
        np.concatenate([rotations for rotations, _ in batches]) if batches else np.zeros((0, 3, 3)),
        np.concatenate([directions for _, directions in batches]) if batches else np.zeros((0, 3)),
    )


@dataclass(frozen=True)
class Rank0Estimate:
    """Monte Carlo estimate of the point-mass weight of a rank-0 path (module docstring item 4).

    ``value`` is ``E_Haar[[valid] rho_H A_P T_P]`` (Haar-probability
    normalisation, the one the pixel values of a rank-2 path use);
    ``value_dvol_g`` the same mass as an integral against ``dVol_g``;
    ``error_estimate`` the standard error of the sample mean;
    ``valid_fraction`` the fraction of Haar poses inside the smooth domain of
    the path (before the finite-crystal footprint gate).
    """

    path_id: str
    value: float
    error_estimate: float
    valid_fraction: float
    sample_count: int
    rng_seed: int

    @property
    def value_dvol_g(self) -> float:
        return self.value / HAAR_TO_DVOL_G_FACTOR

    def provenance(self) -> dict[str, Any]:
        return {
            "path_id": self.path_id,
            "value": self.value,
            "value_dvol_g": self.value_dvol_g,
            "error_estimate": self.error_estimate,
            "valid_fraction": self.valid_fraction,
            "sample_count": self.sample_count,
            "rng_seed": self.rng_seed,
            "normalization": "E_Haar[[domain valid] rho_pose * entry_measure * fresnel_transmission]",
        }


def estimate_rank0_contribution(
    crystal: HexPrism,
    faces: Sequence[int],
    incident_direction: np.ndarray,
    refractive_index: float,
    pose_density: PoseDensity,
    *,
    rng_seed: int = RANK0_RNG_SEED,
    sample_count: int = RANK0_SAMPLE_COUNT,
    batch_size: int = RANK0_BATCH_SIZE,
) -> Rank0Estimate:
    """Sample ``sample_count`` Haar poses (:func:`.so3.haar_rotations`, ``batch_size`` at a time) and average the integrand.

    Per pose the integrand is ``[path_domain valid] * rho_H * entry_measure *
    fresnel_transmission`` with the three factors read from their single
    authorities (:func:`.optics.path_domain_batch`,
    :func:`.geometry.entry_measure_batch`,
    :func:`.optics.fresnel_transmission_path_batch`, the pose density's
    ``evaluate_batch``); the mean and its standard error are accumulated
    exactly over the batches (sum and sum of squares).
    """
    faces = normalize_faces(faces)
    if halo_map_rank(crystal, faces) != 0:
        raise ValueError(f"{path_id_of(faces)} is not a rank-0 path")
    incident = np.asarray(incident_direction, dtype=np.float64)
    index = float(refractive_index)
    total = 0.0
    total_squares = 0.0
    valid_count = 0
    for valid, _ in haar_domain_batches(faces, incident, index, sample_count=sample_count, rng_seed=rng_seed, batch_size=batch_size):
        valid_count += len(valid)
        if len(valid) == 0:
            continue
        integrand = (
            np.asarray(pose_density.evaluate_batch(valid), dtype=np.float64)
            * entry_measure_batch(valid, faces, incident, crystal, n_ice=index)
            * fresnel_transmission_path_batch(valid, faces, incident, index)
        )
        total += float(integrand.sum())
        total_squares += float((integrand * integrand).sum())
    mean = total / sample_count
    variance = max(total_squares / sample_count - mean * mean, 0.0)
    return Rank0Estimate(
        path_id=path_id_of(faces),
        value=mean,
        error_estimate=float(np.sqrt(variance / sample_count)),
        valid_fraction=valid_count / sample_count,
        sample_count=int(sample_count),
        rng_seed=int(rng_seed),
    )


# ---- class scene and class pixel -----------------------------------------------


def sun_pixel(render: Mapping[str, Any], incident_direction: np.ndarray) -> tuple[int, int] | None:
    """``(row, column)`` of the pixel containing the sun (``-incident_direction``), or ``None`` if outside the window."""
    sky = -np.asarray(incident_direction, dtype=np.float64)
    try:
        u, v = project_linear(sky, **render)
    except ValueError:
        return None
    column, row = int(np.floor(u)), int(np.floor(v))
    if 0 <= column < int(render["width"]) and 0 <= row < int(render["height"]):
        return row, column
    return None


def pixel_solid_angle(render: Mapping[str, Any], row: int, column: int) -> float:
    """Solid angle (steradian) of pixel ``(row, column)`` of the linear camera, centre approximation.

    On the tangent plane at distance ``scale`` (:func:`.camera.linear_scale`)
    a unit pixel subtends ``cos^3(theta) / scale^2`` where ``theta`` is the
    angle between the pixel centre and the optical axis; the variation across
    one pixel is second order in the pixel size.
    """
    direction = linear_pixel_sky_direction(row, column, **render)
    axis = camera_rotation(render.get("view")) @ np.array([0.0, 0.0, 1.0])
    cosine = float(np.clip(direction @ axis, -1.0, 1.0))
    return float(cosine**3 / linear_scale(float(render["fov_deg"]), int(render["width"]), int(render["height"])) ** 2)


@dataclass(frozen=True)
class ClassScene:
    """A :class:`PathClass` bound to one scene: one :class:`.strip_pixel.StripScene` per rank-2 member.

    Every member scene shares the constants, the pose density and the render
    window; its seeds are the class's :func:`store_plan` stores of
    ``seed_store_n`` points, each member posed through its group's element
    (module docstring item 3).  A rank-0 class has no member scenes
    (``member_scenes`` is empty) and is estimated from the Haar stream
    ``(rank0_rng_seed, rank0_sample_count)`` instead
    (:func:`estimate_rank0_contribution`).  ``sun_direction`` is ``s_hat``;
    :attr:`incident_direction` the propagation ``-s_hat``.
    """

    path_class: PathClass
    sun_direction: np.ndarray
    refractive_index: float
    crystal: HexPrism
    pose_density: PoseDensity
    render: Mapping[str, Any]
    seed_store_n: int
    rank0_sample_count: int
    rank0_rng_seed: int
    member_scenes: Mapping[Faces, StripScene]

    def __post_init__(self) -> None:
        object.__setattr__(self, "sun_direction", np.asarray(self.sun_direction, dtype=np.float64))
        object.__setattr__(self, "refractive_index", float(self.refractive_index))
        expected = () if self.path_class.halo_map_rank == 0 else self.path_class.members
        if tuple(self.member_scenes) != expected:
            raise ValueError("member_scenes must cover exactly the class members (none for a rank-0 class)")
        for member, scene in self.member_scenes.items():
            if scene.faces != member:
                raise ValueError(f"scene of member {path_id_of(member)} is of path {scene.path_id!r}")
            if scene.seeds.n != self.seed_store_n:
                raise ValueError(f"member {path_id_of(member)} seeds come from a store of {scene.seeds.n} points, not {self.seed_store_n}")
            if not np.array_equal(scene.incident_direction, self.incident_direction) or scene.refractive_index != self.refractive_index:
                raise ValueError(f"member {path_id_of(member)} scene constants differ from the class scene")

    @property
    def incident_direction(self) -> np.ndarray:
        """The propagation direction ``s = -s_hat`` of the optics."""
        return incident_direction_from_sun(self.sun_direction)

    @property
    def sun_pixel(self) -> tuple[int, int] | None:
        return sun_pixel(self.render, self.incident_direction)

    @property
    def sun_in_field_of_view(self) -> bool:
        return self.sun_pixel is not None


def build_class_scene(
    path_class: PathClass,
    *,
    sun_direction: np.ndarray,
    refractive_index: float,
    crystal: HexPrism,
    pose_density: PoseDensity,
    render: Mapping[str, Any],
    seed_store_n: int = DEFAULT_SEED_STORE_N,
    seed_store_cache_dir: Path | None = None,
    seed_stores: Mapping[tuple[Faces, ...], S2EventStore] | None = None,
    rank0_sample_count: int = RANK0_SAMPLE_COUNT,
    rank0_rng_seed: int = RANK0_RNG_SEED,
) -> ClassScene:
    """Build the member scenes of ``path_class`` with :func:`.strip_pixel.build_strip_scene`.

    The seeds follow :func:`store_plan`: per :class:`StoreGroup` one store
    (``seed_stores[group.members]`` if a caller built or loaded it, else
    :func:`.strip_pixel.seed_store` of ``seed_store_n`` points, cached in
    ``seed_store_cache_dir`` if given), and every member of a
    :class:`Transport` of that group seeds from it through the transport's
    ``g``.  A rank-0 class builds nothing.
    """
    stores = dict(seed_stores or {})
    sun = np.asarray(sun_direction, dtype=np.float64)
    member_scenes: dict[Faces, StripScene] = {}
    for group in store_plan(path_class, crystal):
        store = stores.get(group.members)
        if store is None:
            store = seed_store(crystal, refractive_index, group.members, seed_store_n, cache_dir=seed_store_cache_dir)
        for transport in group.transports:
            for member in transport.members:
                member_scenes[member] = build_strip_scene(
                    member,
                    sun_direction=sun,
                    refractive_index=refractive_index,
                    crystal=crystal,
                    pose_density=pose_density,
                    render=render,
                    seeds=StoreSeeds(store, member, sun, transport.g),
                )
    return ClassScene(
        path_class=path_class,
        sun_direction=sun,
        refractive_index=refractive_index,
        crystal=crystal,
        pose_density=pose_density,
        render=dict(render),
        seed_store_n=int(seed_store_n),
        rank0_sample_count=int(rank0_sample_count),
        rank0_rng_seed=int(rank0_rng_seed),
        member_scenes={member: member_scenes[member] for member in path_class.members if member in member_scenes},
    )


def canonical_class_scene(
    representative: Sequence[int] = (3, 5),
    *,
    seed_store_n: int = DEFAULT_SEED_STORE_N,
    seed_stores: Mapping[tuple[Faces, ...], S2EventStore] | None = None,
    rank0_sample_count: int = RANK0_SAMPLE_COUNT,
    rank0_rng_seed: int = RANK0_RNG_SEED,
    pose_density: PoseDensity | None = None,
    render: Mapping[str, Any] | None = None,
    crystal: HexPrism | None = None,
) -> ClassScene:
    """The ch06 canonical scene for the class of ``representative`` (``canonical_strip_scene`` per member).

    ``pose_density`` / ``render`` override the canonical column density and
    the canonical strip window (the latter to put the sun inside the window
    for a rank-0 class); ``crystal`` overrides ``canonical_crystal()`` the way
    :func:`.strip_pixel.canonical_strip_scene` does (tests keep fixtures
    recorded on the pre-2026-09-20 ``h/a = 1`` crystal).
    """
    crystal = canonical_crystal() if crystal is None else crystal
    return build_class_scene(
        build_path_class(crystal, representative),
        sun_direction=canonical_sun_direction(),
        refractive_index=CANONICAL_REFRACTIVE_INDEX,
        crystal=crystal,
        pose_density=canonical_pose_density() if pose_density is None else pose_density,
        render=CANONICAL_RENDER if render is None else render,
        seed_store_n=seed_store_n,
        seed_stores=seed_stores,
        rank0_sample_count=rank0_sample_count,
        rank0_rng_seed=rank0_rng_seed,
    )


@dataclass(frozen=True)
class ClassDiscoveryResult:
    """Per-member component discovery of a class for one target direction (no quadrature).

    ``per_member`` maps each rank-2 member to its
    :class:`.discovery.ComponentDiscoveryResult`; ``rank0_estimate`` is set
    for a rank-0 class instead (module docstring item 4; ``None`` when the sun
    is outside the window).  ``completeness`` is ``"complete"`` iff every
    member's discovery is.
    """

    path_class: PathClass
    per_member: Mapping[Faces, ComponentDiscoveryResult]
    rank0_estimate: Rank0Estimate | None
    completeness: Completeness

    @property
    def component_count(self) -> int:
        return sum(result.component_count for result in self.per_member.values())


def discover_class_components(
    target_direction: np.ndarray,
    scene: ClassScene,
    options: PixelOptions | None = None,
    *,
    extra_seeds: Mapping[Faces, Sequence[np.ndarray]] | None = None,
) -> ClassDiscoveryResult:
    """:func:`.discovery.discover_components` on every member scene (rank-0: the point-mass estimate)."""
    options = options or PixelOptions()
    path_class = scene.path_class
    if path_class.halo_map_rank == 0:
        estimate = (
            estimate_rank0_contribution(
                scene.crystal,
                path_class.representative,
                scene.incident_direction,
                scene.refractive_index,
                scene.pose_density,
                rng_seed=scene.rank0_rng_seed,
                sample_count=scene.rank0_sample_count,
            )
            if scene.sun_in_field_of_view
            else None
        )
        return ClassDiscoveryResult(path_class, {}, estimate, "complete")
    target = np.asarray(target_direction, dtype=np.float64)
    seeds = extra_seeds or {}
    per_member = {
        member: discover_components(
            target,
            member_scene.seeds,
            template=member_scene.discovery_template,
            extra_seeds=tuple(seeds.get(member, ())),
            **options.discovery_kwargs(),
        )
        for member, member_scene in scene.member_scenes.items()
    }
    complete = all(result.completeness == "complete" for result in per_member.values())
    return ClassDiscoveryResult(path_class, per_member, None, "complete" if complete else "unknown")


@dataclass(frozen=True)
class ClassPixelResult:
    """One pixel of a class: member results summed, provenance of every contribution.

    ``value`` / ``error_estimate`` are the linear sums over the members'
    :class:`.strip_pixel.PixelResult` (a rank-0 class: the pixel-averaged
    point mass on the sun pixel, ``0.0`` elsewhere, see module docstring);
    ``members`` keeps every member's full :class:`.strip_pixel.PixelResult`
    (its ``components`` are the :class:`.strip_pixel.ComponentRecord` tuples
    the strip pipeline reports); ``completeness`` is ``"complete"`` iff every
    member's is.  ``provenance`` is JSON-serialisable: the class, the
    sampling policy, each member's ``value`` / ``error_estimate`` /
    ``completeness`` / ``component_count`` / ``arc_count`` /
    ``incomplete_count`` (the :class:`.strip_pixel.PixelResult` names), and
    the rank-0 block with its field-of-view verdict.
    """

    row: int
    column: int
    value: float
    error_estimate: float
    members: Mapping[Faces, PixelResult]
    rank0_estimate: Rank0Estimate | None
    completeness: Completeness
    events: Mapping[str, int]
    timings: Mapping[str, float]
    provenance: Mapping[str, Any]

    @property
    def components(self) -> Mapping[Faces, tuple[ComponentRecord, ...]]:
        return {member: result.components for member, result in self.members.items()}

    @property
    def component_count(self) -> int:
        return sum(result.component_count for result in self.members.values())

    @property
    def warm_seeds(self) -> Mapping[Faces, tuple[np.ndarray, ...]]:
        """Per-member converged poses for a neighbouring pixel (:attr:`.strip_pixel.PixelResult.warm_seeds`)."""
        return {member: result.warm_seeds for member, result in self.members.items()}


def _rank0_pixel(scene: ClassScene, row: int, column: int, total_start: float) -> ClassPixelResult:
    path_class = scene.path_class
    sun = scene.sun_pixel
    block: dict[str, Any] = {"sun_in_field_of_view": sun is not None, "sun_pixel": list(sun) if sun else None}
    estimate: Rank0Estimate | None = None
    value = 0.0
    error = 0.0
    if sun is None:
        block["reason"] = "render window does not contain the sun direction; point mass contributes 0 without sampling"
    else:
        estimate = estimate_rank0_contribution(
            scene.crystal,
            path_class.representative,
            scene.incident_direction,
            scene.refractive_index,
            scene.pose_density,
            rng_seed=scene.rank0_rng_seed,
            sample_count=scene.rank0_sample_count,
        )
        block["estimate"] = estimate.provenance()
        block["this_pixel_contains_sun"] = sun == (int(row), int(column))
        if sun == (int(row), int(column)):
            solid_angle = pixel_solid_angle(scene.render, row, column)
            block["pixel_solid_angle_sr"] = solid_angle
            value = estimate.value / solid_angle
            error = estimate.error_estimate / solid_angle
        else:
            block["reason"] = "point mass lies in another pixel of the window; this pixel contributes 0"
    timings = {name: 0.0 for name in STAGE_NAMES}
    timings["total_s"] = time.perf_counter() - total_start
    return ClassPixelResult(
        row=int(row),
        column=int(column),
        value=float(value),
        error_estimate=float(error),
        members={},
        rank0_estimate=estimate,
        completeness="complete",
        events={name: 0 for name in EVENT_NAMES},
        timings=timings,
        provenance={
            **path_class.provenance(),
            "rank0_sampling": {"sample_count": scene.rank0_sample_count, "rng_seed": scene.rank0_rng_seed},
            "per_member": {},
            "rank0": block,
        },
    )


def render_class_pixel(
    scene: ClassScene,
    row: int,
    column: int,
    options: PixelOptions,
    *,
    target: np.ndarray | None = None,
    warm_seeds: Mapping[Faces, Sequence[np.ndarray]] | None = None,
) -> ClassPixelResult:
    """:func:`.strip_pixel.render_pixel` on every member scene, summed (module docstring items 3-4).

    ``target`` overrides the pixel-centre direction; ``warm_seeds`` maps a
    member to the converged poses of a neighbouring pixel of *that* member
    (:attr:`ClassPixelResult.warm_seeds`).  A rank-0 class ignores both.
    """
    total_start = time.perf_counter()
    path_class = scene.path_class
    if path_class.halo_map_rank == 0:
        return _rank0_pixel(scene, row, column, total_start)
    seeds = warm_seeds or {}
    members: dict[Faces, PixelResult] = {}
    events: Counter = Counter()
    timings: Counter = Counter({name: 0.0 for name in STAGE_NAMES})
    for member, member_scene in scene.member_scenes.items():
        result = render_pixel(
            member_scene, row, column, options, target=target, warm_seeds=tuple(seeds.get(member, ())) or None
        )
        members[member] = result
        events.update(result.events)
        for name in STAGE_NAMES:
            timings[name] += result.timings[name]
    timings["total_s"] = time.perf_counter() - total_start
    complete = all(result.completeness == "complete" for result in members.values())
    return ClassPixelResult(
        row=int(row),
        column=int(column),
        value=float(sum(result.value for result in members.values())),
        error_estimate=float(sum(result.error_estimate for result in members.values())),
        members=members,
        rank0_estimate=None,
        completeness="complete" if complete else "unknown",
        events={name: int(events[name]) for name in EVENT_NAMES},
        timings={name: float(timings[name]) for name in STAGE_NAMES},
        provenance={
            **path_class.provenance(),
            "seed_store": {"N": scene.seed_store_n, "plan": [group.as_json() for group in store_plan(path_class, scene.crystal)]},
            "per_member": {
                path_id_of(member): {
                    "value": result.value,
                    "error_estimate": result.error_estimate,
                    "completeness": result.completeness,
                    "component_count": result.component_count,
                    "arc_count": result.arc_count,
                    "incomplete_count": result.incomplete_count,
                    "pool_count": result.pool_count,
                    "admissible_count": result.admissible_count,
                }
                for member, result in members.items()
            },
            "rank0": None,
        },
    )


__all__ = [
    "ClassDiscoveryResult",
    "ClassPixelResult",
    "ClassScene",
    "PathClass",
    "RANK0_BATCH_SIZE",
    "RANK0_RNG_SEED",
    "RANK0_SAMPLE_COUNT",
    "Rank0Estimate",
    "StoreGroup",
    "Transport",
    "build_class_scene",
    "build_path_class",
    "canonical_class_scene",
    "discover_class_components",
    "estimate_rank0_contribution",
    "haar_domain_batches",
    "haar_domain_samples",
    "hexprism_symmetry_matrices",
    "path_class_symmetry",
    "pbd_orbit_hexprism",
    "phi_key",
    "pixel_solid_angle",
    "render_class_pixel",
    "single_path_class",
    "store_plan",
    "sun_pixel",
]
