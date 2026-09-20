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
   per member, all built from the same ``(rng_seed, sample_count)`` Haar
   stream (:class:`ClassScene`), :func:`.strip_pixel.render_pixel` per member,
   contributions summed.  Members are neither merged nor assumed equal: the
   symmetric-density identity "3-7 equals 3-5 pointwise" is a *test* of the
   canonical scene, not an assumption of the code.
4. A rank-0 class (``M = I`` and ``W = 0``: ``1-2``, ``3-6``, ...; ch8 A0-06)
   sends every pose to the sun direction.  Its contribution is a point mass
   at ``-incident_direction`` of total weight

       m = E_Haar[ [R in V_P] rho_H(R) A_P(R) T_P(R) ]
         = (1 / 8 pi^2) integral_(V_P) rho_H A_P T_P dVol_g,

   estimated by :func:`estimate_rank0_contribution` from the raw Haar stream
   (same generator as the prescan; the prescan table itself keeps only the
   domain-valid poses of *its* path, which is the wrong subset here).  ``m``
   is in the same normalisation as the pixel values of a rank-2 path (the
   push-forward of ``rho_H W dmu_Haar``, whose sphere density the fiber
   quadrature reports; ``docs/phase1-math-contract.md`` section 7), so on
   the one pixel that contains the sun the pixel-averaged value is
   ``m / pixel_solid_angle``, every other pixel gets ``0.0``, and a render
   window that does not contain the sun gets ``0.0`` without sampling
   (recorded in ``provenance``).  A rank-0 class never enters the fiber
   pipeline: almost every pose is aligned, the "one-dimensional fiber"
   premise is false there.

Nothing here imports or calls Lumice.
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .camera import camera_rotation, linear_pixel_sky_direction, linear_scale, project_linear
from .canonical_scene import (
    CANONICAL_REFRACTIVE_INDEX,
    CANONICAL_RENDER,
    canonical_crystal,
    canonical_incident_direction,
    canonical_pose_density,
)
from .discovery import ComponentDiscoveryResult, discover_components
from .geometry import HexPrism, Polyhedron, entry_measure_batch, halo_map_rank, wedge_angle_deg
from .optics import fresnel_transmission_path_batch, normalize_faces, path_domain_batch, path_id_of
from .pose_density import PoseDensity
from .prescan import DEFAULT_BATCH_SIZE, DEFAULT_RNG_SEED, DEFAULT_SAMPLE_COUNT, PrescanTable, haar_rotations
from .quadrature import HAAR_TO_DVOL_G_FACTOR
from .strip_pixel import (
    EVENT_NAMES,
    STAGE_NAMES,
    ComponentRecord,
    PixelOptions,
    PixelResult,
    StripScene,
    build_strip_scene,
    render_pixel,
)

Faces = tuple[int, ...]
Completeness = str  # "complete" | "unknown"

# ---- PBD orbit ---------------------------------------------------------------


def _hexprism_symmetry_matrices() -> tuple[np.ndarray, ...]:
    """The 24 orthogonal matrices of ``D6h`` in the body frame (c axis = +z, face 3 normal = +x).

    Generated as the closure of three generators -- the 60-degree rotation
    about z, the vertical mirror ``y -> -y`` and the horizontal mirror
    ``z -> -z`` -- so the element list is derived, not typed in.
    """
    angle = np.radians(60.0)
    generators = [
        np.array([[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]]),
        np.diag([1.0, -1.0, 1.0]),
        np.diag([1.0, 1.0, -1.0]),
    ]
    elements: list[np.ndarray] = [np.eye(3)]
    frontier = [np.eye(3)]
    while frontier:
        next_frontier: list[np.ndarray] = []
        for element in frontier:
            for generator in generators:
                candidate = generator @ element
                if not any(np.allclose(candidate, known, atol=1e-12) for known in elements):
                    elements.append(candidate)
                    next_frontier.append(candidate)
        frontier = next_frontier
    if len(elements) != 24:
        raise RuntimeError(f"D6h closure produced {len(elements)} elements, expected 24")
    return tuple(elements)


_D6H = _hexprism_symmetry_matrices()


def pbd_orbit_hexprism(faces: Sequence[int], crystal: HexPrism | None = None) -> frozenset[Faces]:
    """Orbit of a face sequence under ``D6h`` (Lumice ``PBD``) as a set of face sequences.

    Each group element ``g`` maps face ``f`` to the face whose outward normal
    is ``g @ n_f`` (normals from ``crystal``, default :class:`.geometry.HexPrism`);
    the face sequence is mapped elementwise.  Only the point group acts, so
    the result does not depend on the prism's aspect ratio.
    """
    faces = normalize_faces(faces)
    crystal = HexPrism() if crystal is None else crystal
    normals = {face.number: crystal.normal(face) for face in crystal.faces}
    orbit: set[Faces] = set()
    for element in _D6H:
        image: list[int] = []
        for face in faces:
            target = element @ normals[face]
            matches = [number for number, normal in normals.items() if np.allclose(normal, target, atol=1e-9)]
            if len(matches) != 1:
                raise RuntimeError(f"symmetry image of face {face} matched {matches}")
            image.append(matches[0])
        orbit.add(tuple(image))
    return frozenset(orbit)


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


# ---- rank-0 point mass -------------------------------------------------------


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
    rng_seed: int = DEFAULT_RNG_SEED,
    sample_count: int = DEFAULT_SAMPLE_COUNT,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Rank0Estimate:
    """Sample the same Haar stream as :func:`.prescan.build_prescan_table` and average the integrand.

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
    if sample_count < 1 or batch_size < 1:
        raise ValueError("sample_count and batch_size must be positive")
    incident = np.asarray(incident_direction, dtype=np.float64)
    index = float(refractive_index)
    rng = np.random.default_rng(rng_seed)
    total = 0.0
    total_squares = 0.0
    valid_count = 0
    for start in range(0, sample_count, batch_size):
        count = min(batch_size, sample_count - start)
        chunk = haar_rotations(count, rng)
        domain = path_domain_batch(chunk, faces, incident, index)
        rows = np.flatnonzero(domain.valid)
        valid_count += int(rows.size)
        if rows.size == 0:
            continue
        valid = chunk[rows]
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

    Every member scene shares the constants, the pose density, the render
    window and the prescan sampling policy ``(prescan_rng_seed,
    prescan_sample_count)``; each has its own :class:`.prescan.PrescanTable`
    of its own path, all drawn from the same Haar stream.  A rank-0 class has
    no member scenes (``member_scenes`` is empty) and is estimated from the
    raw stream instead (:func:`estimate_rank0_contribution`).
    """

    path_class: PathClass
    incident_direction: np.ndarray
    refractive_index: float
    crystal: HexPrism
    pose_density: PoseDensity
    render: Mapping[str, Any]
    prescan_sample_count: int
    prescan_rng_seed: int
    member_scenes: Mapping[Faces, StripScene]

    def __post_init__(self) -> None:
        object.__setattr__(self, "incident_direction", np.asarray(self.incident_direction, dtype=np.float64))
        object.__setattr__(self, "refractive_index", float(self.refractive_index))
        expected = () if self.path_class.halo_map_rank == 0 else self.path_class.members
        if tuple(self.member_scenes) != expected:
            raise ValueError("member_scenes must cover exactly the class members (none for a rank-0 class)")
        for member, scene in self.member_scenes.items():
            if scene.faces != member:
                raise ValueError(f"scene of member {path_id_of(member)} is of path {scene.path_id!r}")
            table = scene.prescan_table
            if (table.sample_count, table.rng_seed) != (self.prescan_sample_count, self.prescan_rng_seed):
                raise ValueError(f"member {path_id_of(member)} prescan table does not share the scene's Haar stream")
            if not np.array_equal(scene.incident_direction, self.incident_direction) or scene.refractive_index != self.refractive_index:
                raise ValueError(f"member {path_id_of(member)} scene constants differ from the class scene")

    @property
    def sun_pixel(self) -> tuple[int, int] | None:
        return sun_pixel(self.render, self.incident_direction)

    @property
    def sun_in_field_of_view(self) -> bool:
        return self.sun_pixel is not None


def build_class_scene(
    path_class: PathClass,
    *,
    incident_direction: np.ndarray,
    refractive_index: float,
    crystal: HexPrism,
    pose_density: PoseDensity,
    render: Mapping[str, Any],
    prescan_sample_count: int = DEFAULT_SAMPLE_COUNT,
    prescan_rng_seed: int = DEFAULT_RNG_SEED,
    prescan_tables: Mapping[Faces, PrescanTable] | None = None,
) -> ClassScene:
    """Build the member scenes of ``path_class`` with :func:`.strip_pixel.build_strip_scene`.

    ``prescan_tables`` (by member) supplies tables a driver built or loaded;
    a member without one gets a table built from ``(prescan_rng_seed,
    prescan_sample_count)``.  A rank-0 class builds nothing.
    """
    tables = dict(prescan_tables or {})
    member_scenes: dict[Faces, StripScene] = {}
    if path_class.halo_map_rank != 0:
        for member in path_class.members:
            member_scenes[member] = build_strip_scene(
                member,
                incident_direction=incident_direction,
                refractive_index=refractive_index,
                crystal=crystal,
                pose_density=pose_density,
                render=render,
                prescan_table=tables.get(member),
                prescan_sample_count=prescan_sample_count,
                prescan_rng_seed=prescan_rng_seed,
            )
    return ClassScene(
        path_class=path_class,
        incident_direction=incident_direction,
        refractive_index=refractive_index,
        crystal=crystal,
        pose_density=pose_density,
        render=dict(render),
        prescan_sample_count=int(prescan_sample_count),
        prescan_rng_seed=int(prescan_rng_seed),
        member_scenes=member_scenes,
    )


def canonical_class_scene(
    representative: Sequence[int] = (3, 5),
    *,
    prescan_sample_count: int = DEFAULT_SAMPLE_COUNT,
    prescan_rng_seed: int = DEFAULT_RNG_SEED,
    pose_density: PoseDensity | None = None,
    render: Mapping[str, Any] | None = None,
    prescan_tables: Mapping[Faces, PrescanTable] | None = None,
) -> ClassScene:
    """The ch06 canonical scene for the class of ``representative`` (``canonical_strip_scene`` per member).

    ``pose_density`` / ``render`` override the canonical column density and
    the canonical strip window (the latter to put the sun inside the window
    for a rank-0 class).
    """
    crystal = canonical_crystal()
    return build_class_scene(
        build_path_class(crystal, representative),
        incident_direction=canonical_incident_direction(),
        refractive_index=CANONICAL_REFRACTIVE_INDEX,
        crystal=crystal,
        pose_density=canonical_pose_density() if pose_density is None else pose_density,
        render=CANONICAL_RENDER if render is None else render,
        prescan_sample_count=prescan_sample_count,
        prescan_rng_seed=prescan_rng_seed,
        prescan_tables=prescan_tables,
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
                rng_seed=scene.prescan_rng_seed,
                sample_count=scene.prescan_sample_count,
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
            member_scene.crystal,
            member_scene.prescan_table,
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
            rng_seed=scene.prescan_rng_seed,
            sample_count=scene.prescan_sample_count,
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
            "prescan": {"sample_count": scene.prescan_sample_count, "rng_seed": scene.prescan_rng_seed},
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
            "prescan": {"sample_count": scene.prescan_sample_count, "rng_seed": scene.prescan_rng_seed},
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
    "Rank0Estimate",
    "build_class_scene",
    "build_path_class",
    "canonical_class_scene",
    "discover_class_components",
    "estimate_rank0_contribution",
    "pbd_orbit_hexprism",
    "pixel_solid_angle",
    "render_class_pixel",
    "sun_pixel",
]
