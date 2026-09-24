"""The S^2 event store: per-path weight fields sampled on ``u = R^-1 s_hat`` and sorted by deviation.

Convention (``docs/conventions.md``; framework theorem 8 of the writing
series): ``s_hat`` points *toward* the sun and ``u = R^-1 s_hat`` is the sun
in the crystal frame.  The optics take the propagation direction
``-s_hat`` (:func:`.camera.incident_direction_from_sun`, the contract's
``s``), so the body-frame incoming ray is ``-u``, ``phi`` is the body-frame
outgoing propagation direction ``Phi_P(-u)`` and the deviation is
``D = angle(phi, -u) = angle(-Phi_P(-u), u)``, the framework's ``D_P(u)``.

Roadmap section 4.1(a): validity, entry measure ``A``, Fresnel transmission
``T``, ``phi`` and ``D`` of a fixed ray path depend on the pose ``R`` only
through ``u``.  Section 4.2 discretises the level-set integral with
a store of events on ``S^2``: ``N`` points (the antipodal Fibonacci lattice
by default, :func:`store_lattice`, ``4 pi / N`` each), one rotation per point, the production batch evaluators
(:func:`.optics.path_domain_batch`, :func:`.optics.fresnel_transmission_path_batch`,
:func:`.geometry.entry_measure_batch`), only ``w = A T > 0`` events kept,
sorted by ``D``.  A renderer takes the band ``[delta_lo, delta_hi]`` of a
pixel with :meth:`S2EventStore.band_slice` and rebuilds each event's pose
with :func:`event_rotations`.  This module is the store only; the band-sum
estimator on top of it belongs to the renderer.

Provenance: migrated from ``scripts/probe_band_sum.py`` (task
``band-sum-quadrature-probe``) with the numerics unchanged -- ``CHUNK``,
chunk order, the stable argsort and the float64 evaluation are the ones the
task 13/14 stores were built with, and ``tests/test_s2_store.py`` pins the
rebuild of those stores bit for bit.  Those stores (and ``SCHEMA_VERSION``
1) recorded ``u = R^-1 s`` with the propagation direction ``s = -s_hat``;
schema 2 (task ``notation-alignment``) records ``u = R^-1 s_hat``, which is
the old ``u`` negated.  The default lattice is the antipode of the old one
and every formula below is the old one with ``(u, s)`` replaced by
``(-u, -s_hat)``, under which frames, cross and outer products are
unchanged bit for bit: every pose, ``phi``, ``D``, ``w`` and band sum is
the schema 1 one exactly, only the sign of the stored ``u`` differs.
:meth:`S2EventStore.load` refuses a schema 1 store.

Members and ``Phi`` groups.  A store is built for a tuple of ``members``
(face sequences).  One member is the ordinary per-path store.  Several
members must share one :func:`.path_class.phi_key` (same fold matrix, entry
normal and unfolded exit normal, hence the same ``Phi(u)`` and ``D(u)``);
their weights are then summed on the same ``u`` (``w_Phi = sum_m w_m``) and
an event is kept where any member has ``w_m > 0``.

Symmetry transport.  For any crystal symmetry ``g`` of ``D6h``, proper or
improper, mapping the representative's faces onto a member's
(:func:`.path_class.path_class_symmetry`), section 4.1(d) gives
``Phi_member(-u) = g Phi_rep(-g^-1 u)`` with ``D``, ``w`` and the valid domain
unchanged: the member's events are the representative's with ``u' = g u``,
``phi' = g phi``.  The pose of a transported event is rebuilt from
``(g u, g phi, D)`` and the pixel's azimuth by :func:`event_rotations`,
whose two orthonormal frames always give a rotation, whatever ``det g``
(a mirror needs no store of its own on ``S^2``; that restriction belongs to
Phase I, where ``R g^-1`` would have to stay in SO(3)).  The production
interface is :func:`transported_rotations`, the closed form of that rebuild
on the representative's poses: ``L_g R g^T`` with ``L_g = I`` for a proper
``g`` and the reflection in the plane of ``s_hat`` and the pixel centre for an
improper one (equal to :func:`event_rotations` of the transported events,
pinned by ``tests/test_s2_store_symmetry.py`` for all 24 elements), so one
store in memory serves the whole ``D6h`` orbit.  Symmetry saves repeated
evaluation, not samples: the transported events are the same precomputed
events.

Memory layout: every array is C-contiguous, ``u`` / ``phi`` ``(K, 3)``,
``D`` / ``w`` (and ``iw`` for a non-uniform sampler) ``(K,)``, in the
store's ``dtype`` (float64 by default; the fields are always evaluated in
float64 and cast once after sorting).  :meth:`S2EventStore.band_slice`
returns views into the store, not copies.  float32 halves the memory
(102.5 -> 51.3 MB for the task 13 ``3-5`` store at ``N = 1e7``); on its
column 126 (511 lit pixels, canonical density) the band-sum estimates move
by a median ``4.5e-8`` and an RMS ``3.1e-4`` relative to float64, but by up
to ``5.5e-3`` on the worst pixel -- comparable to the float64 estimate's own
median error against the Phase I reference (``7.1e-3``), so float32 is not
the default (2026-09-23, task ``s2-event-store``).

Independent of the sun (schema 3, task ``s2-store-schema-3``): the
arrays depend on the crystal, the ``Phi`` group, the refractive index and
the sampling only, never on ``s_hat``, so one store serves every sun
direction.  The build aligns ``R u = s_hat`` to one fixed reference
direction (``_REFERENCE_SUN_DIRECTION``) and :class:`S2StoreSpec` records
no sun.

Disk cache: ``<base_dir>/<cache_key>/<name>.npy`` per array (``D``, ``u``,
``phi``, ``w``, ``iw``) + ``provenance.json`` with each array's SHA-256 and
size.  :meth:`S2EventStore.load` reads them (hash checked) or maps them
read-only (``mmap_mode="r"``, size checked; :meth:`S2EventStore.verify`
hashes on demand).  :func:`build_or_load` builds a missing store bucket by
bucket in ``D`` straight into the cache (:func:`_build_into`), so the build
never holds all events in memory.
The key hashes the build parameters (:meth:`S2StoreSpec.build_parameters`)
and ``SCHEMA_VERSION``; ``SCHEMA_VERSION`` is bumped by hand when the build
algorithm changes.  The git commit is recorded in the provenance for
forensics but deliberately *not* part of the key or of the comparison: a
large store (``N = 1e8`` takes minutes to hours) must not be invalidated by
unrelated commits.  Unlike :func:`.prescan.build_or_load_prescan_table`
(which rebuilds on mismatch), :func:`build_or_load` refuses a cache whose
recorded parameters differ from the request, and :meth:`S2EventStore.load`
refuses an array whose SHA-256 differs from the recorded one --
neither silently rebuilds nor silently reuses.

Nothing here imports or calls Lumice.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json
import platform
import resource
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from . import geometry, optics
from .camera import incident_direction_from_sun, sun_direction
from .geometry import HexPrism, Polyhedron
from .optics import normalize_faces, path_id_of
from .path_class import phi_key
from .provenance import git_commit, sha256_of

# 3: no sun direction in the spec (the arrays do not depend on it), one .npy per array, task s2-store-schema-3;
# 2: u = R^-1 s_hat (toward the sun) with the sun direction recorded, task notation-alignment;
# 1: u = R^-1 s (propagation).  Schemas 1 and 2 are refused on load.
SCHEMA_VERSION = 3
CHUNK = 250_000  # rotations per batch call: ~0.5 GB transient in the eager jax.vmap (task 13/14 value)
DEFAULT_CACHE_DIR = Path("artifacts/s2-store")
DEFAULT_BUCKET_COUNT = 1024  # equal-width D buckets of the build (0.18 deg on [0, pi]); an I/O knob, not in the key
PROVENANCE_FILE = "provenance.json"
_STAGING_STALE_SECONDS = 24 * 3600  # builds are documented to take minutes to hours; idle past this is a dead process, not one still writing
FIBONACCI_SAMPLING = (
    "antipodal Fibonacci lattice on S^2 (u_i = -f_i, f_i the equal-area spiral z_i = 1 - (2i+1)/N, "
    "golden-angle azimuth)"
)
ROTATION_PER_POINT = (
    "R = [s_hat, p_s, s_hat x p_s][u, p_u, u x p_u]^T (any R with R u = s_hat, s_hat toward the sun; section 4.1(a))"
)
RANDOM_SEED = 20260923
# The reference ``s_hat`` of the build (``R u = s_hat``).  A store does not depend on the sun: validity, ``A``,
# ``T``, ``phi`` and ``D`` depend on the pose only through ``u = R^-1 s_hat`` (section 4.1(a); measured in
# ``docs/phase2.md`` section 1.1 and pinned by ``tests/test_s2_store.py``), so any direction gives the same events
# to round-off.  This one is numerically the canonical sun (altitude 15 deg, azimuth 0) -- a coincidence chosen to
# keep the schema 2 canonical builds bit for bit, not a dependence on the canonical scene (nothing imported from it).
REFERENCE_SUN_ALTITUDE_DEG = 15.0
REFERENCE_SUN_AZIMUTH_DEG = 0.0
_REFERENCE_SUN_DIRECTION = sun_direction(REFERENCE_SUN_ALTITUDE_DEG, REFERENCE_SUN_AZIMUTH_DEG)
DTYPES = ("float64", "float32")

Faces = tuple[int, ...]
# ``sampler(first, stop)`` -> points ``first:stop`` of an ``N``-point set on ``S^2`` and, for a
# non-uniform set, ``1 / (4 pi q(u))`` per point (``q`` its density w.r.t. area; ``None`` = uniform).
PointSampler = Callable[[int, int], tuple[np.ndarray, np.ndarray | None]]


def max_rss_mb() -> float:
    """Process peak resident set size (macOS reports bytes, Linux kilobytes)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 2**20 if platform.system() == "Darwin" else peak / 2**10


# ---------------------------------------------------------------- geometry
def fibonacci_sphere(n: int, start: int = 0, stop: int | None = None) -> np.ndarray:
    """Points ``start:stop`` of the ``n``-point Fibonacci lattice on ``S^2`` (equal-area spiral, ``(k, 3)``)."""
    i = np.arange(start, n if stop is None else stop, dtype=np.float64) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)


def store_lattice(n: int, start: int = 0, stop: int | None = None) -> np.ndarray:
    """The store's default ``u``: points ``start:stop`` of the antipodal Fibonacci lattice, ``-f_i``.

    Schema 1 sampled ``R^-1 s`` (propagation) on the lattice ``f_i``; its
    antipode keeps every schema 1 pose and event bit for bit (module docstring).
    """
    return -fibonacci_sphere(n, start, stop)


@dataclasses.dataclass(frozen=True)
class RandomSphereSampler:
    """i.i.d. uniform points on ``S^2`` (``q = 1 / 4 pi``, so no inverse weights); chunk-seeded, reproducible.

    The points are negated for the same reason as :func:`store_lattice`: the
    schema 1 draws, antipodal, so the poses are unchanged bit for bit.
    """

    seed: int = RANDOM_SEED
    description = "i.i.d. uniform on S^2 (negated normalised Gaussian triples, numpy default_rng([seed, first]) per chunk)"

    def __call__(self, first: int, stop: int) -> tuple[np.ndarray, None]:
        points = np.random.default_rng([self.seed, first]).normal(size=(stop - first, 3))
        return -(points / np.linalg.norm(points, axis=1, keepdims=True)), None


def _any_perpendicular(v: np.ndarray) -> np.ndarray:
    """A unit vector perpendicular to each row of ``v`` (cross with the least aligned axis; no degenerate case)."""
    axis = np.zeros_like(v)
    axis[np.arange(len(v)), np.argmin(np.abs(v), axis=1)] = 1.0
    p = np.cross(v, axis)
    return p / np.linalg.norm(p, axis=1, keepdims=True)


def frame(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Orthonormal right-handed frames with columns ``(first, second, first x second)``, ``(n, 3, 3)``."""
    return np.stack([first, second, np.cross(first, second)], axis=2)


def align_rotations(u: np.ndarray, sun: np.ndarray) -> np.ndarray:
    """One rotation per row with ``R u = s_hat`` (the free twist about ``s_hat`` is arbitrary, section 4.1(a))."""
    sun_rows = np.broadcast_to(sun, u.shape)
    return np.einsum("nij,nkj->nik", frame(sun_rows, _any_perpendicular(sun_rows)), frame(u, _any_perpendicular(u)))


def twist_about(axis: np.ndarray, angle: float) -> np.ndarray:
    a = axis / np.linalg.norm(axis)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)


def event_rotations(
    u: np.ndarray, phi: np.ndarray, deviation: np.ndarray, sun: np.ndarray, centre: np.ndarray
) -> np.ndarray:
    """``R_i`` with ``R_i u_i = s_hat`` and ``R_i phi_i`` at deviation ``D_i``, azimuth of ``centre``.

    The pose of each event for a pixel whose centre (outgoing) direction is
    ``centre`` (Gislen eq. 19 at ``omega = D_i``, built from two orthonormal
    frames): ``R = [s_hat, e, s_hat x e][u, f, u x f]^T`` with ``e`` the unit
    component of ``centre`` normal to ``s_hat`` and ``f`` that of ``phi``
    normal to ``u`` (``phi . (-u) = cos D``, so ``f ~ phi + cos(D) u``).
    For the events transported by a crystal symmetry ``g`` see
    :func:`transported_rotations` (module docstring).
    """
    e = centre - (centre @ sun) * sun
    e /= np.linalg.norm(e)
    f2 = phi + np.cos(deviation)[:, None] * u
    f2 /= np.linalg.norm(f2, axis=1, keepdims=True)
    world = np.stack([sun, e, np.cross(sun, e)], axis=1)
    return np.einsum("ij,nkj->nik", world, frame(u, f2))


def transported_rotations(rotations: np.ndarray, g: np.ndarray, sun: np.ndarray, centre: np.ndarray) -> np.ndarray:
    """:func:`event_rotations` of the events transported by ``g`` (``u' = g u``, ``phi' = g phi``), from their poses.

    ``rotations`` are :func:`event_rotations` of the untransported events
    for the same ``s_hat`` and ``centre``; ``g`` is any orthogonal crystal
    symmetry.  :func:`event_rotations` is ``R = W F^T`` with the world frame
    ``W`` and the event frame ``F`` of ``(u, f)``.  The transported frame is
    ``g F J`` with ``J = diag(1, 1, det g)`` (a cross product changes sign
    under a mirror), so ``R' = W J F^T g^T = L_g R g^T`` with
    ``L_g = W J W^T = I - (1 - det g) m m^T``, ``m = W e_3`` the normal of
    the plane of ``s_hat`` and ``centre``.  For a proper ``g`` this is ``R g^T``
    (``L_g`` skipped: the same floating-point operation as the proper-only
    transport of task ``band-sum-renderer``); for an improper one ``L_g`` is
    the reflection in the ``(s_hat, centre)`` plane.  The result is a rotation
    either way.
    """
    g = np.asarray(g, dtype=np.float64)
    moved = rotations @ g.T
    if np.linalg.det(g) > 0.0:
        return moved
    e = centre - (centre @ sun) * sun
    m = np.cross(sun, e / np.linalg.norm(e))
    return (np.eye(3) - 2.0 * np.outer(m, m)) @ moved


def evaluate_fields(
    rotations: np.ndarray, sun: np.ndarray, crystal: Polyhedron, index: float, members: Sequence[Faces]
) -> dict[str, np.ndarray]:
    """Production batch evaluators of the ``members`` at ``rotations``, for the sun direction ``s_hat``.

    Returns validity (any member), ``A`` and ``T`` per member (``(m, n)``),
    ``w = sum_m A_m T_m``, body-frame ``phi = Phi_P(-u)``, ``D`` and
    ``u = R^-1 s_hat``.  The evaluators take the propagation direction
    ``s = -s_hat`` (:func:`.camera.incident_direction_from_sun`, the only
    place the sign changes).  ``phi`` is
    the first member's direction: members of one ``Phi`` group share the
    closed form of the outgoing direction (entry and exit refraction on the
    same normals, the same fold matrix), so it is the same wherever any of
    them is valid.
    """
    s = incident_direction_from_sun(sun)
    valid, areas, transmissions = [], [], []
    for index_m, faces in enumerate(members):
        check = optics.path_domain_batch(rotations, faces, s, index)
        transmissions.append(optics.fresnel_transmission_path_batch(rotations, faces, s, index))
        areas.append(geometry.entry_measure_batch(rotations, faces, s, crystal, n_ice=index))
        valid.append(check.valid)
        if index_m == 0:
            direction = check.direction
    w = areas[0] * transmissions[0]
    for area, transmission in zip(areas[1:], transmissions[1:]):
        w = w + area * transmission
    u = np.einsum("nji,j->ni", rotations, sun)
    phi = np.einsum("nji,nj->ni", rotations, direction)
    deviation = np.arccos(np.clip(np.sum(phi * -u, axis=1), -1.0, 1.0))  # angle(phi, -u), -u the incoming ray
    return {
        "valid": np.any(valid, axis=0),
        "A": np.stack(areas),
        "T": np.stack(transmissions),
        "w": w,
        "phi": phi,
        "D": deviation,
        "u": u,
    }


# --------------------------------------------------------- self-checks (a02)
def self_check_psi_invariance(
    sun: np.ndarray, crystal: Polyhedron, index: float, members: Sequence[Faces], sample: int = 4000
) -> dict[str, Any]:
    """Section 4.1(a): validity, ``A``, ``T``, ``phi`` and ``D`` do not change under a twist about ``s_hat``."""
    u = store_lattice(200_000)[:: 200_000 // sample]
    base = align_rotations(u, sun)
    reference = evaluate_fields(base, sun, crystal, index, members)
    worst = {"valid_mismatch": 0, "A": 0.0, "T": 0.0, "phi": 0.0, "D": 0.0}
    for angle in (0.7, 2.1, -2.9):
        fields = evaluate_fields(np.einsum("ij,njk->nik", twist_about(sun, angle), base), sun, crystal, index, members)
        both = reference["valid"] & fields["valid"]
        worst["valid_mismatch"] += int(np.count_nonzero(reference["valid"] != fields["valid"]))
        worst["A"] = max(worst["A"], float(np.max(np.abs(fields["A"] - reference["A"]))))
        worst["T"] = max(worst["T"], float(np.max(np.abs(fields["T"] - reference["T"]))))
        worst["phi"] = max(worst["phi"], float(np.max(np.abs(fields["phi"][both] - reference["phi"][both]))))
        worst["D"] = max(worst["D"], float(np.max(np.abs(fields["D"][both] - reference["D"][both]))))
    worst["points"] = int(len(u))
    worst["valid_points"] = int(np.count_nonzero(reference["valid"]))
    worst["passed"] = bool(
        worst["valid_mismatch"] == 0 and max(worst["A"], worst["T"], worst["phi"], worst["D"]) < 1e-10
    )
    return worst


def self_check_gate_coverage(
    sun: np.ndarray, crystal: Polyhedron, index: float, members: Sequence[Faces], sample: int = 3000
) -> dict[str, int]:
    """Every ``entry_measure`` failure reason occurs on the sphere (the scalar form reports the reason)."""
    u = store_lattice(sample)
    s = incident_direction_from_sun(sun)
    counts: dict[str, int] = {}
    for faces in members:
        for rotation in align_rotations(u, sun):
            status = geometry.entry_measure(rotation, faces, s, crystal, n_ice=index).status
            counts[status] = counts.get(status, 0) + 1
    return counts


def self_check_haar_mean(
    sun: np.ndarray,
    crystal: Polyhedron,
    index: float,
    fibonacci_mean_w: float,
    members: Sequence[Faces],
    n: int = 1_000_000,
) -> dict[str, Any]:
    """``E_Haar[A T]`` from independent uniform quaternions against the Fibonacci ``sum w / N``.

    Checks the fibration claim ``u = R^-1 s_hat`` is uniform on ``S^2`` under Haar together with
    the ``4 pi / N`` area element, independently of the ``u`` parametrisation.
    """
    rng = np.random.default_rng(20260923)
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    w0, x, y, z = q.T
    rotations = np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w0), 2 * (x * z + y * w0)], axis=1),
            np.stack([2 * (x * y + z * w0), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w0)], axis=1),
            np.stack([2 * (x * z - y * w0), 2 * (y * z + x * w0), 1 - 2 * (x * x + y * y)], axis=1),
        ],
        axis=1,
    )
    values = []
    for start in range(0, n, CHUNK):
        values.append(evaluate_fields(rotations[start : start + CHUNK], sun, crystal, index, members)["w"])
    w = np.concatenate(values)
    mean, stderr = float(np.mean(w)), float(np.std(w) / np.sqrt(n))
    return {
        "haar_mc_samples": n,
        "haar_mc_mean_w": mean,
        "haar_mc_stderr": stderr,
        "fibonacci_mean_w": fibonacci_mean_w,
        "z_score": (fibonacci_mean_w - mean) / stderr,
        "passed": bool(abs(fibonacci_mean_w - mean) < 4.0 * stderr),
    }


# ------------------------------------------------------------------ store
def crystal_description(crystal: Polyhedron) -> dict[str, Any]:
    """JSON description of an untransformed :class:`.geometry.HexPrism` (the only crystal supported)."""
    if not isinstance(crystal, HexPrism):
        raise TypeError("the S^2 event store is implemented for the hexagonal prism only")
    if not np.array_equal(crystal.vertices, HexPrism(crystal.a, crystal.h).vertices):
        raise ValueError("the crystal must be an untransformed HexPrism (body frame, centred at the origin)")
    return {"type": "HexPrism", "a": crystal.a, "h": crystal.h}


def crystal_from_description(description: Mapping[str, Any]) -> HexPrism:
    if description.get("type") != "HexPrism":
        raise ValueError(f"unsupported crystal description {description!r}")
    return HexPrism(float(description["a"]), float(description["h"]))


@dataclasses.dataclass(frozen=True)
class S2StoreSpec:
    """Every parameter a store's arrays depend on; its canonical JSON is the cache key."""

    members: tuple[Faces, ...]
    crystal: Mapping[str, Any]
    refractive_index: float
    n: int
    sampling: str = FIBONACCI_SAMPLING
    deviation_window: tuple[float, float] | None = None
    dtype: str = "float64"

    def __post_init__(self) -> None:
        members = tuple(normalize_faces(m) for m in self.members)
        if not members or len(set(members)) != len(members):
            raise ValueError("members must be a non-empty tuple of distinct face sequences")
        if self.n < 1:
            raise ValueError("n must be positive")
        if self.dtype not in DTYPES:
            raise ValueError(f"dtype must be one of {DTYPES}")
        window = self.deviation_window
        object.__setattr__(self, "members", members)
        object.__setattr__(self, "crystal", dict(self.crystal))
        object.__setattr__(self, "refractive_index", float(self.refractive_index))
        object.__setattr__(self, "n", int(self.n))
        object.__setattr__(self, "deviation_window", None if window is None else (float(window[0]), float(window[1])))

    @property
    def path_id(self) -> str:
        """``"3-5"`` for one member, ``"3-5+3-1-2-5"`` for a ``Phi`` group."""
        return "+".join(path_id_of(m) for m in self.members)

    def build_parameters(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "members": [list(m) for m in self.members],
            "crystal": dict(self.crystal),
            "refractive_index": self.refractive_index,
            "N": self.n,
            "sampling": self.sampling,
            "deviation_window_rad": None if self.deviation_window is None else list(self.deviation_window),
            "dtype": self.dtype,
            "chunk": CHUNK,
            "rotation_per_point": ROTATION_PER_POINT,
        }

    @classmethod
    def from_build_parameters(cls, parameters: Mapping[str, Any]) -> S2StoreSpec:
        window = parameters["deviation_window_rad"]
        return cls(
            members=tuple(tuple(m) for m in parameters["members"]),
            crystal=parameters["crystal"],
            refractive_index=parameters["refractive_index"],
            n=parameters["N"],
            sampling=parameters["sampling"],
            deviation_window=None if window is None else tuple(window),
            dtype=parameters["dtype"],
        )

    def cache_key(self) -> str:
        """``<path_id>_N<n>_<sha256[:12]>`` of the canonical build-parameter JSON (git commit excluded)."""
        canonical = json.dumps(self.build_parameters(), sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(canonical.encode()).hexdigest()
        return f"{self.path_id}_N{self.n}_{digest[:12]}"


@dataclasses.dataclass(frozen=True)
class S2Events:
    """Events sorted by ``D`` (or a band of them): ``u``, ``phi`` ``(K, 3)``; ``D``, ``w``, ``iw`` ``(K,)``.

    ``u = R^-1 s_hat`` (toward the sun), ``phi = Phi_P(-u)`` (outgoing propagation), both in the body frame.

    ``iw`` (``1 / (4 pi q(u))``) is ``None`` for a uniform point set; a
    contribution is ``w * iw * rho`` otherwise.
    """

    u: np.ndarray
    phi: np.ndarray
    D: np.ndarray
    w: np.ndarray
    iw: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.D)

    def arrays(self) -> dict[str, np.ndarray]:
        """The arrays by name in the on-disk order (``D`` first, ``iw`` only when present)."""
        out = {"D": self.D, "u": self.u, "phi": self.phi, "w": self.w}
        if self.iw is not None:
            out["iw"] = self.iw
        return out


@dataclasses.dataclass(frozen=True)
class S2EventStore:
    """A built store: its :class:`S2StoreSpec`, the sorted :class:`S2Events` and build diagnostics."""

    spec: S2StoreSpec
    events: S2Events
    diagnostics: Mapping[str, Any]

    def band_slice(self, delta_lo: float, delta_hi: float) -> S2Events:
        """Events with ``delta_lo <= D < delta_hi`` (``searchsorted``; views, not copies)."""
        lo, hi = np.searchsorted(self.events.D, [delta_lo, delta_hi])
        hi = max(hi, lo)
        e = self.events
        return S2Events(e.u[lo:hi], e.phi[lo:hi], e.D[lo:hi], e.w[lo:hi], None if e.iw is None else e.iw[lo:hi])

    def save(self, base_dir: Path) -> Path:
        """Write ``<base_dir>/<cache_key>/<name>.npy`` per array then ``provenance.json``; never overwrites."""
        directory = Path(base_dir) / self.spec.cache_key()
        arrays = self.events.arrays()
        for path in [directory / f"{name}.npy" for name in arrays] + [directory / PROVENANCE_FILE]:
            if path.exists():
                raise FileExistsError(f"{path} exists; refusing to overwrite an event store")
        directory.mkdir(parents=True, exist_ok=True)
        for name, array in arrays.items():
            np.save(directory / f"{name}.npy", array)
        _write_provenance(directory, self.spec, list(arrays), self.diagnostics)
        return directory

    @classmethod
    def load(cls, directory: Path, *, mmap_mode: str | None = None) -> S2EventStore:
        """Read a saved store; refuses another ``SCHEMA_VERSION``.

        ``mmap_mode=None`` (default) reads every array into memory after
        checking its SHA-256 against the provenance (:meth:`verify`), so a
        modified array is refused.  ``mmap_mode="r"`` maps the arrays
        read-only (``numpy.memmap``; :meth:`band_slice` stays a view that
        touches only the pages of its band) and checks only each file's size
        against the recorded one: the content is deliberately *not* hashed,
        which would read the whole store and defeat the mapping.  A consumer
        that needs the content guarantee on that path calls :meth:`verify`
        once (task ``s2-store-schema-3``).
        """
        if mmap_mode not in (None, "r"):
            raise ValueError(f"mmap_mode must be None or 'r' (a store is read-only), not {mmap_mode!r}")
        directory = Path(directory)
        provenance = _read_provenance(directory)
        records = provenance["arrays"]
        if mmap_mode is None:
            cls.verify(directory)
        else:
            for name, record in records.items():
                size = (directory / record["file"]).stat().st_size
                if size != record["bytes"]:
                    raise ValueError(f"{directory / record['file']}: {size} bytes, the provenance records {record['bytes']}")
        arrays = {name: np.load(directory / record["file"], mmap_mode=mmap_mode) for name, record in records.items()}
        events = S2Events(arrays["u"], arrays["phi"], arrays["D"], arrays["w"], arrays.get("iw"))
        spec = S2StoreSpec.from_build_parameters(provenance["build"])
        return cls(spec, events, provenance["diagnostics"])

    @staticmethod
    def verify(directory: Path) -> None:
        """Full integrity check of a saved store: every array's SHA-256 against the provenance, else ``ValueError``."""
        directory = Path(directory)
        for name, record in _read_provenance(directory)["arrays"].items():
            path = directory / record["file"]
            digest = sha256_of(path)
            if digest != record["sha256"]:
                raise ValueError(f"{path}: SHA-256 of array {name!r} is {digest}, the provenance records {record['sha256']}")


def _read_provenance(directory: Path) -> dict[str, Any]:
    provenance = json.loads((directory / PROVENANCE_FILE).read_text())
    schema = provenance["build"].get("schema_version")
    if schema != SCHEMA_VERSION:
        raise ValueError(
            f"{directory}: schema_version {schema} is not {SCHEMA_VERSION} (schema 1 stored u = R^-1 s with the "
            "propagation direction s; schema 2 u = R^-1 s_hat with the sun direction in the key and one events.npz; "
            "schema 3 is independent of the sun, one .npy per array); rebuild the store, it is not converted"
        )
    return provenance


def _write_provenance(directory: Path, spec: S2StoreSpec, names: Sequence[str], diagnostics: Mapping[str, Any]) -> None:
    """``provenance.json`` of the arrays ``<name>.npy`` already written in ``directory`` (SHA-256 and size each)."""
    arrays = {}
    for name in names:
        path = directory / f"{name}.npy"
        arrays[name] = {"file": path.name, "sha256": sha256_of(path), "bytes": path.stat().st_size}
    provenance = {
        "build": spec.build_parameters(),
        "cache_key": spec.cache_key(),
        "git_commit": git_commit(None),
        "arrays": arrays,
        "diagnostics": dict(diagnostics),
        "created": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    (directory / PROVENANCE_FILE).write_text(json.dumps(provenance, indent=2) + "\n")


def events_from_schema1(arrays: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Arrays of a schema 1 store (``u = R^-1 s``, ``s`` the propagation direction) in schema 2: ``u`` negated.

    The one named conversion for reading legacy artifacts (the task 13/14
    flat ``events_N<n>.npz`` files); ``phi``, ``D``, ``w`` and ``iw`` are the
    same in both schemas.  :meth:`S2EventStore.load` and :func:`build_or_load`
    never apply it: a schema 1 cache directory is refused, not converted.
    """
    out = dict(arrays)
    out["u"] = -np.asarray(arrays["u"])
    return out


# Build pipeline (schema 3): _spec_of builds the cache key -> _build_into does the bucketed scan/merge onto
# disk (using _bucket_edges, _write_array_file) -> _write_provenance records it -> S2EventStore.save/load and
# .verify() read it back (_read_provenance, _array_names).  build_event_store and build_or_load both call
# _build_into so the two entry points share one build implementation (see its docstring for how).
def _bucket_edges(deviation_window: tuple[float, float] | None, bucket_count: int) -> np.ndarray:
    """``bucket_count + 1`` equal-width edges of the deviation domain (``[0, pi]`` or the window), data independent."""
    lo, hi = (0.0, np.pi) if deviation_window is None else deviation_window
    return np.linspace(lo, hi, bucket_count + 1)


def _write_array_file(path: Path, dtype: str, shape: tuple[int, ...]):
    """Open ``path`` for writing a C-order ``.npy`` of ``shape`` sequentially (the header ``np.save`` writes)."""
    handle = path.open("wb")
    header = {"descr": np.lib.format.dtype_to_descr(np.dtype(dtype)), "fortran_order": False, "shape": shape}
    np.lib.format.write_array_header_1_0(handle, header)
    return handle


def _build_into(
    directory: Path,
    spec: S2StoreSpec,
    crystal: HexPrism,
    sampler: PointSampler | None,
    bucket_count: int,
    run_checks: bool,
    log: Callable[[str], None] | None,
) -> dict[str, Any]:
    """Build the arrays of ``spec`` as ``<name>.npy`` in the existing ``directory``; returns the diagnostics.

    Deviation buckets.  Kept rows ``(D, u, phi, w[, iw])`` are appended, in
    chunk order, to one scratch file per bucket of the fixed equal-width
    edges :func:`_bucket_edges`; afterwards each bucket is read back alone,
    sorted with a stable argsort on ``D`` and appended to the final files,
    buckets in increasing ``D``.  The buckets are disjoint consecutive
    ``D`` intervals, so equal ``D`` values share a bucket and every bucket
    holds its rows in their global order: the concatenation is the stable
    argsort of all rows, the same permutation as sorting everything at
    once (bucket sort; pinned by ``tests/test_s2_store.py`` against
    ``bucket_count=1``).  Only one chunk of fields or one bucket is in
    memory, never all kept events.  ``bucket_count`` is an I/O knob and not
    a build parameter: it changes neither the arrays nor the cache key.
    """
    if bucket_count < 1:
        raise ValueError("bucket_count must be positive")
    sun, index, members, n = _REFERENCE_SUN_DIRECTION, spec.refractive_index, spec.members, spec.n
    checks: dict[str, Any] = {}
    if run_checks:
        checks["psi_invariance"] = self_check_psi_invariance(sun, crystal, index, members)
        if not checks["psi_invariance"]["passed"]:
            raise RuntimeError(f"psi-invariance self-check failed: {checks['psi_invariance']}")
        checks["gate_coverage"] = self_check_gate_coverage(sun, crystal, index, members)
        if log is not None:
            log("self-checks: " + json.dumps(checks))

    start = time.perf_counter()
    edges = _bucket_edges(spec.deviation_window, bucket_count)
    counts = np.zeros(bucket_count, dtype=np.int64)
    columns = {"D": slice(0, 1), "u": slice(1, 4), "phi": slice(4, 7), "w": slice(7, 8)}
    w_sum = 0.0
    valid_count = 0
    with tempfile.TemporaryDirectory(prefix=".buckets-", dir=directory) as scratch:
        bucket_path = [Path(scratch) / f"{b:05d}.f64" for b in range(bucket_count)]
        for first in range(0, n, CHUNK):
            stop = min(first + CHUNK, n)
            if sampler is None:
                u, inverse_weight = store_lattice(n, first, stop), None
            else:
                u, inverse_weight = sampler(first, stop)
            if first == 0 and inverse_weight is not None:
                columns["iw"] = slice(8, 9)
            if (inverse_weight is not None) != ("iw" in columns):
                raise ValueError("a sampler must return inverse weights for every chunk or for none")
            fields = evaluate_fields(align_rotations(u, sun), sun, crystal, index, members)
            w = fields["w"]
            valid_count += int(np.count_nonzero(fields["valid"]))
            keep = w > 0.0
            if spec.deviation_window is not None:
                keep &= (fields["D"] >= spec.deviation_window[0]) & (fields["D"] <= spec.deviation_window[1])
            w_sum += float(np.sum(w if inverse_weight is None else w * inverse_weight))
            parts = [fields["D"][keep, None], u[keep], fields["phi"][keep], w[keep, None]]
            if inverse_weight is not None:
                parts.append(inverse_weight[keep, None])
            rows = np.concatenate(parts, axis=1)
            del fields, parts
            bucket = np.searchsorted(edges[1:-1], rows[:, 0], side="right")
            order = np.argsort(bucket, kind="stable")  # chunk order kept inside each bucket
            chunk_counts = np.bincount(bucket, minlength=bucket_count)
            offset = 0
            sorted_rows = rows[order]
            for b in range(bucket_count):
                size = int(chunk_counts[b])
                if size:
                    with bucket_path[b].open("ab") as handle:
                        handle.write(sorted_rows[offset : offset + size].tobytes())
                    offset += size
            counts += chunk_counts
            del rows, sorted_rows, order, bucket

        kept_count = int(counts.sum())
        width = 9 if "iw" in columns else 8
        handles = {
            name: _write_array_file(directory / f"{name}.npy", spec.dtype, (kept_count, 3) if name in ("u", "phi") else (kept_count,))
            for name in columns
        }
        d_range = None
        try:
            for b in np.flatnonzero(counts):
                rows = np.fromfile(bucket_path[b], dtype=np.float64).reshape(-1, width)
                bucket_path[b].unlink()
                rows = rows[np.argsort(rows[:, 0], kind="stable")]
                d_range = [float(rows[0, 0]) if d_range is None else d_range[0], float(rows[-1, 0])]
                for name, column in columns.items():
                    values = rows[:, column] if name in ("u", "phi") else rows[:, column.start]
                    handles[name].write(np.ascontiguousarray(values, dtype=spec.dtype).tobytes())
                del rows
        finally:
            for handle in handles.values():
                handle.close()
    wall = time.perf_counter() - start

    if run_checks:
        checks["haar_mean"] = self_check_haar_mean(sun, crystal, index, w_sum / n, members)
        if log is not None:
            log("haar mean check: " + json.dumps(checks["haar_mean"]))
    return {
        "kept_events": kept_count,
        "valid_domain_points": valid_count,
        "kept_fraction": kept_count / n,
        "fibonacci_mean_w": w_sum / n,
        "D_range_deg": None if d_range is None else [float(np.degrees(d_range[0])), float(np.degrees(d_range[1]))],
        "wall_clock_s": wall,
        "max_rss_mb_process": max_rss_mb(),
        "bucket_count": bucket_count,
        "largest_bucket_events": int(counts.max()),
        "self_checks": checks,
    }


def _spec_of(
    crystal: HexPrism,
    refractive_index: float,
    members: Sequence[Sequence[int]],
    n: int,
    sampler: PointSampler | None,
    sampling: str,
    deviation_window: tuple[float, float] | None,
    dtype: str,
) -> S2StoreSpec:
    if (sampler is None) != (sampling == FIBONACCI_SAMPLING):
        raise ValueError("a custom sampler needs its own sampling description, and the Fibonacci lattice the default one")
    spec = S2StoreSpec(
        members=tuple(tuple(m) for m in members),
        crystal=crystal_description(crystal),
        refractive_index=refractive_index,
        n=n,
        sampling=sampling,
        deviation_window=deviation_window,
        dtype=dtype,
    )
    keys = {phi_key(crystal, m) for m in spec.members}
    if len(keys) != 1:
        raise ValueError(f"members {spec.path_id} do not share one phi_key: {sorted(keys)}")
    return spec


def build_event_store(
    crystal: HexPrism,
    refractive_index: float,
    members: Sequence[Sequence[int]],
    n: int,
    *,
    sampler: PointSampler | None = None,
    sampling: str = FIBONACCI_SAMPLING,
    deviation_window: tuple[float, float] | None = None,
    dtype: str = "float64",
    bucket_count: int = DEFAULT_BUCKET_COUNT,
    run_checks: bool = True,
    log: Callable[[str], None] | None = None,
) -> S2EventStore:
    """Event store of ``members`` in memory: ``w > 0`` events sorted by ``D`` (``[lo, hi]`` rad only, if given).

    The store is independent of the sun: it is built with the fixed
    reference ``s_hat`` (``_REFERENCE_SUN_DIRECTION``) and serves every sun
    direction (a renderer rebuilds the poses for its own ``s_hat``).
    ``sampler`` replaces :func:`store_lattice`, returns points ``u`` (toward
    the sun, body frame) and must come with its own
    ``sampling`` description (it enters the cache key); its inverse weights
    are stored as ``iw``.  Several ``members`` must share one
    :func:`.path_class.phi_key`.  ``run_checks`` runs the section 4.1(a)
    self-checks (psi invariance and gate coverage before the build, the
    Haar mean after it) and raises if psi invariance fails.  The arrays are
    built through a temporary directory (:func:`_build_into`) and read back;
    :func:`build_or_load` builds into the cache directory instead and never
    needs all events in memory.
    """
    spec = _spec_of(crystal, refractive_index, members, n, sampler, sampling, deviation_window, dtype)
    with tempfile.TemporaryDirectory(prefix="s2-store-build-") as scratch:
        directory = Path(scratch)
        diagnostics = _build_into(directory, spec, crystal, sampler, bucket_count, run_checks, log)
        arrays = {name: np.load(directory / f"{name}.npy") for name in _array_names(directory)}
    events = S2Events(arrays["u"], arrays["phi"], arrays["D"], arrays["w"], arrays.get("iw"))
    return S2EventStore(spec, events, diagnostics)


def build_or_load(
    crystal: HexPrism,
    refractive_index: float,
    members: Sequence[Sequence[int]],
    n: int,
    *,
    base_dir: Path = DEFAULT_CACHE_DIR,
    sampler: PointSampler | None = None,
    sampling: str = FIBONACCI_SAMPLING,
    deviation_window: tuple[float, float] | None = None,
    dtype: str = "float64",
    mmap_mode: str | None = None,
    run_checks: bool = True,
    log: Callable[[str], None] | None = None,
) -> S2EventStore:
    """The cached store of these parameters, built and saved on first use; returned by :meth:`S2EventStore.load`.

    An existing cache directory is loaded (schema and, unless
    ``mmap_mode="r"``, SHA-256 checked) and its recorded build parameters
    must equal the request's, else ``ValueError``; a directory without its
    provenance is refused too.  Nothing is rebuilt or overwritten silently.
    A missing store is built by :func:`_build_into` straight into a hidden
    staging directory next to it (all events are never in memory at once),
    which is renamed into place once its provenance is written, so an
    interrupted build leaves no directory *under the cache key*.  The
    staging directory itself is cleaned up on the Python exception path;
    a hard kill (OOM, ``kill -9``, power loss) instead leaves it on disk
    until a later call's :func:`_sweep_stale_staging` reclaims it once it
    has been idle past :data:`_STAGING_STALE_SECONDS` (age, not mere
    existence, is what tells a dead staging directory apart from one a
    concurrent builder still owns).  A concurrent builder that renamed
    first wins: its store is loaded and ours dropped.
    """
    spec = _spec_of(crystal, refractive_index, members, n, sampler, sampling, deviation_window, dtype)
    directory = Path(base_dir) / spec.cache_key()
    _sweep_stale_staging(Path(base_dir))
    if not directory.exists():
        staging = Path(tempfile.mkdtemp(prefix=f".{spec.cache_key()}.building-", dir=_mkdir(Path(base_dir))))
        try:
            diagnostics = _build_into(staging, spec, crystal, sampler, DEFAULT_BUCKET_COUNT, run_checks, log)
            _write_provenance(staging, spec, list(_array_names(staging)), diagnostics)
            try:
                staging.rename(directory)
            except OSError:
                if not directory.exists():
                    raise
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    if not (directory / PROVENANCE_FILE).exists():
        raise ValueError(f"{directory} has no {PROVENANCE_FILE} (interrupted save?); remove it to rebuild")
    store = S2EventStore.load(directory, mmap_mode=mmap_mode)
    recorded, requested = store.spec.build_parameters(), spec.build_parameters()
    if recorded != requested:
        differing = sorted(k for k in requested.keys() | recorded.keys() if recorded.get(k) != requested.get(k))
        raise ValueError(f"{directory}: recorded build parameters differ from the request in {differing}")
    return store


def _mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _sweep_stale_staging(base_dir: Path) -> None:
    """Remove :func:`build_or_load` staging directories abandoned by a killed builder.

    A normal build cleans up its own staging directory (the ``finally:
    shutil.rmtree`` in :func:`build_or_load`); that only runs on the Python
    exception path.  A hard kill (OOM, ``kill -9``, power loss) during a
    build leaves the staging directory behind: it is named
    ``.<cache_key>.building-<random>``, never matches a cache key, and no
    load/verify/build path ever looks for it, so it is a permanent leak
    without this sweep.  Age is what distinguishes an abandoned staging
    directory from one a concurrent, still-running builder legitimately
    owns (builds are documented to take minutes to hours): only directories
    idle past :data:`_STAGING_STALE_SECONDS` are removed.
    """
    for candidate in base_dir.glob(".*building-*"):
        if not candidate.is_dir():
            continue
        try:
            age = time.time() - candidate.stat().st_mtime
        except OSError:
            continue
        if age > _STAGING_STALE_SECONDS:
            shutil.rmtree(candidate, ignore_errors=True)


def _array_names(directory: Path) -> tuple[str, ...]:
    """The arrays of a built directory in the on-disk order of :meth:`S2Events.arrays`."""
    return tuple(name for name in ("D", "u", "phi", "w", "iw") if (directory / f"{name}.npy").exists())


__all__ = [
    "CHUNK",
    "DEFAULT_BUCKET_COUNT",
    "DEFAULT_CACHE_DIR",
    "FIBONACCI_SAMPLING",
    "ROTATION_PER_POINT",
    "SCHEMA_VERSION",
    "PointSampler",
    "RandomSphereSampler",
    "S2EventStore",
    "S2Events",
    "S2StoreSpec",
    "align_rotations",
    "build_event_store",
    "build_or_load",
    "crystal_description",
    "crystal_from_description",
    "evaluate_fields",
    "event_rotations",
    "events_from_schema1",
    "fibonacci_sphere",
    "frame",
    "max_rss_mb",
    "self_check_gate_coverage",
    "self_check_haar_mean",
    "self_check_psi_invariance",
    "store_lattice",
    "transported_rotations",
    "twist_about",
]
