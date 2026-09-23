"""The S^2 event store: per-path weight fields sampled on ``u = R^-1 s`` and sorted by deviation.

Roadmap section 4.1(a): validity, entry measure ``A``, Fresnel transmission
``T``, the body-frame outgoing direction ``Phi`` and the deviation
``D = angle(u, Phi)`` of a fixed ray path depend on the pose ``R`` only
through ``u = R^-1 s``.  Section 4.2 discretises the level-set integral with
a store of events on ``S^2``: ``N`` points (Fibonacci lattice by default,
``4 pi / N`` each), one rotation per point, the production batch evaluators
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
rebuild of those stores bit for bit.

Members and ``Phi`` groups.  A store is built for a tuple of ``members``
(face sequences).  One member is the ordinary per-path store.  Several
members must share one :func:`.path_class.phi_key` (same fold matrix, entry
normal and unfolded exit normal, hence the same ``Phi(u)`` and ``D(u)``);
their weights are then summed on the same ``u`` (``w_Phi = sum_m w_m``) and
an event is kept where any member has ``w_m > 0``.

Symmetry transport.  For any crystal symmetry ``g`` of ``D6h``, proper or
improper, mapping the representative's faces onto a member's
(:func:`.path_class.path_class_symmetry`), section 4.1(d) gives
``Phi_member(u) = g Phi_rep(g^-1 u)`` with ``D``, ``w`` and the valid domain
unchanged: the member's events are the representative's with ``u' = g u``,
``Phi' = g Phi``.  The pose of a transported event is rebuilt from
``(g u, g Phi, D)`` and the pixel's azimuth by :func:`event_rotations`,
whose two orthonormal frames always give a rotation, whatever ``det g``
(a mirror needs no store of its own on ``S^2``; that restriction belongs to
Phase I, where ``R g^-1`` would have to stay in SO(3)).  The production
interface is :func:`transported_rotations`, the closed form of that rebuild
on the representative's poses: ``L_g R g^T`` with ``L_g = I`` for a proper
``g`` and the reflection in the plane of ``s`` and the pixel centre for an
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

Disk cache: ``<base_dir>/<cache_key>/events.npz`` + ``provenance.json``.
The key hashes the build parameters (:meth:`S2StoreSpec.build_parameters`)
and ``SCHEMA_VERSION``; ``SCHEMA_VERSION`` is bumped by hand when the build
algorithm changes.  The git commit is recorded in the provenance for
forensics but deliberately *not* part of the key or of the comparison: a
large store (``N = 1e8`` takes minutes to hours) must not be invalidated by
unrelated commits.  Unlike :func:`.prescan.build_or_load_prescan_table`
(which rebuilds on mismatch), :func:`build_or_load` refuses a cache whose
recorded parameters differ from the request, and :meth:`S2EventStore.load`
refuses an ``events.npz`` whose SHA-256 differs from the recorded one --
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
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from . import geometry, optics
from .geometry import HexPrism, Polyhedron
from .optics import normalize_faces, path_id_of
from .path_class import phi_key
from .provenance import git_commit, sha256_of

SCHEMA_VERSION = 1
CHUNK = 250_000  # rotations per batch call: ~0.5 GB transient in the eager jax.vmap (task 13/14 value)
DEFAULT_CACHE_DIR = Path("artifacts/s2-store")
EVENTS_FILE = "events.npz"
PROVENANCE_FILE = "provenance.json"
FIBONACCI_SAMPLING = "Fibonacci lattice on S^2 (equal-area spiral, z_i = 1 - (2i+1)/N, golden-angle azimuth)"
ROTATION_PER_POINT = "R = [s, p_s, s x p_s][u, p_u, u x p_u]^T (any R with R u = s; section 4.1(a))"
RANDOM_SEED = 20260923
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


@dataclasses.dataclass(frozen=True)
class RandomSphereSampler:
    """i.i.d. uniform points on ``S^2`` (``q = 1 / 4 pi``, so no inverse weights); chunk-seeded, reproducible."""

    seed: int = RANDOM_SEED
    description = "i.i.d. uniform on S^2 (normalised Gaussian triples, numpy default_rng([seed, first]) per chunk)"

    def __call__(self, first: int, stop: int) -> tuple[np.ndarray, None]:
        points = np.random.default_rng([self.seed, first]).normal(size=(stop - first, 3))
        return points / np.linalg.norm(points, axis=1, keepdims=True), None


def _any_perpendicular(v: np.ndarray) -> np.ndarray:
    """A unit vector perpendicular to each row of ``v`` (cross with the least aligned axis; no degenerate case)."""
    axis = np.zeros_like(v)
    axis[np.arange(len(v)), np.argmin(np.abs(v), axis=1)] = 1.0
    p = np.cross(v, axis)
    return p / np.linalg.norm(p, axis=1, keepdims=True)


def frame(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Orthonormal right-handed frames with columns ``(first, second, first x second)``, ``(n, 3, 3)``."""
    return np.stack([first, second, np.cross(first, second)], axis=2)


def align_rotations(u: np.ndarray, s: np.ndarray) -> np.ndarray:
    """One rotation per row with ``R u = s`` (the free twist about ``s`` is arbitrary, section 4.1(a))."""
    s_rows = np.broadcast_to(s, u.shape)
    return np.einsum("nij,nkj->nik", frame(s_rows, _any_perpendicular(s_rows)), frame(u, _any_perpendicular(u)))


def twist_about(axis: np.ndarray, angle: float) -> np.ndarray:
    a = axis / np.linalg.norm(axis)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)


def event_rotations(
    u: np.ndarray, phi: np.ndarray, deviation: np.ndarray, s: np.ndarray, centre: np.ndarray
) -> np.ndarray:
    """``R_i`` with ``R_i u_i = s`` and ``R_i Phi_i`` at deviation ``D_i``, azimuth of ``centre``.

    The pose of each event for a pixel whose centre direction is ``centre``
    (Gislen eq. 19 at ``omega = D_i``, built from two orthonormal frames).
    For the events transported by a crystal symmetry ``g`` see
    :func:`transported_rotations` (module docstring).
    """
    e = centre - (centre @ s) * s
    e /= np.linalg.norm(e)
    f2 = phi - np.cos(deviation)[:, None] * u
    f2 /= np.linalg.norm(f2, axis=1, keepdims=True)
    world = np.stack([s, e, np.cross(s, e)], axis=1)
    return np.einsum("ij,nkj->nik", world, frame(u, f2))


def transported_rotations(rotations: np.ndarray, g: np.ndarray, s: np.ndarray, centre: np.ndarray) -> np.ndarray:
    """:func:`event_rotations` of the events transported by ``g`` (``u' = g u``, ``Phi' = g Phi``), from their poses.

    ``rotations`` are :func:`event_rotations` of the untransported events
    for the same ``s`` and ``centre``; ``g`` is any orthogonal crystal
    symmetry.  :func:`event_rotations` is ``R = W F^T`` with the world frame
    ``W`` and the event frame ``F`` of ``(u, f)``.  The transported frame is
    ``g F J`` with ``J = diag(1, 1, det g)`` (a cross product changes sign
    under a mirror), so ``R' = W J F^T g^T = L_g R g^T`` with
    ``L_g = W J W^T = I - (1 - det g) m m^T``, ``m = W e_3`` the normal of
    the plane of ``s`` and ``centre``.  For a proper ``g`` this is ``R g^T``
    (``L_g`` skipped: the same floating-point operation as the proper-only
    transport of task ``band-sum-renderer``); for an improper one ``L_g`` is
    the reflection in the ``(s, centre)`` plane.  The result is a rotation
    either way.
    """
    g = np.asarray(g, dtype=np.float64)
    moved = rotations @ g.T
    if np.linalg.det(g) > 0.0:
        return moved
    e = centre - (centre @ s) * s
    m = np.cross(s, e / np.linalg.norm(e))
    return (np.eye(3) - 2.0 * np.outer(m, m)) @ moved


def evaluate_fields(
    rotations: np.ndarray, s: np.ndarray, crystal: Polyhedron, index: float, members: Sequence[Faces]
) -> dict[str, np.ndarray]:
    """Production batch evaluators of the ``members`` at ``rotations``.

    Returns validity (any member), ``A`` and ``T`` per member (``(m, n)``),
    ``w = sum_m A_m T_m``, body-frame ``Phi``, ``D`` and ``u``.  ``Phi`` is
    the first member's direction: members of one ``Phi`` group share the
    closed form of the outgoing direction (entry and exit refraction on the
    same normals, the same fold matrix), so it is the same wherever any of
    them is valid.
    """
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
    u = np.einsum("nji,j->ni", rotations, s)
    phi = np.einsum("nji,nj->ni", rotations, direction)
    deviation = np.arccos(np.clip(np.sum(phi * u, axis=1), -1.0, 1.0))
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
    s: np.ndarray, crystal: Polyhedron, index: float, members: Sequence[Faces], sample: int = 4000
) -> dict[str, Any]:
    """Section 4.1(a): validity, ``A``, ``T``, ``Phi`` and ``D`` do not change under a twist about ``s``."""
    u = fibonacci_sphere(200_000)[:: 200_000 // sample]
    base = align_rotations(u, s)
    reference = evaluate_fields(base, s, crystal, index, members)
    worst = {"valid_mismatch": 0, "A": 0.0, "T": 0.0, "phi": 0.0, "D": 0.0}
    for angle in (0.7, 2.1, -2.9):
        fields = evaluate_fields(np.einsum("ij,njk->nik", twist_about(s, angle), base), s, crystal, index, members)
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
    s: np.ndarray, crystal: Polyhedron, index: float, members: Sequence[Faces], sample: int = 3000
) -> dict[str, int]:
    """Every ``entry_measure`` failure reason occurs on the sphere (the scalar form reports the reason)."""
    u = fibonacci_sphere(sample)
    counts: dict[str, int] = {}
    for faces in members:
        for rotation in align_rotations(u, s):
            status = geometry.entry_measure(rotation, faces, s, crystal, n_ice=index).status
            counts[status] = counts.get(status, 0) + 1
    return counts


def self_check_haar_mean(
    s: np.ndarray,
    crystal: Polyhedron,
    index: float,
    fibonacci_mean_w: float,
    members: Sequence[Faces],
    n: int = 1_000_000,
) -> dict[str, Any]:
    """``E_Haar[A T]`` from independent uniform quaternions against the Fibonacci ``sum w / N``.

    Checks the fibration claim ``u = R^-1 s`` is uniform on ``S^2`` under Haar together with
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
        values.append(evaluate_fields(rotations[start : start + CHUNK], s, crystal, index, members)["w"])
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
    incident_direction: tuple[float, float, float]
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
        object.__setattr__(self, "incident_direction", tuple(float(v) for v in self.incident_direction))
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
            "incident_direction": list(self.incident_direction),
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
            incident_direction=tuple(parameters["incident_direction"]),
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
        """Write ``<base_dir>/<cache_key>/events.npz`` then ``provenance.json``; never overwrites."""
        directory = Path(base_dir) / self.spec.cache_key()
        events_path, provenance_path = directory / EVENTS_FILE, directory / PROVENANCE_FILE
        for path in (events_path, provenance_path):
            if path.exists():
                raise FileExistsError(f"{path} exists; refusing to overwrite an event store")
        directory.mkdir(parents=True, exist_ok=True)
        np.savez(events_path, **self.events.arrays())
        provenance = {
            "build": self.spec.build_parameters(),
            "cache_key": self.spec.cache_key(),
            "git_commit": git_commit(None),
            "arrays": {"file": EVENTS_FILE, "sha256": sha256_of(events_path)},
            "diagnostics": dict(self.diagnostics),
            "created": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
        return directory

    @classmethod
    def load(cls, directory: Path) -> S2EventStore:
        """Read a saved store; refuses an ``events.npz`` whose SHA-256 differs from the provenance."""
        directory = Path(directory)
        provenance = json.loads((directory / PROVENANCE_FILE).read_text())
        events_path = directory / EVENTS_FILE
        digest = sha256_of(events_path)
        if digest != provenance["arrays"]["sha256"]:
            raise ValueError(f"{events_path}: SHA-256 {digest} differs from the recorded {provenance['arrays']['sha256']}")
        with np.load(events_path) as data:
            arrays = {key: data[key] for key in data.files}
        events = S2Events(arrays["u"], arrays["phi"], arrays["D"], arrays["w"], arrays.get("iw"))
        spec = S2StoreSpec.from_build_parameters(provenance["build"])
        return cls(spec, events, provenance["diagnostics"])


def build_event_store(
    crystal: HexPrism,
    refractive_index: float,
    members: Sequence[Sequence[int]],
    n: int,
    *,
    incident_direction: np.ndarray,
    sampler: PointSampler | None = None,
    sampling: str = FIBONACCI_SAMPLING,
    deviation_window: tuple[float, float] | None = None,
    dtype: str = "float64",
    run_checks: bool = True,
    log: Callable[[str], None] | None = None,
) -> S2EventStore:
    """Event store of ``members``: ``w > 0`` events sorted by ``D`` (``[lo, hi]`` rad only, if given).

    ``sampler`` replaces the Fibonacci lattice and must come with its own
    ``sampling`` description (it enters the cache key); its inverse weights
    are stored as ``iw``.  Several ``members`` must share one
    :func:`.path_class.phi_key`.  ``run_checks`` runs the section 4.1(a)
    self-checks (psi invariance and gate coverage before the build, the
    Haar mean after it) and raises if psi invariance fails.
    """
    if (sampler is None) != (sampling == FIBONACCI_SAMPLING):
        raise ValueError("a custom sampler needs its own sampling description, and the Fibonacci lattice the default one")
    s = np.asarray(incident_direction, dtype=np.float64)
    index = float(refractive_index)
    spec = S2StoreSpec(
        members=tuple(tuple(m) for m in members),
        crystal=crystal_description(crystal),
        refractive_index=index,
        incident_direction=tuple(s),
        n=n,
        sampling=sampling,
        deviation_window=deviation_window,
        dtype=dtype,
    )
    members = spec.members
    keys = {phi_key(crystal, m) for m in members}
    if len(keys) != 1:
        raise ValueError(f"members {spec.path_id} do not share one phi_key: {sorted(keys)}")
    checks: dict[str, Any] = {}
    if run_checks:
        checks["psi_invariance"] = self_check_psi_invariance(s, crystal, index, members)
        if not checks["psi_invariance"]["passed"]:
            raise RuntimeError(f"psi-invariance self-check failed: {checks['psi_invariance']}")
        checks["gate_coverage"] = self_check_gate_coverage(s, crystal, index, members)
        if log is not None:
            log("self-checks: " + json.dumps(checks))

    start = time.perf_counter()
    kept: dict[str, list[np.ndarray]] = {"u": [], "phi": [], "D": [], "w": []}
    w_sum = 0.0
    valid_count = 0
    for first in range(0, n, CHUNK):
        stop = min(first + CHUNK, n)
        if sampler is None:
            u, inverse_weight = fibonacci_sphere(n, first, stop), None
        else:
            u, inverse_weight = sampler(first, stop)
        fields = evaluate_fields(align_rotations(u, s), s, crystal, index, members)
        w = fields["w"]
        valid_count += int(np.count_nonzero(fields["valid"]))
        keep = w > 0.0
        if deviation_window is not None:
            keep &= (fields["D"] >= spec.deviation_window[0]) & (fields["D"] <= spec.deviation_window[1])
        w_sum += float(np.sum(w if inverse_weight is None else w * inverse_weight))
        kept["u"].append(u[keep])
        kept["phi"].append(fields["phi"][keep])
        kept["D"].append(fields["D"][keep])
        kept["w"].append(w[keep])
        if inverse_weight is not None:
            kept.setdefault("iw", []).append(inverse_weight[keep])
    deviation = np.concatenate(kept.pop("D"))
    order = np.argsort(deviation, kind="stable")
    arrays = {"D": deviation[order]}
    del deviation
    for key in list(kept):  # one key at a time keeps the peak at about one extra copy of one array
        arrays[key] = np.concatenate(kept.pop(key))[order]
    if dtype != "float64":
        arrays = {key: np.ascontiguousarray(value, dtype=dtype) for key, value in arrays.items()}
    wall = time.perf_counter() - start
    events = S2Events(arrays["u"], arrays["phi"], arrays["D"], arrays["w"], arrays.get("iw"))

    if run_checks:
        checks["haar_mean"] = self_check_haar_mean(s, crystal, index, w_sum / n, members)
        if log is not None:
            log("haar mean check: " + json.dumps(checks["haar_mean"]))
    kept_count = len(events)
    diagnostics = {
        "kept_events": kept_count,
        "valid_domain_points": valid_count,
        "kept_fraction": kept_count / n,
        "fibonacci_mean_w": w_sum / n,
        "D_range_deg": [float(np.degrees(events.D[0])), float(np.degrees(events.D[-1]))] if kept_count else None,
        "wall_clock_s": wall,
        "max_rss_mb_process": max_rss_mb(),
        "self_checks": checks,
    }
    return S2EventStore(spec, events, diagnostics)


def build_or_load(
    crystal: HexPrism,
    refractive_index: float,
    members: Sequence[Sequence[int]],
    n: int,
    *,
    base_dir: Path = DEFAULT_CACHE_DIR,
    incident_direction: np.ndarray,
    sampler: PointSampler | None = None,
    sampling: str = FIBONACCI_SAMPLING,
    deviation_window: tuple[float, float] | None = None,
    dtype: str = "float64",
    run_checks: bool = True,
    log: Callable[[str], None] | None = None,
) -> S2EventStore:
    """The cached store of these parameters, built and saved on first use.

    An existing cache directory is loaded (SHA-256 checked) and its recorded
    build parameters must equal the request's, else ``ValueError``; a
    directory without its provenance (an interrupted save) is refused too.
    Nothing is rebuilt or overwritten silently.
    """
    spec = S2StoreSpec(
        members=tuple(tuple(m) for m in members),
        crystal=crystal_description(crystal),
        refractive_index=refractive_index,
        incident_direction=tuple(np.asarray(incident_direction, dtype=np.float64)),
        n=n,
        sampling=sampling,
        deviation_window=deviation_window,
        dtype=dtype,
    )
    directory = Path(base_dir) / spec.cache_key()
    if directory.exists():
        if not (directory / PROVENANCE_FILE).exists():
            raise ValueError(f"{directory} has no {PROVENANCE_FILE} (interrupted save?); remove it to rebuild")
        store = S2EventStore.load(directory)
        recorded, requested = store.spec.build_parameters(), spec.build_parameters()
        if recorded != requested:
            differing = sorted(k for k in requested.keys() | recorded.keys() if recorded.get(k) != requested.get(k))
            raise ValueError(f"{directory}: recorded build parameters differ from the request in {differing}")
        return store
    store = build_event_store(
        crystal,
        refractive_index,
        members,
        n,
        incident_direction=incident_direction,
        sampler=sampler,
        sampling=sampling,
        deviation_window=deviation_window,
        dtype=dtype,
        run_checks=run_checks,
        log=log,
    )
    store.save(base_dir)
    return store


__all__ = [
    "CHUNK",
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
    "fibonacci_sphere",
    "frame",
    "max_rss_mb",
    "self_check_gate_coverage",
    "self_check_haar_mean",
    "self_check_psi_invariance",
    "transported_rotations",
    "twist_about",
]
