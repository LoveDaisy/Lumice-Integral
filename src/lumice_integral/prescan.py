"""Scene-level prescan table: which poses send the incident ray where.

For a fixed ray path, incident direction ``s`` and refractive index ``n`` the
question "does Haar sample ``R`` pass the path's smooth domain, and where does
it send ``s``?" does not depend on the pixel.  :func:`discover_components` used
to answer it afresh for every cold pixel (400k samples, about 2.6 s each,
roughly 7000 times over the 58-column strip); :class:`PrescanTable` answers
it once per scene and indexes the domain-valid samples by outgoing direction
so a pixel only asks ``candidates(d, angle_tolerance_deg)``.

What the table holds (and does not):

- Every *domain-valid* sample: its rotation, outgoing direction, the margins
  of :func:`.optics.path_domain_batch` (:func:`.optics.domain_margin_names`
  of the path: four for ``3-5``, two more per internal reflection) and its
  index in the original sampling stream.  Nothing is pre-filtered by any pixel direction;
  the angular tolerance is a query parameter, so one table serves the image.
- It does not depend on the crystal: :func:`.optics.path_direction` refracts
  and reflects on the rotated face normals of an infinite prism, and the
  finite-crystal :func:`.geometry.entry_measure` gate stays in discovery,
  applied per candidate after the query.  Swapping crystal sizes therefore
  needs no rebuild.
- ``path_id`` names the ray path (``"3-5"``, ``"3-1-2-5"``, ...; the format
  of :func:`.optics.path_id_of`) and is the table's single source of the
  face sequence (:attr:`PrescanTable.faces`); a multi-path scene keeps one
  table per member, all built from the same ``(rng_seed, sample_count)``
  stream (:mod:`.path_class`).

Sampling is one ``numpy`` generator per ``rng_seed`` consumed in
``batch_size`` chunks, so (a) ``batch_size`` only bounds memory and never
changes the sampled stream, and (b) the stream of ``sample_count = N`` is a
prefix of the stream of ``2N``; :meth:`PrescanTable.prefix` exposes the
latter so a density survey can compare ``N`` against ``2N`` from one build.
Both properties are pinned by ``tests/test_prescan.py``.

The direction index is a ``scipy.spatial.cKDTree`` on the unit vectors; a
query is a ball of chord radius ``2 sin(tol / 2)`` (the exact image of the
great-circle cap) followed by the exact ``direction . d >= cos(tol)`` test on
the returned points, so the result is *identical* to the brute-force
comparison discovery used to make, with the tree only as a prefilter.

``save``/``load`` write the arrays as ``.npz`` next to a JSON provenance
sidecar (build parameters, git commit, SHA-256 of the ``.npz``);
:func:`build_or_load_prescan_table` rebuilds and overwrites a cache whose
provenance does not match the requested build.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
from scipy.spatial import cKDTree

from .optics import PATH_3_5_FACES, domain_margin_names, faces_of_path_id, path_domain_batch, path_id_of
from .provenance import git_commit, sha256_of

DEFAULT_PATH_ID = path_id_of(PATH_3_5_FACES)
# Pinned by the density survey of docs/ch06-reference-fixture.md (section 7,
# stage 4): on 32 strip pixels the discovered components and arclengths are
# unchanged from 2M to 16M samples (the one exception is a dedup-tolerance
# effect on a 0.165 rad loop, not a missed component), so 4M is the first
# rung whose halving changes nothing; scripts/prescan_density_survey.py.
DEFAULT_SAMPLE_COUNT = 4_000_000
DEFAULT_RNG_SEED = 20260916
DEFAULT_BATCH_SIZE = 200_000
PROVENANCE_SUFFIX = ".provenance.json"
LogCallback = Callable[[str], None]


def haar_rotations(count: int, rng: np.random.Generator) -> np.ndarray:
    """Haar-uniform rotation matrices from normalised Gaussian quaternions."""
    quaternion = rng.standard_normal((count, 4))
    quaternion /= np.linalg.norm(quaternion, axis=1, keepdims=True)
    w, x, y, z = quaternion.T
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
            np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
            np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
        ],
        axis=1,
    )


def chord_radius(angle_tolerance_deg: float) -> float:
    """Chord length on the unit sphere subtending ``angle_tolerance_deg``."""
    return float(2.0 * np.sin(np.radians(angle_tolerance_deg) / 2.0))


@dataclass(frozen=True)
class PrescanTable:
    """Domain-valid Haar samples of one ``(path_id, s, n)`` indexed by outgoing direction.

    Arrays are aligned row by row: ``rotations[i]`` sends ``incident_direction``
    to ``directions[i]`` with margins ``margins[name][i]`` and was sample
    number ``sample_indices[i]`` of the ``sample_count``-long stream seeded by
    ``rng_seed``.  The kd-tree is derived in ``__post_init__`` and excluded
    from equality and pickling; a worker that receives the table over
    ``multiprocessing`` rebuilds the tree on unpickle.
    """

    path_id: str
    incident_direction: np.ndarray
    refractive_index: float
    rotations: np.ndarray
    directions: np.ndarray
    margins: Mapping[str, np.ndarray]
    sample_indices: np.ndarray
    sample_count: int
    rng_seed: int
    _tree: cKDTree = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        incident = np.asarray(self.incident_direction, dtype=np.float64)
        object.__setattr__(self, "incident_direction", incident)
        object.__setattr__(self, "refractive_index", float(self.refractive_index))
        if self.rotations.shape != (self.valid_count, 3, 3):
            raise ValueError("rotations must have shape (M, 3, 3)")
        if self.directions.shape != (self.valid_count, 3):
            raise ValueError("directions must have shape (M, 3)")
        margin_names = self.margin_names
        if set(self.margins) != set(margin_names):
            raise ValueError(f"margins must carry exactly {margin_names}")
        if self.sample_indices.shape != (self.valid_count,):
            raise ValueError("sample_indices must have shape (M,)")
        object.__setattr__(self, "_tree", cKDTree(self.directions))

    @property
    def valid_count(self) -> int:
        return int(self.rotations.shape[0])

    @property
    def faces(self) -> tuple[int, ...]:
        """The face sequence named by ``path_id`` (:func:`.optics.faces_of_path_id`)."""
        return faces_of_path_id(self.path_id)

    @property
    def margin_names(self) -> tuple[str, ...]:
        return domain_margin_names(self.faces)

    def candidates(self, target_direction: np.ndarray, angle_tolerance_deg: float) -> np.ndarray:
        """Row indices whose direction lies within ``angle_tolerance_deg`` of ``target_direction``.

        Sorted ascending (i.e. in original sampling order).  Exactly the rows
        with ``directions @ target >= cos(tol)``: the tree ball is padded by
        a relative ``1e-9`` and the dot-product test is applied afterwards,
        so no boundary sample is ever decided by the tree's rounding.
        """
        target = np.asarray(target_direction, dtype=np.float64)
        if target.shape != (3,):
            raise ValueError("target_direction must have shape (3,)")
        if angle_tolerance_deg <= 0.0:
            raise ValueError("angle_tolerance_deg must be positive")
        prefilter = np.asarray(
            self._tree.query_ball_point(target, r=chord_radius(angle_tolerance_deg) * (1.0 + 1e-9)),
            dtype=np.int64,
        )
        if prefilter.size == 0:
            return prefilter
        prefilter.sort()
        alignment = self.directions[prefilter] @ target
        return prefilter[alignment >= np.cos(np.radians(angle_tolerance_deg))]

    def prefix(self, sample_count: int) -> PrescanTable:
        """The table that ``build_prescan_table`` would produce for the first ``sample_count`` samples.

        Valid because the sampling stream is prefix-stable (module docstring);
        pinned by ``test_prefix_of_a_larger_table_equals_the_smaller_build``.
        """
        if not 0 < sample_count <= self.sample_count:
            raise ValueError("sample_count must lie in (0, self.sample_count]")
        keep = self.sample_indices < sample_count
        return PrescanTable(
            path_id=self.path_id,
            incident_direction=self.incident_direction,
            refractive_index=self.refractive_index,
            rotations=self.rotations[keep],
            directions=self.directions[keep],
            margins={name: values[keep] for name, values in self.margins.items()},
            sample_indices=self.sample_indices[keep],
            sample_count=int(sample_count),
            rng_seed=self.rng_seed,
        )

    def build_parameters(self) -> dict[str, Any]:
        """The inputs that determine the table's content (JSON-serialisable)."""
        return {
            "path_id": self.path_id,
            "incident_direction": [float(x) for x in self.incident_direction],
            "refractive_index": self.refractive_index,
            "sample_count": int(self.sample_count),
            "rng_seed": int(self.rng_seed),
        }

    def __getstate__(self) -> dict[str, Any]:
        return {name: value for name, value in self.__dict__.items() if name != "_tree"}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        object.__setattr__(self, "_tree", cKDTree(self.directions))

    # --- disk cache ------------------------------------------------------------

    def save(self, path: Path, *, repo: Path | None = None) -> dict[str, Path]:
        """Write ``path`` (``.npz``) and ``path + PROVENANCE_SUFFIX``; returns both."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            rotations=self.rotations,
            directions=self.directions,
            sample_indices=self.sample_indices,
            **{f"margin_{name}": self.margins[name] for name in self.margin_names},
        )
        provenance = {
            "format": "lumice-integral-prescan-table-v1",
            "build": self.build_parameters(),
            "valid_count": self.valid_count,
            "git_commit": git_commit(repo),
            "arrays": {"name": path.name, "sha256": sha256_of(path)},
        }
        provenance_path = provenance_path_of(path)
        provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True))
        return {"arrays": path, "provenance": provenance_path}

    @classmethod
    def load(cls, path: Path) -> PrescanTable:
        """Read a table written by :meth:`save`; verifies the ``.npz`` hash first."""
        path = Path(path)
        provenance = json.loads(provenance_path_of(path).read_text())
        actual = sha256_of(path)
        if actual != provenance["arrays"]["sha256"]:
            raise ValueError(f"{path.name}: sha256 mismatch (provenance {provenance['arrays']['sha256']}, file {actual})")
        build = provenance["build"]
        path_id = str(build["path_id"])
        with np.load(path) as payload:
            return cls(
                path_id=path_id,
                incident_direction=np.asarray(build["incident_direction"], dtype=np.float64),
                refractive_index=float(build["refractive_index"]),
                rotations=payload["rotations"],
                directions=payload["directions"],
                margins={name: payload[f"margin_{name}"] for name in domain_margin_names(faces_of_path_id(path_id))},
                sample_indices=payload["sample_indices"],
                sample_count=int(build["sample_count"]),
                rng_seed=int(build["rng_seed"]),
            )


def provenance_path_of(path: Path) -> Path:
    path = Path(path)
    return path.with_name(path.name + PROVENANCE_SUFFIX)


def build_prescan_table(
    incident_direction: np.ndarray,
    refractive_index: float,
    *,
    sample_count: int = DEFAULT_SAMPLE_COUNT,
    rng_seed: int = DEFAULT_RNG_SEED,
    batch_size: int = DEFAULT_BATCH_SIZE,
    path_id: str = DEFAULT_PATH_ID,
) -> PrescanTable:
    """Sample ``sample_count`` Haar poses in ``batch_size`` chunks and keep the domain-valid ones.

    ``path_id`` names the face sequence (:func:`.optics.path_id_of` format;
    parsed once by :func:`.optics.faces_of_path_id`, which rejects malformed
    ids and unknown faces).  ``batch_size`` bounds the transient memory of
    one :func:`.optics.path_domain_batch` call and nothing else (module
    docstring).
    """
    faces = faces_of_path_id(path_id)
    path_id = path_id_of(faces)
    if sample_count < 1 or batch_size < 1:
        raise ValueError("sample_count and batch_size must be positive")
    incident = np.asarray(incident_direction, dtype=np.float64)
    index = float(refractive_index)
    rng = np.random.default_rng(rng_seed)
    margin_names = domain_margin_names(faces)
    rotations: list[np.ndarray] = []
    directions: list[np.ndarray] = []
    margins: dict[str, list[np.ndarray]] = {name: [] for name in margin_names}
    sample_indices: list[np.ndarray] = []
    for start in range(0, sample_count, batch_size):
        count = min(batch_size, sample_count - start)
        chunk = haar_rotations(count, rng)
        domain = path_domain_batch(chunk, faces, incident, index)
        keep = np.nonzero(domain.valid)[0]
        rotations.append(chunk[keep])
        directions.append(domain.direction[keep])
        for name in margin_names:
            margins[name].append(domain.margins[name][keep])
        sample_indices.append(keep + start)
    return PrescanTable(
        path_id=path_id,
        incident_direction=incident,
        refractive_index=index,
        rotations=np.concatenate(rotations) if rotations else np.zeros((0, 3, 3)),
        directions=np.concatenate(directions) if directions else np.zeros((0, 3)),
        margins={name: np.concatenate(values) for name, values in margins.items()},
        sample_indices=np.concatenate(sample_indices).astype(np.int64),
        sample_count=int(sample_count),
        rng_seed=int(rng_seed),
    )


def build_or_load_prescan_table(
    cache_path: Path | None,
    incident_direction: np.ndarray,
    refractive_index: float,
    *,
    sample_count: int = DEFAULT_SAMPLE_COUNT,
    rng_seed: int = DEFAULT_RNG_SEED,
    batch_size: int = DEFAULT_BATCH_SIZE,
    path_id: str = DEFAULT_PATH_ID,
    repo: Path | None = None,
    log: LogCallback | None = None,
) -> PrescanTable:
    """Load ``cache_path`` if its provenance matches the requested build; otherwise build (and cache).

    ``cache_path=None`` builds in memory only.  A cache whose build
    parameters differ, whose hash fails, or whose files are missing is
    rebuilt and overwritten, with the reason reported through ``log``
    (same policy as :func:`.strip_driver.load_checkpoints`).
    """
    requested = {
        "path_id": path_id_of(faces_of_path_id(path_id)),
        "incident_direction": [float(x) for x in np.asarray(incident_direction, dtype=np.float64)],
        "refractive_index": float(refractive_index),
        "sample_count": int(sample_count),
        "rng_seed": int(rng_seed),
    }
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists() and provenance_path_of(cache_path).exists():
            try:
                table = PrescanTable.load(cache_path)
            except (ValueError, KeyError, OSError) as error:
                if log is not None:
                    log(f"prescan cache {cache_path} unreadable ({error}); rebuilding")
            else:
                if table.build_parameters() == requested:
                    if log is not None:
                        log(f"prescan table loaded from {cache_path} ({table.valid_count} valid of {table.sample_count})")
                    return table
                if log is not None:
                    log(f"prescan cache {cache_path} was built with different parameters; rebuilding")
        elif log is not None:
            if cache_path.exists():
                log(f"prescan cache {cache_path} missing provenance sidecar; rebuilding")
            else:
                log(f"prescan cache {cache_path} absent; building")
    start = time.perf_counter()
    table = build_prescan_table(
        incident_direction,
        refractive_index,
        sample_count=sample_count,
        rng_seed=rng_seed,
        batch_size=batch_size,
        path_id=path_id,
    )
    if log is not None:
        log(
            f"prescan table built: {table.valid_count} valid of {table.sample_count} samples "
            f"in {time.perf_counter() - start:.1f}s"
        )
    if cache_path is not None:
        table.save(cache_path, repo=repo)
        if log is not None:
            log(f"prescan table cached at {cache_path}")
    return table


__all__ = [
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_PATH_ID",
    "DEFAULT_RNG_SEED",
    "DEFAULT_SAMPLE_COUNT",
    "PROVENANCE_SUFFIX",
    "PrescanTable",
    "build_or_load_prescan_table",
    "build_prescan_table",
    "chord_radius",
    "haar_rotations",
    "provenance_path_of",
]
