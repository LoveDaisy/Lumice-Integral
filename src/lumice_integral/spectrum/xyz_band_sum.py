"""Colour band sums: one monochrome :func:`..band_sum.render_band_sum_window` per distinct ``n(lambda)``, summed to XYZ.

This module orchestrates :func:`lumice_integral.band_sum.render_band_sum_window`;
``band_sum`` does not depend on :mod:`lumice_integral.spectrum`.  The estimator
is untouched: a wavelength is a refractive index, i.e. a monochrome scene and
its own S^2 stores (``s2_store`` keys them by ``n``), and the colour image is
the Lumice-normalised pool average of the monochrome values

    XYZ[p] = (1 / M) sum_{i < M} spd_i (xbar_i, ybar_i, zbar_i) V_{n_i}(p) / S_bar

with ``V_n`` the band-sum value at index ``n`` (crystal length^2 per
steradian), the ``M`` slots of :func:`.wl_pool.wavelength_pool` and ``S_bar``
the weight :func:`.wl_pool.emitted_weight` charges each emitted ray.  This is
Lumice's per-ray slot draw (uniform over ``M``, weight ``spd_i``) over its
``emitted_energy`` denominator, so a Lumice float export satisfies, channel by
channel, ``raw[p, c] / E = XYZ[p, c] Omega_p / (S / 2)``
(``scripts/compare_lumice_family.py``).  A single discrete wavelength gives
``XYZ = CMF(lambda) V`` whatever its weight and ``M``; at 550 nm the ``Y``
channel is ``ybar(550) V``, the monochrome convention ``raw / E = ybar(550)
Omega_p V / (S / 2)``, bit for bit.

Slots with exactly equal ``n`` (a discrete pool; ``==``, the rule of
:func:`.store.build_wavelength_pool_stores`) are rendered once and weighted by
the sum of their slots' weights.  Per pixel, ``K_min`` / ``K_rho_pos_min`` /
``K_eff_min`` are the minimum over the rendered indices whose value at that
pixel is non-zero (``0`` where none is): the noisiest contributing wavelength,
independent of the channel.

Output (:func:`write_xyz_band_sum_strip`, :func:`read_xyz_band_sum_strip`):
the ``strip_io`` directory layout with its own float payloads
(``xyz_float64.bin`` / ``xyz_float32.bin``, ``(H, W, 3)`` channel-last X, Y, Z,
row 0 at the top: the layout of Lumice's ``img_0N.npy``), the ``status`` and
``component_count`` images, a colour ``pixels.csv`` and ``provenance.json``
with ``format`` :data:`FORMAT_VERSION`.  ``strip_io.read_strip`` does not read
it (no ``strip_float64.bin``).

Nothing here imports or calls Lumice.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..band_sum import FORMAT_VERSION as MONOCHROME_FORMAT_VERSION
from ..band_sum import ESTIMATOR, K_EFF_SEMANTICS, BandSumScene, render_band_sum_window
from ..provenance import git_commit, sha256_of
from ..s2_store import DEFAULT_CACHE_DIR
from ..strip_io import FILE_NAMES, Window, environment_block, scene_block
from ..strip_pixel import STATUS_HAS_COMPONENT, STATUS_RENDERED
from .wl_pool import WlPoolEntry

FORMAT_VERSION = "lumice-integral.band-sum-xyz/v1"
CHANNELS = ("X", "Y", "Z")
XYZ_FILE_NAMES = {
    "xyz_float64": "xyz_float64.bin",
    "xyz_float32": "xyz_float32.bin",
    "status": FILE_NAMES["status"],
    "component_count": FILE_NAMES["component_count"],
    "pixels": FILE_NAMES["pixels"],
    "provenance": FILE_NAMES["provenance"],
}
PIXEL_CSV_COLUMNS = ("row", "column", "X", "Y", "Z", "delta_deg", "band_width_rad", "K_min", "K_rho_pos_min", "K_eff_min")
XYZ_FORMULA = (
    "XYZ[p] = (1/M) sum_i spd_i (xbar_i, ybar_i, zbar_i) V_{n_i}(p) / emitted_weight; V_n the monochrome band sum "
    "at refractive index n (options.estimator); Lumice raw[p, c] / emitted_energy = XYZ[p, c] Omega_p / (S / 2)"
)


@dataclasses.dataclass(frozen=True)
class XyzBandSumPixelResult:
    """One pixel: its XYZ, its deviation band and the worst diagnostics over the contributing wavelengths."""

    row: int
    column: int
    xyz: tuple[float, float, float]
    delta: float
    band_width_rad: float
    K_min: int
    K_rho_pos_min: int
    K_eff_min: float

    def csv_row(self) -> dict[str, Any]:
        return {
            "row": self.row,
            "column": self.column,
            **{c: repr(float(v)) for c, v in zip(CHANNELS, self.xyz)},
            "delta_deg": repr(float(np.degrees(self.delta))),
            "band_width_rad": repr(self.band_width_rad),
            "K_min": self.K_min,
            "K_rho_pos_min": self.K_rho_pos_min,
            "K_eff_min": repr(self.K_eff_min),
        }


@dataclasses.dataclass(frozen=True)
class IndexGroup:
    """The pool slots sharing one refractive index, and their summed ``spd * CMF`` weights."""

    refractive_index: float
    entries: tuple[WlPoolEntry, ...]

    @property
    def weights(self) -> tuple[float, float, float]:
        x = y = z = 0.0
        for e in self.entries:
            x += e.spd_weight * e.cmf_x
            y += e.spd_weight * e.cmf_y
            z += e.spd_weight * e.cmf_z
        return x, y, z

    def as_json(self) -> dict[str, Any]:
        return {
            "refractive_index": self.refractive_index,
            "slots": len(self.entries),
            "wavelength_nm": sorted({e.wavelength_nm for e in self.entries}),
            "weights_xyz": list(self.weights),
        }


def index_groups(pool: Sequence[WlPoolEntry]) -> list[IndexGroup]:
    """The pool's slots grouped by exactly equal refractive index, in order of first appearance."""
    groups: dict[float, list[WlPoolEntry]] = {}
    for entry in pool:
        groups.setdefault(entry.refractive_index, []).append(entry)
    return [IndexGroup(n, tuple(entries)) for n, entries in groups.items()]


def pool_json(pool: Sequence[WlPoolEntry]) -> list[dict[str, float]]:
    return [dataclasses.asdict(e) for e in pool]


def render_xyz_band_sum_window(
    scene: BandSumScene,
    pool: Sequence[WlPoolEntry],
    emitted_weight: float,
    window: Window,
    store_n: int,
    *,
    workers: int = 1,
    base_dir: Path = DEFAULT_CACHE_DIR,
    run_checks: bool = True,
    log: Callable[[str], None] | None = None,
) -> tuple[list[XyzBandSumPixelResult], dict[str, Any]]:
    """XYZ of every pixel of ``window`` (module docstring): one monochrome render per distinct ``n`` of ``pool``.

    ``scene.refractive_index`` must be NaN: each index group renders
    ``dataclasses.replace(scene, refractive_index=n)``, and a real index there
    would be silently ignored.  ``emitted_weight`` is
    :func:`.wl_pool.emitted_weight` of the spectrum ``pool`` was built from.
    Returns the pixel results (in the window's order) and an execution record
    with one monochrome execution record per index group.

    ``delta`` and ``band_width_rad`` come from ``pixel_band(row, column, sun,
    render)`` (``band_sum.py``): camera and sun geometry only, independent of
    ``n`` (a rank-0 path class instead gives the fixed NaN placeholder of
    :func:`.band_sum.render_band_sum_window`).  Either way they must be
    identical across index groups; this is asserted rather than assumed, so a
    future change coupling them to ``n`` fails loudly here instead of
    silently keeping only the last group's value.
    """
    if not math.isnan(scene.refractive_index):
        raise ValueError("a colour render takes n from the pool: build the scene with refractive_index=nan")
    if not pool:
        raise ValueError("the wavelength pool is empty")
    if not emitted_weight > 0.0:
        raise ValueError(f"emitted_weight must be positive, got {emitted_weight}")
    start = time.perf_counter()
    groups = index_groups(pool)
    scale = 1.0 / (len(pool) * emitted_weight)
    pixels = [(row, column) for column in window.column_range for row in window.row_range]
    xyz = np.zeros((len(pixels), 3))
    k_min = np.full(len(pixels), np.iinfo(np.int64).max)
    k_pos_min = np.full(len(pixels), np.iinfo(np.int64).max)
    k_eff_min = np.full(len(pixels), np.inf)
    records = []
    for index, group in enumerate(groups):
        if log is not None:
            log(f"index group {index + 1}/{len(groups)}: n = {group.refractive_index!r} ({len(group.entries)} slot(s))")
        results, execution = render_band_sum_window(
            dataclasses.replace(scene, refractive_index=group.refractive_index),
            window,
            store_n,
            workers=workers,
            base_dir=base_dir,
            run_checks=run_checks,
            log=log,
        )
        if [(r.row, r.column) for r in results] != pixels:
            raise RuntimeError("render_band_sum_window returned the pixels in another order")
        value = np.array([r.value for r in results])
        xyz += np.outer(value, group.weights)
        lit = value != 0.0
        k_min[lit] = np.minimum(k_min[lit], [r.K for r, on in zip(results, lit) if on])
        k_pos_min[lit] = np.minimum(k_pos_min[lit], [r.K_rho_pos for r, on in zip(results, lit) if on])
        k_eff_min[lit] = np.minimum(k_eff_min[lit], [r.K_eff for r, on in zip(results, lit) if on])
        group_delta = np.array([r.delta for r in results])
        group_width = np.array([r.band_width_rad for r in results])
        if index == 0:
            delta, width = group_delta, group_width
        elif not (
            np.array_equal(delta, group_delta, equal_nan=True) and np.array_equal(width, group_width, equal_nan=True)
        ):
            raise RuntimeError(
                "delta/band_width_rad differ across wavelength index groups; they are camera/sun-only "
                "geometry (band_sum.pixel_band), or the rank-0 NaN placeholder, and must not depend on n(lambda)"
            )
        records.append({**group.as_json(), "execution": execution})
    xyz *= scale
    unlit = ~np.isfinite(k_eff_min)
    k_min[unlit], k_pos_min[unlit], k_eff_min[unlit] = 0, 0, 0.0
    out = [
        XyzBandSumPixelResult(
            row, column, (float(xyz[i, 0]), float(xyz[i, 1]), float(xyz[i, 2])),
            float(delta[i]), float(width[i]), int(k_min[i]), int(k_pos_min[i]), float(k_eff_min[i]),
        )
        for i, (row, column) in enumerate(pixels)
    ]  # fmt: skip
    execution = {
        "workers": workers,
        "N": store_n,
        "store_base_dir": str(base_dir),
        "slots": len(pool),
        "emitted_weight": emitted_weight,
        "index_groups": records,
        "wall_clock_s": time.perf_counter() - start,
    }
    return out, execution


# ---------------------------------------------------------------- output
def write_xyz_band_sum_strip(
    output_dir: Path,
    results: Sequence[XyzBandSumPixelResult],
    *,
    scene: BandSumScene,
    pool: Sequence[WlPoolEntry],
    spectrum: Mapping[str, Any],
    emitted_weight: float,
    store_n: int,
    window: Window,
    pose_density_block: Mapping[str, Any],
    execution: Mapping[str, Any],
    repo: Path | None = None,
) -> dict[str, Path]:
    """The colour output directory (module docstring); ``spectrum`` is the run's spectrum options (illuminant or wavelength, ``M``)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    height, width = int(scene.render["height"]), int(scene.render["width"])
    values = np.zeros((height, width, 3), dtype=np.float64)
    status = np.zeros((height, width), dtype=np.uint8)
    component_count = np.zeros((height, width), dtype=np.uint8)
    for r in results:
        lit = any(v != 0.0 for v in r.xyz)
        values[r.row, r.column] = r.xyz
        status[r.row, r.column] = STATUS_RENDERED | (STATUS_HAS_COMPONENT if lit else 0)
        component_count[r.row, r.column] = 1 if lit else 0
    files = {key: output_dir / name for key, name in XYZ_FILE_NAMES.items()}
    values.astype("<f8").tofile(files["xyz_float64"])
    values.astype("<f4").tofile(files["xyz_float32"])
    status.tofile(files["status"])
    component_count.tofile(files["component_count"])
    with files["pixels"].open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PIXEL_CSV_COLUMNS)
        writer.writeheader()
        for r in sorted(results, key=lambda r: (r.column, r.row)):
            writer.writerow(r.csv_row())
    rendered = (status & STATUS_RENDERED) != 0
    is_rank0 = scene.path_class.halo_map_rank == 0  # a point mass: K_eff_min is a placeholder, as in band_sum
    k_eff = np.array([]) if is_rank0 else np.array([r.K_eff_min for r in results if any(v != 0.0 for v in r.xyz)])
    scene_json = scene_block(pose_density_block, crystal=scene.crystal)
    scene_json["refractive_index"] = {"value": "per wavelength: options.pool[].refractive_index", "provenance": "run-option"}
    scene_json["wavelength_nm"] = {"value": "per wavelength: options.pool[].wavelength_nm", "provenance": "run-option"}
    scene_json["path"] = {"value": list(scene.path_class.representative), "provenance": "run-option"}
    scene_json["camera"] = {"value": {"lens": "linear", **dict(scene.render)}, "provenance": "run-option"}
    scene_json["image_shape"] = {"value": [height, width], "provenance": "run-option"}
    groups = index_groups(pool)
    provenance = {
        "format": FORMAT_VERSION,
        "product": "S^2 band-sum colour render (CIE 1931 XYZ, band-averaged pixels, point light source)",
        "generator": {
            "package": "lumice_integral",
            "modules": ["spectrum.xyz_band_sum", "band_sum", "s2_store", "spectrum.wl_pool"],
            "git_commit": git_commit(repo),
            "lumice_dependency": "none (independent implementation; Lumice is neither imported nor invoked)",
        },
        "scene": scene_json,
        "options": {
            "xyz": XYZ_FORMULA,
            "estimator": ESTIMATOR,
            "monochrome_format": MONOCHROME_FORMAT_VERSION,
            "N": store_n,
            "spectrum": dict(spectrum),
            "slots": len(pool),
            "emitted_weight": emitted_weight,
            "pool": pool_json(pool),
            "index_groups": [g.as_json() for g in groups],
            "path_class": scene.path_class.provenance(),
            "symmetry_transport": scene.transport,
            "store_plan": [g.as_json() for g in scene.plan],
            "k_eff_semantics": K_EFF_SEMANTICS,
            "diagnostics": "K_min, K_rho_pos_min, K_eff_min: minimum over the index groups whose value at the pixel is non-zero",
        },
        "window": window.as_json(),
        "arrays": {
            "shape": [height, width, 3],
            "channels": list(CHANNELS),
            "order": "row-major, row 0 at the top of the image, column 0 at the left, channel last (X, Y, Z)",
            "files": {
                "xyz_float64": {"name": XYZ_FILE_NAMES["xyz_float64"], "dtype": "<f8", "sha256": sha256_of(files["xyz_float64"])},
                "xyz_float32": {"name": XYZ_FILE_NAMES["xyz_float32"], "dtype": "<f4", "sha256": sha256_of(files["xyz_float32"])},
                "status": {"name": XYZ_FILE_NAMES["status"], "dtype": "u1", "sha256": sha256_of(files["status"])},
                "component_count": {"name": XYZ_FILE_NAMES["component_count"], "dtype": "u1", "sha256": sha256_of(files["component_count"])},
                "pixels": {"name": XYZ_FILE_NAMES["pixels"], "sha256": sha256_of(files["pixels"]), "columns": list(PIXEL_CSV_COLUMNS)},
            },
            "status_bits": {"rendered": STATUS_RENDERED, "has_component": STATUS_HAS_COMPONENT},
            "value_semantics": (
                "linear CIE 1931 XYZ in Lumice's raw / emitted_energy normalisation divided by Omega_p / (S / 2) (options.xyz); "
                "each monochrome term a band average (options.estimator), not a point value"
            ),
        },
        "summary": {
            "rendered_pixels": int(rendered.sum()),
            "pixels_with_light": int((values != 0.0).any(axis=2).sum()),
            "xyz_max": [float(values[..., c].max()) for c in range(3)],
            "xyz_sum": [float(values[..., c].sum()) for c in range(3)],
            "K_eff_min_median_lit": float(np.median(k_eff)) if k_eff.size else None,
            "K_eff_min_min_lit": float(np.min(k_eff)) if k_eff.size else None,
        },
        "execution": dict(execution),
        "environment": environment_block(),
    }
    files["provenance"].write_text(json.dumps(provenance, indent=2) + "\n")
    return files


def read_xyz_band_sum_strip(output_dir: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """``(xyz (H, W, 3) float64, status (H, W), provenance)`` of a colour render, SHA-256 checked; ``ValueError`` on another format."""
    output_dir = Path(output_dir)
    provenance = json.loads((output_dir / XYZ_FILE_NAMES["provenance"]).read_text())
    if provenance.get("format") != FORMAT_VERSION:
        raise ValueError(f"{output_dir}: format {provenance.get('format')!r} is not {FORMAT_VERSION!r}")
    height, width, channels = provenance["arrays"]["shape"]
    payloads = {}
    for key, dtype, shape in (("xyz_float64", "<f8", (height, width, channels)), ("status", "u1", (height, width))):
        entry = provenance["arrays"]["files"][key]
        path = output_dir / entry["name"]
        actual = sha256_of(path)
        if actual != entry["sha256"]:
            raise ValueError(f"{entry['name']}: sha256 {actual} != recorded {entry['sha256']}")
        payloads[key] = np.fromfile(path, dtype=dtype).reshape(shape)
    return payloads["xyz_float64"], payloads["status"], provenance


__all__ = [
    "CHANNELS",
    "FORMAT_VERSION",
    "PIXEL_CSV_COLUMNS",
    "XYZ_FILE_NAMES",
    "IndexGroup",
    "XyzBandSumPixelResult",
    "index_groups",
    "read_xyz_band_sum_strip",
    "render_xyz_band_sum_window",
    "write_xyz_band_sum_strip",
]
