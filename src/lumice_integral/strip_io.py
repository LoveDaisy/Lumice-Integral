"""On-disk format of a rendered strip: raw arrays, status layer, provenance.

Layout (all arrays ``(height, width)`` row-major, row 0 at the top, matching
the historical ``data_251x801.bin`` read as ``(801, 251)``):

- ``strip_float64.bin``: headerless little-endian float64 pixel values
  (partial sums of the integrated components; ``0.0`` where nothing was
  integrated *and* outside the rendered window -- the status layer tells them
  apart);
- ``strip_float32.bin``: the same values cast to float32, the historical
  storage type;
- ``status_uint8.bin``: :data:`STATUS_BITS` bit mask per pixel; ``0`` means the
  pixel was not rendered;
- ``component_count_uint8.bin``: integrated components per pixel;
- ``pixels.csv``: one diagnostic row per rendered pixel (value, error
  estimate, completeness, seed source, counts, events, timings);
- ``provenance.json``: scene binding with provenance tags, every numerical
  option, the pixel model, window, environment, timings, and the SHA-256 of
  every payload above.

Nothing here depends on Lumice or on solver objects; a reader needs numpy and
this file's constants only (:func:`read_strip`).
"""

from __future__ import annotations

import csv
import json
import os
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .canonical_scene import (
    CANONICAL_HEIGHT_RATIO,
    CANONICAL_REFRACTIVE_INDEX,
    CANONICAL_RENDER,
    CANONICAL_SUN_ALTITUDE_DEG,
    CANONICAL_SUN_AZIMUTH_DEG,
    CANONICAL_WAVELENGTH_NM,
    CANONICAL_ZENITH_MEAN_DEG,
    CANONICAL_ZENITH_STD_DEG,
)
from .provenance import git_commit as _git_commit, sha256_of
from .quadrature import INTEGRAND_FACTOR_NAMES, QUADRATURE_METHOD
from .strip_pixel import (
    EVENT_NAMES,
    STAGE_NAMES,
    STATUS_ARCLENGTH_JUMP,
    STATUS_COLD_CHECK_MISMATCH,
    STATUS_COLD_DISCOVERY,
    STATUS_DEPTH_EXHAUSTED,
    STATUS_HAS_COMPONENT,
    STATUS_PRODUCTION_FAILURE,
    STATUS_RENDERED,
    STATUS_UNKNOWN_COMPLETENESS,
    PixelOptions,
    PixelResult,
)

FORMAT_VERSION = "lumice-integral.strip/v1"
FILE_NAMES = {
    "float64": "strip_float64.bin",
    "float32": "strip_float32.bin",
    "status": "status_uint8.bin",
    "component_count": "component_count_uint8.bin",
    "pixels": "pixels.csv",
    "provenance": "provenance.json",
}
STATUS_BITS: dict[str, int] = {
    "rendered": STATUS_RENDERED,
    "unknown_completeness": STATUS_UNKNOWN_COMPLETENESS,
    "has_component": STATUS_HAS_COMPONENT,
    "cold_discovery": STATUS_COLD_DISCOVERY,
    "arclength_jump": STATUS_ARCLENGTH_JUMP,
    "production_failure": STATUS_PRODUCTION_FAILURE,
    "depth_exhausted": STATUS_DEPTH_EXHAUSTED,
    "cold_check_mismatch": STATUS_COLD_CHECK_MISMATCH,
}
STATUS_BIT_MEANINGS: dict[str, str] = {
    "rendered": "pixel was computed (0 = outside the rendered window; its value is a placeholder 0)",
    "unknown_completeness": (
        "procedural completeness is 'unknown': an unclassified candidate, a "
        "production trace that did not close or changed arclength, or an "
        "unavailable quadrature; the value is the partial sum of what was integrated"
    ),
    "has_component": "at least one closed component was integrated into the value",
    "cold_discovery": "the prescan ran for this pixel (first pixel, fallback, or cold check); unset = pure hot start",
    "arclength_jump": "hot start from the previous pixel was rejected by detect_arclength_jump (topology boundary)",
    "production_failure": "a production retrace did not close or a quadrature was unavailable",
    "depth_exhausted": "adaptive quadrature hit maximum_refinement_depth on some edge",
    "cold_check_mismatch": "a scheduled cold check disagreed with the hot-start chain; the cold result was kept",
}
PIXEL_CSV_COLUMNS = (
    "row",
    "column",
    "value",
    "error_estimate",
    "component_count",
    "completeness",
    "discovery_completeness",
    "seed_source",
    "incomplete_count",
    "pool_count",
    "raw_cluster_count",
    "admissible_count",
    "status_bits",
    "component_arclengths",
    "production_pose_counts",
    "quadrature_refinements",
    "quadrature_max_depth",
    *(f"event_{name}" for name in EVENT_NAMES),
    *STAGE_NAMES,
)


@dataclass(frozen=True)
class Window:
    """Half-open pixel ranges ``rows[0]:rows[1]`` x ``columns[0]:columns[1]``.

    ``column_step > 1`` renders every ``column_step``-th column only (a coarse
    full-height preview); rows are always contiguous because the hot-start
    chain runs down a column.
    """

    rows: tuple[int, int]
    columns: tuple[int, int]
    column_step: int = 1

    def __post_init__(self) -> None:
        if not (0 <= self.rows[0] < self.rows[1]) or not (0 <= self.columns[0] < self.columns[1]):
            raise ValueError("window ranges must be non-empty and non-negative")
        if self.column_step < 1:
            raise ValueError("column_step must be positive")

    @property
    def row_range(self) -> range:
        return range(*self.rows)

    @property
    def column_range(self) -> range:
        return range(self.columns[0], self.columns[1], self.column_step)

    @property
    def pixel_count(self) -> int:
        return len(self.row_range) * len(self.column_range)

    def as_json(self) -> dict[str, Any]:
        return {
            "rows": list(self.rows),
            "columns": list(self.columns),
            "column_step": self.column_step,
            "pixel_count": self.pixel_count,
        }


@dataclass(frozen=True)
class StripArrays:
    """The four image-shaped payloads of a render."""

    values: np.ndarray  # float64 (height, width)
    status: np.ndarray  # uint8
    component_count: np.ndarray  # uint8

    @property
    def rendered(self) -> np.ndarray:
        return (self.status & STATUS_RENDERED) != 0


def assemble_arrays(results: Iterable[PixelResult], *, height: int, width: int) -> StripArrays:
    values = np.zeros((height, width), dtype=np.float64)
    status = np.zeros((height, width), dtype=np.uint8)
    component_count = np.zeros((height, width), dtype=np.uint8)
    for result in results:
        values[result.row, result.column] = result.value
        status[result.row, result.column] = result.status_bits
        component_count[result.row, result.column] = min(result.component_count, 255)
    return StripArrays(values, status, component_count)


def pixel_csv_row(result: PixelResult) -> dict[str, Any]:
    timings = result.timings
    return {
        "row": result.row,
        "column": result.column,
        "value": repr(result.value),
        "error_estimate": repr(result.error_estimate),
        "component_count": result.component_count,
        "completeness": result.completeness,
        "discovery_completeness": result.discovery_completeness,
        "seed_source": result.seed_source,
        "incomplete_count": result.incomplete_count,
        "pool_count": result.pool_count,
        "raw_cluster_count": result.raw_cluster_count,
        "admissible_count": result.admissible_count,
        "status_bits": result.status_bits,
        "component_arclengths": ";".join(f"{c.discovery_arclength:.6f}" for c in result.components),
        "production_pose_counts": ";".join(str(c.production_pose_count) for c in result.components),
        "quadrature_refinements": ";".join(str(c.refinements) for c in result.components),
        "quadrature_max_depth": ";".join(str(c.maximum_depth_reached) for c in result.components),
        **{f"event_{name}": result.events.get(name, 0) for name in EVENT_NAMES},
        **{name: f"{timings.get(name, 0.0):.4f}" for name in STAGE_NAMES},
    }


def write_pixel_csv(path: Path, results: Sequence[PixelResult]) -> None:
    ordered = sorted(results, key=lambda r: (r.column, r.row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PIXEL_CSV_COLUMNS)
        writer.writeheader()
        for result in ordered:
            writer.writerow(pixel_csv_row(result))


def environment_block() -> dict[str, Any]:
    import jax

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "hostname": platform.node(),
        "jax": jax.__version__,
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "numpy": np.__version__,
        # Process-level knobs that change memory/threading but not values.
        "env": {
            name: os.environ.get(name)
            for name in (
                "XLA_FLAGS",
                "XLA_PYTHON_CLIENT_PREALLOCATE",
                "OMP_NUM_THREADS",
                "JAX_PLATFORMS",
                "MALLOC_ARENA_MAX",
                "MALLOC_TRIM_THRESHOLD_",
                "MALLOC_MMAP_THRESHOLD_",
            )
        },
    }


def scene_block() -> dict[str, Any]:
    """Canonical scene constants, each tagged with its ``docs/ch06-reference-fixture.md`` provenance."""
    return {
        "specification": "docs/ch06-reference-fixture.md section 3.3",
        "path": {"value": [3, 5], "provenance": "historical-direct"},
        "crystal": {
            "value": {"type": "hexagonal_column", "height_ratio": CANONICAL_HEIGHT_RATIO},
            "provenance": "historical-direct",
        },
        "sun": {
            "value": {"altitude_deg": CANONICAL_SUN_ALTITUDE_DEG, "azimuth_deg": CANONICAL_SUN_AZIMUTH_DEG, "diameter_deg": 0.0},
            "provenance": "historical-inferred",
        },
        "refractive_index": {"value": CANONICAL_REFRACTIVE_INDEX, "provenance": "canonical-new"},
        "wavelength_nm": {"value": CANONICAL_WAVELENGTH_NM, "provenance": "canonical-new"},
        "pose_density": {
            "value": {
                "model": "zenith-gaussian column",
                "zenith_mean_deg": CANONICAL_ZENITH_MEAN_DEG,
                "zenith_std_deg": CANONICAL_ZENITH_STD_DEG,
            },
            "provenance": "canonical-new",
        },
        "camera": {"value": {"lens": "linear", **CANONICAL_RENDER}, "provenance": "canonical-new"},
        "image_shape": {"value": [CANONICAL_RENDER["height"], CANONICAL_RENDER["width"]], "provenance": "historical-direct"},
    }


def options_block(options: PixelOptions, prescan: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """``PixelOptions`` plus the scene-level prescan build parameters (``prescan``, if any)."""
    return {
        "discovery": {
            "prescan": dict(prescan) if prescan is not None else None,
            "discovery_step_budget": options.discovery_step_budget,
            "retry_step_budget": options.effective_retry_step_budget,
            "stall_floor_window": options.stall_floor_window,
            "angle_tolerance_deg": options.angle_tolerance_deg,
            "cluster_radius_rad": options.cluster_radius_rad,
            "arclength_rtol": options.arclength_rtol,
            "jump_relative_threshold": options.jump_relative_threshold,
            "strategy": (
                "column-wise top-down scan; hot start every integrated component of "
                "the pixel above with the production step budget, reject on arclength "
                "jump and fall back to cold discovery; cold discovery queries the scene-level "
                "prescan table (built once per run from prescan.sample_count Haar samples with "
                "prescan.rng_seed, domain-valid poses indexed by outgoing direction) for the "
                "candidates within angle_tolerance_deg, then runs the small "
                "discovery budget, incomplete candidates retraced once with retry_step_budget "
                "unless a step_budget candidate's last stall_floor_window accepted discovery "
                "steps all sat at continuation.minimum_step (counted as incomplete_stall_skip, "
                "kept incomplete without a retrace); "
                "components deduplicated by (status, reason, arclength within arclength_rtol)"
            ),
        },
        "continuation": asdict(options.continuation),
        "quadrature": {
            **asdict(options.quadrature),
            "method": QUADRATURE_METHOD,
            "integrand_factors": list(INTEGRAND_FACTOR_NAMES),
            "density_factor": "rho_pose",
            "haar_to_dvol_g_factor_applied": True,
            "component_sum": "linear sum of component values; error estimates summed linearly (conservative bound)",
        },
    }


def write_strip(
    output_dir: Path,
    results: Sequence[PixelResult],
    *,
    options: PixelOptions,
    window: Window,
    height: int,
    width: int,
    pixel_model: Mapping[str, Any],
    execution: Mapping[str, Any],
    repo: Path | None = None,
    prescan: Mapping[str, Any] | None = None,
) -> dict[str, Path]:
    """Write every payload plus ``provenance.json``; returns the file map.

    ``prescan`` is the scene-level prescan build record
    (:meth:`.strip_driver.PrescanBuildOptions.as_json`), stored under
    ``options.discovery.prescan``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays = assemble_arrays(results, height=height, width=width)
    files = {key: output_dir / name for key, name in FILE_NAMES.items()}
    arrays.values.astype("<f8").tofile(files["float64"])
    arrays.values.astype("<f4").tofile(files["float32"])
    arrays.status.astype(np.uint8).tofile(files["status"])
    arrays.component_count.astype(np.uint8).tofile(files["component_count"])
    write_pixel_csv(files["pixels"], results)

    rendered = arrays.rendered
    values = arrays.values[rendered]
    unknown = ((arrays.status & STATUS_UNKNOWN_COMPLETENESS) != 0) & rendered
    timings = {key: float(sum(r.timings.get(key, 0.0) for r in results)) for key in STAGE_NAMES}
    provenance = {
        "format": FORMAT_VERSION,
        "product": "ch06 251 x 801 direct 3-5 strip (partial physical integrand, point light source)",
        "generator": {
            "package": "lumice_integral",
            "modules": ["strip_pixel", "strip_driver", "strip_io"],
            "git_commit": _git_commit(repo),
            "lumice_dependency": "none (independent implementation; Lumice is neither imported nor invoked)",
        },
        "scene": scene_block(),
        "options": options_block(options, prescan),
        "pixel_model": dict(pixel_model),
        "window": window.as_json(),
        "arrays": {
            "shape": [height, width],
            "order": "row-major, row 0 at the top of the image, column 0 at the left",
            "files": {
                "float64": {"name": FILE_NAMES["float64"], "dtype": "<f8", "sha256": sha256_of(files["float64"])},
                "float32": {"name": FILE_NAMES["float32"], "dtype": "<f4", "sha256": sha256_of(files["float32"])},
                "status": {"name": FILE_NAMES["status"], "dtype": "u1", "sha256": sha256_of(files["status"])},
                "component_count": {"name": FILE_NAMES["component_count"], "dtype": "u1", "sha256": sha256_of(files["component_count"])},
                "pixels": {"name": FILE_NAMES["pixels"], "sha256": sha256_of(files["pixels"])},
            },
            "status_bits": STATUS_BITS,
            "status_bit_meanings": STATUS_BIT_MEANINGS,
            "value_semantics": (
                "sum over integrated closed components of the Haar-converted partial integral "
                "(1/(8 pi^2)) rho_pose * entry_measure * fresnel_transmission * path_validity / (J_perp + epsilon) dH^1; "
                "unknown-completeness pixels hold the partial sum of what was integrated (never NaN); "
                "unrendered pixels hold 0 with status 0"
            ),
            "radiometric_normalization": "not aligned with the historical raw or Lumice; morphology/relative profiles only",
        },
        "summary": {
            "rendered_pixels": int(rendered.sum()),
            "unknown_completeness_pixels": int(unknown.sum()),
            "pixels_with_components": int(((arrays.status & STATUS_HAS_COMPONENT) != 0).sum()),
            "cold_discovery_pixels": int(((arrays.status & STATUS_COLD_DISCOVERY) != 0).sum()),
            "arclength_jump_pixels": int(((arrays.status & STATUS_ARCLENGTH_JUMP) != 0).sum()),
            "cold_check_mismatch_pixels": int(((arrays.status & STATUS_COLD_CHECK_MISMATCH) != 0).sum()),
            "value_min": float(values.min()) if values.size else None,
            "value_max": float(values.max()) if values.size else None,
            "value_mean": float(values.mean()) if values.size else None,
            "component_count_max": int(arrays.component_count.max()),
            "stage_cpu_seconds": timings,
            "stage_fractions": {
                key: (timings[key] / timings["total_s"] if timings["total_s"] else None)
                for key in STAGE_NAMES
                if key != "total_s"
            },
            "per_pixel_mean_s": timings["total_s"] / max(len(results), 1),
        },
        "execution": dict(execution),
        "environment": environment_block(),
    }
    files["provenance"].write_text(json.dumps(provenance, indent=2, sort_keys=False) + "\n")
    return files


def read_strip(output_dir: Path) -> tuple[StripArrays, dict[str, Any]]:
    """Read a render back (verifying the recorded SHA-256 of every array)."""
    output_dir = Path(output_dir)
    provenance = json.loads((output_dir / FILE_NAMES["provenance"]).read_text())
    height, width = provenance["arrays"]["shape"]
    payloads = {}
    for key, dtype in (("float64", "<f8"), ("status", "u1"), ("component_count", "u1")):
        entry = provenance["arrays"]["files"][key]
        path = output_dir / entry["name"]
        actual = sha256_of(path)
        if actual != entry["sha256"]:
            raise ValueError(f"{entry['name']}: sha256 {actual} != recorded {entry['sha256']}")
        payloads[key] = np.fromfile(path, dtype=dtype).reshape(height, width)
    return StripArrays(payloads["float64"], payloads["status"], payloads["component_count"]), provenance


__all__ = [
    "FILE_NAMES",
    "FORMAT_VERSION",
    "PIXEL_CSV_COLUMNS",
    "STATUS_BITS",
    "STATUS_BIT_MEANINGS",
    "StripArrays",
    "Window",
    "assemble_arrays",
    "environment_block",
    "options_block",
    "pixel_csv_row",
    "read_strip",
    "scene_block",
    "sha256_of",
    "write_pixel_csv",
    "write_strip",
]
