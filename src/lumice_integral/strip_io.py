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
  estimate, completeness, component kinds, counts, events, timings);
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
from .quadrature import INTEGRAND_FACTOR_NAMES, RESAMPLED_QUADRATURE_METHOD
from .strip_pixel import (
    EVENT_NAMES,
    STAGE_NAMES,
    STATUS_HAS_ARC,
    STATUS_HAS_COMPONENT,
    STATUS_NODE_COUNT_EXHAUSTED,
    STATUS_QUADRATURE_UNAVAILABLE,
    STATUS_RENDERED,
    STATUS_UNKNOWN_COMPLETENESS,
    PixelOptions,
    PixelResult,
)

# v2 (task-pixel-pipeline-v2): single-trace pipeline with open-arc
# components; status bits, CSV columns and the checkpoint payload changed.
FORMAT_VERSION = "lumice-integral.strip/v2"
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
    "has_arc": STATUS_HAS_ARC,
    "quadrature_unavailable": STATUS_QUADRATURE_UNAVAILABLE,
    "node_count_exhausted": STATUS_NODE_COUNT_EXHAUSTED,
}
STATUS_BIT_MEANINGS: dict[str, str] = {
    "rendered": "pixel was computed (0 = outside the rendered window; its value is a placeholder 0)",
    "unknown_completeness": (
        "procedural completeness is 'unknown': an admissible candidate did not converge to a "
        "closed loop or an open arc (incomplete), or a component's quadrature was unavailable; "
        "the value is the partial sum of what was integrated.  'complete' is not a certificate "
        "that every connected component of the fiber was found"
    ),
    "has_component": "at least one component (closed loop or open arc) was integrated into the value",
    "has_arc": (
        "at least one integrated component is an open arc: a fiber piece cut by a named event "
        "(TIR, branch, path infeasibility, visibility, chart) at both ends, traced forward and "
        "backward from one seed; its truncation estimates are in pixels.csv, not in the value"
    ),
    "quadrature_unavailable": "a component's quadrature was unavailable (it contributes 0 to the value)",
    "node_count_exhausted": (
        "the resampled quadrature reached maximum_node_count on some component with its "
        "error estimate still above relative_tolerance; the value is reported as is"
    ),
}
PIXEL_CSV_COLUMNS = (
    "row",
    "column",
    "value",
    "error_estimate",
    "component_count",
    "arc_count",
    "completeness",
    "incomplete_count",
    "pool_count",
    "extra_seed_count",
    "raw_cluster_count",
    "admissible_count",
    "status_bits",
    "component_kinds",
    "component_arclengths",
    "component_pose_counts",
    "component_end_reasons",
    "component_start_truncations",
    "component_end_truncations",
    "quadrature_node_counts",
    "quadrature_refinement_rounds",
    *(f"event_{name}" for name in EVENT_NAMES),
    *STAGE_NAMES,
)


@dataclass(frozen=True)
class Window:
    """Half-open pixel ranges ``rows[0]:rows[1]`` x ``columns[0]:columns[1]``.

    ``column_step > 1`` renders every ``column_step``-th column only (a coarse
    full-height preview); rows are always contiguous because the warm-seed
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
    components = result.components
    return {
        "row": result.row,
        "column": result.column,
        "value": repr(result.value),
        "error_estimate": repr(result.error_estimate),
        "component_count": result.component_count,
        "arc_count": result.arc_count,
        "completeness": result.completeness,
        "incomplete_count": result.incomplete_count,
        "pool_count": result.pool_count,
        "extra_seed_count": result.extra_seed_count,
        "raw_cluster_count": result.raw_cluster_count,
        "admissible_count": result.admissible_count,
        "status_bits": result.status_bits,
        "component_kinds": ";".join(c.kind for c in components),
        "component_arclengths": ";".join(f"{c.arclength:.6f}" for c in components),
        "component_pose_counts": ";".join(str(c.pose_count) for c in components),
        # ``start|end`` event names of each component (``closed_loop`` alone for a loop).
        "component_end_reasons": ";".join(
            f"{c.start_reason}|{c.reason}" if c.kind == "arc" else c.reason for c in components
        ),
        "component_start_truncations": ";".join(repr(c.start_truncation_estimate) for c in components),
        "component_end_truncations": ";".join(repr(c.end_truncation_estimate) for c in components),
        "quadrature_node_counts": ";".join(str(c.node_count) for c in components),
        "quadrature_refinement_rounds": ";".join(str(c.refinement_rounds) for c in components),
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
            "angle_tolerance_deg": options.angle_tolerance_deg,
            "cluster_radius_rad": options.cluster_radius_rad,
            "distance_threshold": options.distance_threshold,
            "strategy": (
                "column-wise top-down scan; per pixel one discovery: the candidate pool is the "
                "scene-level prescan table (built once per run from prescan.sample_count Haar "
                "samples with prescan.rng_seed, domain-valid poses indexed by outgoing direction) "
                "queried within angle_tolerance_deg, plus the integrated components of the pixel "
                "above as warm Gauss-Newton starts (never traced on their own); greedy geodesic "
                "clustering with cluster_radius_rad; each representative is Newton-corrected, gated "
                "(residual, path domain, entry measure), deduplicated by SO(3) distance below "
                "distance_threshold to any accepted curve, and traced once with the production "
                "continuation options; closed -> closed component; a named event (tir, branch, "
                "path_infeasible, visibility, chart) -> traced backward from the same seed and "
                "stitched into an open-arc component; anything else -> incomplete"
            ),
        },
        "continuation": asdict(options.continuation),
        "quadrature": {
            **asdict(options.quadrature),
            "method": RESAMPLED_QUADRATURE_METHOD,
            "integrand_factors": list(INTEGRAND_FACTOR_NAMES),
            "density_factor": "rho_pose",
            "haar_to_dvol_g_factor_applied": True,
            "component_sum": (
                "linear sum of component values (closed loops and open arcs); error estimates "
                "summed linearly (conservative bound); open-arc truncation estimates reported per "
                "component in pixels.csv, not added"
            ),
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
                "sum over integrated components (closed loops and open arcs) of the Haar-converted partial integral "
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
            "pixels_with_arcs": int(((arrays.status & STATUS_HAS_ARC) != 0).sum()),
            "quadrature_unavailable_pixels": int(((arrays.status & STATUS_QUADRATURE_UNAVAILABLE) != 0).sum()),
            "event_totals": {name: int(sum(r.events.get(name, 0) for r in results)) for name in EVENT_NAMES},
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
