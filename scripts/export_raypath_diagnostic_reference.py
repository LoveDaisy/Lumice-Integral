"""Export the fixed LI ray-path diagnostic reference as JSON plus NPZ arrays.

The command writes ``reference.json``, ``arrays.npz`` and
``provenance.json``.  It runs only LI's independent geometry, optics,
focusing, chromatic and finite-crystal weight chains; Lumice is neither an
input nor a runtime dependency.

Usage::

    uv run python scripts/export_raypath_diagnostic_reference.py \
      --output-dir artifacts/raypath-diagnostic-reference
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from lumice_integral.provenance import git_commit, sha256_of
from lumice_integral.raypath_diagnostic_reference import (
    PLATE_COARSE_N,
    PLATE_FINE_N,
    RANDOM_LATTICE_N,
    DiagnosticReference,
    build_reference,
)

REPOSITORY = Path(__file__).resolve().parents[1]
SOURCES = (
    "src/lumice_integral/raypath_diagnostic_reference.py",
    "src/lumice_integral/chromatic.py",
    "src/lumice_integral/focusing.py",
    "src/lumice_integral/dp_field",
    "src/lumice_integral/path_weight.py",
    "src/lumice_integral/optics.py",
    "src/lumice_integral/geometry",
    "src/lumice_integral/symmetry",
    "scripts/export_raypath_diagnostic_reference.py",
)


def _last_commit(path: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "log", "-1", "--format=%H", "--", path],
            cwd=REPOSITORY,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _dirty(paths: tuple[str, ...]) -> bool | None:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain", "--", *paths],
            cwd=REPOSITORY,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return bool(result.stdout.strip())


def _array_semantics(name: str) -> str:
    if name.endswith("_rotations"):
        return "body-to-world SO(3) rotation matrices"
    if name.endswith("_u"):
        return "body-frame directions toward the sun, u = R^T s_hat"
    if "_outgoing_" in name:
        return "world propagation directions after the fixed path"
    if "_A_" in name:
        return "finite-crystal entry measure A in units of a^2"
    if "_T_" in name:
        return "dimensionless path power T, including every internal Fresnel R"
    if "_AT_" in name:
        return "finite-crystal weighted power A*T in units of a^2"
    if name.endswith("_theta_rad"):
        return "ideal horizontal-plate spin angles in radians"
    if name.endswith("_valid_red") or name.endswith("_valid_blue"):
        return "physical smooth-branch validity mask"
    return "numerical evidence array; see docs/raypath-diagnostic-reference.md"


def write_reference(reference: DiagnosticReference, output_dir: Path, provenance: dict[str, Any]) -> dict[str, Path]:
    """Write one assembled reference and return its three output paths."""

    output_dir.mkdir(parents=True, exist_ok=True)
    arrays_path = output_dir / "arrays.npz"
    np.savez_compressed(arrays_path, **reference.arrays)
    array_index = {
        name: {
            "shape": list(np.asarray(value).shape),
            "dtype": str(np.asarray(value).dtype),
            "semantics": _array_semantics(name),
        }
        for name, value in sorted(reference.arrays.items())
    }
    payload = {**reference.metadata, "array_store": {"file": arrays_path.name, "arrays": array_index}}
    reference_path = output_dir / "reference.json"
    reference_path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    provenance_path = output_dir / "provenance.json"
    provenance_payload = {
        **provenance,
        "files": {
            "reference.json": {"sha256": sha256_of(reference_path)},
            "arrays.npz": {"sha256": sha256_of(arrays_path)},
        },
    }
    provenance_path.write_text(
        json.dumps(provenance_payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {"reference": reference_path, "arrays": arrays_path, "provenance": provenance_path}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--random-lattice-n", type=int, default=RANDOM_LATTICE_N)
    parser.add_argument("--plate-coarse-n", type=int, default=PLATE_COARSE_N)
    parser.add_argument("--plate-fine-n", type=int, default=PLATE_FINE_N)
    parser.add_argument("--auxiliary-samples", type=int, default=4096)
    args = parser.parse_args(argv)

    command = " ".join([Path(sys.argv[0]).name, *(argv if argv is not None else sys.argv[1:])])
    start = time.perf_counter()
    reference = build_reference(
        random_lattice_n=args.random_lattice_n,
        plate_coarse_n=args.plate_coarse_n,
        plate_fine_n=args.plate_fine_n,
        auxiliary_samples=args.auxiliary_samples,
    )
    elapsed = time.perf_counter() - start
    provenance = {
        "schema": "lumice-integral.raypath-diagnostic-reference-provenance/v1",
        "repository_commit": git_commit(REPOSITORY),
        "source_commits": {path: _last_commit(path) for path in SOURCES},
        "sources_modified_in_worktree": _dirty(SOURCES),
        "command": command,
        "parameters": {
            "random_lattice_n": args.random_lattice_n,
            "plate_coarse_n": args.plate_coarse_n,
            "plate_fine_n": args.plate_fine_n,
            "auxiliary_samples": args.auxiliary_samples,
        },
        "wall_clock_s": elapsed,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    files = write_reference(reference, args.output_dir, provenance)
    print(f"wrote {files['reference']} and {files['arrays']} ({elapsed:.2f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
