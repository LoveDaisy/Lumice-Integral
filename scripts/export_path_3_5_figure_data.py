"""Export the canonical 3-5 single-fiber figure-data fixture."""

from __future__ import annotations

import argparse
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from lumice_integral.continuation import trace_fiber
from lumice_integral.figure_data import export_fiber_figure_data
from lumice_integral.optics import (
    ICE_REFRACTIVE_INDEX,
    minimum_deviation_incident,
    path_3_5,
    path_3_5_problem,
)
from lumice_integral.so3 import exp


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="output directory")
    args = parser.parse_args()

    seed_vector = np.array([0.15, 0.08, -0.05], dtype=np.float64)
    seed = exp(jnp.asarray(seed_vector))
    incident = minimum_deviation_incident()
    target = path_3_5(seed, incident).direction
    result = trace_fiber(path_3_5_problem(seed, incident, target_direction=target))
    files = export_fiber_figure_data(
        result,
        args.output,
        fixture={
            "name": "canonical-path-3-5-single-fiber",
            "path": [3, 5],
            "refractive_index": float(ICE_REFRACTIVE_INDEX),
            "incident_direction": np.asarray(incident),
            "target_direction": np.asarray(target),
            "seed_exponential_coordinates": seed_vector,
        },
        provenance={
            "specification": "docs/ch06-reference-fixture.md",
            "classification": "canonical-new",
        },
    )
    print(files.metadata)
    print(files.arrays)


if __name__ == "__main__":
    main()
