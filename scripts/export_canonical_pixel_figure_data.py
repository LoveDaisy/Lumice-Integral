"""Export the canonical ch06 pixel 3-5 fiber with its named physical factors.

Writes ``lumice-integral.figure-data/v2`` metadata plus arrays, including one
``weight_<name>`` array per available factor (``rho_pose``, ``entry_measure``,
``fresnel_transmission``, ``path_validity``); the remaining contract factors
are exported as explicitly unavailable.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from lumice_integral.canonical_scene import canonical_fixture_metadata, canonical_pixel_problem
from lumice_integral.continuation import trace_fiber
from lumice_integral.figure_data import export_fiber_figure_data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="output directory")
    args = parser.parse_args()

    result = trace_fiber(canonical_pixel_problem())
    files = export_fiber_figure_data(
        result,
        args.output,
        fixture=canonical_fixture_metadata(),
        provenance={
            "specification": "docs/ch06-reference-fixture.md",
            "classification": "canonical-new",
            "scene_binding": "section 3.3 canonical validation scene",
        },
    )
    print(files.metadata)
    print(files.arrays)
    print(
        f"status={result.status.value} reason={result.reason.value} "
        f"samples={len(result.poses)} length={float(result.arclength_increments.sum()):.6f} rad"
    )
    for name, observable in result.weight_observables.items():
        if observable.status == "available":
            values = observable.values
            print(f"{name}: available [{values.min():.6g}, {values.max():.6g}] {observable.unit}")
        else:
            print(f"{name}: unavailable")


if __name__ == "__main__":
    main()
