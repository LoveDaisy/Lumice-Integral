"""Export the canonical ch06 pixel 3-5 fiber with its factors and line integral.

Writes ``lumice-integral.figure-data/v2`` metadata plus arrays, including one
``weight_<name>`` array per available factor (``rho_pose``, ``entry_measure``,
``fresnel_transmission``, ``path_validity``), the pointwise ``integrand`` array
and the ``result.quadrature`` block of the adaptive line quadrature; the
remaining contract factors are exported as explicitly unavailable.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from lumice_integral.canonical_scene import canonical_fixture_metadata, canonical_pixel_problem
from lumice_integral.continuation import trace_fiber
from lumice_integral.figure_data import export_fiber_figure_data
from lumice_integral.quadrature import integrate_fiber


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="output directory")
    args = parser.parse_args()

    problem = canonical_pixel_problem()
    result = trace_fiber(problem)
    quadrature = integrate_fiber(problem, result)
    files = export_fiber_figure_data(
        result,
        args.output,
        fixture=canonical_fixture_metadata(),
        provenance={
            "specification": "docs/ch06-reference-fixture.md",
            "classification": "canonical-new",
            "scene_binding": "section 3.3 canonical validation scene",
        },
        quadrature=quadrature,
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
    print(
        f"quadrature: status={quadrature.status} value={quadrature.value:.12g} "
        f"error_estimate={quadrature.error_estimate:.3g} epsilon={quadrature.epsilon:g} "
        f"refinements={quadrature.refinements} node_count={quadrature.node_count} "
        f"maximum_depth_reached={quadrature.maximum_depth_reached} "
        f"convergence_order_estimate={quadrature.convergence_order_estimate} "
        f"median_edge_convergence_order={quadrature.median_edge_convergence_order} "
        f"low_order_edges={list(quadrature.low_order_edges)} "
        f"depth_exhausted_edges={list(quadrature.depth_exhausted_edges)} "
        f"refinement_failures={len(quadrature.refinement_failures)}"
    )
    print(f"quadrature method: {quadrature.method}")


if __name__ == "__main__":
    main()
