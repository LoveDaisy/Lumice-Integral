"""Locate a strip pixel where the reflecting member ``3-1-2-5`` carries weight (AC4 fixture).

``3-1-2-5`` (entry face 3, total internal reflections at both basal faces,
exit face 5) has the fold matrix ``M = S_2 S_1 = I`` and the 60-deg wedge of
``3-5``: the *same direction map*, so wherever its finite-crystal footprint is
non-empty its fiber is a piece of the ``3-5`` fiber of the same target, with
a different weight (a footprint near the top edge of face 3 that climbs to
face 1, drops to face 2 and leaves through face 5).  This script finds where
that happens in the canonical ch06 scene:

1. Draw the ``3-1-2-5`` domain-valid Haar samples
   (``path_class.haar_domain_samples``, ``--landing-samples`` poses,
   ``--landing-seed``) and evaluate ``geometry.entry_measure_batch`` and the
   canonical column density on them.
2. Keep the poses whose footprint is positive and whose outgoing direction
   lies inside the canonical strip window; rank the pixels they land in by
   ``rho_pose * entry_measure`` and take the best one (``--rank`` picks
   another).
3. Render that pixel with the single-path pipeline for both ``3-5`` and
   ``3-1-2-5`` (:func:`lumice_integral.strip_pixel.build_strip_scene` /
   :func:`render_pixel`) and verify on the accepted poses of every
   ``3-1-2-5`` component that :func:`lumice_integral.optics.path_3_5` sends
   the pose to the same target (direction maps agree) while the integrated
   values differ (weights differ).  Print the constants to freeze in
   ``tests/test_path_class_ac4_reflection.py``.

Run::

    uv run python scripts/discover_3_1_2_5_seed.py
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from typing import Any

import jax.numpy as jnp
import numpy as np

from lumice_integral.camera import project_linear
from lumice_integral.canonical_scene import (
    CANONICAL_REFRACTIVE_INDEX,
    CANONICAL_RENDER,
    canonical_crystal,
    canonical_incident_direction,
    canonical_pose_density,
    canonical_sun_direction,
)
from lumice_integral.geometry import entry_measure_batch
from lumice_integral.optics import PATH_3_5_FACES, path_3_5
from lumice_integral.path_class import haar_domain_samples
from lumice_integral.so3 import log as so3_log
from lumice_integral.strip_pixel import PixelOptions, build_strip_scene, pixel_target, render_pixel

REFLECTING = (3, 1, 2, 5)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--landing-samples", type=int, default=400_000)
    parser.add_argument("--landing-seed", type=int, default=20260916)
    parser.add_argument("--rank", type=int, default=0, help="which candidate pixel to render (0 = best)")
    args = parser.parse_args(argv)

    incident = canonical_incident_direction()
    crystal = canonical_crystal()
    density = canonical_pose_density()
    rotations, directions = haar_domain_samples(
        REFLECTING, incident, CANONICAL_REFRACTIVE_INDEX, sample_count=args.landing_samples, rng_seed=args.landing_seed
    )
    footprint = entry_measure_batch(rotations, REFLECTING, incident, crystal, n_ice=CANONICAL_REFRACTIVE_INDEX)
    positive = np.flatnonzero(footprint > 0.0)
    weight = density.evaluate_batch(rotations[positive]) * footprint[positive]
    per_pixel: dict[tuple[int, int], float] = defaultdict(float)
    counts: dict[tuple[int, int], int] = defaultdict(int)
    for index, w in zip(positive, weight):
        try:
            u, v = project_linear(-directions[index], **CANONICAL_RENDER)
        except ValueError:
            continue
        if 0.0 <= u < CANONICAL_RENDER["width"] and 0.0 <= v < CANONICAL_RENDER["height"]:
            per_pixel[(int(v), int(u))] += float(w)
            counts[(int(v), int(u))] += 1
    ranked = sorted(per_pixel.items(), key=lambda item: -item[1])
    if not ranked:
        raise SystemExit("no 3-1-2-5 footprint lands inside the strip window")
    (row, column), score = ranked[args.rank]

    options = PixelOptions()
    kwargs = dict(
        sun_direction=canonical_sun_direction(),
        refractive_index=CANONICAL_REFRACTIVE_INDEX,
        crystal=crystal,
        pose_density=density,
        render=CANONICAL_RENDER,
    )
    results = {
        "3-5": render_pixel(build_strip_scene(PATH_3_5_FACES, **kwargs), row, column, options),
        "3-1-2-5": render_pixel(build_strip_scene(REFLECTING, **kwargs), row, column, options),
    }
    target = pixel_target(CANONICAL_RENDER, row, column)
    direction_map_residuals = []
    for component in results["3-1-2-5"].components:
        # The seed pose through the *3-5* map must hit the same target: same Phi, same direction map.
        direction_map_residuals.append(
            float(np.linalg.norm(np.asarray(path_3_5(jnp.asarray(component.seed), jnp.asarray(incident)).direction) - target))
        )
    out: dict[str, Any] = {
        "script": "scripts/discover_3_1_2_5_seed.py",
        "command": "uv run python scripts/discover_3_1_2_5_seed.py" + (f" --rank {args.rank}" if args.rank else ""),
        "landing_samples": {"sample_count": args.landing_samples, "rng_seed": args.landing_seed},
        "valid_count": int(len(rotations)),
        "footprint_positive_count": int(positive.size),
        "candidate_pixels": [{"row": r, "column": c, "score": s, "samples": counts[(r, c)]} for (r, c), s in ranked[:8]],
        "selection": f"rank {args.rank} by summed rho_pose * entry_measure of the landing Haar samples",
        "pixel": {"row": row, "column": column, "score": score},
        "members": {
            path_id: {
                "value": result.value,
                "error_estimate": result.error_estimate,
                "completeness": result.completeness,
                "components": [
                    {
                        "kind": component.kind,
                        "arclength": component.arclength,
                        "value": component.value,
                        "reason": component.reason,
                        "start_reason": component.start_reason,
                        "seed_exponential_coordinates": np.asarray(so3_log(jnp.asarray(component.seed))).tolist(),
                    }
                    for component in result.components
                ],
            }
            for path_id, result in results.items()
        },
        "direction_map_residuals_of_3_1_2_5_seeds_through_3_5": direction_map_residuals,
        "value_ratio_3_1_2_5_over_3_5": results["3-1-2-5"].value / results["3-5"].value if results["3-5"].value else None,
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
