"""One-off coarse prescan that finds a 3-5 fiber seed for one canonical pixel.

This is a discovery tool, not a component-discovery algorithm and not part of
the package API.  Its *output* (pixel, seed exponential coordinates, target
direction) is frozen into ``lumice_integral.canonical_scene``; the scan
parameters used for that run are recorded there and in
``docs/ch06-reference-fixture.md``.

Procedure for one pixel ``(row, column)`` of the canonical linear render:

1. ``d`` = pixel-centre outgoing direction (``lumice_integral.camera``).
2. Draw ``samples`` Haar-uniform rotations (random unit quaternions), evaluate
   the smooth 3-5 map with ``jax.vmap(path_3_5)`` and keep poses whose four
   branch margins are positive and whose outgoing direction lies within
   ``angle_tolerance_deg`` of ``d``.
3. Gauss-Newton-correct the closest candidates on the solver's own
   target-chart residual (minimum-norm update in right-trivialized
   coordinates) until the residual is below the solver seed tolerance.
4. Keep corrected seeds with ``path_3_5_domain(...).valid`` and
   ``entry_measure > 0``, deduplicate by rotation distance, order the
   survivors by ``|c-axis zenith - 90 deg|`` (closest to the pose-density
   peak first), trace the fiber from every survivor and print seed
   coordinates plus trace evidence.  The frozen fixture takes the first
   survivor of that ordering.

Example::

    uv run python scripts/discover_canonical_pixel_seed.py --row 150 --column 200
"""

from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp
import numpy as np

from lumice_integral.camera import linear_pixel_outgoing_direction, sun_incident_direction
from lumice_integral.continuation import (
    ContinuationOptions,
    local_residual_jacobian,
    target_residual,
    trace_fiber,
)
from lumice_integral.geometry import HexPrism, entry_measure
from lumice_integral.optics import path_3_5, path_3_5_domain, path_3_5_problem
from lumice_integral.so3 import exp, rotation_distance

CANONICAL_RENDER = {
    "width": 251,
    "height": 801,
    "fov_deg": 6.0,
    "view": {"azimuth": 0.0, "elevation": -15.0},
}
SUN_ALTITUDE_DEG = 15.0
REFRACTIVE_INDEX = 1.31
HEIGHT_RATIO = 1.0


def haar_rotations(count: int, rng: np.random.Generator) -> np.ndarray:
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


def log_map(rotation: np.ndarray) -> np.ndarray:
    """Rotation matrix -> rotation vector (principal angle in [0, pi))."""
    skew = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ]
    ) / 2.0
    sine = np.linalg.norm(skew)
    cosine = (np.trace(rotation) - 1.0) / 2.0
    angle = np.arctan2(sine, cosine)
    if sine < 1e-12:
        return np.zeros(3)
    return skew / sine * angle


def newton_correct(problem, rotation: jnp.ndarray, tolerance: float, iterations: int = 30):
    for _ in range(iterations):
        residual = target_residual(problem, rotation)
        norm = float(jnp.linalg.norm(residual))
        if norm <= tolerance:
            return rotation, norm
        jacobian = local_residual_jacobian(problem, rotation)
        delta = -jacobian.T @ jnp.linalg.solve(jacobian @ jacobian.T, residual)
        rotation = rotation @ exp(delta)
    return rotation, float(jnp.linalg.norm(target_residual(problem, rotation)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--row", type=int, required=True)
    parser.add_argument("--column", type=int, required=True)
    parser.add_argument("--samples", type=int, default=400_000)
    parser.add_argument("--angle-tolerance-deg", type=float, default=2.0)
    parser.add_argument("--candidates", type=int, default=12)
    parser.add_argument("--rng-seed", type=int, default=20260916)
    args = parser.parse_args()

    incident = sun_incident_direction(SUN_ALTITUDE_DEG)
    target = linear_pixel_outgoing_direction(args.row, args.column, **CANONICAL_RENDER)
    sky = -target
    print(f"pixel row={args.row} column={args.column}")
    print(f"incident s = {incident.tolist()}")
    print(f"target d   = {target.tolist()}")
    print(
        "sky elevation/azimuth = "
        f"{np.degrees(np.arcsin(sky[2])):.4f} / {np.degrees(np.arctan2(sky[1], sky[0])):.4f} deg, "
        f"deviation from sun = {np.degrees(np.arccos(np.dot(-incident, sky))):.4f} deg"
    )

    rng = np.random.default_rng(args.rng_seed)
    rotations = haar_rotations(args.samples, rng)
    evaluation = jax.vmap(lambda r: path_3_5(r, jnp.asarray(incident), jnp.asarray(REFRACTIVE_INDEX)))(
        jnp.asarray(rotations)
    )
    valid = (
        (np.asarray(evaluation.entry.incidence_cosine) > 0)
        & (np.asarray(evaluation.entry.discriminant) > 0)
        & (np.asarray(evaluation.exit.incidence_cosine) > 0)
        & (np.asarray(evaluation.exit.discriminant) > 0)
    )
    directions = np.asarray(evaluation.direction)
    alignment = np.where(valid, directions @ target, -np.inf)
    order = np.argsort(-alignment)[: args.candidates]
    print(
        f"samples={args.samples} valid={int(valid.sum())} "
        f"within {args.angle_tolerance_deg} deg: "
        f"{int((alignment >= np.cos(np.radians(args.angle_tolerance_deg))).sum())}"
    )

    problem = path_3_5_problem(
        jnp.asarray(rotations[order[0]]),
        jnp.asarray(incident),
        target_direction=jnp.asarray(target),
        refractive_index=jnp.asarray(REFRACTIVE_INDEX),
    )
    options = ContinuationOptions()
    tolerance = options.residual_tolerance + options.relative_residual_tolerance
    crystal = HexPrism.from_ratio(HEIGHT_RATIO)
    seeds: list[np.ndarray] = []
    for index in order:
        if alignment[index] < np.cos(np.radians(args.angle_tolerance_deg)):
            continue
        corrected, norm = newton_correct(problem, jnp.asarray(rotations[index]), tolerance * 1e-2)
        corrected_np = np.asarray(corrected)
        domain = path_3_5_domain(corrected_np, incident, REFRACTIVE_INDEX)
        measure = entry_measure(corrected_np, (3, 5), incident, crystal, n_ice=REFRACTIVE_INDEX)
        duplicate = any(
            float(rotation_distance(jnp.asarray(seed), corrected)) < 1e-6 for seed in seeds
        )
        print(
            f"candidate {index}: residual={norm:.3e} valid={domain.valid} "
            f"entry_measure={measure.value:.6f} ({measure.status}) duplicate={duplicate}"
        )
        if norm <= tolerance and domain.valid and measure.value > 0 and not duplicate:
            seeds.append(corrected_np)
    if not seeds:
        raise SystemExit("no admissible seed found; widen the scan or choose another pixel")

    seeds.sort(key=lambda seed: abs(np.degrees(np.arccos(abs(seed[2, 2]))) - 90.0))
    for seed in seeds:
        coordinates = log_map(seed)
        roundtrip = np.asarray(exp(jnp.asarray(coordinates)))
        seed_problem = path_3_5_problem(
            jnp.asarray(roundtrip),
            jnp.asarray(incident),
            target_direction=jnp.asarray(target),
            refractive_index=jnp.asarray(REFRACTIVE_INDEX),
        )
        result = trace_fiber(seed_problem, options)
        c_axis = roundtrip @ np.array([0.0, 0.0, 1.0])
        print("seed exponential coordinates =", repr(coordinates.tolist()))
        print(
            f"  exp round-trip residual={float(jnp.linalg.norm(target_residual(seed_problem, jnp.asarray(roundtrip)))):.3e} "
            f"c-axis zenith={np.degrees(np.arccos(abs(c_axis[2]))):.3f} deg"
        )
        print(
            f"  trace: status={result.status.value} reason={result.reason.value} "
            f"poses={len(result.poses)} length={float(result.arclength_increments.sum()):.6f} "
            f"max_residual={float(result.residual_norms.max()) if len(result.poses) else float('nan'):.2e} "
            f"seed_distance={result.closure_diagnostics.seed_distance}"
        )


if __name__ == "__main__":
    main()
