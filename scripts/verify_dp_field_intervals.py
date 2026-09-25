"""Independent check of ``DPField.interval_partition``: level-set counts from a dense grid through the pose chain.

For every interval of the partition, at its midpoint ``delta``, the counts are
recomputed from scratch on a dense grid, sharing nothing with the field layer
but the gates' authority (:func:`.optics.path_domain_batch`):

- the grid is the orthographic chart of the entry hemisphere
  (``U_P`` lies in ``u . n_a > 0``), ``u = x e1 + y e2 + sqrt(1 - x^2 - y^2) n_a``;
- validity and ``D`` come from :func:`.s2_store.evaluate_fields` (the event
  store's production evaluator: poses ``R u = s_hat``, ``arccos`` deviation);
- ``n_closed``: a closed level loop in a disk bounds exactly one region of
  ``U_P`` minus ``{D = delta}`` that touches no boundary, so ``n_closed`` is the
  number of 4-connected components of ``{D < delta}`` and ``{D > delta}``
  (the sublevel-component method of explore ``dp-field-topology`` run #5,
  on a grid) with no node on the edge of the valid mask;
- ``n_open``: half the number of crossings of ``delta`` by ``D`` along the
  edge of the valid mask, traced in order (Moore neighbourhood), each edge
  node moved onto ``dU_P`` by bisection towards an outside neighbour.  Node
  values themselves would not do: ``D`` falls like the square root of the
  distance to an exit-TIR boundary, so the thin superlevel band along it is
  below the grid and would be counted as many fragments.

Run ``uv run python scripts/verify_dp_field_intervals.py`` (the five fixture
paths, ``--grid`` points per side); exit status 1 on any mismatch.
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import NamedTuple, Sequence

import numpy as np
from scipy import ndimage

from lumice_integral import optics
from lumice_integral.canonical_scene import canonical_crystal
from lumice_integral.dp_field import DPField
from lumice_integral.s2_store import align_rotations, evaluate_fields

FIXTURES: tuple[tuple[int, ...], ...] = ((3, 5), (1, 3), (3, 1, 6), (1, 3, 2), (3, 5, 6, 7, 3))
SUN = np.array([0.0, 0.0, 1.0])
CHUNK = 200_000


class Grid(NamedTuple):
    """The chart: unit vectors at every node (off-chart nodes pushed onto the rim ``u . n_a = 0``), validity and ``D``."""

    u: np.ndarray
    valid: np.ndarray
    deviation: np.ndarray


def _evaluate(u: np.ndarray, faces: tuple[int, ...], index: float) -> tuple[np.ndarray, np.ndarray]:
    crystal = canonical_crystal()
    valid = np.zeros(len(u), dtype=bool)
    deviation = np.full(len(u), np.nan)
    for start in range(0, len(u), CHUNK):
        chunk = u[start : start + CHUNK]
        fields = evaluate_fields(align_rotations(chunk, SUN), SUN, crystal, index, [faces])
        valid[start : start + CHUNK] = fields["valid"]
        deviation[start : start + CHUNK] = fields["D"]
    return valid, deviation


def grid_field(faces: tuple[int, ...], index: float, grid: int) -> Grid:
    """The ``grid x grid`` orthographic chart of the entry hemisphere (module docstring)."""
    n_a = np.asarray(optics.HEXPRISM_BODY_NORMALS[faces[0]])
    e1 = np.cross(n_a, np.eye(3)[int(np.argmin(np.abs(n_a)))])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n_a, e1)
    xs = np.linspace(-1.0, 1.0, grid)
    x, y = np.meshgrid(xs, xs, indexing="ij")
    r = np.sqrt(x**2 + y**2)
    on_chart = r < 1.0
    scale = np.where(on_chart, 1.0, 1.0 / np.maximum(r, 1e-300))
    height = np.sqrt(np.clip(1.0 - r**2, 0.0, None))
    u = (x * scale)[..., None] * e1 + (y * scale)[..., None] * e2 + height[..., None] * n_a
    u /= np.linalg.norm(u, axis=-1, keepdims=True)
    valid = np.zeros((grid, grid), dtype=bool)
    deviation = np.full((grid, grid), np.nan)
    valid[on_chart], deviation[on_chart] = _evaluate(u[on_chart], faces, index)
    return Grid(u, valid, deviation)


_MOORE = ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1))


def trace_edge(valid: np.ndarray) -> list[tuple[int, int]]:
    """The outer edge of the (single) valid region in order: Moore-neighbour tracing, Jacob's stopping criterion."""
    starts = np.argwhere(valid)
    start = (int(starts[0][0]), int(starts[0][1]))
    path = [start]
    current, came_from = start, 0  # entered from the north (raster order: the node above is outside)
    for _ in range(4 * valid.size):
        for step in range(8):
            direction = (came_from + step) % 8
            di, dj = _MOORE[direction]
            ni, nj = current[0] + di, current[1] + dj
            if 0 <= ni < valid.shape[0] and 0 <= nj < valid.shape[1] and valid[ni, nj]:
                # next search starts from the outside neighbour just before the hit, seen from the new node
                came_from = (direction + 5) % 8 if direction % 2 == 0 else (direction + 6) % 8
                current = (ni, nj)
                break
        else:
            return path  # an isolated node
        if current == start and len(path) > 1:
            return path
        path.append(current)
    raise RuntimeError("edge tracing did not close")


def edge_values(grid: Grid, faces: tuple[int, ...], index: float, iterations: int = 60) -> np.ndarray:
    """``D`` on ``dU_P`` next to every traced edge node, in tracing order (bisection towards an outside neighbour)."""
    nodes = trace_edge(grid.valid)
    inside, outside = [], []
    n = grid.valid.shape[0]
    for i, j in nodes:
        for di, dj in _MOORE:
            ni, nj = min(max(i + di, 0), n - 1), min(max(j + dj, 0), n - 1)
            if not grid.valid[ni, nj]:
                inside.append(grid.u[i, j])
                outside.append(grid.u[ni, nj])
                break
    inside, outside = np.array(inside), np.array(outside)
    for _ in range(iterations):
        middle = inside + outside
        middle /= np.linalg.norm(middle, axis=1, keepdims=True)
        valid, _ = _evaluate(middle, faces, index)
        inside[valid] = middle[valid]
        outside[~valid] = middle[~valid]
    return _evaluate(inside, faces, index)[1]


def level_counts(grid: Grid, edge: np.ndarray, delta: float) -> tuple[int, int, int]:
    """``(n_components, n_closed, n_open)`` of ``{D = delta}`` from the grid (module docstring)."""
    valid, deviation = grid.valid, grid.deviation
    rim = valid & ~ndimage.binary_erosion(valid, structure=ndimage.generate_binary_structure(2, 1), border_value=0)
    n_closed = 0
    for mask in (valid & (deviation < delta), valid & (deviation > delta)):
        labels, count = ndimage.label(mask)
        touching = set(np.unique(labels[rim & mask]).tolist())
        n_closed += sum(1 for label in range(1, count + 1) if label not in touching)
    above = edge > delta
    crossings = int(np.count_nonzero(above != np.roll(above, 1)))
    n_open = crossings // 2
    return n_closed + n_open, n_closed, n_open


def verify(faces: Sequence[int], index: float, grid: int, lattice_n: int = 20000) -> list[tuple]:
    """One row per interval: ``(lower_deg, upper_deg, predicted (n, closed, open), grid (n, closed, open))``.

    ``lattice_n`` is the field's lattice (:meth:`.dp_field.DPField.build`): ``3-5-6-7`` needs 50000, its
    ``U_P`` has a neck that 20000 points split into two components.
    """
    field = DPField.build(canonical_crystal(), faces, index, lattice_n=lattice_n)
    chart = grid_field(field.faces, index, grid)
    edge = edge_values(chart, field.faces, index)
    rows = []
    for interval in field.interval_partition():
        delta = 0.5 * (interval.lower + interval.upper)
        predicted = (int(interval.n_components), int(interval.n_closed), int(interval.n_open))
        rows.append((np.degrees(interval.lower), np.degrees(interval.upper), predicted, level_counts(chart, edge, delta)))
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--grid", type=int, default=1601, help="grid points per side of the chart")
    parser.add_argument("--refractive-index", type=float, default=float(optics.ICE_REFRACTIVE_INDEX))
    parser.add_argument("--path", type=int, nargs="+", action="append", help="face sequence (repeatable); default: the fixtures")
    parser.add_argument("--lattice-n", type=int, default=20000, help="Fibonacci lattice of the field layer (DPField.build)")
    args = parser.parse_args(argv)
    paths = [tuple(p) for p in args.path] if args.path else list(FIXTURES)
    failures = 0
    for faces in paths:
        start = time.perf_counter()
        rows = verify(faces, args.refractive_index, args.grid, args.lattice_n)
        print(f"{optics.path_id_of(faces)}  ({time.perf_counter() - start:.1f} s, grid {args.grid})")
        for lower, upper, predicted, measured in rows:
            ok = predicted == measured
            failures += not ok
            print(f"  ({lower:10.5f}, {upper:10.5f}) deg  predicted {predicted}  grid {measured}  {'ok' if ok else 'MISMATCH'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
