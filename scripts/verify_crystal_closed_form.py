"""Verify the closed-form crystals (``geometry.closed_form``) against Lumice's own mesh and by independent geometry.

Two chains, run together whenever both are available and reported side by side:

1. **Lumice oracle** (validation boundary only; nothing here is imported by the package): the shared library's
   ``LUMICE_GetCrystalMesh`` is called through ``ctypes`` with the same shape (prism ``height`` /
   ``face_distance``; pyramid ``prism_h`` / ``upper_h`` / ``lower_h`` / the two wedge angles / ``face_distance``),
   and the present face numbers, face normals and per-face vertex sets are compared with Lumice Integral's
   crystal (``HexPrism.from_lumice`` / ``Pyramid.from_lumice``).  Lumice builds its mesh at circumscribed
   diameter 1, i.e. ``a = 1/2``.  Lumice computes in float32, so the comparison tolerance is ``1e-5`` of the
   crystal size.  A shape Lumice rejects comes back as an empty mesh; Lumice Integral must raise ``ValueError``
   for exactly those.  The C API takes wedge angles only, so a pyramid given by Miller indices is converted
   here, independently of ``geometry.pyramid`` (``doc/configuration.md`` §11: ``wedge = atan((√3/2)·(l/h) / c)``,
   ``c = 1.629``); that crystal is also built from the indices by ``Pyramid.from_lumice`` and must agree with
   the one built from the converted angle.
2. **Independent geometry** (always): every vertex lies in every face's half-space (tolerance ``1e-12`` of the
   size, scaled up by a face's aspect: its plane comes from its own corners), every face is a polygon of at least
   three distinct corners, every edge has two faces, and ``V - E + F = 2``.

A Lumice disagreement on a crystal with an edge shorter than ``RULER_BAND`` times its size is reported apart, as
*in the ruler band*, and listed but not failed: Lumice resolves corners and events with ``5e-5``-scale rulers
(``GapToleranceForScale``, ``ApexCollapsedAt``) where this project decides presence at ``1e-9``
(``closed_form.PRESENT_REL_TOL``), so a sliver face, an event within ~1e-4 of a truncation or apex, or a cone under
~1e-5 thick is resolved differently by construction (measured on ``--random 10000``: 13 of 20842 cases, every one
with an edge under 2.2e-4, 2 of them prisms).  The independent chain is never excused.

The library is taken from ``--lumice-lib`` or ``$LUMICE_LIB`` (default: the Lumice checkout's
``build/cmake_install/shared/lib/liblumice.dylib``); when it cannot be loaded the script says so and runs the
second chain alone.  The exit status is non-zero iff a check that ran failed.

    uv run python scripts/verify_crystal_closed_form.py [--lumice-lib PATH] [--random 300] [--output out.json]
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
from pathlib import Path

import numpy as np

from lumice_integral.geometry import HexPrism, Polyhedron, Pyramid

DEFAULT_LIB = Path.home() / "Codes/Ice Halo Simulation/build/cmake_install/shared/lib/liblumice.dylib"
LUMICE_A = 0.5          # Lumice meshes have circumscribed diameter 1
TOL = 1e-5              # float32 mesh against float64 construction, relative to the crystal size
RULER_BAND = 3e-4       # shortest edge / crystal size below which the two projects' rulers decide differently


# ---- Lumice C API (src/include/lumice.h, LUMICE_CrystalParam / LUMICE_CrystalMesh) -------------------------

class _Distribution(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("center", ctypes.c_float), ("spread", ctypes.c_float)]


class _CrystalParam(ctypes.Structure):
    _fields_ = [
        ("id", ctypes.c_int), ("type", ctypes.c_int),
        ("height", _Distribution), ("prism_h", _Distribution), ("upper_h", _Distribution), ("lower_h", _Distribution),
        ("upper_wedge_angle", ctypes.c_float), ("lower_wedge_angle", ctypes.c_float),
        ("face_distance", _Distribution * 6),
        ("zenith", _Distribution), ("azimuth", _Distribution), ("roll", _Distribution),
        ("sync_group", ctypes.c_int * 10),
    ]


class _CrystalMesh(ctypes.Structure):
    _fields_ = [
        ("vertices", ctypes.c_float * (128 * 3)), ("vertex_count", ctypes.c_int),
        ("edges", ctypes.c_int * (256 * 2)), ("edge_count", ctypes.c_int),
        ("triangles", ctypes.c_int * (128 * 3)), ("triangle_count", ctypes.c_int),
        ("edge_face_normals", ctypes.c_float * (256 * 6)),
        ("face_numbers", ctypes.c_int * 128),
        ("face_count", ctypes.c_int),
        ("face_numbers_by_face", ctypes.c_int * 24), ("face_vtx_offsets", ctypes.c_int * 24),
        ("face_vtx_counts", ctypes.c_int * 24), ("face_vtx_pool", ctypes.c_int * 192),
        ("face_normals", ctypes.c_float * (24 * 3)),
    ]


class LumiceMesher:
    def __init__(self, path: Path):
        self.lib = ctypes.CDLL(str(path))
        self.lib.LUMICE_GetCrystalMesh.restype = ctypes.c_int

    def mesh(self, shape: dict) -> dict[int, tuple[np.ndarray, np.ndarray]] | None:
        """``{face number: (unit normal, vertices (k, 3))}`` of Lumice's mesh; ``None`` for a rejected shape."""
        param = _CrystalParam()
        if shape["type"] == "prism":
            param.type = 0
            param.height.center = shape["height"]
        else:
            param.type = 1
            param.prism_h.center = shape["prism_h"]
            param.upper_h.center = shape["upper_h"]
            param.lower_h.center = shape["lower_h"]
            param.upper_wedge_angle = shape["upper_wedge_deg"]
            param.lower_wedge_angle = shape["lower_wedge_deg"]
        for i, d in enumerate(shape["face_distance"]):
            param.face_distance[i].center = d
        out = _CrystalMesh()
        code = self.lib.LUMICE_GetCrystalMesh(ctypes.byref(param), ctypes.c_ulonglong(0), ctypes.byref(out))
        if code != 0:
            raise RuntimeError(f"LUMICE_GetCrystalMesh returned {code} for {shape}")
        if out.vertex_count == 0:
            return None
        vertices = np.array(out.vertices[: 3 * out.vertex_count], dtype=float).reshape(-1, 3)
        faces = {}
        for k in range(out.face_count):
            start, count = out.face_vtx_offsets[k], out.face_vtx_counts[k]
            ids = list(out.face_vtx_pool[start:start + count])
            faces[out.face_numbers_by_face[k]] = (np.array(out.face_normals[3 * k:3 * k + 3], dtype=float), vertices[ids])
        return faces


# ---- Lumice Integral side -----------------------------------------------------------------------------------

LUMICE_ICE_C = 1.629    # Lumice kIceCrystalC (src/core/geo3d.hpp), the c/a its Miller-index conversion uses


def _f32(x: float) -> float:
    return float(np.float32(x))   # the float32 value Lumice receives


def miller_wedge_deg(indices: tuple[int, int, int]) -> float:
    """Wedge angle of a legal ``(h, 0, l)`` with ``h > 0`` (``doc/configuration.md`` §11), independent of the package."""
    h, _, l = indices
    return float(np.degrees(np.arctan(np.sqrt(3.0) / 2.0 * l / h / LUMICE_ICE_C)))


def build(shape: dict) -> Polyhedron:
    """The Lumice shape in Lumice Integral (raises ``ValueError`` for a rejected shape)."""
    fd = tuple(_f32(d) for d in shape["face_distance"])
    if shape["type"] == "prism":
        return HexPrism.from_lumice(_f32(shape["height"]), fd, a=LUMICE_A)
    return Pyramid.from_lumice(_f32(shape["prism_h"]), _f32(shape["upper_h"]), _f32(shape["lower_h"]),
                               face_distance=fd, a=LUMICE_A, upper_wedge_deg=_f32(shape["upper_wedge_deg"]),
                               lower_wedge_deg=_f32(shape["lower_wedge_deg"]))


def miller_check(shape: dict, crystal: Polyhedron | None) -> list[str]:
    """For a shape given by Miller indices: the crystal built from the indices equals the one built from the
    angle this script converted them to (the package's conversion against this script's)."""
    if "upper_indices" not in shape:
        return []
    fd = tuple(_f32(d) for d in shape["face_distance"])
    try:
        ours = Pyramid.from_lumice(_f32(shape["prism_h"]), _f32(shape["upper_h"]), _f32(shape["lower_h"]),
                                   shape["upper_indices"], shape["lower_indices"], fd, a=LUMICE_A)
    except ValueError:
        ours = None
    if ours is None or crystal is None:
        return [] if ours is None and crystal is None else ["Miller-index and wedge-angle builds disagree on rejection"]
    if sorted(f.number for f in ours.faces) != sorted(f.number for f in crystal.faces):
        return ["Miller-index and wedge-angle builds have different faces"]
    reference = {f.number: (crystal.normal(f), crystal.face_vertices(f)) for f in crystal.faces}
    return [f"Miller-index build: {problem}" for problem in lumice_check(ours, reference)]


def shortest_edge(crystal: Polyhedron) -> float:
    """The shortest edge relative to the crystal size (the largest vertex radius)."""
    scale = float(np.max(np.linalg.norm(crystal.vertices, axis=1)))
    return min(float(np.linalg.norm(crystal.vertices[a] - crystal.vertices[b])) for a, b in crystal.edges) / scale


def independent_check(crystal: Polyhedron) -> list[str]:
    problems = []
    scale = float(np.max(np.linalg.norm(crystal.vertices, axis=1)))
    v, e, f = len(crystal.vertices), len(crystal.edges), len(crystal.faces)
    if v - e + f != 2:
        problems.append(f"Euler V-E+F = {v - e + f}")
    for face in crystal.faces:
        pts = crystal.face_vertices(face)
        gaps = np.linalg.norm(pts - np.roll(pts, -1, axis=0), axis=1)
        if len(pts) < 3 or gaps.min() <= 1e-9 * scale:
            problems.append(f"face {face.number} has a degenerate outline")
        # every vertex inside this face's plane; the plane is taken from the vertices (Newell), good to ~eps·size/width,
        # so a sliver's plane is that much less exact: the tolerance scales with its aspect, never below 1e-12 of the size
        tolerance = 1e-12 * scale * max(1.0, scale / float(gaps.min()))
        if np.max((crystal.vertices - pts[0]) @ crystal.normal(face)) > tolerance:
            problems.append(f"a corner lies outside the half-space of face {face.number}")
    if not all(len(crystal.edge_faces(edge)) == 2 for edge in crystal.edges):
        problems.append("an edge does not have exactly two faces")
    return problems


def _same_point_set(a: np.ndarray, b: np.ndarray, tol: float) -> bool:
    if len(a) != len(b):
        return False
    distance = np.linalg.norm(a[:, None] - b[None], axis=-1)
    return bool(distance.min(axis=1).max() <= tol and distance.min(axis=0).max() <= tol)


def lumice_check(crystal: Polyhedron | None, reference: dict | None) -> list[str]:
    if crystal is None or reference is None:
        return [] if crystal is None and reference is None else [
            f"rejection disagrees: Lumice Integral {'rejects' if crystal is None else 'accepts'}, "
            f"Lumice {'rejects' if reference is None else 'accepts'}"]
    problems = []
    ours = {f.number: f for f in crystal.faces}
    if set(ours) != set(reference):
        return [f"present faces differ: ours {sorted(ours)}, Lumice {sorted(reference)}"]
    scale = float(np.max(np.linalg.norm(crystal.vertices, axis=1)))
    for number, (normal, vertices) in reference.items():
        face = ours[number]
        if np.linalg.norm(crystal.normal(face) - normal) > TOL:
            problems.append(f"face {number} normal differs")
        if not _same_point_set(crystal.face_vertices(face), vertices, TOL * scale):
            problems.append(f"face {number} corners differ")
    return problems


def cases(n_random: int) -> list[dict]:
    prism = [
        (1, 1, 1, 1, 1, 1), (1, 1.2, 1, 1.2, 1, 1.2), (1, 1.9, 1, 1.9, 1, 1.9), (1, 2, 1, 2, 1, 2),
        (1, 2.5, 1, 2.5, 1, 2.5), (1, 0.5, 1, 0.5, 1, 0.5), (1.9, 1, 1, 1.9, 1, 1), (2, 1, 1, 2, 1, 1),
        (1.0, 1.3, 0.7, 1.9, 1.1, 0.4), (1, 1, 0.01, 1, 1, 1), (1, 1, -0.2, 1, 1, 1), (1, 1, 1, 1.5, 1.5, 1.5),
        (1, 1, -1.5, 1, 1, 1), (1, 1, -1, 1, 1, -1), (-1, -1, -1, -1, -1, -1), (1, -1, 1, -1, 1, -1),
        (1, 1, -0.5, -0.9, -0.9, 1),
    ]
    out = [{"type": "prism", "height": h, "face_distance": fd} for fd in prism for h in (1.0, 0.3)]
    rng = np.random.default_rng(20260927)
    out += [{"type": "prism", "height": float(rng.uniform(0.1, 3.0)),
             "face_distance": tuple(float(x) for x in rng.uniform(-0.6, 2.4, 6))} for _ in range(n_random)]
    return out + pyramid_cases(n_random, rng)


def _pyramid(prism_h, upper_h, lower_h, upper_wedge, lower_wedge, face_distance=(1,) * 6) -> dict:
    return {"type": "pyramid", "prism_h": prism_h, "upper_h": upper_h, "lower_h": lower_h,
            "upper_wedge_deg": upper_wedge, "lower_wedge_deg": lower_wedge, "face_distance": tuple(face_distance)}


def pyramid_cases(n_random: int, rng: np.random.Generator) -> list[dict]:
    """Symmetric regular cones (the delegation to ``Pyramid(a, h, c_over_a, tip_ratio)``), the legality boundaries of
    ``doc/configuration.md`` §Pyramid Shape Legality crossed with irregular cross-sections, Miller indices, and
    ``n_random`` random shapes over every field."""
    out = [_pyramid(ph, ch, ch, w, w) for ph, ch, w in
           [(1.0, 0.5, 28.0), (0.4, 0.2, 28.0), (2.0, 0.9, 28.0), (1.0, 0.5, 45.0), (0.7, 0.3, 15.0)]]
    irregular = [(1,) * 6, (1, 1.2, 1, 1.2, 1, 1.2), (1, 2, 1, 2, 1, 2), (1.9, 1, 1, 1.9, 1, 1),
                 (1.0, 1.3, 0.7, 1.9, 1.1, 0.4), (1, 1, -0.5, -0.9, -0.9, 1), (2, 2, 2, 2, 2, 2), (1, 1, 1, 1.5, 1.5, 1.5)]
    heights = [0.0, 1e-6, 0.3, 0.9999, 0.99995, 1.0, 1.7, -0.4]
    for fd in irregular:
        for prism_h in (0.0, 0.6):
            for upper_h in heights:
                for lower_h in (0.0, 0.5, 1.0):
                    out.append(_pyramid(prism_h, upper_h, lower_h, 28.0, 40.0, fd))
    for wedge in (0.05, 0.1, 5.0, 60.0, 89.9, 90.0):   # outside [0.1, 89.9] the cone is absent
        for prism_h in (0.0, 0.5):
            out.append(_pyramid(prism_h, 0.6, 1.0, wedge, 28.0))
            out.append(_pyramid(prism_h, 1.0, 0.0, wedge, wedge, (1, 1.2, 1, 1.2, 1, 1.2)))
    for upper, lower in [((1, 0, 1), (1, 0, 1)), ((2, 0, 3), (1, 0, 1)), ((1, 0, 2), (3, 0, 1)), ((0, 0, 1), (1, 0, 1))]:
        for fd in ((1,) * 6, (1.0, 1.3, 0.7, 1.9, 1.1, 0.4)):
            shape = _pyramid(0.5, 0.7, 1.0, miller_wedge_deg(upper) if upper[0] else 0.0,
                             miller_wedge_deg(lower) if lower[0] else 0.0, fd)
            out.append({**shape, "upper_indices": upper, "lower_indices": lower})
    for _ in range(n_random):
        def height() -> float:
            kind = rng.integers(4)
            return (0.0, float(rng.uniform(0.0, 1.0)), float(rng.uniform(0.0, 1.0)), float(rng.uniform(1.0, 1.6)))[kind]
        def wedge() -> float:
            return float(rng.uniform(0.0, 90.0)) if rng.random() < 0.1 else float(rng.uniform(3.0, 87.0))
        fd = (1,) * 6 if rng.random() < 0.3 else tuple(float(x) for x in rng.uniform(-0.6, 2.4, 6))
        prism_h = 0.0 if rng.random() < 0.2 else float(rng.uniform(0.05, 2.0))
        out.append(_pyramid(prism_h, height(), height(), wedge(), wedge(), fd))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lumice-lib", type=Path, default=Path(os.environ.get("LUMICE_LIB", DEFAULT_LIB)))
    parser.add_argument("--random", type=int, default=300,
                        help="random samples of each type (prism face_distance; pyramid over every field)")
    parser.add_argument("--output", type=Path, help="write the per-case results as JSON")
    args = parser.parse_args(argv)

    try:
        mesher = LumiceMesher(args.lumice_lib)
        print(f"Lumice oracle: {args.lumice_lib}")
    except OSError as error:
        mesher = None
        print(f"Lumice oracle UNAVAILABLE ({error}); running the independent geometry chain alone")

    results, failed = [], 0
    counts = {"independent_ok": 0, "lumice_ok": 0, "rejected_both": 0, "pyramid_cases": 0, "pyramid_accepted": 0,
              "ruler_band": 0}
    for shape in cases(args.random):
        try:
            crystal = build(shape)
        except ValueError:
            crystal = None
        independent = ([] if crystal is None else independent_check(crystal)) + miller_check(shape, crystal)
        lumice = None if mesher is None else lumice_check(crystal, mesher.mesh(shape))
        in_band = bool(lumice) and crystal is not None and shortest_edge(crystal) < RULER_BAND
        ok = not independent and (not lumice or in_band)
        failed += not ok
        counts["ruler_band"] += in_band
        counts["independent_ok"] += crystal is not None and not independent
        counts["lumice_ok"] += lumice is not None and not lumice
        counts["rejected_both"] += crystal is None and lumice == []
        counts["pyramid_cases"] += shape["type"] == "pyramid"
        counts["pyramid_accepted"] += shape["type"] == "pyramid" and crystal is not None
        results.append({"shape": {k: list(v) if isinstance(v, tuple) else v for k, v in shape.items()},
                        "faces": None if crystal is None else sorted(f.number for f in crystal.faces),
                        "independent": independent, "lumice": lumice, "ruler_band": in_band, "ok": ok})
        if not ok:
            print(f"FAIL {shape}: independent {independent}, lumice {lumice}")
        elif in_band:
            print(f"RULER BAND (shortest edge {shortest_edge(crystal):.2e}) {shape}: lumice {lumice}")
    total = len(results)
    print(f"{total} cases: independent geometry ok on {counts['independent_ok']} accepted crystals; "
          + ("Lumice oracle not run" if mesher is None else
             f"Lumice agrees on {counts['lumice_ok']} (rejected by both: {counts['rejected_both']}; "
             f"disagrees in the ruler band on {counts['ruler_band']})")
          + f" [pyramids: {counts['pyramid_cases']} cases, {counts['pyramid_accepted']} accepted]; {failed} failed")
    if args.output:
        args.output.write_text(json.dumps({"lumice_lib": None if mesher is None else str(args.lumice_lib),
                                           "failed": failed, "counts": counts, "cases": results}, indent=1))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
