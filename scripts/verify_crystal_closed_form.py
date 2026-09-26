"""Verify the closed-form crystals (``geometry.closed_form``) against Lumice's own mesh and by independent geometry.

Two chains, run together whenever both are available and reported side by side:

1. **Lumice oracle** (validation boundary only; nothing here is imported by the package): the shared library's
   ``LUMICE_GetCrystalMesh`` is called through ``ctypes`` with the same shape (prism ``height`` /
   ``face_distance``; pyramid ``prism_h`` / ``upper_h = lower_h`` / wedge angle), and the present face numbers,
   face normals and per-face vertex sets are compared with Lumice Integral's crystal.  Lumice builds its mesh at
   circumscribed diameter 1, i.e. ``a = 1/2``, with ``h = 2a·height`` (prism) or ``2a·prism_h`` (pyramid),
   ``tip_ratio = upper_h`` and ``c_over_a = (√3/2)·cot(wedge)`` (the wedge is the face-to-c-axis angle).
   Lumice computes in float32, so the comparison tolerance is ``1e-5`` of the crystal size.  A shape Lumice
   rejects comes back as an empty mesh; Lumice Integral must raise ``ValueError`` for exactly those.
2. **Independent geometry** (always): every vertex of every face satisfies all half-spaces, every face is a
   polygon of at least three distinct corners, every edge has two faces, and ``V - E + F = 2``.

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
            param.upper_h.center = param.lower_h.center = shape["cone_h"]
            param.upper_wedge_angle = param.lower_wedge_angle = shape["wedge_deg"]
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

def build(shape: dict) -> Polyhedron:
    """The Lumice shape in Lumice Integral (raises ``ValueError`` for a rejected shape)."""
    fd = tuple(float(np.float32(d)) for d in shape["face_distance"])   # the float32 values Lumice receives
    if shape["type"] == "prism":
        return HexPrism.from_lumice(float(np.float32(shape["height"])), fd, a=LUMICE_A)
    if fd != (1.0,) * 6:
        raise NotImplementedError("Pyramid is regular in cross-section (irregular cones are not in scope)")
    wedge = np.radians(float(np.float32(shape["wedge_deg"])))
    return Pyramid(a=LUMICE_A, h=2.0 * LUMICE_A * float(np.float32(shape["prism_h"])),
                   c_over_a=np.sqrt(3.0) / 2.0 / np.tan(wedge), tip_ratio=float(np.float32(shape["cone_h"])))


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
        if not all(crystal.contains(p, eps=1e-12 * scale) for p in pts):
            problems.append(f"face {face.number} has a corner outside another half-space")
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
    out += [{"type": "pyramid", "prism_h": ph, "cone_h": ch, "wedge_deg": w, "face_distance": (1,) * 6}
            for ph, ch, w in [(1.0, 0.5, 28.0), (0.4, 0.2, 28.0), (2.0, 0.9, 28.0), (1.0, 0.5, 45.0), (0.7, 0.3, 15.0)]]
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lumice-lib", type=Path, default=Path(os.environ.get("LUMICE_LIB", DEFAULT_LIB)))
    parser.add_argument("--random", type=int, default=300, help="random prism face_distance samples")
    parser.add_argument("--output", type=Path, help="write the per-case results as JSON")
    args = parser.parse_args(argv)

    try:
        mesher = LumiceMesher(args.lumice_lib)
        print(f"Lumice oracle: {args.lumice_lib}")
    except OSError as error:
        mesher = None
        print(f"Lumice oracle UNAVAILABLE ({error}); running the independent geometry chain alone")

    results, failed = [], 0
    counts = {"independent_ok": 0, "lumice_ok": 0, "rejected_both": 0}
    for shape in cases(args.random):
        try:
            crystal = build(shape)
        except ValueError:
            crystal = None
        independent = [] if crystal is None else independent_check(crystal)
        lumice = None if mesher is None else lumice_check(crystal, mesher.mesh(shape))
        ok = not independent and not lumice
        failed += not ok
        counts["independent_ok"] += crystal is not None and not independent
        counts["lumice_ok"] += lumice is not None and not lumice
        counts["rejected_both"] += crystal is None and lumice == []
        results.append({"shape": {k: list(v) if isinstance(v, tuple) else v for k, v in shape.items()},
                        "faces": None if crystal is None else sorted(f.number for f in crystal.faces),
                        "independent": independent, "lumice": lumice, "ok": ok})
        if not ok:
            print(f"FAIL {shape}: independent {independent}, lumice {lumice}")
    total = len(results)
    print(f"{total} cases: independent geometry ok on {counts['independent_ok']} accepted crystals; "
          + ("Lumice oracle not run" if mesher is None else
             f"Lumice agrees on {counts['lumice_ok']} (rejected by both: {counts['rejected_both']})")
          + f"; {failed} failed")
    if args.output:
        args.output.write_text(json.dumps({"lumice_lib": None if mesher is None else str(args.lumice_lib),
                                           "failed": failed, "counts": counts, "cases": results}, indent=1))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
