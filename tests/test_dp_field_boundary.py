"""``dp_field.boundary``: the walk of ``dU_P``, its corners, curve kinds and margin identities."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial import cKDTree

from lumice_integral import optics
from lumice_integral.canonical_scene import canonical_crystal
from lumice_integral.dp_field import DPField
from lumice_integral.dp_field import boundary as B
from lumice_integral.dp_field import field as F
from lumice_integral.s2_store import fibonacci_sphere

N = 1.31
FIXTURES = ((3, 5), (1, 3), (3, 1, 6), (1, 3, 2), (3, 5, 6, 7, 3))


@pytest.fixture(scope="module")
def fields() -> dict[tuple[int, ...], DPField]:
    return {faces: DPField.build(canonical_crystal(), faces, N) for faces in FIXTURES}


def _angle(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.arctan2(np.linalg.norm(np.cross(a, b)), a @ b))


def test_identical_margins_of_60_degree_side_face_triples() -> None:
    """Three consecutive reflections off side faces stepping by the same +-60 degrees: third step = first step."""
    assert B.identical_margins((3, 5, 6, 7, 3)) == {
        "internal_3_incidence_cosine": "internal_1_incidence_cosine",
        "internal_3_tir_discriminant": "internal_1_tir_discriminant",
    }
    assert B.identical_margins((3, 4, 5, 6, 3)) == {
        "internal_3_incidence_cosine": "internal_1_incidence_cosine",
        "internal_3_tir_discriminant": "internal_1_tir_discriminant",
    }
    assert B.identical_margins((3, 8, 7, 6, 3))["internal_3_tir_discriminant"] == "internal_1_tir_discriminant"
    # negative controls: steps 120 / -60, a basal face in the triple, only two reflections
    assert B.identical_margins((3, 5, 7, 6, 3)) == {}
    assert B.identical_margins((3, 5, 1, 7, 3)) == {}
    assert B.identical_margins((3, 1, 6)) == {}


@pytest.mark.parametrize("faces", [(3, 5, 6, 7, 3), (3, 4, 5, 6, 3)])
def test_dropped_margins_equal_the_kept_ones_as_functions(faces) -> None:
    u = np.random.default_rng(1).normal(size=(20000, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    margins = F.margins_batch(u, faces, N)
    names = optics.domain_margin_names(faces)
    for dropped, kept in B.identical_margins(faces).items():
        assert np.max(np.abs(margins[:, names.index(dropped)] - margins[:, names.index(kept)])) <= 1e-14


def test_great_circle_margins_are_decided_per_path() -> None:
    """The entry margin always; an internal / exit incidence cosine iff ``m . n_a = 0`` (and it checks out on the circle)."""
    crystal = canonical_crystal()
    circles = {faces: B.great_circle_margins(crystal, faces, N) for faces in FIXTURES}
    for faces, found in circles.items():
        assert "entry_incidence_cosine" in found
        np.testing.assert_allclose(found["entry_incidence_cosine"], optics.HEXPRISM_BODY_NORMALS[faces[0]], atol=1e-15)
    assert set(circles[(1, 3, 2)]) == {"entry_incidence_cosine", "internal_1_incidence_cosine"}
    assert set(circles[(3, 1, 6)]) == {"entry_incidence_cosine", "internal_1_incidence_cosine"}
    assert set(circles[(1, 3)]) == {"entry_incidence_cosine", "exit_incidence_cosine"}
    # 3-5-6-7-3: m = n_5 for the first reflection, n_5 . n_3 = -1/2: not a great circle, marched
    assert set(circles[(3, 5, 6, 7, 3)]) == {"entry_incidence_cosine"}
    assert set(circles[(3, 5)]) == {"entry_incidence_cosine"}


@pytest.mark.parametrize("faces", FIXTURES)
def test_entry_margin_vanishes_on_its_great_circle(faces) -> None:
    n_a = np.asarray(optics.HEXPRISM_BODY_NORMALS[faces[0]])
    e1 = np.cross(n_a, np.eye(3)[int(np.argmin(np.abs(n_a)))])
    e1 /= np.linalg.norm(e1)
    t = np.linspace(0.0, 2.0 * np.pi, 3600, endpoint=False)
    circle = np.cos(t)[:, None] * e1 + np.sin(t)[:, None] * np.cross(n_a, e1)
    assert np.max(np.abs(F.margins_batch(circle, faces, N)[:, 0])) <= 1e-15


# (pieces in walk order as margins, from an arbitrary first corner): the loop is compared up to rotation
EXPECTED_PIECES = {
    (3, 5): ("entry_incidence_cosine", "exit_snell_discriminant"),
    (1, 3): ("entry_incidence_cosine", "exit_snell_discriminant"),
    (3, 1, 6): ("entry_incidence_cosine", "internal_1_tir_discriminant", "entry_incidence_cosine", "internal_1_incidence_cosine"),
    (1, 3, 2): ("entry_incidence_cosine", "internal_1_tir_discriminant", "entry_incidence_cosine", "internal_1_incidence_cosine"),
    (3, 5, 6, 7, 3): (
        "internal_1_tir_discriminant",
        "entry_incidence_cosine",
        "internal_2_tir_discriminant",
        "entry_incidence_cosine",
    ),
}


def _rotations(sequence: tuple[str, ...]) -> set[tuple[str, ...]]:
    return {sequence[i:] + sequence[:i] for i in range(len(sequence))}


@pytest.mark.parametrize("faces", FIXTURES)
def test_walk_closes_with_the_expected_pieces(fields, faces) -> None:
    loop = fields[faces].boundary
    assert tuple(p.margin for p in loop.pieces) in _rotations(EXPECTED_PIECES[faces])
    assert len(loop.corners) == len(EXPECTED_PIECES[faces])
    for i, piece in enumerate(loop.pieces):
        np.testing.assert_array_equal(piece.points[-1], loop.corners[i].position)
        np.testing.assert_array_equal(piece.points[0], loop.corners[i - 1].position)
        assert loop.corners[i].incoming == piece.margin
        assert loop.corners[i].outgoing == loop.pieces[(i + 1) % len(loop.pieces)].margin
        assert piece.kind == ("great_circle" if piece.margin in loop.great_circle_margins else "marched")


@pytest.mark.parametrize("faces", FIXTURES)
def test_pieces_stay_on_their_zero_set_inside_the_closure(fields, faces) -> None:
    names = optics.domain_margin_names(faces)
    for piece in fields[faces].boundary.pieces:
        margins = F.margins_batch(piece.points, faces, N)
        assert np.max(np.abs(margins[:, names.index(piece.margin)])) <= 2e-15
        others = [k for k, name in enumerate(names) if name != piece.margin and name not in piece.coincident]
        assert np.min(margins[:, others]) >= -B.VIOLATION_ATOL
        for name in piece.coincident:
            assert np.max(np.abs(margins[:, names.index(name)])) <= B.COINCIDENT_ATOL


@pytest.mark.parametrize("faces", FIXTURES)
def test_corners_are_exact(fields, faces) -> None:
    for corner in fields[faces].corners:
        assert corner.residual <= 3e-16
        assert corner.transversal == ()


def test_corners_on_the_entry_circle_match_a_1d_scan_with_path_domain() -> None:
    """Independent of the walk (explore ``dp-field-boundary-corners`` #2): bisection on the entry circle with the scalar gates."""
    faces = (3, 5)
    walked = DPField.build(canonical_crystal(), faces, N).corners
    n_a = np.array([1.0, 0.0, 0.0])
    e1, e2 = np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])

    def inside(t: float) -> bool:
        u = np.cos(t) * e1 + np.sin(t) * e2 + 1e-9 * n_a
        return optics.path_domain(np.eye(3), faces, -u / np.linalg.norm(u), N).valid

    grid = np.linspace(0.0, 2.0 * np.pi, 3601)
    flags = [inside(t) for t in grid]
    found = []
    for k in range(3600):
        if flags[k] != flags[k + 1]:
            lo, hi = grid[k], grid[k + 1]
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                (lo, hi) = (mid, hi) if inside(mid) == flags[k] else (lo, mid)
            found.append(np.cos(lo) * e1 + np.sin(lo) * e2)
    assert len(found) == len(walked) == 2
    for corner in walked:
        assert min(_angle(corner.position, f) for f in found) <= 1e-10


def test_liljequist_corners_carry_every_vanishing_margin(fields) -> None:
    """``3-5-6-7-3``: four margins vanish at every corner, two bound ``U_P``; the third is tangent, ``exit_snell`` coincident.

    Measured structure (gradient test, angle ``0.0``): at the pair on
    ``internal_1_tir`` the third curve ``internal_2_incidence_cosine`` is
    tangent to it (explore ``dp-field-bigon-other-corner-pair``), and at the
    pair on ``internal_2_tir`` the third curve ``internal_1_incidence_cosine``
    is tangent to *that* edge as well -- explore
    ``dp-field-boundary-deep-internal-faces`` read this pair as transversal
    (a bigon); the lens between the two tangent curves lies outside ``U_P``.
    """
    corners = fields[(3, 5, 6, 7, 3)].corners
    by_edge: dict[str, list] = {"internal_1_tir_discriminant": [], "internal_2_tir_discriminant": []}
    for corner in corners:
        edges = {corner.incoming, corner.outgoing}
        assert "entry_incidence_cosine" in edges
        tir = (edges - {"entry_incidence_cosine"}).pop()
        by_edge[tir].append(corner)
    for tir, third in (("internal_1_tir_discriminant", "internal_2_incidence_cosine"), ("internal_2_tir_discriminant", "internal_1_incidence_cosine")):
        assert len(by_edge[tir]) == 2
        for corner in by_edge[tir]:
            assert set(corner.margins) == {"entry_incidence_cosine", tir, third, "exit_snell_discriminant"}
            assert corner.tangent == (third,)
            assert corner.coincident == ("exit_snell_discriminant",)
            assert "internal_3_tir_discriminant" not in corner.margins  # dropped before walking


def test_liljequist_corner_positions_on_the_entry_circle(fields) -> None:
    """The entry circle ``x = 0`` parametrised as ``(0, cos t, sin t)``: corners at ``t = +-60.753, +-119.247`` degrees."""
    t = sorted(np.degrees(np.arctan2(c.position[2], c.position[1])) for c in fields[(3, 5, 6, 7, 3)].corners)
    np.testing.assert_allclose(t, [-119.2465917, -60.7534083, 60.7534083, 119.2465917], atol=1e-6)


@pytest.mark.parametrize("faces", FIXTURES)
def test_walk_accounts_for_every_lattice_edge_point(fields, faces) -> None:
    """Completeness spot check: every lattice point of ``U_P`` with an outside neighbour is next to the walked loop."""
    lattice = fibonacci_sphere(20000)
    valid = F.valid_batch(lattice, faces, N)
    _, neighbours = cKDTree(lattice).query(lattice, k=7)
    edge = lattice[valid & np.any(~valid[neighbours[:, 1:]], axis=1)]
    loop = np.concatenate([p.points for p in fields[faces].boundary_curves])
    distances, _ = cKDTree(loop).query(edge)
    spacing = np.sqrt(4.0 * np.pi / 20000)
    assert np.max(distances) <= 1.5 * spacing


@pytest.mark.parametrize("faces", [(3, 5), (1, 3)])
def test_restricted_extrema_come_in_mirror_pairs(fields, faces) -> None:
    """The two-face fixtures are symmetric under the mirror through ``n_a`` and ``n_b``: extrema of the entry and exit pieces pair up."""
    values = sorted(p.value for p in fields[faces].boundary_critical_points)
    assert len(values) % 2 == 0
    pairs = np.array(values).reshape(-1, 2)
    assert np.max(np.abs(pairs[:, 0] - pairs[:, 1])) <= 5e-8  # exit-TIR pieces carry the ~1e-8 sqrt error
