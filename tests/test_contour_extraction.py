"""``contour``: every component of ``{D_P = delta} ∩ U_P``, certified against the interval partition of ``dp_field``."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from lumice_integral import contour
from lumice_integral.canonical_scene import canonical_crystal
from lumice_integral.dp_field import DPField, TopologyEscape
from lumice_integral.dp_field import field as F
from lumice_integral.path_class import build_path_class, hexprism_symmetry_matrices, path_class_symmetry
from lumice_integral.s2_store import build_event_store

N = 1.31
FIXTURES = ((3, 5), (1, 3), (3, 1, 6), (1, 3, 2), (3, 5, 6, 7, 3))
STORE_N = 200_000
OFFSETS = (1e-6, 1e-3)
IMPROPER = [e for e in hexprism_symmetry_matrices() if np.linalg.det(e) < 0.0]


@pytest.fixture(scope="module")
def fields() -> dict[tuple[int, ...], DPField]:
    return {faces: DPField.build(canonical_crystal(), faces, N) for faces in FIXTURES}


@pytest.fixture(scope="module")
def stores() -> dict[tuple[int, ...], object]:
    return {faces: build_event_store(canonical_crystal(), N, [faces], STORE_N, run_checks=False) for faces in FIXTURES}


def _breaks(field: DPField) -> list[float]:
    partition = field.interval_partition()
    return [partition[0].lower] + [interval.upper for interval in partition]


def _certificate_deltas(field: DPField) -> np.ndarray:
    """Every interval midpoint and every critical value +- 1e-6 and +- 1e-3 rad (both ends of the range included)."""
    partition = field.interval_partition()
    deltas = [0.5 * (iv.lower + iv.upper) for iv in partition]
    for value in _breaks(field):
        deltas += [value + sign * offset for offset in OFFSETS for sign in (-1.0, 1.0)]
    return np.array(deltas)


@pytest.fixture(scope="module")
def certified(fields, stores) -> dict[tuple[int, ...], tuple[contour.LevelSet, ...]]:
    return {faces: contour.extract_level_sets(fields[faces], _certificate_deltas(fields[faces]), stores[faces]) for faces in FIXTURES}


def test_dp_field_margin_forwarders(fields) -> None:
    field = fields[(3, 5)]
    u = np.random.default_rng(0).normal(size=(256, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    np.testing.assert_array_equal(field.margins_batch(u), F.margins_batch(u, field.faces, N))
    np.testing.assert_array_equal(field.valid_batch(u), F.valid_batch(u, field.faces, N))


@pytest.mark.parametrize("faces", FIXTURES)
def test_certificate_across_every_critical_value(fields, certified, faces) -> None:
    """Counts equal the partition's at midpoints and at +-1e-6 / +-1e-3 rad of every critical value, and change across it.

    ``extract_level_sets`` raises on a mismatch; this re-checks the counts
    and that each offset pair really straddles a topology change (interior
    extremum, boundary loop extremum or corner, and both ends of the range).
    """
    field = fields[faces]
    by_delta = {level_set.delta: level_set for level_set in certified[faces]}
    for level_set in certified[faces]:
        expected = (0, 0) if level_set.interval is None else (level_set.interval.n_closed, level_set.interval.n_open)
        assert (level_set.n_closed, level_set.n_open) == expected
    for value in _breaks(field):
        for offset in OFFSETS:
            below, above = by_delta[value - offset], by_delta[value + offset]
            assert (below.n_closed, below.n_open) != (above.n_closed, above.n_open), (np.degrees(value), offset)


@pytest.mark.parametrize("faces", FIXTURES)
def test_nodes_are_on_the_level_set(fields, certified, faces) -> None:
    """``|D - delta| <= 1e-12`` rad wherever the rounding of ``u`` allows it, ``64 eps |grad D|`` next to an exit-TIR curve."""
    field = fields[faces]
    components = [c for level_set in certified[faces] for c in level_set.components]
    points = np.vstack([c.points for c in components])
    # recomputed through DPField.d_p_batch (not the walker's kernel), in one batch: a call per component shape recompiles
    residuals = field.d_p_batch(points) - np.concatenate([np.full(len(c.points), c.delta) for c in components])
    gradient = np.linalg.norm(field.gradient_batch(points), axis=1)
    tolerance = np.maximum(contour.LEVEL_RESIDUAL_TOL, contour.ROUNDING_ULPS * np.finfo(float).eps * gradient)
    assert np.all(np.abs(residuals) <= tolerance)
    assert np.all(np.abs(residuals[gradient <= 1e3]) <= contour.LEVEL_RESIDUAL_TOL)
    assert np.all(field.valid_batch(points))


@pytest.mark.parametrize("faces", FIXTURES)
def test_open_arcs_end_on_the_boundary(certified, faces) -> None:
    for level_set in certified[faces]:
        for component in level_set.components:
            if component.closed:
                assert component.start_margin is None and component.end_margin is None
                continue
            # the walker keeps every margin above INSIDE_MARGIN_FLOOR; another compilation rounds it by a few eps
            assert 0.0 < component.start_margin_value <= contour.ENDPOINT_MARGIN_ATOL
            assert 0.0 < component.end_margin_value <= contour.ENDPOINT_MARGIN_ATOL
            assert np.linalg.norm(component.points[0] - component.points[-1]) > 1e-9


def test_small_loop_is_walked_once(fields, certified) -> None:
    """Phase I defect 1: the loop ``1e-6`` rad above the ``3-5`` minimum (``~2e-3`` rad across) winds once, not twice."""
    field = fields[(3, 5)]
    (minimum,) = field.interior_critical_points
    (level_set,) = [ls for ls in certified[(3, 5)] if ls.delta == minimum.value + 1e-6]
    (loop,) = level_set.components
    assert loop.closed
    centre = minimum.position
    e1 = np.cross(centre, np.eye(3)[int(np.argmin(np.abs(centre)))])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(centre, e1)
    angles = np.unwrap(np.arctan2(loop.points @ e2, loop.points @ e1))
    closing = np.angle(np.exp(1j * (angles[0] - angles[-1])))
    assert abs(angles[-1] - angles[0] + closing) == pytest.approx(2.0 * np.pi, abs=1e-9)
    radius = np.arccos(np.clip(loop.points @ centre, -1.0, 1.0))
    assert 1e-4 < radius.min() and radius.max() < 1e-2
    # the closing gap is at most one step, never an absolute distance
    steps = np.linalg.norm(np.diff(loop.points, axis=0), axis=1)
    assert np.linalg.norm(loop.points[-1] - loop.points[0]) <= 1.5 * steps.max()
    assert len(loop.points) >= 2.0 * np.pi / contour.MAX_TURN_RAD


def test_slab_cone_loop_next_to_pi(fields, certified) -> None:
    """``3-5-6-7-3``: ``D = pi - 2 r`` around ``+n_M``, so the loop at ``pi - 1e-6`` has radius ``5e-7`` rad."""
    (level_set,) = [ls for ls in certified[(3, 5, 6, 7, 3)] if abs(ls.delta - (np.pi - 1e-6)) < 1e-9]
    (loop,) = level_set.components
    centre = fields[(3, 5, 6, 7, 3)].fold.axis
    centre = centre if centre @ loop.points[0] > 0.0 else -centre
    radius = np.arctan2(np.linalg.norm(np.cross(loop.points, centre), axis=1), loop.points @ centre)
    assert radius == pytest.approx(np.full(len(radius), 5e-7), rel=1e-6)


def test_same_delta_same_components(fields, stores) -> None:
    """The output depends on ``delta`` only: two pixels of one ``delta`` share one extraction, and a second call repeats it."""
    field, store = fields[(3, 5)], stores[(3, 5)]
    a, b = np.radians(35.0), np.radians(45.0)
    first = contour.extract_level_sets(field, [a, b, a], store)
    assert first[0] is first[2]
    again = contour.extract_level_sets(field, a, store)
    for mine, theirs in zip(first[0].components, again[0].components, strict=True):
        assert mine.kind == theirs.kind
        np.testing.assert_array_equal(mine.points, theirs.points)


@pytest.mark.parametrize(
    ("representative", "member", "elements", "det"),
    [
        ((3, 5), (3, 7), None, 1.0),
        ((3, 5), (3, 7), IMPROPER, -1.0),
        ((3, 5, 6, 7, 3), (4, 8, 7, 6, 4), None, 1.0),
        ((3, 5, 6, 7, 3), (4, 8, 7, 6, 4), IMPROPER, -1.0),
    ],
)
def test_transport_matches_independent_extraction(fields, stores, representative, member, elements, det) -> None:
    """A class member's level sets are the representative's moved by ``u -> g u`` (orientation kept), not re-extracted."""
    crystal = canonical_crystal()
    g = path_class_symmetry(build_path_class(crystal, representative), symmetry_elements=elements)[member]
    assert np.linalg.det(g) == pytest.approx(det)
    rep = fields[representative]
    other = DPField.build(crystal, member, N)
    other_store = build_event_store(crystal, N, [member], STORE_N, run_checks=False)
    deltas = [0.5 * (iv.lower + iv.upper) for iv in rep.interval_partition()]
    moved = [ls.transported(g) for ls in contour.extract_level_sets(rep, deltas, stores[representative])]
    mine = contour.extract_level_sets(other, deltas, other_store)
    for transported, extracted in zip(moved, mine, strict=True):
        assert (transported.n_closed, transported.n_open) == (extracted.n_closed, extracted.n_open)
        # arcs meet an exit-TIR piece tangentially: two walks of one arc stop ~sqrt(WALK_MIN_STEP_RAD) apart there
        slack = 10.0 * np.sqrt(contour.WALK_MIN_STEP_RAD)
        for a, b in ((transported, extracted), (extracted, transported)):
            polylines = [(0, c.points, c.closed) for c in b.components]
            for component in a.components:
                groups = np.zeros(len(component.points), dtype=int)
                assert np.all(contour._covered(component.points, groups, polylines, slack=slack))
                away = np.ones(len(component.points), dtype=bool)
                if not component.closed:
                    for end in (component.points[0], component.points[-1]):
                        away &= np.linalg.norm(component.points - end, axis=1) > slack
                assert np.all(contour._covered(component.points[away], groups[away], polylines))
        for component in transported.components:
            residual = other.d_p_batch(component.points) - transported.delta
            gradient = np.linalg.norm(other.gradient_batch(component.points), axis=1)
            assert np.all(np.abs(residual) <= np.maximum(1e-11, 1e3 * np.finfo(float).eps * gradient))
            # orientation: points run along cross(u, grad D) on the member too
            tangent = np.diff(component.points, axis=0)
            along = np.cross(component.points[:-1], other.gradient_batch(component.points[:-1]))
            assert np.all(np.einsum("ij,ij->i", tangent, along) > 0.0)


def test_store_and_grid_seeds_find_a_component_without_critical_seeds(fields, stores, monkeypatch) -> None:
    """Without the ray seed of the ``3-5`` minimum the loop is found by the independent (store / grid) seeds."""
    monkeypatch.setattr(contour, "_extremum_seeds", lambda field, deltas: (np.zeros((0, 3)), np.zeros(0, dtype=int)))
    (level_set,) = contour.extract_level_sets(fields[(3, 5)], np.radians(30.0), stores[(3, 5)])
    assert (level_set.n_closed, level_set.n_open) == (1, 0)


def test_certificate_refuses_a_missed_component(fields, stores, monkeypatch) -> None:
    """With every seed of the loop removed the extraction is short of the prediction: an error, not an empty level set."""
    monkeypatch.setattr(contour, "_extremum_seeds", lambda field, deltas: (np.zeros((0, 3)), np.zeros(0, dtype=int)))
    with pytest.raises(contour.ContourCertificateError, match=r"predicted \(closed, open\) = \(1, 0\), extracted \(0, 0\)"):
        contour.extract_level_sets(fields[(3, 5)], np.radians(30.0), stores[(3, 5)], grid=3, band_halfwidth=0.0)


def test_certificate_refuses_an_extra_component(fields, stores, monkeypatch) -> None:
    """A component the partition does not predict (here: the partition told there is none) is an error too."""
    field = fields[(3, 5)]
    partition = field.interval_partition()
    empty = tuple(iv._replace(n_components=0, n_closed=0, n_open=0) for iv in partition)
    monkeypatch.setattr(DPField, "interval_partition", lambda self: empty)
    with pytest.raises(contour.ContourCertificateError, match=r"predicted \(closed, open\) = \(0, 0\), extracted \(1, 0\)"):
        contour.extract_level_sets(field, np.radians(30.0), stores[(3, 5)])


def test_saddle_escapes_before_extraction(stores) -> None:
    """A saddle makes the partition escape (``dp_field.certificate``); extraction propagates it and extracts nothing.

    No face sequence of the prism has shown an interior saddle (explore
    ``dp-field-saddle-search``: 87 non-empty candidate paths searched), so
    the saddle branch of the certificate is exercised by planting one; the
    level-set splitting at a real saddle is not covered by any fixture.
    """
    field = DPField.build(canonical_crystal(), (3, 5), N)
    (minimum,) = field.interior_critical_points
    saddle = dataclasses.replace(minimum, kind="saddle", morse_index=1)
    field.__dict__["_interior"] = ((saddle,), None)
    with pytest.raises(TopologyEscape, match="saddle"):
        contour.extract_level_sets(field, np.radians(30.0), stores[(3, 5)])


def test_fixtures_have_no_saddle(fields) -> None:
    """The documented limit of the saddle coverage above: every fixture's interior critical points are extrema."""
    for field in fields.values():
        assert all(p.kind in ("minimum", "maximum", "degenerate") for p in field.interior_critical_points)


def test_delta_at_a_critical_value_is_refused(fields, stores) -> None:
    field = fields[(3, 5)]
    with pytest.raises(ValueError, match="critical value"):
        contour.extract_level_sets(field, field.interior_critical_points[0].value, stores[(3, 5)])


@pytest.mark.parametrize("faces", [(1, 3, 2), (3, 1, 6)])
def test_boundary_seeds_clear_a_square_coincident_margin(fields, faces, monkeypatch) -> None:
    """Along the entry piece of ``1-3-2`` / ``3-1-6`` the exit Snell discriminant is the entry cosine squared.

    A seed pulled ``1e-8`` inside sits at ``1e-16`` on it, below ``INSIDE_MARGIN_FLOOR``: with the targets
    ``(1e-14, 1e-8)`` no deviation had a boundary seed and every component came from the fallback seeds,
    whose rounds ran out past ~180 deviations per call.  The later targets seed every deviation.
    """
    field = fields[faces]
    deltas = np.radians(np.linspace(0.5, 179.5, 179))
    _, j = contour._boundary_seeds(field, deltas)
    assert set(j.tolist()) == set(range(len(deltas)))
    monkeypatch.setattr(contour, "BOUNDARY_SEED_MARGINS", (1e-14, 1e-8))
    _, j = contour._boundary_seeds(field, deltas)
    assert len(j) == 0


@pytest.mark.slow
@pytest.mark.parametrize("faces", [(1, 3, 2), (3, 1, 6), (3, 5, 6, 7, 3), (3, 5, 6, 7), (3, 4, 5, 7)])
def test_one_call_certifies_a_column_of_deviations(faces) -> None:
    """801 deviations in one call (a full column of the canonical strip), certified; no fallback round needed.

    The A60-10 members ``3-5-6-7`` (a necked ``U_P``: 50000 lattice points) and ``3-4-5-7`` included.
    ~10 s per path on an M2 Max, most of it the store.
    """
    field = DPField.build(canonical_crystal(), faces, N, **({"lattice_n": 50000} if faces == (3, 5, 6, 7) else {}))
    store = build_event_store(canonical_crystal(), N, [faces], STORE_N, run_checks=False)
    breaks = np.array(_breaks(field))
    deltas = np.linspace(breaks[0], breaks[-1], 803)[1:-1]
    deltas = deltas[np.min(np.abs(deltas[:, None] - breaks[None]), axis=1) > 1e-6]
    rounds = []
    original = contour._components_of

    def counting(field_, seeds, j, deltas_):
        rounds.append(len(seeds))
        return original(field_, seeds, j, deltas_)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(contour, "_components_of", counting)
        level_sets = contour.extract_level_sets(field, deltas, store)
    assert len(rounds) == 1
    assert len(level_sets) == len(deltas) >= 800
    for level_set in level_sets:
        assert (level_set.n_closed, level_set.n_open) == (level_set.interval.n_closed, level_set.interval.n_open)
