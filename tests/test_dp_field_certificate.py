"""``dp_field.certificate``: the ``delta`` partition, its counts, the escape hatch, and the independent grid check."""

from __future__ import annotations

import dataclasses
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from lumice_integral.canonical_scene import canonical_crystal
from lumice_integral.dp_field import DPField, TopologyEscape
from lumice_integral.dp_field import certificate as C

N = 1.31
FIXTURES = ((3, 5), (1, 3), (3, 1, 6), (1, 3, 2), (3, 5, 6, 7, 3))
SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "verify_dp_field_intervals.py"

# (lower deg, upper deg, n_components, n_closed, n_open), checked by scripts/verify_dp_field_intervals.py
EXPECTED = {
    (3, 5): [(21.8393, 42.99086, 1, 1, 0), (42.99086, 43.46516, 4, 0, 4), (43.46516, 50.06262, 2, 0, 2)],
    (1, 3): [(45.73342, 57.80363, 1, 1, 0), (57.80363, 73.50689, 2, 0, 2)],
    (3, 1, 6): [(0.0, 180.0, 1, 0, 1)],
    (1, 3, 2): [(0.0, 180.0, 1, 0, 1)],
    (3, 5, 6, 7, 3): [(0.0, 98.16070, 2, 0, 2), (98.16070, 180.0, 1, 1, 0)],
}
# The A60-10 members (two internal reflections, no slab): the loop maximum 141.8393 = 120 + 21.8393 degrees sits
# on a grazing internal-reflection piece, where the smooth extension of D_P has a saddle (not interior).  The
# U_P of 3-5-6-7 has a neck that the default 20000-point lattice splits in two: its field takes 50000 points.
A60_10 = {
    (3, 5, 6, 7): (50000, [(50.06262, 141.83930, 2, 0, 2), (141.83930, 163.46516, 1, 0, 1)]),
    (3, 4, 5, 7): (20000, [(50.06262, 141.83930, 2, 0, 2), (141.83930, 163.46516, 1, 0, 1)]),
}


@pytest.fixture(scope="module")
def fields() -> dict[tuple[int, ...], DPField]:
    return {faces: DPField.build(canonical_crystal(), faces, N) for faces in FIXTURES}


@pytest.mark.parametrize("faces", FIXTURES)
def test_domains_are_disks(fields, faces) -> None:
    topology = fields[faces].domain_topology
    assert (topology.domain_components, topology.complement_components) == (1, 1)


@pytest.mark.parametrize("faces", FIXTURES)
def test_interval_partition(fields, faces) -> None:
    partition = fields[faces].interval_partition()
    assert [iv[2:] for iv in partition] == [e[2:] for e in EXPECTED[faces]]
    for interval, expected in zip(partition, EXPECTED[faces]):
        assert np.degrees(interval.lower) == pytest.approx(expected[0], abs=5e-6)
        assert np.degrees(interval.upper) == pytest.approx(expected[1], abs=5e-6)
        assert interval.n_components == interval.n_closed + interval.n_open


@pytest.mark.parametrize("faces", sorted(A60_10))
def test_a60_10_members_partition(faces) -> None:
    """Two partial internal reflections: a disk, no interior critical point, two open arcs then one."""
    lattice_n, expected = A60_10[faces]
    field = DPField.build(canonical_crystal(), faces, N, lattice_n=lattice_n)
    assert field.domain_topology.is_disk
    assert field.interior_critical_points == ()
    partition = field.interval_partition()
    assert [iv[2:] for iv in partition] == [e[2:] for e in expected]
    for interval, e in zip(partition, expected):
        assert np.degrees(interval.lower) == pytest.approx(e[0], abs=5e-6)
        assert np.degrees(interval.upper) == pytest.approx(e[1], abs=5e-6)


def test_3_5_has_one_closed_loop_from_the_minimum_to_the_boundary(fields) -> None:
    """Explore ``dp-field-topology`` #5: one component all along; it is a closed loop until ``D`` reaches ``dU_P``."""
    field = fields[(3, 5)]
    first = field.interval_partition()[0]
    assert first.lower == pytest.approx(field.interior_critical_points[0].value, abs=1e-15)
    assert first.upper == pytest.approx(min(p.value for p in field.boundary_critical_points), abs=1e-15)
    assert (first.n_closed, first.n_open) == (1, 0)


def test_parhelic_circle_has_no_fold_and_one_arc(fields) -> None:
    """``3-1-6``: ``D = 2 |elevation|`` on ``U_P``, a lune: 0 along the crease arc of ``dU_P``, up to 180 along the entry arc.

    With a partial basal reflection the pose stays in ``U_P``: the arc on
    ``internal_1_tir_discriminant`` (``D = 115.6`` degrees) is gone, and the
    parhelic circle runs round to the anthelion at ``-n_M`` on the entry arc.
    """
    field = fields[(3, 1, 6)]
    assert field.interior_critical_points == ()
    (interval,) = field.interval_partition()
    (crease,) = [p for p in field.boundary_curves if p.margin == "internal_1_incidence_cosine"]
    (entry,) = [p for p in field.boundary_curves if p.margin == "entry_incidence_cosine"]
    assert np.max(crease.values) <= 1e-12
    assert np.degrees(np.max(entry.values)) == pytest.approx(180.0, abs=1e-2)  # the 0.25 degree walk samples
    assert np.degrees(interval.upper) == pytest.approx(180.0, abs=1e-9)
    assert [where for _, where in field.degenerate_fold.axis_points] == ["boundary", "exterior"]


def test_liljequist_pair_critical_sets(fields) -> None:
    """``1-3-2`` and ``3-5-6-7-3``: one field, ``D = angle(S_x u, u)``, two domains, two different critical sets.

    Both fold matrices are the mirror of face 3 (``phi_key`` differs only in
    the entry normal), so the fields agree pointwise; the critical sets of
    ``D_P`` on ``U_P`` do not: ``{0, 180}`` degrees on ``1-3-2`` (``-n_M`` on
    the entry arc) and ``{0, 98.161, 180}`` on ``3-5-6-7-3`` (``+n_M``
    inside).  Neither has a value near ``142 = 120 + 21.84`` degrees (the
    A60-10 members do, :data:`A60_10`); the chapter-8 reading is task
    ``ch10-liljequist-unblock-and-docs``'.  Before partial internal
    reflections the TIR arcs gave ``{0, 115.607}`` and ``{0, 153.070, 180}``.
    """
    short, long = fields[(1, 3, 2)], fields[(3, 5, 6, 7, 3)]
    np.testing.assert_allclose(short.slab, long.slab, atol=1e-15)
    u = np.random.default_rng(4).normal(size=(1000, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    np.testing.assert_allclose(short.d_p_batch(u), long.d_p_batch(u), atol=1e-15)
    np.testing.assert_allclose(np.degrees(short.critical_values), [0.0, 180.0], atol=1e-6)
    np.testing.assert_allclose(np.degrees(long.critical_values), [0.0, 98.160700, 180.0], atol=1e-6)
    assert short.critical_set.mismatch(long.critical_set) == (float("inf"), float("inf"))  # interior sets differ
    for field in (short, long):
        assert np.all(np.abs(np.degrees(field.critical_values) - (120.0 + 21.8393)) > 10.0)


def test_escape_hatch_not_a_disk(fields) -> None:
    field = fields[(3, 5)]
    annulus = C.DomainTopology(20000, 1, 2)
    with pytest.raises(TopologyEscape, match="not a disk"):
        C.interval_partition(field.faces, N, field.interior_critical_points, None, field.boundary, annulus, None)


def test_escape_hatch_two_interior_critical_points(fields) -> None:
    field = fields[(3, 5)]
    (minimum,) = field.interior_critical_points
    saddle = dataclasses.replace(minimum, kind="saddle", morse_index=1)
    with pytest.raises(TopologyEscape, match="2 interior critical points"):
        C.interval_partition(field.faces, N, (minimum, saddle), None, field.boundary, field.domain_topology, None)
    with pytest.raises(TopologyEscape, match="saddle"):
        C.interval_partition(field.faces, N, (saddle,), None, field.boundary, field.domain_topology, None)


def test_escape_hatch_crease_inside(fields) -> None:
    field = fields[(3, 1, 6)]
    crease = dataclasses.replace(field.degenerate_fold, circle_interior_fraction=0.1)
    with pytest.raises(TopologyEscape, match="crease"):
        C.interval_partition(field.faces, N, (), crease, field.boundary, field.domain_topology, field.slab)


def test_escape_hatch_minimum_that_never_reaches_the_boundary_first(fields) -> None:
    """An interior minimum above the loop minimum cannot be the first to touch ``dU_P``: refused, not guessed."""
    field = fields[(3, 5)]
    (minimum,) = field.interior_critical_points
    high = dataclasses.replace(minimum, value=np.radians(45.0))
    with pytest.raises(TopologyEscape, match="not shown to reach"):
        C.interval_partition(field.faces, N, (high,), None, field.boundary, field.domain_topology, None)


@pytest.mark.slow
def test_partition_agrees_with_the_independent_grid() -> None:
    """``scripts/verify_dp_field_intervals.py`` on the five fixtures and the A60-10 members (~2.5 min on an M2 Max)."""
    spec = importlib.util.spec_from_file_location("verify_dp_field_intervals", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    paths = [(faces, 20000) for faces in FIXTURES] + [(faces, lattice_n) for faces, (lattice_n, _) in A60_10.items()]
    for faces, lattice_n in paths:
        for lower, upper, predicted, measured in module.verify(faces, N, 1201, lattice_n):
            assert predicted == measured, (faces, lower, upper)

