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
from lumice_integral.focusing import classify, field_onsets
from lumice_integral.geometry import HexPrism
from lumice_integral.pose_density import build_pose_density

N = 1.31
FIXTURES = ((3, 5), (1, 3), (3, 1, 6), (1, 3, 2), (3, 5, 6, 7, 3), (1, 2, 1))
SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "verify_dp_field_intervals.py"
# The beta scenario crystal of explore ``u-space-dissolution-probe`` at Lumice's ``n(550)``: the one
# fixture whose slab crease crosses ``U_P`` (22.76% of its sampling), the three-check gate's positive
# case (task ``dp-slab-partition-completion``; values from the task's probe_partition.json).
BETA = HexPrism.from_lumice(height=3.0, face_distance=[2.0, 1.0, 1.0, 2.0, 1.0, 1.0])
PATH_BETA = (4, 8, 7, 5)
N_BETA = 1.3110129

# (lower deg, upper deg, n_components, n_closed, n_open), checked by scripts/verify_dp_field_intervals.py
EXPECTED = {
    (3, 5): [(21.8393, 42.99086, 1, 1, 0), (42.99086, 43.46516, 4, 0, 4), (43.46516, 50.06262, 2, 0, 2)],
    (1, 3): [(45.73342, 57.80363, 1, 1, 0), (57.80363, 73.50689, 2, 0, 2)],
    (3, 1, 6): [(0.0, 180.0, 1, 0, 1)],
    (1, 3, 2): [(0.0, 180.0, 1, 0, 1)],
    # the crease circle is dU_P itself (a constant loop, no corners): every level set a closed
    # circle around the axis cone point, none of them touching the boundary
    (1, 2, 1): [(0.0, 180.0, 1, 1, 0)],
    (3, 5, 6, 7, 3): [(0.0, 98.16070, 2, 0, 2), (98.16070, 180.0, 1, 1, 0)],
}
# The A60-10 members (two internal reflections, no slab): the loop maximum 141.8393 = 120 + 21.8393 degrees sits
# on a grazing internal-reflection piece, where the smooth extension of D_P has a saddle (not interior).  The
# U_P of 3-5-6-7 has a neck the default lattice splits in two; the chart audit of task dp-thin-neck-topology
# corrects that artefact, and this fixture stays on the 50000-point lattice that resolves the neck itself,
# so its counts come from the lattice and the audit does not run.
A60_10 = {
    (3, 5, 6, 7): (50000, [(50.06262, 141.83930, 2, 0, 2), (141.83930, 163.46516, 1, 0, 1)]),
    (3, 4, 5, 7): (20000, [(50.06262, 141.83930, 2, 0, 2), (141.83930, 163.46516, 1, 0, 1)]),
}


@pytest.fixture(scope="module")
def fields() -> dict[tuple[int, ...], DPField]:
    return {faces: DPField.build(canonical_crystal(), faces, N) for faces in FIXTURES}


@pytest.fixture(scope="module")
def beta_field() -> DPField:
    return DPField.build(BETA, PATH_BETA, N_BETA)


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


def test_4_8_7_5_slab_crease_partition(beta_field) -> None:
    """The beta crystal's slab crease crosses ``U_P`` and still partitions (task ``dp-slab-partition-completion``).

    The crease (the c-axis great circle, ``D_P = 120 deg`` along it) enters
    ``U_P`` as one 1.43-rad arc; its two transversal ends are the walk's two
    *strict* blade maxima, the four corners (``entry`` / ``internal_i`` /
    ``exit_snell`` triple gates) sit at the 50.1617 deg closure value, and the
    two axis points on ``dU_P`` are the loop minima at ``D = 0``.  The pinned
    partition equals the task's Step 1 probe draft (the generic mechanism with
    the escape branch disabled) and the independent chart grid of
    ``scripts/verify_dp_field_intervals.py`` at 1201 points per side (the
    slow test below); C05's display support ``[0.54, 120] deg`` is bracketed.
    """
    field = beta_field
    assert field.domain_topology.is_disk
    assert field.interior_critical_points == ()
    fold = field.degenerate_fold
    assert fold.circle_interior_fraction == pytest.approx(0.22764, abs=2e-4)
    assert [where for _, where in fold.axis_points] == ["boundary", "boundary"]
    assert (fold.crease_interior_arcs, fold.crease_closed_ridge, fold.crease_touching_arc) == (1, False, False)
    expected = [(0.0, 50.161740, 2, 0, 2), (50.161740, 120.0, 2, 0, 2)]
    partition = field.interval_partition()
    assert [iv[2:] for iv in partition] == [e[2:] for e in expected]
    for interval, e in zip(partition, expected):
        assert np.degrees(interval.lower) == pytest.approx(e[0], abs=5e-6)
        assert np.degrees(interval.upper) == pytest.approx(e[1], abs=5e-6)
        assert interval.n_components == interval.n_closed + interval.n_open
    # C05 anchor: the display support [0.54, 120] deg lies inside the partition's range
    assert np.degrees(partition[0].lower) <= 0.54 < np.degrees(partition[-1].upper)
    maxima = [p for p in field.boundary_critical_points if p.kind == "maximum"]
    assert len(maxima) == 2 and all(p.strict for p in maxima)  # one per crease-arc end, not plateaus
    assert all(p.value == pytest.approx(partition[-1].upper, abs=1e-15) for p in maxima)
    assert np.degrees(partition[-1].upper) == pytest.approx(120.0, abs=1e-9)


def test_4_8_7_5_classifies_jacobian_at_the_blade(beta_field) -> None:
    """``focusing`` on the partitioned slab: ``jacobian``, the vanishing-gradient pair at the 120 deg blade.

    The 120 deg boundary pair carries the vanishing gradient (the crease is a
    critical ridge: ``grad D_P = 0`` at the blade maxima), the 0 deg pair are
    cone-point boundary extrema with finite gradient, and the closed-form
    ``slab_axis`` onsets sit at exactly 0 (explore insight 11).
    """
    field = beta_field
    label = classify(BETA, PATH_BETA, build_pose_density("random"), N_BETA, field=field)
    assert label.mechanism == "jacobian"
    assert label.jacobian_focusing
    onsets = {(o.source, o.profile): o for o in field_onsets(field)}
    blade = onsets[("boundary_extremum", "degenerate")]
    assert np.degrees(blade.value) == pytest.approx(120.0, abs=1e-9)
    assert blade.multiplicity == 2
    axis = onsets[("slab_axis", "cone_point")]
    assert axis.value == 0.0 and axis.multiplicity == 2
    zero = onsets[("boundary_extremum", "boundary_onset")]
    assert np.degrees(zero.value) == pytest.approx(0.0, abs=1e-9) and zero.multiplicity == 2
    assert np.degrees(onsets[("slab_circle", "inverse_sqrt_divergence")].value) == pytest.approx(120.0, abs=1e-9)


def test_3_5_6_7_at_lumice_n550_walks_the_full_stack() -> None:
    """The exit-Snell closure convention end to end on ``3-5-6-7`` at Lumice's ``n(550)`` (1.3110129).

    Pre-convention the boundary walk raised at the triple-gate corner (the
    pre-fix certification run captured ``D_P is not finite`` at exactly that
    point, evidence/ in the task directory).  Now every layer completes with
    finite values: the walk, the weight kinks (the internal-1 onset coincides
    with the exit-Snell boundary piece), the partition and the focusing
    onsets; the closure limit ``50.16174`` deg is the corner value, the loop
    minimum, the first interval's lower end and the kink range's minimum.
    Component counts are not asserted: at this ``n`` the thin neck of
    ``U_P`` splits on the lattice in a non-monotone way (50000: 2
    components, 100000: 1, 200000: 2 again) -- the k-NN resolution artifact
    of task ``dp-thin-neck-topology`` -- so the lattice density is merely
    chosen where the disk check passes.  A corner's ``gradient_batch`` is
    ``NaN`` by design (an exit-TIR end: profile ``boundary_onset``, norm
    ``inf``, ``focusing.field_onsets``).
    """
    field = DPField.build(canonical_crystal(), (3, 5, 6, 7), 1.3110129, lattice_n=100000)
    assert field.domain_topology.is_disk
    assert field.domain_topology.grid_audit is None  # the lattice resolves the neck here: no audit runs
    assert field.interior_critical_points == ()
    assert np.isfinite(field.boundary.values).all()
    for corner in field.corners:
        assert np.degrees(corner.value) == pytest.approx(50.16174445450327, abs=2e-5)
    kink_values = np.concatenate([curve.values for curve in field.weight_kinks if curve.arcs])
    assert np.isfinite(kink_values).all()
    assert np.degrees(kink_values.min()) == pytest.approx(50.16174445450327, abs=2e-5)
    partition = field.interval_partition()
    assert len(partition) == 2
    assert np.degrees(partition[0].lower) == pytest.approx(50.16174445450327, abs=2e-5)
    assert np.degrees(partition[0].upper) == pytest.approx(141.916126, abs=1e-3)
    assert np.degrees(partition[1].upper) == pytest.approx(163.545132, abs=1e-3)
    onsets = field_onsets(field)
    assert all(np.isfinite(onset.value) for onset in onsets)
    (corner_onset,) = [onset for onset in onsets if onset.source == "corner"]
    assert corner_onset.profile == "boundary_onset" and corner_onset.gradient_norm == float("inf")
    assert corner_onset.multiplicity == 2
    assert np.degrees(corner_onset.value) == pytest.approx(50.16174445450327, abs=2e-5)


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
    A60-10 members do, :data:`A60_10`); near 153 degrees ``3-5-6-7-3`` keeps
    a corner at the internal TIR onset, not a critical value
    (``ch10_verdicts.tir_onset_maximum``).  Before partial internal
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
    """A patched ``circle_interior_fraction`` on a grazing crease: the gate's contradiction form.

    ``3-1-6``'s real crease sampling holds no interior arc (its crease is a
    grazing boundary -- a 3.14-rad coincident arc, the tangency evidence), so
    the patched fraction is caught by the first check: evidence contradicting
    the premise.  One of the four fail-closed forms of the slab-crease gate;
    the others are pinned synthetically below.
    """
    field = fields[(3, 1, 6)]
    crease = dataclasses.replace(field.degenerate_fold, circle_interior_fraction=0.1)
    with pytest.raises(TopologyEscape, match="crease"):
        C.interval_partition(field.faces, N, (), crease, field.boundary, field.domain_topology, field.slab)


# ---- the four fail-closed forms of the slab-crease gate (task dp-slab-partition-completion) ------------


def _beta_partition_args(beta_field: DPField, crease) -> tuple:
    """``interval_partition`` args of the beta field: its own everything, a hand-built disk topology, the given crease."""
    return (beta_field.faces, beta_field.index, (), crease, beta_field.boundary, C.DomainTopology(20000, 1, 1), beta_field.slab)


def test_slab_gate_escape_contradiction(beta_field) -> None:
    """Form 0: a fraction claiming interior arcs the crease sampling does not hold."""
    crease = dataclasses.replace(beta_field.degenerate_fold, crease_interior_arcs=0)
    with pytest.raises(TopologyEscape, match="holds no interior arc"):
        C.interval_partition(*_beta_partition_args(beta_field, crease), crystal=BETA)


def test_slab_gate_escape_blade_not_carried(beta_field, fields) -> None:
    """Form (a): a loop with no strict local maximum at the blade value (here a foreign, non-slab loop)."""
    foreign = fields[(3, 5)].boundary  # maxima at 43.0 / 50.1 deg: nothing near the 120 deg blade
    with pytest.raises(TopologyEscape, match="not carried by the boundary walk as a strict local maximum"):
        C.interval_partition(
            beta_field.faces, beta_field.index, (), beta_field.degenerate_fold, foreign,
            C.DomainTopology(20000, 1, 1), beta_field.slab, crystal=BETA,
        )


def test_slab_gate_escape_tangency(beta_field) -> None:
    """Form 2a: the crease hugging ``dU_P`` over an arc (a tangency or coincidence, not a crossing)."""
    crease = dataclasses.replace(beta_field.degenerate_fold, crease_touching_arc=True)
    with pytest.raises(TopologyEscape, match="non-transversal contact"):
        C.interval_partition(*_beta_partition_args(beta_field, crease), crystal=BETA)


def test_slab_gate_escape_closed_ridge(beta_field) -> None:
    """Form 2b: an interior crease arc that never reaches ``dU_P`` (the level loops around it are not the walk's to count)."""
    crease = dataclasses.replace(beta_field.degenerate_fold, crease_closed_ridge=True)
    with pytest.raises(TopologyEscape, match="closed ridge"):
        C.interval_partition(*_beta_partition_args(beta_field, crease), crystal=BETA)


def test_escape_hatch_minimum_that_never_reaches_the_boundary_first(fields) -> None:
    """An interior minimum above the loop minimum cannot be the first to touch ``dU_P``: refused, not guessed."""
    field = fields[(3, 5)]
    (minimum,) = field.interior_critical_points
    high = dataclasses.replace(minimum, value=np.radians(45.0))
    with pytest.raises(TopologyEscape, match="not shown to reach"):
        C.interval_partition(field.faces, N, (high,), None, field.boundary, field.domain_topology, None)


# ---- the chart-grid audit on synthetic masks (task ``dp-thin-neck-topology``; no physical field here) --


def _chart_geometry(grid: int) -> tuple[np.ndarray, np.ndarray]:
    """The chart's index coordinates and off-chart mask of a ``grid``-point side (:func:`C._chart_grid`)."""
    m = (grid - 1) // 2
    i, j = np.meshgrid(np.arange(-m, m + 1), np.arange(-m, m + 1), indexing="ij")
    off_chart = i**2 + j**2 >= m**2
    return i, j, off_chart


def _two_islands(i: np.ndarray, j: np.ndarray) -> np.ndarray:
    """Two valid blobs a 1-node corridor could join; on an 81-point side they sit clear of the rim."""
    return ((i + 22) ** 2 + j**2 <= 100) | ((i - 22) ** 2 + j**2 <= 100)


def test_chart_counts_dumbbell_and_two_islands() -> None:
    """A neck the coarse view splits and the corridor view joins; a genuine two-island mask."""
    i, j, off_chart = _chart_geometry(81)
    islands = _two_islands(i, j)
    joined = islands | ((j == 0) & (np.abs(i) <= 13))
    assert C._chart_component_counts(islands, off_chart) == (2, 1)
    assert C._chart_component_counts(joined, off_chart) == (1, 1)


def test_chart_counts_complement_islands_and_rim_merge() -> None:
    """An interior invalid hole is a second complement island; an invalid band reaching the rim merges into one."""
    i, j, off_chart = _chart_geometry(81)
    on_chart = ~off_chart
    holed = on_chart & ~(i**2 + j**2 <= 9)
    rim_band = on_chart & (i < 20)
    assert C._chart_component_counts(holed, off_chart) == (1, 2)
    assert C._chart_component_counts(rim_band, off_chart) == (1, 1)


def test_chart_counts_valid_on_the_rim_voids_the_counts() -> None:
    """A valid node on the pushed rim breaches the chart premise: ``None``, not counts merged through it."""
    i, j, off_chart = _chart_geometry(81)
    breached = _two_islands(i, j)
    breached[0, 0] = True  # a grid corner is off-chart
    assert off_chart[0, 0]
    assert C._chart_component_counts(breached, off_chart) is None


def test_audit_verdict_three_states_and_rollup() -> None:
    """Grids agreeing against the lattice correct, agreeing with it confirm, disagreeing establish nothing."""
    assert C._audit_verdict((1, 1), (1, 1), 2, 1) == "corrected"
    assert C._audit_verdict((2, 2), (1, 1), 2, 1) == "confirmed"
    assert C._audit_verdict((1, 2), (1, 1), 2, 1) == "unconverged"
    assert C._audit_verdict((-1, 1), (1, 1), 2, 1) == "unconverged"
    # one count corrected, the other confirmed (or not converged): the roll-up takes the worst
    assert C._audit_verdict((1, 1), (2, 2), 2, 1) == "corrected"
    assert C._audit_verdict((2, 2), (1, 2), 2, 1) == "unconverged"


def test_chart_audit_record_keeps_the_lattice_counts_visible() -> None:
    """``ChartAudit`` carries the audited lattice counts next to the verdict (no silent correction)."""
    audit = C.ChartAudit((801, 1601), (1, 1), (1, 1), 2, 1, "corrected")
    assert (audit.lattice_domain_count, audit.lattice_complement_count) == (2, 1)
    assert audit.grids == (801, 1601)


@pytest.mark.parametrize("faces", FIXTURES)
def test_healthy_paths_pay_no_audit(fields, faces) -> None:
    """A singular lattice count never triggers the chart audit: ``grid_audit is None``, no chart cost."""
    assert fields[faces].domain_topology.grid_audit is None


def test_3_5_6_7_partitions_on_the_default_lattice() -> None:
    """The chart audit corrects the thin-neck artefact: a disk on 20000 points, the A60-10 partition.

    The lattice says (2, 1); the audit's grids (801, 1601) both say (1, 1),
    so the adjudicated topology is a disk and the partition runs with the
    same intervals the 50000-point fixture pins.  The interval values are
    the default-lattice run of ``scripts/verify_dp_field_intervals.py
    --path 3 5 6 7 --grid 801`` (every interval predicted == grid there,
    the M3 evidence of task ``dp-thin-neck-topology``).
    """
    field = DPField.build(canonical_crystal(), (3, 5, 6, 7), N)
    topology = field.domain_topology
    assert topology.grid_audit.verdict == "corrected"
    assert topology.grid_audit.lattice_domain_count == 2
    assert (topology.domain_components, topology.complement_components) == (1, 1)
    expected = [(50.06262, 141.83930, 2, 0, 2), (141.83930, 163.46516, 1, 0, 1)]
    partition = field.interval_partition()
    assert [iv[2:] for iv in partition] == [e[2:] for e in expected]
    for interval, e in zip(partition, expected):
        assert np.degrees(interval.lower) == pytest.approx(e[0], abs=5e-6)
        assert np.degrees(interval.upper) == pytest.approx(e[1], abs=5e-6)


def test_empty_ladder_is_the_rollback_switch(monkeypatch) -> None:
    """``AUDIT_LADDER = ()`` turns the audit off: the plural path raises the legacy escape, ``grid_audit is None``.

    The trigger gate must short-circuit on the empty ladder instead of entering ``chart_audit``
    (whose roll-up would crash on the empty counts): the rollback path is the old "on the
    N-point lattice" escape with no audit record -- the degraded-to-status-quo semantics the
    constant's comment and the plan's rollback clause promise.
    """
    monkeypatch.setattr(C, "AUDIT_LADDER", ())
    field = DPField.build(canonical_crystal(), (3, 5, 6, 7), N)
    topology = field.domain_topology
    assert topology.grid_audit is None
    assert not topology.is_disk
    with pytest.raises(TopologyEscape, match=r"U_P is not a disk on the 20000-point lattice: 2 component\(s\), complement 1"):
        field.interval_partition()


def test_escape_text_reports_the_audit_branches(fields) -> None:
    """The disk escape says what the two chains said: agreed, corrected but still not a disk, or not converged."""
    field = fields[(3, 5)]
    args = (field.faces, N, field.interior_critical_points, None, field.boundary)
    confirmed = C.DomainTopology(20000, 2, 1, grid_audit=C.ChartAudit((801, 1601), (2, 2), (1, 1), 2, 1, "confirmed"))
    with pytest.raises(TopologyEscape, match=r"2 component\(s\), complement 1 \(lattice 20000 and chart grids \(801, 1601\) agree\)"):
        C.interval_partition(*args, confirmed, None)
    corrected = C.DomainTopology(20000, 1, 2, grid_audit=C.ChartAudit((801, 1601), (1, 1), (2, 2), 2, 1, "corrected"))
    with pytest.raises(TopologyEscape, match=r"1 component\(s\), complement 2 \(chart grids \(801, 1601\) agree, correcting the lattice 20000 counts 2/1\)"):
        C.interval_partition(*args, corrected, None)
    unconverged = C.DomainTopology(20000, 2, 1, grid_audit=C.ChartAudit((801, 1601), (1, 2), (1, 1), 2, 1, "unconverged"))
    with pytest.raises(TopologyEscape, match=r"lattice 20000 says 2/1, chart grids \(801, 1601\) say \(1, 2\)/\(1, 1\); counts are not resolution-converged, the topology is not established"):
        C.interval_partition(*args, unconverged, None)


@pytest.mark.slow
def test_partition_agrees_with_the_independent_grid() -> None:
    """``scripts/verify_dp_field_intervals.py`` on the fixtures, the A60-10 members, 3-5-6-7's audited default lattice, and the beta slab (~3 min on an M2 Max).

    The ``(3, 5, 6, 7)`` row at the default 20000 points runs the partition
    through the chart audit's correction; this script's own independent grid
    is the second chain of that audit, so its agreement here is the standing
    cross-check of the deliberately dual implementations (module docstring,
    ``certificate.py``).  The beta row verifies the slab-crease gate's one
    positive fixture end to end (task ``dp-slab-partition-completion``).
    """
    spec = importlib.util.spec_from_file_location("verify_dp_field_intervals", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    paths = [(faces, 20000) for faces in FIXTURES] + [(faces, lattice_n) for faces, (lattice_n, _) in A60_10.items()]
    paths.append(((3, 5, 6, 7), 20000))
    for faces, lattice_n in paths:
        for lower, upper, predicted, measured in module.verify(faces, N, 1201, lattice_n):
            assert predicted == measured, (faces, lower, upper)
    for lower, upper, predicted, measured in module.verify(PATH_BETA, N_BETA, 1201, crystal=BETA):
        assert predicted == measured, (PATH_BETA, lower, upper)

