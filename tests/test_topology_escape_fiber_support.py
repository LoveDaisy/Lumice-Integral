"""Fiber components against the store's finite-crystal support on the paths the interval partition rejects.

``explore-level-set-vs-finite-crystal-gate`` (experiments.md #2, #3): on the
canonical prism ``DPField.interval_partition()`` raises ``TopologyEscape`` for
3-1-5-7 and 3-5-7 (the slab crease runs through ``U_P``), so the contour layer
has no component counts to compare fiber by fiber.  The substitute check is
the fiber layer against the store: the store keeps only ``w = A T > 0``
events, so its deviation range is the finite crystal's support, and
:func:`.discovery.discover_components` must find no component outside it and
the recorded components, all converged, inside it.  This is a consistency
check between the fiber layer and the store, not a one-to-one contour check.

3-5-6-7's ``U_P`` has a neck thinner than the default 20000-point lattice
resolves (task ``dp-thin-neck-topology``): the k-NN count says two
components and the chart-grid audit corrects it to one, so the path
partitions on the default lattice too and its contour layer is compared
one to one: below the support its two open arcs are ``gated_out`` where the
fiber layer finds nothing, inside it both layers have two arcs.

Targets are ``cos d prop + sin d perp`` (``perp = prop x z``), ``N = 1e6``
stores, 0.2 deg bands; every delta is at least ``SUPPORT_MARGIN_DEG`` from a support
edge (the closest, 3-1-5-7 at 80 deg, is 8.75 deg below 88.75 deg).
Component counts are integers and pinned exactly; arclengths are not pinned.
"""

from __future__ import annotations

import numpy as np
import pytest

from lumice_integral import contour_quadrature as cq
from lumice_integral.canonical_scene import (
    CANONICAL_REFRACTIVE_INDEX,
    canonical_crystal,
    canonical_incident_direction,
    canonical_sun_direction,
)
from lumice_integral.contour import extract_level_sets
from lumice_integral.discovery import discover_components
from lumice_integral.dp_field import DPField, TopologyEscape
from lumice_integral.pose_density import build_pose_density
from lumice_integral.s2_store import StoreSeeds, build_event_store

INDEX = CANONICAL_REFRACTIVE_INDEX
STORE_N = 1_000_000
BAND_HALF_WIDTH_DEG = 0.2

# path -> ((delta_deg, component kinds), ...): the first delta is below the store's support, the others inside it
FIBER_COMPONENTS: dict[tuple[int, ...], tuple[tuple[float, tuple[str, ...]], ...]] = {
    (3, 5, 6, 7): ((70.0, ()), (100.0, ("arc", "arc")), (130.0, ("arc", "arc"))),
    (3, 1, 5, 7): ((80.0, ()), (110.0, ("arc",)), (150.0, ("arc", "arc"))),
    (3, 5, 7): ((45.0, ()), (90.0, ("arc",)), (140.0, ("closed",))),
}
# the store's w > 0 deviation range in degrees (experiments.md #2), to 0.01 deg
STORE_SUPPORT_DEG = {(3, 5, 6, 7): (81.17, 163.14), (3, 1, 5, 7): (88.75, 180.00), (3, 5, 7): (55.21, 179.87)}
ESCAPE_MESSAGE = {
    (3, 1, 5, 7): "the slab crease",
    (3, 5, 7): "the slab crease",
}
NECK_LATTICE_N = 50000
SUPPORT_MARGIN_DEG = 5.0

CASES = [(faces, delta, kinds) for faces, rows in FIBER_COMPONENTS.items() for delta, kinds in rows]


def _target(delta_deg: float) -> np.ndarray:
    prop = canonical_incident_direction()
    perp = np.cross(prop, [0.0, 0.0, 1.0])
    perp /= np.linalg.norm(perp)
    delta = np.radians(delta_deg)
    return np.cos(delta) * prop + np.sin(delta) * perp


@pytest.fixture(scope="module")
def stores() -> dict:
    """One ``N = 1e6`` store per path (~2-4 s each)."""
    return {faces: build_event_store(canonical_crystal(), INDEX, [faces], STORE_N, run_checks=False) for faces in FIBER_COMPONENTS}


@pytest.mark.parametrize("faces", list(FIBER_COMPONENTS), ids=lambda f: "-".join(map(str, f)))
def test_deltas_bracket_the_store_support(stores, faces) -> None:
    """The fixture deltas lie outside / inside the store's own ``w > 0`` range, ``SUPPORT_MARGIN_DEG`` to spare."""
    deviations = np.degrees(stores[faces].events.D)
    low, high = deviations.min(), deviations.max()
    np.testing.assert_allclose([low, high], STORE_SUPPORT_DEG[faces], atol=0.01)
    assert np.all(stores[faces].events.w > 0.0)
    for delta, kinds in FIBER_COMPONENTS[faces]:
        inside = low + SUPPORT_MARGIN_DEG <= delta <= high - SUPPORT_MARGIN_DEG
        outside = delta <= low - SUPPORT_MARGIN_DEG or delta >= high + SUPPORT_MARGIN_DEG
        assert (inside, outside) == ((True, False) if kinds else (False, True)), delta


@pytest.mark.parametrize(("faces", "delta", "kinds"), CASES, ids=[f"{"-".join(map(str, f))}@{d:g}" for f, d, _ in CASES])
def test_fiber_components_follow_the_store_support(stores, faces, delta, kinds) -> None:
    """None outside the support (an empty band), the recorded ones inside it, all converged."""
    seeds = StoreSeeds(stores[faces], faces, canonical_sun_direction())
    result = discover_components(_target(delta), seeds, band_half_width_deg=BAND_HALF_WIDTH_DEG)
    assert tuple(c.kind for c in result.components) == kinds
    assert result.component_count == len(kinds)
    assert len(result.incomplete) == 0 and result.completeness == "complete"
    if not kinds:
        assert result.pool_count == 0
    else:
        assert result.pool_count > 0


@pytest.mark.parametrize("faces", [(3, 1, 5, 7), (3, 5, 7)], ids=lambda f: "-".join(map(str, f)))
def test_interval_partition_escapes_on_the_default_lattice(faces) -> None:
    """Fail-closed on the default lattice; the reason the substitute check exists."""
    with pytest.raises(TopologyEscape, match=ESCAPE_MESSAGE[faces]):
        DPField.build(canonical_crystal(), faces, INDEX).interval_partition()


def test_3_5_6_7_default_lattice_partitions_through_the_chart_audit() -> None:
    """The neck the 20000-point lattice splits is corrected to one component by the chart audit.

    The lattice k-NN count (2 components) is a resolution artefact of the
    neck (< 1e-3 rad); the audit's two chart grids agree on one component
    and one complement, the adjudicated topology is a disk and the partition
    runs (task ``dp-thin-neck-topology``, the interval values are pinned in
    ``test_dp_field_certificate.py``).
    """
    field = DPField.build(canonical_crystal(), (3, 5, 6, 7), INDEX)
    topology = field.domain_topology
    audit = topology.grid_audit
    assert audit.lattice_domain_count == 2 and audit.lattice_complement_count == 1
    assert audit.verdict == "corrected"
    assert audit.domain_counts == audit.complement_counts == (1, 1)
    assert (topology.domain_components, topology.complement_components) == (1, 1)
    assert [(iv.n_components, iv.n_closed, iv.n_open) for iv in field.interval_partition()] == [(2, 0, 2), (1, 0, 1)]


@pytest.mark.parametrize("faces", [(3, 1, 5, 7), (3, 5, 7)], ids=lambda f: "-".join(map(str, f)))
def test_slab_crease_escape_is_not_a_lattice_artefact(faces) -> None:
    """3-1-5-7 and 3-5-7 still escape on the finer lattice: no contour layer to compare with."""
    with pytest.raises(TopologyEscape, match="the slab crease"):
        DPField.build(canonical_crystal(), faces, INDEX, lattice_n=NECK_LATTICE_N).interval_partition()


def test_3_5_6_7_contour_layer_matches_the_fiber_layer(stores) -> None:
    """On 50000 points 3-5-6-7 partitions; its level sets agree with the fibers, the gate included.

    Below the support the two open arcs exist on ``U_P`` but every point is at
    ``w = 0`` (``gated_out``, value exactly 0) where the fiber layer finds no
    component; inside it both layers have two open arcs and the value is live.
    """
    faces = (3, 5, 6, 7)
    field = DPField.build(canonical_crystal(), faces, INDEX, lattice_n=NECK_LATTICE_N)
    partition = field.interval_partition()
    rows = FIBER_COMPONENTS[faces]
    deltas = np.radians([delta for delta, _ in rows])
    for delta, kinds in rows:
        (interval,) = [i for i in partition if i.lower < np.radians(delta) < i.upper]
        assert (interval.n_components, interval.n_closed, interval.n_open) == (2, 0, 2)
    level_sets = extract_level_sets(field, deltas, stores[faces])
    geometry = cq.LevelSetGeometry.build(field, level_sets, cq.QuadratureOptions(relative_tolerance=1e-6))
    sun = canonical_sun_direction()
    results = geometry.integrate(sun, [_target(delta) for delta, _ in rows], build_pose_density("random"))
    for (delta, kinds), level_set, result in zip(rows, level_sets, results, strict=True):
        assert [c.closed for c in level_set.components] == [False, False], delta
        assert result.status == "integrated", delta
        if kinds:
            assert len(kinds) == len(level_set.components) and all(k == "arc" for k in kinds)
            assert not result.gated_out and result.value > 0.0, delta
        else:
            assert result.gated_out and result.value == 0.0, delta
