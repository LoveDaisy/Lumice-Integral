"""The SUN^2 band-sum renderer (``lumice_integral.band_sum``, roadmap section 4.2).

Hand-built events pin the estimator's closed form; small real stores pin the
scatter form of the window renderer against the per-pixel gather (the oracle), the
class aggregation (one store per ``D6h`` orbit, proper and improper
transports, against every ``Phi`` group's own store on the ``g``-moved
lattice, the same points), the per-event ``K`` / ``K_eff``, the rank-0
hand-off, the store plan, the window driver and the ``strip_io``-compatible
output.
"""

from __future__ import annotations

import csv
import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from lumice_integral import s2_store
from lumice_integral.band_sum import (
    K_EFF_SEMANTICS,
    PIXEL_CSV_COLUMNS,
    BandSumScene,
    ScatterSums,
    StoreGroup,
    Transport,
    band_contributions,
    band_poses,
    band_sum_estimate,
    band_sum_pixel,
    class_band_sum_pixel,
    kish_k_eff,
    pixel_band,
    pixel_bands,
    render_band_sum_window,
    scatter_results,
    scatter_store,
    single_path_class,
    store_plan,
    write_band_sum_strip,
)
from lumice_integral.canonical_scene import (
    CANONICAL_REFRACTIVE_INDEX,
    CANONICAL_RENDER,
    canonical_crystal,
    canonical_pose_density,
    canonical_sun_direction,
)
from lumice_integral.camera import sun_direction
from lumice_integral.optics import path_id_of
from lumice_integral.path_class import build_path_class, pixel_solid_angle
from lumice_integral.pose_density import build_pose_density
from lumice_integral.pose_density_provenance import pose_density_provenance
from lumice_integral.s2_store import S2Events, build_event_store, store_lattice
from lumice_integral.strip_io import Window, read_strip
from lumice_integral.strip_pixel import STATUS_HAS_COMPONENT, STATUS_RENDERED

SUN = canonical_sun_direction()  # s_hat, toward the sun
N_SMALL = 100_000
UNIFORM = build_pose_density("random")
SUN_WINDOW = {"width": 21, "height": 21, "fov_deg": 6.0, "view": {"azimuth": 0.0, "elevation": 15.0}}


def identity_group(members) -> StoreGroup:
    return StoreGroup(tuple(members), (Transport(tuple(members), None),))


def rotated_lattice(g: np.ndarray, n: int):
    def sampler(first: int, stop: int):
        return np.einsum("ij,nj->ni", g, store_lattice(n, first, stop)), None

    return sampler


def build(members, n: int, g: np.ndarray | None = None) -> s2_store.S2EventStore:
    extra = {} if g is None else {"sampler": rotated_lattice(g, n), "sampling": f"Fibonacci lattice rotated by {g.round(12).tolist()}"}
    return build_event_store(
        canonical_crystal(), CANONICAL_REFRACTIVE_INDEX, members, n, run_checks=False, **extra
    )


# ------------------------------------------------------------- estimator
def test_band_sum_of_hand_built_events_matches_the_closed_form():
    """Events inside ``[delta_lo, delta_hi)`` count with ``w``; ``rho = 1`` makes the sum closed form."""
    row, column = 400, 126
    centre, delta, lo, hi = pixel_band(row, column, SUN)
    assert lo < delta < hi and np.isclose(hi - lo, 4.4e-4, rtol=0.1)
    d = np.array([lo - 1e-6, lo, 0.5 * (lo + hi), hi - 1e-9, hi, hi + 1e-3])
    w = np.array([10.0, 1.0, 2.0, 3.0, 20.0, 30.0])
    u = np.tile([-1.0, 0.0, 0.0], (6, 1))
    phi = np.stack([np.cos(d), np.sin(d), np.zeros(6)], axis=1)  # angle(-u, phi) = D (-u the incoming ray)
    events = {"D": d, "u": u, "phi": phi, "w": w}
    n = 1000
    value, delta_out, k, k_pos, k_eff, width = band_sum_pixel(events, SUN, UNIFORM, row, column, n)
    assert (k, k_pos, delta_out, width) == (3, 3, delta, hi - lo)
    assert value == pytest.approx(6.0 / (2.0 * np.pi * n * (hi - lo) * np.sin(delta)), rel=1e-15)
    assert k_eff == pytest.approx(36.0 / 14.0, rel=1e-15)
    (contribution,) = band_contributions(events, SUN, [UNIFORM], centre, lo, hi)
    np.testing.assert_array_equal(contribution, [1.0, 2.0, 3.0])
    # A non-uniform store multiplies by iw; the class path equals the single-path path.
    events["iw"] = np.full(6, 0.5)
    result = class_band_sum_pixel([(events, identity_group([(3, 5)]))], SUN, UNIFORM, row, column, n)
    assert result.value == pytest.approx(0.5 * value, rel=1e-15) and result.K == 3
    assert kish_k_eff(0.0, 0.0) == 0.0 and band_sum_estimate(0.0, n, 1e-3, 0.3) == 0.0


@pytest.fixture(scope="module")
def canonical_store():
    return build([(3, 5)], N_SMALL)


def test_band_sum_on_a_real_store_is_finite_and_equals_the_class_path(canonical_store):
    events = canonical_store.events.arrays()
    density = canonical_pose_density()
    values = []
    for row in range(0, 801, 40):
        single = band_sum_pixel(events, SUN, density, row, 126, N_SMALL)
        pooled = class_band_sum_pixel([(events, identity_group([(3, 5)]))], SUN, density, row, 126, N_SMALL)
        assert np.isfinite(single[0]) and single[0] >= 0.0
        assert (pooled.value, pooled.K, pooled.K_rho_pos, pooled.K_eff) == (single[0], single[2], single[3], single[4])
        values.append(single[0])
    assert max(values) > 0.0


# ------------------------------------------------------------ store plan
def test_store_plan_is_one_store_per_class():
    crystal = canonical_crystal()
    plan = store_plan(build_path_class(crystal, (3, 5)), crystal)
    assert len(plan) == 1 and plan[0].members == ((3, 5),) and len(plan[0].transports) == 12
    assert plan[0].transports[0].g is None
    assert all(np.linalg.det(t.g) > 0.0 for t in plan[0].transports[1:])
    assert len(store_plan(build_path_class(crystal, (3, 5)), crystal, transport=False)) == 12
    # 3-1-2-5: 24 members in Phi pairs; every Phi pair is reached by a proper element (through the partner
    # of a mirror-only member), so the proper element is kept.
    cls = build_path_class(crystal, (3, 1, 2, 5))
    plan = store_plan(cls, crystal)
    assert [len(g.members) for g in plan] == [2] and len(plan[0].transports) == 12
    assert all(np.linalg.det(t.g) > 0.0 for t in plan[0].transports[1:])
    assert sorted(plan[0].served_members) == sorted(cls.members)
    # 1-3-5-2: four Phi groups of six; two are reached only by mirrors (two stores before improper
    # elements were transported), now one store with two improper transports.
    cls = build_path_class(crystal, (1, 3, 5, 2))
    plan = store_plan(cls, crystal)
    assert [(len(g.members), len(g.transports)) for g in plan] == [(6, 4)]
    assert sorted(int(np.sign(np.linalg.det(t.g))) for t in plan[0].transports[1:]) == [-1, -1, 1]
    assert sorted(plan[0].served_members) == sorted(cls.members)
    assert len(store_plan(cls, crystal, transport=False)) == 4
    assert store_plan(build_path_class(crystal, (1, 3, 6, 2)), crystal) == ()  # rank 0
    assert single_path_class(crystal, (3, 5)).members == ((3, 5),)


@pytest.mark.parametrize("representative", [(3, 5), (1, 3, 5, 2)])
def test_class_value_equals_the_sum_over_every_groups_own_store(representative):
    """One store per class with its transports vs each ``Phi`` group's own store on the ``g``-moved lattice.

    Same points, so the totals agree to rounding (``1e-10``) and so do the
    per-event diagnostics of each group.  ``1-3-5-2`` has ``Phi`` groups of
    six members, two of them reached by mirrors only: the ``Phi`` sum and the
    improper transport (``L_g R g^T``) both enter.
    """
    crystal = canonical_crystal()
    cls = build_path_class(crystal, representative)
    plan = store_plan(cls, crystal)
    # A roll-locked density: rho(R g^T) != rho(R) for the rotations about the c axis, so the factor matters.
    density = build_pose_density("parry", zenith_std_deg=2.0, roll_std_deg=20.0)
    stores = [(build(group.members, N_SMALL).events.arrays(), group) for group in plan]
    own_by_group = {
        t.members: (build(t.members, N_SMALL, g=t.g).events.arrays(), identity_group(t.members))
        for group in plan
        for t in group.transports
    }
    own = list(own_by_group.values())
    # Per group, not only the class sum: the elements come in inverse pairs, so the class sum is blind to
    # g^T vs g (it swaps the two members' values); one transport at a time is not.
    per_group = [
        ([(events, StoreGroup(group.members, (t,)))], [own_by_group[t.members]])
        for events, group in stores
        for t in group.transports
    ]
    render = CANONICAL_RENDER if representative == (3, 5) else {"width": 41, "height": 41, "fov_deg": 150.0, "view": {"azimuth": 0.0, "elevation": 15.0}}
    pixels = [(r, 126) for r in range(0, 801, 25)] if representative == (3, 5) else [(r, c) for r in range(0, 41, 4) for c in range(0, 41, 4)]
    lit = 0
    for row, column in pixels:
        a = class_band_sum_pixel(stores, SUN, density, row, column, N_SMALL, render)
        b = class_band_sum_pixel(own, SUN, density, row, column, N_SMALL, render)
        # Distinct events: one store's band against the groups' own bands, which hold as many events each.
        assert a.K * len(own) == b.K and a.K_rho_pos <= b.K_rho_pos
        assert a.value == pytest.approx(b.value, rel=1e-10, abs=1e-300), (row, column)
        lit += a.value > 0.0
        for transported, mine in per_group:
            x = class_band_sum_pixel(transported, SUN, density, row, column, N_SMALL, render)
            y = class_band_sum_pixel(mine, SUN, density, row, column, N_SMALL, render)
            assert x.value == pytest.approx(y.value, rel=1e-10, abs=1e-300), (row, column, mine[0][1].members)
            assert (x.K, x.K_rho_pos) == (y.K, y.K_rho_pos)
            assert x.K_eff == pytest.approx(y.K_eff, rel=1e-8, abs=1e-300)
    assert lit >= 5


def test_k_eff_counts_distinct_events_not_transport_copies():
    """``K`` / ``K_eff`` of ``c_i = w_i sum_t rho(R_i^(t))``; the value is the per-transport sum, unchanged.

    Hand-built: three events, two transports.  With ``rho = 1`` both
    transports give the same contribution, so the pooled task
    ``band-sum-renderer`` count (6 samples, ``K_eff = 36 / 14 * 2``) would
    double the effective size; per event it is the one-transport size.
    """
    row, column = 400, 126
    _, delta, lo, hi = pixel_band(row, column, SUN)
    d = np.array([lo, 0.5 * (lo + hi), hi - 1e-9])
    w = np.array([1.0, 2.0, 3.0])
    u = np.tile([-1.0, 0.0, 0.0], (3, 1))  # angle(-u, phi) = D
    events = {"D": d, "u": u, "phi": np.stack([np.cos(d), np.sin(d), np.zeros(3)], axis=1), "w": w}
    mirror = np.diag([1.0, 1.0, -1.0])
    group = StoreGroup(((3, 5),), (Transport(((3, 5),), None), Transport(((3, 7),), mirror)))
    one = class_band_sum_pixel([(events, identity_group([(3, 5)]))], SUN, UNIFORM, row, column, 1000)
    two = class_band_sum_pixel([(events, group)], SUN, UNIFORM, row, column, 1000)
    assert (two.K, two.K_rho_pos) == (3, 3) and two.value == 2.0 * one.value
    assert two.total == 12.0 and two.square == 4.0 * 14.0
    assert two.K_eff == pytest.approx(36.0 / 14.0, rel=1e-15) == one.K_eff
    # A transport where rho vanishes adds nothing to c_i, so neither to K_rho_pos nor to K_eff.
    zero = type("Zero", (), {"evaluate_batch": staticmethod(lambda r: np.zeros(len(r)))})()
    assert class_band_sum_pixel([(events, group)], SUN, zero, row, column, 1000).K_rho_pos == 0


def test_class_value_is_unchanged_and_k_eff_is_the_single_member_scale(canonical_store):
    """Class ``[3,5]`` with a plate density: the value is the task ``band-sum-renderer`` sum bit for bit.

    The task ``band-sum-renderer`` aggregation is rebuilt literally here
    (every transport's contributions pooled, pose factor ``R g^T``).  Six of
    the twelve transports give ``rho(R g^T) = rho(R)`` for a plate density,
    so the pooled Kish size was six times one member's; per event those six
    only scale ``c_i``, which the Kish size ignores: one member's size.
    """
    crystal = canonical_crystal()
    (group,) = store_plan(build_path_class(crystal, (3, 5)), crystal)
    events = canonical_store.events.arrays()
    density = build_pose_density("plate", zenith_std_deg=1.0)
    render = {"width": 81, "height": 41, "fov_deg": 60.0, "view": {"azimuth": 0.0, "elevation": 15.0}}
    ratios = []
    for row, column in [(r, c) for r in range(0, 41, 5) for c in range(0, 81, 5)]:
        new = class_band_sum_pixel([(events, group)], SUN, density, row, column, N_SMALL, render)
        centre, delta, lo, hi = pixel_band(row, column, SUN, render)
        rotations, weight = band_poses(events, SUN, centre, lo, hi)
        total, square = 0.0, 0.0
        for t in group.transports:
            contribution = weight * density.evaluate_batch(rotations if t.g is None else rotations @ t.g.T)
            total += float(np.sum(contribution))
            square += float(np.sum(contribution**2))
        old = band_sum_estimate(total, N_SMALL, hi - lo, delta) if total != 0.0 else 0.0
        assert new.value == old and new.total == total
        if new.value > 0.0:
            single = class_band_sum_pixel([(events, identity_group([(3, 5)]))], SUN, density, row, column, N_SMALL, render)
            if single.K_eff > 20.0:
                ratios.append((kish_k_eff(total, square) / single.K_eff, new.K_eff / single.K_eff))
    assert len(ratios) >= 5
    pooled, per_event = np.median(np.array(ratios), axis=0)
    assert 5.5 < pooled < 6.5 and 0.9 < per_event < 1.5, ratios


def test_transport_and_store_group_compare_by_identity():
    """``eq=False``: ndarray fields would make the generated ``__eq__`` ambiguous and ``__hash__`` impossible."""
    g = np.diag([1.0, 1.0, -1.0])
    a, b = Transport(((3, 5),), g), Transport(((3, 5),), g.copy())
    assert a == a and a != b and len({a, b, a}) == 2
    group = StoreGroup(((3, 5),), (a, b))
    other = StoreGroup(((3, 5),), (a, b))
    assert group == group and group != other and len({group, other}) == 2
    assert group.served_members == ((3, 5), (3, 5))
    plan = store_plan(build_path_class(canonical_crystal(), (3, 5)), canonical_crystal())
    assert plan[0] in plan and plan[0].transports[3] in plan[0].transports


TASK14_WINDOWS = Path(__file__).resolve().parents[1] / "scratchpad/task-narrow-density-band-sum-probe/artifacts/profile_windows.json"


@pytest.mark.slow
def test_per_event_k_eff_is_the_iid_noise_of_a_class_band_sum():
    """The ruler of ``K_eff`` (a16): two i.i.d. stores, task 14's three profiles, ``N = 1e7`` each.

    With i.i.d. points the Kish size predicts the Monte Carlo noise, so
    ``z = (a - b) / sqrt(a^2 / K_a + b^2 / K_b)`` is standard normal when
    ``K_eff`` counts independent samples.  Per event it is (``z_rms``
    1.20 / 0.85 / 0.93 for plate / Parry / Lowitz on seeds 1, 2; 1.01 and
    0.97 on plate seeds 3, 4 and 5, 6); the pooled count of task
    ``band-sum-renderer`` gives plate ``z_rms`` 2.9: its six ``rho``-equal
    transports are one sample, not six.  ``scripts/regress_band_sum.py
    --stage k-eff`` is the report form of this test.
    """
    if not TASK14_WINDOWS.exists():
        pytest.skip(f"{TASK14_WINDOWS} not present (scratchpad artifact of task 14)")
    families = json.loads(TASK14_WINDOWS.read_text())["families"]
    crystal = canonical_crystal()
    (group,) = store_plan(build_path_class(crystal, (3, 5)), crystal)
    bands = [pixel_band(r, c, SUN, spec["render"])[2:] for spec in families.values() for r, c in spec["pixels"]]
    window = (min(lo for lo, _ in bands), max(hi for _, hi in bands))
    n = 10_000_000
    stores = [
        build_event_store(
            crystal, CANONICAL_REFRACTIVE_INDEX, [(3, 5)], n, sampler=s2_store.RandomSphereSampler(seed),
            sampling=f"i.i.d. seed {seed}", deviation_window=window, run_checks=False,
        ).events.arrays()
        for seed in (1, 2)
    ]
    z_rms = {}
    for family, spec in families.items():
        density = build_pose_density(family, **spec["density"])
        rows = []
        for row, column in spec["pixels"]:
            a, b = (class_band_sum_pixel([(events, group)], SUN, density, row, column, n, spec["render"]) for events in stores)
            pooled = []
            for events in stores:  # task band-sum-renderer's per_transport_sample count, literally
                centre, _, lo, hi = pixel_band(row, column, SUN, spec["render"])
                rotations, weight = band_poses(events, SUN, centre, lo, hi)
                c = [weight * density.evaluate_batch(rotations if t.g is None else rotations @ t.g.T) for t in group.transports]
                pooled.append(kish_k_eff(float(sum(x.sum() for x in c)), float(sum((x**2).sum() for x in c))))
            rows.append((a.value, b.value, a.K_eff, b.K_eff, *pooled))
        va, vb, ka, kb, pa, pb = (np.array(v) for v in zip(*rows))
        used = (va > 0.0) & (vb > 0.0) & (np.minimum(ka, kb) >= 30.0)
        assert used.sum() >= 100, family
        rms = lambda k1, k2: float(np.sqrt(np.mean((va - vb)[used] ** 2 / (va[used] ** 2 / k1[used] + vb[used] ** 2 / k2[used]))))  # noqa: E731
        z_rms[family] = (rms(ka, kb), rms(pa, pb))
    assert all(0.75 < per_event < 1.35 for per_event, _ in z_rms.values()), z_rms
    assert z_rms["plate"][1] > 2.0 and z_rms["parry"][1] > z_rms["parry"][0], z_rms


# -------------------------------------------------- scatter vs gather oracle
FAMILIES = {
    "random": {},
    "column": {"zenith_std_deg": 0.5},
    "plate": {"zenith_std_deg": 1.0},
    "parry": {"zenith_std_deg": 1.0, "roll_std_deg": 1.0},
    "lowitz": {"zenith_std_deg": 1.0, "roll_std_deg": 1.0},
}
WIDE_RENDER = {"width": 41, "height": 41, "fov_deg": 150.0, "view": {"azimuth": 0.0, "elevation": 15.0}}
PARHELIA_RENDER = {"width": 81, "height": 41, "fov_deg": 60.0, "view": {"azimuth": 0.0, "elevation": 15.0}}


def scatter(stores, density, pixels, sun=SUN, render=CANONICAL_RENDER, **kwargs):
    bands = pixel_bands(pixels, sun, render)
    sums = ScatterSums.zeros(len(bands))
    for events, group in stores:
        scatter_store(events, group, density, bands, sums, **kwargs)
    return scatter_results(bands, sums, N_SMALL)


def assert_scatter_equals_gather(scattered, gathered, *, rel=1e-12, k_eff_rel=1e-12):
    """Same band (bit for bit: delta, width, K, K_rho_pos), value and K_eff to round-off (summation order).

    The pose enters as ``W[2, :] . F_i[j, :]`` instead of the third row of
    ``W F_i^T``: the same three products, rounded differently.  A narrow
    density amplifies that: ``d log rho = (theta - mean) / std^2 d theta``,
    and the roll ``atan2(-e2, e1)`` of a nearly vertical c axis (Lowitz) has
    ``d psi ~ eps / sin(theta)``.  A value is a sum over many events and
    keeps ``1e-12``; a ``K_eff`` near 1 is one or two events' ratio and
    carries their error undiluted (``k_eff_rel``).
    """
    lit = 0
    for a, b in zip(scattered, gathered, strict=True):
        assert (a.row, a.column, a.delta, a.band_width_rad, a.K, a.K_rho_pos) == (
            b.row, b.column, b.delta, b.band_width_rad, b.K, b.K_rho_pos
        )
        assert (a.value == 0.0) == (b.value == 0.0)
        assert a.value == pytest.approx(b.value, rel=rel, abs=0.0), (a.row, a.column)
        assert a.K_eff == pytest.approx(b.K_eff, rel=k_eff_rel, abs=0.0), (a.row, a.column, a.K_eff)
        lit += b.value > 0.0
    return lit


@pytest.fixture(scope="module")
def class_stores():
    """Class ``[3,5]`` (12 proper transports) and ``[1,3,5,2]`` (improper ones), one store each; ``[3,5]`` per group too."""
    crystal = canonical_crystal()
    out = {}
    for representative in ((3, 5), (1, 3, 5, 2)):
        (group,) = store_plan(build_path_class(crystal, representative), crystal)
        out[representative] = [(build(group.members, N_SMALL).events, group)]
    per_group = store_plan(build_path_class(crystal, (3, 5)), crystal, transport=False)
    out["3-5 per group"] = [(build(group.members, N_SMALL).events, group) for group in per_group]
    return out


@pytest.mark.parametrize("family", sorted(FAMILIES))
@pytest.mark.parametrize("case", [(3, 5), (1, 3, 5, 2), "3-5 per group"])
def test_scatter_form_equals_the_gather_oracle(class_stores, family, case):
    """``scatter_store`` against ``class_band_sum_pixel`` on every family, proper and improper transports, many stores.

    Small chunks and blocks put chunk and block edges inside most bands;
    rows 40-60 of the canonical camera cover the inner-edge caustic.
    """
    stores = class_stores[case]
    density = build_pose_density(family, **FAMILIES[family])
    arrays = [(events.arrays(), group) for events, group in stores]
    if case == (1, 3, 5, 2):
        views = [(WIDE_RENDER, [(r, c) for r in range(0, 41, 3) for c in range(0, 41, 4)])]
    else:  # the canonical camera (the columns are dark under plate and Lowitz densities) and one about the parhelia
        views = [
            (CANONICAL_RENDER, [(r, c) for c in (0, 126, 250) for r in [*range(40, 61), *range(61, 801, 53)]]),
            (PARHELIA_RENDER, [(r, c) for r in range(0, 41, 5) for c in range(0, 81, 5)]),
        ]
    lit = 0
    for render, pixels in views:
        scattered = scatter(stores, density, pixels, render=render, event_chunk=257, pixel_block=19)
        gathered = [class_band_sum_pixel(arrays, SUN, density, r, c, N_SMALL, render) for r, c in pixels]
        lit += assert_scatter_equals_gather(scattered, gathered, k_eff_rel=1e-12 if family != "lowitz" else 1e-10)
    assert lit >= 5


def test_scatter_band_edges_are_the_gathers_left_closed_right_open():
    """Hand-built events on and next to both band ends: the same ``K`` and value as the gather, bit for bit."""
    row, column = 400, 126
    _, delta, lo, hi = pixel_band(row, column, SUN)
    d = np.array([lo - 1e-6, np.nextafter(lo, -1.0), lo, 0.5 * (lo + hi), np.nextafter(hi, -1.0), hi, hi + 1e-3])
    w = np.arange(1.0, 8.0)
    u = np.tile([-1.0, 0.0, 0.0], (len(d), 1))
    events = S2Events(u, np.stack([np.cos(d), np.sin(d), np.zeros(len(d))], axis=1), d, w)
    group = identity_group([(3, 5)])
    for density in (UNIFORM, canonical_pose_density()):
        for chunk in (1, 2, 3, 4096):
            (a,) = scatter([(events, group)], density, [(row, column)], event_chunk=chunk, pixel_block=1)
            b = class_band_sum_pixel([(events.arrays(), group)], SUN, density, row, column, N_SMALL)
            assert (a.K, a.K_rho_pos, a.total) == (b.K, b.K_rho_pos, b.total) == (3, a.K_rho_pos, b.total)
    (a,) = scatter([(events, group)], UNIFORM, [(row, column)])
    assert a.total == 3.0 + 4.0 + 5.0
    # A band with no event, and pixels that share no event with the chunk walk, stay exactly 0.
    empty = S2Events(u[:1], events.phi[:1], np.array([hi + 1.0]), w[:1])
    (z,) = scatter([(empty, group)], UNIFORM, [(row, column)])
    assert (z.value, z.K, z.K_rho_pos, z.K_eff) == (0.0, 0, 0, 0.0)


# ----------------------------------------------------------- window + io
def scene_of(path_class, render=CANONICAL_RENDER, **kwargs) -> BandSumScene:
    return BandSumScene(
        path_class=path_class,
        crystal=canonical_crystal(),
        refractive_index=CANONICAL_REFRACTIVE_INDEX,
        sun_direction=SUN,
        pose_density=canonical_pose_density(),
        render=render,
        **kwargs,
    )


def test_window_render_workers_agree_and_the_cache_is_reused(tmp_path, monkeypatch):
    scene = scene_of(single_path_class(canonical_crystal(), (3, 5)))
    window = Window((395, 400), (124, 129))
    one, execution = render_band_sum_window(scene, window, N_SMALL, workers=1, base_dir=tmp_path, run_checks=False)
    assert len(one) == 25 and len(execution["stores"]) == 1
    assert (tmp_path / execution["stores"][0]["cache_key"] / "D.npy").exists()

    def no_rebuild(*args, **kwargs):
        raise AssertionError("the cached store must be loaded, not rebuilt")

    monkeypatch.setattr(s2_store, "_build_into", no_rebuild)
    two, execution_two = render_band_sum_window(scene, window, N_SMALL, workers=2, base_dir=tmp_path, run_checks=False)
    # Column by column, rows within a column (the order does not depend on the deviation segments).
    assert [(r.row, r.column) for r in one] == [(r.row, r.column) for r in two] == [(r, c) for c in range(124, 129) for r in range(395, 400)]
    # Another segmentation groups other pixels into a block: the sums agree to round-off, the bands exactly.
    assert_scatter_equals_gather(two, one, rel=1e-13)
    assert any(r.value > 0.0 for r in one)
    assert len(execution["segments"]) == 1 and len(execution_two["segments"]) == 2
    assert sum(s["pixels"] for s in execution_two["segments"]) == 25
    # The gather oracle, pixel for pixel.
    events = s2_store.S2EventStore.load(tmp_path / execution["stores"][0]["cache_key"]).events.arrays()
    (group,) = scene.plan
    gathered = [class_band_sum_pixel([(events, group)], SUN, scene.pose_density, r.row, r.column, N_SMALL) for r in one]
    assert_scatter_equals_gather(one, gathered)


def test_one_store_serves_every_sun_altitude(tmp_path, monkeypatch):
    """Schema 3: the store key has no sun, so another sun altitude reuses the cached store instead of rebuilding."""
    scene = scene_of(single_path_class(canonical_crystal(), (3, 5)))
    higher = dataclasses.replace(scene, sun_direction=sun_direction(25.0, 0.0))
    window = Window((395, 400), (124, 129))
    first, first_execution = render_band_sum_window(scene, window, N_SMALL, base_dir=tmp_path / "shared", run_checks=False)
    fresh, _ = render_band_sum_window(higher, window, N_SMALL, base_dir=tmp_path / "fresh", run_checks=False)

    def no_rebuild(*args, **kwargs):
        raise AssertionError("a store serves every sun direction; another altitude must not rebuild it")

    monkeypatch.setattr(s2_store, "_build_into", no_rebuild)
    second, second_execution = render_band_sum_window(higher, window, N_SMALL, base_dir=tmp_path / "shared", run_checks=False)
    assert second_execution["stores"][0]["cache_key"] == first_execution["stores"][0]["cache_key"]
    assert [p.name for p in (tmp_path / "shared").iterdir()] == [first_execution["stores"][0]["cache_key"]]
    key = lambda r: (r.row, r.column)  # noqa: E731
    assert [(r.value, r.K) for r in sorted(second, key=key)] == [(r.value, r.K) for r in sorted(fresh, key=key)]
    assert len(second) == 25 and any(r.value > 0.0 for r in first)
    # Both altitudes from the one store against the gather oracle at that altitude.
    events = s2_store.S2EventStore.load(tmp_path / "shared" / first_execution["stores"][0]["cache_key"]).events.arrays()
    (group,) = scene.plan
    for rendered, sun in ((first, scene.sun_direction), (second, higher.sun_direction)):
        gathered = [class_band_sum_pixel([(events, group)], sun, scene.pose_density, r.row, r.column, N_SMALL) for r in rendered]
        assert_scatter_equals_gather(rendered, gathered)
    assert [r.K for r in first] != [r.K for r in second]


def test_output_is_readable_by_read_strip(tmp_path):
    scene = scene_of(build_path_class(canonical_crystal(), (3, 5)))
    window = Window((390, 400), (120, 123))
    results, execution = render_band_sum_window(scene, window, N_SMALL, base_dir=tmp_path / "stores", run_checks=False)
    density_block = pose_density_provenance("column", zenith_std_deg=0.5)
    files = write_band_sum_strip(
        tmp_path / "out", results, scene=scene, n=N_SMALL, window=window, pose_density_block=density_block, execution=execution
    )
    arrays, provenance = read_strip(tmp_path / "out")
    assert arrays.values.shape == (801, 251)
    rendered = np.zeros((801, 251), dtype=bool)
    rendered[390:400, 120:123] = True
    np.testing.assert_array_equal(arrays.rendered, rendered)
    for r in results:
        assert arrays.values[r.row, r.column] == r.value
        assert bool(arrays.status[r.row, r.column] & STATUS_HAS_COMPONENT) == (r.value != 0.0)
    assert set(np.unique(arrays.status)) <= {0, STATUS_RENDERED, STATUS_RENDERED | STATUS_HAS_COMPONENT}
    provenance = json.loads(json.dumps(provenance))
    options = provenance["options"]
    assert options["N"] == N_SMALL and options["path_class"]["size"] == 12
    assert options["stores"][0]["cache_key"] == execution["stores"][0]["cache_key"]
    assert len(options["store_plan"][0]["transports"]) == 12
    assert options["k_eff_semantics"] == K_EFF_SEMANTICS == "per_event"
    assert "distinct" in provenance["arrays"]["value_semantics"] and "distinct" in options["class_value"]
    assert provenance["scene"]["pose_density"]["value"] == density_block
    header = files["pixels"].read_text().splitlines()[0].split(",")
    assert tuple(header) == PIXEL_CSV_COLUMNS
    assert len(files["pixels"].read_text().splitlines()) == 1 + window.pixel_count


def test_pixels_csv_columns_are_float_parseable(tmp_path):
    """``csv_row`` must stay ``float()``-parseable even where the source is a numpy scalar (``band_sum_estimate``)."""
    scene = scene_of(single_path_class(canonical_crystal(), (3, 5)))
    window = Window((395, 400), (124, 129))
    results, execution = render_band_sum_window(scene, window, N_SMALL, base_dir=tmp_path / "stores", run_checks=False)
    assert any(r.value > 0.0 for r in results)  # exercises band_sum_estimate, not just the total == 0.0 shortcut
    files = write_band_sum_strip(
        tmp_path / "out", results, scene=scene, n=N_SMALL, window=window,
        pose_density_block=pose_density_provenance("column", zenith_std_deg=0.5), execution=execution,
    )
    with files["pixels"].open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    for row in rows:
        for name in PIXEL_CSV_COLUMNS:
            float(row[name])


def test_rank0_class_puts_the_task9_point_mass_on_the_sun_pixel_only(tmp_path):
    crystal = canonical_crystal()
    cls = build_path_class(crystal, (1, 2))
    assert cls.halo_map_rank == 0
    scene = scene_of(cls, render=SUN_WINDOW, rank0_sample_count=50_000)
    results, execution = render_band_sum_window(scene, Window((9, 12), (9, 12)), N_SMALL, base_dir=tmp_path)
    assert execution["stores"] == [] and not any(tmp_path.iterdir())
    values = {(r.row, r.column): r.value for r in results}
    mass = execution["rank0"]["estimate"]["value"]
    assert values.pop((10, 10)) == mass / pixel_solid_angle(SUN_WINDOW, 10, 10) > 0.0
    assert set(values.values()) == {0.0}
    # Outside the sun pixel nothing is sampled.
    results, execution = render_band_sum_window(scene, Window((0, 3), (0, 3)), N_SMALL, base_dir=tmp_path)
    assert all(r.value == 0.0 for r in results) and "estimate" not in execution["rank0"]


def test_rank0_k_eff_placeholder_is_not_in_the_summary(tmp_path):
    """The rank-0 point mass carries ``K_eff = 0.0`` as a placeholder; the lit-pixel summary excludes it."""
    cls = build_path_class(canonical_crystal(), (1, 2))
    scene = scene_of(cls, render=SUN_WINDOW, rank0_sample_count=50_000)
    window = Window((9, 12), (9, 12))
    results, execution = render_band_sum_window(scene, window, N_SMALL, base_dir=tmp_path / "stores")
    lit = [r for r in results if r.value > 0.0]
    assert len(lit) == 1 and lit[0].K_eff == 0.0
    write_band_sum_strip(
        tmp_path / "out", results, scene=scene, n=N_SMALL, window=window,
        pose_density_block=pose_density_provenance("column", zenith_std_deg=0.5), execution=execution,
    )
    summary = json.loads((tmp_path / "out" / "provenance.json").read_text())["summary"]
    assert summary["pixels_with_light"] == 1
    assert summary["K_eff_median_lit"] is None and summary["K_eff_min_lit"] is None


def test_store_members_are_recorded_by_path_id():
    plan = store_plan(build_path_class(canonical_crystal(), (3, 5)), canonical_crystal())
    block = plan[0].as_json()
    assert block["store_members"] == ["3-5"]
    assert sorted(t["members"][0] for t in block["transports"]) == sorted(
        path_id_of(m) for m in build_path_class(canonical_crystal(), (3, 5)).members
    )
