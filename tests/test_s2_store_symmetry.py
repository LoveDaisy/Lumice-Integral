"""Symmetry transport of the S^2 event store (roadmap section 4.1(d)): one store serves a proper orbit.

For a class member reached from the representative by a proper ``D6h``
element ``g`` (:func:`.path_class.path_class_symmetry`), the member's
events are the representative's with ``u' = g u``, ``Phi' = g Phi`` and
``D``, ``w`` unchanged.  The Fibonacci lattice is not closed under ``g``,
so the member's own store is built on the ``g``-rotated lattice (the same
point set the transported events live on) through the production builder,
which evaluates the member's own faces: equal events there are the
physical statement, not a tautology.  The production interface is the pose
factor ``g^-1`` (``s2_store`` module docstring); :func:`transport_events`
lives here, in the tests, as the literal form it is checked against.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from lumice_integral.camera import linear_pixel_outgoing_direction
from lumice_integral.canonical_scene import (
    CANONICAL_REFRACTIVE_INDEX,
    CANONICAL_RENDER,
    canonical_crystal,
    canonical_incident_direction,
)
from lumice_integral.geometry import HexPrism
from lumice_integral.optics import path_id_of
from lumice_integral.path_class import build_path_class, hexprism_symmetry_matrices, path_class_symmetry
from lumice_integral.pose_density import build_pose_density
from lumice_integral.s2_store import S2Events, S2EventStore, build_event_store, event_rotations, fibonacci_sphere

TASK14_ARTIFACTS = Path(__file__).resolve().parents[1] / "scratchpad/task-narrow-density-band-sum-probe/artifacts"
N_EVENTWISE = 1_000_000


def transport_events(events: S2Events, g: np.ndarray) -> S2Events:
    """The member's events: ``u' = g u``, ``Phi' = g Phi``; ``D``, ``w``, ``iw`` unchanged (still sorted)."""
    return S2Events(
        np.einsum("ij,nj->ni", g, events.u), np.einsum("ij,nj->ni", g, events.phi), events.D, events.w, events.iw
    )


def rotated_lattice(g: np.ndarray, n: int):
    """The ``n``-point Fibonacci lattice rotated by ``g`` (a uniform point set, so no inverse weights)."""

    def sampler(first: int, stop: int) -> tuple[np.ndarray, None]:
        return np.einsum("ij,nj->ni", g, fibonacci_sphere(n, first, stop)), None

    return sampler


def build(members, n: int, *, crystal=None, g: np.ndarray | None = None, **kwargs) -> S2EventStore:
    extra = {} if g is None else {"sampler": rotated_lattice(g, n), "sampling": f"Fibonacci lattice rotated by {g.round(12).tolist()}"}
    return build_event_store(
        canonical_crystal() if crystal is None else crystal,
        CANONICAL_REFRACTIVE_INDEX,
        members,
        n,
        incident_direction=canonical_incident_direction(),
        run_checks=False,
        **extra,
        **kwargs,
    )


def max_event_difference(a: S2Events, b: S2Events) -> dict[str, float]:
    assert len(a) == len(b) > 0
    return {name: float(np.max(np.abs(a.arrays()[name] - b.arrays()[name]))) for name in ("u", "phi", "D", "w")}


@pytest.fixture(scope="module")
def class_3_5():
    crystal = canonical_crystal()
    path_class = build_path_class(crystal, (3, 5))
    return path_class, path_class_symmetry(path_class, crystal)


@pytest.fixture(scope="module")
def representative() -> S2EventStore:
    return build([(3, 5)], N_EVENTWISE)


def test_transported_events_equal_each_members_own_store(class_3_5, representative) -> None:
    """Class ``[3,5]``, ``N = 1e6``: all 12 members, sorted ``u / Phi / D / w`` within ``1e-12``."""
    _, symmetry = class_3_5
    worst: dict[str, float] = {}
    for member, g in symmetry.items():
        own = build([member], N_EVENTWISE, g=g)
        difference = max_event_difference(transport_events(representative.events, g), own.events)
        worst = {k: max(worst.get(k, 0.0), v) for k, v in difference.items()}
        assert max(difference.values()) <= 1e-12, (member, difference)
    assert set(worst) == {"u", "phi", "D", "w"}


def test_pose_factor_is_the_transport(class_3_5, representative) -> None:
    """``event_rotations`` of the transported events equals the representative's poses times ``g^-1``."""
    _, symmetry = class_3_5
    s = canonical_incident_direction()
    centre = linear_pixel_outgoing_direction(400, 126, **CANONICAL_RENDER)
    band = representative.band_slice(np.radians(22.0), np.radians(23.0))
    assert len(band) > 1000
    poses = event_rotations(band.u, band.phi, band.D, s, centre)
    for member, g in symmetry.items():
        moved = transport_events(band, g)
        direct = event_rotations(moved.u, moved.phi, moved.D, s, centre)
        assert np.max(np.abs(direct - poses @ g.T)) <= 1e-12, member
        assert np.max(np.abs(np.einsum("nij,nj->ni", direct, moved.u) - s)) <= 1e-12


def test_none_is_not_a_transport(class_3_5, representative) -> None:
    with pytest.raises((TypeError, ValueError)):
        transport_events(representative.band_slice(0.4, 0.41), None)


def test_improper_member_needs_its_own_store() -> None:
    """Class ``[3,1,2,5]`` on a plate: a mirror still maps the fields, but ``R g^-1`` is not a rotation.

    ``path_class_symmetry`` gives ``None`` for ``3-2-1-5`` (the basal swap).
    The transported events equal that member's own store on the mirrored
    lattice -- the optics are mirror symmetric -- but the pose factor has
    determinant ``-1``, so the representative's store cannot serve it.
    """
    plate = HexPrism.from_ratio(0.3)
    path_class = build_path_class(plate, (3, 1, 2, 5))
    member = (3, 2, 1, 5)
    assert path_class_symmetry(path_class, plate)[member] is None
    mirror = next(e for e in hexprism_symmetry_matrices() if np.allclose(e, np.diag([1.0, 1.0, -1.0])))
    n = 100_000
    representative = build([(3, 1, 2, 5)], n, crystal=plate)
    own = build([member], n, crystal=plate, g=mirror)
    transported = transport_events(representative.events, mirror)
    assert max(max_event_difference(transported, own.events).values()) <= 1e-12
    s = canonical_incident_direction()
    poses = event_rotations(representative.events.u, representative.events.phi, representative.events.D, s, np.array([0.0, 1.0, 0.0]))
    assert np.allclose(np.linalg.det(poses @ mirror.T), -1.0)


# ------------------------------------------------- task 14 class regression
def pixel_band(row: int, column: int, s: np.ndarray, render) -> tuple[np.ndarray, float, float]:
    """Pixel-centre direction and ``[delta_lo, delta_hi]`` from the four corners (the task 14 pixel band)."""
    centre = linear_pixel_outgoing_direction(row, column, **render)
    deviations = [
        float(np.arccos(np.clip(linear_pixel_outgoing_direction(row + dr, column + dc, **render) @ s, -1.0, 1.0)))
        for dr in (-0.5, 0.5)
        for dc in (-0.5, 0.5)
    ]
    return centre, min(deviations), max(deviations)


def class_totals(stores, factors, density, pixels, s) -> np.ndarray:
    """Per pixel ``sum_members sum_band w rho(R g^-1)``; ``stores[k]`` is paired with pose factor ``factors[k]``."""
    totals = np.zeros(len(pixels))
    for store, factor in zip(stores, factors):
        for p, (centre, lo, hi) in enumerate(pixels):
            band = store.band_slice(lo, hi)
            if len(band) == 0:
                continue
            poses = event_rotations(band.u, band.phi, band.D, s, centre)
            totals[p] += float(np.sum(band.w * density.evaluate_batch(poses @ factor.T)))
    return totals


@pytest.mark.slow
def test_task14_class_band_sums_single_store_equals_twelve_stores(class_3_5) -> None:
    """Task 14's three profiles, ``N = 1e7``: one store with pose factors ``g^-1`` against 12 member stores.

    The 12 member stores are built on the ``g``-rotated lattices (same point
    sets), so the class sums agree to ``1e-12`` relative.  Task 14's own 12
    stores (independent Fibonacci lattices per member) sample different
    points; against those the agreement is at the discretisation level and
    is reported, not held to ``1e-12`` (the ratio is asserted near 1).
    """
    windows_path = TASK14_ARTIFACTS / "profile_windows.json"
    if not windows_path.exists():
        pytest.skip(f"{windows_path} not present (scratchpad artifact of task 14)")
    families = json.loads(windows_path.read_text())["families"]
    s = canonical_incident_direction()
    profiles = {
        family: [pixel_band(row, column, s, spec["render"]) for row, column in spec["pixels"]]
        for family, spec in families.items()
    }
    window = (min(lo for p in profiles.values() for _, lo, _ in p), max(hi for p in profiles.values() for _, _, hi in p))
    path_class, symmetry = class_3_5
    n = 10_000_000
    single = build([path_class.representative], n, deviation_window=window)
    members = [build([m], n, g=g, deviation_window=window) for m, g in symmetry.items()]
    identity = np.eye(3)
    artifact_dirs = [TASK14_ARTIFACTS / "members" / path_id_of(m) for m in symmetry]
    have_artifacts = all((d / f"events_N{n}.npz").exists() for d in artifact_dirs)
    report = {}
    for family, pixels in profiles.items():
        density = build_pose_density(family, **families[family]["density"])
        one = class_totals([single] * len(symmetry), list(symmetry.values()), density, pixels, s)
        twelve = class_totals(members, [identity] * len(members), density, pixels, s)
        lit = twelve > 0.0
        assert np.array_equal(one > 0.0, lit) and np.count_nonzero(lit) > 10, family
        relative = float(np.max(np.abs(one[lit] - twelve[lit]) / twelve[lit]))
        report[family] = {"lit_pixels": int(np.count_nonzero(lit)), "max_rel_single_vs_twelve": relative}
        assert relative <= 1e-12, (family, relative)
        if have_artifacts:
            stores = []
            for directory in artifact_dirs:
                with np.load(directory / f"events_N{n}.npz") as data:
                    stores.append(S2EventStore(single.spec, S2Events(data["u"], data["phi"], data["D"], data["w"]), {}))
            task14 = class_totals(stores, [identity] * len(stores), density, pixels, s)
            del stores
            peak = task14 > 0.1 * task14.max()
            ratio = one[peak] / task14[peak]
            report[family].update(
                task14_peak_pixels=int(np.count_nonzero(peak)),
                task14_ratio_median=float(np.median(ratio)),
                task14_ratio_min=float(ratio.min()),
                task14_ratio_max=float(ratio.max()),
                task14_sum_ratio=float(one[peak].sum() / task14[peak].sum()),
            )
            assert abs(report[family]["task14_sum_ratio"] - 1.0) < 0.05, report[family]
    print(json.dumps(report, indent=2))
