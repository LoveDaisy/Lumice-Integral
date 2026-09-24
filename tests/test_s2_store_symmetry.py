"""Symmetry transport of the S^2 event store (roadmap section 4.1(d)): one store serves a ``D6h`` orbit.

For a class member reached from the representative by a ``D6h`` element
``g``, proper or improper (:func:`.path_class.path_class_symmetry`), the
member's events are the representative's with ``u' = g u``, ``phi' = g phi``
and ``D``, ``w`` and the valid domain unchanged.  The Fibonacci lattice is not closed under ``g``,
so the member's own store is built on the ``g``-rotated lattice (the same
point set the transported events live on) through the production builder,
which evaluates the member's own faces: equal events there are the
physical statement, not a tautology.  The production interface is
:func:`.s2_store.transported_rotations` (``L_g R g^T``, ``s2_store`` module
docstring); :func:`transport_events` lives here, in the tests, as the
literal form it is checked against.
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
    canonical_sun_direction,
)
from lumice_integral.geometry import HexPrism
from lumice_integral.optics import path_id_of
from lumice_integral.path_class import (
    PathClass,
    _hexprism_normals,
    _symmetry_image_of_faces,
    build_path_class,
    hexprism_symmetry_matrices,
    path_class_symmetry,
)
from lumice_integral.pose_density import build_pose_density
from lumice_integral.s2_store import (
    S2Events,
    S2EventStore,
    align_rotations,
    build_event_store,
    evaluate_fields,
    event_rotations,
    events_from_schema1,
    store_lattice,
    transported_rotations,
)

TASK14_ARTIFACTS = Path(__file__).resolve().parents[1] / "scratchpad/task-narrow-density-band-sum-probe/artifacts"
N_EVENTWISE = 1_000_000


def transport_events(events: S2Events, g: np.ndarray) -> S2Events:
    """The member's events: ``u' = g u``, ``phi' = g phi``; ``D``, ``w``, ``iw`` unchanged (still sorted)."""
    return S2Events(
        np.einsum("ij,nj->ni", g, events.u), np.einsum("ij,nj->ni", g, events.phi), events.D, events.w, events.iw
    )


def rotated_lattice(g: np.ndarray, n: int):
    """The store's ``n``-point (antipodal Fibonacci) lattice rotated by ``g`` (uniform, so no inverse weights)."""

    def sampler(first: int, stop: int) -> tuple[np.ndarray, None]:
        return np.einsum("ij,nj->ni", g, store_lattice(n, first, stop)), None

    return sampler


def build(members, n: int, *, crystal=None, g: np.ndarray | None = None, **kwargs) -> S2EventStore:
    extra = {} if g is None else {"sampler": rotated_lattice(g, n), "sampling": f"Fibonacci lattice rotated by {g.round(12).tolist()}"}
    return build_event_store(
        canonical_crystal() if crystal is None else crystal,
        CANONICAL_REFRACTIVE_INDEX,
        members,
        n,
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


def test_transported_rotations_rebuild_the_transported_events_for_all_24_elements(class_3_5, representative) -> None:
    """``L_g R g^T`` equals ``event_rotations`` of ``(g u, g Phi, D)`` for every ``D6h`` element, and is a rotation.

    For a proper ``g`` it is the plain pose factor ``R g^T``, bit for bit
    (the task ``band-sum-renderer`` operation); for a mirror the reflection
    ``L_g`` in the ``(s_hat, centre)`` plane makes it a rotation again.
    """
    sun = canonical_sun_direction()
    band = representative.band_slice(np.radians(22.0), np.radians(23.0))
    assert len(band) > 1000
    for row, column in ((400, 126), (120, 30), (700, 240)):
        centre = linear_pixel_outgoing_direction(row, column, **CANONICAL_RENDER)
        poses = event_rotations(band.u, band.phi, band.D, sun, centre)
        for g in hexprism_symmetry_matrices():
            moved = transport_events(band, g)
            direct = event_rotations(moved.u, moved.phi, moved.D, sun, centre)
            fast = transported_rotations(poses, g, sun, centre)
            assert np.max(np.abs(direct - fast)) <= 1e-13, (row, column, g.round(3).tolist())
            if np.linalg.det(g) > 0.0:
                np.testing.assert_array_equal(fast, poses @ g.T)
            assert np.max(np.abs(np.linalg.det(fast) - 1.0)) <= 1e-13
            assert np.max(np.abs(np.einsum("nij,nkj->nik", fast, fast) - np.eye(3))) <= 1e-13
            assert np.max(np.abs(np.einsum("nij,nj->ni", fast, moved.u) - sun)) <= 1e-12  # R u = s_hat
    # The class's own elements are among them.
    _, symmetry = class_3_5
    assert all(np.linalg.det(g) > 0.0 for g in symmetry.values())


def test_none_is_not_a_transport(class_3_5, representative) -> None:
    with pytest.raises((TypeError, ValueError)):
        transport_events(representative.band_slice(0.4, 0.41), None)


@pytest.mark.parametrize("faces", [(3, 5), (1, 3), (3, 1, 2, 5)])
def test_fields_are_equivariant_under_all_24_elements(faces) -> None:
    """``w_{gPg^-1}(g u) = w_P(u)``, ``Phi_{gPg^-1}(g u) = g Phi_P(u)``, the same valid domain, all of ``D6h``.

    Random ``u`` on the canonical crystal (``h/a = 2``), the production
    evaluators on each image face sequence; ``3-1-2-5`` is a four-face path
    that is feasible there.  The owner probe of this task
    (``owner_probe_mirror.py``) checked the twelve mirrors; here every
    element, proper ones included.
    """
    crystal = canonical_crystal()
    sun = canonical_sun_direction()
    normals = _hexprism_normals(crystal)
    u = np.random.default_rng(3).normal(size=(20_000, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    base = evaluate_fields(align_rotations(u, sun), sun, crystal, CANONICAL_REFRACTIVE_INDEX, [faces])
    assert 0 < np.count_nonzero(base["valid"])
    for g in hexprism_symmetry_matrices():
        image = _symmetry_image_of_faces(g, faces, normals)
        moved = evaluate_fields(align_rotations(u @ g.T, sun), sun, crystal, CANONICAL_REFRACTIVE_INDEX, [image])
        np.testing.assert_array_equal(moved["valid"], base["valid"])
        valid = base["valid"]
        assert np.max(np.abs(moved["w"] - base["w"])) <= 1e-12, (faces, image)
        assert np.max(np.abs(moved["phi"][valid] - base["phi"][valid] @ g.T)) <= 1e-12, (faces, image)
        assert np.max(np.abs(moved["D"][valid] - base["D"][valid])) <= 1e-12, (faces, image)


def test_improper_member_is_served_by_the_representatives_store() -> None:
    """Class ``[3,1,2,5]`` on a plate: ``3-2-1-5`` is reached by the basal mirror only, and one store serves it.

    ``path_class_symmetry`` gives it an improper element.  The transported
    events equal that member's own store on the mirrored lattice (the optics
    are mirror symmetric), and the poses rebuilt from them are rotations,
    equal to :func:`transported_rotations` of the representative's poses.
    """
    plate = HexPrism.from_ratio(0.3)
    path_class = build_path_class(plate, (3, 1, 2, 5))
    member = (3, 2, 1, 5)
    g = path_class_symmetry(path_class, plate)[member]
    assert np.linalg.det(g) < 0.0
    assert not any(
        np.linalg.det(e) > 0.0 and _symmetry_image_of_faces(e, (3, 1, 2, 5), _hexprism_normals(plate)) == member
        for e in hexprism_symmetry_matrices()
    )
    n = 100_000
    representative = build([(3, 1, 2, 5)], n, crystal=plate)
    own = build([member], n, crystal=plate, g=g)
    transported = transport_events(representative.events, g)
    assert max(max_event_difference(transported, own.events).values()) <= 1e-12
    sun = canonical_sun_direction()
    centre = np.array([0.0, 1.0, 0.0])
    events = representative.events
    poses = event_rotations(events.u, events.phi, events.D, sun, centre)
    assert np.allclose(np.linalg.det(poses @ g.T), -1.0)  # the plain pose factor is not a rotation
    fast = transported_rotations(poses, g, sun, centre)
    own_poses = event_rotations(own.events.u, own.events.phi, own.events.D, sun, centre)
    assert np.max(np.abs(fast - own_poses)) <= 1e-12
    assert np.allclose(np.linalg.det(fast), 1.0)


def test_a_member_outside_the_orbit_is_an_error() -> None:
    """A hand-made class whose members are not one ``D6h`` orbit is a construction error, not a fallback."""
    crystal = canonical_crystal()
    broken = PathClass((3, 5), ((3, 5), (3, 7)), 60.0, 2)
    assert set(path_class_symmetry(broken, crystal)) == {(3, 5), (3, 7)}
    broken = PathClass((3, 5), ((3, 5), (1, 3)), 60.0, 2)
    with pytest.raises(RuntimeError, match="not a D6h image"):
        path_class_symmetry(broken, crystal)


# ------------------------------------------------- task 14 class regression
def pixel_band(row: int, column: int, s: np.ndarray, render) -> tuple[np.ndarray, float, float]:
    """Pixel-centre direction and ``[delta_lo, delta_hi]`` from the four corners (the task 14 pixel band).

    ``s`` is the propagation direction ``-s_hat``: deviations are angles between propagation directions.
    """
    centre = linear_pixel_outgoing_direction(row, column, **render)
    deviations = [
        float(np.arccos(np.clip(linear_pixel_outgoing_direction(row + dr, column + dc, **render) @ s, -1.0, 1.0)))
        for dr in (-0.5, 0.5)
        for dc in (-0.5, 0.5)
    ]
    return centre, min(deviations), max(deviations)


def class_totals(stores, factors, density, pixels, sun) -> np.ndarray:
    """Per pixel ``sum_members sum_band w rho(R g^T)``; ``stores[k]`` is paired with pose factor ``factors[k]``."""
    totals = np.zeros(len(pixels))
    for store, factor in zip(stores, factors):
        for p, (centre, lo, hi) in enumerate(pixels):
            band = store.band_slice(lo, hi)
            if len(band) == 0:
                continue
            poses = event_rotations(band.u, band.phi, band.D, sun, centre)
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
    sun, s = canonical_sun_direction(), canonical_incident_direction()
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
        one = class_totals([single] * len(symmetry), list(symmetry.values()), density, pixels, sun)
        twelve = class_totals(members, [identity] * len(members), density, pixels, sun)
        lit = twelve > 0.0
        assert np.array_equal(one > 0.0, lit) and np.count_nonzero(lit) > 10, family
        relative = float(np.max(np.abs(one[lit] - twelve[lit]) / twelve[lit]))
        report[family] = {"lit_pixels": int(np.count_nonzero(lit)), "max_rel_single_vs_twelve": relative}
        assert relative <= 1e-12, (family, relative)
        if have_artifacts:
            stores = []
            for directory in artifact_dirs:
                with np.load(directory / f"events_N{n}.npz") as data:  # task 14 stores are schema 1
                    arrays = events_from_schema1({key: data[key] for key in data.files})
                stores.append(S2EventStore(single.spec, S2Events(arrays["u"], arrays["phi"], arrays["D"], arrays["w"]), {}))
            task14 = class_totals(stores, [identity] * len(stores), density, pixels, sun)
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
