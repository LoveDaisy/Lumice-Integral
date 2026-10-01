"""``chromatic``: the weight-kink colour criterion (edges under random orientation, tints of plate classes)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial import cKDTree

from lumice_integral import chromatic as C
from lumice_integral.dp_field import DPField
from lumice_integral.geometry import HexPrism
from lumice_integral.s2_store import fibonacci_sphere

RHOMBIC = [1.5, 1.0, 1.0, 1.5, 1.0, 1.0]  # the user's plate: Lumice face_distance of faces 3..8


def _rim(index: float) -> float:
    return 2.0 * np.arcsin(np.sqrt(index * index - 1.0))


# ---- random orientation ----------------------------------------------------------------------------------


@pytest.mark.parametrize("faces", [(3, 1, 6), (1, 3, 2)])
def test_single_mirror_slab_is_a_visible_blue_edge_at_the_hole_rim(faces) -> None:
    """The dark hole of ``3-1-6`` / ``1-3-2``: its rim sits at ``2 arcsin sqrt(n^2 - 1)`` and is blue."""
    verdict = C.diagnose(HexPrism(), faces)
    assert (verdict.kind, verdict.color, verdict.visible) == ("edge", "blue", True)
    (edge,) = verdict.features
    assert edge.source == "internal_1_tir_discriminant" and edge.positive_fraction == 1.0
    assert edge.delta_red == pytest.approx(_rim(C.N_RED), abs=1e-12)
    assert edge.delta_blue == pytest.approx(_rim(C.N_BLUE), abs=1e-12)
    assert verdict.position == pytest.approx(_rim(C.N_BLUE), abs=1e-12)
    assert edge.shift == pytest.approx(_rim(C.N_BLUE) - _rim(C.N_RED), abs=1e-12)  # 3.354 deg
    assert edge.spread < 1e-12 and edge.direction_dispersion == 0.0
    assert 0.0 < edge.lit_fraction <= 1.0 and edge.weight > 0.0
    # 1 - R at n_red on the blue rim: the red partial reflection there, sin theta = sqrt(n_b^2 - 1) / n_r ...
    assert 0.5 < edge.contrast < 0.8


def test_exit_gate_of_3_1_5_is_a_red_candidate_that_is_not_visible() -> None:
    """The red-edge candidate: red exits where blue is totally reflected, but the gate spreads over ~100 deg of D."""
    verdict = C.diagnose(HexPrism(), (3, 1, 5))
    (gate,) = [f for f in verdict.features if f.kind == "gate_edge"]
    assert gate.source == "exit_snell_discriminant"
    assert (gate.color, gate.visible) == ("red", False)
    assert gate.positive_fraction == 0.0
    assert gate.spread > np.radians(90.0) and abs(gate.shift) < np.radians(1.0)
    assert gate.score < 0.01


def test_internal_kink_of_3_1_5_is_a_visible_blue_band() -> None:
    """``3-1-5``'s kink (a closed-form circle, ``D`` not constant on it) moves by more than its own spread.

    Not the issue's expectation ("no visible colour edge" for 3-1-5, which was about the red gate): the band sum
    renders a blue band over delta ~ 130 - 142 deg with blue / red up to ~2.3 and no red edge
    (``scripts/verify_chromatic_kink.py``, task progress).  The verdict of the path is this band.
    """
    verdict = C.diagnose(HexPrism(), (3, 1, 5))
    (kink,) = [f for f in verdict.features if f.kind == "edge"]
    assert (kink.color, kink.visible) == ("blue", True)
    assert np.radians(4.0) < kink.spread < kink.shift < np.radians(8.0)
    assert np.degrees(kink.delta_red) == pytest.approx(130.45, abs=0.05)
    assert np.degrees(kink.delta_blue) == pytest.approx(138.78, abs=0.05)
    assert kink.direction_dispersion > np.radians(1.0)  # the direction map disperses too: D moves at fixed u
    assert (verdict.kind, verdict.color, verdict.visible) == ("edge", "blue", True)


def test_internal_gate_of_slab_3_7_5_is_a_blue_candidate() -> None:
    """``3-7-5`` is a slab (no exit gate); its moving gate is the internal incidence cosine (``m . n_a = -1/2``), blue."""
    verdict = C.diagnose(HexPrism(), (3, 7, 5))
    gates = [f for f in verdict.features if f.kind == "gate_edge"]
    assert [g.source for g in gates] == ["internal_1_incidence_cosine"]
    assert gates[0].color == "blue" and not gates[0].visible
    assert not verdict.visible


PROBE_MONO_PATHS = [(3, 1, 6), (3, 1, 5), (3, 7, 5), (3, 5, 1, 7), (1, 3, 5, 2)]


@pytest.mark.parametrize("faces", PROBE_MONO_PATHS)
def test_internal_reflection_only_ever_favours_blue(faces) -> None:
    """``probe_mono``: no pose reflects totally at red and partially at blue; ``d disc_k / dn > 0`` on every kink point.

    Two chains: the kink's AD sign, and a count over 20000 directions of ``U_P`` at both indices.
    """
    verdict = C.diagnose(HexPrism(), faces)
    edges = [f for f in verdict.features if f.kind == "edge"]
    assert edges and all(f.color == "blue" and f.positive_fraction == 1.0 for f in edges)
    red, blue = (DPField.build(HexPrism(), faces, n) for n in (C.N_RED, C.N_BLUE))
    u = fibonacci_sphere(20000)
    valid = red.valid_batch(u) & blue.valid_batch(u)
    m_red, m_blue = red.margins_batch(u[valid]), blue.margins_batch(u[valid])
    from lumice_integral.optics import domain_margin_names

    for k, name in enumerate(domain_margin_names(faces)):
        if name.endswith("_tir_discriminant"):
            assert not np.any((m_red[:, k] > 0.0) & (m_blue[:, k] <= 0.0)), name


def test_path_without_colour_source() -> None:
    """``3-6`` on the regular prism is rank 0 (a point mass): refused like :meth:`DPField.build`."""
    with pytest.raises(ValueError, match="rank 0"):
        C.diagnose(HexPrism(), (3, 6))


# ---- classes --------------------------------------------------------------------------------------------------


def test_class_members_are_lumice_pbd_orbits() -> None:
    members = C.class_members((3, 5, 6, 8))
    assert (3, 5, 6, 8) in members and (8, 4, 5, 7) in members
    assert len(members) == 12  # D6 alone: B swaps basal faces only
    assert C.class_members((8, 4, 5, 7)) == members


def test_3_5_6_8_class_on_the_rhombic_plate_is_lit_only_through_other_members() -> None:
    """The literal ``3-5-6-8`` is impossible on ``[1.5, 1, 1, 1.5, 1, 1]``; four other members carry the class."""
    crystal = HexPrism(a=1.0, h=2.0, face_distance=RHOMBIC)
    verdict = C.diagnose_class(crystal, (3, 5, 6, 8), C.PlateFamily(9.0, samples=20000))
    expected = ((4, 8, 7, 5), (5, 7, 8, 4), (7, 5, 4, 8), (8, 4, 5, 7))
    assert verdict.lit_members == {"red": expected, "blue": expected}
    assert (verdict.verdict.kind, verdict.verdict.color) == ("none", "white")


def test_3_5_6_8_class_is_impossible_on_the_regular_prism() -> None:
    verdict = C.diagnose_class(HexPrism(a=1.0, h=2.0), (3, 5, 6, 8), C.RandomOrientation())
    assert verdict.lit_members == {"red": (), "blue": ()}
    assert (verdict.verdict.kind, verdict.verdict.color) == ("none", "none")


def test_random_class_of_3_1_6_is_one_member_verdict_on_the_regular_prism() -> None:
    """``G_true = D6h`` on the regular prism: the lit members are one symmetry orbit, diagnosed once."""
    verdict = C.diagnose_class(HexPrism(), (3, 1, 6), C.RandomOrientation())
    assert len(verdict.member_verdicts) == 1
    assert len(verdict.lit_members["red"]) == len(C.class_members((3, 1, 6)))
    assert (verdict.verdict.kind, verdict.verdict.color, verdict.verdict.visible) == ("edge", "blue", True)
    assert verdict.verdict.position == pytest.approx(_rim(C.N_BLUE), abs=1e-12)


def test_random_class_feasibility_resolution_is_the_lattice_covering_radius() -> None:
    """Every point of ``S^2`` is within ``feasibility_resolution_rad`` of the lattice (checked on 2e5 random points)."""
    verdict = C.diagnose_class(HexPrism(a=1.0, h=2.0), (3, 5, 6, 8), C.RandomOrientation(), lattice_n=5000)
    radius = verdict.feasibility_resolution_rad
    probe = np.random.default_rng(1).normal(size=(200000, 3))
    probe /= np.linalg.norm(probe, axis=1)[:, None]
    farthest = np.max(2.0 * np.arcsin(0.5 * cKDTree(fibonacci_sphere(5000)).query(probe)[0]))
    assert 0.9 * radius < farthest <= radius  # a bound, and a tight one
    assert np.radians(1.5) < radius < np.radians(3.0)
    assert C.diagnose_class(HexPrism(), (1, 3, 5, 2), C.PlateFamily(9.0, samples=2000)).feasibility_resolution_rad is None


def test_random_class_finds_every_lit_set_wider_than_its_resolution(monkeypatch) -> None:
    """A synthetic member lit on one cap just wider than the resolution is found wherever the cap sits.

    (No natural member with a lit set that small was found on five prisms x nine classes, 2e4 lattice points.)
    """
    radius = C._covering_radius(2000)
    centres = np.random.default_rng(2).normal(size=(40, 3))
    centres /= np.linalg.norm(centres, axis=1)[:, None]
    for centre in centres:
        def cap(points, faces, index, crystal, centre=centre):
            return (np.arccos(np.clip(points @ centre, -1.0, 1.0)) <= 1.0001 * radius).astype(float)

        monkeypatch.setattr(C, "_body_weight", cap)
        verdict = C.diagnose_class(HexPrism(), (3, 6), C.RandomOrientation(), lattice_n=2000)
        assert verdict.lit_members["red"] == C.class_members((3, 6)), centre


def test_plate_1_3_5_2_is_blue_on_a_small_sample() -> None:
    """The fast form of the 120 deg parhelion fixture (N = 2e4; the pinned N = 1e5 values are the slow test)."""
    crystal = HexPrism(a=1.0, h=2.0, face_distance=RHOMBIC)
    verdict = C.diagnose_class(crystal, (1, 3, 5, 2), C.PlateFamily(9.0, samples=20000)).verdict
    assert (verdict.kind, verdict.color, verdict.visible) == ("tint", "blue", True)
    assert 1.4 < verdict.tint.ratio < 1.6
    assert verdict.tint.direction_dispersion < 1e-12


def test_dispersive_plate_class_gets_no_tint() -> None:
    """``3-5`` on a plate at a 20 deg sun: red and blue land apart, the power ratio says nothing about a spot colour."""
    verdict = C.diagnose_class(HexPrism(a=1.0, h=0.3), (3, 5), C.PlateFamily(20.0, samples=20000)).verdict
    assert verdict.tint.direction_dispersion > C.EDGE_MIN_SHIFT_RAD
    assert (verdict.kind, verdict.color, verdict.visible) == ("none", "none", False)
    assert verdict.notes


# 120 deg parhelion fixture, sun 9 deg (the user's case) and 0 deg; the values are probe_120_cls.out.txt's (the same
# sampler and seed through an independent pose construction, scipy's Rotation, and a monkeypatched exit gate that
# #48 made the library's own).  The [1.4, 1.6] and [0.9, 1.1] bands separate "blue" from "white": the probe spans
# 1.36-1.53 for 1-3-5-2 and 0.96-1.03 for the white classes; the pinned values match to the printed digits.
PLATE_FIXTURE = {
    (9.0, 1.0): {(3, 5, 6, 8): 1.029, (1, 3, 5, 2): 1.492, (1, 3, 4, 2): 0.965},
    (9.0, 2.0): {(3, 5, 6, 8): 1.026, (1, 3, 5, 2): 1.528, (1, 3, 4, 2): 1.010},
    (0.0, 2.0): {(3, 5, 6, 8): 1.025, (1, 3, 5, 2): 1.407, (1, 3, 4, 2): 1.014},
}
LIT_1_3_5_2_AT_9_DEG = {1.0: 4, 2.0: 12}


# slow: 12 classes at N = 1e5 samples, ~5 min on an M2 Max
@pytest.mark.slow
@pytest.mark.parametrize("altitude,height", list(PLATE_FIXTURE))
def test_plate_120_deg_parhelion_fixture(altitude, height) -> None:
    crystal = HexPrism(a=1.0, h=height, face_distance=RHOMBIC)
    for representative, ratio in PLATE_FIXTURE[(altitude, height)].items():
        verdict = C.diagnose_class(crystal, representative, C.PlateFamily(altitude))
        assert verdict.verdict.tint.ratio == pytest.approx(ratio, abs=6e-4), representative
        if representative == (1, 3, 5, 2):
            assert 1.4 <= verdict.verdict.tint.ratio <= 1.6 or altitude == 0.0
            assert (verdict.verdict.kind, verdict.verdict.color) == ("tint", "blue")
            if altitude == 9.0:
                assert len(verdict.lit_members["red"]) == LIT_1_3_5_2_AT_9_DEG[height]
        else:
            assert 0.9 <= verdict.verdict.tint.ratio <= 1.1
            assert (verdict.verdict.kind, verdict.verdict.color) == ("none", "white")


# Calibration of TINT_RATIO_MIN on crystals and sun heights that are not acceptance fixtures: non-dispersive classes
# whose reflections stay total at both indices.  Measured (N = 2e4, seed 11): 0.955 - 1.023.
CALIBRATION = [
    (HexPrism(a=1.0, h=0.3), (1, 3, 4, 2)),
    (HexPrism(a=1.0, h=1.0, face_distance=[1.0, 1.2, 1.0, 1.2, 1.0, 1.2]), (1, 3, 4, 2)),
    (HexPrism(a=1.0, h=2.0, face_distance=[1.3, 1.0, 1.0, 1.3, 1.0, 1.0]), (1, 3, 4, 2)),
    (HexPrism(a=1.0, h=2.0, face_distance=[1.3, 1.0, 1.0, 1.3, 1.0, 1.0]), (3, 5, 6, 8)),
]


# slow: 8 plate classes at N = 2e4, ~45 s on an M2 Max
@pytest.mark.slow
def test_tint_threshold_calibration() -> None:
    deviations = []
    for crystal, representative in CALIBRATION:
        for altitude in (5.0, 20.0):
            verdict = C.diagnose_class(crystal, representative, C.PlateFamily(altitude, samples=20000, seed=11)).verdict
            assert verdict.tint.direction_dispersion < 1e-12
            assert verdict.tint.tir_fraction_red > 0.99 and verdict.tint.tir_fraction_blue > 0.99
            deviations.append(abs(verdict.tint.ratio - 1.0))
    assert 0.03 < max(deviations) <= C.CALIBRATION_WHITE_MAX_DEVIATION
    assert C.TINT_RATIO_MIN == 1.0 + 2.0 * C.CALIBRATION_WHITE_MAX_DEVIATION


def test_dependency_direction() -> None:
    """``chromatic`` sits on top of ``dp_field``; no module of the field layer (or below) imports it."""
    from _dependency_direction import PACKAGE_ROOT, imported_modules

    offenders = {}
    for path in PACKAGE_ROOT.rglob("*.py"):
        # The fixed diagnostic reference is a documented assembler above
        # chromatic, not part of the field layer whose dependency direction
        # this guard protects.
        if path.name in {"chromatic.py", "raypath_diagnostic_reference.py"}:
            continue
        relative = path.relative_to(PACKAGE_ROOT).with_suffix("")
        package = ".".join(("lumice_integral", *relative.parts[:-1]))  # a module's (or an __init__'s) own package
        hits = {m for m in imported_modules(path, package) if m.startswith("lumice_integral.chromatic")}
        if hits:
            offenders[str(relative)] = sorted(hits)
    assert offenders == {}
    assert "lumice_integral.dp_field.DPField" in imported_modules(PACKAGE_ROOT / "chromatic.py", "lumice_integral")


# ---- one-sided cases are reported, never "no colour" (owner, code-review Minor 4/6/8) ----------------------


def _tint(red: float, blue: float):
    from dataclasses import fields

    from lumice_integral.chromatic import TintMetrics

    values = {"energy_red": red, "energy_blue": blue}
    for f in fields(TintMetrics):
        values.setdefault(f.name, float("nan") if f.name == "ratio" else 0.0)
    return TintMetrics(**values)


@pytest.mark.parametrize("red, blue, kind, color", [(0.0, 1.0, "tint", "blue"), (1.0, 0.0, "tint", "red"), (0.0, 0.0, "none", "none")])
def test_class_lit_at_one_index_is_the_extreme_tint(red, blue, kind, color):
    from lumice_integral.chromatic import _tint_verdict

    verdict = _tint_verdict((1, 3, 5, 2), _tint(red, blue), 1.307, 1.317)
    assert (verdict.kind, verdict.color) == (kind, color)
    assert verdict.notes


def test_one_sided_line_without_features_is_unresolved_not_none():
    from lumice_integral.chromatic import _verdict

    verdict = _verdict((3, 1, 5), (), ("internal_1_tir_discriminant: weight kink at n = 1.317 only (not assessed)",), 1.307, 1.317, True)
    assert (verdict.kind, verdict.color, verdict.visible) == ("unresolved", "none", False)
    assert _verdict((3, 1, 5), (), (), 1.307, 1.317).kind == "none"


# ---- coverage and lookups (task chromatic-consolidation) ------------------------------------------------------


def test_a_failed_kink_walk_marks_the_verdict_incomplete(monkeypatch) -> None:
    """``3-7-5``'s kink is marched; one failed seed walk leaves the verdict standing but ``coverage_complete`` False."""
    from lumice_integral.dp_field import weight_kink as W

    assert C.diagnose(HexPrism(), (3, 7, 5)).coverage_complete
    walk = W._walk_both_ways
    calls = []

    def first_fails(walker, start, margin):
        calls.append(start)
        if len(calls) == 1:
            raise RuntimeError("injected walk failure")
        return walk(walker, start, margin)

    monkeypatch.setattr(W, "_walk_both_ways", first_fails)
    verdict = C.diagnose(HexPrism(), (3, 7, 5))
    assert not verdict.coverage_complete
    assert any("injected walk failure" in note for note in verdict.notes)


def test_kink_feature_reads_the_incidence_cosine_by_name(monkeypatch) -> None:
    """Reverse the margin layout (names and columns together): the kink feature is unchanged."""
    import types

    from lumice_integral import optics

    red, blue = (DPField.build(HexPrism(), (3, 1, 5), n) for n in (C.N_RED, C.N_BLUE))
    kink_red, kink_blue = red.weight_kinks[0], blue.weight_kinks[0]
    expected = C._kink_feature(red, blue, kink_red, kink_blue)

    class Reversed:
        def __init__(self, field):
            self._field = field

        def __getattr__(self, name):
            return getattr(self._field, name)

        def margins_batch(self, u):
            return self._field.margins_batch(u)[:, ::-1]

        def index_derivatives_batch(self, u):
            d, m = self._field.index_derivatives_batch(u)
            return d, m[:, ::-1]

    proxy = types.SimpleNamespace(**{name: getattr(optics, name) for name in dir(optics) if not name.startswith("__")})
    proxy.domain_margin_names = lambda faces, *args: tuple(reversed(optics.domain_margin_names(faces, *args)))
    monkeypatch.setattr(C, "optics", proxy)
    assert C._kink_feature(Reversed(red), Reversed(blue), kink_red, kink_blue) == expected


def test_every_fixture_verdict_is_complete() -> None:
    for faces in PROBE_MONO_PATHS:
        assert C.diagnose(HexPrism(), faces).coverage_complete, faces
    assert C.diagnose_class(HexPrism(), (3, 1, 6), C.RandomOrientation()).verdict.coverage_complete
