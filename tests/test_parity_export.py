"""LI -> Lumice parity fixtures (``lumice_integral.parity_export``, ``scripts/export_analytic_parity.py``).

The format (bit-exact float round trip, crystal blocks, field shapes), one ``3-5`` cell exported on a small
store and read back, tampered fixtures that the read-back must reject, and the full matrix exported twice by
two independent interpreters, byte for byte, and read back.
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from lumice_integral import parity_export as pe
from lumice_integral.canonical_scene import CANONICAL_REFRACTIVE_INDEX, canonical_crystal, canonical_sun_direction
from lumice_integral.discovery import DISCOVERY_EVENT_NAMES
from lumice_integral.geometry import Pyramid
from lumice_integral.optics import fresnel_transmission_path, path_direction
from lumice_integral.s2_store import StoreSeeds

REPO = Path(__file__).resolve().parent.parent
SUN = tuple(float(x) for x in canonical_sun_direction())
PRISM = pe.prism_crystal(1.0)
SCENE_3_5 = pe.Scene(PRISM, (3, 5), CANONICAL_REFRACTIVE_INDEX, SUN)
SMALL_N = 10_000


def test_json_round_trip_is_bit_exact_for_float64() -> None:
    values = np.array([1.0 / 3.0, 1e-300, -0.0, 5e-324, np.nextafter(1.0, 2.0), -2.718281828459045, 1e300])
    back = np.asarray(json.loads(pe.dumps({"x": values}))["x"])
    assert back.tobytes() == values.tobytes()  # -0.0 and subnormals included, bit for bit
    with pytest.raises(ValueError):
        pe.dumps({"x": float("nan")})


def test_dumps_is_canonical() -> None:
    assert pe.dumps({"b": 1, "a": np.float64(0.1)}) == pe.dumps({"a": 0.1, "b": np.int64(1)})


def test_prism_block_builds_the_canonical_column() -> None:
    assert np.array_equal(pe.build_crystal(PRISM).vertices, canonical_crystal().vertices)
    assert PRISM["upper_h"] == PRISM["lower_h"] == PRISM["upper_wedge_deg"] == PRISM["lower_wedge_deg"] == 0.0


def test_pyramid_block_is_the_miller_pyramid_through_its_wedge_angles() -> None:
    by_miller = Pyramid.from_lumice(0.5, 0.25, 0.6, (1, 0, 1), (2, 0, 3), face_distance=(1, 1.1, 0.9, 1, 1.2, 0.95))
    spec = pe.pyramid_crystal(
        0.5, 0.25, 0.6, pe.miller_wedge_deg((1, 0, 1)), pe.miller_wedge_deg((2, 0, 3)), (1, 1.1, 0.9, 1, 1.2, 0.95)
    )
    by_wedge = pe.build_crystal(spec)
    assert by_wedge.shape.upper_c_over_a == pytest.approx(by_miller.shape.upper_c_over_a, rel=1e-14)
    assert by_wedge.shape.lower_c_over_a == pytest.approx(by_miller.shape.lower_c_over_a, rel=1e-14)
    np.testing.assert_allclose(by_wedge.vertices, by_miller.vertices, rtol=0.0, atol=1e-13)


def test_evaluate_path_fields_follow_the_api_shapes() -> None:
    """3-5-6-7: face_count + 1 body-frame segments, face_count interfaces, their product is the Fresnel factor."""
    scene = pe.Scene(PRISM, (3, 5, 6, 7), CANONICAL_REFRACTIVE_INDEX, SUN)
    u, _ = pe.random_u(scene, 20260928)
    pose = pe.pose_of_u(scene, u)
    result = pe.evaluate_path(scene, pose)
    assert result["valid"]
    assert result["segment_directions"].shape == (5, 3) and result["interface_transmittances"].shape == (4,)
    np.testing.assert_allclose(result["segment_directions"][0], pose.T @ scene.incident_direction, atol=1e-15)
    np.testing.assert_allclose(result["segment_directions"][-1], pose.T @ result["outgoing_direction"], atol=1e-15)
    assert np.prod(result["interface_transmittances"]) == pytest.approx(result["fresnel_transmission"], rel=1e-14)
    direct = path_direction(pose, scene.faces, scene.incident_direction, scene.refractive_index, crystal=scene.crystal)
    assert np.array_equal(result["outgoing_direction"], np.asarray(direct.direction))
    assert result["fresnel_transmission"] == fresnel_transmission_path(
        pose, scene.faces, scene.incident_direction, scene.refractive_index, crystal=scene.crystal
    )


def test_pose_of_u_puts_u_on_the_sun_and_the_target_in_the_reference_half_plane() -> None:
    u, _ = pe.random_u(SCENE_3_5, 1)
    pose = pe.pose_of_u(SCENE_3_5, u)
    np.testing.assert_allclose(pose @ u, SCENE_3_5.sun, atol=1e-15)
    target = np.asarray(path_direction(pose, (3, 5), SCENE_3_5.incident_direction, CANONICAL_REFRACTIVE_INDEX).direction)
    _, deviation = pe.body_outgoing(SCENE_3_5, u)
    assert np.arccos(target @ SCENE_3_5.incident_direction) == pytest.approx(deviation, abs=1e-12)
    normal = np.cross(SCENE_3_5.incident_direction, pe.AZIMUTH_REFERENCE)
    assert abs(target @ normal) < 1e-12 and target @ pe.AZIMUTH_REFERENCE > SCENE_3_5.incident_direction @ pe.AZIMUTH_REFERENCE


def test_a_path_without_an_interior_extremum_has_no_critical_cell() -> None:
    scene = pe.Scene(PRISM, (3, 5, 6, 7), CANONICAL_REFRACTIVE_INDEX, SUN)
    with pytest.raises(pe.PointUnavailable, match="no interior extremum"):
        pe.choose_point(pe.Cell(scene, "critical", "test", store_n=SMALL_N))


def test_near_boundary_point_sits_at_the_requested_margin_with_its_mirror_outside() -> None:
    choice = pe.choose_point(pe.Cell(SCENE_3_5, "near_boundary", "test", store_n=SMALL_N))
    assert choice.selection["nearest_margin_value"] == pytest.approx(1e-3, rel=1e-9)
    assert pe.evaluate_path(SCENE_3_5, choice.pose)["valid"]
    assert not pe.evaluate_path(SCENE_3_5, choice.outside_pose)["valid"]


def test_curve_distance_measures_the_polyline_not_its_samples() -> None:
    from lumice_integral.so3 import exp

    axis = np.array([0.0, 0.0, 1.0])
    coarse = np.stack([np.asarray(exp(theta * axis)) for theta in np.linspace(0.0, 1.0, 11)])
    fine = np.stack([np.asarray(exp(theta * axis)) for theta in np.linspace(0.0, 1.0, 101)])
    # same geodesic, different sampling: only the densification spacing is left
    assert pe.curve_distance(coarse, False, fine, False) <= pe.CURVE_DENSIFY_SPACING / 2 + 1e-12
    shifted = fine @ np.asarray(exp(np.array([0.01, 0.0, 0.0])))
    assert pe.curve_distance(coarse, False, shifted, False) == pytest.approx(0.01, abs=pe.CURVE_DENSIFY_SPACING / 2)
    assert pe.curve_distance(coarse[:6], False, coarse, False) == pytest.approx(0.5, rel=1e-9)  # a missing half


@pytest.fixture(scope="module")
def exported_cell(tmp_path_factory) -> tuple[Path, dict]:
    directory = tmp_path_factory.mktemp("parity")
    cell = pe.Cell(SCENE_3_5, "random", "fast-tier smoke", store_n=SMALL_N)
    manifest = pe.export_matrix([cell], directory)
    return directory, manifest


def test_one_cell_exports_every_fixture_kind_with_provenance(exported_cell) -> None:
    directory, manifest = exported_cell
    entry = manifest["cells"][0]
    assert entry["skipped"] == []
    kinds = {}
    for name in entry["files"]:
        fixture = pe.read_json(directory / name)
        kinds[fixture["fixture_kind"]] = fixture
        assert fixture["symmetry_semantics"] == "none" and fixture["schema_version"] == pe.SCHEMA_VERSION
        assert len(fixture["provenance"]["li_rev"]) == 40 and len(fixture["provenance"]["conventions_sha256"]) == 64
        for tolerance in fixture["tolerance"].values():
            assert tolerance["basis"]
    assert set(kinds) == {"evaluate_path", "trace_fiber", "seed_search"}
    expected = kinds["seed_search"]["expected"]
    assert set(expected["events"]) == set(DISCOVERY_EVENT_NAMES)
    assert expected["admissible_count"] == (
        expected["events"]["dedup_merged"] + len(expected["components"]) + len(expected["incomplete"])
    )
    assert expected["pool_count"] == len(kinds["seed_search"]["input"]["sample"]["band_deviation"])


def test_exported_band_reproduces_the_store_candidates(exported_cell) -> None:
    directory, manifest = exported_cell
    fixture = pe.read_json(directory / "3-5__random__seed_search.json")
    target = np.asarray(fixture["input"]["target_direction"])
    sample = fixture["input"]["sample"]
    half_width = np.radians(fixture["input"]["band_half_width_deg"])
    band = pe.BandSeeds(
        SCENE_3_5, np.asarray(sample["band_u"]), np.asarray(sample["band_phi"]), np.asarray(sample["band_deviation"]), target, half_width
    )
    store = StoreSeeds(pe.scene_store(SCENE_3_5, SMALL_N), (3, 5), canonical_sun_direction())
    for mine, theirs in zip(band.candidates(target, half_width), store.candidates(target, half_width)):
        assert np.array_equal(mine, theirs)


def test_every_exported_fixture_reads_back(exported_cell) -> None:
    directory, _ = exported_cell
    checks = pe.verify_directory(directory)
    assert len(checks) == 4
    assert all(not check.failures for check in checks), [check.failures for check in checks]


def _tampered(directory: Path, name: str, edit) -> pe.Check:
    fixture = copy.deepcopy(pe.read_json(directory / name))
    edit(fixture)
    return pe.VERIFIERS[fixture["fixture_kind"]](fixture, name)


def _nudge_curve(fixture: dict) -> None:
    from lumice_integral.so3 import exp

    turn = np.asarray(exp(np.array([0.0, 0.02, 0.0])))
    poses = np.asarray(fixture["expected"]["traces"][0]["poses"]).reshape(-1, 3, 3) @ turn
    fixture["expected"]["traces"][0]["poses"] = poses.reshape(-1, 9).tolist()


@pytest.mark.parametrize(
    "name, edit, message",
    [
        (
            "3-5__random__evaluate_path__point.json",
            lambda f: f["expected"]["outgoing_direction"].__setitem__(0, f["expected"]["outgoing_direction"][0] + 1e-9),
            "outgoing_direction",
        ),
        ("3-5__random__evaluate_path__point.json", lambda f: f["expected"].__setitem__("valid", False), "valid"),
        ("3-5__random__trace_fiber.json", _nudge_curve, "curve distance"),
        ("3-5__random__trace_fiber.json", lambda f: f["expected"]["traces"][0].__setitem__("reason", "tir_boundary"), "status/reason"),
        ("3-5__random__seed_search.json", lambda f: f["expected"].__setitem__("pool_count", f["expected"]["pool_count"] + 1), "pool_count"),
        (
            "3-5__random__seed_search.json",
            lambda f: f["expected"]["events"].__setitem__("dedup_merged", f["expected"]["events"]["dedup_merged"] + 1),
            "events",
        ),
    ],
)
def test_the_read_back_rejects_a_tampered_fixture(exported_cell, name, edit, message) -> None:
    directory, _ = exported_cell
    check = _tampered(directory, name, edit)
    assert any(message in failure for failure in check.failures), check.failures


def _export(output: Path) -> subprocess.CompletedProcess:
    env = {**os.environ, "JAX_PLATFORMS": "cpu"}
    return subprocess.run(
        [sys.executable, str(REPO / "scripts" / "export_analytic_parity.py"), "--output-dir", str(output)],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )


# Two full-matrix exports in separate interpreters plus a read-back: 16 s with a cold JAX cache on an M2 Max
# (2026-09-28), under the ~20 s slow-tier threshold of AGENTS.md, so CI checks the determinism on Linux too.
def test_full_matrix_export_is_byte_deterministic_and_reads_back(tmp_path: Path) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    _export(first)
    _export(second)
    names = sorted(path.name for path in first.iterdir())
    assert names == sorted(path.name for path in second.iterdir())
    for name in names:
        assert (first / name).read_bytes() == (second / name).read_bytes(), name
    manifest = pe.read_json(first / pe.MANIFEST)
    cells = {entry["name"]: entry for entry in manifest["cells"]}
    assert len(cells) == 9
    assert [item["fixture"] for item in cells["3-5-6-7__critical"]["skipped"]] == ["all"]
    for category in pe.CATEGORIES:
        assert [item["fixture"] for item in cells[f"13-15-26-28__{category}"]["skipped"]] == ["seed_search"]
    checks = pe.verify_directory(first)
    assert len(checks) == len(names) - 1
    assert all(not check.failures for check in checks), [(check.fixture, check.failures) for check in checks if check.failures]
