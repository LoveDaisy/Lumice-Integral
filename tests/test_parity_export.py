"""LI -> Lumice parity fixtures (``lumice_integral.parity_export``, ``scripts/export_analytic_parity.py``).

The format (bit-exact float round trip, crystal blocks, field shapes), one ``3-5`` cell exported on a small
store and read back, tampered fixtures that the read-back must reject, and the full matrix exported twice by
two independent interpreters, byte for byte, and read back.  The module B ``band_sum`` fixtures
(``docs/band-sum-contract.md``): pixel tables, singular pixels, both layers of the read-back and the red
states, and their byte determinism.
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


MATRIX_CELLS = [f"{path}__{category}" for path in ("3-5", "3-5-6-7", "13-15-26-28") for category in pe.CATEGORIES]


def _export(output: Path, cells: list[str] | None = None) -> subprocess.CompletedProcess:
    env = {**os.environ, "JAX_PLATFORMS": "cpu"}
    return subprocess.run(
        [sys.executable, str(REPO / "scripts" / "export_analytic_parity.py"), "--output-dir", str(output)]
        + (["--cells", *cells] if cells else []),
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )


def _assert_byte_identical(first: Path, second: Path) -> list[str]:
    names = sorted(path.name for path in first.iterdir())
    assert names == sorted(path.name for path in second.iterdir())
    for name in names:
        assert (first / name).read_bytes() == (second / name).read_bytes(), name
    return names


# Two exports of the 3 x 3 matrix in separate interpreters plus a read-back: 18 s with a cold JAX cache on an
# idle M2 Max (2026-09-29, with the pyramid's seed search), 22 s with a warm cache at load average 33 after the
# wave 2 per-pose checks; about the ~20 s slow-tier threshold of AGENTS.md, kept fast so CI checks the
# determinism on Linux too.  The wave 2 edge cells are in the slow test below.
def test_full_matrix_export_is_byte_deterministic_and_reads_back(tmp_path: Path) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    _export(first, MATRIX_CELLS)
    _export(second, MATRIX_CELLS)
    names = _assert_byte_identical(first, second)
    manifest = pe.read_json(first / pe.MANIFEST)
    assert "edge_cells" not in manifest and "band_sum_cells" not in manifest  # a matrix-only export keeps the v0 manifest
    cells = {entry["name"]: entry for entry in manifest["cells"]}
    assert len(cells) == 9
    assert [item["fixture"] for item in cells["3-5-6-7__critical"]["skipped"]] == ["all"]
    for category in pe.CATEGORIES:
        assert cells[f"13-15-26-28__{category}"]["skipped"] == []
    # The pyramid's seed search: components at the random and critical targets; the near-boundary point sits
    # next to a Snell gate where the finite crystal's entry measure is already zero (corridor_empty), so its
    # target lies outside the store's lit D range and the band is empty (docs/analytic-parity-fixtures.md 6).
    pyramid = {
        category: pe.read_json(first / f"13-15-26-28__{category}__seed_search.json")["expected"] for category in pe.CATEGORIES
    }
    assert all(pyramid[category]["completeness"] == "complete" for category in pe.CATEGORIES)
    assert len(pyramid["random"]["components"]) > 0 and len(pyramid["critical"]["components"]) > 0
    assert pyramid["near_boundary"]["pool_count"] == 0 and pyramid["near_boundary"]["components"] == []
    checks = pe.verify_directory(first)
    assert len(checks) == len(names) - 1
    assert all(not check.failures for check in checks), [(check.fixture, check.failures) for check in checks if check.failures]


# slow: two full exports (matrix and the wave 2 edge cells, 82 fixtures) plus a read-back, 40 s with a warm JAX cache on a loaded M2 Max
@pytest.mark.slow
def test_export_with_edge_cells_is_byte_deterministic_and_reads_back(tmp_path: Path) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    _export(first)
    _export(second)
    names = _assert_byte_identical(first, second)
    manifest = pe.read_json(first / pe.MANIFEST)
    edge = {entry["name"]: entry for entry in manifest["edge_cells"]}
    assert all(entry["skipped"] == [] for entry in edge.values())
    assert all(entry["serves"] and entry["rationale"] for entry in edge.values())
    # Seed search on the edge targets: rows 700/780 fold every candidate into one closed loop (C16), 1-3 at
    # 60 deg is two arcs (C17/C18), 13-24-26 one arc cut by the path domain at both ends (C19).
    search = {name: pe.read_json(first / f"{name}__seed_search.json")["expected"] for name in edge if f"{name}__seed_search.json" in names}
    for row in (700, 780):
        result = search[f"3-5__boundary_hugging_r{row}_c150"]
        assert [component["kind"] for component in result["components"]] == ["closed"] and result["completeness"] == "complete"
        assert result["events"]["dedup_merged"] == result["admissible_count"] - 1
    arcs = search["1-3__two_arcs_60deg"]["components"]
    assert [c["kind"] for c in arcs] == ["arc", "arc"]
    assert all({c["reason"], c["start_reason"]} == {"tir_boundary", "path_infeasible"} for c in arcs)
    (pyramid,) = search["13-24-26__boundary_arc_90deg"]["components"]
    assert pyramid["kind"] == "arc" and pyramid["reason"] == pyramid["start_reason"] == "path_infeasible"
    checks = pe.verify_directory(first)
    assert len(checks) == len(names) - 1
    assert all(not check.failures for check in checks), [(check.fixture, check.failures) for check in checks if check.failures]


# ------------------------------------------------------------------ band sum (module B)
LINEAR = {"kind": "linear", "render": {"width": 41, "height": 41, "fov_deg": 60.0, "view": {"azimuth": 0.0, "elevation": 15.0}}}
SUN_LAMBERT = {"kind": "lambert_azimuthal_equal_area", "centre_sky": list(SUN), "field_radius_deg": 30.0, "size": 65}
BAND_CELLS = [
    pe.BandSumCell("random", SCENE_3_5, {"family": "random"}, LINEAR, ((20, 20), (20, 12), (5, 20), (3, 33)), SMALL_N, "fast-tier smoke"),
    pe.BandSumCell(
        "plate", SCENE_3_5, {"family": "plate", "zenith_std_deg": 1.0}, SUN_LAMBERT, ((32, 32), (31, 8), (29, 7), (32, 57)), SMALL_N, "fast-tier smoke"
    ),
    pe.BandSumCell("rank0", pe.Scene(PRISM, (3, 6), CANONICAL_REFRACTIVE_INDEX, SUN), {"family": "random"}, LINEAR, ((20, 20), (20, 21)), SMALL_N, "smoke"),
]


@pytest.fixture(scope="module")
def band_fixtures(tmp_path_factory) -> tuple[Path, dict]:
    directory = tmp_path_factory.mktemp("parity-band")
    return directory, pe.export_matrix([], directory, band_sum_cells=BAND_CELLS)


def test_pixel_containment_is_the_closed_spherical_quadrilateral_on_the_corners_side() -> None:
    corners = np.array([[1.0, -0.1, -0.1], [1.0, 0.1, -0.1], [1.0, 0.1, 0.1], [1.0, -0.1, 0.1]])
    corners /= np.linalg.norm(corners, axis=1, keepdims=True)
    assert pe.pixel_contains(corners, np.array([1.0, 0.0, 0.0])) and pe.pixel_contains(corners[::-1], np.array([1.0, 0.0, 0.0]))
    assert not pe.pixel_contains(corners, np.array([-1.0, 0.0, 0.0]))  # the antipode passes the sign test alone
    assert not pe.pixel_contains(corners, np.array([1.0, 0.2, 0.0]) / np.hypot(1.0, 0.2))
    assert pe.pixel_contains(corners, corners[0])  # closed: a corner belongs to the pixel


def test_linear_pixel_table_is_the_cameras_directions_in_cyclic_order() -> None:
    from lumice_integral.band_sum import pixel_band
    from lumice_integral.path_class import sun_pixel

    table = pe.linear_pixel_table(LINEAR["render"], [(20, 20), (5, 20)])
    for label, centre, corners in zip(table["labels"], table["centre"], table["corners"]):
        band = pixel_band(*label, np.asarray(SUN), LINEAR["render"])
        assert np.array_equal(band[0], centre)
        edges = [np.linalg.norm(corners[k] - corners[(k + 1) % 4]) for k in range(4)]
        assert max(edges) < 1.2 * min(edges)  # consecutive corners are neighbours, not diagonals (sqrt 2 longer)
    s = SCENE_3_5.incident_direction
    assert pe.pixel_contains(table["corners"][0], s) and tuple(table["labels"][0]) == sun_pixel(LINEAR["render"], s)


def test_lambert_pixels_are_equal_area_and_tile_the_cap() -> None:
    """Every pixel's solid angle is 1/k^2, and the pixels inside the disk add up to the cap's area."""
    size, radius = 33, 20.0
    view = pe.lambert_view(SUN, radius, size)
    inside = [(y, x) for y in range(size) for x in range(size) if np.hypot(x + 0.5 - size / 2, y + 0.5 - size / 2) <= 0.492 * size - 1.0]
    table = pe.lambert_pixel_table(SUN, radius, size, inside)
    polygon = [
        abs(float(np.linalg.det(np.stack([c[0], c[1], c[2]])))) + abs(float(np.linalg.det(np.stack([c[0], c[2], c[3]])))) for c in table["corners"]
    ]
    # a unit-sphere quadrilateral's area ~ the flat quad spanned by its corners (sub-pixel curvature ~1e-4)
    assert np.allclose(0.5 * np.array(polygon), table["solid_angle"], rtol=1e-3)
    assert np.isclose(table["solid_angle"][0], 1.0 / view["k"] ** 2)
    assert np.allclose(np.linalg.norm(table["centre"], axis=1), 1.0)
    assert np.isclose(-table["centre"][inside.index((16, 16))] @ np.asarray(SUN), 1.0)  # the view is centred on the sky point


def test_band_sum_fixtures_cover_the_statuses_and_read_back(band_fixtures) -> None:
    directory, manifest = band_fixtures
    assert [entry["name"] for entry in manifest["band_sum_cells"]] == [cell.name for cell in BAND_CELLS]
    random = pe.read_json(directory / "3-5__band_sum_random.json")
    assert random["fixture_kind"] == "band_sum" and random["symmetry_semantics"] == "none" and random["cell"]["rank"] == 2
    statuses = [p["status"] for p in random["expected"]["pixels"]]
    assert statuses[0] == "singular" and set(statuses[1:]) == {"ok"}
    assert random["expected"]["pixels"][1]["K"] == 0 and random["expected"]["pixels"][2]["K"] > 10
    plate = pe.read_json(directory / "3-5__band_sum_plate.json")
    mirror = plate["expected"]["pixels"][3]
    assert mirror["K"] > 0 and mirror["K_rho_pos"] == 0 and mirror["value"] == 0.0  # a narrow family: rho = 0 on every band event
    rank0 = pe.read_json(directory / "3-6__band_sum_rank0.json")
    assert [p["status"] for p in rank0["expected"]["pixels"]] == ["point_mass", "ok"] and rank0["cell"]["rank"] == 0
    for fixture in (random, plate):
        for tolerance in fixture["tolerance"].values():
            assert tolerance["basis"]
    checks = pe.verify_directory(directory)
    assert len(checks) == len(BAND_CELLS) and all(not check.failures for check in checks), [check.failures for check in checks]


def _move_one_band_event(fixture: dict) -> None:
    """Shift a band-centre event of the brightest pixel far below every band (not a boundary event)."""
    pixels = [p for p in fixture["expected"]["pixels"] if p["status"] == "ok"]
    brightest = max(pixels, key=lambda p: p["K"])
    deviation = np.asarray(fixture["input"]["events"]["deviation"])
    centre = 0.5 * (brightest["delta_lo"] + brightest["delta_hi"])
    index = int(np.argmin(np.abs(deviation - centre)))
    assert min(abs(deviation[index] - brightest["delta_lo"]), abs(deviation[index] - brightest["delta_hi"])) > 1e-4
    deviation[index] = 1e-3
    fixture["input"]["events"]["deviation"] = deviation
    order = np.argsort(deviation, kind="stable")
    for key in ("u", "phi", "deviation", "w"):
        fixture["input"]["events"][key] = np.asarray(fixture["input"]["events"][key])[order].tolist()


@pytest.mark.parametrize(
    "name, edit, message",
    [
        ("3-5__band_sum_random.json", _move_one_band_event, "layer 1 scatter: pixel"),
        ("3-5__band_sum_random.json", _move_one_band_event, "not the regenerated sample"),
        ("3-5__band_sum_plate.json", lambda f: f["input"].__setitem__("refractive_index", 1.32), "layer 2: pixel"),
        (
            "3-5__band_sum_plate.json",
            lambda f: f["expected"]["pixels"][1].__setitem__("value", f["expected"]["pixels"][1]["value"] * (1.0 + 1e-8)),
            "value",
        ),
        ("3-5__band_sum_plate.json", lambda f: f["expected"]["pixels"][0].__setitem__("status", "ok"), "status"),
        ("3-5__band_sum_plate.json", lambda f: f["input"]["pose_density"].__setitem__("zenith_std_deg", 1.01), "layer 1 gather: pixel"),
        ("3-6__band_sum_rank0.json", lambda f: f["input"]["events"]["w"].pop(), "layer 1 point mass"),
    ],
)
def test_the_band_sum_read_back_rejects_a_tampered_fixture(band_fixtures, name, edit, message) -> None:
    directory, _ = band_fixtures
    check = _tampered(directory, name, edit)
    assert any(message in failure for failure in check.failures), check.failures


# Two exports of three band-sum cells in separate interpreters: about 10 s.  The full set (with the 4e6-sample
# rank-0 plate cell) is exported by the slow test above.
def test_band_sum_export_is_byte_deterministic(tmp_path: Path) -> None:
    cells = ["3-5__band_sum_plate", "3-5__band_sum_parry", "3-5-6-7__band_sum_random"]
    first, second = tmp_path / "first", tmp_path / "second"
    _export(first, cells)
    _export(second, cells)
    names = _assert_byte_identical(first, second)
    assert len(names) == len(cells) + 1
    manifest = pe.read_json(first / pe.MANIFEST)
    assert manifest["cells"] == [] and "edge_cells" not in manifest and len(manifest["band_sum_cells"]) == len(cells)
