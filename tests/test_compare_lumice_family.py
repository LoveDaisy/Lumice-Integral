"""``scripts/compare_lumice_family.py``: its vectorised pixel solid angles are the probe's, on any linear camera."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from lumice_integral.canonical_scene import CANONICAL_RENDER, canonical_crystal
from lumice_integral.geometry import HexPrism
from lumice_integral.path_class import build_path_class

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def family():
    return _load("compare_lumice_family")


@pytest.mark.parametrize(
    "render",
    [
        dict(CANONICAL_RENDER),
        {"width": 321, "height": 161, "fov_deg": 32.0, "view": {"azimuth": 0.0, "elevation": 15.0}},
        {"width": 401, "height": 401, "fov_deg": 100.0, "view": {"azimuth": 0.0, "elevation": 15.0}},
    ],
)
def test_pixel_solid_angles_match_the_probe(render, family):
    probe = _load("probe_absolute_scale")
    grid = family.pixel_solid_angles(render)
    assert grid.shape == (render["height"], render["width"])
    for row, column in ((0, 0), (render["height"] // 2, render["width"] // 2), (render["height"] - 1, render["width"] - 3), (7, render["width"] // 3)):
        assert grid[row, column] == pytest.approx(probe.pixel_solid_angle(row, column, render), rel=1e-12)


def test_family_imports_pixel_solid_angles_and_merged_relative_noise_rather_than_redefining_them(family):
    """a56: these must be re-exported from probe_absolute_scale, not a second local implementation.

    ``__module__`` is set once, at function-definition time in ``probe_absolute_scale.py``, and is
    unaffected by which module *object* currently sits under that name in ``sys.modules`` (this test
    file's ``_load`` helper reloads modules by file path, which would make a plain identity check
    order-dependent); so it is the robust way to assert "``family`` imports this, it does not define
    a second copy of it".
    """
    assert family.pixel_solid_angles.__module__ == "probe_absolute_scale"
    assert family.merged_relative_noise.__module__ == "probe_absolute_scale"


def test_flux_ratio_sums_measured_over_predicted_on_the_mask(family):
    measured = np.array([1.0, 2.0, 3.0, 100.0])
    predicted = np.array([2.0, 2.0, 3.0, 100.0])
    mask = np.array([True, True, True, False])
    assert family.flux_ratio(measured, predicted, mask) == pytest.approx(6.0 / 7.0)


def test_flux_ratio_is_nan_when_predicted_is_zero_on_the_mask(family):
    measured = np.array([1.0, 2.0])
    predicted = np.array([0.0, 0.0])
    mask = np.array([True, True])
    assert np.isnan(family.flux_ratio(measured, predicted, mask))


def test_profile_rms_of_identical_profiles_is_zero(family):
    a = np.array([0.0, 1.0, 2.0, 10.0, 2.0, 1.0, 0.0])
    rms, lit = family.profile_rms(a, a.copy(), floor=0.05)
    assert rms == pytest.approx(0.0, abs=1e-12)
    assert lit == int((a / a.max() > 0.05).sum())


def test_profile_rms_matches_a_direct_computation_on_shifted_profiles(family):
    a = np.array([0.0, 5.0, 10.0, 5.0, 0.0])
    b = np.array([0.0, 4.0, 8.0, 6.0, 0.0])
    floor = 0.1
    na, nb = a / a.max(), b / b.max()
    lit = (na > floor) | (nb > floor)
    expected_rms = float(np.sqrt(np.mean((na[lit] - nb[lit]) ** 2)))
    rms, count = family.profile_rms(a, b, floor)
    assert rms == pytest.approx(expected_rms)
    assert count == int(lit.sum())


def test_profile_rms_raises_when_a_profile_is_all_zero(family):
    zero = np.zeros(5)
    nonzero = np.array([0.0, 1.0, 2.0, 1.0, 0.0])
    with pytest.raises(ValueError):
        family.profile_rms(zero, nonzero, floor=0.05)
    with pytest.raises(ValueError):
        family.profile_rms(nonzero, zero, floor=0.05)


D3H = HexPrism.from_lumice(1.0, (1.0, 1.2, 1.0, 1.2, 1.0, 1.2))
G2 = HexPrism.from_lumice(1.0, (1.0, 1.3, 0.7, 1.9, 1.1, 0.4))


def _config(filters: list[dict], top: int) -> dict:
    return {"filter": filters, "scene": {"scattering": [{"prob": 0, "entries": [{"crystal": 1, "proportion": 100, "filter": top}]}]}}


def _exact(members) -> list[dict]:
    return [{"id": k + 1, "type": "raypath", "raypath": list(m), "action": "filter_in"} for k, m in enumerate(members)]


def _complex(members) -> dict:
    parts = _exact(members)
    return _config([*parts, {"id": 99, "type": "complex", "composition": [p["id"] for p in parts]}], 99)


def _members(crystal, path):
    return [list(m) for m in build_path_class(crystal, path).members]


def test_check_filter_accepts_pbd_only_on_the_regular_prism(family):
    pbd = _config([{"id": 1, "type": "raypath", "raypath": [3, 5], "symmetry": "PBD", "action": "filter_in"}], 1)
    assert family.check_filter(pbd, [3, 5], _members(canonical_crystal(), [3, 5]), canonical_crystal()) == "PBD"
    with pytest.raises(SystemExit, match="smaller than D6h"):
        family.check_filter(pbd, [3, 5], _members(D3H, [3, 5]), D3H)
    with pytest.raises(SystemExit, match="Lumice PBD admits"):  # a single-path render is not the PBD class
        family.check_filter(pbd, [3, 5], [[3, 5]], canonical_crystal())


@pytest.mark.parametrize(("crystal", "path"), [(D3H, [3, 5]), (D3H, [3, 5, 6, 7]), (G2, [3, 5]), (canonical_crystal(), [3, 5])])
def test_check_filter_accepts_exact_members_alone_or_ored(family, crystal, path):
    members = _members(crystal, path)
    assert family.check_filter(_complex(members[::-1]), path, members, crystal) == "complex"
    if len(members) == 1:
        assert family.check_filter(_config(_exact(members), 1), path, members, crystal) == "exact"


@pytest.mark.parametrize(
    "change",
    [
        lambda m: m[:-1],  # a member missing
        lambda m: [*m, [4, 6, 7, 8]],  # an extra path (a PBD image that is not a G_true image on D3h)
        lambda m: [m[0][::-1], *m[1:]],  # a member's faces in the wrong order
        lambda m: [*m, m[0]],  # a member twice
    ],
)
def test_check_filter_rejects_a_different_member_set(family, change):
    members = _members(D3H, [3, 5, 6, 7])
    with pytest.raises(SystemExit, match="admits"):
        family.check_filter(_complex(change(members)), [3, 5, 6, 7], members, D3H)


def test_check_filter_rejects_folded_or_non_raypath_parts(family):
    members = _members(D3H, [3, 5])
    config = _complex(members)
    config["filter"][0]["symmetry"] = "P"
    with pytest.raises(SystemExit, match="without symmetry"):
        family.check_filter(config, [3, 5], members, D3H)
    nested = _complex(members)
    nested["filter"][-1]["composition"][0] = [1, 2]
    with pytest.raises(SystemExit, match="without symmetry"):
        family.check_filter(nested, [3, 5], members, D3H)


# ------------------------------------------------------------------ colour
def _lumice_config(render: dict, spectrum) -> dict:
    config = _config([{"id": 1, "type": "raypath", "raypath": [3, 5], "symmetry": "PBD", "action": "filter_in"}], 1)
    config["scene"]["light_source"] = {"type": "sun", "altitude": 15.0, "azimuth": 0, "diameter": 0, "spectrum": spectrum}
    config["render"] = [
        {"id": 1, "lens": {"type": "linear", "fov": render["fov_deg"]}, "resolution": [render["width"], render["height"]],
         "view": {"azimuth": render["view"]["azimuth"], "elevation": render["view"]["elevation"]}}
    ]  # fmt: skip
    return config


def _fake_run(run_dir: Path, image: np.ndarray, config: dict, emitted_energy: float) -> Path:
    import json

    run_dir.mkdir(parents=True)
    np.save(run_dir / "img_01.npy", image.astype(np.float32))
    (run_dir / "img_01.json").write_text(json.dumps({"emitted_energy": emitted_energy}))
    (run_dir / "config.json").write_text(json.dumps(config))
    return run_dir


@pytest.fixture(scope="module")
def renders(tmp_path_factory):
    """The [3, 5] class on a small window: monochrome at n(550) and colour (discrete 550 nm, D65 M = 3), shared stores."""
    import dataclasses

    from lumice_integral.band_sum import BandSumScene, render_band_sum_window, write_band_sum_strip
    from lumice_integral.canonical_scene import canonical_sun_direction
    from lumice_integral.pose_density import build_pose_density
    from lumice_integral.pose_density_provenance import pose_density_provenance
    from lumice_integral.spectrum import IlluminantType, dispersion, emitted_weight, wavelength_pool
    from lumice_integral.spectrum.xyz_band_sum import render_xyz_band_sum_window, write_xyz_band_sum_strip
    from lumice_integral.strip_io import Window

    root = tmp_path_factory.mktemp("renders")
    render = {"width": 61, "height": 41, "fov_deg": 60.0, "view": {"azimuth": 0.0, "elevation": 15.0}}
    window = Window((0, 41), (0, 61), 1)
    density = pose_density_provenance("random")
    scene = BandSumScene(
        path_class=build_path_class(canonical_crystal(), (3, 5)), crystal=canonical_crystal(), refractive_index=float("nan"),
        sun_direction=canonical_sun_direction(), pose_density=build_pose_density("random"), render=render,
    )  # fmt: skip
    common = {"window": window, "pose_density_block": density}
    mono_scene = dataclasses.replace(scene, refractive_index=dispersion.refractive_index(550.0))
    results, execution = render_band_sum_window(mono_scene, window, 100_000, base_dir=root / "stores", run_checks=False)
    write_band_sum_strip(root / "mono", results, scene=mono_scene, n=100_000, execution=execution, **common)
    out = {"root": root, "render": render}
    for name, pool, weight, spectrum in (
        ("discrete", wavelength_pool(1, discrete_wavelength_nm=550.0), 1.0, {"discrete_wavelength_nm": 550.0}),
        ("d65", wavelength_pool(3, illuminant=IlluminantType.D65), emitted_weight(illuminant=IlluminantType.D65), {"illuminant": "D65", "wavelength_count": 3}),
    ):
        results, execution = render_xyz_band_sum_window(scene, pool, weight, window, 100_000, base_dir=root / "stores", run_checks=False)
        write_xyz_band_sum_strip(
            root / name, results, scene=scene, pool=pool, spectrum=spectrum, emitted_weight=weight, store_n=100_000, execution=execution, **common
        )
    return out


def _runs(family, renders, li: str, spectrum, name: str) -> list[Path]:
    """Two fake Lumice seeds: the render's own prediction times independent noise (every channel)."""
    root, render = renders["root"], renders["render"]
    channels, _, provenance = family.load_li(root / li)
    omega = family.pixel_solid_angles(render) / (0.5 * family.total_surface_area(canonical_crystal()))
    image = np.stack([omega * channels.get(c, channels["Y"]) for c in family.CHANNELS], axis=-1)
    rng = np.random.default_rng(len(name))
    out = []
    for seed, energy in ((1, 2.0e6), (2, 3.0e6)):
        noisy = image * energy * (1.0 + 0.05 * rng.standard_normal(image.shape))
        out.append(_fake_run(root / "lumice" / name / f"run{seed}", np.maximum(noisy, 0.0), _lumice_config(render, spectrum), energy))
    return out


def test_a_discrete_550nm_colour_render_gives_the_monochrome_y_metrics_bit_for_bit(family, renders, tmp_path):
    import json

    runs = _runs(family, renders, "mono", [{"wavelength": 550, "weight": 1.0}], "mono")
    args = [a for r in runs for a in ("--lumice-run", str(r))]
    family.main(["--li-dir", str(renders["root"] / "mono"), *args, "--output", str(tmp_path / "mono.json")])
    family.main(["--li-dir", str(renders["root"] / "discrete"), *args, "--output", str(tmp_path / "colour.json")])
    mono = json.loads((tmp_path / "mono.json").read_text())
    colour = json.loads((tmp_path / "colour.json").read_text())
    assert mono["li_format"] == "lumice-integral.band-sum/v1" and colour["li_format"] == "lumice-integral.band-sum-xyz/v1"
    assert set(colour["channels"]) == {"X", "Y", "Z"} and "channels" not in mono
    for block in ("total", "bright", "regions", "profiles"):
        assert colour["channels"]["Y"][block] == mono[block], block
    assert mono["bright"]["pixels"] >= 20
    assert mono["total"]["measured_over_predicted"] == pytest.approx(1.0, abs=0.02)


def test_colour_metrics_are_the_monochrome_blocks_per_channel(family, renders, tmp_path):
    import json

    runs = _runs(family, renders, "d65", "D65", "d65")
    family.main(["--li-dir", str(renders["root"] / "d65"), *[a for r in runs for a in ("--lumice-run", str(r))], "--output", str(tmp_path / "d65.json")])
    out = json.loads((tmp_path / "d65.json").read_text())
    keys = {block: set(out["channels"]["Y"][block]) for block in ("total", "bright", "regions", "profiles")}
    for c in "XYZ":
        assert {block: set(out["channels"][c][block]) for block in keys} == keys
        assert out["channels"][c]["total"]["measured_over_predicted"] == pytest.approx(1.0, abs=0.02)
        assert len(out["channels"][c]["total"]["per_run"]) == 2
    assert out["channels"]["X"]["total"] != out["channels"]["Z"]["total"]
    chroma = out["chromaticity"]
    assert chroma["difference_xy"] == pytest.approx([0.0, 0.0], abs=5e-3)
    assert 0.2 < chroma["predicted_xy"][0] < 0.5 and out["spectrum"] == {"illuminant": "D65", "wavelength_count": 3}


def test_a_lumice_run_with_another_spectrum_is_refused(family, renders, tmp_path):
    runs = _runs(family, renders, "d65", "A", "wrong-illuminant")
    with pytest.raises(SystemExit, match="!= band-sum illuminant"):
        family.main(["--li-dir", str(renders["root"] / "d65"), "--lumice-run", str(runs[0]), "--output", str(tmp_path / "x.json")])
    runs = _runs(family, renders, "discrete", [{"wavelength": 600, "weight": 1.0}], "wrong-wavelength")
    with pytest.raises(SystemExit, match="!= band-sum wavelength"):
        family.main(["--li-dir", str(renders["root"] / "discrete"), "--lumice-run", str(runs[0]), "--output", str(tmp_path / "x.json")])


def test_an_unknown_or_missing_format_is_refused_not_read_as_monochrome(family, renders, tmp_path):
    import json
    import shutil

    other = tmp_path / "other"
    shutil.copytree(renders["root"] / "mono", other)
    provenance = json.loads((other / "provenance.json").read_text())
    provenance["format"] = "lumice-integral.strip/v2"
    (other / "provenance.json").write_text(json.dumps(provenance))
    with pytest.raises(SystemExit, match="is neither"):
        family.load_li(other)
    del provenance["format"]
    (other / "provenance.json").write_text(json.dumps(provenance))
    with pytest.raises(SystemExit, match="None is neither"):
        family.load_li(other)
    with pytest.raises(SystemExit, match="no readable provenance.json"):
        family.load_li(tmp_path / "missing")
