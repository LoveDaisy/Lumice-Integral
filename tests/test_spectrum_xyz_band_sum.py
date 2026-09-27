"""``spectrum.xyz_band_sum``: colour band sums as Lumice-normalised pool averages of the unchanged monochrome renderer."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from lumice_integral import band_sum
from lumice_integral.band_sum import BandSumScene, render_band_sum_window, single_path_class
from lumice_integral.canonical_scene import CANONICAL_RENDER, canonical_crystal, canonical_pose_density, canonical_sun_direction
from lumice_integral.path_class import build_path_class
from lumice_integral.pose_density_provenance import pose_density_provenance
from lumice_integral.spectrum import IlluminantType, cmf, dispersion, emitted_weight, wavelength_pool
from lumice_integral.spectrum import illuminant as illuminant_module
from lumice_integral.spectrum import xyz_band_sum
from lumice_integral.spectrum.xyz_band_sum import (
    FORMAT_VERSION,
    PIXEL_CSV_COLUMNS,
    index_groups,
    read_xyz_band_sum_strip,
    render_xyz_band_sum_window,
    write_xyz_band_sum_strip,
)
from lumice_integral.strip_io import Window, read_strip
from lumice_integral.strip_pixel import STATUS_HAS_COMPONENT, STATUS_RENDERED

N_SMALL = 100_000
WINDOW = Window((395, 400), (124, 127))
SUN_WINDOW = {"width": 21, "height": 21, "fov_deg": 6.0, "view": {"azimuth": 0.0, "elevation": 15.0}}
N_550 = 1.3110129170742788


def colour_scene(path_class=None, render=CANONICAL_RENDER, **kwargs) -> BandSumScene:
    return BandSumScene(
        path_class=single_path_class(canonical_crystal(), (3, 5)) if path_class is None else path_class,
        crystal=canonical_crystal(),
        refractive_index=math.nan,
        sun_direction=canonical_sun_direction(),
        pose_density=canonical_pose_density(),
        render=render,
        **kwargs,
    )


def count_monochrome_renders(monkeypatch) -> list[float]:
    calls: list[float] = []
    original = xyz_band_sum.render_band_sum_window

    def counting(scene, *args, **kwargs):
        calls.append(scene.refractive_index)
        return original(scene, *args, **kwargs)

    monkeypatch.setattr(xyz_band_sum, "render_band_sum_window", counting)
    return calls


# ------------------------------------------------------------ normalisation
def test_band_mean_spd_is_the_band_average_lumice_charges():
    assert illuminant_module.band_mean_spd(IlluminantType.E) == 1.0
    for kind in IlluminantType:
        dense = 200_001  # an endpoint-inclusive equally spaced average, the form of Lumice's ComputeMeanSpd
        average = sum(illuminant_module.spd(kind, 380.0 + 400.0 * i / (dense - 1)) for i in range(dense)) / dense
        assert illuminant_module.band_mean_spd(kind) == pytest.approx(average, rel=5e-6)


def test_emitted_weight_mirrors_the_pool_arguments():
    assert emitted_weight(illuminant=IlluminantType.D65) == illuminant_module.band_mean_spd(IlluminantType.D65)
    assert emitted_weight(discrete_weight=2.5) == 2.5
    for bad in ({}, {"illuminant": IlluminantType.A, "discrete_weight": 1.0}):
        with pytest.raises(ValueError, match="exactly one"):
            emitted_weight(**bad)


def test_index_groups_merge_exactly_equal_indices_only():
    discrete = wavelength_pool(3, discrete_wavelength_nm=550.0, discrete_weight=2.0)
    (group,) = index_groups(discrete)
    assert group.refractive_index == N_550 and len(group.entries) == 3
    x, y, z = cmf.lookup(550.0)
    assert group.weights == pytest.approx((6.0 * x, 6.0 * y, 6.0 * z), rel=1e-15)
    pool = wavelength_pool(5, illuminant=IlluminantType.D65)
    assert [g.refractive_index for g in index_groups(pool)] == [e.refractive_index for e in pool]


# ------------------------------------------------------------------ render
def test_scene_must_leave_the_index_to_the_pool(tmp_path):
    scene = dataclasses.replace(colour_scene(), refractive_index=N_550)
    pool = wavelength_pool(1, discrete_wavelength_nm=550.0)
    with pytest.raises(ValueError, match="refractive_index=nan"):
        render_xyz_band_sum_window(scene, pool, 1.0, WINDOW, N_SMALL, base_dir=tmp_path)
    with pytest.raises(ValueError, match="emitted_weight"):
        render_xyz_band_sum_window(colour_scene(), pool, 0.0, WINDOW, N_SMALL, base_dir=tmp_path)
    with pytest.raises(ValueError, match="empty"):
        render_xyz_band_sum_window(colour_scene(), (), 1.0, WINDOW, N_SMALL, base_dir=tmp_path)


def test_one_discrete_550nm_wavelength_is_the_monochrome_render_times_ybar_bit_for_bit(tmp_path):
    """The anchor: Y = ybar(550) V, the monochrome convention raw / E = ybar(550) Omega_p V / (S / 2)."""
    scene = colour_scene()
    pool = wavelength_pool(1, discrete_wavelength_nm=550.0)
    colour, execution = render_xyz_band_sum_window(scene, pool, emitted_weight(discrete_weight=1.0), WINDOW, N_SMALL, base_dir=tmp_path, run_checks=False)
    mono, _ = render_band_sum_window(dataclasses.replace(scene, refractive_index=N_550), WINDOW, N_SMALL, base_dir=tmp_path, run_checks=False)
    xbar, ybar, zbar = cmf.lookup(550.0)
    assert ybar == 0.9949501
    assert [(c.row, c.column) for c in colour] == [(m.row, m.column) for m in mono]
    assert sum(m.value > 0.0 for m in mono) >= 3
    for c, m in zip(colour, mono):
        assert c.xyz == (xbar * m.value, ybar * m.value, zbar * m.value)
        assert (c.delta, c.band_width_rad) == (m.delta, m.band_width_rad)
        if m.value != 0.0:
            assert (c.K_min, c.K_rho_pos_min, c.K_eff_min) == (m.K, m.K_rho_pos, m.K_eff)
        else:
            assert (c.K_min, c.K_rho_pos_min, c.K_eff_min) == (0, 0, 0.0)
    assert execution["slots"] == 1 and len(execution["index_groups"]) == 1
    assert execution["index_groups"][0]["refractive_index"] == N_550


def test_a_discrete_pool_renders_once_and_its_weight_and_slot_count_cancel(tmp_path, monkeypatch):
    scene = colour_scene()
    one, _ = render_xyz_band_sum_window(
        scene, wavelength_pool(1, discrete_wavelength_nm=600.0), 1.0, WINDOW, N_SMALL, base_dir=tmp_path, run_checks=False
    )
    calls = count_monochrome_renders(monkeypatch)
    three, execution = render_xyz_band_sum_window(
        scene,
        wavelength_pool(3, discrete_wavelength_nm=600.0, discrete_weight=2.5),
        emitted_weight(discrete_weight=2.5),
        WINDOW,
        N_SMALL,
        base_dir=tmp_path,
        run_checks=False,
    )
    assert calls == [dispersion.refractive_index(600.0)]
    assert execution["index_groups"][0]["slots"] == 3
    for a, b in zip(one, three):
        assert b.xyz == pytest.approx(a.xyz, rel=1e-15, abs=0.0)


def test_an_illuminant_pool_is_the_slot_average_over_the_band_mean(tmp_path, monkeypatch):
    scene = colour_scene()
    pool = wavelength_pool(5, illuminant=IlluminantType.D65)
    weight = emitted_weight(illuminant=IlluminantType.D65)
    calls = count_monochrome_renders(monkeypatch)
    colour, execution = render_xyz_band_sum_window(scene, pool, weight, WINDOW, N_SMALL, base_dir=tmp_path, run_checks=False)
    assert calls == [e.refractive_index for e in pool] and len(set(calls)) == 5
    assert len(execution["index_groups"]) == 5
    monkeypatch.undo()
    expected = np.zeros((len(colour), 3))
    k_eff = np.full(len(colour), np.inf)
    for entry in pool:
        mono, _ = render_band_sum_window(
            dataclasses.replace(scene, refractive_index=entry.refractive_index), WINDOW, N_SMALL, base_dir=tmp_path, run_checks=False
        )
        value = np.array([m.value for m in mono])
        expected += np.outer(value, [entry.spd_weight * entry.cmf_x, entry.spd_weight * entry.cmf_y, entry.spd_weight * entry.cmf_z])
        k_eff = np.where(value != 0.0, np.minimum(k_eff, [m.K_eff for m in mono]), k_eff)
    expected /= 5 * weight
    np.testing.assert_allclose([c.xyz for c in colour], expected, rtol=1e-15, atol=0.0)
    np.testing.assert_array_equal([c.K_eff_min for c in colour], np.where(np.isfinite(k_eff), k_eff, 0.0))
    lit = [c for c in colour if c.xyz[1] > 0.0]
    assert lit and all(c.xyz[0] > 0.0 and c.xyz[2] > 0.0 for c in lit)


def test_rank0_class_takes_each_wavelengths_own_point_mass(tmp_path):
    cls = build_path_class(canonical_crystal(), (1, 2))
    assert cls.halo_map_rank == 0
    scene = colour_scene(cls, render=SUN_WINDOW, rank0_sample_count=50_000)
    pool = wavelength_pool(2, illuminant=IlluminantType.E)
    window = Window((9, 12), (9, 12))
    colour, execution = render_xyz_band_sum_window(scene, pool, emitted_weight(illuminant=IlluminantType.E), window, N_SMALL, base_dir=tmp_path)
    masses = [g["execution"]["rank0"]["estimate"]["value"] for g in execution["index_groups"]]
    assert masses[0] != masses[1]
    expected = np.zeros(3)
    for entry in pool:
        (mono_sun,) = [
            m for m in render_band_sum_window(dataclasses.replace(scene, refractive_index=entry.refractive_index), window, N_SMALL, base_dir=tmp_path)[0]
            if (m.row, m.column) == (10, 10)
        ]
        expected += np.array([entry.cmf_x, entry.cmf_y, entry.cmf_z]) * entry.spd_weight * mono_sun.value
    (sun,) = [c for c in colour if (c.row, c.column) == (10, 10)]
    np.testing.assert_allclose(sun.xyz, expected / 2.0, rtol=1e-15)
    assert all(c.xyz == (0.0, 0.0, 0.0) for c in colour if (c.row, c.column) != (10, 10))


def test_the_monochrome_estimator_is_not_touched():
    """The colour path adds a layer: ``band_sum`` never imports ``spectrum``."""
    import inspect

    assert "spectrum" not in inspect.getsource(band_sum)


# ------------------------------------------------------------------ output
def test_output_round_trips_and_is_not_a_monochrome_strip(tmp_path):
    scene = colour_scene(build_path_class(canonical_crystal(), (3, 5)))
    pool = wavelength_pool(5, illuminant=IlluminantType.D65)
    weight = emitted_weight(illuminant=IlluminantType.D65)
    window = Window((390, 400), (120, 123))
    results, execution = render_xyz_band_sum_window(scene, pool, weight, window, N_SMALL, base_dir=tmp_path / "stores", run_checks=False)
    write_xyz_band_sum_strip(
        tmp_path / "out", results, scene=scene, pool=pool, spectrum={"illuminant": "D65", "wavelength_count": 5},
        emitted_weight=weight, store_n=N_SMALL, window=window,
        pose_density_block=pose_density_provenance("column", zenith_std_deg=0.5), execution=execution,
    )  # fmt: skip
    xyz, status, provenance = read_xyz_band_sum_strip(tmp_path / "out")
    assert xyz.shape == (801, 251, 3) and provenance["format"] == FORMAT_VERSION
    raw = np.fromfile(tmp_path / "out" / "xyz_float64.bin", dtype="<f8").reshape(801, 251, 3)
    np.testing.assert_array_equal(raw, xyz)
    for r in results:
        assert tuple(xyz[r.row, r.column]) == r.xyz
        assert status[r.row, r.column] == STATUS_RENDERED | (STATUS_HAS_COMPONENT if any(r.xyz) else 0)
    assert int((status & STATUS_RENDERED != 0).sum()) == 30 and xyz[..., 1].max() > 0.0
    assert provenance["options"]["slots"] == 5 and provenance["options"]["emitted_weight"] == weight
    assert [g["refractive_index"] for g in provenance["options"]["index_groups"]] == [e.refractive_index for e in pool]
    assert provenance["options"]["pool"][2]["wavelength_nm"] == 580.0
    assert provenance["scene"]["refractive_index"]["provenance"] == "run-option"
    header = (tmp_path / "out" / "pixels.csv").read_text().splitlines()[0]
    assert tuple(header.split(",")) == PIXEL_CSV_COLUMNS
    with pytest.raises(ValueError):  # the (H, W, 3) shape is not a monochrome strip
        read_strip(tmp_path / "out")
    # A tampered payload is refused, and a monochrome directory is not read as colour.
    path = tmp_path / "out" / "xyz_float64.bin"
    path.write_bytes(path.read_bytes()[:-1] + b"\x01")
    with pytest.raises(ValueError, match="sha256"):
        read_xyz_band_sum_strip(tmp_path / "out")
    mono_scene = dataclasses.replace(scene, refractive_index=N_550)
    mono, mono_execution = render_band_sum_window(mono_scene, window, N_SMALL, base_dir=tmp_path / "stores", run_checks=False)
    band_sum.write_band_sum_strip(
        tmp_path / "mono", mono, scene=mono_scene, n=N_SMALL, window=window,
        pose_density_block=pose_density_provenance("column", zenith_std_deg=0.5), execution=mono_execution,
    )  # fmt: skip
    with pytest.raises(ValueError, match="is not"):
        read_xyz_band_sum_strip(tmp_path / "mono")
