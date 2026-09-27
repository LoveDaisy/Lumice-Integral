"""CLI of ``scripts/render_band_sum.py``: argument validation, defaults shared with ``render_ch06_strip.py``, one small run."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from lumice_integral.canonical_scene import CANONICAL_HEIGHT_RATIO, CANONICAL_REFRACTIVE_INDEX, canonical_crystal
from lumice_integral.geometry import HexPrism
from lumice_integral.strip_io import read_strip, scene_block, scene_crystal

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def cli():
    return load("render_band_sum")


def _no_render(monkeypatch, cli) -> list:
    calls = []

    def fake(scene, window, n, **kwargs):
        calls.append((scene, window, n, kwargs))
        raise RuntimeError("stop after validation")

    monkeypatch.setattr(cli, "render_band_sum_window", fake)
    return calls


def _error(cli, capsys, argv) -> str:
    with pytest.raises(SystemExit) as excinfo:
        cli.main(argv)
    assert excinfo.value.code == 2
    return capsys.readouterr().err


def test_defaults_match_render_ch06_strip(cli):
    ch06 = load("render_ch06_strip")
    assert cli.MAC_MAX_WORKERS == ch06.MAC_MAX_WORKERS == 4
    parsers = {}
    for module in (cli, ch06):
        captured = {}

        def fake_parse(self, argv=None, namespace=None, _captured=captured):
            _captured["parser"] = self
            raise SystemExit(0)

        original = module.argparse.ArgumentParser.parse_args
        module.argparse.ArgumentParser.parse_args = fake_parse
        try:
            with pytest.raises(SystemExit):
                module.main([])
        finally:
            module.argparse.ArgumentParser.parse_args = original
        parsers[module.__name__] = {a.dest: a.default for a in captured["parser"]._actions}
    ours, theirs = parsers["render_band_sum"], parsers["render_ch06_strip"]
    for name in (
        "workers",
        "column_step",
        "pose_density_family",
        "pose_density_zenith_mean_deg",
        "pose_density_zenith_std_deg",
        "pose_density_roll_mean_deg",
        "pose_density_roll_std_deg",
    ):
        assert ours[name] == theirs[name], name
    assert ours["path"] == [3, 5] and ours["path_class"] is False


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--workers", "0"], "--workers must be positive"),
        (["--rows", "10:900"], "must satisfy 0 <= a < b <= 801"),
        (["--columns", "5:5"], "must satisfy 0 <= a < b <= 251"),
        (["--no-symmetry-transport"], "needs --path-class"),
        (["--path", "3", "9", "--path-class"], "unknown hexagonal-prism face numbers"),
        (["--path", "3"], "at least two faces"),
        (["--pose-density-family", "plate"], "requires zenith_std_deg"),
        (["--pose-density-family", "column", "--pose-density-roll-std-deg", "1"], "takes no roll_std_deg"),
        (["--store-n", "0"], "--store-n must be positive"),
        (["--face-distance", "-1", "-1", "-1", "-1", "-1", "-1"], "the cross-section has no area"),
        (["--face-distance", "1", "1.3", "0.7", "1.9", "1.1", "0.4", "--path", "3", "5", "6", "7"], "faces [6] do not exist"),
        (["--refractive-index", "0.9"], "--refractive-index must exceed 1"),
    ],
)
def test_invalid_arguments_are_rejected(cli, capsys, monkeypatch, tmp_path, extra, message):
    calls = _no_render(monkeypatch, cli)
    argv = ["--output-dir", str(tmp_path / "out"), "--store-n", "1000", *extra]
    assert message in _error(cli, capsys, argv)
    assert calls == []


def test_workers_above_the_mac_cap_and_a_non_empty_output_dir_are_rejected(cli, capsys, monkeypatch, tmp_path):
    calls = _no_render(monkeypatch, cli)
    monkeypatch.setattr(cli.platform, "system", lambda: "Darwin")
    assert "exceeds the macOS limit" in _error(cli, capsys, ["--output-dir", str(tmp_path), "--store-n", "1", "--workers", "5"])
    (tmp_path / "old").write_text("previous render")
    assert "--overwrite" in _error(cli, capsys, ["--output-dir", str(tmp_path), "--store-n", "1"])
    with pytest.raises(RuntimeError, match="stop after validation"):
        cli.main(["--output-dir", str(tmp_path), "--store-n", "1", "--overwrite", "--workers", "4"])
    assert calls[0][3]["workers"] == 4


def test_path_class_and_density_reach_the_scene(cli, monkeypatch, tmp_path):
    calls = _no_render(monkeypatch, cli)
    with pytest.raises(RuntimeError):
        cli.main(
            [
                "--output-dir", str(tmp_path / "out"), "--store-n", "1000", "--path", "3", "5", "--path-class",
                "--pose-density-family", "parry", "--pose-density-zenith-std-deg", "1", "--pose-density-roll-std-deg", "1",
                "--width", "31", "--height", "21", "--fov-deg", "20", "--view-elevation", "15", "--rows", "2:4",
            ]
        )  # fmt: skip
    scene, window, n, kwargs = calls[0]
    assert scene.path_class.size == 12 and scene.transport is True and n == 1000
    assert scene.render == {"width": 31, "height": 21, "fov_deg": 20.0, "view": {"azimuth": 0.0, "elevation": 15.0}}
    assert window.rows == (2, 4) and window.columns == (0, 31)
    assert type(scene.pose_density).__name__ == "ZenithRollGaussianPoseDensity"


def test_small_end_to_end_run(cli, tmp_path):
    out = tmp_path / "out"
    cli.main(
        [
            "--output-dir", str(out), "--store-n", "100000", "--rows", "395:400", "--columns", "124:127",
            "--store-cache-dir", str(tmp_path / "stores"), "--skip-store-self-checks", "--quiet",
        ]
    )  # fmt: skip
    arrays, provenance = read_strip(out)
    assert int(arrays.rendered.sum()) == 15 and arrays.values.max() > 0.0
    assert provenance["options"]["N"] == 100_000 and provenance["execution"]["workers"] == 1
    assert provenance["scene"]["pose_density"]["value"]["family"] == "column"
    assert provenance["scene"]["crystal"] == {
        "value": {"type": "hexagonal_column", "height_ratio": CANONICAL_HEIGHT_RATIO},
        "provenance": "canonical-new",
    }
    assert provenance["scene"]["refractive_index"] == {"value": CANONICAL_REFRACTIVE_INDEX, "provenance": "canonical-new"}


D3H = [1.0, 1.2, 1.0, 1.2, 1.0, 1.2]


def test_face_distance_and_refractive_index_reach_the_scene(cli, monkeypatch, tmp_path):
    calls = _no_render(monkeypatch, cli)
    with pytest.raises(RuntimeError):
        cli.main(
            [
                "--output-dir", str(tmp_path / "out"), "--store-n", "1000", "--path-class",
                "--face-distance", *map(str, D3H), "--refractive-index", "1.3110129",
            ]
        )  # fmt: skip
    scene = calls[0][0]
    assert scene.crystal.face_distance_ratios == tuple(D3H) and scene.crystal.h / scene.crystal.a == CANONICAL_HEIGHT_RATIO
    assert scene.refractive_index == 1.3110129
    assert scene.path_class.size == 6  # D3h: |G_true| = 12, the [3, 5] orbit halves


def test_scene_block_records_a_non_canonical_crystal_and_index():
    assert scene_block() == scene_block(crystal=canonical_crystal(), refractive_index=CANONICAL_REFRACTIVE_INDEX)
    assert scene_crystal(scene_block()).vertices.tolist() == canonical_crystal().vertices.tolist()
    crystal = HexPrism.from_lumice(1.0, [1.0, 1.3, 0.7, 1.9, 1.1, 0.4])
    block = scene_block(crystal=crystal, refractive_index=1.3110129)
    assert block["crystal"]["provenance"] == "run-option" and block["crystal"]["value"]["face_distance"] == [1.0, 1.3, 0.7, 1.9, 1.1, 0.4]
    assert block["refractive_index"] == {"value": 1.3110129, "provenance": "run-option"}
    assert scene_crystal(block).vertices.tolist() == crystal.vertices.tolist()
    # a render written before scene_block recorded the crystal: height_ratio only, the regular prism
    assert scene_crystal({"crystal": {"value": {"type": "hexagonal_column", "height_ratio": 0.2}}}).h == pytest.approx(0.2)


def test_small_end_to_end_run_low_symmetry(cli, tmp_path):
    out = tmp_path / "out"
    cli.main(
        [
            "--output-dir", str(out), "--store-n", "100000", "--path-class", "--face-distance", *map(str, D3H),
            "--refractive-index", "1.3110129", "--pose-density-family", "random", "--width", "41", "--height", "41", "--fov-deg", "60",
            "--view-elevation", "15", "--rows", "18:23", "--columns", "0:41",
            "--store-cache-dir", str(tmp_path / "stores"), "--skip-store-self-checks", "--quiet",
        ]
    )  # fmt: skip
    arrays, provenance = read_strip(out)
    assert arrays.values.max() > 0.0
    assert scene_crystal(provenance["scene"]).face_distance_ratios == tuple(D3H)
    assert provenance["scene"]["refractive_index"]["value"] == 1.3110129
    assert provenance["options"]["path_class"]["size"] == 6
