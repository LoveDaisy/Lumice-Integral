"""CLI of ``scripts/render_band_sum.py``: argument validation, defaults shared with ``render_ch06_strip.py``, one small run."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from lumice_integral.strip_io import read_strip

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
