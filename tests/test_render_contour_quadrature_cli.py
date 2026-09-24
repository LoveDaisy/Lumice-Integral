"""CLI of ``scripts/render_contour_quadrature.py``: flags shared with ``render_band_sum.py``, validation, one small run."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
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
    return load("render_contour_quadrature")


def _parser(module):
    captured = {}

    def fake_parse(self, argv=None, namespace=None):
        captured["parser"] = self
        raise SystemExit(0)

    original = module.argparse.ArgumentParser.parse_args
    module.argparse.ArgumentParser.parse_args = fake_parse
    try:
        with pytest.raises(SystemExit):
            module.main([])
    finally:
        module.argparse.ArgumentParser.parse_args = original
    return {a.dest: a.default for a in captured["parser"]._actions}


def test_scene_and_camera_flags_match_render_band_sum(cli) -> None:
    """The copied flags (scene, camera, window, density, workers) keep ``render_band_sum.py``'s names and defaults."""
    ours, theirs = _parser(cli), _parser(load("render_band_sum"))
    shared = ("width", "height", "fov_deg", "view_azimuth", "view_elevation", "rows", "columns", "column_step", "workers", "path",
              "pose_density_family", "pose_density_zenith_mean_deg", "pose_density_zenith_std_deg", "pose_density_roll_mean_deg",
              "pose_density_roll_std_deg", "store_cache_dir", "label")
    assert {k: ours[k] for k in shared} == {k: theirs[k] for k in shared}
    assert cli.MAC_MAX_WORKERS == load("render_band_sum").MAC_MAX_WORKERS


def test_validation(cli, capsys, tmp_path) -> None:
    for argv in (["--band-nodes", "-1"], ["--path", "3", "6"], ["--rows", "5:2"]):
        with pytest.raises(SystemExit) as excinfo:
            cli.main(["--output-dir", str(tmp_path / "x"), *argv])
        assert excinfo.value.code == 2
    assert "rank 0" in capsys.readouterr().err


@pytest.mark.parametrize("band_nodes", [0, 2])
def test_small_window(cli, tmp_path, band_nodes) -> None:
    out = tmp_path / f"render-{band_nodes}"
    cli.main(["--rows", "150:152", "--columns", "150:151", "--store-n", "20000", "--store-cache-dir", str(tmp_path / "stores"),
              "--skip-store-self-checks", "--band-nodes", str(band_nodes), "--output-dir", str(out), "--quiet"])
    arrays, provenance = read_strip(out)
    assert provenance["format"] == "lumice-integral.contour-quadrature/v1"
    assert provenance["options"]["band_nodes"] == band_nodes
    lit = arrays.values[150:152, 150]
    assert np.all(lit > 0.0) and arrays.rendered.sum() == 2
    assert set(json.loads((out / "provenance.json").read_text())["execution"]["cpu_seconds"]) >= {"extract_s", "geometry_s", "integrate_s"}
    if band_nodes == 0:
        # the canonical pixel's row: (150, 150) at 1e-9 (tests/test_contour_quadrature.py holds it at 1e-11)
        assert abs(arrays.values[150, 150] / 6.58152199151 - 1.0) < 1e-8
