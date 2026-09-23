"""CLI argument validation of ``scripts/render_ch06_strip.py`` (no rendering)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "render_ch06_strip.py"


@pytest.fixture(scope="module")
def cli():
    spec = importlib.util.spec_from_file_location("render_ch06_strip", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _no_render(monkeypatch, cli):
    """Stop ``main`` right after argument validation."""
    calls = []

    def fake_render_window(*args, **kwargs):
        calls.append(kwargs.get("workers"))
        raise RuntimeError("stop after validation")

    monkeypatch.setattr(cli, "render_window", fake_render_window)
    return calls


def test_workers_above_the_mac_cap_are_rejected_on_darwin(monkeypatch, cli, tmp_path):
    calls = _no_render(monkeypatch, cli)
    monkeypatch.setattr(cli.platform, "system", lambda: "Darwin")
    assert cli.MAC_MAX_WORKERS == 4
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--output-dir", str(tmp_path), "--workers", "5"])
    assert excinfo.value.code == 2 and calls == []
    with pytest.raises(RuntimeError, match="stop after validation"):
        cli.main(["--output-dir", str(tmp_path), "--workers", "4"])
    assert calls == [4]


def test_the_mac_cap_does_not_apply_on_linux(monkeypatch, cli, tmp_path):
    calls = _no_render(monkeypatch, cli)
    monkeypatch.setattr(cli.platform, "system", lambda: "Linux")
    with pytest.raises(RuntimeError, match="stop after validation"):
        cli.main(["--output-dir", str(tmp_path), "--workers", "30"])
    assert calls == [30]
    with pytest.raises(SystemExit):
        cli.main(["--output-dir", str(tmp_path), "--workers", "0"])


def test_retired_flags_are_gone_and_the_distance_threshold_is_exposed(cli, tmp_path):
    for flag in ("--cold-check-interval", "--discovery-step-budget", "--retry-step-budget", "--stall-floor-window"):
        with pytest.raises(SystemExit):
            cli.main(["--output-dir", str(tmp_path), flag, "1"])
    with pytest.raises(SystemExit):
        cli.main(["--output-dir", str(tmp_path), "--distance-threshold", "not-a-number"])


# --- pose-density family flags ------------------------------------------------------------

POSE_DENSITY_FIELDS = (
    "pose_density_family",
    "pose_density_zenith_mean_deg",
    "pose_density_zenith_std_deg",
    "pose_density_roll_mean_deg",
    "pose_density_roll_std_deg",
)


def _capture_options(monkeypatch, cli):
    """Stop ``main`` at ``render_window`` and keep the ``DriverOptions`` it was handed."""
    captured = []

    def fake_render_window(window, options, **kwargs):
        captured.append(options)
        raise RuntimeError("stop after validation")

    monkeypatch.setattr(cli, "render_window", fake_render_window)
    return captured


def test_pose_density_flags_default_to_the_canonical_column_density(monkeypatch, cli, tmp_path):
    from lumice_integral.strip_driver import DriverOptions

    captured = _capture_options(monkeypatch, cli)
    with pytest.raises(RuntimeError, match="stop after validation"):
        cli.main(["--output-dir", str(tmp_path)])
    (options,) = captured
    assert {name: getattr(options, name) for name in POSE_DENSITY_FIELDS} == {
        "pose_density_family": "column",
        "pose_density_zenith_mean_deg": 90.0,
        "pose_density_zenith_std_deg": 0.5,
        "pose_density_roll_mean_deg": None,
        "pose_density_roll_std_deg": None,
    }
    assert options == DriverOptions()  # same resume fingerprint as a run without the flags


def test_pose_density_flags_reach_every_driver_option_field(monkeypatch, cli, tmp_path):
    captured = _capture_options(monkeypatch, cli)
    argv = [
        "--output-dir", str(tmp_path),
        "--pose-density-family", "parry",
        "--pose-density-zenith-mean-deg", "88",
        "--pose-density-zenith-std-deg", "1.5",
        "--pose-density-roll-mean-deg", "3",
        "--pose-density-roll-std-deg", "2",
    ]  # fmt: skip
    with pytest.raises(RuntimeError, match="stop after validation"):
        cli.main(argv)
    (options,) = captured
    assert {name: getattr(options, name) for name in POSE_DENSITY_FIELDS} == {
        "pose_density_family": "parry",
        "pose_density_zenith_mean_deg": 88.0,
        "pose_density_zenith_std_deg": 1.5,
        "pose_density_roll_mean_deg": 3.0,
        "pose_density_roll_std_deg": 2.0,
    }
    # plate: width required, no canonical default outside the column family
    captured.clear()
    with pytest.raises(RuntimeError, match="stop after validation"):
        cli.main(["--output-dir", str(tmp_path), "--pose-density-family", "plate", "--pose-density-zenith-std-deg", "0.7"])
    assert (captured[0].pose_density_family, captured[0].pose_density_zenith_mean_deg, captured[0].pose_density_zenith_std_deg) == ("plate", 0.0, 0.7)
    # random takes no parameter at all
    captured.clear()
    with pytest.raises(RuntimeError, match="stop after validation"):
        cli.main(["--output-dir", str(tmp_path), "--pose-density-family", "random"])
    assert [getattr(captured[0], name) for name in POSE_DENSITY_FIELDS] == ["random", None, None, None, None]


@pytest.mark.parametrize(
    ("argv", "family", "given"),
    [
        (["--pose-density-family", "plate"], "plate", {}),
        (["--pose-density-family", "parry", "--pose-density-zenith-std-deg", "1"], "parry", {"zenith_std_deg": 1.0}),
        (["--pose-density-family", "random", "--pose-density-zenith-std-deg", "1"], "random", {"zenith_std_deg": 1.0}),
        (["--pose-density-roll-std-deg", "1"], "column", {"zenith_std_deg": 0.5, "roll_std_deg": 1.0}),
    ],
)
def test_pose_density_parameter_errors_pass_through_unchanged(monkeypatch, cli, tmp_path, capsys, argv, family, given):
    from lumice_integral.pose_density import resolve_pose_density_parameters

    captured = _capture_options(monkeypatch, cli)
    with pytest.raises(ValueError) as expected:
        resolve_pose_density_parameters(family, **given)
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--output-dir", str(tmp_path), *argv])
    assert excinfo.value.code == 2 and captured == []
    assert capsys.readouterr().err.rstrip().endswith(f"error: {expected.value}")


def test_unknown_pose_density_family_is_an_argparse_choice_error(cli, tmp_path):
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--output-dir", str(tmp_path), "--pose-density-family", "needle"])
    assert excinfo.value.code == 2


# --- small-window renders through the CLI (pose density reaches the pixels and provenance) ---

RENDER_WINDOW = ["--rows", "300:302", "--columns", "126:127", "--prescan-samples", "400000", "--quiet"]
PARRY = ["--pose-density-family", "parry", "--pose-density-zenith-std-deg", "1", "--pose-density-roll-std-deg", "1"]


def _render(cli, directory: Path, *extra: str):
    import json

    import numpy as np

    cli.main(["--output-dir", str(directory), *RENDER_WINDOW, *extra])
    provenance = json.loads((directory / "provenance.json").read_text())
    values = np.fromfile(directory / "strip_float64.bin", dtype="<f8").reshape(801, 251)  # full-image layout
    return provenance, values[300:302, 126]


def test_explicit_column_family_renders_bit_identical_to_the_default(cli, tmp_path):
    default_provenance, _ = _render(cli, tmp_path / "default")
    explicit_provenance, _ = _render(cli, tmp_path / "explicit", "--pose-density-family", "column")
    for name in ("strip_float64.bin", "strip_float32.bin", "status_uint8.bin", "component_count_uint8.bin"):
        assert (tmp_path / "default" / name).read_bytes() == (tmp_path / "explicit" / name).read_bytes(), name
    assert default_provenance["scene"]["pose_density"] == explicit_provenance["scene"]["pose_density"] == {
        "value": {"model": "zenith-gaussian column", "zenith_mean_deg": 90.0, "zenith_std_deg": 0.5, "family": "column"},
        "provenance": "canonical-new",
    }


def _assert_parry_run(provenance, values) -> None:
    from lumice_integral.pose_density_provenance import pose_density_provenance

    assert provenance["scene"]["pose_density"] == {
        "value": pose_density_provenance("parry", zenith_std_deg=1.0, roll_std_deg=1.0),
        "provenance": "run-option",
    }
    # Parry puts its 3-5 light elsewhere on the sky: exactly zero on column 126
    # (docs/roadmap.md section 3.5 item 5), where the canonical column density is lit.
    assert values.shape == (2,) and (values == 0.0).all()


def test_non_default_family_reaches_the_in_process_pixels_and_provenance(cli, tmp_path):
    _, column_values = _render(cli, tmp_path / "column")
    assert (column_values > 0.0).all()
    _assert_parry_run(*_render(cli, tmp_path / "parry", *PARRY, "--workers", "1"))


@pytest.mark.slow
def test_non_default_family_reaches_the_spawned_workers_and_provenance(cli, tmp_path):
    provenance, values = _render(cli, tmp_path / "parry", *PARRY, "--workers", "2")
    assert provenance["execution"]["workers"] == 2
    _assert_parry_run(provenance, values)
