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
