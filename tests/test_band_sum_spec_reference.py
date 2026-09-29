"""A band-sum backend written from ``docs/band-sum-contract.md`` alone, replayed on the exported fixtures.

The functions below implement sections 2.2 (densities), 4 (the band sum) and 5 (rank-0 point masses) of the
contract from its text: they do not import ``band_sum``, ``s2_store``, ``pose_density`` or the parity
exporter's estimator, and their normalisation integrals use scipy's adaptive quadrature and the closed form
of a one-period Gaussian instead of LI's Gauss-Legendre rule.  The fixtures (``BAND_SUM_CELLS`` of
``scripts/export_analytic_parity.py``, exported here) are read as JSON and compared by the layer-1 recipe of
contract section 7.1.

Evidence level (a01): this shows that the contract's text is enough to encode the estimator.  It does not
show that Lumice reads the text the same way -- the reading here is the contract author's, and the events
are LI's.  A disagreement found by Lumice's first implementation is a revision of the contract (section 10).
The statistical rank-0 cell carries no events and is not replayed.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest
from scipy import integrate, special

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "export_analytic_parity.py"


# ------------------------------------------------------------------ section 2.2: pose densities
def density(block: dict):
    """``rho`` of a fixture's ``pose_density`` as a function of the third row ``(e1, e2, e3)`` of ``R``."""
    family = block["family"]
    if family == "random":
        return lambda e1, e2, e3: np.ones_like(e3)
    mu, sigma = math.radians(block["zenith_mean_deg"]), math.radians(block["zenith_std_deg"])
    lower, upper = max(0.0, mu - 14.0 * sigma), min(math.pi, mu + 14.0 * sigma)
    zenith_integral, _ = integrate.quad(
        lambda t: math.exp(-((t - mu) ** 2) / (2.0 * sigma**2)) * math.sin(t), lower, upper, epsabs=0.0, epsrel=1e-13, limit=500
    )

    def zenith(e3):
        theta = np.arccos(np.clip(e3, -1.0, 1.0))
        return 2.0 * np.exp(-((theta - mu) ** 2) / (2.0 * sigma**2)) / zenith_integral

    if family in ("column", "plate"):
        return lambda e1, e2, e3: zenith(e3)
    mu_r, sigma_r = math.radians(block["roll_mean_deg"]), math.radians(block["roll_std_deg"])
    roll_integral = sigma_r * math.sqrt(2.0 * math.pi) * special.erf(math.pi / (sigma_r * math.sqrt(2.0)))  # one period, closed form

    def rho(e1, e2, e3):
        psi = np.arctan2(-e2, e1)
        offset = np.mod(psi - mu_r + math.pi, 2.0 * math.pi) - math.pi
        return zenith(e3) * 2.0 * math.pi * np.exp(-(offset**2) / (2.0 * sigma_r**2)) / roll_integral

    return rho


# ------------------------------------------------------------------ section 4: the band sum
def unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def contains(corners: np.ndarray, x: np.ndarray) -> bool:
    """Section 4.1: the closed spherical quadrilateral of the cyclic corners, on the corners' side."""
    dets = [float(np.dot(np.cross(corners[k], corners[(k + 1) % 4]), x)) for k in range(4)]
    return float(np.dot(corners.sum(axis=0), x)) > 0.0 and (min(dets) >= 0.0 or max(dets) <= 0.0)


def band_sum(fixture: dict) -> list[dict]:
    data = fixture["input"]
    s = np.asarray(data["incident_direction"], dtype=np.float64)
    sun = -s
    n = data["sample"]["n"]
    rho = density(data["pose_density"])
    events = data["events"]
    u, phi = np.asarray(events["u"]).reshape(-1, 3), np.asarray(events["phi"]).reshape(-1, 3)
    d, w = np.asarray(events["deviation"], dtype=np.float64), np.asarray(events["w"], dtype=np.float64)
    f = unit(phi + np.cos(d)[:, None] * u)
    event_rows = np.stack([u, f, np.cross(u, f)], axis=2)  # F_i, columns u, f, u x f
    out = []
    for label, centre, corners in zip(data["pixels"]["labels"], np.asarray(data["pixels"]["centre"]), np.asarray(data["pixels"]["corners"])):
        corner_deviation = [math.acos(max(-1.0, min(1.0, float(np.dot(c, s))))) for c in corners]
        lo, hi = min(corner_deviation), max(corner_deviation)
        delta = math.acos(max(-1.0, min(1.0, float(np.dot(centre, s)))))
        if contains(corners, s) or contains(corners, -s):
            out.append({"label": label, "status": "singular"})
            continue
        inside = (d >= lo) & (d < hi)
        e = unit(centre - np.dot(centre, sun) * sun)
        world = np.stack([sun, e, np.cross(sun, e)], axis=1)  # W, columns s_hat, e, s_hat x e
        third = event_rows[inside] @ world[2]  # (R_i)_{3j} = W_{3,.} . (F_i)_{j,.}
        c = w[inside] * rho(third[:, 0], third[:, 1], third[:, 2])
        total, square = float(np.sum(c)), float(np.sum(c * c))
        value = total / (2.0 * math.pi * n * (hi - lo) * math.sin(delta)) if total != 0.0 else 0.0
        out.append(
            {
                "label": label,
                "status": "ok",
                "value": value,
                "K": int(np.count_nonzero(inside)),
                "K_rho_pos": int(np.count_nonzero(c > 0.0)),
                "K_eff": total * total / square if square > 0.0 else 0.0,
            }
        )
    return out


# ------------------------------------------------------------------ section 5: rank 0
def point_mass(fixture: dict) -> tuple[float, list[dict]]:
    data = fixture["input"]
    s = np.asarray(data["incident_direction"], dtype=np.float64)
    m = float(np.sum(np.asarray(data["events"]["w"], dtype=np.float64))) / data["sample"]["n"]
    pixels, placed = [], False
    for label, corners, omega in zip(data["pixels"]["labels"], np.asarray(data["pixels"]["corners"]), data["pixels"]["solid_angle"]):
        lit = not placed and contains(corners, s)
        placed = placed or lit
        pixels.append({"label": label, "status": "point_mass" if lit else "ok", "value": m / omega if lit else 0.0})
    return m, pixels


# ------------------------------------------------------------------ the fixtures
def _export_script():
    spec = importlib.util.spec_from_file_location("export_analytic_parity", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def fixtures(tmp_path_factory) -> dict[str, dict]:
    from lumice_integral.parity_export import export_matrix  # produces the fixtures only; nothing below uses it

    script = _export_script()
    cells = [cell for cell in script.BAND_SUM_CELLS if cell.rank0_sample_count is None]
    directory = tmp_path_factory.mktemp("band-sum-spec")
    manifest = export_matrix([], directory, band_sum_cells=cells)
    return {entry["name"]: json.loads((directory / entry["files"][0]).read_text()) for entry in manifest["band_sum_cells"]}


def test_every_deterministic_band_sum_cell_is_replayed(fixtures) -> None:
    assert len(fixtures) >= 6 and {f["cell"]["rank"] for f in fixtures.values()} == {0, 2}
    families = {f["input"]["pose_density"]["family"] for f in fixtures.values()}
    assert {"random", "plate", "parry"} <= families


def test_the_contract_text_reproduces_the_layer_1_fixtures(fixtures) -> None:
    for name, fixture in fixtures.items():
        rtol = fixture["tolerance"]["value_relative"]["value"] if fixture["cell"]["rank"] == 2 else fixture["tolerance"]["point_mass_relative"]["value"]
        if fixture["cell"]["rank"] == 0:
            m, pixels = point_mass(fixture)
            reference = fixture["expected"]
            assert abs(m - reference["point_mass"]["m"]) <= rtol * reference["point_mass"]["m"], name
            for mine, expected in zip(pixels, reference["pixels"]):
                assert mine["status"] == expected["status"] and abs(mine["value"] - expected["value"]) <= rtol * abs(expected["value"]), (name, mine)
            continue
        got = band_sum(fixture)
        assert len(got) == len(fixture["expected"]["pixels"])
        for mine, expected in zip(got, fixture["expected"]["pixels"]):
            where = (name, expected["label"])
            assert mine["status"] == expected["status"], where
            if expected["status"] != "ok":
                continue
            assert mine["K"] == expected["K"], where
            assert abs(mine["K_rho_pos"] - expected["K_rho_pos"]) <= expected["allowance"]["K_rho_pos_subnormal"], where
            assert abs(mine["value"] - expected["value"]) <= rtol * abs(expected["value"]), (where, mine["value"], expected["value"])
            assert abs(mine["K_eff"] - expected["K_eff"]) <= rtol * abs(expected["K_eff"]), (where, mine["K_eff"], expected["K_eff"])
