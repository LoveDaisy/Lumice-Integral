from __future__ import annotations

from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.integrate import cumulative_simpson


REPO = Path(__file__).parents[1]
ILLUSTRATIONS = REPO / "scripts" / "illustrations"
REFERENCE = Path(__file__).parent / "data" / "theory_illustrations" / "world_body_reference.json"
sys.path.insert(0, str(ILLUSTRATIONS))


def _load_module(name: str):
    path = ILLUSTRATIONS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


canonical_data = _load_module("_canonical_data")
world_body_data = _load_module("_world_body_data")


# slow: traces, resamples, and evaluates the canonical production fiber.
@pytest.fixture(scope="module")
def canonical(tmp_path_factory):
    return canonical_data.recompute_data(tmp_path_factory.mktemp("canonical-illustration-data"))


def test_canonical_cache_loader_reports_missing_pair_with_recompute_command(tmp_path):
    with pytest.raises(canonical_data.IllustrationDataError) as caught:
        canonical_data.load_data(tmp_path / "data with spaces")

    assert "missing metadata.json, arrays.npz" in str(caught.value)
    assert "Recompute with:" in str(caught.value)
    assert str((tmp_path / "data with spaces").resolve()) in str(caught.value)


@pytest.mark.slow
def test_canonical_presentation_grid_preserves_accepted_numbered_poses(canonical):
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    arrays = canonical.arrays

    np.testing.assert_array_equal(arrays["selected_indices"], [0, 128, 256, 384])
    np.testing.assert_allclose(
        arrays["presentation_rotations"][arrays["selected_indices"]],
        np.asarray(reference["rotations"]),
        rtol=0.0,
        atol=reference["rotation_atol"],
    )
    np.testing.assert_array_equal(
        arrays["selected_s"], arrays["presentation_s"][arrays["selected_indices"]]
    )
    assert np.all(np.diff(arrays["presentation_s"]) > 0.0)
    assert np.all(arrays["presentation_speed"] > 0.0)
    assert canonical.metadata["source"]["component_scope"] == "one component reached from one seed"
    assert canonical.metadata["source"]["component_completeness"] == "unknown"


@pytest.mark.slow
def test_canonical_quadrature_grid_reconstructs_factors_and_integral(canonical):
    arrays = canonical.arrays
    metadata = canonical.metadata
    epsilon = metadata["quadrature"]["epsilon"]
    factors = (
        arrays["rho_pose"]
        * arrays["entry_measure"]
        * arrays["fresnel_transmission"]
        * arrays["path_validity"]
        / (arrays["normal_jacobian"] + epsilon)
    )
    np.testing.assert_allclose(factors, arrays["integrand"], rtol=1e-12, atol=0.0)
    np.testing.assert_array_equal(arrays["path_validity"], 1.0)

    cumulative = cumulative_simpson(
        arrays["integrand"] * arrays["quadrature_speed"],
        x=arrays["quadrature_parameters"],
        initial=0.0,
    )
    assert cumulative[-1] == pytest.approx(metadata["quadrature"]["raw_value"], rel=1e-12)
    assert metadata["quadrature"]["node_count_exhausted"] is False
    assert abs(
        metadata["quadrature"]["value"] - metadata["quadrature"]["refined_value"]
    ) <= (
        metadata["quadrature"]["error_estimate"]
        + metadata["quadrature"]["refined_error_estimate"]
    )


@pytest.mark.slow
def test_world_body_leaf_is_assembled_from_shared_canonical_data(canonical, tmp_path):
    result = world_body_data.recompute_data(tmp_path, canonical.directory)
    arrays = result.arrays
    selected = canonical.arrays["selected_indices"]

    np.testing.assert_array_equal(
        arrays["rotations"], canonical.arrays["presentation_rotations"][selected]
    )
    np.testing.assert_array_equal(arrays["source_body"], canonical.arrays["presentation_u"][selected])
    np.testing.assert_array_equal(
        arrays["outgoing_directions"], canonical.arrays["presentation_outgoing"][selected]
    )
    assert result.metadata["source"]["canonical_data_sha256"] == sha256(
        (canonical.directory / "arrays.npz").read_bytes()
    ).hexdigest()
