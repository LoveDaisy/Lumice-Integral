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
deviation_data = _load_module("_deviation_data")
pose_fiber_data = _load_module("_pose_fiber_data")
world_body_data = _load_module("_world_body_data")


# slow: traces, resamples, and evaluates the canonical production fiber.
@pytest.fixture(scope="module")
def canonical(tmp_path_factory):
    return canonical_data.recompute_data(tmp_path_factory.mktemp("canonical-illustration-data"))


@pytest.fixture(scope="module")
def deviation(canonical, tmp_path_factory):
    return deviation_data.recompute_data(
        tmp_path_factory.mktemp("deviation-illustration-data"), canonical.directory
    )


@pytest.fixture(scope="module")
def pose_fiber(canonical, tmp_path_factory):
    return pose_fiber_data.recompute_data(
        tmp_path_factory.mktemp("pose-fiber-illustration-data"), canonical.directory
    )


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


@pytest.mark.slow
def test_deviation_leaf_matches_production_field_and_shared_numbering(canonical, deviation):
    from lumice_integral.canonical_scene import CANONICAL_REFRACTIVE_INDEX, canonical_crystal
    from lumice_integral.dp_field import DPField

    arrays = deviation.arrays
    field = DPField.build(canonical_crystal(), [3, 5], CANONICAL_REFRACTIVE_INDEX)
    np.testing.assert_array_equal(arrays["u_curve"], canonical.arrays["presentation_u"])
    np.testing.assert_array_equal(arrays["selected_indices"], canonical.arrays["selected_indices"])
    assert np.all(field.valid_batch(arrays["u_curve"]))
    assert np.max(np.abs(field.d_p_batch(arrays["u_curve"]) - arrays["delta_rad"])) < 1e-10

    chart_points = (
        arrays["chart_x"][..., None] * arrays["chart_e1"]
        + arrays["chart_y"][..., None] * arrays["chart_e2"]
        + np.sqrt(
            np.maximum(1.0 - arrays["chart_x"] ** 2 - arrays["chart_y"] ** 2, 0.0)
        )[..., None]
        * arrays["chart_center"]
    )
    sample = np.flatnonzero(arrays["chart_valid"].ravel())[::1000]
    points = chart_points.reshape(-1, 3)[sample]
    np.testing.assert_allclose(
        arrays["chart_d_deg"].ravel()[sample],
        np.degrees(field.d_p_batch(points)),
        rtol=0.0,
        atol=1e-12,
    )


@pytest.mark.slow
def test_pose_fiber_rays_hit_3_then_5_and_obey_independent_snell(canonical, pose_fiber):
    from lumice_integral.canonical_scene import CANONICAL_REFRACTIVE_INDEX, canonical_crystal

    arrays = pose_fiber.arrays
    crystal = canonical_crystal()
    incident = arrays["incident_direction"]
    target = arrays["target_direction"]
    np.testing.assert_array_equal(arrays["selected_s"], canonical.arrays["selected_s"])
    np.testing.assert_array_equal(
        arrays["selected_rotations"],
        canonical.arrays["presentation_rotations"][canonical.arrays["selected_indices"]],
    )
    for rotation, points, directions in zip(
        arrays["selected_rotations"], arrays["ray_points"], arrays["ray_segment_directions"]
    ):
        body_points = points @ rotation
        internal_body = rotation.T @ directions[1]
        hit = crystal.intersect_ray(np.mean(body_points[1:3], axis=0), internal_body)
        assert hit is not None
        _, entry_face, _, exit_face = hit
        assert [entry_face.number, exit_face.number] == [3, 5]
        np.testing.assert_allclose(directions[0], incident, rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(directions[2], target, rtol=0.0, atol=1e-12)

        entry_normal = rotation @ crystal.normal(entry_face)
        exit_normal = rotation @ crystal.normal(exit_face)
        entry_residual = (
            pose_fiber_data._tangential(incident, entry_normal)
            - CANONICAL_REFRACTIVE_INDEX
            * pose_fiber_data._tangential(directions[1], entry_normal)
        )
        exit_residual = (
            CANONICAL_REFRACTIVE_INDEX
            * pose_fiber_data._tangential(directions[1], exit_normal)
            - pose_fiber_data._tangential(target, exit_normal)
        )
        assert max(np.linalg.norm(entry_residual), np.linalg.norm(exit_residual)) < 1e-10


@pytest.mark.slow
def test_spatial_figures_render_from_leaf_caches_only(deviation, pose_fiber, tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    deviation_module = _load_module("deviation_level_set")
    pose_module = _load_module("pose_fiber")

    deviation_figure = deviation_module.render_figure(deviation)
    pose_figure = pose_module.render_figure(pose_fiber)
    try:
        deviation_path = tmp_path / "deviation.png"
        pose_path = tmp_path / "pose.png"
        deviation_figure.savefig(deviation_path, dpi=150)
        pose_figure.savefig(pose_path, dpi=150)
        assert matplotlib.image.imread(deviation_path).shape[:2] == (1200, 2175)
        assert matplotlib.image.imread(pose_path).shape[:2] == (1500, 2400)
        assert len(deviation_figure.axes) == 4
        assert len(pose_figure.axes) == 8
    finally:
        matplotlib.pyplot.close(deviation_figure)
        matplotlib.pyplot.close(pose_figure)
