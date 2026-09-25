from __future__ import annotations

from hashlib import sha256
import json

import jax.numpy as jnp
import numpy as np

from lumice_integral.analytic import BODY_AXIS, direction_map, tangent_basis
from lumice_integral.canonical_scene import canonical_fixture_metadata, canonical_pixel_problem
from lumice_integral.continuation import FiberProblem, TargetChart, trace_fiber
from lumice_integral.figure_data import SCHEMA_VERSION, export_fiber_figure_data
from lumice_integral.quadrature import integrate_fiber_resampled, pointwise_integrand


def test_figure_data_round_trip_preserves_geometry_and_unavailable_weights(tmp_path):
    problem = FiberProblem(
        path="analytic-body-axis",
        incident_direction=jnp.array([1.0, 0.0, 0.0], dtype=jnp.float64),
        target_chart=TargetChart(BODY_AXIS, tangent_basis(BODY_AXIS)),
        direction_evaluator=direction_map,
        seed=jnp.eye(3, dtype=jnp.float64),
    )
    result = trace_fiber(problem)
    files = export_fiber_figure_data(
        result,
        tmp_path,
        fixture={"name": "analytic-circle", "target": np.asarray(BODY_AXIS)},
        provenance={"classification": "analytic"},
    )

    metadata = json.loads(files.metadata.read_text(encoding="utf-8"))
    with np.load(files.arrays) as arrays:
        np.testing.assert_array_equal(arrays["poses"], result.poses)
        np.testing.assert_allclose(
            arrays["cumulative_arclength"][-1], 2.0 * np.pi, atol=1e-8
        )
        assert arrays["branch_margins"].shape == (len(result.poses), 0)
        assert set(arrays.files) == set(metadata["payload"]["arrays"])

    assert metadata["schema"] == SCHEMA_VERSION == "lumice-integral.figure-data/v3"
    assert metadata["result"]["status"] == "closed"
    assert metadata["result"]["quadrature"] is None
    assert "integrand" not in metadata["payload"]["arrays"]
    assert metadata["result"]["sample_count"] == len(result.poses)
    weights = metadata["result"]["weight_observables"]
    assert len(weights) == 8
    assert all(
        entry == {"status": "unavailable", "unit": None, "normalization": None, "array": None}
        for entry in weights.values()
    )
    assert not any(name.startswith("weight_") for name in metadata["payload"]["arrays"])
    assert metadata["payload"]["branch_margin_columns"] == []
    assert metadata["payload"]["sha256"] == sha256(files.arrays.read_bytes()).hexdigest()


def test_figure_data_exports_available_weight_arrays_for_the_canonical_pixel(tmp_path):
    result = trace_fiber(canonical_pixel_problem())
    files = export_fiber_figure_data(
        result,
        tmp_path,
        fixture=canonical_fixture_metadata(),
        provenance={"classification": "canonical-new"},
    )

    metadata = json.loads(files.metadata.read_text(encoding="utf-8"))
    weights = metadata["result"]["weight_observables"]
    available = {name for name, entry in weights.items() if entry["status"] == "available"}
    assert available == {"rho_pose", "entry_measure", "fresnel_transmission", "path_validity"}
    assert {name for name, entry in weights.items() if entry["status"] == "unavailable"} == {
        "visibility",
        "source_factor",
        "pixel_factor",
        "other_radiometric",
    }
    with np.load(files.arrays) as arrays:
        for name in available:
            entry = weights[name]
            assert entry["array"] == f"weight_{name}"
            assert entry["unit"] and entry["normalization"]
            np.testing.assert_array_equal(
                arrays[entry["array"]], result.weight_observables[name].values
            )
            assert arrays[entry["array"]].shape == (len(result.poses),)
            assert metadata["payload"]["arrays"][entry["array"]]["unit"] == entry["unit"]
        assert "weight_visibility" not in arrays.files
        assert set(arrays.files) == set(metadata["payload"]["arrays"])
    pixel = metadata["fixture"]["pixel"]
    assert (pixel["row"], pixel["column"]) == (150, 150)
    np.testing.assert_allclose(pixel["projected_center"], [150.5, 150.5], atol=1e-9)
    assert metadata["result"]["conventions"]["haar_to_dvol_g_factor"] == "1/(8*pi**2)"
    assert metadata["payload"]["sha256"] == sha256(files.arrays.read_bytes()).hexdigest()


def test_figure_data_rejects_unserializable_fixture_metadata(tmp_path):
    problem = FiberProblem(
        path="analytic-body-axis",
        incident_direction=jnp.array([1.0, 0.0, 0.0], dtype=jnp.float64),
        target_chart=TargetChart(BODY_AXIS, tangent_basis(BODY_AXIS)),
        direction_evaluator=direction_map,
        seed=jnp.eye(3, dtype=jnp.float64),
    )

    with np.testing.assert_raises_regex(TypeError, "cannot encode object"):
        export_fiber_figure_data(
            trace_fiber(problem),
            tmp_path,
            fixture={"bad": object()},
        )


def test_figure_data_represents_empty_failure_without_nonstandard_json(tmp_path):
    problem = FiberProblem(
        path="rank-loss",
        incident_direction=jnp.array([1.0, 0.0, 0.0], dtype=jnp.float64),
        target_chart=TargetChart(BODY_AXIS, tangent_basis(BODY_AXIS)),
        direction_evaluator=lambda _: BODY_AXIS,
        seed=jnp.eye(3, dtype=jnp.float64),
    )
    files = export_fiber_figure_data(
        trace_fiber(problem),
        tmp_path,
        fixture={"name": "rank-loss"},
    )

    metadata = json.loads(files.metadata.read_text(encoding="utf-8"))
    with np.load(files.arrays) as arrays:
        assert arrays["cumulative_arclength"].shape == (0,)
        assert arrays["poses"].shape == (0, 3, 3)

    assert metadata["result"]["status"] == "event_terminated"
    assert metadata["result"]["closure"]["seed_distance"] is None
    assert metadata["result"]["closure"]["tangent_dot"] is None


def test_figure_data_exports_the_quadrature_block_and_pointwise_integrand(tmp_path):
    problem = canonical_pixel_problem()
    result = trace_fiber(problem)
    quadrature = integrate_fiber_resampled(problem, result)
    files = export_fiber_figure_data(
        result,
        tmp_path,
        fixture=canonical_fixture_metadata(),
        provenance={"classification": "canonical-new"},
        quadrature=quadrature,
    )

    metadata = json.loads(files.metadata.read_text(encoding="utf-8"))
    block = metadata["result"]["quadrature"]
    assert block["status"] == "available"
    assert block["method"] == quadrature.method
    assert block["value"] == quadrature.value
    assert block["error_estimate"] == quadrature.error_estimate
    assert block["raw_value"] == quadrature.raw_value
    assert block["epsilon"] == 1e-6
    assert block["haar_to_dvol_g_factor"] == 1.0 / (8.0 * np.pi**2)
    assert block["relative_tolerance"] == 1e-4
    # 513 / 2 since the panel-wise error estimate (task phase1-quadrature-start-and-speed; 257 / 1 before).
    assert block["node_count"] == quadrature.node_count == 513
    assert block["refinement_rounds"] == quadrature.refinement_rounds == 2
    assert block["node_count_exhausted"] is False
    assert block["node_count_history"] == [[count, value] for count, value in quadrature.node_count_history]
    assert block["residual_after_max"] == quadrature.residual_after_max < 1e-14
    assert block["non_finite_node_count"] == 0
    assert block["endpoint_truncation_estimate"] is None  # NaN on a closed loop -> JSON null
    assert block["endpoint_truncation_note"].startswith("closed loop")
    for gone in ("refinements", "maximum_refinement_depth", "convergence_order_estimate", "depth_exhausted_edges", "factor_seconds"):
        assert gone not in block
    assert block["coverage"].startswith("partial")
    assert block["component_completeness"] == "unknown"
    assert block["factor_names"] == ["entry_measure", "fresnel_transmission", "path_validity"]
    assert block["integrand_array"] == "integrand"
    entry = metadata["payload"]["arrays"]["integrand"]
    assert entry["shape"] == [len(result.poses)]
    assert "normal_jacobian stays unregularised" in entry["semantic"]
    with np.load(files.arrays) as arrays:
        np.testing.assert_array_equal(
            arrays["integrand"], pointwise_integrand(result, epsilon=quadrature.epsilon)
        )
        # The raw J_perp array is exported untouched next to the regularised integrand.
        np.testing.assert_array_equal(
            arrays["normal_jacobian"],
            [d.normal_jacobian for d in result.jacobian_diagnostics],
        )
        assert set(arrays.files) == set(metadata["payload"]["arrays"])
    assert metadata["payload"]["sha256"] == sha256(files.arrays.read_bytes()).hexdigest()
