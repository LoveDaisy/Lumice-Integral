from __future__ import annotations

from hashlib import sha256
import json

import jax.numpy as jnp
import numpy as np

from lumice_integral.analytic import BODY_AXIS, direction_map, tangent_basis
from lumice_integral.continuation import FiberProblem, TargetChart, trace_fiber
from lumice_integral.figure_data import SCHEMA_VERSION, export_fiber_figure_data


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

    assert metadata["schema"] == SCHEMA_VERSION
    assert metadata["result"]["status"] == "closed"
    assert metadata["result"]["sample_count"] == len(result.poses)
    assert set(metadata["result"]["weight_observables"].values()) == {
        "unavailable"
    }
    assert metadata["payload"]["branch_margin_columns"] == []
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
