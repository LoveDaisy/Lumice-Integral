from __future__ import annotations

from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest

from lumice_integral.analytic import BODY_AXIS, direction_map, tangent_basis
from lumice_integral.canonical_scene import (
    canonical_crystal,
    canonical_incident_direction,
    canonical_pixel_problem,
    canonical_pose_density,
    canonical_seed,
)
from lumice_integral.continuation import FiberProblem, FiberStatus, TargetChart, TerminationReason, trace_fiber
from lumice_integral.geometry import entry_measure
from lumice_integral.optics import minimum_deviation_incident, path_3_5_domain, path_3_5_problem
from lumice_integral.so3 import exp
from lumice_integral.weights import (
    STANDARD_WEIGHT_NAMES,
    WeightEvaluator,
    WeightObservable,
    build_3_5_weight_evaluators,
    entry_measure_weight,
    evaluate_weights,
    fresnel_transmission_weight,
    path_validity_weight,
)

FOUR_FACTORS = ("rho_pose", "entry_measure", "fresnel_transmission", "path_validity")
OPEN_FACTORS = ("visibility", "source_factor", "pixel_factor", "other_radiometric")


def test_entry_measure_weight_is_the_geometry_value_unchanged():
    rotation = canonical_seed()
    incident = canonical_incident_direction()
    crystal = canonical_crystal()
    direct = entry_measure(rotation, (3, 5), incident, crystal, n_ice=1.31)
    adapted = entry_measure_weight(
        jnp.asarray(rotation), faces=(3, 5), incident_direction=incident, crystal=crystal, refractive_index=1.31
    )
    assert direct.status == "ok" and direct.value > 0.0
    assert adapted == direct.value


def test_path_validity_gate_and_fresnel_on_feasible_and_infeasible_poses():
    incident = np.asarray(minimum_deviation_incident())
    crystal = canonical_crystal()
    feasible = np.asarray(exp(jnp.array([0.15, 0.08, -0.05], dtype=jnp.float64)))
    tir = np.asarray(
        exp(jnp.array([0.5447316801391622, -1.506228738763967, -1.190186580432801], dtype=jnp.float64))
    )
    backface = np.asarray(exp(jnp.array([0.0, 0.0, np.pi], dtype=jnp.float64)))  # face 3 turned away

    assert path_3_5_domain(feasible, incident).valid
    assert path_3_5_domain(tir, incident).event_kind == "tir_boundary"
    assert path_3_5_domain(backface, incident).event_kind == "path_infeasible"
    common = dict(faces=(3, 5), incident_direction=incident, crystal=crystal, refractive_index=1.31)
    assert path_validity_weight(feasible, **common) == 1.0
    assert path_validity_weight(tir, **common) == 0.0
    assert path_validity_weight(backface, **common) == 0.0
    assert 0.0 < fresnel_transmission_weight(feasible, faces=(3, 5), incident_direction=incident, refractive_index=1.31) < 1.0
    assert fresnel_transmission_weight(tir, faces=(3, 5), incident_direction=incident, refractive_index=1.31) == 0.0


def test_weight_schema_rejects_inconsistent_observables_and_evaluators():
    with pytest.raises(ValueError):
        WeightObservable("available", "dimensionless", "x", None)
    with pytest.raises(ValueError):
        WeightObservable("unavailable", values=np.ones(3))
    with pytest.raises(ValueError):
        WeightObservable("available", "dimensionless", "x", np.ones((3, 1)))
    with pytest.raises(ValueError):
        WeightObservable("partial")
    with pytest.raises(ValueError):
        WeightEvaluator(lambda rotation: 1.0, "", "x")
    with pytest.raises(ValueError):
        FiberProblem(
            path="analytic-body-axis",
            incident_direction=jnp.array([1.0, 0.0, 0.0], dtype=jnp.float64),
            target_chart=TargetChart(BODY_AXIS, tangent_basis(BODY_AXIS)),
            direction_evaluator=direction_map,
            seed=jnp.eye(3, dtype=jnp.float64),
            weight_evaluators={"rho_pose": lambda rotation: 1.0},
        )


def test_evaluate_weights_reports_registered_factors_and_keeps_the_rest_unavailable():
    poses = np.stack([np.eye(3), np.asarray(exp(jnp.array([0.1, 0.0, 0.0])))])
    evaluators = {
        "rho_pose": WeightEvaluator(lambda rotation: float(rotation[2, 2]), "dimensionless", "test"),
        "custom_factor": WeightEvaluator(lambda rotation: 2.0, "unit", "test"),
    }
    observables = evaluate_weights(evaluators, poses)
    assert set(observables) == set(STANDARD_WEIGHT_NAMES) | {"custom_factor"}
    np.testing.assert_allclose(observables["rho_pose"].values, [1.0, np.cos(0.1)])
    assert observables["custom_factor"].values.tolist() == [2.0, 2.0]
    for name in STANDARD_WEIGHT_NAMES:
        if name != "rho_pose":
            assert observables[name] == WeightObservable("unavailable")
    empty = evaluate_weights(evaluators, np.empty((0, 3, 3)))
    assert empty["rho_pose"].status == "available" and empty["rho_pose"].values.shape == (0,)


def test_build_3_5_weight_evaluators_declares_exactly_the_four_factors():
    evaluators = build_3_5_weight_evaluators(
        incident_direction=canonical_incident_direction(),
        refractive_index=1.31,
        crystal=canonical_crystal(),
        pose_density=canonical_pose_density(),
    )
    assert set(evaluators) == set(FOUR_FACTORS)
    for name, evaluator in evaluators.items():
        assert isinstance(evaluator, WeightEvaluator)
        assert evaluator.unit and evaluator.normalization
    assert "Haar" in evaluators["rho_pose"].normalization
    assert "not divided" in evaluators["entry_measure"].normalization
    assert "not a gain" in evaluators["path_validity"].normalization


def test_canonical_pixel_fiber_exposes_four_available_factors_pointwise():
    problem = canonical_pixel_problem()
    result = trace_fiber(problem)
    sample_count = len(result.poses)

    assert result.status == FiberStatus.CLOSED
    assert result.reason == TerminationReason.CLOSED_LOOP
    assert result.residual_norms.max() <= 1e-10
    assert set(result.weight_observables) == set(STANDARD_WEIGHT_NAMES)
    for name in FOUR_FACTORS:
        observable = result.weight_observables[name]
        assert observable.status == "available"
        assert observable.values.shape == (sample_count,)
        assert observable.values.dtype == np.float64
        assert np.all(np.isfinite(observable.values))
        assert observable.unit == problem.weight_evaluators[name].unit
        assert observable.normalization == problem.weight_evaluators[name].normalization
    for name in OPEN_FACTORS:
        assert result.weight_observables[name] == WeightObservable("unavailable")

    rho = result.weight_observables["rho_pose"].values
    fresnel = result.weight_observables["fresnel_transmission"].values
    measure = result.weight_observables["entry_measure"].values
    validity = result.weight_observables["path_validity"].values
    assert np.all(rho > 0.0) and rho.max() > 1.0  # the loop crosses the horizontal-column peak
    assert np.all((fresnel > 0.0) & (fresnel < 1.0))
    assert np.all(measure >= 0.0) and measure.max() > 0.0
    assert set(np.unique(validity)) <= {0.0, 1.0} and validity.min() == 1.0

    # Pointwise alignment: recomputing factor i at pose i reproduces the arrays.
    index = sample_count // 3
    pose = result.poses[index]
    for name in FOUR_FACTORS:
        assert result.weight_observables[name].values[index] == problem.weight_evaluators[name].evaluate(pose)
    # J_perp and the Haar conversion stay separate from every factor.
    assert result.conventions["haar_to_dvol_g_factor"] == "1/(8*pi**2)"
    assert "J_perp" in result.conventions["coarea_denominator"]
    assert len(result.jacobian_diagnostics) == sample_count


def test_weights_do_not_change_geometry_or_termination():
    with_weights = trace_fiber(canonical_pixel_problem())
    without = trace_fiber(canonical_pixel_problem(with_weights=False))
    np.testing.assert_array_equal(with_weights.poses, without.poses)
    np.testing.assert_array_equal(with_weights.arclength_increments, without.arclength_increments)
    assert with_weights.reason == without.reason
    assert all(observable.status == "unavailable" for observable in without.weight_observables.values())


def test_event_terminated_fiber_still_reports_weights_on_accepted_poses_only():
    tir_seed = exp(jnp.array([0.5447316801391622, -1.506228738763967, -1.190186580432801], dtype=jnp.float64))
    incident = minimum_deviation_incident()
    problem = replace(
        path_3_5_problem(tir_seed, incident, target_direction=jnp.array([0.0, 0.0, 1.0], dtype=jnp.float64)),
        weight_evaluators=build_3_5_weight_evaluators(
            incident_direction=np.asarray(incident),
            refractive_index=1.31,
            crystal=canonical_crystal(),
            pose_density=canonical_pose_density(),
        ),
    )
    result = trace_fiber(problem)
    assert result.status == FiberStatus.EVENT_TERMINATED
    assert len(result.poses) == 0
    for name in FOUR_FACTORS:
        assert result.weight_observables[name].status == "available"
        assert result.weight_observables[name].values.shape == (0,)


# --- batch factor forms (task-resample-and-integrate Step 3) -------------------


def test_batch_factors_agree_with_the_scalar_evaluators_on_fiber_and_random_poses():
    """The four 3-5 factors have scalar and batch forms that must not drift apart.

    Checked on the canonical fiber poses (all gates open) and on random poses
    (every gate verdict), the batch form of each named factor equals the loop
    over its scalar form to ``atol=1e-12`` (float64 is the package-wide
    baseline).  ``path_validity``'s batch reuses the ``entry_measure`` memo of
    the same pose array, so the memo must be transparent as well.
    """
    from lumice_integral.weights import evaluate_weights_batch

    problem = canonical_pixel_problem()
    result = trace_fiber(problem)
    rng = np.random.default_rng(9)
    random_poses = np.asarray([np.asarray(exp(jnp.asarray(rng.normal(size=3)))) for _ in range(400)])
    poses = np.concatenate([np.asarray(result.poses), random_poses])
    names = ("rho_pose", "entry_measure", "fresnel_transmission", "path_validity")

    batch = evaluate_weights_batch(problem.weight_evaluators, poses, names)

    assert set(batch.values) == set(batch.seconds) == set(names)
    for name in names:
        scalar = np.array([problem.weight_evaluators[name].evaluate(pose) for pose in poses])
        assert batch.values[name].shape == (len(poses),) and batch.values[name].dtype == np.float64
        np.testing.assert_allclose(batch.values[name], scalar, rtol=0.0, atol=1e-12, err_msg=name)
        assert batch.seconds[name] >= 0.0
    assert 0 < np.count_nonzero(batch.values["path_validity"]) < len(poses)
    # A second, different array with the same shape is recomputed, not served from the memo.
    other = np.asarray(poses[::-1])
    np.testing.assert_allclose(
        evaluate_weights_batch(problem.weight_evaluators, other, ("entry_measure",)).values["entry_measure"],
        batch.values["entry_measure"][::-1], rtol=0.0, atol=0.0,
    )


def test_batch_evaluation_falls_back_to_the_scalar_loop_and_refuses_missing_factors():
    from lumice_integral.weights import evaluate_weights_batch

    evaluators = {
        "rho_pose": WeightEvaluator(lambda rotation: float(rotation[0, 0]) + 2.0, "dimensionless", "test"),
    }
    poses = np.asarray([np.eye(3), np.asarray(exp(jnp.array([0.0, 0.0, 1.0])))])

    batch = evaluate_weights_batch(evaluators, poses, ("rho_pose",))
    np.testing.assert_allclose(batch.values["rho_pose"], [3.0, 2.0 + np.cos(1.0)])
    with pytest.raises(KeyError):
        evaluate_weights_batch(evaluators, poses, ("rho_pose", "entry_measure"))
    with pytest.raises(ValueError):
        WeightEvaluator(lambda _: 1.0, "u", "n", evaluate_batch=3)


def test_fresnel_weight_is_the_optics_path_power_factor_across_partial_reflections():
    """The Phase I ``fresnel_transmission`` weight is the S^2 store's power factor, called not copied.

    On a path with three internal reflections, over poses where some are
    partial (``R < 1``) and some total, both the scalar and the batch
    evaluator return :func:`.optics.fresnel_transmission_path(_batch)` bit
    for bit.
    """
    from lumice_integral.geometry import HexPrism
    from lumice_integral.optics import fresnel_transmission_path, fresnel_transmission_path_batch, path_domain_batch
    from lumice_integral.pose_density import HaarUniformPoseDensity
    from lumice_integral.so3 import haar_rotations
    from lumice_integral.weights import build_path_weight_evaluators

    faces = (3, 5, 6, 7, 3)
    incident = np.asarray(minimum_deviation_incident(), dtype=np.float64)
    rotations = haar_rotations(4000, np.random.default_rng(11))
    check = path_domain_batch(rotations, faces, incident, 1.31)
    partial = check.valid & np.any(
        np.stack([check.margins[f"internal_{k}_tir_discriminant"] <= 0.0 for k in (1, 2, 3)]), axis=0
    )
    assert partial.any() and (check.valid & ~partial).any()
    weight = build_path_weight_evaluators(
        faces=faces, incident_direction=incident, refractive_index=1.31,
        crystal=HexPrism.from_ratio(2.0), pose_density=HaarUniformPoseDensity(),
    )["fresnel_transmission"]
    batch = weight.evaluate_batch(rotations)
    np.testing.assert_array_equal(batch, fresnel_transmission_path_batch(rotations, faces, incident, 1.31))
    assert np.any((batch > 0.0) & partial)
    for rotation in rotations[check.valid][:20]:
        assert weight.evaluate(jnp.asarray(rotation)) == fresnel_transmission_path(rotation, faces, incident, 1.31)
