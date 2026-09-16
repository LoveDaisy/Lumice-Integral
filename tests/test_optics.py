from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lumice_integral.analytic import tangent_basis
from lumice_integral.continuation import FiberStatus, TerminationReason, trace_fiber
from lumice_integral.optics import (
    ICE_REFRACTIVE_INDEX,
    fresnel_transmission_3_5,
    fresnel_unpolarized_transmittance,
    minimum_deviation_incident,
    path_3_5,
    path_3_5_domain,
    path_3_5_problem,
)
from lumice_integral.so3 import exp
from test_analytic_fiber import central_difference_jacobian


def test_symmetric_3_5_path_is_regular_and_unit_length():
    evaluation = path_3_5(jnp.eye(3, dtype=jnp.float64), minimum_deviation_incident())

    np.testing.assert_allclose(np.linalg.norm(evaluation.direction), 1.0, atol=1e-14)
    assert evaluation.entry.discriminant > 0.0
    assert evaluation.exit.discriminant > 0.0
    assert evaluation.entry.incidence_cosine > 0.0
    assert evaluation.exit.incidence_cosine > 0.0


def test_3_5_direction_jacobian_matches_central_difference():
    rotation = exp(jnp.array([0.15, 0.08, -0.05], dtype=jnp.float64))
    incident = minimum_deviation_incident()
    target = path_3_5(rotation, incident).direction
    basis = tangent_basis(target)

    def local_residual(delta):
        direction = path_3_5(rotation @ exp(delta), incident).direction
        return basis.T @ (direction - target)

    point = np.zeros(3)
    actual = np.asarray(jax.jacfwd(local_residual)(jnp.asarray(point)))
    expected = central_difference_jacobian(
        lambda delta: np.asarray(local_residual(jnp.asarray(delta))), point
    )

    relative_error = np.linalg.norm(actual - expected) / np.linalg.norm(expected)
    assert relative_error <= 1e-6
    assert np.linalg.matrix_rank(actual) == 2
    assert np.linalg.svd(actual, compute_uv=False)[-1] >= 0.05


def test_safe_3_5_adapter_traces_a_regular_component():
    initial = exp(jnp.array([0.15, 0.08, -0.05], dtype=jnp.float64))
    problem = path_3_5_problem(initial, minimum_deviation_incident())

    result = trace_fiber(problem)

    assert result.status == FiberStatus.CLOSED
    assert result.reason == TerminationReason.CLOSED_LOOP
    assert result.residual_norms.max() <= 1e-10
    assert all(
        margins["entry_snell_discriminant"] > 0.0
        and margins["exit_snell_discriminant"] > 0.0
        for margins in result.branch_diagnostics.accepted_margins
    )


def test_3_5_tir_is_reported_before_the_unsafe_exit_square_root():
    initial = exp(
        jnp.array(
            [0.5447316801391622, -1.506228738763967, -1.190186580432801],
            dtype=jnp.float64,
        )
    )
    incident = minimum_deviation_incident()
    domain = path_3_5_domain(initial, incident)
    problem = path_3_5_problem(
        initial,
        incident,
        target_direction=jnp.array([0.0, 0.0, 1.0], dtype=jnp.float64),
    )

    result = trace_fiber(problem)

    assert not domain.valid
    assert domain.event_kind == "tir_boundary"
    assert domain.margins["entry_snell_discriminant"] > 0.0
    assert domain.margins["exit_snell_discriminant"] < 0.0
    assert result.status == FiberStatus.EVENT_TERMINATED
    assert result.reason == TerminationReason.TIR_BOUNDARY
    assert result.terminal_payload.event.margin < 0.0


@pytest.mark.parametrize(
    ("argument", "value"),
    [
        ("seed", jnp.eye(3, dtype=jnp.float32)),
        ("incident_direction", jnp.array([1.0, 0.0, 0.0], dtype=jnp.float32)),
        ("refractive_index", jnp.asarray(1.31, dtype=jnp.float32)),
        ("target_direction", jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32)),
    ],
)
def test_3_5_problem_rejects_float32_inputs_before_coercion(argument, value):
    arguments = {
        "seed": jnp.eye(3, dtype=jnp.float64),
        "incident_direction": minimum_deviation_incident(),
        "target_direction": jnp.array([0.0, 0.0, 1.0], dtype=jnp.float64),
        "refractive_index": jnp.asarray(1.31, dtype=jnp.float64),
    }
    arguments[argument] = value

    with pytest.raises(ValueError, match=f"{argument}.*float64"):
        path_3_5_problem(**arguments)


@pytest.mark.parametrize(
    "refractive_index",
    [
        jnp.asarray(0.0, dtype=jnp.float64),
        jnp.asarray(-1.31, dtype=jnp.float64),
        jnp.asarray(jnp.nan, dtype=jnp.float64),
        jnp.asarray([1.31], dtype=jnp.float64),
    ],
)
def test_3_5_domain_and_problem_reject_invalid_refractive_indices(refractive_index):
    seed = jnp.eye(3, dtype=jnp.float64)
    incident = minimum_deviation_incident()

    with pytest.raises(ValueError, match="refractive_index.*finite positive scalar"):
        path_3_5_domain(seed, incident, refractive_index)
    with pytest.raises(ValueError, match="refractive_index.*finite positive scalar"):
        path_3_5_problem(seed, incident, refractive_index=refractive_index)


@pytest.mark.parametrize("index", [float(ICE_REFRACTIVE_INDEX), 1.5, 2.4])
def test_fresnel_normal_incidence_matches_the_closed_form(index):
    expected = 1.0 - ((index - 1.0) / (index + 1.0)) ** 2
    assert fresnel_unpolarized_transmittance(1.0, 1.0, index, 1.0) == pytest.approx(expected, rel=1e-15)
    # Reciprocity: the same interface crossed from inside transmits the same power.
    assert fresnel_unpolarized_transmittance(index, 1.0, 1.0, 1.0) == pytest.approx(expected, rel=1e-15)


def test_fresnel_transmittance_is_energy_consistent_with_independent_amplitudes():
    """Compare against the textbook s/p reflectances written in angles, not cosines."""
    index = 1.31
    for incidence_deg in (0.0, 10.0, 35.0, 60.0, 85.0):
        incidence = np.radians(incidence_deg)
        transmitted = np.arcsin(np.sin(incidence) / index)
        r_s = np.sin(incidence - transmitted) / np.sin(incidence + transmitted) if incidence_deg else (1 - index) / (1 + index)
        r_p = np.tan(incidence - transmitted) / np.tan(incidence + transmitted) if incidence_deg else (1 - index) / (1 + index)
        expected = 1.0 - 0.5 * (r_s**2 + r_p**2)
        actual = fresnel_unpolarized_transmittance(1.0, np.cos(incidence), index, np.cos(transmitted))
        assert actual == pytest.approx(expected, rel=1e-12)
        assert 0.0 < actual <= 1.0


def test_fresnel_3_5_lies_in_the_unit_interval_and_drops_toward_the_critical_angle():
    incident = minimum_deviation_incident()
    symmetric = fresnel_transmission_3_5(jnp.eye(3, dtype=jnp.float64), incident)
    grazing = fresnel_transmission_3_5(exp(jnp.array([0.0, 0.0, 0.55], dtype=jnp.float64)), incident)
    assert 0.0 < grazing < symmetric < 1.0
    # Product structure: the two interface factors multiply, no hidden extra factor.
    check = path_3_5_domain(jnp.eye(3, dtype=jnp.float64), incident)
    entry = fresnel_unpolarized_transmittance(
        1.0, check.margins["entry_incidence_cosine"], 1.31, np.sqrt(check.margins["entry_snell_discriminant"])
    )
    exit = fresnel_unpolarized_transmittance(
        1.31, check.margins["exit_incidence_cosine"], 1.0, np.sqrt(check.margins["exit_snell_discriminant"])
    )
    assert symmetric == pytest.approx(entry * exit, rel=1e-15)
    # Outside the smooth domain (exit TIR) no power is transmitted.
    tir_pose = exp(jnp.array([0.5447316801391622, -1.506228738763967, -1.190186580432801], dtype=jnp.float64))
    assert path_3_5_domain(tir_pose, incident).event_kind == "tir_boundary"
    assert fresnel_transmission_3_5(tir_pose, incident) == 0.0
