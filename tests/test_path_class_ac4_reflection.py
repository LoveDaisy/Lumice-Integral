"""AC4: a reflecting member of the same Phi (``3-1-2-5``) traced and integrated; same map, different weight.

``3-1-2-5`` has fold matrix ``S_2 S_1 = I`` and the 60-deg wedge of ``3-5``
(the necessary "same Phi" conditions this task verifies; the full
Phi-fingerprint machinery stays in the writing repository), hence the same
direction map: on the accepted poses of its fiber the ``3-5`` map hits the
same target.  Its weight differs (the footprint is the part of face 3 whose
internal ray climbs to face 1 and drops to face 2 before leaving through
face 5).  The fixture pixel :data:`REFLECTION_DISCOVERY` is the frozen output
of ``scripts/discover_3_1_2_5_seed.py``: the strip pixel where the reflecting
member carries the most column-density weight -- row 651 of the ch06 strip,
inside the rows 475-650 tail where the single-path render is about 4x darker
than the historical raw (``defect2_findings.md`` section 8); there the
reflecting member alone is 3.6x the ``3-5`` value.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from lumice_integral.canonical_scene import (
    CANONICAL_REFRACTIVE_INDEX,
    CANONICAL_RENDER,
    canonical_incident_direction,
    canonical_pose_density,
    canonical_sun_direction,
)
from lumice_integral.continuation import FiberStatus, TerminationReason, trace_fiber
from lumice_integral.discovery import discover_components
from lumice_integral.geometry import HexPrism, entry_measure, fold_matrix, halo_map_rank, wedge_angle_deg
from lumice_integral.optics import PATH_3_5_FACES, path_3_5, path_domain, path_problem
from lumice_integral.path_class import build_path_class
from lumice_integral.so3 import exp
from lumice_integral.strip_pixel import PixelOptions, StripScene, build_strip_scene, pixel_target, render_pixel

REFLECTING = (3, 1, 2, 5)
# The frozen seed and values below were recorded on 2026-09-20 on the ``h/a = 1``
# crystal; task defect2-crystal-height-convention then moved the canonical scene
# to ``h/a = 2`` (the seed's corridor and ``entry_measure`` change with h), so
# this module binds that crystal explicitly -- the same-map/different-weight
# property under test does not depend on which crystal it runs on.
RECORDED_CRYSTAL = HexPrism.from_ratio(1.0)
# Frozen output of one recorded run of scripts/discover_3_1_2_5_seed.py (2026-09-20).
REFLECTION_DISCOVERY = {
    "script": "scripts/discover_3_1_2_5_seed.py",
    "command": "uv run python scripts/discover_3_1_2_5_seed.py",
    "prescan": {"sample_count": 400_000, "rng_seed": 20260916},  # of the pixel selection; the renders seed from the store
    "selection": "strip pixel with the largest summed rho_pose * entry_measure of the landing 3-1-2-5 prescan samples",
    "pixel": {"row": 651, "column": 13},
    "recorded": {
        "3-5": {"value": 5.8239478920284775e-3, "kind": "closed", "arclength": 5.266994836084206},
        "3-1-2-5": {
            "value": 2.1184291530835113e-2,
            "kind": "arc",
            "arclength": 2.6051425949472597,
            "reason": "path_infeasible",
            "start_reason": "path_infeasible",
            "seed_exponential_coordinates": [-1.766544719946978, 0.09861605958874967, 1.3808563762094437],
        },
        "value_ratio": 3.6374452387925875,
    },
}
ROW, COLUMN = REFLECTION_DISCOVERY["pixel"]["row"], REFLECTION_DISCOVERY["pixel"]["column"]
# The 3-1-2-5 arc as traced from the seed-store seed (task phase1-seeds-from-store, 2026-09-25): both ends on
# the internal_1_incidence_cosine boundary (margin ~4e-6).  The recorded trace from the prescan seed stopped
# 0.028 rad short of it at one end, on a path_infeasible trial whose last accepted pose still had margin 0.02;
# the continuation's event localisation depends on the step sequence, i.e. on the seed.  The value moves by
# 7e-5 relative (the entry measure vanishes towards both ends), inside the pins below.
STORE_SEED_ARC_ARCLENGTH = 2.633485946


def _scene(faces: tuple[int, ...]) -> StripScene:
    return build_strip_scene(
        faces,
        sun_direction=canonical_sun_direction(),
        refractive_index=CANONICAL_REFRACTIVE_INDEX,
        crystal=RECORDED_CRYSTAL,
        pose_density=canonical_pose_density(),
        render=CANONICAL_RENDER,
    )


@pytest.fixture(scope="module")
def scenes() -> dict[tuple[int, ...], StripScene]:
    return {PATH_3_5_FACES: _scene(PATH_3_5_FACES), REFLECTING: _scene(REFLECTING)}


@pytest.fixture(scope="module")
def options() -> PixelOptions:
    return PixelOptions()


def test_3_1_2_5_shares_the_phi_invariants_of_3_5():
    crystal = RECORDED_CRYSTAL
    assert np.allclose(fold_matrix(crystal, REFLECTING), np.eye(3), atol=1e-15)
    assert wedge_angle_deg(crystal, REFLECTING) == pytest.approx(wedge_angle_deg(crystal, PATH_3_5_FACES), abs=1e-9)
    assert halo_map_rank(crystal, REFLECTING) == halo_map_rank(crystal, PATH_3_5_FACES) == 2
    path_class = build_path_class(crystal, REFLECTING)
    assert path_class.size == 24 and path_class.wedge_deg == pytest.approx(60.0, abs=1e-9)
    assert (3, 2, 1, 5) in path_class.members and PATH_3_5_FACES not in path_class.members


def test_frozen_seed_is_inside_the_reflecting_domain_and_traces_an_arc():
    seed = np.asarray(exp(jnp.asarray(REFLECTION_DISCOVERY["recorded"]["3-1-2-5"]["seed_exponential_coordinates"])))
    incident = canonical_incident_direction()
    check = path_domain(seed, REFLECTING, incident, CANONICAL_REFRACTIVE_INDEX)
    assert check.valid
    assert check.margins["internal_1_incidence_cosine"] > 0 and check.margins["internal_1_tir_discriminant"] > 0
    assert check.margins["internal_2_incidence_cosine"] > 0 and check.margins["internal_2_tir_discriminant"] > 0
    footprint = entry_measure(seed, REFLECTING, incident, RECORDED_CRYSTAL, n_ice=CANONICAL_REFRACTIVE_INDEX)
    assert footprint.status == "ok" and footprint.value > 0.0
    problem = path_problem(
        jnp.asarray(seed), REFLECTING, jnp.asarray(incident),
        target_direction=jnp.asarray(pixel_target(CANONICAL_RENDER, ROW, COLUMN)),
        refractive_index=jnp.asarray(CANONICAL_REFRACTIVE_INDEX, dtype=jnp.float64),
    )
    result = trace_fiber(problem, PixelOptions().continuation)
    assert result.status == FiberStatus.EVENT_TERMINATED and result.reason == TerminationReason.PATH_INFEASIBLE
    assert result.residual_norms.max() <= 1e-8


def test_reflecting_member_integrates_with_the_same_map_and_a_different_weight(scenes, options):
    results = {faces: render_pixel(scene, ROW, COLUMN, options) for faces, scene in scenes.items()}
    recorded = REFLECTION_DISCOVERY["recorded"]
    plain, reflecting = results[PATH_3_5_FACES], results[REFLECTING]
    assert plain.completeness == "complete" and reflecting.completeness == "complete"
    assert plain.component_count == 1 and plain.components[0].kind == "closed"
    assert plain.value == pytest.approx(recorded["3-5"]["value"], rel=1e-3)
    assert reflecting.component_count == 1
    component = reflecting.components[0]
    assert component.kind == "arc" and component.reason == "path_infeasible" and component.start_reason == "path_infeasible"
    assert component.arclength == pytest.approx(STORE_SEED_ARC_ARCLENGTH, rel=1e-6)
    assert component.arclength > recorded["3-1-2-5"]["arclength"]
    assert reflecting.value == pytest.approx(recorded["3-1-2-5"]["value"], rel=1e-3)
    assert reflecting.value / plain.value == pytest.approx(recorded["value_ratio"], rel=2e-3)
    assert reflecting.value != plain.value  # different weight ...

    # ... same direction map: every accepted pose of the reflecting fiber hits the target through the 3-5 map,
    # and lies on the 3-5 fiber (a pose on both smooth domains with a positive footprint of each path).
    target = pixel_target(CANONICAL_RENDER, ROW, COLUMN)
    incident = canonical_incident_direction()
    discovered = discover_components(
        target, scenes[REFLECTING].seeds,
        template=scenes[REFLECTING].discovery_template, **options.discovery_kwargs(),
    )
    assert discovered.component_count == 1 and discovered.components[0].kind == "arc"
    poses = np.asarray(discovered.components[0].result.poses)
    assert len(poses) > 10
    for pose in poses:
        assert np.linalg.norm(np.asarray(path_3_5(jnp.asarray(pose), jnp.asarray(incident)).direction) - target) < 1e-8
        assert path_domain(pose, PATH_3_5_FACES, incident, CANONICAL_REFRACTIVE_INDEX).valid
    weights_3_5 = scenes[PATH_3_5_FACES].production_template.weight_evaluators
    weights_reflecting = scenes[REFLECTING].production_template.weight_evaluators
    footprint_3_5 = weights_3_5["entry_measure"].evaluate_batch(poses)
    footprint_reflecting = weights_reflecting["entry_measure"].evaluate_batch(poses)
    # The reflecting footprint closes (continuously, to zero) before the infinite-prism ``path_infeasible``
    # event that ends the arc, so it is positive on part of the curve only; ``path_validity`` zeroes the rest.
    positive = footprint_reflecting > 0.0
    assert 0 < positive.sum() < len(poses)
    assert np.all(footprint_3_5[positive] > 0.0)
    # Different partitions of the face-3 entrants (direct vs two basal bounces): neither dominates the other.
    assert not np.allclose(footprint_reflecting[positive], footprint_3_5[positive], rtol=1e-3)
    assert np.any(footprint_reflecting[positive] > footprint_3_5[positive]) and np.any(footprint_reflecting[positive] < footprint_3_5[positive])
    # Fresnel only counts the two refracting interfaces, shared by the two paths at the same pose.
    np.testing.assert_allclose(
        weights_reflecting["fresnel_transmission"].evaluate_batch(poses),
        weights_3_5["fresnel_transmission"].evaluate_batch(poses),
        rtol=0.0, atol=1e-12,
    )
