"""C16 local, non-certified target/exact-family conformance (literal path 3-5)."""
import json

import numpy as np
import pytest

from lumice_integral.camera import sun_direction
from lumice_integral.family_restricted import ExactFamily
from lumice_integral.family_target import (
    C16_INDEX,
    check_target_pose,
    intersect_target,
    locate_elevation_witness,
)
from lumice_integral.geometry import HexPrism

SUN = sun_direction(20)
COLUMN = ExactFamily("column")
PARRY = ExactFamily("parry")


def assert_replay(result):
    # No numpy containers or nonfinite numbers in the public fixture payload.
    json.dumps(result, allow_nan=False)
    assert result["coverage"] == "local_numerical_not_certified"
    assert result["completeness"] == "unknown"
    for point in result["records"]:
        replay = check_target_pose(ExactFamily(result["family"], roll=result["roll_lock"]),
                                   point["q"], result["sun"], result["target"],
                                   refractive_index=result["refractive_index"])["witness"]
        assert replay["status"] == "computed"
        assert replay["direction_residual"] < 1e-10
        assert replay["so3_rank"] == 2
        assert replay["A"] > 0
        assert replay["T"] > 0
        np.testing.assert_allclose(replay["rotation"], point["rotation"], atol=1e-12)


@pytest.fixture(scope="module")
def column_fold():
    return locate_elevation_witness(COLUMN, SUN, 40, [3.767, -0.05])


@pytest.fixture(scope="module")
def parry_point():
    return locate_elevation_witness(PARRY, SUN, 40, [3.767])


def test_old_40_30_target_has_two_transverse_roots_not_a_tangent():
    result = intersect_target(COLUMN, SUN, sun_direction(40, 30),
                              [[3.766, -0.11], [3.766, 0.023]])
    assert result["status"] == "computed"
    assert len(result["records"]) == 2
    assert not result["unresolved_seeds"]
    assert_replay(result)
    for point in result["records"]:
        assert point["intersection"] == "transverse"
        assert abs(point["first_derivative"]) > 1e-3
        assert point["constraint_residual"] < 1e-12
    assert np.linalg.norm(np.asarray(result["records"][0]["rotation"]) -
                          result["records"][1]["rotation"]) > 0.1


def test_column_tangent_recomputed_not_rounded_target_or_merged_roots(column_fold):
    assert column_fold["status"] == "computed"
    assert_replay(column_fold)
    # Published scratch coordinate is a locator, not an exact target authority.
    assert column_fold["target_azimuth_deg"] == pytest.approx(29.680390583, abs=2e-6)
    assert column_fold["target_azimuth_deg"] == pytest.approx(29.680392011006834, abs=1e-9)
    point, = column_fold["records"]
    assert point["intersection"] == "tangent"
    assert abs(point["first_derivative"]) < 1e-9
    assert abs(point["second_derivative"]) > 0.1
    assert point["so3_singular_values"][1] > 0.1
    assert point["A"] == pytest.approx(0.08547, abs=1e-5)
    offsets = point["corrected_offsets"]
    for offset in offsets:
        assert offset["direction_residual"] < 1e-10
        # Both sides have the same sign and the leading term is quadratic.
        assert offset["constraint"] * point["second_derivative"] > 0
        assert 2 * offset["constraint"] / offset["phase"] ** 2 == pytest.approx(
            point["second_derivative"], rel=0.02)
    assert offsets[0]["constraint"] / offsets[1]["constraint"] == pytest.approx(4, rel=0.02)
    assert offsets[3]["constraint"] / offsets[2]["constraint"] == pytest.approx(4, rel=0.02)
    # Explicitly replay the newly located target, rather than widening tolerance.
    replay = intersect_target(COLUMN, SUN, column_fold["target"], [point["q"]])
    assert replay["records"][0]["intersection"] == "tangent"
    # A local least-squares stop near this quartic minimum is NOT a pair of
    # resolved transverse roots. A precise locator is needed for this regime.
    near = intersect_target(COLUMN, SUN, column_fold["target"],
                            [np.asarray(point["q"]) + [0, 1e-3]])
    assert all(p["intersection"] == "tangent" for p in near["records"])
    if not near["records"]:
        assert near["status"] == "unsupported"
    # Nor is the old rounded coordinate silently snapped onto the recomputed fold.
    rounded = intersect_target(COLUMN, SUN, sun_direction(40, 29.680390583), [point["q"]])
    assert rounded["status"] == "unsupported"
    assert rounded["records"] == []


def test_parry_is_a_positive_regular_pose_witness_not_a_distance_threshold(parry_point):
    assert parry_point["status"] == "computed"
    assert_replay(parry_point)
    assert parry_point["target_azimuth_deg"] == pytest.approx(29.835010151, abs=2e-6)
    assert parry_point["target_azimuth_deg"] == pytest.approx(29.835011461218983, abs=1e-9)
    point, = parry_point["records"]
    assert point["intersection"] == "isolated"
    assert point["family_sky_speed"] > 0.1
    result = intersect_target(PARRY, SUN, parry_point["target"], [[3.766]])
    assert len(result["records"]) == 1
    assert_replay(result)
    miss = intersect_target(PARRY, SUN, sun_direction(40, 30), [[3.766]])
    assert miss["status"] == "unsupported"
    assert miss["records"] == []
    assert miss["reason"] == "no_verified_local_intersection"


@pytest.mark.parametrize("name", ["parry", "lowitz"])
def test_u_projection_and_deviation_match_are_not_target_hit(parry_point, name):
    point = parry_point["records"][0]
    family = ExactFamily(name)
    wrong_target = np.asarray(parry_point["target"]).copy()
    wrong_target[1] *= -1  # same deviation from sun, a different sky azimuth
    np.testing.assert_allclose(wrong_target @ SUN, np.asarray(point["sky"]) @ SUN, atol=1e-14)
    lifts = family.lifts(np.asarray(point["u"]), SUN)
    assert len(lifts) == 1
    result = check_target_pose(family, lifts[0], SUN, wrong_target)
    assert result["witness"]["status"] == "rejected"
    assert result["witness"]["reason"] == "full_sky_direction_mismatch"
    assert result["witness"]["direction_residual"] > 0.5
    json.dumps(result, allow_nan=False)


def test_lowitz_is_exact_roll_zero_with_free_azimuth_and_zenith(parry_point):
    # Parry is the zenith=pi/2 section of this exact Lowitz sheet.
    q = [parry_point["records"][0]["q"][0], np.pi / 2]
    result = intersect_target(ExactFamily("lowitz"), SUN, parry_point["target"], [q])
    assert result["status"] == "computed"
    assert len(result["records"]) == 1
    assert_replay(result)
    p = result["records"][0]
    assert p["constraint"] == "roll_lock"
    assert abs(p["rotation"][2][1]) < 1e-12
    assert 0 < p["q"][1] < np.pi
    # Do not silently add the roll+pi sheet by folding a negative zenith.
    with pytest.raises(ValueError, match="Lowitz zenith"):
        intersect_target(ExactFamily("lowitz"), SUN, parry_point["target"], [[q[0], -0.1]])


def test_mirrored_sun_target_has_whole_closed_fiber_inside_column():
    target = sun_direction(-20)
    # This seed is itself solved first; coincidence does not accept a nearby pose.
    # A precise seed is obtained from an arbitrary regular column direction pair.
    from scipy.optimize import least_squares
    from lumice_integral import optics
    import jax
    import jax.numpy as jnp

    sky = jax.jit(lambda q: -optics.path_direction(COLUMN.rotation(q), (3, 5),
                                                  jnp.asarray(-SUN), jnp.asarray(C16_INDEX)).direction)
    solved = least_squares(lambda q: np.asarray(sky(q)) - target, [0.45, 2.1],
                           jac=jax.jit(jax.jacfwd(sky)), gtol=1e-14, ftol=1e-14, xtol=1e-14)
    # The public input remains a locator, with the same bounded correction used
    # for other target solves; a 550nm hard-coded exact seed is unnecessary.
    result = intersect_target(COLUMN, SUN, target, [[.45, 2.1]])
    assert result["status"] == "computed"
    assert result["records"] == []
    curve, = result["curves"]
    assert curve["intersection"] == "coincident"
    assert curve["dimension"] == 1
    assert curve["closed"]
    assert curve["trace_reason"] == "closed_loop"
    assert curve["component_completeness"] == "unknown"
    assert len(curve["points"]) > 20
    assert "conservation" in curve["containment"]
    for p in curve["points"]:
        assert p["constraint_residual"] < 1e-9
        assert p["direction_residual"] < 1e-10
        assert p["A"] > 0
        assert p["T"] > 0
    json.dumps(result, allow_nan=False)
    # A stopped trace is not silently relabeled a coincident closed component.
    stopped = intersect_target(COLUMN, SUN, target, [solved.x], maximum_trace_steps=2)
    assert stopped["status"] == "unsupported"
    assert stopped["curves"] == []
    assert stopped["trace_reason"] == "step_budget"


def test_plate_pose_and_lowitz_endpoint_stratum_are_distinct():
    from lumice_integral import optics
    import jax.numpy as jnp

    plate = ExactFamily("plate")
    q = [3.76]
    target = -np.asarray(optics.path_direction(plate.rotation(jnp.asarray(q)), (3, 5),
                                              jnp.asarray(-SUN), jnp.asarray(C16_INDEX)).direction)
    plate_result = intersect_target(plate, SUN, target, [q])
    assert plate_result["status"] == "computed"
    assert plate_result["records"][0]["intersection"] == "isolated"
    assert_replay(plate_result)
    lowitz_result = intersect_target(ExactFamily("lowitz"), SUN, target, [[q[0], 0.]])
    assert lowitz_result["status"] == "unsupported"
    assert lowitz_result["records"] == []
    assert lowitz_result["unresolved_seeds"][0]["reason"] == "degenerate_or_endpoint_contact_unresolved"


def test_unsupported_geometry_and_unreached_roots_are_fail_closed():
    asymmetric = HexPrism.from_lumice(1, face_distance=[1, 1.2, 1, 1.2, 1, 1.2])
    result = intersect_target(COLUMN, SUN, sun_direction(40, 30), [[3.766, 0.02]], crystal=asymmetric)
    assert result["status"] == "unsupported"
    assert result["records"] == []
    empty_seeds = intersect_target(COLUMN, SUN, sun_direction(40, 30), [])
    assert empty_seeds["status"] == "unsupported"
    assert empty_seeds["reason"] == "explicit_seeds_required"
    unsupported = locate_elevation_witness(ExactFamily("lowitz"), SUN, 40, [3.766, np.pi / 2])
    assert unsupported["status"] == "unsupported"
    exhausted = intersect_target(COLUMN, SUN, sun_direction(40, 30), [[3.7, 0.02]], maximum_evaluations=1)
    assert exhausted["status"] == "unsupported"
    assert exhausted["unresolved_seeds"]
