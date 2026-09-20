"""AC2: the normalisation of the class ``[3, 5]`` under a symmetric and an asymmetric pose density.

1. Column density (canonical scene, zenith Gaussian about 90 deg, spin
   uniform): every one of the 12 members images the canonical pixel
   identically -- the six ``C6`` rotations because the spin is uniform, the
   ``3-7`` half because the crystal's ``C2'`` rotation about the face-3
   normal maps ``3-5`` onto ``3-7`` and the density is invariant under it
   (``theta -> pi - theta``, ``psi -> -psi``).  The class value is therefore
   ``12 x`` the single ``3-5`` value (Lumice ``PBD`` = ``P x 6`` times two),
   and the test asserts the *pointwise* member identity, not only the sum.
2. Tilted Parry density (``build_pose_density("parry", roll_mean_deg=20)``):
   the roll lock away from ``0`` breaks the ``C2'`` invariance.  At the frozen
   pixel (:data:`PARRY_DISCOVERY`, found by
   ``scripts/discover_parry_class_pixel.py``) ``3-5`` is bright while ``3-7``
   is zero -- the *strong* inequality of the issue's dispatch note (chosen
   over "both non-zero, ratio != 1" because the exploration found such a
   pixel at the first attempt, and a zero/non-zero pair cannot be a rounding
   artefact) -- and the class value is not ``12 x`` (nor any ``k x``) the
   ``3-5`` value.  The stock Parry (``roll_mean = 0``) at the same pixel is
   the control: ``3-5`` and ``3-7`` agree pointwise there too, which is why
   the issue's "Parry-type asymmetric density" has to be a tilted one.

The dispatch note's reason for leaving the strip: on the ch06 strip (below
the sun) ``3-5`` is exactly zero under every roll-locked family
(``docs/ch11-pose-density-families.md`` section 5), so the asymmetric case is
rendered on a single-pixel window centred on the upper Parry-arc region.
"""

from __future__ import annotations

import numpy as np
import pytest

from lumice_integral.canonical_scene import CANONICAL_PIXEL_COLUMN, CANONICAL_PIXEL_ROW
from lumice_integral.geometry import HexPrism
from lumice_integral.path_class import ClassPixelResult, canonical_class_scene, render_class_pixel
from lumice_integral.pose_density import build_pose_density
from lumice_integral.strip_pixel import PixelOptions

# The recorded values below (twelve-fold identity, tilted-Parry separation) were
# taken on 2026-09-20 on the ``h/a = 1`` crystal; task defect2-crystal-height-convention
# then moved the canonical scene to ``h/a = 2`` (``entry_measure`` changes with h),
# so these fixtures bind that crystal explicitly -- the class mechanism under test
# does not depend on which crystal it runs on.
RECORDED_CRYSTAL = HexPrism.from_ratio(1.0)
TEST_PRESCAN_SAMPLES = 400_000  # the tests/test_strip_pixel.py table; the canonical value 2.364412980 is pinned to it
# The quadrature's relative tolerance is 1e-4 (PixelOptions.quadrature); the
# members are integrated on independently discovered fibers of the same curve
# family, so their agreement is bounded by twice that.
MEMBER_RTOL = 2.0e-4

# Frozen output of one recorded run of scripts/discover_parry_class_pixel.py
# (2026-09-20).  The landing map that selected the bin used the 5x widened
# density (1 deg x 1 deg leaves a handful of significant samples in 400k);
# the values below are the production render at the selected pixel.
PARRY_DISCOVERY = {
    "script": "scripts/discover_parry_class_pixel.py",
    "command": "uv run python scripts/discover_parry_class_pixel.py --roll-mean-deg 20",
    "pose_density": {"family": "parry", "zenith_std_deg": 1.0, "roll_mean_deg": 20.0, "roll_std_deg": 1.0},
    "prescan": {"sample_count": 400_000, "rng_seed": 20260916},
    "selection": "strong: 3-7 carries no weight where 3-5 does (bin of largest 3-5 weight, 1 deg bins, 5x widened density)",
    "render": {"width": 21, "height": 21, "fov_deg": 6.0, "view": {"azimuth": -32.5, "elevation": 46.5}},
    "pixel": {"row": 10, "column": 10},
    "recorded": {
        "tilted": {"3-5": 8.576691422962443e-4, "3-7": 8.40575516739801e-48, "8-4": 2.2132716242285644e-3, "class_over_12x_3_5": 0.29838},
        "stock_parry": {"3-5": 1.8385041849251435e-37, "3-7": 1.838504176716809e-37},
    },
}


@pytest.fixture(scope="module")
def options() -> PixelOptions:
    return PixelOptions()


@pytest.fixture(scope="module")
def column_class_pixel(options) -> ClassPixelResult:
    scene = canonical_class_scene((3, 5), prescan_sample_count=TEST_PRESCAN_SAMPLES, crystal=RECORDED_CRYSTAL)
    return render_class_pixel(scene, CANONICAL_PIXEL_ROW, CANONICAL_PIXEL_COLUMN, options)


def test_column_density_class_value_is_twelve_times_the_single_3_5_value(column_class_pixel):
    result = column_class_pixel
    assert result.completeness == "complete" and len(result.members) == 12
    single = result.members[(3, 5)].value
    assert single == pytest.approx(2.364412980, rel=1e-6)  # the pinned canonical pixel value
    assert result.value == pytest.approx(12.0 * single, rel=MEMBER_RTOL)
    assert result.error_estimate <= 12.0 * max(member.error_estimate for member in result.members.values())


def test_column_density_images_every_member_pointwise_the_same(column_class_pixel):
    """Not only the sum: each member is one closed loop of the same arclength and value as 3-5."""
    reference = column_class_pixel.members[(3, 5)]
    assert reference.component_count == 1 and reference.components[0].kind == "closed"
    for member, result in column_class_pixel.members.items():
        assert result.completeness == "complete", member
        assert result.component_count == 1 and result.components[0].kind == "closed", member
        assert result.components[0].arclength == pytest.approx(reference.components[0].arclength, rel=1e-4), member
        assert result.components[0].value == pytest.approx(reference.components[0].value, rel=MEMBER_RTOL), member
        assert result.value == pytest.approx(reference.value, rel=MEMBER_RTOL), member
    # The issue's C2' statement, spelled out: 3-7 equals 3-5 at this pixel.
    assert column_class_pixel.members[(3, 7)].value == pytest.approx(reference.value, rel=MEMBER_RTOL)
    provenance = column_class_pixel.provenance
    assert set(provenance["per_member"]) == set(provenance["path_ids"]) and len(provenance["path_ids"]) == 12
    assert all(entry["completeness"] == "complete" for entry in provenance["per_member"].values())


def _parry_pixel(options, **overrides) -> ClassPixelResult:
    parameters = {**PARRY_DISCOVERY["pose_density"], **overrides}
    family = parameters.pop("family")
    scene = canonical_class_scene(
        (3, 5),
        prescan_sample_count=PARRY_DISCOVERY["prescan"]["sample_count"],
        prescan_rng_seed=PARRY_DISCOVERY["prescan"]["rng_seed"],
        pose_density=build_pose_density(family, **parameters),
        render=PARRY_DISCOVERY["render"],
        crystal=RECORDED_CRYSTAL,
    )
    return render_class_pixel(scene, PARRY_DISCOVERY["pixel"]["row"], PARRY_DISCOVERY["pixel"]["column"], options)


def test_tilted_parry_density_separates_3_5_from_3_7_and_breaks_the_simple_multiple(options):
    result = _parry_pixel(options)
    recorded = PARRY_DISCOVERY["recorded"]["tilted"]
    assert result.completeness == "complete"
    value_3_5 = result.members[(3, 5)].value
    value_3_7 = result.members[(3, 7)].value
    assert value_3_5 == pytest.approx(recorded["3-5"], rel=1e-3)
    assert value_3_5 > 1e-4 and value_3_7 < 1e-30  # strong inequality: 3-5 bright, 3-7 dark
    assert result.members[(8, 4)].value == pytest.approx(recorded["8-4"], rel=1e-3)
    ratio = result.value / (12.0 * value_3_5)
    assert ratio == pytest.approx(recorded["class_over_12x_3_5"], rel=1e-3)
    # Not 12x, and not k x for any integer k: the members carry different weights.
    assert abs(12.0 * ratio - round(12.0 * ratio)) > 0.1
    values = np.array([member.value for member in result.members.values()])
    assert (values > 1e-4).sum() == 2 and (values < 1e-30).sum() == 10


def test_stock_parry_density_still_images_3_5_and_3_7_the_same(options):
    """Control of the previous test: with the roll lock at 0 the C2' symmetry holds again."""
    result = _parry_pixel(options, roll_mean_deg=0.0)
    recorded = PARRY_DISCOVERY["recorded"]["stock_parry"]
    value_3_5 = result.members[(3, 5)].value
    value_3_7 = result.members[(3, 7)].value
    assert value_3_5 == pytest.approx(recorded["3-5"], rel=1e-3) and value_3_5 > 0.0
    assert value_3_7 == pytest.approx(value_3_5, rel=1e-6)
    assert result.members[(4, 8)].value == pytest.approx(result.members[(8, 4)].value, rel=1e-6)
