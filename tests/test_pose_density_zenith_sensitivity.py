"""Zenith-width sweep of the column family reproduces ``defect2_findings.md`` section 4.

The 2026-09-17 probe (``scripts/probe_defect2_factors.py``, task
strip-rerender-and-compare, ``artifacts/defect2_probe.json``) reweighted the
traced integrand of the column-126 probe pixels by ``rho(std') / rho(0.5 deg)``
-- with an *independent* ``scipy.integrate.quad``-normalised Gaussian -- and
trapezoid-integrated along arclength.  Here the same sweep runs through the
production density classes (``build_pose_density("column", ...)``) and
``scripts/compare_pose_density_families.py::zenith_width_sensitivity``; the 16
cells must each agree with the recorded table to 1.5 % (the table is rounded to
three decimals; the probe's own run-to-run scatter is well below that).

The table was recorded on the ``h/a = 1`` crystal of 2026-09-17; the canonical
scene moved to ``h/a = 2`` on 2026-09-20 (task
defect2-crystal-height-convention), which reweights the fiber through
``entry_measure`` and changes the ratios, so the scene here binds the
recording's crystal explicitly (``TABLE_CRYSTAL``).  The check is about the
density classes, not about the crystal.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from lumice_integral.continuation import trace_fiber
from lumice_integral.discovery import retarget_problem
from lumice_integral.geometry import HexPrism
from lumice_integral.pose_density import ZenithGaussianPoseDensity
from lumice_integral.quadrature import HAAR_TO_DVOL_G_FACTOR
from lumice_integral.strip_pixel import PixelOptions, canonical_strip_scene, pixel_target, render_pixel

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "compare_pose_density_families.py"

# defect2_findings.md section 4 (rows x std): value ratio to the 0.5 deg render.
DEFECT2_SECTION_4 = {
    150: {"0.25": 1.001, "0.5": 1.0, "1": 0.921, "2": 0.595},
    300: {"0.25": 1.149, "0.5": 1.0, "1": 0.651, "2": 0.353},
    450: {"0.25": 1.526, "0.5": 1.0, "1": 0.545, "2": 0.279},
    600: {"0.25": 1.078, "0.5": 1.0, "1": 0.828, "2": 0.536},
}
RELATIVE_TOLERANCE = 0.015
TABLE_CRYSTAL = HexPrism.from_ratio(1.0)


@pytest.fixture(scope="module")
def families_script():
    spec = importlib.util.spec_from_file_location("compare_pose_density_families", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def scene():
    return canonical_strip_scene(crystal=TABLE_CRYSTAL)


@pytest.mark.parametrize("row", sorted(DEFECT2_SECTION_4))
def test_column_zenith_width_sweep_reproduces_the_defect2_table_cell_by_cell(families_script, scene, row):
    assert families_script.PROBE_COLUMN == 126 and families_script.WIDTH_SENSITIVITY_STD_DEG == (0.25, 0.5, 1.0, 2.0)
    options = PixelOptions()
    result = render_pixel(scene, row, families_script.PROBE_COLUMN, options)
    assert result.completeness == "complete" and result.component_count == 1, row
    record = result.components[0]
    fiber = trace_fiber(retarget_problem(scene.production_template, pixel_target(scene.render, row, 126), record.seed))
    assert isinstance(scene.pose_density, ZenithGaussianPoseDensity)
    ratios = families_script.zenith_width_sensitivity(
        fiber, base_density=scene.pose_density, epsilon=options.quadrature.epsilon
    )
    expected = DEFECT2_SECTION_4[row]
    assert set(ratios) == set(expected)
    assert ratios["0.5"] == pytest.approx(1.0, rel=1e-12)
    for std, value in expected.items():
        assert ratios[std] == pytest.approx(value, rel=RELATIVE_TOLERANCE), f"row {row}, std {std}: {ratios[std]:.4f} vs {value}"
    # the trapezoid base value is the integral the resampled quadrature converged to (Haar-converted; the
    # trapezoid on the raw trace is a coarser rule, hence the loose tolerance)
    integrand = families_script.pointwise_integrand(fiber, epsilon=options.quadrature.epsilon)
    trapezoid = np.trapezoid(integrand, families_script.fiber_arclength(fiber)) * HAAR_TO_DVOL_G_FACTOR
    assert trapezoid == pytest.approx(result.value, rel=0.05)
