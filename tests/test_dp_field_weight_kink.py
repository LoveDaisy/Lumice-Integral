"""``dp_field.weight_kink``: the TIR onsets ``C_k``, closed form on small circles and marched otherwise."""

from __future__ import annotations

import numpy as np
import pytest

from lumice_integral import optics
from lumice_integral.dp_field import DPField
from lumice_integral.dp_field import weight_kink as W
from lumice_integral.geometry import HexPrism

SLABS = ((3, 1, 6), (1, 3, 2))
INDICES = (1.307, 1.31, 1.317)
# the antisolar dark hole of a single-mirror slab: 180 - 2 arcsin sqrt(n^2 - 1), degrees (issue, probe_hole.py)
HOLE_RADIUS_DEG = {1.307: 65.4, 1.31: 64.4, 1.317: 62.0}


def _gates(field: DPField, points: np.ndarray) -> np.ndarray:
    return field.validity_margins_batch(points)


def _discriminant(field: DPField, curve: W.KinkCurve) -> np.ndarray:
    return field.margins_batch(curve.points)[:, optics.domain_margin_names(field.faces).index(curve.margin)]


@pytest.mark.parametrize("faces", SLABS)
@pytest.mark.parametrize("index", INDICES)
def test_single_mirror_slab_kink_is_one_level_of_d(faces, index) -> None:
    """``C_1`` is the small circle ``m . u = -sqrt(n^2 - 1)`` and ``D_P = 2 arcsin sqrt(n^2 - 1)`` all along it."""
    field = DPField.build(HexPrism(), faces, index)
    (curve,) = field.weight_kinks
    assert curve.method == "great_circle" and curve.margin == "internal_1_tir_discriminant"
    assert curve.arcs and len(curve.points) > 1000
    expected = 2.0 * np.arcsin(np.sqrt(index * index - 1.0))
    assert np.max(np.abs(curve.values - expected)) < 1e-12
    assert curve.spread < 1e-12
    assert np.max(np.abs(_discriminant(field, curve))) < 1e-14
    assert np.all(_gates(field, curve.points) > -1e-12)  # in the closure of U_P
    assert 180.0 - np.degrees(expected) == pytest.approx(HOLE_RADIUS_DEG[index], abs=0.05)
    np.testing.assert_allclose(np.abs(curve.points @ curve.normal), np.sqrt(index * index - 1.0), atol=1e-14)


@pytest.mark.parametrize("faces", SLABS)
def test_kink_arc_ends_on_the_gate_it_names(faces) -> None:
    field = DPField.build(HexPrism(), faces, 1.307)
    names = optics.validity_margin_names(faces)
    for arc in field.weight_kinks[0].arcs:
        assert not arc.closed
        for point, gate in zip((arc.points[0], arc.points[-1]), arc.ends, strict=True):
            assert abs(_gates(field, point[None, :])[0, names.index(gate)]) < 1e-9


def test_slab_index_derivatives() -> None:
    """A slab's direction does not disperse (``dD/dn = 0``); on its kink ``d disc / dn = 2 n > 0`` (blue side total)."""
    index = 1.31
    field = DPField.build(HexPrism(), (3, 1, 6), index)
    (curve,) = field.weight_kinks
    d_dn, margin_dn = field.index_derivatives_batch(curve.points[::50])
    assert np.max(np.abs(d_dn)) == 0.0
    k = optics.domain_margin_names(field.faces).index(curve.margin)
    np.testing.assert_allclose(margin_dn[:, k], 2.0 * index, rtol=1e-12)


def test_non_orthogonal_normal_is_marched() -> None:
    """``3-7-5``: ``m_1 . n_a = -1/2``, not a circle of the closed form; the walk stays on the zero set and ends on gates."""
    field = DPField.build(HexPrism(), (3, 7, 5), 1.31)
    (curve,) = field.weight_kinks
    assert curve.method == "marched" and curve.note == "" and curve.arcs
    assert np.max(np.abs(_discriminant(field, curve))) < 1e-12
    assert np.all(_gates(field, curve.points) > -1e-12)
    names = optics.validity_margin_names(field.faces)
    for arc in curve.arcs:
        for point, gate in zip((arc.points[0], arc.points[-1]), arc.ends, strict=True):
            assert abs(_gates(field, point[None, :])[0, names.index(gate)]) < 1e-9


def test_walk_agrees_with_the_closed_form_on_a_slab() -> None:
    """The marched walk forced onto ``3-1-6`` reproduces the closed-form circle (consistency; the closed form is the authority)."""
    index = 1.31
    field = DPField.build(HexPrism(), (3, 1, 6), index)
    (closed_form,) = field.weight_kinks
    marched = W.marched_kink(field.crystal, field.faces, index, 1, slab=field.slab, lattice_n=field.lattice_n)
    assert marched.method == "marched" and marched.arcs
    np.testing.assert_allclose(marched.points @ closed_form.normal, -np.sqrt(index * index - 1.0), atol=1e-12)
    np.testing.assert_allclose(marched.values, 2.0 * np.arcsin(np.sqrt(index * index - 1.0)), atol=1e-12)
    # the walk covers the same arc: its ends are the closed form's ends
    ends = np.array([closed_form.arcs[0].points[0], closed_form.arcs[0].points[-1]])
    for end in (marched.arcs[0].points[0], marched.arcs[0].points[-1]):
        assert np.min(np.linalg.norm(ends - end, axis=1)) < 1e-6


def test_liljequist_onset_maximum_is_on_the_marched_kinks() -> None:
    """``3-5-6-7-3`` (h/a 2): the largest ``D_P`` on each onset is ``ch10_verdicts.tir_onset_maximum``'s 153.0697 deg.

    The walk samples at 0.25 deg steps, so its maximum is below the optimiser's by at most the curvature over half
    a step; the three reflections step by the same 60 deg, so the three onsets carry the same values.
    """
    field = DPField.build(HexPrism.from_ratio(2.0), (3, 5, 6, 7, 3), 1.31)
    assert [curve.method for curve in field.weight_kinks] == ["marched"] * 3
    for curve in field.weight_kinks:
        assert np.max(np.abs(_discriminant(field, curve))) < 1e-12
        top = np.degrees(np.max(curve.values))
        assert 153.0697 - 5e-3 < top <= 153.0697 + 1e-4


def test_a_b_path_has_no_kink() -> None:
    assert DPField.build(HexPrism(), (3, 5), 1.31).weight_kinks == ()


def test_kinks_do_not_change_the_boundary() -> None:
    """The TIR discriminants stay out of ``dU_P``: no boundary piece is walked on one."""
    field = DPField.build(HexPrism(), (3, 1, 5), 1.31)
    assert field.weight_kinks
    assert not any(piece.margin.endswith("_tir_discriminant") for piece in field.boundary_curves)


def test_weight_kink_uses_only_the_public_walk_interface() -> None:
    """No private name of ``boundary`` is imported by ``weight_kink``, and no walker state is mutated there."""
    import ast
    from pathlib import Path

    source = Path(W.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = [
        alias.name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module == "boundary" for alias in node.names
    ]
    assert imported and not [name for name in imported if name.startswith("_")]
    assigned = [
        target.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Attribute)
    ]
    assert assigned == []


def test_a_failed_seed_walk_does_not_stop_the_others(monkeypatch) -> None:
    """``3-5-6-7-3``: the first seed's walk raises; the remaining seeds still give arcs and the failure is counted."""
    faces, index = (3, 5, 6, 7, 3), 1.31
    crystal = HexPrism.from_ratio(2.0)
    reference = W.marched_kink(crystal, faces, index, 1, slab=None)
    assert reference.complete and reference.note == "" and reference.arcs
    walk = W._walk_both_ways
    calls = []

    def first_fails(walker, start, margin):
        calls.append(start)
        if len(calls) == 1:
            raise RuntimeError("injected walk failure")
        return walk(walker, start, margin)

    monkeypatch.setattr(W, "_walk_both_ways", first_fails)
    curve = W.marched_kink(crystal, faces, index, 1, slab=None)
    assert curve.failed_seeds == 1 and not curve.complete
    assert "1 seed walk(s) failed" in curve.note and "injected walk failure" in curve.note
    assert curve.arcs and len(calls) > 1
    # the seeds after the failure recover most of the onset (not all: the failed seed's own stretch may be lost)
    gap = np.linalg.norm(reference.points[::20, None, :] - curve.points[None, ::5], axis=2).min(axis=1)
    assert np.mean(gap < 0.01) > 0.9


@pytest.mark.filterwarnings("error")
def test_index_above_sqrt_2_has_no_onset_on_the_sphere() -> None:
    """``n^2 - 1 >= 1``: the closed-form circle is empty by construction (said in the note), no NaN comparison."""
    (curve,) = DPField.build(HexPrism(), (3, 1, 6), 1.5).weight_kinks
    assert curve.method == "great_circle" and curve.arcs == ()
    assert "n^2 - 1 >= 1" in curve.note and curve.complete
