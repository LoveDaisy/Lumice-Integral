"""``path_class.phi_key``: face sequences with one key share the outgoing-direction map ``Phi``."""

from __future__ import annotations

import itertools
from collections import defaultdict

import numpy as np
import pytest
from _geometry_oracles import phi_group_key

from lumice_integral import optics
from lumice_integral.canonical_scene import CANONICAL_REFRACTIVE_INDEX, canonical_crystal, canonical_incident_direction
from lumice_integral.geometry import HexPrism
from lumice_integral.path_class import phi_key
from lumice_integral.prescan import haar_rotations

ALL_SEQUENCES = [faces for length in (2, 3, 4) for faces in itertools.product(range(1, 9), repeat=length)]


@pytest.fixture(scope="module")
def keys() -> dict[tuple[int, ...], tuple[int, int, int]]:
    crystal = canonical_crystal()
    return {faces: phi_key(crystal, faces) for faces in ALL_SEQUENCES}


def test_named_pairs() -> None:
    crystal = canonical_crystal()
    assert phi_key(crystal, (3, 5)) == phi_key(crystal, (3, 1, 2, 5))
    assert phi_key(crystal, (3, 5)) != phi_key(crystal, (3, 7))
    assert all(isinstance(v, int) for v in phi_key(crystal, (3, 1, 2, 5)))


def test_key_does_not_depend_on_prism_aspect(keys) -> None:
    plate = HexPrism.from_ratio(0.1)
    assert all(phi_key(plate, faces) == keys[faces] for faces in ALL_SEQUENCES[::37])


def test_partition_matches_independent_oracle(keys) -> None:
    ours: dict = defaultdict(set)
    oracle: dict = defaultdict(set)
    for faces, key in keys.items():
        ours[key].add(faces)
        oracle[phi_group_key(faces)].add(faces)
    assert {frozenset(g) for g in ours.values()} == {frozenset(g) for g in oracle.values()}


def test_equal_key_means_equal_direction(keys) -> None:
    """Every 2/3/4-face sequence against the first member of its key group, 200 Haar poses.

    ``path_direction`` is compared wherever both directions are finite: the
    validity gates of the internal reflections differ between members, the
    closed-form direction does not.  Members reach the same ``M`` through
    different mirror products, so the directions differ by rounding, which
    the exit refraction amplifies by ``1 / sqrt(discriminant)`` near grazing
    exit (one of the 200 poses has discriminant ``7e-8``): ``<= 1e-12`` is
    asserted where the exit discriminant is ``>= 1e-6``, and the
    ``sqrt(discriminant)``-weighted difference everywhere.
    """
    s = canonical_incident_direction()
    rotations = haar_rotations(200, np.random.default_rng(20260923))
    groups: dict = defaultdict(list)
    for faces, key in keys.items():
        groups[key].append(faces)
    worst, worst_weighted, compared = 0.0, 0.0, 0
    for members in groups.values():
        if len(members) < 2:
            continue
        first = optics.path_domain_batch(rotations, members[0], s, CANONICAL_REFRACTIVE_INDEX)
        finite = np.all(np.isfinite(first.direction), axis=1)
        discriminant = first.margins["exit_snell_discriminant"]
        conditioned = finite & (discriminant >= 1e-6)
        assert np.count_nonzero(conditioned) > 0, members[0]
        for faces in members[1:]:
            other = optics.path_domain_batch(rotations, faces, s, CANONICAL_REFRACTIVE_INDEX).direction
            assert np.array_equal(np.all(np.isfinite(other), axis=1), finite), faces
            difference = np.max(np.abs(first.direction - other), axis=1)
            worst = max(worst, float(np.max(difference[conditioned])))
            worst_weighted = max(worst_weighted, float(np.max(difference[finite] * np.sqrt(discriminant[finite]))))
            compared += 1
    assert compared > 1000
    assert worst <= 1e-12, worst
    assert worst_weighted <= 1e-12, worst_weighted
