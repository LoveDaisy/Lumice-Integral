"""Caller-owned family fixtures; independent NumPy direction and displacement oracles.

The chain is Rz(az-pi) Ry(-zenith) Rz(roll), all parameters in radians.
No writing-repository imports or data are needed.
"""

import numpy as np
import pytest

from lumice_integral.camera import sun_direction
from lumice_integral.family_sky import diagnose_sky, sky_differential
from lumice_integral.geometry import HexPrism, Polyhedron, fold_matrix
from lumice_integral.so3 import haar_rotations
from _geometry_oracles import reflect, refract
from lumice_integral.optics import path_domain_batch
from lumice_integral.symmetry.signature import phi_batch


CRYSTAL = HexPrism()
SUN = sun_direction(15.0)
PATH = (3, 5)


def rz(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def ry(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def chain(parameters):
    az, zenith, roll = parameters
    return rz(az - np.pi) @ ry(-zenith) @ rz(roll)


def generators(parameters, axes):
    """Analytic body angular velocities, independently checked by displacements."""
    az, _, _ = parameters
    r = chain(parameters)
    world = (np.array([0.0, 0.0, 1.0]), rz(az - np.pi) @ np.array([0.0, -1.0, 0.0]))
    all_axes = (r.T @ world[0], r.T @ world[1], np.array([0.0, 0.0, 1.0]))
    return np.array([all_axes[k] for k in axes])


def independent_sky(rotations, sun=SUN):
    r = np.asarray(rotations).reshape(-1, 3, 3)
    incoming = -np.einsum("nji,j->ni", r, sun)
    outgoing = phi_batch(CRYSTAL, fold_matrix(CRYSTAL, PATH), *PATH, incoming)
    return -np.einsum("nij,nj->ni", r, outgoing)


def parameter_fd(parameters, axes, step=1e-5, sun=SUN):
    columns = []
    for axis in axes:
        delta = np.eye(3)[axis] * step
        pair = independent_sky([chain(parameters + delta), chain(parameters - delta)], sun)
        columns.append((pair[0] - pair[1]) / (2 * step))
    return np.array(columns)


def lowitz_valleys():
    """Original 3-deg az / 0.25-deg zenith search, including the special point.

    Keep the original endpoint inset, <= comparisons and all candidates. The
    special symmetry point is separated by coordinates, never by measured T.
    """
    zeniths = np.radians(np.arange(1.0, 179.0, 0.25))
    rows = []
    for az in np.radians(np.arange(0.0, 360.0, 3.0)):
        params = np.array([[az, z, 0.0] for z in zeniths])
        sky = independent_sky([chain(p) for p in params])
        idx = np.flatnonzero(np.isfinite(sky[:, 0]))
        if len(idx) < 20:
            continue
        inner = idx[(idx >= idx[0] + 4) & (idx <= idx[-1] - 4)]
        if len(inner) < 10:
            continue
        deviation = np.arccos(np.clip(sky[inner] @ SUN, -1, 1))
        minima = np.flatnonzero((deviation[1:-1] <= deviation[:-2]) & (deviation[1:-1] <= deviation[2:])) + 1
        for k in minima:
            p = params[inner[k]]
            if np.isfinite(parameter_fd(p, (0, 1), np.radians(0.02))).all():
                rows.append(p)
    return np.array(rows)


def test_analytic_generators_and_independent_direction():
    parameters = np.radians([[270, 90, 0], [90, 90, 115.90], [210, 90, 5.10], [282, 83, 4]])
    rotations = np.array([chain(p) for p in parameters])
    reference = independent_sky(rotations)
    domain = path_domain_batch(rotations, PATH, -SUN)
    assert domain.valid.all()
    np.testing.assert_allclose(reference, -domain.direction, atol=2e-15, rtol=0)
    for p, r in zip(parameters, rotations):
        for k, omega in enumerate(generators(p, (0, 1, 2))):
            delta = np.eye(3)[k] * 1e-5
            dr = (chain(p + delta) - chain(p - delta)) / 2e-5
            # R^T dR acts as omega cross, without using production SO(3).
            np.testing.assert_allclose(r.T @ dr, np.cross(omega, np.eye(3)).T, atol=3e-10, rtol=0)


def test_lowitz_reference_search_and_correct_speed_units():
    candidates = lowitz_valleys()
    special = np.all(np.isclose(candidates, np.radians([270, 90, 0]), atol=1e-12), axis=1)
    assert len(candidates) == 51
    assert special.sum() == 1
    general = candidates[~special]
    derivatives = np.array([parameter_fd(p, (0, 1)) for p in general])
    sky = independent_sky([chain(p) for p in general])
    area = np.abs(np.einsum("ni,ni->n", sky, np.cross(derivatives[:, 0], derivatives[:, 1])))
    assert area.min() > 1e-4
    assert np.median(area) == pytest.approx(0.040, rel=0.2)
    assert area.max() == pytest.approx(0.132, rel=0.2)
    speed = np.linalg.norm(parameter_fd(np.radians([270, 90, 0]), (0, 1)), axis=1)
    np.testing.assert_allclose(speed, [0.2771, 0.4661], atol=5e-5, rtol=0)


def rodrigues(vector):
    angle = np.linalg.norm(vector)
    if angle == 0:
        return np.eye(3)
    axis = vector / angle
    skew = np.cross(axis, np.eye(3)).T
    return np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * (skew @ skew)


def optical_oracle(rotation, faces, index, crystal=CRYSTAL):
    """NumPy Snell/mirror oracle, not optics.trace_path or signature's fixed n."""
    normals = [rotation @ crystal.normal(crystal.face(f)) for f in faces]
    direction = refract(-SUN, normals[0], 1.0, index)
    for normal in normals[1:-1]:
        direction = reflect(direction, normal)
    return -refract(direction, -normals[-1], index, 1.0)


@pytest.mark.parametrize("k", [1, 2, 3])
def test_raw_batch_parameter_fd_and_single_row(k):
    parameters = np.radians([[282, 83, 4], [270, 90, 0], [261, 104, 2]])
    rotations = np.array([chain(p) for p in parameters])
    omega = np.array([generators(p, tuple(range(k))) for p in parameters])
    result = sky_differential(rotations, omega, PATH, SUN)
    assert result.valid.all() and result.available.all()
    assert result.dx.shape == (3, k, 3)
    np.testing.assert_allclose(result.sky, independent_sky(rotations), atol=2e-15, rtol=0)
    for i, p in enumerate(parameters):
        expected = parameter_fd(p, tuple(range(k)))
        np.testing.assert_allclose(result.dx[i], expected, atol=2e-9, rtol=0)
        single = sky_differential(rotations[i:i+1], omega[i:i+1], PATH, SUN)
        np.testing.assert_allclose(result.dx[i], single.dx[0], atol=2e-15, rtol=0)


@pytest.mark.parametrize("faces,index,off_star", [((3, 5), 1.31, False), ((3, 1, 2, 5), 1.307, False), ((3, 5), 1.317, True)])
def test_random_interior_fd_step_ladder_and_input_sensitivity(faces, index, off_star):
    # Rotate the complete crystal about its c axis: all side normals leave
    # the historical star, but the independent oracle reads the actual mesh.
    crystal = Polyhedron(CRYSTAL.vertices @ rz(0.13).T, CRYSTAL.faces) if off_star else CRYSTAL
    rotations = haar_rotations(2048, np.random.default_rng(52))
    domain = path_domain_batch(rotations, faces, -SUN, index, crystal=crystal)
    from lumice_integral.optics import validity_margin_names
    margin = np.min([domain.margins[name] for name in validity_margin_names(faces)], axis=0)
    rotations = rotations[domain.valid & (margin > 0.05)][:16]
    assert len(rotations) == 16
    omega = np.random.default_rng(5).normal(size=(len(rotations), 2, 3))
    result = sky_differential(rotations, omega, faces, SUN, crystal=crystal, refractive_index=index)
    assert result.available.all()
    expected = np.array([optical_oracle(r, faces, index, crystal) for r in rotations])
    np.testing.assert_allclose(result.sky, expected, atol=3e-15, rtol=0)
    errors = []
    for step in (1e-3, 1e-4, 1e-5, 1e-6):
        fd = np.array([[(optical_oracle(r @ rodrigues(step*w), faces, index, crystal)
                        - optical_oracle(r @ rodrigues(-step*w), faces, index, crystal)) / (2*step)
                       for w in generators_] for r, generators_ in zip(rotations, omega)])
        errors.append(np.max(np.linalg.norm(fd - result.dx, axis=2)))
    assert errors[1] < errors[0] / 50  # central-difference second-order regime
    assert errors[2] < 1e-7 * max(1, np.linalg.norm(result.dx, axis=2).max())
    assert errors[3] < 1e-7 * max(1, np.linalg.norm(result.dx, axis=2).max())
    if index != 1.31 or off_star:
        baseline = sky_differential(rotations, omega, faces, SUN)
        common = baseline.available
        assert common.any()
        assert np.max(np.abs(result.sky[common] - baseline.sky[common])) > 1e-4
        assert np.max(np.abs(result.dx[common] - baseline.dx[common])) > 1e-4


def test_mixed_domain_empty_and_partial_reflection():
    rotations = haar_rotations(2048, np.random.default_rng(3))
    omega = np.broadcast_to(np.eye(3)[:2], (len(rotations), 2, 3))
    faces = (3, 5, 6, 7)
    result = sky_differential(rotations, omega, faces, SUN)
    domain = path_domain_batch(rotations, faces, -SUN)
    np.testing.assert_array_equal(result.valid, domain.valid)
    assert result.valid.any() and (~result.valid).any()
    assert (result.valid & (result.margins["internal_1_tir_discriminant"] < 0)).any()
    assert (~result.valid & (result.margins["entry_incidence_cosine"] < 0)).any()
    assert (~result.valid & (result.margins["exit_snell_discriminant"] < 0)).any()
    assert np.isnan(result.sky[~result.valid]).all()
    assert np.isnan(result.dx[~result.valid]).all()
    diagnostic = diagnose_sky(result)
    assert (diagnostic.mechanism[~result.valid] == "invalid").all()
    assert np.isnan(diagnostic.singular_values[~result.valid]).all()
    empty = sky_differential(np.empty((0, 3, 3)), np.empty((0, 2, 3)), PATH, SUN)
    assert empty.sky.shape == (0, 3) and empty.dx.shape == (0, 2, 3)
    assert diagnose_sky(empty).gram.shape == (0, 2, 2)


@pytest.mark.parametrize("name,value", [
    ("rotations", np.eye(3)), ("rotations", np.zeros((1, 3, 3))),
    ("rotations", -np.eye(3)[None]), ("rotations", np.full((1, 3, 3), np.nan)),
    ("tangent_generators", np.ones((2, 2, 3))), ("tangent_generators", np.ones((1, 0, 3))),
    ("tangent_generators", np.full((1, 2, 3), np.inf)), ("sun_direction", SUN*2),
    ("sun_direction", SUN[None]), ("sun_direction", np.full(3, np.nan)),
    ("refractive_index", 0), ("refractive_index", np.inf), ("refractive_index", [1.31]),
    ("faces", (3,)), ("faces", (3, 99)),
])
def test_input_validation(name, value):
    p = np.radians([270, 90, 0])
    inputs = dict(rotations=chain(p)[None], tangent_generators=generators(p, (0, 1))[None], faces=PATH, sun_direction=SUN)
    inputs[name] = value
    with pytest.raises(ValueError):
        sky_differential(**inputs)


def test_nonfinite_ad_is_explicit_and_does_not_poison_batch(monkeypatch):
    import lumice_integral.family_sky as module
    p = np.radians([270, 90, 0])
    kernel = module._differential_batch

    def broken(*args):
        sky, dx = map(np.array, kernel(*args))
        dx[0, 0, 0] = np.nan
        return sky, dx

    monkeypatch.setattr(module, "_differential_batch", broken)
    result = sky_differential(np.repeat(chain(p)[None], 2, axis=0), np.repeat(generators(p, (0, 1))[None], 2, axis=0), PATH, SUN)
    assert result.valid.all()
    assert result.available.tolist() == [False, True]
    assert np.isnan(result.sky[0]).all() and np.isnan(result.dx[0]).all()
    diagnostic = diagnose_sky(result)
    assert diagnostic.mechanism.tolist() == ["unavailable", "parallel_collapse"]
