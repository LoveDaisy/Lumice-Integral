"""Caller-owned family fixtures; independent NumPy direction and displacement oracles.

The chain is Rz(az-pi) Ry(-zenith) Rz(roll), all parameters in radians.
No writing-repository imports or data are needed.
"""

import numpy as np
import pytest
from scipy.optimize import brentq, minimize_scalar

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
    ("faces", (3,)), ("faces", (3, 99)), ("faces", (3.2, 5)),
    ("faces", [[3, 5]]), ("tangent_generators", np.ones((1, 2, 3), dtype=complex)),
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


def family_result(parameters, axes, sun=SUN):
    params = np.asarray(parameters).reshape(-1, 3)
    return sky_differential(np.array([chain(p) for p in params]),
                            np.array([generators(p, axes) for p in params]), PATH, sun)


@pytest.mark.parametrize("az,roll", [(90, 115.90), (210, 5.10)])
def test_column_signed_area_root_matches_independent_deviation_valley(az, roll):
    def parameters(r):
        return np.array([np.radians(az), np.pi/2, r])

    def area(r):
        return diagnose_sky(family_result(parameters(r), (0, 2))).signed_area[0]

    center = np.radians(roll)
    historical = area(center)
    # Historical coordinates came from a 0.05-degree grid, not exact roots.
    # At az=210 the actual 1.24164e-4 is slightly above plan's provisional
    # 1e-4 bound; pin the independent observation instead of snapping T.
    expected_historical = 8.92011964e-6 if az == 90 else 1.24164019e-4
    assert historical == pytest.approx(expected_historical, abs=1e-11)
    fd = parameter_fd(parameters(center), (0, 2), step=1e-6)
    independent_area = independent_sky([chain(parameters(center))])[0] @ np.cross(*fd)
    assert historical == pytest.approx(independent_area, abs=1e-10)
    root = brentq(area, center - 0.01, center + 0.01, xtol=1e-14)
    minimum = minimize_scalar(lambda r: -float(independent_sky([chain(parameters(r))])[0] @ SUN),
                              bounds=(center - 0.01, center + 0.01), method="bounded",
                              options={"xatol": 1e-13})
    assert abs(root - center) < np.radians(0.025)  # half the source grid cell
    assert minimum.success and abs(root - minimum.x) < 1e-5
    assert abs(area(root)) < 1e-12
    at_root = diagnose_sky(family_result(parameters(root), (0, 2)))
    assert at_root.mechanism[0] == "axis_stationary" and at_root.rank[0] == 1
    off = diagnose_sky(family_result(parameters(root + 0.05), (0, 2)))
    assert off.mechanism[0] == "regular" and abs(off.signed_area[0]) > 1e-3
    # A true turn in this known fixture, not inferred merely from a label.
    x0 = independent_sky([chain(parameters(root))])[0]
    sides = independent_sky([chain(parameters(root + d)) for d in (-1e-3, 1e-3)]) - x0
    assert np.dot(*sides) / np.prod(np.linalg.norm(sides, axis=1)) > 0.999


def test_lowitz_all_general_valleys_remain_regular_under_ad():
    candidates = lowitz_valleys()
    special = np.all(np.isclose(candidates, np.radians([270, 90, 0]), atol=1e-12), axis=1)
    result = family_result(candidates, (0, 1))
    diagnosis = diagnose_sky(result)
    assert result.available.all()
    assert (diagnosis.mechanism[~special] == "regular").all()
    assert diagnosis.mechanism[special].tolist() == ["parallel_collapse"]
    assert np.min(np.abs(diagnosis.signed_area[~special])) > 0.0018
    assert np.min(diagnosis.singular_values[~special, -1]) > 0.001
    fd = np.array([parameter_fd(p, (0, 1), step=1e-6) for p in candidates])
    np.testing.assert_allclose(result.dx, fd, atol=3e-9, rtol=0)


@pytest.mark.parametrize("height", [10, 15, 20, 30])
def test_lowitz_special_symmetry_null_and_parry_passage(height):
    sun = sun_direction(height)
    p = np.radians([270, 90, 0])
    result = family_result(p, (0, 1), sun)
    d = diagnose_sky(result)
    assert result.available.all()
    assert d.generator_rank[0] == 2 and d.rank[0] == 1
    assert d.mechanism[0] == "parallel_collapse"
    assert (d.speeds[0] > 0.1).all()
    assert abs(d.signed_area[0]) < 1e-12
    assert np.linalg.norm(np.cross(*result.dx[0])) < 1e-12
    assert d.singular_values[0, -1] < 1e-12
    # Independent premises of the mirror-plane proof: body c is the plane
    # normal, both ray segments, both angular-velocity axes are in that plane.
    r = chain(p)
    normal = r[:, 2]
    assert abs(normal @ sun) < 1e-15
    assert abs(normal @ result.sky[0]) < 1e-15
    world_axes = (r @ generators(p, (0, 1)).T).T
    np.testing.assert_allclose(world_axes @ normal, 0, atol=1e-15)
    # The optical side-face normals also lie in the same principal section.
    np.testing.assert_allclose((r @ np.array([CRYSTAL.normal(CRYSTAL.face(f)) for f in PATH]).T).T @ normal, 0, atol=1e-15)
    np.testing.assert_allclose(np.cross(result.dx[0], normal), 0, atol=1e-12)
    v = d.null_direction[0]
    assert np.linalg.norm(v @ result.dx[0]) < 1e-12
    independent_v = np.array([d.speeds[0, 1], -d.speeds[0, 0]])
    independent_v /= np.linalg.norm(independent_v)
    assert abs(v @ independent_v) > 1 - 1e-12
    if height == 15:
        assert abs(v[0]/v[1]) == pytest.approx(1.68, abs=0.01)
        np.testing.assert_allclose(d.speeds[0], [0.27710757, 0.46611503], atol=1e-8, rtol=0)
    # Real family curves, not the tangent's one-parameter subgroup. The null
    # is mixed az/zenith; its second-order displacement ratio is ~4.
    displacement, axial = [], []
    x0 = independent_sky([chain(p)], sun)[0]
    for step in np.radians([0.125, 0.25, 0.5]):
        mixed = p + step * np.array([v[0], v[1], 0])
        displacement.append(np.linalg.norm(independent_sky([chain(mixed)], sun)[0] - x0))
        axial.append(np.linalg.norm(independent_sky([chain(p + [step, 0, 0])], sun)[0] - x0))
    assert np.all((np.array(displacement[1:])/displacement[:-1] > 3.5) & (np.array(displacement[1:])/displacement[:-1] < 4.5))
    np.testing.assert_allclose(np.array(axial[1:])/axial[:-1], 2, atol=0.01, rtol=0)
    parry = family_result(p, (0,), sun)
    parry_d = diagnose_sky(parry)
    assert parry_d.mechanism[0] == "passing" and parry_d.signed_area is None
    assert parry_d.speeds[0, 0] == pytest.approx(d.speeds[0, 0], abs=1e-15)
    # Opposite sides of the Parry curve pass through, rather than turn back.
    sides = independent_sky([chain(p + [t, 0, 0]) for t in (-1e-3, 1e-3)], sun) - x0
    assert np.dot(*sides) / np.prod(np.linalg.norm(sides, axis=1)) < -0.999


def test_plate_s1_ordinary_point_and_fd():
    p = np.radians([210, 0, 0])
    result = family_result(p, (0,))
    d = diagnose_sky(result)
    assert result.available[0] and d.mechanism[0] == "passing"
    assert d.signed_area is None and d.speeds[0, 0] > 0.1
    np.testing.assert_allclose(result.dx[0], parameter_fd(p, (0,)), atol=2e-9, rtol=0)


def test_random_sphere_tangency_area_gram_and_parameter_transformations():
    rotations = haar_rotations(4096, np.random.default_rng(82))
    domain = path_domain_batch(rotations, PATH, -SUN)
    margin = np.min(list(domain.margins.values()), axis=0)
    rotations = rotations[domain.valid & (margin > 0.1)]
    assert len(rotations) >= 100
    omega = np.random.default_rng(29).normal(size=(len(rotations), 2, 3))
    result = sky_differential(rotations, omega, PATH, SUN)
    d = diagnose_sky(result)
    assert result.available.all()
    eps = np.finfo(float).eps
    np.testing.assert_allclose(np.linalg.norm(result.sky, axis=1), 1, atol=32*eps, rtol=0)
    tangent_residual = np.abs(np.einsum("ni,nki->nk", result.sky, result.dx))
    assert np.all(tangent_residual < 128*eps*np.maximum(1, d.speeds))
    error = np.abs(d.signed_area**2 - np.linalg.det(d.gram))
    assert np.all(error <= 512*eps*np.maximum(1, np.linalg.norm(d.gram, axis=(1,2))**2))
    swapped = diagnose_sky(sky_differential(rotations, omega[:, ::-1], PATH, SUN))
    np.testing.assert_allclose(swapped.signed_area, -d.signed_area, atol=2e-13, rtol=1e-13)
    scaled = diagnose_sky(sky_differential(rotations, omega * np.array([2., -3.])[None, :, None], PATH, SUN))
    np.testing.assert_allclose(scaled.signed_area, -6*d.signed_area, atol=2e-12, rtol=1e-13)
    assert (scaled.generator_rank == d.generator_rank).all()


def synthetic_differential(dx, omega=None):
    from lumice_integral.family_sky import SkyDifferential
    derivative = np.asarray(dx, dtype=float)[None]
    k = derivative.shape[1]
    generators_ = np.eye(3)[:k][None] if omega is None else np.asarray(omega, dtype=float)[None]
    return SkyDifferential(np.array([[0., 0., 1.]]), derivative, generators_, np.array([True]), np.array([True]), {})


@pytest.mark.parametrize("dx,label,rank", [
    ([[1,0,0], [0,1,0]], "regular", 2),
    ([[0,0,0], [0,1,0]], "axis_stationary", 1),
    ([[1,0,0], [2,0,0]], "parallel_collapse", 1),
    ([[0,0,0], [0,0,0]], "rank_zero", 0),
    ([[0,0,0]], "stationary", 0),
    ([[1,0,0]], "passing", 1),
])
def test_parameter_space_mechanisms(dx, label, rank):
    d = diagnose_sky(synthetic_differential(dx))
    assert d.mechanism[0] == label and d.rank[0] == rank
    np.testing.assert_allclose(d.gram[0], np.array(dx) @ np.array(dx).T, atol=1e-15)
    np.testing.assert_allclose(d.gram_eigenvalues[0], np.linalg.eigvalsh(d.gram[0]), atol=1e-15)
    if rank < len(dx):
        assert np.linalg.norm(d.null_direction[0] @ np.array(dx)) < 1e-14


@pytest.mark.parametrize("omega,rank", [([[0,0,0], [0,1,0]], 1), ([[1,0,0], [2,0,0]], 1),
                                         ([[1e-20,0,0], [0,1e20,0]], 2), ([[0,0,0], [0,0,0]], 0)])
def test_generator_rank_is_length_scaled_and_not_optical_compression(omega, rank):
    result = synthetic_differential([[1,0,0], [0,1,0]], omega)
    d = diagnose_sky(result)
    assert d.generator_rank[0] == rank
    assert d.mechanism[0] == ("regular" if rank == 2 else "degenerate_generators")


def test_explicit_tolerances_do_not_snap_area_or_derivative():
    result = synthetic_differential([[1,0,0], [0,1e-8,0]])
    strict, loose = diagnose_sky(result), diagnose_sky(result, atol=1e-6)
    assert strict.rank[0] == 2 and loose.rank[0] == 1
    assert strict.signed_area[0] == loose.signed_area[0] == 1e-8
    np.testing.assert_array_equal(result.dx[0], [[1,0,0], [0,1e-8,0]])
    # Direct SVD resolves a nearly parallel pair whose Gram spectrum suffers
    # cancellation. Do not obtain sigma_min by sqrt(lambda_min).
    pair = diagnose_sky(synthetic_differential([[1,0,0], [1,1e-9,0]]))
    assert pair.singular_values[0, -1] > 7e-10 and pair.rank[0] == 2
    for kwargs in ({"atol": -1}, {"rtol": np.nan}, {"generator_rtol": [1e-12]}):
        with pytest.raises(ValueError):
            diagnose_sky(result, **kwargs)
    with pytest.raises(ValueError, match="K=1 or K=2"):
        diagnose_sky(synthetic_differential(np.eye(3)))


def test_quadratic_diagnostic_overflow_is_unavailable():
    result = synthetic_differential([[1e200, 0, 0], [0, 1e200, 0]])
    diagnosis = diagnose_sky(result)
    assert diagnosis.mechanism[0] == "unavailable" and diagnosis.rank[0] == -1
    assert np.isnan(diagnosis.gram).all()


def test_triple_product_does_not_detect_general_r3_rank():
    x = np.array([1., 0., 0.])
    da, db = np.array([1., 0., 0.]), np.array([0., 1., 0.])
    assert x @ np.cross(da, db) == 0
    assert np.linalg.matrix_rank(np.stack([da, db], axis=1)) == 2
    assert np.linalg.det(np.array([da, db]) @ np.array([da, db]).T) == 1


def test_near_exit_critical_angle_has_explicit_domain_semantics():
    index = 1.31
    critical = np.arcsin(1/index)
    for offset in (1e-7, -1e-7):
        theta_inside = np.pi/3 - critical + offset
        incidence = np.arcsin(index*np.sin(theta_inside))
        sun = np.array([np.cos(incidence), -np.sin(incidence), 0.])
        result = sky_differential(np.eye(3)[None], np.eye(3)[[0, 2]][None], PATH, sun)
        assert result.valid[0] == (offset > 0)
        if offset > 0:
            assert 0 < result.margins["exit_snell_discriminant"][0] < 1e-6
            assert result.available[0] and np.linalg.norm(result.dx) > 100
        else:
            assert result.margins["exit_snell_discriminant"][0] < 0
            assert not result.available[0] and np.isnan(result.dx).all()
