"""Local numerical conformance for target fibers intersecting exact pose supports.

Only the literal regular-prism path (3, 5) is implemented.  Seeds are explicit,
angles are radians, and targets point *toward* the observed sky.  A projected u
or a deviation match never establishes a target hit.  Results contain ordinary
JSON values; neither root searches nor a closed traced component certify global
coverage.  This is not a Gaussian-family or a brightness/Dirac-mass classifier.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares, root

from . import optics, so3
from .camera import incident_direction_from_sun
from .continuation import (
    ContinuationOptions,
    local_residual_jacobian,
    target_residual,
    trace_fiber,
)
from .family_restricted import ExactFamily
from .geometry import HexPrism
from .weights import entry_measure_weight, fresnel_transmission_weight

C16_INDEX = 1.3110129170742788
_HIT_TOL = 1e-10


def _unit(value, name):
    value = np.asarray(value, dtype=np.float64)
    if value.shape != (3,) or not np.isfinite(value).all() or abs(value @ value - 1) > 1e-12:
        raise ValueError(f"{name} must be a finite unit vector")
    return value


class _Model:
    def __init__(self, family, sun, target, refractive_index, crystal):
        self.family = family
        self.sun, self.target = _unit(sun, "sun"), _unit(target, "target")
        self.index = float(refractive_index)
        if not np.isfinite(self.index) or self.index <= 1:
            raise ValueError("refractive_index must be finite and greater than one")
        # C16: unit reference diameter, Lumice height=1 -> a=0.5, h=1.
        self.crystal = HexPrism.from_lumice(height=1, a=0.5) if crystal is None else crystal
        self.incident = incident_direction_from_sun(self.sun)
        normals = optics.face_normals(self.crystal, (3, 5))
        self.sky = jax.jit(lambda q: -optics.trace_path(
            family.rotation(q), normals, jnp.asarray(self.incident), jnp.asarray(self.index)
        ).direction)
        self.sky_jacobian = jax.jit(jax.jacfwd(self.sky))

    def result(self, status, **extra):
        return dict(status=status, family=self.family.name, roll_lock=float(self.family.roll),
                    path=[3, 5], sun=self.sun.tolist(), target=self.target.tolist(),
                    refractive_index=self.index,
                    crystal=dict(a=float(self.crystal.a), h=float(self.crystal.h)),
                    coverage="local_numerical_not_certified", completeness="unknown", **extra)

    def supported(self):
        return isinstance(self.crystal, HexPrism) and np.array_equal(
            np.asarray(self.crystal.face_distance_ratios), np.ones(6))

    def canonical(self, q):
        q = np.asarray(q, dtype=np.float64).copy()
        if q.shape != (self.family.dimension,) or not np.isfinite(q).all():
            raise ValueError("each seed must be a finite family parameter vector")
        q[0] %= 2 * np.pi
        if self.family.name == "column":
            q[1] %= 2 * np.pi
        if self.family.name == "lowitz" and not 0 <= q[1] <= np.pi:
            raise ValueError("Lowitz zenith must be in [0, pi]; no pole-folded roll+pi sheet")
        return q

    def problem(self, rotation):
        return optics.path_problem(
            np.asarray(rotation, dtype=np.float64), (3, 5), self.incident,
            target_direction=-self.target, refractive_index=np.float64(self.index), crystal=self.crystal,
        )

    def point(self, q):
        q = self.canonical(q)
        rotation = np.asarray(self.family.rotation(jnp.asarray(q)))
        check = optics.path_domain(rotation, (3, 5), self.incident, self.index, crystal=self.crystal)
        if not check.valid:
            return dict(status="unsupported", reason="pose_outside_smooth_optical_domain", q=q.tolist())
        sky = np.asarray(self.sky(jnp.asarray(q)))
        error = float(np.linalg.norm(sky - self.target))
        if not np.isfinite(error):
            return dict(status="unsupported", reason="nonfinite_direction", q=q.tolist())
        a = entry_measure_weight(rotation, faces=(3, 5), incident_direction=self.incident,
                                 crystal=self.crystal, refractive_index=self.index)
        t = fresnel_transmission_weight(rotation, faces=(3, 5), incident_direction=self.incident,
                                        crystal=self.crystal, refractive_index=self.index)
        p = dict(q=q.tolist(), rotation=rotation.tolist(), u=(rotation.T @ self.sun).tolist(),
                 sky=sky.tolist(), direction_residual=error, A=a, T=t)
        if error > _HIT_TOL:
            return dict(**p, status="rejected", reason="full_sky_direction_mismatch")
        if not (np.isfinite(a) and np.isfinite(t) and a > 0 and t > 0):
            return dict(**p, status="unsupported", reason="no_positive_AT_witness")
        problem = self.problem(rotation)
        jac = np.asarray(local_residual_jacobian(problem, jnp.asarray(rotation)))
        singular = np.linalg.svd(jac, compute_uv=False)
        p.update(so3_singular_values=singular.tolist(), so3_rank=int(np.sum(singular > 1e-8)),
                 chart_residual=float(np.linalg.norm(target_residual(problem, jnp.asarray(rotation)))))
        if p["so3_rank"] != 2:
            return dict(**p, status="unsupported", reason="singular_full_SO3_direction_map")
        return dict(**p, status="computed")


def check_target_pose(family: ExactFamily, q, sun, target, *, refractive_index=C16_INDEX, crystal=None):
    """Replay one exact-family pose; reject a projected/deviation-only false hit."""
    model = _Model(family, sun, target, refractive_index, crystal)
    if not model.supported():
        return model.result("unsupported", reason="only_regular_hexprism_3_5_implemented")
    point = model.point(q)
    return model.result(point["status"], witness=point)


def check_target_poses(family: ExactFamily, qs, sun, target, *, refractive_index=C16_INDEX, crystal=None):
    """Replay many witnesses with one model/JIT identity, preserving per-pose evidence."""
    model = _Model(family, sun, target, refractive_index, crystal)
    if not model.supported():
        return model.result("unsupported", reason="only_regular_hexprism_3_5_implemented", witnesses=[])
    points = [model.point(q) for q in qs]
    return model.result("computed", witnesses=points)


def _constraint(family, rotation):
    if family.name == "column":
        return rotation[2, 2]
    # Lowitz's roll lock; the canonical zenith sheet is checked separately.
    return jnp.sin(family.roll) * rotation[2, 0] + jnp.cos(family.roll) * rotation[2, 1]


def _contact(model, point):
    """Differentiate the family constraint ALONG the full SO3 target fiber.

    x(t)=t*v+1/2*t^2*a in R exp(x); J*a=-r''(v,v), v.a=0.
    Thus g'' includes the fiber's curvature, not just a straight Lie step.
    """
    q = np.asarray(point["q"])
    if model.family.dimension == 1:
        speed = float(np.linalg.norm(model.sky_jacobian(jnp.asarray(q))))
        return dict(intersection="isolated", family_sky_speed=speed) if speed > 1e-8 else None
    if model.family.name == "lowitz" and min(q[1], np.pi - q[1]) < 1e-7:
        return None  # endpoint circles need a separate stratum treatment
    rotation = jnp.asarray(point["rotation"])
    problem = model.problem(rotation)
    residual = jax.jit(lambda x: target_residual(problem, rotation @ so3.exp(x)))
    g = jax.jit(lambda x: _constraint(model.family, rotation @ so3.exp(x)))
    zero = jnp.zeros(3)
    jac = np.asarray(local_residual_jacobian(problem, rotation))
    v = np.linalg.svd(jac, full_matrices=True)[2][-1]
    # A null LINE has arbitrary LAPACK sign. Pin its largest component for
    # replayable signed jets without changing the underlying tangent line.
    if v[int(np.argmax(abs(v)))] < 0:
        v = -v
    vj = jnp.asarray(v)
    def second(f):
        return np.asarray(jax.jvp(lambda x: jax.jvp(f, (x,), (vj,))[1], (zero,), (vj,))[1])
    acceleration = np.linalg.lstsq(jac, -second(residual), rcond=None)[0]
    dg = np.asarray(jax.grad(g)(zero))
    slope = float(dg @ v)
    curvature = float(second(g) + dg @ acceleration)
    out = dict(constraint="Rzz" if model.family.name == "column" else "roll_lock",
               constraint_residual=abs(float(g(zero))), fiber_tangent=v.tolist(),
               first_derivative=slope, second_derivative=curvature)
    if abs(slope) > 1e-7:
        # A quartic least-squares objective can stop just off a fold. Its small
        # direction residual alone is not a transverse-root certificate: demand
        # residual error small compared with the local distance-to-fold scale.
        if point["direction_residual"] > slope**2 / (100 * max(1., abs(curvature))):
            return None
        return dict(**out, intersection="transverse")
    if abs(slope) > 1e-9 or abs(curvature) < 1e-6:
        return None
    # Independent corrected offsets: target residual and Lie phase, not u.
    equation = jax.jit(lambda x, h: jnp.concatenate((residual(x), jnp.array([vj @ x - h]))))
    derivative = jax.jit(jax.jacfwd(equation, argnums=0))
    offsets = []
    for h in (-0.01, -0.005, 0.005, 0.01):
        solved = root(lambda x: np.asarray(equation(x, h)), h * v,
                      jac=lambda x: np.asarray(derivative(x, h)), tol=1e-11)
        if np.linalg.norm(equation(solved.x, h)) > 1e-10:
            return None
        r = np.asarray(rotation @ so3.exp(jnp.asarray(solved.x)))
        domain = optics.path_domain(r, (3, 5), model.incident, model.index, crystal=model.crystal)
        if not domain.valid:
            return None
        offsets.append(dict(phase=h, constraint=float(g(solved.x)),
                            direction_residual=float(np.linalg.norm(problem.direction_evaluator(r) + model.target))))
    if any(p["constraint"] * curvature <= 0 for p in offsets):
        return None
    return dict(**out, intersection="tangent", corrected_offsets=offsets)


def locate_elevation_witness(family: ExactFamily, sun, elevation_deg: float, seed, *,
                             refractive_index=C16_INDEX, crystal=None):
    """Recompute a column fold or a regular Parry point on a chosen elevation.

    Column solves sky_z=sin(elevation) AND det(dsky/dq0,dsky/dq1,sky)=0.
    Parry solves sky_z=sin(elevation).  This does not snap a supplied target
    onto the image: the returned full target must be used explicitly by callers.
    Only a local root reached from the supplied seed is claimed.
    """
    if not np.isfinite(elevation_deg) or not -90 < elevation_deg < 90:
        raise ValueError("elevation must be finite and strictly between -90 and 90 degrees")
    model = _Model(family, sun, [1., 0., 0.], refractive_index, crystal)
    if not model.supported() or family.name not in ("column", "parry"):
        return model.result("unsupported", reason="locator_requires_regular_3_5_column_or_parry")
    seed = model.canonical(seed)
    height = np.sin(np.deg2rad(elevation_deg))
    def equations(q):
        sky = model.sky(q)
        if family.name == "parry":
            return jnp.array([sky[2] - height])
        return jnp.array([sky[2] - height,
                          jnp.linalg.det(jnp.column_stack((model.sky_jacobian(q), sky)))])
    equation, derivative = jax.jit(equations), jax.jit(jax.jacfwd(equations))
    solved = root(lambda q: np.asarray(equation(q)), seed,
                  jac=lambda q: np.asarray(derivative(q)), tol=1e-11)
    error = np.linalg.norm(equation(solved.x))
    if not np.isfinite(error) or error > 1e-11:
        return model.result("unsupported", reason="elevation_locator_failed")
    q = model.canonical(solved.x)
    model.target = _unit(np.asarray(model.sky(jnp.asarray(q))), "computed sky")
    if abs(np.rad2deg(np.arcsin(model.target[2])) - elevation_deg) > 1e-8:
        return model.result("unsupported", reason="wrong_elevation_branch")
    point = model.point(q)
    contact = _contact(model, point) if point["status"] == "computed" else None
    expected = "tangent" if family.name == "column" else "isolated"
    if contact is None or contact["intersection"] != expected:
        return model.result("unsupported", reason="locator_contact_unresolved", witness=point)
    point.update(contact)
    return model.result("computed", target_elevation_deg=float(elevation_deg),
                        target_azimuth_deg=float(np.rad2deg(np.arctan2(model.target[1], model.target[0]))),
                        records=[point], curves=[])


def _solve_seed(model, seed, maximum_evaluations):
    bounds = ([-np.inf, 0.], [np.inf, np.pi]) if model.family.name == "lowitz" else (-np.inf, np.inf)
    solved = least_squares(lambda q: np.asarray(model.sky(q)) - model.target, seed,
                           jac=lambda q: np.asarray(model.sky_jacobian(q)), bounds=bounds,
                           max_nfev=maximum_evaluations, ftol=1e-14, xtol=1e-14, gtol=1e-14)
    return model.point(solved.x)


def _coincidence(model, seeds, maximum_steps, maximum_evaluations):
    """Side-face tangential Snell conservation proves fiber containment.

    (target-sun).R ez=0. For a nonzero vertical difference this forces Rzz=0
    on EVERY physical pose of the target fiber. The trace only witnesses one
    connected component and its positive AT domain; it does not prove exhaustion.
    """
    for seed in seeds:
        point = model.point(seed)
        if point["status"] != "computed":
            try:
                point = _solve_seed(model, seed, maximum_evaluations)
            except ValueError:
                continue
        if point["status"] == "computed":
            break
    else:
        return model.result("unsupported", reason="coincidence_requires_a_target_pose_seed", records=[], curves=[])
    problem = model.problem(point["rotation"])
    traced = trace_fiber(problem, ContinuationOptions(maximum_accepted_steps=maximum_steps))
    if str(traced.reason) != "closed_loop":
        return model.result("unsupported", reason="coincident_component_not_closed",
                            trace_reason=str(traced.reason), records=[], curves=[])
    points = []
    for rotation in np.asarray(traced.poses):
        # Rzz=0 fixes zen=pi/2; the two remaining ZYZ angles are unambiguous.
        az = (np.arctan2(-rotation[1, 2], -rotation[0, 2]) + np.pi) % (2 * np.pi)
        roll = np.arctan2(-rotation[2, 1], rotation[2, 0]) % (2 * np.pi)
        p = model.point([az, roll])
        if p["status"] != "computed" or np.linalg.norm(np.asarray(p["rotation"]) - rotation) > 1e-8:
            return model.result("unsupported", reason="coincident_pose_replay_failed", records=[], curves=[])
        p["constraint_residual"] = abs(float(rotation[2, 2]))
        points.append(p)
    return model.result("computed", records=[], curves=[dict(
        intersection="coincident", dimension=1, closed=True, points=points,
        containment="side_face_Snell_conservation_and_vertical_target_minus_sun",
        trace_reason=str(traced.reason), component_completeness=traced.component_completeness,
    )])


def intersect_target(family: ExactFamily, sun, target, seeds, *, refractive_index=C16_INDEX,
                     crystal=None, maximum_evaluations=100, maximum_trace_steps=1000):
    """Find local exact-family hits from explicit seeds, never report certified empty.

    Two-parameter families use bounded (Lowitz) or periodic (column) numerical
    least squares followed by full-direction/positive-AT checks and SO3 jets.
    Parry/plate give isolated pose witnesses, not minimum-distance certificates.
    Mirrored-sun column targets use the conservation identity plus one closed
    SO3 continuation. Failed/unresolved seeds remain explicit, even if others hit.
    """
    model = _Model(family, sun, target, refractive_index, crystal)
    if not model.supported():
        return model.result("unsupported", reason="only_regular_hexprism_3_5_implemented", records=[], curves=[])
    if maximum_evaluations < 1 or maximum_trace_steps < 1:
        raise ValueError("solver budgets must be positive")
    seeds = [model.canonical(q) for q in seeds]
    if not seeds:
        return model.result("unsupported", reason="explicit_seeds_required", records=[], curves=[])
    delta = model.target - model.sun
    if family.name == "column" and np.linalg.norm(delta[:2]) < 1e-12 and abs(delta[2]) > 1e-10:
        return _coincidence(model, seeds, maximum_trace_steps, maximum_evaluations)
    records, unresolved = [], []
    for seed in seeds:
        try:
            point = _solve_seed(model, seed, maximum_evaluations)
        except ValueError:
            unresolved.append(dict(seed=seed.tolist(), reason="nonfinite_or_failed_local_solve"))
            continue
        if point["status"] != "computed":
            unresolved.append(dict(seed=seed.tolist(), reason=point["reason"]))
            continue
        contact = _contact(model, point)
        if contact is None:
            unresolved.append(dict(seed=seed.tolist(), reason="degenerate_or_endpoint_contact_unresolved"))
            continue
        point.update(contact)
        if not any(np.linalg.norm(np.asarray(point["rotation"]) - p["rotation"]) < 1e-7 for p in records):
            records.append(point)
    return model.result("computed" if records else "unsupported", records=records, curves=[],
                        unresolved_seeds=unresolved,
                        **({} if records else dict(reason="no_verified_local_intersection")))
