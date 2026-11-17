"""Exact pose supports and restricted structures, independent of Gaussian densities.

Angles and derivatives are in radians.  R maps body to world; u=R.T@sun,
nominal sky=-R@Phi(-u).  Projection never supplies a pose witness by itself.
Coverage is explicit: numerical root seeds and existing kink marches do not
certify global exhaustion.  Unsupported/nonisolated numerical regimes raise
RestrictedEscape rather than being reported as empty intersections.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import brentq

from . import optics
from .dp_field import DPField, TopologyEscape
from .dp_field.field import d_value, margin_vector
from .dp_field.boundary import incidence_normals
from .so3 import vee
from .symmetry.reflection_group import commutes_with_rotation_about

TAU = 2 * np.pi
ATOL = 1e-9


class RestrictedEscape(TopologyEscape):
    """A named, replayable unsupported regime (not an empty computed set)."""

    def __init__(self, regime: str, message: str):
        self.regime = regime
        super().__init__(f"{regime}: {message}")


def _rz(t):
    c, s = jnp.cos(t), jnp.sin(t)
    return jnp.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def _ry(t):
    c, s = jnp.cos(t), jnp.sin(t)
    return jnp.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])


@dataclass(frozen=True)
class ExactFamily:
    """Canonical exact support, not the zero-width interpretation of a density.

    q order: plate/Parry (azimuth,), column (azimuth,roll), Lowitz
    (azimuth,zenith). All circular angles are [0,2pi); Lowitz zenith is
    [0,pi], including both endpoint circles. Negative-zenith coordinates
    require the ZYZ pole rewrite (az+pi,-zenith,roll+pi); those roll+pi
    sheets are NOT silently added to the explicitly roll-locked support.
    """

    name: Literal["plate", "column", "parry", "lowitz"]
    roll: float = 0.

    def __post_init__(self):
        if self.name not in ("plate", "column", "parry", "lowitz") or not np.isfinite(self.roll):
            raise ValueError("exact family must be plate/column/parry/lowitz with finite roll")
        if self.name in ("plate", "column") and self.roll != 0:
            raise ValueError("roll lock only applies to Parry/Lowitz")

    @property
    def dimension(self):
        return 2 if self.name in ("column", "lowitz") else 1

    @partial(jax.jit, static_argnums=0)
    def rotation(self, q):
        az = q[0]
        zen = q[1] if self.name == "lowitz" else (0. if self.name == "plate" else np.pi / 2)
        roll = q[1] if self.name == "column" else self.roll
        return _rz(az - np.pi) @ _ry(-zen) @ _rz(roll)

    def u(self, q, sun):
        return self.rotation(q).T @ sun

    @partial(jax.jit, static_argnums=0)
    def generators(self, q):
        """Right trivialized dR/dq_i=R[Omega_i]x, order matching q."""
        r = self.rotation(q)
        dr = jax.jacfwd(self.rotation)(jnp.asarray(q))
        return jnp.stack([vee(r.T @ dr[:, :, i]) for i in range(self.dimension)])

    def circle(self, sun):
        """Plane a.u=h of a one-dimensional image; refuses point images."""
        if self.dimension != 1:
            raise RestrictedEscape("not_circle", "two-parameter supports have a two-dimensional image")
        h = float(sun[2])
        if abs(h) >= 1 - 1e-12:
            raise RestrictedEscape("projection_rank_zero", "sun on the vertical axis collapses the u circle")
        a = np.asarray(self.rotation(jnp.zeros(1))).T @ np.array([0., 0., 1.])
        return a, h

    def lifts(self, u, sun):
        """All canonical poses above u (including both column/Lowitz sheets).

        Solves z_world.Ru=sun_z analytically, then azimuth from the horizontal
        vectors. At rank-loss sheet boundaries deduplicates poses, not just u.
        A continuous inverse image is refused, never replaced by one pose.
        """
        u, sun = np.asarray(u, float), np.asarray(sun, float)
        if abs(np.linalg.norm(u) - 1) > 1e-9 or abs(np.linalg.norm(sun) - 1) > 1e-12:
            raise ValueError("u and sun must be unit vectors")
        if np.hypot(*sun[:2]) < 1e-10:
            raise RestrictedEscape("sun_pole_lifts", "azimuth preimage is continuous")
        h = sun[2]
        candidates = []
        if self.name == "column":
            radius = np.hypot(*u[:2])
            if radius < 1e-12:
                if abs(h) < ATOL:
                    raise RestrictedEscape("continuous_lifts", "column projection has a circle preimage")
                return ()
            if abs(h) > radius + 1e-12:
                return ()
            phi = np.arctan2(u[1], u[0])
            cosine = np.clip(h / radius, -1, 1)
            if abs(abs(cosine) - 1) < 1e-14:
                cosine = np.sign(cosine)
            theta = np.arccos(cosine)
            candidates = [(np.pi / 2, (sign * theta - phi) % TAU) for sign in (-1, 1)]
        elif self.name == "lowitz":
            v = np.asarray(_rz(self.roll)) @ u
            radius = np.hypot(v[0], v[2])
            if radius < 1e-12:
                if abs(h) < ATOL:
                    raise RestrictedEscape("continuous_lifts", "Lowitz projection has a continuous preimage")
                return ()
            if abs(h) > radius + 1e-12:
                return ()
            phase = np.arctan2(v[0], v[2])
            cosine = np.clip(h / radius, -1, 1)
            if abs(abs(cosine) - 1) < 1e-14:
                cosine = np.sign(cosine)
            angle = np.arccos(cosine)
            for sign in (-1, 1):
                for turn in (-1, 0, 1):
                    z = phase + sign * angle + turn * TAU
                    if -1e-12 <= z <= np.pi + 1e-12:
                        candidates.append((float(np.clip(z, 0, np.pi)), self.roll))
        else:
            a, height = self.circle(sun)
            if abs(a @ u - height) > ATOL:
                return ()
            candidates = [(0. if self.name == "plate" else np.pi / 2, self.roll)]
        out = []
        for z, roll in candidates:
            v = np.asarray(_ry(-z) @ _rz(roll)) @ u
            az = (np.arctan2(sun[1], sun[0]) - np.arctan2(v[1], v[0]) + np.pi) % TAU
            q = np.array([az, roll] if self.name == "column" else [az, z] if self.name == "lowitz" else [az])
            r = np.asarray(self.rotation(q))
            if np.linalg.norm(r @ u - sun) > 5e-9:
                raise RestrictedEscape("lift_residual", "analytic inverse failed its forward check")
            if not any(np.linalg.norm(r - np.asarray(self.rotation(p))) < 1e-8 for p in out):
                out.append(q)
        return tuple(out)


def _sun(sun):
    sun = np.asarray(sun, float)
    if sun.shape != (3,) or not np.isfinite(sun).all() or abs(sun @ sun - 1) > 1e-12:
        raise ValueError("sun must be a finite unit vector")
    return sun


def _point(field, family, sun, q):
    r = np.asarray(family.rotation(q))
    u = r.T @ sun
    margins = field.validity_margins_batch(u[None])[0]
    evaluation = optics.trace_path(jnp.asarray(r), field.normals, jnp.asarray(-sun), jnp.asarray(field.index))
    sky = -np.asarray(evaluation.direction)
    # Exit closure: recompute the dropped-root limit, never epsilon-inpush.
    routed = 0
    if field.slab is not None:
        sky = r @ field.slab @ u
        routed = int(not np.isfinite(np.asarray(evaluation.direction)).all())
    elif (not np.isfinite(sky).all() or abs(float(evaluation.exit.discriminant)) < 1e-12) and min(margins) >= -ATOL:
        d = evaluation.internal[-1].direction if evaluation.internal else evaluation.entry.direction
        sky = -np.asarray(field.index * d - field.index * evaluation.exit.incidence_cosine * (r @ field.normals[-1]))
        sky /= np.linalg.norm(sky)
        routed = 1
    if not np.isfinite(sky).all():
        raise RestrictedEscape("nonfinite_sky", "point is outside the defined optical closure")
    return dict(q=np.asarray(q), rotation=r, u=u, sky=sky,
                tangent=np.asarray(jax.jacfwd(lambda p: family.u(p, jnp.asarray(sun)))(jnp.asarray(q))),
                value=float(np.arctan2(np.linalg.norm(np.cross(sky, sun)), sky @ sun)),
                min_gate=float(min(margins)), routed_nonfinite=routed)


def _roots_periodic(f, df, *, samples=360, budget=4096):
    """Numerical roots with derivative brackets for even contacts; not a certificate.

    Dense almost-zero data are not evidence of coincidence. Caller must supply
    an analytic identity for that case, otherwise this regime is refused.
    """
    if samples < 16 or samples * 2 > budget:
        raise RestrictedEscape("root_budget", "insufficient root enumeration budget")
    grid = np.linspace(0, TAU, samples + 1)
    values = np.asarray(jax.jit(jax.vmap(f))(jnp.asarray(grid)))
    slopes = np.asarray(jax.jit(jax.vmap(df))(jnp.asarray(grid)))
    finite = np.isfinite(values)
    if not finite.any():
        raise RestrictedEscape("nonfinite_root_scan", "no finite scalar samples; not a computed empty set")
    if finite.all() and max(abs(values)) < 1e-10:
        raise RestrictedEscape("unproved_coincidence", "near-zero scan is not an identity proof")
    if finite.all() and np.ptp(values) < 1e-12 and min(abs(values)) > 1e-8:
        return []  # numerical nonzero constant; still not an exhaustion certificate
    evaluations = 2 * (samples + 1)
    def bounded(function, t):
        nonlocal evaluations
        evaluations += 1
        if evaluations > budget:
            raise RestrictedEscape("root_budget", "root refinement exhausted its evaluation budget")
        return float(function(t))
    stationary = []
    roots = []
    def add(t):
        t = float(t % TAU)
        if min(t, TAU - t) < 1e-9:
            t = 0.
        if not any(abs((t - p + np.pi) % TAU - np.pi) < 1e-7 for p in roots):
            roots.append(t)
    for i in range(samples):
        a, b = grid[i:i+2]
        if finite[i] and abs(values[i]) < 1e-12:
            add(a)
        if finite[i:i+2].all() and values[i] * values[i+1] < 0:
            left, right = bounded(f, a), bounded(f, b)
            if left * right < 0:
                add(brentq(lambda t: bounded(f, t), a, b, xtol=1e-13))
        if np.isfinite(slopes[i:i+2]).all() and slopes[i] * slopes[i+1] < 0:
            left, right = bounded(df, a), bounded(df, b)
            if left * right < 0:
                stationary.append(brentq(lambda t: bounded(df, t), a, b, xtol=1e-13))
    for t in stationary:
        if np.isfinite(float(f(t))) and abs(float(f(t))) < 1e-10:
            add(t)
    return sorted(roots)


def _circle_roots(a, h, b, k):
    """Analytic plane-circle intersections on the unit sphere; None=coincident."""
    dot = float(a @ b)
    direction = np.cross(a, b)
    sine = float(np.linalg.norm(direction))
    identity_atol = 8*np.finfo(float).eps
    if sine <= identity_atol:
        return None if abs(k - dot * h) <= identity_atol*max(1., abs(h), abs(k)) else []
    # Gram inversion loses sin(theta)^2 near parallel planes. Build the
    # orthogonal basis from the cross product instead: b=dot*a+sine*e2.
    direction /= sine
    perpendicular = np.cross(direction, a)
    centre = h*a + ((k-dot*h)/sine)*perpendicular
    square = 1 - centre @ centre
    # Only roundoff-scale residuals collapse to one contact. A small but
    # resolved positive radicand is TWO roots, not a tolerance-merged tangent.
    roundoff = 8*np.finfo(float).eps*max(1., float(centre @ centre))
    if square < -roundoff:
        return []
    if abs(square) <= roundoff:
        return [(centre / np.linalg.norm(centre), "tangent")]
    return [(centre + sign * np.sqrt(square) * direction, "transverse") for sign in (-1, 1)]


def restricted_critical(field: DPField, family: ExactFamily, sun, *, samples=360, budget=4096):
    """Critical set of D on the regular image strata, with explicit omitted strata.

    S1 profiles refer to smooth nonzero density on the image: 1D Morse extrema
    push forward as |D-D0|^-1/2, not the 2D finite jump. A constant support
    has a delta mass; its location set has no finite multiplicity.
    """
    sun = _sun(sun)
    if family.dimension == 2:
        if field.slab is not None:
            raise RestrictedEscape("nonisolated_2d_critical", "slab critical strata need a dedicated two-dimensional restriction")
        records = []
        for critical in field.interior_critical_points:
            if critical.kind == "degenerate":
                raise RestrictedEscape("degenerate_jet", "ambient critical stratum is non-Morse")
            lifts = family.lifts(critical.position, sun)
            for q in lifts:
                p = _point(field, family, sun, q)
                rank = np.linalg.matrix_rank(p["tangent"], tol=1e-8)
                if rank < 2:
                    raise RestrictedEscape("projection_rank_loss", "critical point meets an unclassified image stratum")
                records.append(dict(**p, location="interior", source="restricted_" + critical.kind,
                                    profile="log_divergence" if critical.kind == "saddle" else "finite_jump",
                                    multiplicity=1, derivative_residual=float(np.linalg.norm(field.gradient_batch(p["u"][None])[0]))))
        return dict(records=records, coverage="regular_image_strata_only", omitted=["projection_rank_boundary", "parameter_endpoints"])
    a, _ = family.circle(sun)
    normals = field.normals  # resolve cached properties before entering JAX tracing
    slab = field.slab
    f = jax.jit(lambda t: d_value(family.u(jnp.array([t]), jnp.asarray(sun)), field.faces,
                                jnp.asarray(field.index), slab, normals))
    df = jax.jit(jax.grad(f))
    ddf = jax.jit(jax.grad(df))
    endpoints, intervals = _circle_domain(field, family, sun, samples, budget)
    if field.slab is not None and commutes_with_rotation_about(field.slab, a):
        # Algebraic rotational invariance, not a sampled near-constant inference.
        if not intervals:
            return dict(records=[], endpoints=endpoints, coverage="analytic_invariance_numerical_domain", omitted=[])
        t = float(sum(intervals[0])/2)
        return dict(records=[dict(**_point(field, family, sun, [t]), location="constant_set", source="restricted_constant",
                                  profile="delta_mass", multiplicity=None, derivative_residual=0.,
                                  parameter_domain=[0., TAU], parameter_intervals=intervals, domain_clipped=True)],
                    endpoints=endpoints, coverage="analytic_invariance_numerical_domain", omitted=[])
    roots = _roots_periodic(df, ddf, samples=samples, budget=budget)
    records = []
    for t in roots:
        u = np.asarray(family.u(jnp.array([t]), sun))
        if not field.valid_batch(u[None])[0]:
            continue
        second = float(ddf(t))
        if not np.isfinite(second) or abs(second) < 1e-7:
            raise RestrictedEscape("degenerate_jet", "restricted Hessian does not certify a Morse extremum")
        p = _point(field, family, sun, [t])
        records.append(dict(**p, location="interior", source="restricted_minimum" if second > 0 else "restricted_maximum",
                            profile="inverse_sqrt", multiplicity=1, second_derivative=second,
                            derivative_residual=abs(float(df(t)))))
    return dict(records=records, endpoints=endpoints, coverage="numerical_roots_not_certified", omitted=[])


def _circle_domain(field, family, sun, samples, budget):
    """Numerically partition the optical circle; retain singular endpoint jets as unavailable."""
    endpoints = []
    for margin in optics.validity_margin_names(field.faces):
        result = intersect_margin(field, family, sun, margin, samples=samples, budget=budget)
        if result["curves"]:
            raise RestrictedEscape("support_on_gate", "the family circle lies on a validity gate")
        for p in result["records"]:
            existing = next((e for e in endpoints if np.linalg.norm(e["rotation"]-p["rotation"]) < 1e-7), None)
            if existing is not None:
                existing["margins"].append(margin)
                continue
            endpoints.append(dict(**p, location="endpoint", source="restricted_domain_endpoint",
                                  profile=None, profile_unavailable="boundary_jet_not_assumed_smooth",
                                  multiplicity=1, derivative_residual=None,
                                  derivative_unavailable="closed_domain_limit", margins=[margin]))
    cuts = sorted({0., TAU, *(float(p["q"][0]) for p in endpoints)})
    intervals = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        u = np.asarray(family.u(jnp.array([(a+b)/2]), sun))
        if field.valid_batch(u[None])[0]:
            intervals.append([a, b])
    return endpoints, intervals


def _margin_plane(field, margin):
    incidence = incidence_normals(field.crystal, field.faces)
    entry = incidence["entry_incidence_cosine"]
    if margin == "entry_incidence_cosine" or (margin == "exit_snell_discriminant" and field.slab is not None):
        # For a slab, refractions cancel and exit_disc=(entry_cosine)^2.
        # Its ZERO SET is the entry great circle, not a newly tangent curve.
        return entry / np.linalg.norm(entry), 0.
    if margin.endswith("tir_discriminant") and margin.startswith("internal_"):
        m = incidence[margin.replace("tir_discriminant", "incidence_cosine")]
        if abs(m @ entry) < 1e-12 and 1 < field.index < np.sqrt(2):
            return m / np.linalg.norm(m), -np.sqrt(field.index**2 - 1)
    if margin.endswith("incidence_cosine") and margin in incidence:
        m = incidence[margin]
        if abs(m @ entry) < 1e-12:
            return m / np.linalg.norm(m), 0.
    return None


def intersect_margin(field: DPField, family: ExactFamily, sun, margin: str, *, samples=360, budget=4096):
    """kind-2/3 zero set intersected with the exact u image, keeping every lift.

    Circle intersections (including tangencies/coincidences) are analytic.
    Other 1D intersections carry a numerical, non-exhaustive coverage claim.
    For 2D supports curve points come from the existing boundary/kink authority.
    """
    sun = _sun(sun)
    names = optics.domain_margin_names(field.faces)
    if margin not in names:
        raise ValueError(f"unknown margin {margin}")
    kind = "kind_2" if margin in optics.validity_margin_names(field.faces) else "kind_3"
    if kind == "kind_3" and not (margin.startswith("internal_") and margin.endswith("tir_discriminant")):
        raise ValueError("only validity gates and internal TIR onsets are structural margins")
    k = names.index(margin)
    records = []
    plane = _margin_plane(field, margin)
    coverage = "analytic_circle_intersection"
    if family.dimension == 1:
        a, h = family.circle(sun)
        roots = _circle_roots(a, h, *plane) if plane is not None else ...
        if roots is None:
            # Analytic identity of planes proves a continuous coincidence. Its
            # clipped arcs remain arcs, never hundreds of isolated roots.
            return _lift_margin_arcs(field, family, sun, margin, kind, samples,
                                     "analytic_coincidence_upstream_domain", "coincident")
        elif roots is ...:
            normals = field.normals
            f = jax.jit(lambda t: margin_vector(family.u(jnp.array([t]), jnp.asarray(sun)), field.faces,
                                               jnp.asarray(field.index), normals)[k])
            df = jax.jit(jax.grad(f))
            ts = _roots_periodic(f, df, samples=samples, budget=budget)
            ddf = jax.jit(jax.grad(df))
            candidates = []
            for t in ts:
                candidate_u = np.asarray(family.u(jnp.array([t]), sun))
                if min(field.validity_margins_batch(candidate_u[None])[0]) < -ATOL:
                    continue
                slope, second, residual = float(df(t)), float(ddf(t)), abs(float(f(t)))
                if abs(slope) > 1e-7 and residual < min(1e-10, slope*slope/(100*max(1., abs(second)))):
                    state = "transverse"
                elif abs(slope) < 1e-9 and abs(second) > 1e-7 and residual < 1e-12:
                    sides = [float(f(t+sign*1e-3)) for sign in (-1,1)]
                    if not all(v*second > 0 for v in sides):
                        raise RestrictedEscape("unresolved_contact", "two-sided contact evidence disagrees with its jet")
                    state = "tangent"
                else:
                    raise RestrictedEscape("unresolved_contact", "nearby roots are not a certified tangent")
                candidates.append((np.asarray(family.u(jnp.array([t]), sun)), state, [np.array([t])]))
            coverage = "numerical_roots_not_certified"
        else:
            candidates = [(u, state, family.lifts(u, sun)) for u, state in roots]
        for u, state, lifts in candidates:
            if min(field.validity_margins_batch(u[None])[0]) < -ATOL:
                continue
            # A kink also needs the physical reflecting (positive incidence) branch.
            if kind == "kind_3":
                inc = margin.replace("tir_discriminant", "incidence_cosine")
                if field.margins_batch(u[None])[0, names.index(inc)] < -ATOL:
                    continue
            for q in lifts:
                records.append(dict(**_point(field, family, sun, q), intersection=state,
                                    residual=abs(float(field.margins_batch(u[None])[0, k]))))
        return dict(kind=kind, margin=margin, records=records, curves=[], coverage=coverage)
    return _lift_margin_arcs(field, family, sun, margin, kind, samples,
                             "upstream_curve_sampling_not_certified", "curve")


def _lift_margin_arcs(field, family, sun, margin, kind, samples, coverage, state):
    names = optics.domain_margin_names(field.faces)
    k = names.index(margin)
    if kind == "kind_3":
        curve = next(c for c in field.weight_kinks if c.margin == margin)
        if curve.failed_seeds:
            raise RestrictedEscape("kink_walk_failed", curve.note)
        arcs = [(arc.points, arc.closed, arc.ends) for arc in curve.arcs]
        coverage = "upstream_kink_sampling_not_certified"
    else:
        arcs = [(piece.points, False, (None, None)) for piece in field.boundary_curves if piece.margin == margin]
        coverage = "upstream_boundary_sampling"
    curves = []
    plane = _margin_plane(field, margin)
    for points, closed, ends in arcs:
        if plane is None:
            # No analytic clipping model for a general marched curve. A chart
            # transition must not be silently spliced into a fake sky branch.
            raise RestrictedEscape("noncircular_family_clip", "no certified chart cuts for a marched margin")
        normal, height = plane
        radius = np.sqrt(1 - height**2)
        e1 = np.cross(normal, np.eye(3)[np.argmin(abs(normal))])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(normal, e1)
        ts = np.unwrap(np.arctan2(points @ e2, points @ e1))
        if ts[-1] < ts[0]:
            ts = -ts
            e2 = -e2
        lo, hi = float(ts[0]), float(ts[-1])
        cuts = [lo, hi]
        if family.dimension == 2:
            # Boundaries of the analytic inverse: column folds |u_z|=cos h;
            # Lowitz folds |(Rz(roll)u)_y|=cos h and zenith endpoints u_z=+-h.
            ch = np.hypot(*sun[:2])
            constraints = [(np.array([0.,0.,1.]), ch), (np.array([0.,0.,1.]), -ch)]
            if family.name == "lowitz":
                b = np.asarray(_rz(family.roll)).T @ np.array([0.,1.,0.])
                constraints = [(b, ch), (b, -ch), (np.array([0.,0.,1.]), sun[2]),
                               (np.array([0.,0.,1.]), -sun[2])]
            for b, k0 in constraints:
                hits = _circle_roots(normal, height, b, k0)
                if hits is None:
                    raise RestrictedEscape("rank_boundary_coincidence", "margin lies on an image-chart boundary")
                for u, _ in hits:
                    angle = np.arctan2(u @ e2, u @ e1)
                    for turn in range(-3, 4):
                        t = float(angle + turn * TAU)
                        if lo + 1e-10 < t < hi - 1e-10:
                            cuts.append(t)
        cuts = sorted(set(cuts))
        def at(t):
            return height*normal + radius*(np.cos(t)*e1 + np.sin(t)*e2)
        for start, stop in zip(cuts[:-1], cuts[1:]):
            middle = (start+stop)/2
            witnesses = family.lifts(at(middle), sun)
            if not witnesses:
                continue
            grid = np.linspace(start, stop, max(3, int(np.ceil(samples*(stop-start)/TAU))+1))
            for sheet, witness in enumerate(witnesses):
                # Track each analytic branch outward from the midpoint. Matching
                # rotation matrices survives periodic azimuth seams and folds.
                run = {}
                for direction in (-1, 1):
                    previous = np.asarray(family.rotation(witness))
                    order = sorted(grid, key=lambda t: abs(t-middle))
                    for t in order:
                        if (t-middle)*direction < 0:
                            continue
                        qs = family.lifts(at(t), sun)
                        if not qs:
                            raise RestrictedEscape("lift_endpoint", "analytic clipping endpoint lost its preimage")
                        q = min(qs, key=lambda p: np.linalg.norm(np.asarray(family.rotation(p))-previous))
                        previous = np.asarray(family.rotation(q))
                        p = _point(field, family, sun, q)
                        p.update(residual=abs(float(field.margins_batch(p["u"][None])[0, k])), intersection=state)
                        run[float(t)] = p
                ordered = [run[t] for t in sorted(run)]
                curves.append(dict(points=ordered, closed=bool(closed and len(cuts)==2),
                                   ends=(ends[0] if start==lo else "family_image_boundary",
                                         ends[1] if stop==hi else "family_image_boundary"),
                                   sheet=sheet, intersection=state, dimension=1))
    return dict(kind=kind, margin=margin, records=[], curves=curves, coverage=coverage)


def kink_sky(field: DPField, family: ExactFamily, sun, margin: str, **options):
    """Actual oriented-family sky image, not an arbitrary alignment of u to sun."""
    result = intersect_margin(field, family, sun, margin, **options)
    if result["kind"] != "kind_3":
        raise ValueError("kink_sky requires an internal TIR onset")
    if result["curves"]:
        dimension = 1
        for curve in result["curves"]:
            sky = np.array([p["sky"] for p in curve["points"]])
            if np.max(np.ptp(sky, axis=0)) < 1e-9:
                # A single mirror normal parallel to body z, constrained to
                # horizontal by column: a.u=k fixes R a to the intersection
                # of two planes and S2, hence each connected branch is a point.
                basal_mirror = field.slab is not None and np.allclose(field.slab, np.diag([1.,1.,-1.]), atol=1e-12, rtol=0)
                circle_pinned = family.dimension == 1 and field.slab is not None and commutes_with_rotation_about(field.slab, family.circle(_sun(sun))[0])
                if not ((family.name == "column" and basal_mirror) or circle_pinned):
                    raise RestrictedEscape("unclassified_sky_collapse", "constant sky samples have no structural identity proof")
                curve["image_dimension"] = 0
            else:
                curve["image_dimension"] = 1
        dimension = max(c["image_dimension"] for c in result["curves"])
    else:
        dimension = 0 if result["records"] else None
    return dict(**result, image_dimension=dimension, mapping="-R Phi_P(-R.T sun)")
