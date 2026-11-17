"""S6: opportunistic intersections of existing S2 structural strata.

This is not a completeness certificate. Sources retain upstream identities;
small polyline distances generate seeds, never intersection certificates.
Only regular isolated roots, analytic circle contacts, and structural identities
are accepted. Numerical rank loss is an unresolved contact, not a tangency.
See ``docs/junction.md`` for the result and parity contract.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.spatial import cKDTree

from . import optics
from .dp_field import DPField
from .dp_field.boundary import VIOLATION_ATOL, Walker, incidence_normals
from .family_restricted import ExactFamily, RestrictedEscape, _circle_roots, _margin_plane, _sun, intersect_margin
from .spherical_newton import refine_pair

RESIDUAL_ATOL = 1e-11  # dimensionless margins / unit-plane residuals
MEMBERSHIP_RAD = 0.012  # > two upstream 0.25-degree steps; locality only
RANK_RATIO_MIN = 1e-7  # numerical contacts below this are explicitly unresolved
MERGE_RAD = 1e-10  # repeat numerical seeds only; analytic twin roots are never merged


def _angle(a, b):
    return float(np.arctan2(np.linalg.norm(np.cross(a, b)), a @ b))


@dataclass(frozen=True, eq=False)
class Source:
    """One upstream stratum; several arcs can have the same physical margin."""

    id: str
    kind: str
    dimension: int
    arcs: tuple[np.ndarray, ...]
    origin: tuple[str, ...]
    margin: str | None = None
    plane: tuple[np.ndarray, float] | None = None
    point: np.ndarray | None = None
    equivalent_to: str | None = None
    failed_seeds: int = 0

    @property
    def points(self):
        return np.concatenate(self.arcs) if self.arcs else np.zeros((0, 3))

    def descriptor(self):
        return dict(id=self.id, kind=self.kind, dimension=self.dimension,
                    margin=self.margin, origin=list(self.origin),
                    equivalent_to=self.equivalent_to, failed_seeds=self.failed_seeds,
                    model="point" if self.point is not None else "circle" if self.plane is not None else "numerical_zero_set")


def structural_sources(field: DPField) -> tuple[Source, ...]:
    """Consume boundary pieces, kink arcs, interior points, and known slab strata.

    A coincident gate keeps its own identity. Repeated reflection identities
    likewise stay visible, rather than disappearing during geometric dedup.
    """
    sources = []
    boundary = field.boundary
    names = sorted({n for p in boundary.pieces for n in (p.margin, *p.coincident)})
    for name in names:
        pieces = [(i, p) for i, p in enumerate(boundary.pieces) if name in (p.margin, *p.coincident)]
        sources.append(Source("kind_2:" + name, "kind_2", 1,
                              tuple(p.points for _, p in pieces),
                              tuple(f"boundary_piece:{i}" for i, _ in pieces),
                              name, _margin_plane(field, name)))
    for curve in field.weight_kinks:
        equivalent = boundary.identical_margins.get(curve.margin)
        sources.append(Source("kind_3:" + curve.margin, "kind_3", 1,
                              tuple(a.points for a in curve.arcs),
                              tuple(f"kink:{curve.step}:arc:{i}" for i in range(len(curve.arcs))),
                              curve.margin, _margin_plane(field, curve.margin),
                              equivalent_to=None if equivalent is None else "kind_3:" + equivalent,
                              failed_seeds=curve.failed_seeds))
    for i, point in enumerate(field.interior_critical_points):
        sources.append(Source(f"kind_1:interior:{i}", "kind_1", 0, (),
                              (f"interior_critical:{i}:{point.kind}",), point=point.position))
    fold = field.degenerate_fold
    if fold is not None and fold.axis is not None:
        for i, (u, where) in enumerate(fold.axis_points):
            if where == "interior":
                sources.append(Source(f"kind_1:axis:{i}", "kind_1", 0, (),
                                      (f"degenerate_fold:axis:{i}",), point=u))
        # The existing degenerate-fold authority defines this crease analytically,
        # not by treating |grad D|^2 (a nonregular zero set) as an equation.
        axis = fold.axis
        e = np.cross(axis, np.eye(3)[np.argmin(np.abs(axis))])
        e /= np.linalg.norm(e)
        t = np.linspace(0, 2*np.pi, 1441)
        circle = np.cos(t)[:, None]*e + np.sin(t)[:, None]*np.cross(axis, e)
        sources.append(Source("kind_1:crease", "kind_1", 1, (circle,),
                              ("degenerate_fold:crease",), plane=(axis, 0.)))
    return tuple(sources)


def _constraint(walker, source, u):
    if source.plane is not None:
        a, h = source.plane
        return float(a @ u-h), a
    if source.margin is None:
        raise ValueError("point strata have no regular zero-set equation")
    values, jac = walker.margins_jacobian(u)
    k = walker.k(source.margin)
    return float(values[k]), jac[k]


def _raw_residual(walker, source, u):
    if source.point is not None:
        return _angle(u, source.point)
    if source.margin is not None:
        return abs(float(walker.margins(u)[walker.k(source.margin)]))
    a, h = source.plane
    return abs(float(a @ u-h))


def _member(walker, source, u):
    if _raw_residual(walker, source, u) > RESIDUAL_ATOL:
        return False
    if source.point is not None:
        return True
    if source.id == "kind_1:crease":
        return True  # analytic circle, clipped by all gates below
    points = source.points
    return bool(len(points) and cKDTree(points).query(u)[0] <= MEMBERSHIP_RAD)


def _admissible(walker, u):
    return (np.isfinite(u).all() and abs(float(u @ u)-1) <= 1e-12
            and not walker.violated(u, set()))


def _record(walker, a, b, u, intersection, evidence, singular_values=None):
    margins = walker.margins(u)
    return dict(sources=[a.id, b.id], u=np.asarray(u), intersection=intersection,
                residuals=[_raw_residual(walker, s, u) for s in (a, b)],
                min_gate=float(min(margins[walker.k(n)] for n in optics.validity_margin_names(walker.faces))),
                value=walker.d(u), evidence=evidence,
                differential=dict(available=singular_values is not None,
                                  reason=None if singular_values is not None else "analytic_or_upstream_nonsmooth_stratum",
                                  singular_values=singular_values))


def _identity(walker, a, b):
    """Prove a zero-set identity algebraically, never from sampled closeness."""
    if a.equivalent_to == b.id or b.equivalent_to == a.id:
        return "upstream_identical_margins"
    if a.plane is not None and b.plane is not None:
        if _circle_roots(*a.plane, *b.plane) is None:
            return "analytic_identical_planes"
    # exit disc = -(internal TIR disc) when the unfolded normals agree up
    # to sign. Both are 1-n^2*(1-c^2), with opposite sign conventions.
    names = (a.margin, b.margin)
    if "exit_snell_discriminant" in names:
        other = names[1] if names[0] == "exit_snell_discriminant" else names[0]
        if other and other.startswith("internal_") and other.endswith("tir_discriminant"):
            normals = incidence_normals(walker.crystal, walker.faces)
            x = normals["exit_incidence_cosine"]
            y = normals[other.replace("tir_discriminant", "incidence_cosine")]
            if min(np.linalg.norm(x-y), np.linalg.norm(x+y)) <= 8*np.finfo(float).eps:
                return "analytic_equal_squared_incidence"
    return None


def _continuous(walker, a, b, evidence):
    """Keep contiguous validated upstream runs; never list these as junction points."""
    source = b if a.id == "kind_1:crease" else a
    curves = []
    for i, points in enumerate(source.arcs):
        flags = np.array([_admissible(walker, u) and _raw_residual(walker, a, u) <= RESIDUAL_ATOL
                          and _raw_residual(walker, b, u) <= RESIDUAL_ATOL for u in points])
        for run in np.split(np.arange(len(points)), np.flatnonzero(np.diff(flags)) + 1):
            if len(run) >= 2 and flags[run[0]]:
                curves.append(dict(sources=[a.id, b.id], intersection="coincident", dimension=1,
                                   upstream_arc=i, points=points[run], evidence=evidence,
                                   coverage="analytic_identity_upstream_arc_clipping"))
    return curves


def _seeds(walker, a, b, nearby):
    candidates = []
    for first, second in ((a, b), (b, a)):
        for points in first.arcs:
            if not len(points):
                continue
            values = np.array([_constraint(walker, second, u)[0] for u in points])
            for i in np.flatnonzero(values[:-1]*values[1:] < 0):
                t = abs(values[i])/(abs(values[i])+abs(values[i+1]))
                u = (1-t)*points[i] + t*points[i+1]
                candidates.append(u/np.linalg.norm(u))
            # Endpoints, near-zero local extrema (including contacts), and
            # nearest samples provide non-sign-changing candidate routes.
            indices = [0, len(points)-1]
            middle, before, after = abs(values[1:-1]), abs(values[:-2]), abs(values[2:])
            indices.extend((np.flatnonzero((middle <= before) & (middle <= after) &
                                          ((middle < before) | (middle < after)))+1).tolist())
            if len(second.points):
                d, _ = cKDTree(second.points).query(points)
                indices.extend((np.flatnonzero((d[1:-1] <= d[:-2]) & (d[1:-1] <= d[2:]) &
                                              ((d[1:-1] < d[:-2]) | (d[1:-1] < d[2:])))+1).tolist())
                candidates.extend(points[i] for i in sorted(set(indices)) if d[i] < nearby)
            else:
                candidates.extend(points[i] for i in sorted(set(indices)) if abs(values[i]) < nearby)
    unique = []
    for u in candidates:
        if not any(_angle(u, p) < 1e-7 for p in unique):
            unique.append(u)
    return unique


def _finish(result, status):
    if result["diagnostics"]:
        status = "partial" if result["records"] or result["curves"] else "unresolved"
    result["status"] = status
    return result


def intersect_sources(field: DPField, a: Source, b: Source, *, budget=64, nearby=0.02,
                      max_iterations=30, max_step=1e-3):
    """Intersect a selected pair; explicit diagnostics survive partial success.

    ``budget`` caps local refinements per pair, not upstream construction.
    Empty numerical searches are ``not_found``, never ``disjoint``.
    """
    if not isinstance(budget, int) or budget < 0 or not np.isfinite(nearby) or nearby <= 0:
        raise ValueError("budget must be a nonnegative integer and nearby must be positive")
    if max_iterations < 0 or not np.isfinite(max_step) or max_step <= 0:
        raise ValueError("invalid local refinement budget")
    walker = Walker(field.crystal, field.faces, field.index, field.slab)
    result = dict(sources=[a.id, b.id], status="not_found", coverage="opportunistic_not_exhaustive",
                  candidates=0, attempted=0, converged=0, rejected=0, records=[], curves=[], diagnostics=[])
    for source in (a, b):
        if source.failed_seeds:
            result["diagnostics"].append(dict(reason="upstream_failed_seeds", source=source.id, count=source.failed_seeds))
    if a.point is not None or b.point is not None:
        u = a.point if a.point is not None else b.point
        if _admissible(walker, u) and all(_member(walker, s, u) for s in (a, b)):
            result["records"].append(_record(walker, a, b, u, "point_membership", "upstream_point_raw_membership"))
        return _finish(result, "computed" if result["records"] else "not_found")
    identity = _identity(walker, a, b)
    if identity:
        result["curves"] = _continuous(walker, a, b, identity)
        return _finish(result, "continuous" if result["curves"] else "not_found")
    analytic = a.plane is not None and b.plane is not None
    if analytic:
        roots = _circle_roots(*a.plane, *b.plane)
        if not roots:
            result["coverage"] = "analytic_circles_disjoint"
            return _finish(result, "disjoint")
        seeds = [u for u, _ in roots]
    else:
        seeds = _seeds(walker, a, b, nearby)
    result["candidates"] = len(seeds)
    if len(seeds) > budget:
        result["diagnostics"].append(dict(reason="budget_exhausted", omitted_candidates=len(seeds)-budget))
    for i, seed in enumerate(seeds[:budget]):
        result["attempted"] += 1
        if analytic:
            u, state = roots[i]
            evidence, singular = "analytic_circle_intersection", None
        else:
            def evaluate(u):
                parts = [_constraint(walker, s, u) for s in (a, b)]
                return np.array([p[0] for p in parts]), np.array([p[1] for p in parts])

            def residual(u):
                return max(_raw_residual(walker, s, u) for s in (a, b))

            refined = refine_pair(seed, evaluate, residual, max_iterations=max_iterations, max_step=max_step,
                                  residual_tolerance=RESIDUAL_ATOL)
            u = refined.position
            if refined.termination != "converged":
                result["diagnostics"].append(dict(reason=refined.termination, seed=seed, residual=refined.residual))
                result["rejected"] += 1
                continue
            _, raw_jac = walker.margins_jacobian(u)
            jac = np.array([raw_jac[walker.k(s.margin)] if s.margin is not None else s.plane[0] for s in (a, b)])
            singular = np.linalg.svd(jac @ (np.eye(3)-np.outer(u, u)), compute_uv=False) if np.isfinite(jac).all() else None
            if singular is None or singular[0] == 0 or singular[-1] <= RANK_RATIO_MIN*singular[0]:
                result["diagnostics"].append(dict(reason="unresolved_contact", seed=seed))
                result["rejected"] += 1
                continue
            singular = singular.tolist()
            state, evidence = "transverse", "regular_local_newton"
        reason = None
        if not _admissible(walker, u):
            reason = "outside_closure_or_nonfinite"
        elif not all(_member(walker, s, u) for s in (a, b)):
            reason = "outside_source_arc_or_raw_residual"
        elif not analytic and _angle(seed, u) > MEMBERSHIP_RAD:
            reason = "nonlocal_root"
        if reason:
            result["diagnostics"].append(dict(reason=reason, seed=seed))
            result["rejected"] += 1
            continue
        result["converged"] += 1
        # Analytic roots have their own distinct algebraic branches, even near
        # tangency. Only repeated numerical seeds are proximity-deduplicated.
        if not analytic and any(_angle(u, p["u"]) < MERGE_RAD for p in result["records"]):
            continue
        result["records"].append(_record(walker, a, b, u, state, evidence, singular))
    result["records"].sort(key=lambda p: tuple(p["u"]))
    if result["diagnostics"]:
        result["status"] = "partial" if result["records"] else "unresolved"
    else:
        result["status"] = "computed" if result["records"] else "not_found"
    return result


def intersect_family(field: DPField, source: Source, family: ExactFamily, sun, *, samples=32, budget=4096):
    """Reuse exact-family intersections and retain every pose above each u.

    A two-dimensional image meeting a curve remains a curve. Dedicated
    unsupported regimes are data, while programming errors still propagate.
    """
    sun = _sun(sun)
    walker = Walker(field.crystal, field.faces, field.index, field.slab)
    pair = [source.id, "family:" + family.name]
    result = dict(sources=pair, status="not_found", records=[], curves=[], diagnostics=[],
                  coverage="opportunistic_family_adapter")
    if source.failed_seeds:
        result["diagnostics"].append(dict(reason="upstream_failed_seeds", count=source.failed_seeds))
    try:
        if source.margin is not None:
            if family.dimension == 1 and source.plane is not None:
                roots = _circle_roots(*source.plane, *family.circle(sun))
                if roots == []:
                    result.update(status="disjoint", coverage="analytic_circles_disjoint")
                    return _finish(result, "disjoint")
            data = intersect_margin(field, family, sun, source.margin, samples=samples, budget=budget)
            result["coverage"] = data["coverage"]
            for p in data["records"]:
                u = p["u"]
                if _admissible(walker, u) and _member(walker, source, u):
                    result["records"].append(dict(sources=pair, **p))
                else:
                    result["diagnostics"].append(dict(reason="outside_source_arc_or_closure"))
            for curve in data["curves"]:
                points = curve["points"]
                flags = np.array([_admissible(walker, p["u"]) and _member(walker, source, p["u"]) for p in points])
                if not flags.all():
                    result["diagnostics"].append(dict(reason="outside_source_arc_or_closure", rejected_points=int((~flags).sum())))
                for run in np.split(np.arange(len(points)), np.flatnonzero(np.diff(flags))+1):
                    if len(run) >= 2 and flags[run[0]]:
                        result["curves"].append(dict(curve, sources=pair, points=[points[i] for i in run],
                                                     closed=bool(curve["closed"] and flags.all())))
        elif source.point is not None:
            if _admissible(walker, source.point):
                for q in family.lifts(source.point, sun):
                    r = np.asarray(family.rotation(q))
                    result["records"].append(dict(sources=pair, u=source.point, q=q, rotation=r,
                                                  intersection="point_membership", residual=0.))
        elif source.plane is not None and family.dimension == 1:
            roots = _circle_roots(*source.plane, *family.circle(sun))
            if roots is None:
                result["status"] = "unsupported_continuous"
                result["diagnostics"].append(dict(reason="critical_circle_family_coincidence_clipping"))
                return result
            for u, state in roots:
                if _admissible(walker, u):
                    for q in family.lifts(u, sun):
                        result["records"].append(dict(sources=pair, u=u, q=q, rotation=np.asarray(family.rotation(q)),
                                                      intersection=state, residual=abs(float(source.plane[0] @ u-source.plane[1]))))
            if not roots:
                result["coverage"] = "analytic_circles_disjoint"
                result["status"] = "disjoint"
                return result
        else:
            raise RestrictedEscape("critical_stratum_adapter", "this critical stratum has no certified family clipping adapter")
    except RestrictedEscape as error:
        result["diagnostics"].append(dict(reason=error.regime, detail=str(error)))
        result["status"] = "partial" if result["records"] or result["curves"] else "unsupported"
        return result
    result["status"] = ("partial" if result["records"] or result["curves"] else "unresolved") if result["diagnostics"] else (
        "continuous" if result["curves"] else "computed" if result["records"] else "not_found")
    return result


def enumerate_junctions(field: DPField, *, pairs=None, family: ExactFamily | None = None, sun=None,
                        budget=64, nearby=0.02, max_iterations=30, max_step=1e-3,
                        family_samples=32, family_budget=4096):
    """Enumerate selected or all existing source pairs, optionally against a family.

    ``pairs`` is a sequence of two source IDs. A family pair uses the literal
    second ID ``family:<name>``. None means all structural pairs and, when
    requested, all source/family pairs. The global claim is ALWAYS non-exhaustive.
    """
    sources = structural_sources(field)
    by_id = {s.id: s for s in sources}
    options = dict(budget=budget, nearby=nearby, max_iterations=max_iterations, max_step=max_step)
    if pairs is None:
        selected = [(a.id, b.id) for a, b in combinations(sources, 2)]
        if family is not None:
            selected.extend((s.id, "family:"+family.name) for s in sources)
    else:
        selected = sorted(set(tuple(sorted(p)) for p in pairs))
    output = []
    for pair in selected:
        family_ids = [s for s in pair if s.startswith("family:")]
        if family_ids:
            if len(family_ids) != 1 or family is None or family_ids[0] != "family:"+family.name:
                raise ValueError("family pair must name the supplied exact family")
            source_id = next(s for s in pair if not s.startswith("family:"))
            output.append(intersect_family(field, by_id[source_id], family, sun, samples=family_samples, budget=family_budget))
        else:
            a, b = pair
            output.append(intersect_sources(field, by_id[a], by_id[b], **options))
    return dict(path=list(field.faces), index=field.index, exhaustive=False,
                coverage="opportunistic_existing_strata_only",
                omitted=["corridor_A_zero", "unrepresented_critical_manifolds", "undiscovered_upstream_components"],
                options=dict(**options, family_samples=family_samples, family_budget=family_budget),
                sources=[s.descriptor() for s in sources], pairs=output)
