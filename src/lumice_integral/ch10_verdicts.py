"""The chapter-10 numerical verdicts (``docs/phase2.md`` section 10; task ``ch10-numerical-verdicts``).

Four statements of the writing series' chapter 10 are settled here as
measurements on the Phase II chains, not assumed: every pixel value comes
from :mod:`.contour_quadrature` (the precision authority), every critical
value and Hessian from :class:`.dp_field.DPField`, every focusing label from
:mod:`.focusing`; this module only chooses the pixels, reduces the values to
the numbers a verdict states and records them.  Each function returns a
:class:`Verdict` (statement, numbers, inputs and arrays) that
:func:`.figure_data.export_verdict_figure_data` writes for the writing
repository; ``scripts/ch10_numerical_verdicts.py`` is the command line.

``inner-edge`` (:func:`inner_edge`)
    The 22 deg inner edge of path ``3-5``.  Random orientation: the pixel
    value jumps to a finite value at ``D_min`` (the raw level-set integral
    tends to ``w* 2 pi / sqrt(det H)``), approached as ``1 - a sqrt(eps)``
    because the entry area has a ``|u_z|`` kink at the minimum-deviation
    point.  Column family, at the ring azimuth where its orientation ridge
    passes through the minimum (the tangent-arc contact): ``I ~ eps^(-1/2)``
    between a crossover ``eps_c ~ sigma^2`` and ``~1e-2``, finite
    (``~ 1 / sigma``) below ``eps_c``.
``liljequist`` (:func:`liljequist`)
    (i) the 142 deg edge of class A60-10 (``3-5-6-7``, ``3-4-5-7``, one
    partial internal reflection each): the ``D_P`` saddle
    ``120 deg + D_min(3-5)`` on both members and every ``h / a``.  (ii)
    ``1-3-2`` and ``3-5-6-7-3`` share one mirror-slab field
    ``D = 2 arcsin |u . n_3|`` with ``|grad D| = 2``: no fold; their
    critical values do not depend on ``h / a``, their windows (profiles)
    do.  The Liljequist peak of ``3-5-6-7-3`` is the corner the internal
    TIR onset of ``R_k`` puts at 153.07 deg (a slope jump, not a critical
    value of ``D_P``); its one boundary critical value, 98.16 deg, is a
    grazing internal incidence where ``w`` vanishes: no visible edge.
``parhelic-circle`` (:func:`parhelic_circle`)
    ``1-3-2`` under plates: the ring's elevation-integrated brightness equals
    the window-only prediction ``sum_phi w(phi) / (2 pi |d theta / d phi|)``
    with ``d theta / d phi = 2`` (no Jacobian), up to the plate width.
``parallel-face`` (:func:`parallel_face`)
    The focusing labels of :mod:`.focusing` over fixtures x density families,
    and the dimension-collapse signature of the parhelic circle (peak
    ``~ 1 / sigma`` at a fixed integral) on a parallel-face (wedge 0,
    ``M != I``) class.

``eps`` is ``delta - D_min`` in radians throughout.

Nothing here imports or calls Lumice.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field as dataclass_field
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from . import contour_quadrature as cq
from . import focusing
from .camera import incident_direction_from_sun, sun_direction
from .canonical_scene import CANONICAL_HEIGHT_RATIO, CANONICAL_REFRACTIVE_INDEX, canonical_crystal, canonical_sun_direction
from .contour import extract_level_sets
from .dp_field import DPField
from .dp_field.boundary import EXTREMUM_ATOL
from .dp_field.field import d_value, margin_vector, tangent_basis
from .geometry import HexPrism, halo_map_rank
from .optics import domain_margin_names, path_id_of, validity_margin_indices
from .pose_density import build_pose_density
from .s2_store import S2EventStore, align_rotations, build_event_store, evaluate_fields, event_rotations, fibonacci_sphere

VERDICTS = ("inner-edge", "liljequist", "parhelic-circle", "parallel-face")
# The store only seeds the extraction's independent check (contour module docstring); the tests' size.
SEED_STORE_N = 200_000
LATTICE_N = 200_000
# Class A60-10 (the 142 deg parhelion) and its D_P saddle from the task verify-liljequist-face-numbering numpy probe;
# a60_10_saddle measures it on the production chain.
A60_10_MEMBERS = ((3, 5, 6, 7), (3, 4, 5, 7))
A60_10_REFERENCE_DEG = 141.839300
# tir_onset_maximum: lattice starts within this of a TIR discriminant's zero; accepted solutions satisfy it to this.
ONSET_START_BAND = 1e-3
ONSET_CONSTRAINT_ATOL = 1e-12
# Smallest ring cross-section window, in units of the plate width (theta -> 0, where the ring is dark anyway).
CROSS_WIDTH_FLOOR = 1e-2


@dataclass(frozen=True, eq=False)
class Verdict:
    """One verdict: its statement, the numbers it rests on, its inputs and its arrays.

    ``status`` is ``"measured"``, or ``"partially_blocked"`` when an item
    could not be computed in this repository (``numbers`` says which and
    why).  ``arrays`` are the figure data; ``array_notes`` give each one's
    unit and meaning.
    """

    name: str
    status: str
    statement: str
    numbers: dict[str, Any]
    parameters: dict[str, Any]
    arrays: dict[str, np.ndarray] = dataclass_field(default_factory=dict)
    array_notes: dict[str, str] = dataclass_field(default_factory=dict)


# ---- shared helpers -------------------------------------------------------------------------------


def ring_direction(sun: np.ndarray, delta: float, azimuth: float) -> np.ndarray:
    """The outgoing (centre) direction at deviation ``delta`` from the propagation ``-sun``, at ``azimuth`` about it.

    ``azimuth`` is measured from ``e1 = s x z`` towards ``e2 = s x e1``
    (``s`` the propagation direction); ``pi / 2`` is the top of the ring
    (sky direction ``-centre`` above the sun) for a sun above the horizon.
    The sun must not be at the zenith or nadir (``e1`` undefined).
    """
    s = incident_direction_from_sun(sun)
    if abs(s[2]) > 1.0 - 1e-12:
        raise ValueError("ring_direction needs a sun off the zenith and nadir")
    e1 = np.cross(s, [0.0, 0.0, 1.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(s, e1)
    return np.cos(delta) * s + np.sin(delta) * (np.cos(azimuth) * e1 + np.sin(azimuth) * e2)


def seed_store(crystal: HexPrism, index: float, faces: Sequence[int], n: int = SEED_STORE_N) -> S2EventStore:
    return build_event_store(crystal, index, [tuple(faces)], n, run_checks=False)


def level_set_geometry(
    field: DPField, deltas: np.ndarray, store: S2EventStore, options: cq.QuadratureOptions
) -> cq.LevelSetGeometry:
    """Stage one of the quadrature for ``deltas``."""
    level_sets = extract_level_sets(field, np.asarray(deltas, dtype=np.float64), store)
    return cq.LevelSetGeometry.build(field, level_sets, options)


def _weight_at(u: np.ndarray, sun: np.ndarray, crystal: HexPrism, index: float, faces: Sequence[int]) -> np.ndarray:
    """``w = A_P T_P`` at body directions ``u`` (production evaluators; the pose's twist is irrelevant)."""
    return evaluate_fields(align_rotations(np.atleast_2d(u), sun), sun, crystal, index, [tuple(faces)])["w"]


def _local_slopes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """``d log y / d log x`` by ``np.gradient`` (``x`` need not be uniform in log)."""
    return np.gradient(np.log(y), np.log(x))


def finite_difference_hessian(field: DPField, u: np.ndarray, step: float = 1e-4) -> np.ndarray:
    """The Riemannian Hessian of ``D_P`` at ``u`` by central differences of the field value along geodesics.

    Independent of the AD Hessian of :mod:`.dp_field` (a different failure
    mode: no curvature term to forget, only truncation ``~step^2`` and
    round-off ``~eps / step^2``).  Returned in :func:`.dp_field.field.tangent_basis`
    coordinates at ``u``.
    """
    basis = np.asarray(tangent_basis(np.asarray(u, dtype=np.float64)))

    def at(x: float, y: float) -> np.ndarray:
        v = x * basis[0] + y * basis[1]
        r = np.hypot(x, y)
        return u if r == 0.0 else np.cos(r) * u + np.sin(r) * v / r

    offsets = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    points = np.stack([at(step * a, step * b) for a, b in offsets])
    d = dict(zip(offsets, field.d_p_batch(points)))
    hxx = (d[(1, 0)] - 2.0 * d[(0, 0)] + d[(-1, 0)]) / step**2
    hyy = (d[(0, 1)] - 2.0 * d[(0, 0)] + d[(0, -1)]) / step**2
    hxy = (d[(1, 1)] - d[(1, -1)] - d[(-1, 1)] + d[(-1, -1)]) / (4.0 * step**2)
    return np.array([[hxx, hxy], [hxy, hyy]])


# ---- verdict 1: the 22 deg inner edge --------------------------------------------------------------


@dataclass(frozen=True)
class InnerEdgeOptions:
    """Inputs of :func:`inner_edge` (defaults: the recorded run)."""

    eps_random: tuple[float, ...] = tuple(10.0 ** np.arange(-1.0, -6.01, -0.25))
    eps_column: tuple[float, ...] = tuple(10.0 ** np.arange(-1.0, -6.51, -0.25))
    column_widths_deg: tuple[float, ...] = (0.5, 0.1, 0.02, 0.005)
    azimuth_step_deg: float = 0.1
    # fit of ratio = c - a sqrt(eps) - b eps on eps <= this
    extrapolation_max_eps: float = 1e-3
    # |local slope + 1/2| within this is the power-law window
    power_law_band: float = 0.1
    relative_tolerance: float = 1e-8
    seed_store_n: int = SEED_STORE_N


def _power_law_window(eps: np.ndarray, slopes: np.ndarray, band: float) -> tuple[float, float] | None:
    """The longest run of consecutive ``eps`` with ``|slope + 1/2| <= band``: ``(smallest, largest)`` eps."""
    inside = np.abs(slopes + 0.5) <= band
    best, start = None, None
    for k in range(len(eps) + 1):
        if k < len(eps) and inside[k]:
            start = k if start is None else start
            continue
        if start is not None:
            if best is None or k - start > best[1] - best[0]:
                best = (start, k)
            start = None
    if best is None or best[1] - best[0] < 2:
        return None
    run = eps[best[0]:best[1]]
    return float(run.min()), float(run.max())


def _window_text(window: tuple[float, float] | None) -> str:
    return "empty" if window is None else f"[{window[0]:.1e}, {window[1]:.1e}] rad"


def _crossover(eps: np.ndarray, slopes: np.ndarray, level: float = -0.25) -> float | None:
    """The ``eps`` below the steepest slope where the slope rises through ``level`` (log-interpolated): the cap onset.

    ``eps`` is decreasing.  ``None`` if the slope never rises back to
    ``level`` within the grid (the cap is below the grid).
    """
    k0 = int(np.argmin(slopes))
    for k in range(k0, len(eps) - 1):
        if slopes[k] < level <= slopes[k + 1]:
            t = (level - slopes[k]) / (slopes[k + 1] - slopes[k])
            return float(np.exp((1.0 - t) * np.log(eps[k]) + t * np.log(eps[k + 1])))
    return None


def inner_edge(options: InnerEdgeOptions = InnerEdgeOptions()) -> Verdict:
    """Verdict 1 (module docstring): path ``3-5`` on the canonical crystal, sun and refractive index."""
    faces = (3, 5)
    crystal, index, sun = canonical_crystal(), CANONICAL_REFRACTIVE_INDEX, canonical_sun_direction()
    field = DPField.build(crystal, faces, index)
    (minimum,) = [p for p in field.interior_critical_points if p.kind == "minimum"]
    u_star, d_min = minimum.position, minimum.value
    onset = focusing.interior_onset(d_min, minimum.kind, minimum.hessian_eigenvalues, minimum.gradient_norm)
    w_star = float(_weight_at(u_star, sun, crystal, index, faces)[0])
    raw_limit = w_star * onset.measure_limit
    pixel_limit = raw_limit * cq.HAAR_TO_DVOL_G_FACTOR / np.sin(d_min)

    # independent Hessian (finite differences of the value) against the AD one
    fd_eigenvalues = np.linalg.eigvalsh(finite_difference_hessian(field, u_star))
    ad_eigenvalues = np.sort(minimum.hessian_eigenvalues)
    hessian, basis = field.hessian_tangent_batch(u_star[None, :])
    eigenvalues, vectors = np.linalg.eigh(0.5 * (hessian[0] + hessian[0].T))
    directions = vectors.T @ basis[0]
    # kinks of w = A_P T_P at u*: one-sided slopes along each Hessian eigendirection (u* is in the basal plane,
    # so A_P depends on |u_z|; in plane the corridor switches its bounding vertex there).  With the even part
    # kappa_i = -(s_+ + s_-) / 2 w* per direction, the mean of w over the ellipse D - D_min <= eps gives
    # ratio = 1 - (2 sqrt 2 / pi) sum_i kappa_i sqrt(eps / lambda_i); the odd parts average out.
    t = np.array([1e-5, 2e-5])
    kinks = []
    for eigenvalue, direction in zip(eigenvalues, directions):
        probes = np.concatenate([np.cos(t)[:, None] * u_star + np.sin(sign * t)[:, None] * direction for sign in (1.0, -1.0)])
        w_probe = _weight_at(probes, sun, crystal, index, faces)
        one_sided = [float((w_probe[1] - w_probe[0]) / (t[1] - t[0])), float((w_probe[3] - w_probe[2]) / (t[1] - t[0]))]
        kinks.append({"eigenvalue": float(eigenvalue), "direction": direction.tolist(), "one_sided_slopes": one_sided,
                      "kappa": -0.5 * sum(one_sided) / w_star})
    predicted_sqrt_coefficient = 2.0 * np.sqrt(2.0) / np.pi * sum(k["kappa"] / np.sqrt(k["eigenvalue"]) for k in kinks)

    store = seed_store(crystal, index, faces, options.seed_store_n)
    quadrature = cq.QuadratureOptions(relative_tolerance=options.relative_tolerance)

    eps_r = np.asarray(options.eps_random)
    geometry = level_set_geometry(field, d_min + eps_r, store, quadrature)
    random = geometry.integrate(sun, [ring_direction(sun, d, 0.0) for d in d_min + eps_r], build_pose_density("random"))
    raw_r = np.array([r.raw_value for r in random])
    ratio = raw_r / raw_limit
    fit = eps_r <= options.extrapolation_max_eps
    design = np.stack([np.ones(fit.sum()), -np.sqrt(eps_r[fit]), -eps_r[fit]], axis=1)
    (c, a, b), *_ = np.linalg.lstsq(design, ratio[fit], rcond=None)

    # the ring azimuth where the column ridge passes through u*: the rho maximum at u*
    phi_star = evaluate_fields(align_rotations(u_star[None, :], sun), sun, crystal, index, [faces])["phi"]
    azimuths = np.radians(np.arange(0.0, 360.0, options.azimuth_step_deg))
    poses = np.concatenate([event_rotations(u_star[None, :], phi_star, np.array([d_min]), sun, ring_direction(sun, d_min, a))
                            for a in azimuths])
    rho = build_pose_density("column", zenith_std_deg=options.column_widths_deg[0]).evaluate_batch(poses)
    peaks = azimuths[rho >= rho.max() * (1.0 - 1e-9)]
    sky_elevation = np.array([np.degrees(np.arcsin(-ring_direction(sun, d_min, a)[2])) for a in peaks])
    azimuth = float(peaks[int(np.argmax(sky_elevation))])

    eps_c = np.asarray(options.eps_column)
    centres = [ring_direction(sun, d, azimuth) for d in d_min + eps_c]
    column_values, column_errors, column_slopes, per_width, column_quadrature = [], [], [], [], {}
    for width in options.column_widths_deg:
        # first panels of 2 sigma put Simpson points sigma / 2 apart (QuadratureOptions: 1 deg against 0.5 deg);
        # a coarser panel misses the density ridge and its error estimate cannot see that
        panel = min(float(np.radians(1.0)), 2.0 * float(np.radians(width)))
        options_w = dataclasses.replace(quadrature, initial_panel_rad=panel)
        column_quadrature[width] = options_w.as_json()
        geometry = level_set_geometry(field, d_min + eps_c, store, options_w)
        results = geometry.integrate(sun, centres, build_pose_density("column", zenith_std_deg=width))
        values = np.array([r.value for r in results])
        slopes = _local_slopes(eps_c, values)
        column_values.append(values)
        column_errors.append([r.error_estimate for r in results])
        column_slopes.append(slopes)
        per_width.append({
            "zenith_std_deg": width,
            "local_slope_range_below_1e-2": [float(slopes[eps_c <= 1e-2].min()), float(slopes[eps_c <= 1e-2].max())],
            "power_law_window_eps": _power_law_window(eps_c, slopes, options.power_law_band),
            "cap_crossover_eps": _crossover(eps_c, slopes),
            "value_at_smallest_eps": float(values[-1]),
            "max_relative_error_estimate": float(np.max(np.array(column_errors[-1]) / values)),
        })
    crossovers = [(p["zenith_std_deg"], p["cap_crossover_eps"]) for p in per_width if p["cap_crossover_eps"] is not None]
    crossover_exponent = None
    if len(crossovers) >= 2:
        s, e = np.log(np.radians([c_[0] for c_ in crossovers])), np.log([c_[1] for c_ in crossovers])
        crossover_exponent = float(np.polyfit(s, e, 1)[0])

    canonical = per_width[0]
    numbers = {
        "path": path_id_of(faces),
        "D_min_deg": float(np.degrees(d_min)),
        "hessian_eigenvalues_ad": ad_eigenvalues.tolist(),
        "hessian_eigenvalues_finite_difference": fd_eigenvalues.tolist(),
        "hessian_max_relative_difference": float(np.max(np.abs(fd_eigenvalues - ad_eigenvalues) / ad_eigenvalues)),
        "measure_limit_2pi_over_sqrt_det_H": onset.measure_limit,
        "w_at_minimum": w_star,
        "raw_limit_analytic": raw_limit,
        "pixel_value_limit_analytic": pixel_limit,
        "random": {
            "ratio_to_analytic_at_smallest_eps": float(ratio[-1]),
            "smallest_eps": float(eps_r[-1]),
            "extrapolated_ratio_c": float(c),
            "sqrt_eps_coefficient_fitted": float(a),
            "eps_coefficient_fitted": float(b),
            "sqrt_eps_coefficient_predicted_from_kinks": float(predicted_sqrt_coefficient),
            "kinks": kinks,
            "max_relative_error_estimate": float(max(r.raw_error_estimate / r.raw_value for r in random)),
            "local_slope_at_smallest_eps": float(_local_slopes(eps_r, raw_r)[-1]),
        },
        "column": {
            "ring_azimuth_deg": float(np.degrees(azimuth)),
            "ring_top_sky_elevation_deg": float(sky_elevation.max()),
            "widths": per_width,
            "cap_crossover_exponent_in_sigma": crossover_exponent,
            # the value at the smallest eps is the cap only where the crossover was reached (just above it the local
            # slope overshoots to ~ -0.59 before levelling off)
            "cap_times_sigma_rad": {p["zenith_std_deg"]: float(p["value_at_smallest_eps"] * np.radians(p["zenith_std_deg"]))
                                    for p in per_width if p["cap_crossover_eps"] is not None},
        },
    }
    statement = (
        f"Random orientation: finite jump at D_min = {np.degrees(d_min):.6f} deg. The level-set integral tends to "
        f"w* 2 pi / sqrt(det H) (pixel value {pixel_limit:.6g}); at eps = {eps_r[-1]:.0e} rad the ratio is {ratio[-1]:.5f}, "
        f"approached as 1 - {a:.3f} sqrt(eps) (the kinks of w = A_P T_P at the minimum predict {predicted_sqrt_coefficient:.3f}), "
        f"extrapolated {c:.6f}. Column family at the tangent-arc contact (ring azimuth {np.degrees(azimuth):.1f} deg, the top): "
        f"I ~ eps^(-1/2) (local slope within {options.power_law_band} of -1/2) only between a cap crossover "
        f"eps_c ~ sigma^{'?' if crossover_exponent is None else f'{crossover_exponent:.2f}'} "
        f"(from {len(crossovers)} widths) and {max(p['power_law_window_eps'][1] for p in per_width if p['power_law_window_eps']):.1e} rad, "
        f"finite below eps_c (cap x sigma roughly constant); at the canonical sigma = {canonical['zenith_std_deg']} deg the "
        f"window is {_window_text(canonical['power_law_window_eps'])} and the cap is {canonical['value_at_smallest_eps']:.4g}."
    )
    parameters = {
        "crystal": {"type": "hexagonal_prism", "height_ratio": CANONICAL_HEIGHT_RATIO},
        "refractive_index": index,
        "sun_direction": sun.tolist(),
        "options": dataclasses.asdict(options),
        "quadrature": quadrature.as_json(),
        "quadrature_column": column_quadrature,
    }
    arrays = {
        "eps_random": eps_r,
        "raw_random": raw_r,
        "value_random": np.array([r.value for r in random]),
        "error_random": np.array([r.error_estimate for r in random]),
        "eps_column": eps_c,
        "column_widths_deg": np.asarray(options.column_widths_deg, dtype=float),
        "value_column": np.array(column_values),
        "error_column": np.array(column_errors, dtype=float),
        "local_slope_column": np.array(column_slopes),
    }
    notes = {
        "eps_random": "rad; delta - D_min of the random-orientation pixels",
        "raw_random": "length^2; sum over components of int w / |grad D_P| dl (rho = 1)",
        "value_random": "pixel value I (per steradian, Haar probability)",
        "error_random": "pixel value; the quadrature's error estimate",
        "eps_column": "rad; delta - D_min of the column-family pixels (ring azimuth in metadata)",
        "column_widths_deg": "deg; zenith standard deviations, rows of the column arrays",
        "value_column": "pixel value I per (width, eps)",
        "error_column": "pixel value; error estimate per (width, eps)",
        "local_slope_column": "d log I / d log eps per (width, eps)",
    }
    return Verdict("inner-edge", "measured", statement, numbers, parameters, arrays, notes)


# ---- verdict 2: Liljequist -----------------------------------------------------------------------


@dataclass(frozen=True)
class LiljequistOptions:
    height_ratios: tuple[float, ...] = (0.2, 1.0, 2.0)
    a60_10_members: tuple[tuple[int, ...], ...] = A60_10_MEMBERS
    # the U_P of 3-5-6-7 has a neck the default lattice splits in two (tests/test_dp_field_certificate.py A60_10)
    a60_10_lattice_n: int = 50_000
    # profile grids (deg) per path
    grid_1_3_2: tuple[float, float, float] = (0.25, 115.5, 0.25)
    grid_3_5_6_7_3: tuple[float, float, float] = (0.25, 179.75, 0.25)
    # the fine profile around the Liljequist peak (the TIR onset corner near 153.07 deg)
    peak_window_deg: tuple[float, float] = (145.0, 165.0)
    peak_step_deg: float = 0.05
    cusp_eps: tuple[float, ...] = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6)
    lattice_n: int = LATTICE_N
    relative_tolerance: float = 1e-8
    seed_store_n: int = SEED_STORE_N


def _grid(spec: tuple[float, float, float]) -> np.ndarray:
    lo, hi, step = spec
    return np.radians(np.arange(lo, hi + 0.5 * step, step))


def _random_profile(field: DPField, deltas: np.ndarray, store: S2EventStore, options: cq.QuadratureOptions) -> tuple[np.ndarray, np.ndarray, list]:
    """Random-orientation pixel values on ``deltas`` (those at a critical value dropped); ``(deltas, values, results)``."""
    deltas = np.array([d for d in deltas if not cq.critical_delta(field, d)])
    geometry = level_set_geometry(field, deltas, store, options)
    sun = canonical_sun_direction()  # rho = 1: the value depends on delta only (contour_quadrature), any sun will do
    results = geometry.integrate(sun, [ring_direction(sun, d, 0.0) for d in deltas], build_pose_density("random"))
    return deltas, np.array([r.value for r in results]), results


def _half_maximum_range(deltas: np.ndarray, values: np.ndarray) -> tuple[float, float]:
    k = int(np.argmax(values))
    above = values >= 0.5 * values[k]
    lo = k
    while lo > 0 and above[lo - 1]:
        lo -= 1
    hi = k
    while hi < len(values) - 1 and above[hi + 1]:
        hi += 1
    return float(np.degrees(deltas[lo])), float(np.degrees(deltas[hi]))


def blocked_class_status(members: Sequence[Sequence[int]], index: float, lattice_n: int = LATTICE_N) -> dict[str, Any]:
    """For each member: lattice points in ``U_P`` (0 means no event) and points passing every gate but internal TIR.

    The margins do not depend on the crystal (``U_P`` is shape independent),
    so one count per member covers every ``h / a``.  Since internal
    reflections are partial (task ``optics-partial-reflection``) the two
    counts agree; they differed (0 valid) while internal TIR was a gate.
    """
    lattice = fibonacci_sphere(lattice_n)
    out = {}
    for faces in members:
        faces = tuple(faces)
        field = DPField.build(canonical_crystal(), faces, index)
        margins = field.margins_batch(lattice)
        others = [k for k, name in enumerate(domain_margin_names(faces)) if not name.endswith("_tir_discriminant")]
        out[path_id_of(faces)] = {
            "lattice_points": int(lattice_n),
            "valid_points": int(np.sum(field.valid_batch(lattice))),
            "points_passing_all_but_internal_tir": int(np.sum(np.all(margins[:, others] > 0.0, axis=1))),
        }
    return out


def a60_10_saddle(
    index: float, height_ratios: Sequence[float], members: Sequence[Sequence[int]] = A60_10_MEMBERS, lattice_n: int = 50_000
) -> dict[str, Any]:
    """The 142 deg edge of class A60-10: the middle of each member's three critical values, per ``h / a``.

    ``D_P`` of ``3-5-6-7`` / ``3-4-5-7`` has no interior critical point; its
    critical values are the two corner values and, between them, the
    maximum of the ``dU_P`` loop on a grazing internal-reflection piece (a
    saddle of the smooth extension; ``docs/phase2.md``, A60-10), which the
    reference puts at ``120 + D_min(3-5)``.  Anything but three critical
    values raises: the middle one would not be that saddle.
    """
    per_member: dict[str, list[float]] = {}
    for faces in members:
        values = []
        for ratio_ in height_ratios:
            critical = np.degrees(DPField.build(HexPrism.from_ratio(ratio_), faces, index, lattice_n=lattice_n).critical_values)
            if len(critical) != 3:
                raise RuntimeError(f"{path_id_of(faces)} on h/a = {ratio_}: {len(critical)} critical values {critical}, expected 3")
            values.append(float(critical[1]))
        per_member[path_id_of(faces)] = values
    all_values = np.array(list(per_member.values()))
    d_min_3_5 = float(np.degrees(DPField.build(canonical_crystal(), (3, 5), index).critical_values[0]))
    saddle = float(np.mean(all_values))
    return {
        "saddle_deg_per_member_and_h_over_a": per_member,
        "height_ratios": [float(r) for r in height_ratios],
        "saddle_deg": saddle,
        "spread_deg": float(np.ptp(all_values)),
        "reference_deg": A60_10_REFERENCE_DEG,
        "difference_from_reference_deg": saddle - A60_10_REFERENCE_DEG,
        "d_min_3_5_deg": d_min_3_5,
        "saddle_minus_120_minus_d_min_3_5_deg": saddle - 120.0 - d_min_3_5,
        "lattice_n": int(lattice_n),
    }


def tir_onset_maximum(field: DPField, lattice_n: int = LATTICE_N, starts: int = 5) -> dict[str, Any]:
    """Per internal reflection: the largest ``D_P`` on its TIR onset ``{internal_k_tir_discriminant = 0}`` in ``U_P``.

    ``R_k`` (:func:`.optics.internal_reflectance`) is ``1`` on the total
    side and ``1 - O(sqrt(-disc))`` on the partial side: continuous, with an
    unbounded normal derivative.  Where a level set of ``D_P`` leaves the
    onset curve (its ``D`` maximum) the random-orientation profile keeps
    its value and changes its slope (``int sqrt(c - b x^2) dx ~ c``): a
    corner, not a critical value of ``D_P``.  Lattice starts (the largest
    ``D`` near the curve), then SLSQP in a tangent chart with the ``U_P``
    gates as inequalities, derivatives by JAX through the field's own
    kernels.  ``constraint`` is the discriminant at the solution,
    ``min_gate_margin`` its smallest gate and ``lagrange_sine`` the sine
    between the tangent gradients of ``D_P`` and the discriminant (``0`` at
    a tangency inside ``U_P``); ``None`` for a face without a feasible onset.
    """
    faces, index = field.faces, jnp.asarray(field.index)
    slab = None if field.slab is None else jnp.asarray(field.slab)
    names = domain_margin_names(faces)
    gates = np.asarray(validity_margin_indices(faces))
    lattice = fibonacci_sphere(lattice_n)
    u = lattice[field.valid_batch(lattice)]
    d, margins = field.d_p_batch(u), field.margins_batch(u)

    def stacked(v: jax.Array) -> jax.Array:  # (D_P, every margin) at unit v
        return jnp.concatenate([d_value(v, faces, index, slab)[None], margin_vector(v, faces, index)])

    evaluate, stacked_jacobian = jax.jit(stacked), jax.jit(jax.jacfwd(stacked))
    out: dict[str, Any] = {}
    for k, name in enumerate(names):
        if not name.endswith("_tir_discriminant"):
            continue
        near = np.abs(margins[:, k]) < ONSET_START_BAND
        best = None
        for j in np.argsort(-d[near])[:starts]:
            u0 = u[near][j]
            basis = np.asarray(tangent_basis(u0))

            def point(x, u0=u0, basis=basis):
                v = u0 + x @ basis
                return v / jnp.linalg.norm(v)

            def values(x):
                return np.asarray(evaluate(point(jnp.asarray(x))))

            def jacobian(x, u0=u0, basis=basis):
                x = jnp.asarray(x)
                return np.asarray(stacked_jacobian(point(x)) @ jax.jacfwd(point)(x))

            result = minimize(
                lambda x: -values(x)[0], np.zeros(2), jac=lambda x: -jacobian(x)[0], method="SLSQP",
                constraints=[
                    {"type": "eq", "fun": lambda x, k=k: values(x)[1 + k], "jac": lambda x, k=k: jacobian(x)[1 + k]},
                    {"type": "ineq", "fun": lambda x: values(x)[1 + gates], "jac": lambda x: jacobian(x)[1 + gates]},
                ],
                options={"ftol": 1e-15, "maxiter": 500},
            )
            at = values(result.x)
            if abs(at[1 + k]) < ONSET_CONSTRAINT_ATOL and np.all(at[1 + gates] > 0.0) and (best is None or at[0] > best[0]):
                best = (float(at[0]), np.asarray(point(jnp.asarray(result.x))), at, jacobian(result.x))
        if best is None:
            out[name] = None
            continue
        value, position, at, jac = best
        g_d, g_c = jac[0], jac[1 + k]
        out[name] = {
            "delta_deg": float(np.degrees(value)),
            "position": position.tolist(),
            "constraint": float(at[1 + k]),
            "min_gate_margin": float(np.min(at[1 + gates])),
            "lagrange_sine": float(abs(g_d[0] * g_c[1] - g_d[1] * g_c[0]) / (np.linalg.norm(g_d) * np.linalg.norm(g_c))),
            "lattice_start_band": ONSET_START_BAND,
        }
    return out


def _one_sided(field: DPField, centre: float, eps: np.ndarray, store: S2EventStore, options: cq.QuadratureOptions) -> dict[str, Any]:
    """Values at ``centre +- eps`` (and at ``centre`` unless it is a critical value), one-sided slopes, gap exponent."""
    _, above, _ = _random_profile(field, centre + eps, store, options)
    _, below, _ = _random_profile(field, centre - eps, store, options)
    gap = above - below
    out = {
        "value_above": above.tolist(),
        "value_below": below.tolist(),
        "gap_exponent": float(np.polyfit(np.log(eps[1:]), np.log(np.abs(gap[1:])), 1)[0]),
        "gap_relative_at_smallest_eps": float(abs(gap[-1]) / max(abs(above[-1]), abs(below[-1]))),
    }
    if not cq.critical_delta(field, centre):
        _, (at,), _ = _random_profile(field, np.array([centre]), store, options)
        out["value_at"] = float(at)
        out["slope_above"] = ((above - at) / eps).tolist()
        out["slope_below"] = ((at - below) / eps).tolist()
    return out


def liljequist(options: LiljequistOptions = LiljequistOptions()) -> Verdict:
    """Verdict 2 (module docstring): (i) the A60-10 saddle on every ``h / a``; (ii) ``1-3-2`` / ``3-5-6-7-3`` profiles and the peak corner."""
    index = CANONICAL_REFRACTIVE_INDEX
    quadrature = cq.QuadratureOptions(relative_tolerance=options.relative_tolerance)
    a60_10 = a60_10_saddle(index, options.height_ratios, options.a60_10_members, options.a60_10_lattice_n)
    a60_10["domain"] = blocked_class_status(options.a60_10_members, index, options.lattice_n)

    paths = {(1, 3, 2): _grid(options.grid_1_3_2), (3, 5, 6, 7, 3): _grid(options.grid_3_5_6_7_3)}
    lattice = fibonacci_sphere(options.lattice_n)
    shared = {}
    fields = {faces: DPField.build(canonical_crystal(), faces, index) for faces in paths}
    both = fields[(1, 3, 2)].valid_batch(lattice) | fields[(3, 5, 6, 7, 3)].valid_batch(lattice)
    shared["max_abs_D_difference_on_either_U_P"] = float(np.max(np.abs(
        fields[(1, 3, 2)].d_p_batch(lattice[both]) - fields[(3, 5, 6, 7, 3)].d_p_batch(lattice[both]))))
    shared["fold_axes"] = {path_id_of(f): fields[f].fold.axis.tolist() for f in paths}

    per_path: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {"height_ratios": np.asarray(options.height_ratios, dtype=float)}
    notes = {"height_ratios": "h / a (a the hexagon edge) of the profile rows"}
    for faces, grid in paths.items():
        name = path_id_of(faces)
        key = name.replace("-", "_")
        rows, errors, critical, gradients, peaks, used = [], [], [], [], [], None
        for ratio_ in options.height_ratios:
            crystal = HexPrism.from_ratio(ratio_)
            field = DPField.build(crystal, faces, index)
            store = seed_store(crystal, index, faces, options.seed_store_n)
            deltas, values, results = _random_profile(field, grid, store, quadrature)
            used = deltas if used is None else used
            if not np.array_equal(used, deltas):
                raise RuntimeError(f"{name}: the profile grid lost different critical deltas on h/a = {ratio_}")
            rows.append(values)
            errors.append([r.error_estimate for r in results])
            critical.append(np.degrees(field.critical_values))
            gradients.append(focusing.gradient_norm_range(field))
            k = int(np.argmax(values))
            peaks.append({
                "height_ratio": ratio_,
                "peak_deg": float(np.degrees(deltas[k])),
                "peak_value": float(values[k]),
                "half_maximum_range_deg": _half_maximum_range(deltas, values),
                "max_relative_error_estimate": float(max((r.error_estimate / r.value for r in results if r.value > 0.0), default=0.0)),
                "exhausted_panels": int(sum(r.exhausted_panels for r in results)),
            })
        spread = float(np.max(np.ptp(np.array(critical), axis=0))) if len({len(c) for c in critical}) == 1 else None
        per_path[name] = {
            "critical_values_deg": [c.tolist() for c in critical],
            "critical_value_spread_across_h_over_a_deg": spread,
            "gradient_norm_range": gradients,
            "profiles": peaks,
        }
        arrays[f"delta_deg_{key}"] = np.degrees(used)
        arrays[f"value_{key}"] = np.array(rows)
        arrays[f"error_{key}"] = np.array(errors)
        arrays[f"critical_values_deg_{key}"] = critical[0]
        notes[f"delta_deg_{key}"] = f"deg; deviations of the {name} profile (critical values excluded)"
        notes[f"value_{key}"] = f"random-orientation pixel value of {name} per (h / a, delta)"
        notes[f"error_{key}"] = f"pixel value; error estimate of value_{key}"
        notes[f"critical_values_deg_{key}"] = f"deg; critical values of D_P for {name} (identical on every h / a)"

    # 3-5-6-7-3: the peak is the corner at the internal TIR onset (not a critical value of D_P); the one
    # boundary critical value inside (0, 180) deg is a grazing internal incidence, where w itself vanishes
    faces = (3, 5, 6, 7, 3)
    peak, onset_98, fine_rows = [], [], []
    fine = np.radians(np.arange(options.peak_window_deg[0], options.peak_window_deg[1] + 1e-9, options.peak_step_deg))
    eps = np.asarray(options.cusp_eps)
    for ratio_ in options.height_ratios:
        crystal = HexPrism.from_ratio(ratio_)
        field = DPField.build(crystal, faces, index)
        store = seed_store(crystal, index, faces, options.seed_store_n)
        onsets = tir_onset_maximum(field, options.lattice_n)
        found = [v["delta_deg"] for v in onsets.values() if v is not None]
        corner = np.radians(max(found))
        (boundary,) = [v for v in field.critical_values if EXTREMUM_ATOL < v < np.pi - EXTREMUM_ATOL]
        fine_deltas, fine_values, _ = _random_profile(field, fine, store, quadrature)
        fine_rows.append(fine_values)
        sided = _one_sided(field, corner, eps, store, quadrature)
        peak.append({
            "height_ratio": ratio_,
            "tir_onsets": onsets,
            "onset_spread_across_faces_deg": float(np.ptp(found)),
            "corner_deg": float(np.degrees(corner)),
            "corner_is_a_critical_value": cq.critical_delta(field, corner),
            **sided,
            "slope_jump": sided["slope_above"][-1] - sided["slope_below"][-1],
            "fine_peak_deg": float(np.degrees(fine_deltas[int(np.argmax(fine_values))])),
            "fine_half_maximum_range_deg": _half_maximum_range(fine_deltas, fine_values),
        })
        edge = _one_sided(field, boundary, eps, store, quadrature)
        onset_98.append({
            "height_ratio": ratio_,
            "critical_value_deg": float(np.degrees(boundary)),
            **edge,
            "relative_to_peak": float(edge["value_above"][-1] / sided["value_at"]),
        })
    arrays["delta_deg_3_5_6_7_3_peak"] = np.degrees(fine_deltas)
    arrays["value_3_5_6_7_3_peak"] = np.array(fine_rows)
    arrays["cusp_eps"] = eps
    notes["delta_deg_3_5_6_7_3_peak"] = "deg; fine deviations around the Liljequist peak"
    notes["value_3_5_6_7_3_peak"] = "random-orientation pixel value of 3-5-6-7-3 per (h / a, fine delta)"
    notes["cusp_eps"] = ("rad; offsets of the one-sided values in metadata, from the 153.07 deg TIR onset corner "
                         "(liljequist_peak) and from the 98.16 deg boundary critical value (boundary_onset)")

    numbers = {
        "a60_10": a60_10,
        "shared_field": shared,
        "paths": per_path,
        "liljequist_peak": peak,
        "boundary_onset": onset_98,
    }
    spreads = [per_path[p]["critical_value_spread_across_h_over_a_deg"] for p in per_path]
    p0 = peak[0]
    statement = (
        f"(i) A60-10 (3-5-6-7, 3-4-5-7), the 142 deg edge: its D_P saddle is {a60_10['saddle_deg']:.6f} deg on both members "
        f"and every h/a (spread {a60_10['spread_deg']:.1e} deg; reference {A60_10_REFERENCE_DEG:.6f} deg, "
        f"difference {a60_10['difference_from_reference_deg']:.1e} deg), = 120 deg + D_min(3-5) to "
        f"{a60_10['saddle_minus_120_minus_d_min_3_5_deg']:.1e} deg. (ii) 1-3-2 and 3-5-6-7-3 have one field, "
        f"D = 2 arcsin|u . n_3| (max difference {shared['max_abs_D_difference_on_either_U_P']:.1e} rad), with |grad D_P| = 2 "
        "on U_P: no fold anywhere. Their critical values are the same on every h/a (spread "
        f"{max(s for s in spreads if s is not None):.1e} deg); the profiles are the window. The Liljequist peak of 3-5-6-7-3 "
        f"sits at {p0['corner_deg']:.4f} deg for every h/a, the largest D_P on the internal TIR onsets (not a critical value "
        "of D_P): a corner, the value continuous (gap ~ eps^"
        f"{np.mean([c['gap_exponent'] for c in peak]):.2f}), the slope jumping from "
        f"{', '.join(f'{c['slope_below'][-1]:.3g}' for c in peak)} to {', '.join(f'{c['slope_above'][-1]:.3g}' for c in peak)} "
        f"per rad (h/a {', '.join(str(c['height_ratio']) for c in peak)}); its width and the profile below it change with h/a "
        f"(half maximum {', '.join(f'h/a {c['height_ratio']}: {c['fine_half_maximum_range_deg'][0]:.2f}-{c['fine_half_maximum_range_deg'][1]:.2f}' for c in peak)} deg). "
        f"The boundary critical value {onset_98[0]['critical_value_deg']:.4f} deg (grazing internal incidence) is no "
        f"visible edge: w vanishes there, the profile is continuous (gap ~ eps^{np.mean([c['gap_exponent'] for c in onset_98]):.2f}) "
        f"at {max(c['relative_to_peak'] for c in onset_98):.1e} of the peak or less. The window shapes the peak; it does not move it."
    )
    parameters = {"refractive_index": index, "options": dataclasses.asdict(options), "quadrature": quadrature.as_json()}
    return Verdict("liljequist", "measured", statement, numbers, parameters, arrays, notes)


# ---- verdict 3: the parhelic circle ----------------------------------------------------------------


@dataclass(frozen=True)
class ParhelicCircleOptions:
    height_ratio: float = 0.2
    sun_altitude_deg: float = 15.0
    plate_widths_deg: tuple[float, ...] = (0.5, 0.25)
    theta_step_deg: float = 2.0
    # ring cross-section: Gauss-Legendre nodes on +-(half_width_sigmas x sigma) of elevation
    cross_nodes: int = 48
    cross_half_width_sigmas: float = 8.0
    # theta closer than this many sigma (in ring azimuth) to a jump of the window is not compared
    jump_exclusion_sigmas: float = 12.0
    phi_samples: int = 72000
    relative_tolerance: float = 1e-8
    seed_store_n: int = SEED_STORE_N
    thetas_deg: tuple[float, ...] | None = None


def _plate_poses(phi: np.ndarray) -> np.ndarray:
    """``R_z(phi)``: a plate with its c axis exactly vertical, turned by ``phi``."""
    r = np.zeros((len(phi), 3, 3))
    r[:, 0, 0] = r[:, 1, 1] = np.cos(phi)
    r[:, 0, 1], r[:, 1, 0] = -np.sin(phi), np.sin(phi)
    r[:, 2, 2] = 1.0
    return r


def ring_window(crystal: HexPrism, index: float, sun: np.ndarray, faces: Sequence[int], phi: np.ndarray) -> dict[str, np.ndarray]:
    """The window of ``faces`` for exactly vertical plates at crystal azimuths ``phi``, and where each lands on the sky.

    ``phi`` is a periodic uniform grid on ``[0, 2 pi)``.  Returns ``w``, the
    validity, the sky elevation and the ring azimuth ``theta`` (sky azimuth
    minus the sun's, in ``(-pi, pi]``) of the image, and ``dtheta_dphi``
    (central differences of the wrapped angle).
    """
    rotations = _plate_poses(phi)
    fields = evaluate_fields(rotations, sun, crystal, index, [tuple(faces)])
    sky = -np.einsum("nij,nj->ni", rotations, fields["phi"])
    theta = _wrap(np.arctan2(sky[:, 1], sky[:, 0]) - np.arctan2(sun[1], sun[0]))
    return {
        "w": fields["w"],
        "valid": fields["valid"],
        "elevation": np.arcsin(np.clip(sky[:, 2], -1.0, 1.0)),
        "theta": theta,
        "dtheta_dphi": _wrap(np.roll(theta, -1) - np.roll(theta, 1)) / (2.0 * (phi[1] - phi[0])),
    }


def _wrap(angle: np.ndarray) -> np.ndarray:
    return np.mod(angle + np.pi, 2.0 * np.pi) - np.pi


def window_prediction(window: dict[str, np.ndarray], phi: np.ndarray, thetas: np.ndarray) -> np.ndarray:
    """``L0(theta) = sum_{phi: theta(phi) = theta} w(phi) / (2 pi |d theta / d phi|)``, the ring density with no Jacobian.

    Preimages are the sign changes of ``theta(phi) - theta`` (mod ``2 pi``)
    on the ``phi`` grid (``< 0`` against ``>= 0``), ``w`` and the derivative
    linearly interpolated there.
    """
    wrapped = np.mod(window["theta"], 2.0 * np.pi)
    out = np.zeros(len(thetas))
    for j, target in enumerate(thetas):
        s = np.mod(wrapped - target + np.pi, 2.0 * np.pi) - np.pi
        below = s < 0.0  # half open: a grid point exactly on the target is one crossing, not two
        crossing = np.nonzero((below[:-1] != below[1:]) & (np.abs(s[:-1]) < 0.5))[0]
        for i in crossing:
            t = s[i] / (s[i] - s[i + 1])
            w = (1.0 - t) * window["w"][i] + t * window["w"][i + 1]
            slope = (1.0 - t) * window["dtheta_dphi"][i] + t * window["dtheta_dphi"][i + 1]
            out[j] += w / (2.0 * np.pi * abs(slope))
    return out


def ring_cross_integral(
    geometry_for: Any, sun: np.ndarray, theta: float, elevation: float, sigma: float, options: ParhelicCircleOptions
) -> tuple[float, np.ndarray, np.ndarray]:
    """``L(theta) = int I(el, theta) cos(el) d el`` across the ring by Gauss-Legendre, and the cross profile ``(el, I)``.

    The window is ``+-cross_half_width_sigmas x sigma x 2 sin(|theta| / 2)``:
    a vertical mirror tilted by ``tau`` moves the image of the sun in
    elevation by at most ``2 tau sin(theta / 2)`` (the ring is only
    ``sqrt 2 sin(theta / 2) sigma`` wide in RMS for plates, measured by Monte
    Carlo: ``0.05 sigma`` at 4 deg), so a window fixed in ``sigma`` would
    under-resolve the cross-section near the sun.
    """
    x, weights = np.polynomial.legendre.leggauss(options.cross_nodes)
    half = options.cross_half_width_sigmas * sigma * max(2.0 * abs(np.sin(0.5 * theta)), CROSS_WIDTH_FLOOR)
    elevations = elevation + half * x
    azimuth = np.arctan2(sun[1], sun[0]) + theta
    sky = np.stack([np.cos(elevations) * np.cos(azimuth), np.cos(elevations) * np.sin(azimuth), np.sin(elevations)], axis=1)
    deltas = np.arccos(np.clip(sky @ sun, -1.0, 1.0))
    values = geometry_for(deltas, -sky)
    return float(np.sum(half * weights * values * np.cos(elevations))), elevations, values


def parhelic_circle(options: ParhelicCircleOptions = ParhelicCircleOptions()) -> Verdict:
    """Verdict 3 (module docstring): ``1-3-2`` under plates against its window-only ring prediction."""
    faces = (1, 3, 2)
    crystal, index = HexPrism.from_ratio(options.height_ratio), CANONICAL_REFRACTIVE_INDEX
    sun = sun_direction(options.sun_altitude_deg, 0.0)
    field = DPField.build(crystal, faces, index)
    store = seed_store(crystal, index, faces, options.seed_store_n)
    label = focusing.classify(crystal, faces, build_pose_density("random"), index, field=field)

    phi = np.linspace(0.0, 2.0 * np.pi, options.phi_samples, endpoint=False)
    window = ring_window(crystal, index, sun, faces, phi)
    lit = window["w"] > 0.0
    interior = lit & np.roll(lit, 1) & np.roll(lit, -1)
    elevation_error = float(np.max(np.abs(window["elevation"][lit] - np.radians(options.sun_altitude_deg))))
    slope_range = (float(window["dtheta_dphi"][interior].min()), float(window["dtheta_dphi"][interior].max()))
    # the six prism members' windows are one function shifted by 60 deg
    step = options.phi_samples // 6
    shifts = {}
    for k in range(4, 9):
        other = ring_window(crystal, index, sun, (1, k, 2), phi)["w"]
        # the crystal is invariant under 60 deg turns: face k at phi is face 3 at phi + 60 (k - 3) deg
        shifts[f"1-{k}-2"] = float(np.max(np.abs(other - np.roll(window["w"], -(k - 3) * step))))
    # jumps of w along phi (the internal-TIR-only gate switching a lit window on or off)
    jump = np.abs(np.diff(np.r_[window["w"], window["w"][0]])) > 1e-3 * window["w"].max()
    jump_thetas = np.mod(window["theta"][np.nonzero(jump)[0]], 2.0 * np.pi)

    lit_thetas = np.mod(window["theta"][lit], 2.0 * np.pi)
    lit_thetas = np.where(lit_thetas > np.pi, lit_thetas - 2.0 * np.pi, lit_thetas)
    if options.thetas_deg is None:
        lo, hi = np.degrees(np.abs(lit_thetas).min()), np.degrees(np.abs(lit_thetas).max())
        thetas = np.radians(np.arange(np.floor(lo) - 4.0, min(np.ceil(hi) + 4.0, 180.0) + 1e-9, options.theta_step_deg))
    else:
        thetas = np.radians(np.asarray(options.thetas_deg, dtype=float))
    prediction = window_prediction(window, phi, thetas)
    quadrature_by_width, rows, residual_stats = {}, [], []
    elevation = np.radians(options.sun_altitude_deg)
    for width in options.plate_widths_deg:
        sigma = np.radians(width)
        density = build_pose_density("plate", zenith_std_deg=width)
        quadrature = cq.QuadratureOptions(relative_tolerance=options.relative_tolerance, initial_panel_rad=sigma / 4.0)
        quadrature_by_width[width] = quadrature.as_json()

        def values_at(deltas: np.ndarray, centres: np.ndarray) -> np.ndarray:
            geometry = level_set_geometry(field, deltas, store, quadrature)
            return np.array([r.value for r in geometry.integrate(sun, centres, density)])

        ring = np.array([ring_cross_integral(values_at, sun, t, elevation, sigma, options)[0] for t in thetas])
        rows.append(ring)
        near_jump = np.array([np.min(np.abs(np.mod(jump_thetas - t + np.pi, 2 * np.pi) - np.pi)) if len(jump_thetas) else np.inf
                              for t in thetas]) < options.jump_exclusion_sigmas * sigma
        compare = (prediction > 0.0) & ~near_jump
        relative = ring[compare] / prediction[compare] - 1.0
        residual_stats.append({
            "plate_zenith_std_deg": width,
            "compared_thetas": int(compare.sum()),
            "max_abs_relative_residual": float(np.max(np.abs(relative))) if relative.size else None,
            "median_abs_relative_residual": float(np.median(np.abs(relative))) if relative.size else None,
            "dark_where_predicted_dark": float(np.max(np.abs(ring[prediction == 0.0]))) if np.any(prediction == 0.0) else None,
        })
    order = None
    if len(residual_stats) >= 2 and all(r["max_abs_relative_residual"] for r in residual_stats[:2]):
        order = float(np.log(residual_stats[0]["max_abs_relative_residual"] / residual_stats[1]["max_abs_relative_residual"])
                      / np.log(options.plate_widths_deg[0] / options.plate_widths_deg[1]))

    jumps_deg = np.degrees(np.unique(np.round(jump_thetas, 6))).tolist()
    numbers = {
        "path": path_id_of(faces),
        "focusing": label.as_json(),
        "fold_axis": field.fold.axis.tolist(),
        "image_elevation_max_abs_error_rad": elevation_error,
        "dtheta_dphi_range": slope_range,
        "member_window_shift_max_abs_difference": shifts,
        "window_jump_ring_azimuths_deg": jumps_deg,
        "ring_vs_window": residual_stats,
        "residual_order_in_sigma": order,
    }
    worst = residual_stats[-1]
    statement = (
        f"No fold: 1-3-2 is a mirror slab, |grad D_P| in [{label.gradient_norm_range[0]:.15f}, {label.gradient_norm_range[1]:.15f}] "
        f"on U_P, no Jacobian-focusing critical value. For exactly vertical plates the image stays at the sun's elevation "
        f"(to {elevation_error:.1e} rad) and its ring azimuth moves at d theta / d phi = 2 (range {slope_range[0]:.9f}-{slope_range[1]:.9f}). "
        f"The elevation-integrated ring brightness from the contour quadrature under plates equals the window-only prediction "
        f"sum w / (2 pi x 2) to {worst['max_abs_relative_residual']:.1e} (max over {worst['compared_thetas']} ring azimuths at "
        f"sigma = {worst['plate_zenith_std_deg']} deg), shrinking as sigma^{'?' if order is None else f'{order:.2f}'}; "
        f"within {options.jump_exclusion_sigmas:g} sigma "
        f"of a jump of the window (the TIR-only gate, ring azimuth {', '.join(f'{t:.2f}' for t in jumps_deg if t <= 180.0)} deg) the "
        "ring is that jump smoothed by the plate width and is not compared. The six prism members' windows are one function shifted by 60 deg "
        f"(max difference {max(shifts.values()):.1e}), so under uniform plate azimuth every member draws the same ring."
    )
    arrays = {
        "theta_deg": np.degrees(thetas),
        "window_prediction": prediction,
        "ring_contour": np.array(rows),
        "plate_widths_deg": np.asarray(options.plate_widths_deg, dtype=float),
        "phi_deg": np.degrees(phi),
        "window_w": window["w"],
        "window_theta_deg": np.degrees(window["theta"]),
    }
    notes = {
        "theta_deg": "deg; ring azimuth (sky azimuth minus the sun's)",
        "window_prediction": "per radian of ring azimuth; sum_phi w / (2 pi |d theta / d phi|), the plate limit",
        "ring_contour": "per radian of ring azimuth; int I cos(el) d el from the contour quadrature, rows per plate width",
        "plate_widths_deg": "deg; plate zenith standard deviations, rows of ring_contour",
        "phi_deg": "deg; crystal azimuth of the exactly vertical plate",
        "window_w": "length^2; w = A_P T_P of 1-3-2 at phi_deg",
        "window_theta_deg": "deg; ring azimuth of the image at phi_deg (unwrapped)",
    }
    parameters = {
        "crystal": {"type": "hexagonal_prism", "height_ratio": options.height_ratio},
        "refractive_index": index,
        "sun_direction": sun.tolist(),
        "options": dataclasses.asdict(options),
        "quadrature": quadrature_by_width,
    }
    return Verdict("parhelic-circle", "measured", statement, numbers, parameters, arrays, notes)


# ---- verdict 4: parallel-face classes and the two kinds of focusing ---------------------------------


@dataclass(frozen=True)
class ParallelFaceOptions:
    paths: tuple[tuple[int, ...], ...] = ((3, 5), (1, 3, 2), (3, 5, 6, 7, 3), (3, 1, 6), (1, 3, 5, 2), (3, 6))
    families: tuple[tuple[str, Mapping[str, float]], ...] = (
        ("random", {}),
        ("column", {"zenith_std_deg": 0.5}),
        ("plate", {"zenith_std_deg": 0.5}),
        ("parry", {"zenith_std_deg": 0.5, "roll_std_deg": 1.0}),
        ("lowitz", {"zenith_std_deg": 0.5, "roll_std_deg": 1.0}),
    )
    # the collapse demonstration: 1-3-2 under plates across the parhelic circle at this ring azimuth
    demo_theta_deg: float = 110.0
    demo_widths_deg: tuple[float, ...] = (0.5, 0.25)
    height_ratio: float = 0.2
    sun_altitude_deg: float = 15.0
    cross_nodes: int = 48
    cross_half_width_sigmas: float = 8.0
    relative_tolerance: float = 1e-8
    seed_store_n: int = SEED_STORE_N


def parallel_face(options: ParallelFaceOptions = ParallelFaceOptions()) -> Verdict:
    """Verdict 4 (module docstring): focusing labels over fixtures x families, and the collapse signature of the ring."""
    index = CANONICAL_REFRACTIVE_INDEX
    crystal = canonical_crystal()
    table = []
    for faces in options.paths:
        field = DPField.build(crystal, faces, index) if halo_map_rank(crystal, faces) else None
        for family, parameters in options.families:
            density = build_pose_density(family, **dict(parameters))
            entry = focusing.classify(crystal, faces, density, index, field=field).as_json()
            entry["family"] = family
            entry["family_parameters"] = dict(parameters)
            table.append(entry)

    # dimension collapse on a parallel-face class: the ring cross profile of 1-3-2, peak ~ 1/sigma, integral fixed
    faces = (1, 3, 2)
    plate_crystal = HexPrism.from_ratio(options.height_ratio)
    field = DPField.build(plate_crystal, faces, index)
    store = seed_store(plate_crystal, index, faces, options.seed_store_n)
    sun = sun_direction(options.sun_altitude_deg, 0.0)
    theta, elevation = np.radians(options.demo_theta_deg), np.radians(options.sun_altitude_deg)
    ring_options = ParhelicCircleOptions(cross_nodes=options.cross_nodes, cross_half_width_sigmas=options.cross_half_width_sigmas)
    demo, profiles, random_values = [], [], None
    for width in options.demo_widths_deg:
        sigma = np.radians(width)
        quadrature = cq.QuadratureOptions(relative_tolerance=options.relative_tolerance, initial_panel_rad=sigma / 4.0)
        density = build_pose_density("plate", zenith_std_deg=width)
        random = build_pose_density("random")
        seen: dict[str, np.ndarray] = {}

        def values_at(deltas: np.ndarray, centres: np.ndarray) -> np.ndarray:
            geometry = level_set_geometry(field, deltas, store, quadrature)
            seen["random"] = np.array([r.value for r in geometry.integrate(sun, centres, random)])
            return np.array([r.value for r in geometry.integrate(sun, centres, density)])

        integral, elevations, values = ring_cross_integral(values_at, sun, theta, elevation, sigma, ring_options)
        profiles.append((np.degrees(elevations - elevation), values, seen["random"]))
        demo.append({
            "plate_zenith_std_deg": width,
            "cross_integral": integral,
            "peak_value": float(values.max()),
            "peak_times_sigma_rad": float(values.max() * sigma),
            "random_value_range": [float(seen["random"].min()), float(seen["random"].max())],
        })
    arrays = {
        "demo_widths_deg": np.asarray(options.demo_widths_deg, dtype=float),
        "demo_elevation_offset_deg": np.array([p[0] for p in profiles]),
        "demo_value_plate": np.array([p[1] for p in profiles]),
        "demo_value_random": np.array([p[2] for p in profiles]),
    }
    notes = {
        "demo_widths_deg": "deg; plate zenith standard deviations, rows of the demo arrays",
        "demo_elevation_offset_deg": "deg; sky elevation minus the sun's across the parhelic circle, per width",
        "demo_value_plate": "pixel value I of 1-3-2 under plates across the ring, per width",
        "demo_value_random": "pixel value I of 1-3-2 for random orientation at the same pixels, per width",
    }
    mechanisms = sorted({(row["path"], row["family"], row["mechanism"]) for row in table})
    jacobian_paths = sorted({row["path"] for row in table if row["jacobian_focusing"]})
    gain = demo[-1]["peak_value"] / demo[0]["peak_value"]
    narrowing = options.demo_widths_deg[0] / options.demo_widths_deg[-1]
    statement = (
        "Explicit output (lumice_integral.focusing): Jacobian focusing is read off the D_P critical set, dimension collapse off "
        "the density's confined dimensions. On the fixtures "
        f"{', '.join(path_id_of(f) for f in options.paths)} "
        f"{'no critical value focuses' if not jacobian_paths else 'Jacobian focusing on ' + ', '.join(jacobian_paths)}: 3-5 has a finite jump, the "
        "parallel-face (wedge 0, M != I) slabs have |grad D_P| bounded away from 0 (cone points, creases and boundary cusps; "
        "the rotation slab 1-3-5-2 keeps its fold circle outside U_P), and 3-6 is a point mass. Every sharp image of these "
        f"classes is dimension collapse: across the parhelic circle (1-3-2, plates) narrowing sigma {narrowing:g}-fold multiplies "
        f"the peak by {gain:.3f} while the cross integral stays at {demo[0]['cross_integral']:.6g} / {demo[-1]['cross_integral']:.6g}, and the "
        f"random-orientation value at the same pixels is smooth ({demo[-1]['random_value_range'][0]:.4g}-{demo[-1]['random_value_range'][1]:.4g})."
    )
    numbers = {
        "table": table,
        "mechanisms": [list(m) for m in mechanisms],
        "paths_with_jacobian_focusing": jacobian_paths,
        "collapse_demo": {"path": "1-3-2", "theta_deg": options.demo_theta_deg, "widths": demo},
    }
    parameters = {
        "crystal": {"type": "hexagonal_prism", "height_ratio_table": CANONICAL_HEIGHT_RATIO, "height_ratio_demo": options.height_ratio},
        "refractive_index": index,
        "options": dataclasses.asdict(options),
    }
    return Verdict("parallel-face", "measured", statement, numbers, parameters, arrays, notes)


RUNNERS = {"inner-edge": inner_edge, "liljequist": liljequist, "parhelic-circle": parhelic_circle, "parallel-face": parallel_face}
