"""Colour edges and tints of a halo path: the weight-kink criterion (task ``chromatic-weight-kink-diagnostic``).

A path's colour comes from what changes with the refractive index ``n``
between a red and a blue wavelength (``N_RED`` / ``N_BLUE``, the ends of
Lumice's wavelength pool).  Three things can:

- the direction map (``dD_P/dn``): the ordinary dispersion of refraction
  halos, reported here as ``direction_dispersion`` but not a mechanism of
  its own;
- a **weight kink** ``C_k`` (:mod:`.dp_field.weight_kink`): the TIR onset
  of internal reflection ``k`` moves with ``n``; ``d disc_k / dn > 0``
  means the blue wavelength reflects totally on a larger set, so the kink
  can only push energy toward blue (asserted per point, not assumed);
- a **gate** of ``U_P`` that moves with ``n`` (the exit Snell
  discriminant, or an internal incidence cosine whose normal is not
  orthogonal to the entry face): either sign.  The transmission falls
  continuously to 0 there and ``D_P`` is Hoelder-1/2 across the exit TIR
  curve.

Random orientation (the image is a function of ``delta = D`` alone): a line
of ``S^2`` makes an edge of the ``delta`` profile only where its image in
``D`` is concentrated.  With ``Delta`` the pointwise shift of the line's
``D`` image between the two indices and ``sigma`` its width ``max - min``
at one index, a **colour edge** is ``sigma <= EDGE_SPREAD_PER_SHIFT |Delta|``
and ``|Delta| >= EDGE_MIN_SHIFT_RAD`` on a line that carries weight
(``lit_fraction > 0``).  A single-mirror slab (``3-1-6``, ``1-3-2``) has
``sigma = 0``: the whole kink sits on ``D = 2 arcsin sqrt(n^2 - 1)`` and
makes the blue rim of the dark hole around the antisolar point.  A line
with ``sigma > |Delta|`` still gives a slope corner of the profile at its
``D`` extrema (``ch10_verdicts.tir_onset_maximum``), not a step: its colour
is spread over ``sigma`` and reported as not visible.

Oriented crystals (a plate family): when every pose of a path class lands
near one sky point *at both indices* (the weighted red-blue angle of the
same pose, ``direction_dispersion``, below ``EDGE_MIN_SHIFT_RAD``: a slab-like
class such as ``1-3-5-2`` or ``1-3-4-2`` on a plate), the colour is a
**tint** of the whole spot, measured by
the ratio of the blue to the red weighted power ``sum A T`` over the class
(entry measure ``A`` of :func:`.geometry.entry_measure.entry_measure_batch`,
path power ``T`` of :func:`.optics.fresnel_transmission_path_batch`, the
internal ``R_k`` included) on a sample of the family.  A class whose
direction disperses (``3-5``, ``1-3``) spreads its colours like a spectrum
instead; its power ratio is reported but gives no tint verdict (``kind =
none``).  The sky position of
the kink is not traced for oriented crystals (the task's scope).

Both weights go through :func:`weighted_power`, the one ``A T`` kernel of
this module.  Path classes are Lumice's filter orbits (``PBD``, L1:
:func:`.symmetry.reflection_group.pbd_orbit`), never the literal sequence
(``3-5-6-8`` is geometrically impossible on the rhombic plate
``[1.5, 1, 1, 1.5, 1, 1]`` while four members of its class are lit).
Member feasibility is decided per index by that kernel on the family's
sample, not by the ``n = 1.31`` gates of :mod:`.geometry.feasibility`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.spatial import cKDTree

from . import optics
from .camera import incident_direction_from_sun, sun_direction
from .dp_field import DPField, KinkCurve
from .geometry import HexPrism, Polyhedron, halo_map_rank
from .geometry.entry_measure import entry_measure_batch
from .path_class import g_true_orbit
from .s2_store import align_rotations, fibonacci_sphere
from .so3 import haar_rotations
from .symmetry.reflection_group import pbd_orbit

Faces = tuple[int, ...]

# Red and blue ends of Lumice's wavelength pool, the index pair of every verdict (parameters, not constants of
# the criterion).
N_RED = 1.307
N_BLUE = 1.317

# A colour edge must be displaced by at least the sun's angular diameter: the solar disc smears every sky
# feature over ~0.5 deg, so a smaller red/blue shift is blurred into its own edge.
EDGE_MIN_SHIFT_RAD = float(np.radians(0.5))
# ... and the line's own spread in D must not exceed that shift: with sigma > |Delta| the red and blue images of
# the line overlap more than they separate and the step becomes a gradual slope change (module docstring).
EDGE_SPREAD_PER_SHIFT = 1.0
# A plate tint is reported when blue / red weighted power leaves [1 / TINT_RATIO_MIN, TINT_RATIO_MIN].  Frozen on
# calibration classes that are not acceptance fixtures (tests/test_chromatic.py::test_tint_threshold_calibration):
# classes whose internal reflections do not switch between total and partial across the index pair stay within
# CALIBRATION_WHITE_MAX_DEVIATION of 1 (plain Fresnel dispersion of entry and exit), the threshold is twice that.
CALIBRATION_WHITE_MAX_DEVIATION = 0.05
TINT_RATIO_MIN = 1.0 + 2.0 * CALIBRATION_WHITE_MAX_DEVIATION
# A gate that does not move with n (its margin at the other index vanishes on the curve to this) is no colour source.
GATE_STATIC_ATOL = 1e-9
# Fibonacci lattice of the weight median inside U_P (gate contrast reference).
CONTRAST_LATTICE_N = 20000
# Probe sun of the random-orientation weights: A T of a body direction u is taken at a pose with R u = s_hat.
_PROBE_SUN = np.array([0.0, 0.0, 1.0])


# ---- the weight kernel ------------------------------------------------------------------------------------


def weighted_power(
    rotations: np.ndarray, faces: Sequence[int], incident_direction: np.ndarray, index: float, *, crystal: Polyhedron
) -> np.ndarray:
    """``A T`` per pose: entry measure (at the same ``n``) times the path's power, ``0`` outside the domain.

    The one weight of this module: plate tints sum it over a sample, random
    orientation edges read it on a line of ``S^2``.  ``T`` includes every
    internal ``R_k``.
    """
    faces = optics.normalize_faces(faces, crystal)
    area = entry_measure_batch(rotations, faces, incident_direction, crystal, n_ice=float(index))
    power = optics.fresnel_transmission_path_batch(rotations, faces, incident_direction, float(index), crystal=crystal)
    return area * power


def _body_weight(points: np.ndarray, faces: Faces, index: float, crystal: Polyhedron) -> np.ndarray:
    """:func:`weighted_power` of body directions ``u`` (toward the sun), posed with ``R u = s_hat`` (``A`` is twist invariant)."""
    if len(points) == 0:
        return np.zeros(0)
    rotations = align_rotations(np.asarray(points, dtype=np.float64), _PROBE_SUN)
    return weighted_power(rotations, faces, incident_direction_from_sun(_PROBE_SUN), index, crystal=crystal)


# ---- random orientation: features of one path -----------------------------------------------------------


@dataclass(frozen=True)
class ChromaticFeature:
    """One colour source of a path under random orientation (module docstring for the criterion).

    ``kind`` is ``"edge"`` (a weight kink ``C_k``) or ``"gate_edge"`` (a gate
    of ``U_P`` that moves with ``n``); ``source`` its margin name.
    ``color`` is ``"blue"`` or ``"red"`` from the sign of ``d margin / dn``
    on the line (``positive_fraction`` of its points have ``> 0``: blue
    reflects totally / passes the gate on a larger set).  Angles in
    radians: ``delta_red`` / ``delta_blue`` the median ``D`` of the line at
    each index (its ``delta``), ``shift`` the median pointwise ``Delta``,
    ``spread`` ``sigma`` (the larger of the two indices),
    ``direction_dispersion`` the median ``|dD_P/dn| (N_BLUE - N_RED)`` on
    the line (finite points only: on the exit TIR curve ``dD_P/dn`` diverges
    and a gate's value comes from its sampling off the curve).  ``contrast``: for an edge ``1 - R`` of the disfavoured
    colour on the favoured colour's kink (the fringe's depth), for a gate
    the favoured colour's weight on the disfavoured colour's gate over its
    median weight in ``U_P``.  ``weight`` is the median ``A T`` of the
    favoured colour on its line, ``lit_fraction`` the fraction of its points
    with ``A T > 0``.
    """

    kind: str
    source: str
    color: str
    positive_fraction: float
    delta_red: float
    delta_blue: float
    shift: float
    spread: float
    direction_dispersion: float
    contrast: float
    weight: float
    lit_fraction: float
    visible: bool

    @property
    def score(self) -> float:
        """``|Delta| / sigma``: how far the line is from the edge threshold (``inf`` at ``sigma = 0``)."""
        return abs(self.shift) / self.spread if self.spread > 0.0 else float("inf")


@dataclass(frozen=True)
class ChromaticVerdict:
    """The colour verdict of one face sequence or of a path class.

    ``kind`` in ``{"edge", "gate_edge", "tint", "unresolved", "none"}``; ``color`` in
    ``{"blue", "red", "white", "none"}``: a random-orientation verdict takes
    both from its dominant feature (visible features first, then the largest
    :attr:`ChromaticFeature.score`) and ``visible`` from it; ``none`` /
    ``none`` without features.  ``unresolved`` / ``none``: no assessed feature, but a kink or
    gate exists at one index only (an onset between ``n_red`` and ``n_blue``, the strongest colour
    shape, which the two-index metrics cannot measure); the notes name it.  A plate verdict is ``tint`` with the colour
    of the ratio, ``none`` / ``white`` inside the threshold, or ``none`` /
    ``none`` for a class that disperses (a note says so).  ``position`` is ``delta`` of the
    dominant feature (radians, the blue one's for blue); ``None`` for a
    tint (its sky point is not traced).
    """

    faces: Faces
    kind: str
    color: str
    visible: bool
    position: float | None
    features: tuple[ChromaticFeature, ...] = ()
    tint: TintMetrics | None = None
    notes: tuple[str, ...] = ()
    n_red: float = N_RED
    n_blue: float = N_BLUE


def diagnose(
    crystal: Polyhedron,
    faces: Sequence[int],
    *,
    n_red: float = N_RED,
    n_blue: float = N_BLUE,
    lattice_n: int = 20000,
) -> ChromaticVerdict:
    """The random-orientation colour verdict of one face sequence: every weight kink and every moving gate.

    ``ValueError`` for a rank-0 path (:meth:`.dp_field.DPField.build`).
    """
    faces = optics.normalize_faces(faces, crystal)
    red = DPField.build(crystal, faces, n_red, lattice_n=lattice_n)
    blue = DPField.build(crystal, faces, n_blue, lattice_n=lattice_n)
    notes: list[str] = []
    unresolved: list[str] = []
    features: list[ChromaticFeature] = []
    for kink_red, kink_blue in zip(red.weight_kinks, blue.weight_kinks, strict=True):
        for kink in (kink_red, kink_blue):
            if kink.note:
                notes.append(f"{kink.margin} at n = {kink.index}: {kink.note}")
        if kink_red.arcs and kink_blue.arcs:
            features.append(_kink_feature(red, blue, kink_red, kink_blue))
        elif kink_red.arcs or kink_blue.arcs:
            present = kink_red if kink_red.arcs else kink_blue
            unresolved.append(f"{present.margin}: weight kink at n = {present.index} only (not assessed)")
    try:
        features.extend(_gate_features(red, blue, unresolved))
    except (RuntimeError, ValueError) as error:  # a boundary walk that fails is reported, not hidden
        notes.append(f"gates not analysed: {error}")
    return _verdict(faces, tuple(features), tuple(notes) + tuple(unresolved), n_red, n_blue, bool(unresolved))


def _verdict(
    faces: Faces,
    features: tuple[ChromaticFeature, ...],
    notes: tuple[str, ...],
    n_red: float,
    n_blue: float,
    unresolved: bool = False,
) -> ChromaticVerdict:
    if not features:
        kind = "unresolved" if unresolved else "none"
        return ChromaticVerdict(faces, kind, "none", False, None, (), None, notes, n_red, n_blue)
    top = _dominant(features)
    position = top.delta_blue if top.color == "blue" else top.delta_red
    return ChromaticVerdict(faces, top.kind, top.color, top.visible, position, features, None, notes, n_red, n_blue)


def _dominant(features: Sequence[ChromaticFeature]) -> ChromaticFeature:
    return max(features, key=lambda f: (f.visible, f.score))


def _shift(red_points: np.ndarray, red_values: np.ndarray, blue_points: np.ndarray, blue_values: np.ndarray) -> float:
    """Median over the blue line of ``D_blue(u) - D_red(nearest red point)``."""
    _, nearest = cKDTree(red_points).query(blue_points)
    return float(np.median(blue_values - red_values[nearest]))


def _is_visible(shift: float, spread: float, lit_fraction: float) -> bool:
    return abs(shift) >= EDGE_MIN_SHIFT_RAD and spread <= EDGE_SPREAD_PER_SHIFT * abs(shift) and lit_fraction > 0.0


def _kink_feature(red: DPField, blue: DPField, kink_red: KinkCurve, kink_blue: KinkCurve) -> ChromaticFeature:
    k = optics.domain_margin_names(red.faces).index(kink_red.margin)
    cosine = k - 1  # internal_{j}_incidence_cosine precedes its discriminant
    d_dn, margin_dn = red.index_derivatives_batch(kink_red.points)
    positive = float(np.mean(margin_dn[:, k] > 0.0))
    color = "blue" if positive >= 0.5 else "red"
    favoured, disfavoured = (blue, red) if color == "blue" else (red, blue)
    favoured_kink = kink_blue if color == "blue" else kink_red
    other = disfavoured.margins_batch(favoured_kink.points)
    reflectance = optics.internal_reflectance(disfavoured.index, other[:, cosine], other[:, k])
    weight = _body_weight(favoured_kink.points, favoured.faces, favoured.index, favoured.crystal)
    shift = _shift(kink_red.points, kink_red.values, kink_blue.points, kink_blue.values)
    spread = max(kink_red.spread, kink_blue.spread)
    lit = float(np.mean(weight > 0.0))
    return ChromaticFeature(
        "edge",
        kink_red.margin,
        color,
        positive,
        float(np.median(kink_red.values)),
        float(np.median(kink_blue.values)),
        shift,
        spread,
        float(np.median(np.abs(d_dn))) * abs(blue.index - red.index),
        float(1.0 - np.median(reflectance)),
        float(np.median(weight)),
        lit,
        _is_visible(shift, spread, lit),
    )


def _gate_features(red: DPField, blue: DPField, unresolved: list[str]) -> list[ChromaticFeature]:
    """One feature per gate that bounds ``U_P`` at both indices and moves with ``n``.

    A gate that bounds ``U_P`` at one index only is appended to ``unresolved`` (not assessed).
    """
    names = optics.domain_margin_names(red.faces)
    out = []
    pieces_red = _pieces_by_margin(red)
    pieces_blue = _pieces_by_margin(blue)
    for margin in sorted(set(pieces_red) ^ set(pieces_blue), key=names.index):
        index = red.index if margin in pieces_red else blue.index
        unresolved.append(f"{margin}: bounds U_P at n = {index} only (not assessed)")
    for margin in sorted(set(pieces_red) & set(pieces_blue), key=names.index):
        k = names.index(margin)
        points_red, values_red = pieces_red[margin]
        points_blue, values_blue = pieces_blue[margin]
        if np.max(np.abs(blue.margins_batch(points_red)[:, k])) <= GATE_STATIC_ATOL:
            continue
        d_dn, margin_dn = red.index_derivatives_batch(points_red)
        positive = float(np.mean(margin_dn[:, k] > 0.0))
        color = "blue" if positive >= 0.5 else "red"
        favoured, disfavoured_points, favoured_points = (
            (blue, points_red, points_blue) if color == "blue" else (red, points_blue, points_red)
        )
        reference = _median_weight_inside(favoured)
        across = _body_weight(disfavoured_points, favoured.faces, favoured.index, favoured.crystal)
        weight = _body_weight(favoured_points, favoured.faces, favoured.index, favoured.crystal)
        shift = _shift(points_red, values_red, points_blue, values_blue)
        spread = max(float(np.ptp(values_red)), float(np.ptp(values_blue)))
        lit = float(np.mean(weight > 0.0))
        out.append(
            ChromaticFeature(
                "gate_edge",
                margin,
                color,
                positive,
                float(np.median(values_red)),
                float(np.median(values_blue)),
                shift,
                spread,
                _finite_median(np.abs(d_dn)) * abs(blue.index - red.index),
                float(np.median(across) / reference) if reference > 0.0 else 0.0,
                float(np.median(weight)),
                lit,
                _is_visible(shift, spread, lit),
            )
        )
    return out


def _finite_median(values: np.ndarray) -> float:
    """Median of the finite entries: ``dD_P/dn`` is infinite on the exit TIR curve itself (``D_P`` is Hoelder-1/2 there)."""
    finite = values[np.isfinite(values)]
    return float(np.median(finite)) if len(finite) else float("nan")


def _pieces_by_margin(field_: DPField) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    out: dict[str, tuple[list[np.ndarray], list[np.ndarray]]] = {}
    for piece in field_.boundary_curves:
        points, values = out.setdefault(piece.margin, ([], []))
        finite = np.isfinite(piece.values)
        points.append(piece.points[finite])
        values.append(piece.values[finite])
    return {m: (np.concatenate(p), np.concatenate(v)) for m, (p, v) in out.items() if sum(len(x) for x in p) > 0}


def _median_weight_inside(field_: DPField) -> float:
    lattice = fibonacci_sphere(CONTRAST_LATTICE_N)
    inside = lattice[field_.valid_batch(lattice)]
    weight = _body_weight(inside, field_.faces, field_.index, field_.crystal)
    lit = weight[weight > 0.0]
    return float(np.median(lit)) if len(lit) else 0.0


# ---- families and path classes ------------------------------------------------------------------------------


@dataclass(frozen=True)
class RandomOrientation:
    """Random (Haar) orientation.  ``samples`` Haar poses (``seed``) decide which class members are lit at each index."""

    samples: int = 20000
    seed: int = 0


@dataclass(frozen=True)
class PlateFamily:
    """A plate family: c axis tilted from the zenith by ``|N(0, zenith_std_deg)|`` in a uniform direction, uniform spin.

    Pose ``R = tilt * Rz(spin)`` (body to world), sun at ``sun_altitude_deg``
    and azimuth 180 deg; ``samples`` poses from ``numpy.random.default_rng(seed)``
    drawn as spin, tilt, tilt direction (the sampler of the task's
    ``probe_120_cls.py``).  A half-normal tilt with a uniform tilt direction,
    not :class:`.pose_density.ZenithGaussianPoseDensity`'s density on the
    sphere; at ``1`` deg the two differ far below the tint threshold.
    """

    sun_altitude_deg: float
    zenith_std_deg: float = 1.0
    samples: int = 100_000
    seed: int = 3

    def poses(self) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        spin = rng.uniform(0.0, 2.0 * np.pi, self.samples)
        tilt = np.abs(rng.normal(0.0, np.radians(self.zenith_std_deg), self.samples))
        toward = rng.uniform(0.0, 2.0 * np.pi, self.samples)
        return _rotvec_matrices(np.stack([-np.sin(toward) * tilt, np.cos(toward) * tilt, np.zeros_like(tilt)], axis=1)) @ _rz(spin)

    def incident_direction(self) -> np.ndarray:
        return incident_direction_from_sun(sun_direction(self.sun_altitude_deg, 180.0))


def _rz(angle: np.ndarray) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    out = np.zeros((len(angle), 3, 3))
    out[:, 0, 0], out[:, 0, 1], out[:, 1, 0], out[:, 1, 1], out[:, 2, 2] = c, -s, s, c, 1.0
    return out


def _rotvec_matrices(rotvec: np.ndarray) -> np.ndarray:
    """Rodrigues' formula per row (the rotation by ``|v|`` about ``v / |v|``; identity at ``v = 0``)."""
    angle = np.linalg.norm(rotvec, axis=1)
    axis = rotvec / np.where(angle > 0.0, angle, 1.0)[:, None]
    k = np.zeros((len(rotvec), 3, 3))
    k[:, 0, 1], k[:, 0, 2], k[:, 1, 2] = -axis[:, 2], axis[:, 1], -axis[:, 0]
    k -= np.transpose(k, (0, 2, 1))
    s, c = np.sin(angle)[:, None, None], (1.0 - np.cos(angle))[:, None, None]
    return np.eye(3) + s * k + c * (k @ k)


@dataclass(frozen=True)
class TintMetrics:
    """The weighted power of a class on a family sample, per index (mean over the sample of ``sum_members A T``).

    ``ratio`` is blue / red; ``tir_fraction`` per index the ``A T``-weighted
    fraction of internal reflections that are total (averaged over the
    reflections of each lit member); ``direction_dispersion`` the weighted
    mean angle (radians) between the red and blue outgoing directions of the
    same pose, over poses lit at both indices (``0`` for a path whose
    direction does not disperse).
    """

    energy_red: float
    energy_blue: float
    ratio: float
    tir_fraction_red: float
    tir_fraction_blue: float
    direction_dispersion: float


@dataclass(frozen=True)
class ClassVerdict:
    """:class:`ChromaticVerdict` of a path class plus its members.

    ``members``: the ``PBD`` orbit of ``representative`` (sorted);
    ``lit_members``: per index (``{"red": ..., "blue": ...}``) the members
    with positive weighted power on the family sample (a sampled verdict, not
    a certificate); ``member_verdicts``: random orientation only, one per
    ``G_true`` orbit of lit members (members related by the crystal's own
    symmetry have congruent fields).
    """

    representative: Faces
    members: tuple[Faces, ...]
    lit_members: dict[str, tuple[Faces, ...]]
    verdict: ChromaticVerdict
    member_verdicts: tuple[ChromaticVerdict, ...] = field(default=())


def class_members(representative: Sequence[int]) -> tuple[Faces, ...]:
    """Lumice's ``PBD`` filter orbit of ``representative`` (L1, :func:`.symmetry.reflection_group.pbd_orbit`)."""
    return tuple(sorted(pbd_orbit(representative)))


def diagnose_class(
    crystal: Polyhedron,
    representative: Sequence[int],
    family: RandomOrientation | PlateFamily,
    *,
    n_red: float = N_RED,
    n_blue: float = N_BLUE,
    lattice_n: int = 20000,
) -> ClassVerdict:
    """The colour verdict of the ``PBD`` class of ``representative`` under ``family`` (module docstring)."""
    members = class_members(representative)
    if isinstance(family, PlateFamily):
        rotations, incident = family.poses(), family.incident_direction()
    else:
        rotations, incident = haar_rotations(family.samples, np.random.default_rng(family.seed)), incident_direction_from_sun(_PROBE_SUN)
    weights: dict[str, dict[Faces, np.ndarray]] = {"red": {}, "blue": {}}
    checks: dict[str, dict[Faces, optics.BatchDomainCheck]] = {"red": {}, "blue": {}}
    for label, index in (("red", n_red), ("blue", n_blue)):
        for member in members:
            if not _has_faces(crystal, member):
                continue
            w = weighted_power(rotations, member, incident, index, crystal=crystal)
            if np.any(w > 0.0):
                weights[label][member] = w
                checks[label][member] = optics.path_domain_batch(rotations, member, incident, index, crystal=crystal)
    lit = {label: tuple(sorted(weights[label])) for label in weights}
    if isinstance(family, PlateFamily):
        tint = _tint(weights, checks)
        verdict = _tint_verdict(tuple(representative), tint, n_red, n_blue)
        return ClassVerdict(tuple(representative), members, lit, verdict)
    member_verdicts = []
    for member in _one_per_symmetry_orbit(crystal, sorted(set(lit["red"]) | set(lit["blue"]))):
        if halo_map_rank(crystal, member) == 0:
            continue
        member_verdicts.append(diagnose(crystal, member, n_red=n_red, n_blue=n_blue, lattice_n=lattice_n))
    features = tuple(f for v in member_verdicts for f in v.features)
    notes = tuple(n for v in member_verdicts for n in v.notes)
    return ClassVerdict(
        tuple(representative), members, lit, _verdict(tuple(representative), features, notes, n_red, n_blue), tuple(member_verdicts)
    )


def _has_faces(crystal: Polyhedron, faces: Faces) -> bool:
    try:
        optics.face_normals(crystal, faces)
    except (KeyError, ValueError):
        return False
    return True


def _one_per_symmetry_orbit(crystal: Polyhedron, members: Sequence[Faces]) -> list[Faces]:
    """The first member of each ``G_true`` orbit (:func:`.path_class.g_true_orbit`; every member for a non-prism)."""
    if not isinstance(crystal, HexPrism):
        return list(members)
    out: list[Faces] = []
    seen: set[Faces] = set()
    for member in members:
        if member in seen:
            continue
        out.append(member)
        seen |= g_true_orbit(member, crystal)
    return out


def _tint(weights: dict[str, dict[Faces, np.ndarray]], checks: dict[str, dict[Faces, optics.BatchDomainCheck]]) -> TintMetrics:
    energy = {label: sum(float(np.mean(w)) for w in weights[label].values()) for label in weights}
    fraction = {}
    for label in weights:
        total = tir = 0.0
        for member, w in weights[label].items():
            for name, margin in checks[label][member].margins.items():
                if name.endswith("_tir_discriminant"):
                    total += float(np.sum(w))
                    tir += float(np.sum(w * (np.asarray(margin) > 0.0)))
        fraction[label] = tir / total if total > 0.0 else float("nan")
    spread_sum = weight_sum = 0.0
    for member in set(weights["red"]) & set(weights["blue"]):
        both = (weights["red"][member] > 0.0) & (weights["blue"][member] > 0.0)
        if not both.any():
            continue
        a = checks["red"][member].direction[both]
        b = checks["blue"][member].direction[both]
        angle = np.arctan2(np.linalg.norm(np.cross(a, b), axis=1), np.sum(a * b, axis=1))
        w = weights["red"][member][both]
        spread_sum += float(np.sum(w * angle))
        weight_sum += float(np.sum(w))
    ratio = energy["blue"] / energy["red"] if energy["red"] > 0.0 else float("nan")
    return TintMetrics(
        energy["red"], energy["blue"], ratio, fraction["red"], fraction["blue"], spread_sum / weight_sum if weight_sum > 0.0 else 0.0
    )


def _tint_verdict(representative: Faces, tint: TintMetrics, n_red: float, n_blue: float) -> ChromaticVerdict:
    if tint.energy_red <= 0.0 and tint.energy_blue <= 0.0:
        return ChromaticVerdict(representative, "none", "none", False, None, (), tint, ("class not lit at either index",), n_red, n_blue)
    if tint.energy_red <= 0.0 or tint.energy_blue <= 0.0:  # lit at one index only: the extreme tint, not "no colour"
        color, lit = ("blue", n_blue) if tint.energy_red <= 0.0 else ("red", n_red)
        note = f"class lit at n = {lit} only"
        return ChromaticVerdict(representative, "tint", color, True, None, (), tint, (note,), n_red, n_blue)
    if tint.direction_dispersion >= EDGE_MIN_SHIFT_RAD:
        note = (
            f"red and blue land {np.degrees(tint.direction_dispersion):.2f} deg apart: the ordinary dispersion of the "
            "direction map spreads the colours, the spot has no single tint (outside this criterion)"
        )
        return ChromaticVerdict(representative, "none", "none", False, None, (), tint, (note,), n_red, n_blue)
    if tint.ratio >= TINT_RATIO_MIN:
        return ChromaticVerdict(representative, "tint", "blue", True, None, (), tint, (), n_red, n_blue)
    if tint.ratio <= 1.0 / TINT_RATIO_MIN:
        return ChromaticVerdict(representative, "tint", "red", True, None, (), tint, (), n_red, n_blue)
    return ChromaticVerdict(representative, "none", "white", False, None, (), tint, (), n_red, n_blue)


__all__ = [
    "EDGE_MIN_SHIFT_RAD",
    "EDGE_SPREAD_PER_SHIFT",
    "N_BLUE",
    "N_RED",
    "TINT_RATIO_MIN",
    "ChromaticFeature",
    "ChromaticVerdict",
    "ClassVerdict",
    "PlateFamily",
    "RandomOrientation",
    "TintMetrics",
    "class_members",
    "diagnose",
    "diagnose_class",
    "weighted_power",
]
