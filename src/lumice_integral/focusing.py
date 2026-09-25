"""Two kinds of focusing, as an explicit label of a (path, pose density) pair (``docs/phase2.md`` section 10).

A bright or singular feature in a halo image has one of two causes, and the
solver says which instead of leaving it implicit in a filter:

- *Jacobian focusing* comes from ``Phi``: the level-set measure
  ``int dl / |grad_{S^2} D_P|`` of the random-orientation integral diverges
  as ``delta`` approaches a critical value.  It depends on the path alone
  (its ``D_P`` field on ``U_P``) and is read off the critical set of
  :class:`.dp_field.DPField`.
- *Dimension collapse* comes from ``rho``: a pose density that confines the
  pose to a lower-dimensional family (a narrow Gaussian factor per confined
  dimension) confines ``u`` to a curve or a point for each pixel azimuth, and
  the pixel value then diverges where that curve meets the level sets badly
  (``1 / sqrt`` at a tangency, ``1 / sqrt(delta - D_min)`` where it passes
  through an isolated extremum), capped by the confinement width.  It depends
  on the density family alone.

The profile each critical value produces for a *random* orientation
(``rho = 1``), with the local reason:

``finite_jump``
    Non-degenerate interior minimum or maximum: the level set is a small
    ellipse and ``int dl / |grad D| -> 2 pi / sqrt(det H)``; the pixel value
    jumps from 0 to a finite value (the 22 deg inner edge).
``log_divergence``
    Interior saddle: ``int dl / |grad D| ~ log |delta - c|``.  Jacobian focusing.
``inverse_sqrt_divergence``
    A curve of maxima or minima (the great circle ``u . n_M = 0`` of a
    rotation slab, where ``D = theta_M``): ``~ 1 / sqrt |delta - c|``.
    Jacobian focusing; on ``dU_P`` still a divergence of the level-set
    measure (the window may suppress it, the label does not).
``cone_point``
    The axis ``+-n_M`` of a slab: ``D`` grows linearly away from it
    (``|grad D| -> 2 sin(theta_M / 2)``, ``2`` for a mirror), the level set
    is a circle of radius ``~ |delta - c|`` and its measure vanishes.
``crease``
    The mirror plane ``u . n_M = 0`` of a mirror slab (``D = 2 arcsin |u . n_M|``,
    ``|grad D| = 2`` on both sides): a kink, no focusing.
``boundary_onset``
    A local extremum of ``D_P`` restricted to ``dU_P`` or a corner of it,
    with ``grad D_P`` non-zero there: an arc appears or vanishes with length
    ``~ sqrt |delta - c|`` and a bounded integrand, a square-root cusp.
``degenerate``
    Anything else with a vanishing gradient (a degenerate Hessian): counted
    as Jacobian focusing, strength not classified.

Rank-0 paths (:func:`.geometry.halo_map_rank` ``== 0``: fold matrix ``I``,
wedge 0) have no field at all (:meth:`.dp_field.DPField.build` refuses them):
their image is a point mass at the sun (:func:`.path_class.estimate_rank0_contribution`),
labelled ``point_mass`` here and nothing else.

Nothing here changes a value the quadratures compute; the labels are read
from the same critical data (:class:`.dp_field.DPField`) and density
(:mod:`.pose_density`) the renderers use.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from . import optics
from .dp_field import DPField
from .dp_field.boundary import EXTREMUM_ATOL, ZERO_MARGIN_ATOL
from .dp_field.field import tangent_basis
from .geometry import Polyhedron, halo_map_rank
from .pose_density import HaarUniformPoseDensity, PoseDensity, ZenithGaussianPoseDensity, ZenithRollGaussianPoseDensity
from .s2_store import fibonacci_sphere

PROFILES = (
    "finite_jump",
    "log_divergence",
    "inverse_sqrt_divergence",
    "cone_point",
    "crease",
    "boundary_onset",
    "degenerate",
)
# Profiles whose level-set measure int dl / |grad D| diverges at the critical value.
JACOBIAN_FOCUSING_PROFILES = frozenset({"log_divergence", "inverse_sqrt_divergence", "degenerate"})
# A full S^2 gradient below this at a boundary critical point is a vanishing gradient (the interior
# critical-point tolerance of dp_field.field is 1e-10 on Newton iterates; boundary points are located
# by bisection along a piece to ~1e-12 rad, where a non-degenerate |grad D| is O(1)).
BOUNDARY_GRADIENT_ATOL = 1e-6
# Samples of the slab circle u . n_M = 0 and of the U_P lattice.
CIRCLE_SAMPLES = 7200
LATTICE_N = 20000
# Angle (rad) off the slab axis at which the cone slope |grad D| is read.
CONE_PROBE_RAD = 1e-4


@dataclass(frozen=True)
class CriticalOnset:
    """One critical value of ``D_P`` and the profile it produces for a random orientation (module docstring).

    ``location`` is ``"interior"`` or ``"boundary"`` (of ``U_P``);
    ``source`` names the critical datum (``"interior_minimum"``, ``"slab_axis"``,
    ``"slab_circle"``, ``"boundary_extremum"``, ``"corner"`` ...);
    ``gradient_norm`` is ``|grad_{S^2} D_P|`` at (or, for a cone point,
    ``CONE_PROBE_RAD`` next to) the point; ``measure_limit`` is
    ``2 pi / sqrt(det H)`` for a ``finite_jump`` (the limit of ``int dl /
    |grad D|``), ``None`` otherwise; ``multiplicity`` counts the critical
    points merged into this record (same value within ``EXTREMUM_ATOL``,
    location, source and profile: the mirror images of one extremum, the
    corners where several margins meet), whose smallest ``gradient_norm``
    is kept.
    """

    value: float
    location: str
    source: str
    profile: str
    gradient_norm: float
    measure_limit: float | None = None
    multiplicity: int = 1

    @property
    def jacobian_focusing(self) -> bool:
        return self.profile in JACOBIAN_FOCUSING_PROFILES

    def as_json(self) -> dict[str, Any]:
        return {
            "value_deg": float(np.degrees(self.value)),
            "location": self.location,
            "source": self.source,
            "profile": self.profile,
            "jacobian_focusing": self.jacobian_focusing,
            "gradient_norm": self.gradient_norm,
            "measure_limit": self.measure_limit,
            "multiplicity": self.multiplicity,
        }


@dataclass(frozen=True)
class FocusingClassification:
    """The explicit focusing label of one path under one pose density (module docstring).

    ``onsets`` are every critical value's profile (empty for a rank-0 path);
    ``gradient_norm_range`` is ``(min, max)`` of ``|grad D_P|`` on the ``U_P``
    points of a ``LATTICE_N`` Fibonacci lattice (a sampled bound, not a
    proof); ``confined_dimensions`` / ``confinement_widths_rad`` come from
    :func:`confined_dimensions`.
    """

    path: str
    halo_map_rank: int
    onsets: tuple[CriticalOnset, ...]
    gradient_norm_range: tuple[float, float] | None
    confined_dimensions: int
    confinement_widths_rad: tuple[float, ...]

    @property
    def jacobian_focusing(self) -> bool:
        return any(onset.jacobian_focusing for onset in self.onsets)

    @property
    def dimension_collapse(self) -> bool:
        return self.halo_map_rank > 0 and self.confined_dimensions > 0

    @property
    def mechanism(self) -> str:
        """``"point_mass"``, ``"none"``, ``"jacobian"``, ``"dimension_collapse"`` or ``"jacobian+dimension_collapse"``."""
        if self.halo_map_rank == 0:
            return "point_mass"
        parts = [name for name, on in (("jacobian", self.jacobian_focusing), ("dimension_collapse", self.dimension_collapse)) if on]
        return "+".join(parts) if parts else "none"

    def as_json(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "halo_map_rank": self.halo_map_rank,
            "mechanism": self.mechanism,
            "jacobian_focusing": self.jacobian_focusing,
            "dimension_collapse": self.dimension_collapse,
            "confined_dimensions": self.confined_dimensions,
            "confinement_widths_deg": [float(np.degrees(w)) for w in self.confinement_widths_rad],
            "gradient_norm_range": None if self.gradient_norm_range is None else list(self.gradient_norm_range),
            "onsets": [onset.as_json() for onset in self.onsets],
        }


def confined_dimensions(density: PoseDensity) -> tuple[int, tuple[float, ...]]:
    """Pose dimensions a density confines (one per narrow Gaussian factor) and their widths (rad).

    ``random`` confines none; ``column`` / ``plate`` the c-axis zenith (one);
    ``parry`` / ``lowitz`` the zenith and the roll about the c axis (two).
    Dispatch is on the density's type, not on a family name, and no width
    threshold is applied: a wide factor still confines one dimension, it only
    moves the crossover below which the collapse is resolved (chapter-10
    verdict ``inner-edge``: ``delta - D_min ~ sigma^2``).
    """
    if isinstance(density, HaarUniformPoseDensity):
        return 0, ()
    if isinstance(density, ZenithGaussianPoseDensity):
        return 1, (float(density.zenith_std_rad),)
    if isinstance(density, ZenithRollGaussianPoseDensity):
        return 2, (float(density.zenith_std_rad), float(density.roll_std_rad))
    raise TypeError(f"no confinement rule for pose density {type(density).__name__}")


def interior_onset(value: float, kind: str, hessian_eigenvalues: np.ndarray, gradient_norm: float) -> CriticalOnset:
    """The onset of a non-slab interior critical point from its Morse kind (``dp_field.field.classify``)."""
    eigenvalues = np.asarray(hessian_eigenvalues, dtype=float)
    if kind in ("minimum", "maximum"):
        limit = float(2.0 * np.pi / np.sqrt(abs(np.prod(eigenvalues))))
        return CriticalOnset(float(value), "interior", f"interior_{kind}", "finite_jump", float(gradient_norm), limit)
    if kind == "saddle":
        return CriticalOnset(float(value), "interior", "interior_saddle", "log_divergence", float(gradient_norm))
    return CriticalOnset(float(value), "interior", "interior_degenerate", "degenerate", float(gradient_norm))


def _cone_slope(field: DPField, axis: np.ndarray) -> float:
    e = np.asarray(tangent_basis(np.asarray(axis)))[0]
    probe = np.cos(CONE_PROBE_RAD) * axis + np.sin(CONE_PROBE_RAD) * e
    return float(np.linalg.norm(field.gradient_batch(probe[None, :])[0]))


def _circle_location(field: DPField, axis: np.ndarray) -> str:
    """Where the slab circle ``u . n_M = 0`` lies relative to ``U_P``: interior, boundary (touches) or exterior."""
    e = np.asarray(tangent_basis(np.asarray(axis)))
    t = np.linspace(0.0, 2.0 * np.pi, CIRCLE_SAMPLES, endpoint=False)
    circle = np.cos(t)[:, None] * e[0] + np.sin(t)[:, None] * e[1]
    smallest = np.min(field.validity_margins_batch(circle), axis=1)
    smallest = smallest[np.isfinite(smallest)]
    if smallest.size == 0:
        return "exterior"
    best = float(np.max(smallest))
    if best > ZERO_MARGIN_ATOL:
        return "interior"
    return "boundary" if best >= -ZERO_MARGIN_ATOL else "exterior"


def _slab_onsets(field: DPField) -> list[CriticalOnset]:
    fold = field.degenerate_fold
    m = field.fold.fold_matrix
    mirror = np.linalg.det(m) < 0.0
    onsets = []
    for point, where in fold.axis_points:
        if where == "exterior":
            continue
        value = float(field.d_p_batch(point[None, :])[0])
        onsets.append(CriticalOnset(value, where, "slab_axis", "cone_point", _cone_slope(field, point)))
    where = _circle_location(field, fold.axis)
    if where != "exterior":
        # rotation by theta_M: D = theta_M on the circle, a curve of maxima; mirror: D = 0, a crease
        theta = 0.0 if mirror else float(np.arccos(np.clip(0.5 * (np.trace(m) - 1.0), -1.0, 1.0)))
        profile = "crease" if mirror else "inverse_sqrt_divergence"
        onsets.append(CriticalOnset(theta, where, "slab_circle", profile, 2.0 if mirror else 0.0))
    return onsets


def field_onsets(field: DPField) -> tuple[CriticalOnset, ...]:
    """Every critical value of ``field`` with its random-orientation profile, sorted by value (module docstring)."""
    onsets: list[CriticalOnset] = []
    if field.slab is not None:
        onsets += _slab_onsets(field)
    else:
        for point in field.interior_critical_points:
            onsets.append(interior_onset(point.value, point.kind, point.hessian_eigenvalues, point.gradient_norm))
    # restricted extrema at a corner are the corner itself (field.corners lists every corner)
    boundary = [(p.position, p.value, "boundary_extremum") for p in field.boundary_critical_points if not p.corner]
    boundary += [(c.position, c.value, "corner") for c in field.corners]
    if boundary:
        norms = np.linalg.norm(field.gradient_batch(np.stack([b[0] for b in boundary])), axis=1)
        for (position, value, source), norm in zip(boundary, norms):
            finite = bool(np.isfinite(norm))
            # a non-finite gradient is an exit-TIR end (|grad D| unbounded): not a vanishing one
            profile = "degenerate" if finite and norm <= BOUNDARY_GRADIENT_ATOL else "boundary_onset"
            onsets.append(CriticalOnset(float(value), "boundary", source, profile, float(norm) if finite else float("inf")))
    return _merged(onsets)


def _merged(onsets: Sequence[CriticalOnset]) -> tuple[CriticalOnset, ...]:
    """Records of one value (within ``EXTREMUM_ATOL``), location, source and profile merged, sorted by value."""
    out: list[CriticalOnset] = []
    for onset in sorted(onsets, key=lambda o: (o.value, o.location, o.source, o.profile)):
        for k, kept in enumerate(out):
            if (abs(kept.value - onset.value) <= EXTREMUM_ATOL and (kept.location, kept.source, kept.profile)
                    == (onset.location, onset.source, onset.profile)):
                out[k] = dataclasses.replace(kept, gradient_norm=min(kept.gradient_norm, onset.gradient_norm),
                                             multiplicity=kept.multiplicity + onset.multiplicity)
                break
        else:
            out.append(onset)
    return tuple(sorted(out, key=lambda o: (o.value, o.location, o.source)))


def gradient_norm_range(field: DPField, lattice_n: int = LATTICE_N) -> tuple[float, float] | None:
    """``(min, max)`` of ``|grad D_P|`` on the ``U_P`` points of a Fibonacci lattice; ``None`` if none is inside."""
    lattice = fibonacci_sphere(lattice_n)
    inside = lattice[field.valid_batch(lattice)]
    if len(inside) == 0:
        return None
    norms = np.linalg.norm(field.gradient_batch(inside), axis=1)
    norms = norms[np.isfinite(norms)]
    return (float(norms.min()), float(norms.max())) if norms.size else None


def classify(
    crystal: Polyhedron,
    faces: Sequence[int],
    density: PoseDensity,
    index: float = float(optics.ICE_REFRACTIVE_INDEX),
    *,
    field: DPField | None = None,
) -> FocusingClassification:
    """The focusing label of ``faces`` on ``crystal`` under ``density`` (module docstring).

    ``field`` may pass an already built :class:`.dp_field.DPField` of the
    same path (its layers are cached).  A rank-0 path is labelled
    ``point_mass`` without building a field.
    """
    faces = optics.normalize_faces(faces)
    dims, widths = confined_dimensions(density)
    rank = halo_map_rank(crystal, faces)
    path = optics.path_id_of(faces)
    if rank == 0:
        return FocusingClassification(path, 0, (), None, dims, widths)
    if field is None:
        field = DPField.build(crystal, faces, index)
    elif field.faces != faces:
        raise ValueError(f"field is for {optics.path_id_of(field.faces)}, not {path}")
    return FocusingClassification(path, rank, field_onsets(field), gradient_norm_range(field), dims, widths)


__all__ = [
    "BOUNDARY_GRADIENT_ATOL",
    "CriticalOnset",
    "FocusingClassification",
    "JACOBIAN_FOCUSING_PROFILES",
    "PROFILES",
    "classify",
    "confined_dimensions",
    "field_onsets",
    "gradient_norm_range",
    "interior_onset",
]
