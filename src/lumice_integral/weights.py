"""Named physical weight factors evaluated on accepted fiber poses.

This module owns the weight-layer schema shared by :mod:`.continuation`
(``FiberProblem.weight_evaluators`` / ``FiberResult.weight_observables``) and
the concrete factor implementations of ``docs/phase1-math-contract.md``
section 7 that Phase I can evaluate today:

- ``rho_pose``: :mod:`.pose_density` (relative to Haar probability);
- ``entry_measure``: :func:`.geometry.entry_measure` (absolute area);
- ``fresnel_transmission``: :func:`.optics.fresnel_transmission_path`;
- ``path_validity``: boolean gate ``path_domain(...).valid`` and
  ``entry_measure > 0``.

Every factor takes the face sequence explicitly
(:func:`build_path_weight_evaluators`); :func:`build_3_5_weight_evaluators`
is the same assembly at ``faces == PATH_3_5_FACES``.

Every factor is exposed on its own; nothing here multiplies factors together,
substitutes ``1`` for a missing factor, or folds in ``J_perp`` or the
``1 / (8 pi^2)`` Haar conversion.  Evaluation is host-side numpy on already
accepted poses (after continuation), never inside a ``jax.jit`` kernel.

Each factor has a scalar form (``evaluate``, one pose) and an optional batch
form (``evaluate_batch``, ``(N, 3, 3)`` poses at once) that must agree
elementwise; :func:`evaluate_weights_batch` prefers the batch form and falls
back to a host loop over the scalar one and reports the wall clock of each
factor (``entry_measure`` is numpy polygon clipping and dominates:
task-resample-and-integrate Step 0 fact 5 / Step 3).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Mapping, NamedTuple, Sequence

import numpy as np

from .geometry import HexPrism, entry_measure, entry_measure_batch
from .optics import (
    PATH_3_5_FACES,
    fresnel_transmission_path,
    fresnel_transmission_path_batch,
    normalize_faces,
    path_domain,
    path_domain_batch,
    path_id_of,
)
from .pose_density import PoseDensity

STANDARD_WEIGHT_NAMES = (
    "rho_pose",
    "entry_measure",
    "visibility",
    "fresnel_transmission",
    "path_validity",
    "source_factor",
    "pixel_factor",
    "other_radiometric",
)

WeightFunction = Callable[[np.ndarray], float]
BatchWeightFunction = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class WeightEvaluator:
    """One named factor: a pose -> scalar map plus its declared unit/normalization.

    ``evaluate_batch`` (``(N, 3, 3)`` poses -> ``(N,)`` values) is optional
    and must agree with ``evaluate`` elementwise; ``None`` means "loop over
    ``evaluate``" (see :func:`evaluate_weights_batch`).
    """

    evaluate: WeightFunction
    unit: str
    normalization: str
    evaluate_batch: BatchWeightFunction | None = None

    def __post_init__(self) -> None:
        if not callable(self.evaluate):
            raise ValueError("weight evaluator must be callable")
        if self.evaluate_batch is not None and not callable(self.evaluate_batch):
            raise ValueError("weight batch evaluator must be callable")
        if not self.unit or not self.normalization:
            raise ValueError("weight evaluator must declare unit and normalization")


@dataclass(frozen=True)
class WeightObservable:
    """Per-factor result: ``status`` is ``available`` or ``unavailable``.

    ``values`` aligns with ``FiberResult.poses`` when available and is ``None``
    otherwise; an unavailable factor never carries a stand-in value.
    """

    status: str
    unit: str | None = None
    normalization: str | None = None
    values: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.status not in ("available", "unavailable"):
            raise ValueError("weight status must be 'available' or 'unavailable'")
        if self.status == "unavailable" and (
            self.values is not None or self.unit is not None or self.normalization is not None
        ):
            raise ValueError("unavailable weights carry no unit, normalization, or values")
        if self.status == "available":
            if self.values is None or self.unit is None or self.normalization is None:
                raise ValueError("available weights require unit, normalization, and values")
            values = np.asarray(self.values)
            if values.ndim != 1 or values.dtype != np.float64:
                raise ValueError("weight values must be a float64 vector aligned with poses")


def evaluate_weights(
    evaluators: Mapping[str, WeightEvaluator], poses: np.ndarray
) -> dict[str, WeightObservable]:
    """Evaluate registered factors on accepted poses; report the rest unavailable."""
    poses = np.asarray(poses, dtype=np.float64)
    names = list(STANDARD_WEIGHT_NAMES) + [
        name for name in evaluators if name not in STANDARD_WEIGHT_NAMES
    ]
    observables: dict[str, WeightObservable] = {}
    for name in names:
        evaluator = evaluators.get(name)
        if evaluator is None:
            observables[name] = WeightObservable("unavailable")
            continue
        values = np.asarray(
            [float(evaluator.evaluate(pose)) for pose in poses], dtype=np.float64
        ).reshape(len(poses))
        observables[name] = WeightObservable(
            "available", evaluator.unit, evaluator.normalization, values
        )
    return observables


class BatchWeights(NamedTuple):
    """Batch factor values aligned with the poses, and the wall clock each took."""

    values: dict[str, np.ndarray]
    seconds: dict[str, float]


def evaluate_weights_batch(
    evaluators: Mapping[str, WeightEvaluator], poses: np.ndarray, names: tuple[str, ...]
) -> BatchWeights:
    """Evaluate the named factors on ``(N, 3, 3)`` poses, batch form where available.

    Raises ``KeyError`` for a name without an evaluator: a batch caller has
    already declared the factors it needs (``quadrature.integrand_availability``),
    and an unavailable factor must not be silently replaced.
    """
    poses = np.asarray(poses, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (3, 3):
        raise ValueError("poses must have shape (N, 3, 3)")
    values: dict[str, np.ndarray] = {}
    seconds: dict[str, float] = {}
    for name in names:
        evaluator = evaluators[name]
        start = time.perf_counter()
        if evaluator.evaluate_batch is not None:
            result = np.asarray(evaluator.evaluate_batch(poses), dtype=np.float64)
        else:
            result = np.asarray([float(evaluator.evaluate(pose)) for pose in poses], dtype=np.float64)
        seconds[name] = time.perf_counter() - start
        if result.shape != (len(poses),):
            raise ValueError(f"batch weight {name!r} must return one value per pose")
        values[name] = result
    return BatchWeights(values, seconds)


def entry_measure_weight(
    rotation: np.ndarray,
    *,
    faces: Sequence[int],
    incident_direction: np.ndarray,
    crystal: HexPrism,
    refractive_index: float,
) -> float:
    """``entry_measure(...).value`` of ``faces`` (absolute projected area)."""
    return float(
        entry_measure(
            np.asarray(rotation, dtype=np.float64),
            normalize_faces(faces),
            np.asarray(incident_direction, dtype=np.float64),
            crystal,
            n_ice=refractive_index,
        ).value
    )


def fresnel_transmission_weight(
    rotation: np.ndarray, *, faces: Sequence[int], incident_direction: np.ndarray, refractive_index: float
) -> float:
    return fresnel_transmission_path(
        np.asarray(rotation, dtype=np.float64),
        faces,
        np.asarray(incident_direction, dtype=np.float64),
        refractive_index,
    )


class _EntryMeasureBatch:
    """:func:`.geometry.entry_measure_batch` for one face sequence, memoised once.

    ``entry_measure`` and ``path_validity`` need the same per-pose footprint
    areas; the memo keeps the most recent pose array object and its values so
    that a batch caller evaluating both factors on one array pays the
    clipping once.  The array itself is held (identity test with ``is``), so
    a freed array's id can never be mistaken for a live one.  Explicitly not
    a general cache: one entry, replaced on every new array.
    """

    def __init__(
        self, *, faces: Sequence[int], incident_direction: np.ndarray, crystal: HexPrism, refractive_index: float
    ) -> None:
        self._faces = normalize_faces(faces)
        self._incident = np.asarray(incident_direction, dtype=np.float64)
        self._crystal = crystal
        self._index = float(refractive_index)
        self._last: tuple[object, np.ndarray] | None = None

    def __call__(self, rotations: np.ndarray) -> np.ndarray:
        if self._last is not None and self._last[0] is rotations:
            return self._last[1]
        values = np.asarray(
            entry_measure_batch(
                np.asarray(rotations, dtype=np.float64), self._faces, self._incident, self._crystal,
                n_ice=self._index,
            ),
            dtype=np.float64,
        )
        self._last = (rotations, values)
        return values


def path_validity_weight(
    rotation: np.ndarray,
    *,
    faces: Sequence[int],
    incident_direction: np.ndarray,
    crystal: HexPrism,
    refractive_index: float,
) -> float:
    """``1.0`` when the smooth branch of ``faces`` is valid and the entry footprint is nonempty."""
    rotation = np.asarray(rotation, dtype=np.float64)
    incident = np.asarray(incident_direction, dtype=np.float64)
    if not path_domain(rotation, faces, incident, refractive_index).valid:
        return 0.0
    measure = entry_measure_weight(
        rotation,
        faces=faces,
        incident_direction=incident,
        crystal=crystal,
        refractive_index=refractive_index,
    )
    return 1.0 if measure > 0.0 else 0.0


def build_path_weight_evaluators(
    *,
    faces: Sequence[int],
    incident_direction: np.ndarray,
    refractive_index: float,
    crystal: HexPrism,
    pose_density: PoseDensity,
) -> dict[str, WeightEvaluator]:
    """Assemble the four Phase I factors of one face sequence in one scene (single authority).

    ``pose_density`` is any of the :mod:`.pose_density` families (duck-typed:
    ``unit``, ``normalization``, ``__call__``, ``evaluate_batch``); it only
    changes the ``rho_pose`` factor.  ``visibility``, ``source_factor``,
    ``pixel_factor`` and ``other_radiometric`` are deliberately absent so that
    they stay ``unavailable`` downstream.
    """
    faces = normalize_faces(faces)
    path_id = path_id_of(faces)
    incident = np.asarray(incident_direction, dtype=np.float64)
    index = float(refractive_index)
    entry_measure_batch = _EntryMeasureBatch(
        faces=faces, incident_direction=incident, crystal=crystal, refractive_index=index
    )

    def path_validity_batch(rotations: np.ndarray) -> np.ndarray:
        valid = path_domain_batch(rotations, faces, incident, index).valid
        return np.where(valid & (entry_measure_batch(rotations) > 0.0), 1.0, 0.0)

    return {
        "rho_pose": WeightEvaluator(
            pose_density, pose_density.unit, pose_density.normalization,
            evaluate_batch=pose_density.evaluate_batch,
        ),
        "entry_measure": WeightEvaluator(
            lambda rotation: entry_measure_weight(
                rotation,
                faces=faces,
                incident_direction=incident,
                crystal=crystal,
                refractive_index=index,
            ),
            f"length^2 (crystal length unit; hexagon edge a = {crystal.a:g})",
            (
                f"absolute area of the face-{faces[0]} footprint that follows path {path_id}, "
                "projected perpendicular to the world incident direction; not "
                "divided by any face area or reference cross-section; 0 when "
                "any gate fails (see geometry.entry_measure)"
            ),
            evaluate_batch=entry_measure_batch,
        ),
        "fresnel_transmission": WeightEvaluator(
            lambda rotation: fresnel_transmission_weight(
                rotation, faces=faces, incident_direction=incident, refractive_index=index
            ),
            "dimensionless",
            (
                "product of unpolarized (s/p averaged) power transmittances at "
                f"the face-{faces[0]} entry and face-{faces[-1]} exit interfaces "
                "(internal reflections are total on the smooth branch); "
                f"0 outside the smooth {path_id} domain"
            ),
            evaluate_batch=lambda rotations: fresnel_transmission_path_batch(
                rotations, faces, incident, index
            ),
        ),
        "path_validity": WeightEvaluator(
            lambda rotation: path_validity_weight(
                rotation,
                faces=faces,
                incident_direction=incident,
                crystal=crystal,
                refractive_index=index,
            ),
            "boolean (0 or 1)",
            f"gate: path_domain(..., {path_id}).valid and entry_measure > 0; not a gain",
            evaluate_batch=path_validity_batch,
        ),
    }


def build_3_5_weight_evaluators(
    *,
    incident_direction: np.ndarray,
    refractive_index: float,
    crystal: HexPrism,
    pose_density: PoseDensity,
) -> dict[str, WeightEvaluator]:
    """The four Phase I factors of the 3-5 scene (:func:`build_path_weight_evaluators` at ``(3, 5)``)."""
    return build_path_weight_evaluators(
        faces=PATH_3_5_FACES,
        incident_direction=incident_direction,
        refractive_index=refractive_index,
        crystal=crystal,
        pose_density=pose_density,
    )
