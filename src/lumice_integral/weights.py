"""Named physical weight factors evaluated on accepted fiber poses.

This module owns the weight-layer schema shared by :mod:`.continuation`
(``FiberProblem.weight_evaluators`` / ``FiberResult.weight_observables``) and
the concrete factor implementations of ``docs/phase1-math-contract.md``
section 7 that Phase I can evaluate today:

- ``rho_pose``: :mod:`.pose_density` (relative to Haar probability);
- ``entry_measure``: :func:`.geometry.entry_measure` (absolute area);
- ``fresnel_transmission``: :func:`.optics.fresnel_transmission_3_5`;
- ``path_validity``: boolean gate ``path_3_5_domain(...).valid`` and
  ``entry_measure > 0``.

Every factor is exposed on its own; nothing here multiplies factors together,
substitutes ``1`` for a missing factor, or folds in ``J_perp`` or the
``1 / (8 pi^2)`` Haar conversion.  Evaluation is host-side numpy on already
accepted poses (after continuation), never inside a ``jax.jit`` kernel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np

from .geometry import HexPrism, entry_measure
from .optics import fresnel_transmission_3_5, path_3_5_domain
from .pose_density import ZenithGaussianPoseDensity

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


@dataclass(frozen=True)
class WeightEvaluator:
    """One named factor: a pose -> scalar map plus its declared unit/normalization."""

    evaluate: WeightFunction
    unit: str
    normalization: str

    def __post_init__(self) -> None:
        if not callable(self.evaluate):
            raise ValueError("weight evaluator must be callable")
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


def entry_measure_weight(
    rotation: np.ndarray,
    *,
    incident_direction: np.ndarray,
    crystal: HexPrism,
    refractive_index: float,
) -> float:
    """``entry_measure(...).value`` for the 3-5 path (absolute projected area)."""
    return float(
        entry_measure(
            np.asarray(rotation, dtype=np.float64),
            (3, 5),
            np.asarray(incident_direction, dtype=np.float64),
            crystal,
            n_ice=refractive_index,
        ).value
    )


def fresnel_transmission_weight(
    rotation: np.ndarray, *, incident_direction: np.ndarray, refractive_index: float
) -> float:
    return fresnel_transmission_3_5(
        np.asarray(rotation, dtype=np.float64),
        np.asarray(incident_direction, dtype=np.float64),
        refractive_index,
    )


def path_validity_weight(
    rotation: np.ndarray,
    *,
    incident_direction: np.ndarray,
    crystal: HexPrism,
    refractive_index: float,
) -> float:
    """``1.0`` when the smooth 3-5 branch is valid and the entry footprint is nonempty."""
    rotation = np.asarray(rotation, dtype=np.float64)
    incident = np.asarray(incident_direction, dtype=np.float64)
    if not path_3_5_domain(rotation, incident, refractive_index).valid:
        return 0.0
    measure = entry_measure_weight(
        rotation,
        incident_direction=incident,
        crystal=crystal,
        refractive_index=refractive_index,
    )
    return 1.0 if measure > 0.0 else 0.0


def build_3_5_weight_evaluators(
    *,
    incident_direction: np.ndarray,
    refractive_index: float,
    crystal: HexPrism,
    pose_density: ZenithGaussianPoseDensity,
) -> dict[str, WeightEvaluator]:
    """Assemble the four Phase I factors for one 3-5 scene (single authority).

    ``visibility``, ``source_factor``, ``pixel_factor`` and ``other_radiometric``
    are deliberately absent so that they stay ``unavailable`` downstream.
    """
    incident = np.asarray(incident_direction, dtype=np.float64)
    index = float(refractive_index)
    return {
        "rho_pose": WeightEvaluator(
            pose_density, pose_density.unit, pose_density.normalization
        ),
        "entry_measure": WeightEvaluator(
            lambda rotation: entry_measure_weight(
                rotation,
                incident_direction=incident,
                crystal=crystal,
                refractive_index=index,
            ),
            f"length^2 (crystal length unit; hexagon edge a = {crystal.a:g})",
            (
                "absolute area of the face-3 footprint that follows path 3-5, "
                "projected perpendicular to the world incident direction; not "
                "divided by any face area or reference cross-section; 0 when "
                "any gate fails (see geometry.entry_measure)"
            ),
        ),
        "fresnel_transmission": WeightEvaluator(
            lambda rotation: fresnel_transmission_weight(
                rotation, incident_direction=incident, refractive_index=index
            ),
            "dimensionless",
            (
                "product of unpolarized (s/p averaged) power transmittances at "
                "the face-3 entry and face-5 exit interfaces; 0 outside the smooth "
                "3-5 domain"
            ),
        ),
        "path_validity": WeightEvaluator(
            lambda rotation: path_validity_weight(
                rotation,
                incident_direction=incident,
                crystal=crystal,
                refractive_index=index,
            ),
            "boolean (0 or 1)",
            "gate: path_3_5_domain(...).valid and entry_measure > 0; not a gain",
        ),
    }
