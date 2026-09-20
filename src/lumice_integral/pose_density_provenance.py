"""Provenance / figure-data ``pose_density`` block for the five families.

Kept apart from :mod:`.pose_density` (the density mathematics) because the two
evolve at different rates: this module fixes the JSON *schema* that
``canonical_scene.canonical_fixture_metadata`` and ``strip_io.scene_block``
write and that archived strips are compared against.

The block is flat; ``strip_io.scene_block`` wraps it in its own
``{"value": ..., "provenance": ...}`` envelope like every other field there.
The column block is byte-for-byte the pre-family literal (``model``,
``zenith_mean_deg``, ``zenith_std_deg``) plus the new trailing ``family`` key,
so archived provenance keeps reading the same way; the other families follow
the same shape with their own ``model`` strings and, for the roll-locked ones,
``roll_mean_deg`` / ``roll_std_deg`` before ``family``.
"""

from __future__ import annotations

from typing import Any

from .pose_density import PoseDensityFamily, resolve_pose_density_parameters

POSE_DENSITY_MODEL_NAMES: dict[str, str] = {
    "random": "haar-uniform random",
    "plate": "zenith-gaussian plate",
    "column": "zenith-gaussian column",
    "parry": "zenith-roll-gaussian parry",
    "lowitz": "zenith-roll-gaussian lowitz",
}


def pose_density_provenance(
    family: PoseDensityFamily,
    *,
    zenith_mean_deg: float | None = None,
    zenith_std_deg: float | None = None,
    roll_mean_deg: float | None = None,
    roll_std_deg: float | None = None,
) -> dict[str, Any]:
    """Flat, JSON-serializable ``pose_density`` block of ``family`` (module docstring).

    Same parameter contract as :func:`.pose_density.build_pose_density`; the
    values recorded are the resolved ones (family defaults filled in).
    """
    parameters = resolve_pose_density_parameters(
        family,
        zenith_mean_deg=zenith_mean_deg,
        zenith_std_deg=zenith_std_deg,
        roll_mean_deg=roll_mean_deg,
        roll_std_deg=roll_std_deg,
    )
    return {"model": POSE_DENSITY_MODEL_NAMES[family], **parameters, "family": family}


__all__ = ["POSE_DENSITY_MODEL_NAMES", "pose_density_provenance"]
