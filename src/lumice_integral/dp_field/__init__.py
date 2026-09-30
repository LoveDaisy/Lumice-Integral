"""The ``D_P`` field layer on ``S^2`` (Phase II, ``docs/phase2.md`` sections 3.1 and 4).

For one face sequence ``P`` (every member of a ``path_class.phi_key`` group
has the same field; the valid domain ``U_P`` is the member's own) this
package evaluates the deviation ``D_P`` on the body-frame sphere in batches
(``jax.vmap``, value / tangent gradient / Riemannian Hessian), finds its
interior critical points, walks the boundary ``dU_P`` (pieces, corners with
every vanishing margin, restricted extrema) and partitions the ``delta`` axis
into intervals of constant level-set topology with their component counts,
and traces the weight kinks (TIR onsets of internal reflections, :mod:`.weight_kink`).
The field depends on the crystal's face normals, the face sequence and the
refractive index only; there is no sun direction anywhere in it.

The public surface is :class:`DPField` (with :class:`KinkCurve` /
:class:`KinkArc`, its weight kinks, and :class:`TopologyEscape`, the
exception of its interval partition): its :meth:`DPField.build` is the
one place a rank-0 path (``geometry.halo_map_rank == 0``, a point mass
handled by ``path_class.estimate_rank0_contribution``) is refused, so the
free functions of :mod:`.field`, :mod:`.boundary`, :mod:`.certificate` and :mod:`.weight_kink`
are internal and not re-exported.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Sequence

import jax
import numpy as np

from .. import optics
from ..geometry import Polyhedron, halo_map_rank
from .boundary import BoundaryCriticalPoint, BoundaryLoop, BoundaryPiece, Corner, walk_boundary
from .weight_kink import KinkArc, KinkCurve, weight_kinks
from .certificate import (
    CriticalSet,
    DeviationInterval,
    DomainTopology,
    TopologyEscape,
    critical_values,
    domain_topology,
    interval_partition,
)
from .field import (
    DegenerateFoldSet,
    FoldScreen,
    InteriorCriticalPoint,
    body_normals,
    d_p_batch,
    fold_screen,
    gradient_batch,
    hessian_tangent_batch,
    index_derivatives_batch,
    interior_critical_points,
    margins_batch,
    valid_batch,
    validity_margins_batch,
)

__all__ = ["DPField", "KinkArc", "KinkCurve", "TopologyEscape"]


@dataclass(frozen=True, eq=False)
class DPField:
    """``D_P`` of one face sequence on ``S^2``; deeper layers are computed on first access and cached.

    Build with :meth:`build`.  ``u`` arguments are ``(N, 3)`` unit vectors
    (body frame, toward the sun); values are radians.
    """

    crystal: Polyhedron
    faces: tuple[int, ...]
    index: float
    fold: FoldScreen
    lattice_n: int = 20000

    @classmethod
    def build(
        cls, crystal: Polyhedron, faces: Sequence[int], index: float = float(optics.ICE_REFRACTIVE_INDEX), *, lattice_n: int = 20000
    ) -> "DPField":
        """The field of ``faces``; ``ValueError`` for a rank-0 path (its image is a point mass, not a field)."""
        faces = optics.normalize_faces(faces, crystal)
        if halo_map_rank(crystal, faces) == 0:
            raise ValueError(
                f"{optics.path_id_of(faces, crystal)} has halo-map rank 0 (fold matrix I, wedge 0): its image is a point mass "
                "at the sun (path_class.estimate_rank0_contribution), not a D_P field"
            )
        return cls(crystal, faces, float(index), fold_screen(crystal, faces), lattice_n)

    @cached_property
    def normals(self) -> jax.Array:
        """``crystal``'s body normals of ``faces`` in path order: the ``normals`` argument of the :mod:`.field` kernels."""
        return body_normals(self.crystal, self.faces)

    @property
    def slab(self) -> np.ndarray | None:
        """The fold matrix when the path is a degenerate fold (``D_P = angle(M u, u)``), else ``None``."""
        return self.fold.fold_matrix if self.fold.degenerate else None

    # -- evaluation (not cached)
    def d_p_batch(self, u: np.ndarray) -> np.ndarray:
        return d_p_batch(u, self.faces, self.index, self.slab, crystal=self.crystal)

    def gradient_batch(self, u: np.ndarray) -> np.ndarray:
        return gradient_batch(u, self.faces, self.index, self.slab, crystal=self.crystal)

    def hessian_tangent_batch(self, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return hessian_tangent_batch(u, self.faces, self.index, self.slab, crystal=self.crystal)

    def index_derivatives_batch(self, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``dD_P/dn`` and every ``d margin / dn`` at fixed ``u`` (:func:`.field.index_derivatives_batch`)."""
        return index_derivatives_batch(u, self.faces, self.index, self.slab, crystal=self.crystal)

    def margins_batch(self, u: np.ndarray) -> np.ndarray:
        """Every margin of :func:`.optics.domain_margin_names` at each row of ``u`` (the gates: :meth:`validity_margins_batch`)."""
        return margins_batch(u, self.faces, self.index, crystal=self.crystal)

    def validity_margins_batch(self, u: np.ndarray) -> np.ndarray:
        """The gates of ``U_P`` (:func:`.optics.validity_margin_names`) at each row of ``u``, all positive inside ``U_P``."""
        return validity_margins_batch(u, self.faces, self.index, crystal=self.crystal)

    def valid_batch(self, u: np.ndarray) -> np.ndarray:
        """``u in U_P`` for each row (:func:`.optics.path_domain_batch`, the single authority of the gates)."""
        return valid_batch(u, self.faces, self.index, crystal=self.crystal)

    # -- layers (cached)
    @cached_property
    def _interior(self) -> tuple[tuple[InteriorCriticalPoint, ...], DegenerateFoldSet | None]:
        return interior_critical_points(self.fold, self.faces, self.index, lattice_n=self.lattice_n, crystal=self.crystal)

    @property
    def interior_critical_points(self) -> tuple[InteriorCriticalPoint, ...]:
        return self._interior[0]

    @property
    def degenerate_fold(self) -> DegenerateFoldSet | None:
        """The slab critical set and its location, ``None`` for a path with interior folds."""
        return self._interior[1]

    @cached_property
    def boundary(self) -> BoundaryLoop:
        return walk_boundary(self.crystal, self.faces, self.index, slab=self.slab, lattice_n=self.lattice_n)

    @property
    def boundary_curves(self) -> tuple[BoundaryPiece, ...]:
        return self.boundary.pieces

    @property
    def corners(self) -> tuple[Corner, ...]:
        return self.boundary.corners

    @property
    def boundary_critical_points(self) -> tuple[BoundaryCriticalPoint, ...]:
        return self.boundary.critical_points

    @cached_property
    def weight_kinks(self) -> tuple[KinkCurve, ...]:
        """The TIR onsets ``C_k`` of every internal reflection at this field's ``index`` (:mod:`.weight_kink`).

        The third kind of critical line: a kink of the Fresnel weight ``R_k``,
        not part of ``dU_P`` (:attr:`boundary`) nor of the topology
        certificate.  One field is one ``n``; the chromatic shift of a kink
        is the difference of two fields (:mod:`lumice_integral.chromatic`).
        """
        return weight_kinks(self.crystal, self.faces, self.index, slab=self.slab, lattice_n=self.lattice_n)

    @cached_property
    def domain_topology(self) -> DomainTopology:
        return domain_topology(self.faces, self.index, lattice_n=self.lattice_n, crystal=self.crystal)

    @property
    def critical_set(self) -> CriticalSet:
        return CriticalSet.of(self.interior_critical_points, self.boundary)

    @property
    def critical_values(self) -> np.ndarray:
        return critical_values(self.interior_critical_points, self.boundary)

    def interval_partition(self) -> tuple[DeviationInterval, ...]:
        """``[(delta_a, delta_b, n_components, n_closed, n_open)]``; :class:`TopologyEscape` outside the disk reasoning."""
        return interval_partition(
            self.faces,
            self.index,
            self.interior_critical_points,
            self.degenerate_fold,
            self.boundary,
            self.domain_topology,
            self.slab,
            crystal=self.crystal,
        )

