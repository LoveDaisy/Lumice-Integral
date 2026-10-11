"""Narrow perspective and crystal presentation helpers for theory illustrations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from _style import GRAY, ICE, LIGHT_GRAY


@dataclass(frozen=True)
class Camera:
    azimuth_deg: float
    elevation_deg: float
    projection: str = "perspective"
    distance: float = 7.0
    up: Sequence[float] = (0.0, 0.0, 1.0)
    _basis: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        azimuth = np.deg2rad(self.azimuth_deg)
        elevation = np.deg2rad(self.elevation_deg)
        view = np.array(
            [
                np.cos(elevation) * np.cos(azimuth),
                np.cos(elevation) * np.sin(azimuth),
                np.sin(elevation),
            ]
        )
        up = _unit(self.up)
        if abs(float(view @ up)) > 1.0 - 1e-9:
            raise ValueError("view direction is parallel to up")
        right = _unit(np.cross(up, view))
        camera_up = np.cross(view, right)
        object.__setattr__(self, "_basis", np.stack([right, camera_up, view]))

    @property
    def view_direction(self) -> np.ndarray:
        return self._basis[2]

    @property
    def position(self) -> np.ndarray:
        return self.view_direction * self.distance

    def project(self, points: Sequence[float] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        points_array = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        camera_points = points_array @ self._basis.T
        depth = camera_points[:, 2]
        if self.projection == "orthographic":
            xy = camera_points[:, :2]
        elif self.projection == "perspective":
            camera_distance = self.distance - depth
            if np.any(camera_distance <= 0.0):
                raise ValueError("point behind the perspective camera")
            xy = camera_points[:, :2] * (self.distance / camera_distance)[:, None]
        else:
            raise ValueError(f"unknown projection {self.projection!r}")
        return xy, depth

    def project_xy(self, points: Sequence[float] | np.ndarray) -> np.ndarray:
        return self.project(points)[0]

    def view_vectors(self, points: Sequence[float] | np.ndarray) -> np.ndarray:
        points_array = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        if self.projection == "orthographic":
            return np.broadcast_to(self.view_direction, points_array.shape).copy()
        return _unit_rows(self.position - points_array)


def _unit(vector: Sequence[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float64)
    norm = np.linalg.norm(array)
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError("direction must be finite and nonzero")
    return array / norm


def _unit_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1)
    if np.any(~np.isfinite(norms)) or np.any(norms == 0.0):
        raise ValueError("directions must be finite and nonzero")
    return vectors / norms[:, None]


def face_visibility(normals: np.ndarray, centers: np.ndarray, camera: Camera) -> np.ndarray:
    normals = np.asarray(normals, dtype=np.float64)
    centers = np.asarray(centers, dtype=np.float64)
    if normals.shape != centers.shape or normals.ndim != 2 or normals.shape[1] != 3:
        raise ValueError("normals and centers must have shape (F, 3)")
    return np.sum(normals * camera.view_vectors(centers), axis=1) > 1e-9


def edge_visibility(
    edges: Sequence[Sequence[int]],
    face_vertex_ids: Sequence[Sequence[int]],
    visible_faces: np.ndarray,
) -> np.ndarray:
    visible_faces = np.asarray(visible_faces, dtype=bool)
    result = []
    for edge in edges:
        edge_set = set(edge)
        adjacent = [index for index, ids in enumerate(face_vertex_ids) if edge_set.issubset(ids)]
        if not adjacent:
            raise ValueError(f"edge {tuple(edge)} has no adjacent face")
        result.append(any(visible_faces[index] for index in adjacent))
    return np.asarray(result, dtype=bool)


def draw_ray(ax, points: np.ndarray, *, camera: Camera) -> None:
    from _style import BLUE, INK, ORANGE

    projected = camera.project_xy(points)
    for start, end, color in zip(projected[:-1], projected[1:], (ORANGE, INK, BLUE)):
        ax.annotate(
            "",
            end,
            start,
            arrowprops={
                "arrowstyle": "-|>",
                "color": color,
                "lw": 2.2,
                "mutation_scale": 12,
                "shrinkA": 0,
                "shrinkB": 0,
            },
            zorder=12,
        )
    ax.scatter(*projected[1], color=ORANGE, s=25, zorder=14)
    ax.scatter(*projected[2], color=BLUE, s=25, zorder=14)


def draw_crystal(
    ax,
    *,
    vertices: np.ndarray,
    face_numbers: np.ndarray,
    face_vertex_ids: Sequence[Sequence[int]],
    edges: Sequence[Sequence[int]],
    face_normals: np.ndarray,
    face_centers: np.ndarray,
    camera: Camera,
    highlight_face: int = 3,
    label_highlight: bool = True,
) -> None:
    from matplotlib.patches import Polygon

    vertices = np.asarray(vertices, dtype=np.float64)
    face_numbers = np.asarray(face_numbers)
    visible = face_visibility(face_normals, face_centers, camera)
    _, depth = camera.project(face_centers)
    for face_index in np.argsort(depth):
        face_number = int(face_numbers[face_index])
        xy = camera.project_xy(vertices[list(face_vertex_ids[face_index])])
        fill = "#c6dce8" if face_number == highlight_face else ICE
        alpha = 0.34 if face_number == highlight_face else (0.24 if visible[face_index] else 0.10)
        ax.add_patch(
            Polygon(
                xy,
                closed=True,
                facecolor=fill,
                edgecolor="none",
                alpha=alpha,
                zorder=2 if visible[face_index] else 1,
            )
        )

    visible_edges = edge_visibility(edges, face_vertex_ids, visible)
    for edge, is_visible in zip(edges, visible_edges):
        xy = camera.project_xy(vertices[list(edge)])
        ax.plot(
            xy[:, 0],
            xy[:, 1],
            color=GRAY if is_visible else LIGHT_GRAY,
            linewidth=1.45 if is_visible else 0.9,
            linestyle="-" if is_visible else "--",
            alpha=1.0 if is_visible else 0.9,
            solid_capstyle="round",
            zorder=6 if is_visible else 3,
        )

    matches = np.flatnonzero(face_numbers == highlight_face)
    if len(matches) != 1:
        raise ValueError(f"highlight face {highlight_face} occurs {len(matches)} times")
    if label_highlight:
        label_xy = camera.project_xy(face_centers[matches[0]])[0]
        ax.text(
            *label_xy,
            str(highlight_face),
            color=GRAY,
            fontsize=13,
            ha="center",
            va="center",
            zorder=8,
        )
