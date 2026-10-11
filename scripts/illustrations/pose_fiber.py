"""Recompute or replot the canonical pose-fiber illustration."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from _pose_fiber_data import IllustrationData, IllustrationDataError, OUTPUT_NAME, load_data, recompute_data


CAMERA = {
    "azimuth_deg": 65.0,
    "elevation_deg": 24.0,
    "distance": 7.0,
    "projection": "perspective",
}


def _sphere_grid(ax, radius: float = 1.0) -> None:
    import numpy as np

    angle = np.linspace(0.0, 2.0 * np.pi, 160)
    for latitude in np.deg2rad([-60, -30, 0, 30, 60]):
        points = radius * np.array(
            [
                np.cos(latitude) * np.cos(angle),
                np.cos(latitude) * np.sin(angle),
                np.full_like(angle, np.sin(latitude)),
            ]
        )
        ax.plot(*points, color="#c6d4df", alpha=0.55, linewidth=0.65)
    for longitude in np.deg2rad(np.arange(0, 180, 30)):
        points = radius * np.array(
            [np.cos(angle) * np.cos(longitude), np.cos(angle) * np.sin(longitude), np.sin(angle)]
        )
        ax.plot(*points, color="#c6d4df", alpha=0.55, linewidth=0.65)


def _decorate_pose_axes(ax) -> None:
    ax.set_xlabel(r"$\omega_x$ [rad]", labelpad=8)
    ax.set_ylabel(r"$\omega_y$ [rad]", labelpad=8)
    ax.set_zlabel(r"$\omega_z$ [rad]", labelpad=8)
    ax.grid(True, alpha=0.45)


def _coordinate_triad(figure) -> None:
    import numpy as np
    from _crystal_view import Camera
    from _style import GRAY

    axis = figure.add_axes([0.065, 0.425, 0.073, 0.10])
    camera = Camera(azimuth_deg=-57.0, elevation_deg=24.0, projection="orthographic")
    projected = camera.project_xy(np.eye(3))
    origin = np.array([0.52, 0.42])
    for label, direction in zip((r"$u_x$", r"$u_y$", r"$u_z$"), projected):
        tip = origin + direction * 0.36
        axis.annotate(
            "",
            tip,
            origin,
            arrowprops={"arrowstyle": "->", "color": GRAY, "lw": 1.1},
        )
        axis.text(
            *(origin + direction * 0.47),
            label,
            fontsize=10,
            color=GRAY,
            ha="center",
            va="center",
        )
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")


def _direction_sphere(ax, data: IllustrationData) -> None:
    import numpy as np
    from _crystal_view import Camera
    from _style import BLUE, GRAY, INK, ORANGE

    camera = Camera(**CAMERA)
    radius = 1.25
    angle = np.linspace(0.0, 2.0 * np.pi, 240)
    lines = []
    for latitude in np.deg2rad([-60, -30, 0, 30, 60]):
        lines.append(
            radius
            * np.array(
                [
                    np.cos(latitude) * np.cos(angle),
                    np.cos(latitude) * np.sin(angle),
                    np.full_like(angle, np.sin(latitude)),
                ]
            ).T
        )
    for longitude in np.deg2rad(np.arange(0, 180, 30)):
        lines.append(
            radius
            * np.array(
                [np.cos(angle) * np.cos(longitude), np.cos(angle) * np.sin(longitude), np.sin(angle)]
            ).T
        )
    for points in lines:
        projected = camera.project_xy(points)
        visible = np.sum(points * camera.view_vectors(points), axis=1) > 0.0
        for mask, alpha in ((visible, 0.65), (~visible, 0.25)):
            display = np.where(mask[:, None], projected, np.nan)
            ax.plot(*display.T, color="#b9ccda", linewidth=0.8, alpha=alpha)

    sun = data.arrays["source_world"]
    sky = -data.arrays["target_direction"]
    outgoing = data.arrays["target_direction"]
    for start, end, color in (
        (sun * radius, np.zeros(3), ORANGE),
        (np.zeros(3), outgoing * radius, BLUE),
    ):
        a, b = camera.project_xy(np.array([start, end]))
        ax.annotate(
            "",
            b,
            a,
            arrowprops={"arrowstyle": "-|>", "color": color, "lw": 2.0, "mutation_scale": 14},
        )
    sky_xy, outgoing_xy = camera.project_xy(np.array([sky * radius, outgoing * radius]))
    ax.plot(
        [sky_xy[0], outgoing_xy[0]],
        [sky_xy[1], outgoing_xy[1]],
        color=BLUE,
        linewidth=1.0,
        linestyle=":",
        alpha=0.55,
    )
    for direction, label, offset, color in (
        (sun, "Sun", (-4, 12), ORANGE),
        (sky, "sky point", (-2, -17), BLUE),
    ):
        point = camera.project_xy(direction * radius)[0]
        ax.scatter(*point, color=color, s=38, zorder=5)
        ax.annotate(
            label,
            point,
            xytext=offset,
            textcoords="offset points",
            fontsize=10,
            color=color,
            ha="center",
            zorder=6,
        )
    point = camera.project_xy(outgoing * 0.85)[0]
    ax.annotate(r"$d$", point, xytext=(0, 7), textcoords="offset points", color=BLUE)
    ax.scatter(0, 0, color=INK, s=17, zorder=5)
    ax.set_xlim(-1.7, 1.7)
    ax.set_ylim(-1.55, 1.55)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Fixed world directions", fontsize=11, pad=12, color=INK)


def _pose_panel(ax, data: IllustrationData, number: int) -> None:
    import numpy as np
    from _crystal_view import Camera, draw_crystal, draw_ray
    from _style import INK

    arrays = data.arrays
    index = number - 1
    camera = Camera(**CAMERA)
    geometry = data.metadata["geometry"]
    draw_crystal(
        ax,
        vertices=arrays["vertices_world"][index],
        face_numbers=arrays["face_numbers"],
        face_vertex_ids=geometry["face_vertex_ids"],
        edges=geometry["edges"],
        face_normals=arrays["face_normals_world"][index],
        face_centers=arrays["face_centers_world"][index],
        camera=camera,
        highlight_face=3,
        label_highlight=False,
    )
    draw_ray(ax, arrays["ray_points"][index], camera=camera)
    for point_index, face in zip((1, 2), (3, 5)):
        projected = camera.project_xy(arrays["ray_points"][index, point_index])[0]
        ax.annotate(
            str(face),
            projected,
            xytext=(4, 7),
            textcoords="offset points",
            fontsize=10,
            color=INK,
            zorder=15,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 0.4},
        )
    ax.set_aspect("equal")
    ax.set_xlim(-1.95, 1.95)
    ax.set_ylim(-1.55, 1.55)
    ax.axis("off")


def _assemble_figure(data: IllustrationData, figure):
    from _style import BLUE, GRAY, INK, MARK_COLORS

    arrays = data.arrays
    selected = arrays["selected_indices"]
    figure.suptitle(
        "3–5: fixed incident and outgoing directions, varying crystal poses",
        x=0.04,
        y=0.982,
        ha="left",
        fontsize=19,
        weight="bold",
        color=INK,
    )

    sphere = figure.add_axes([0.025, 0.415, 0.48, 0.53], projection="3d")
    _sphere_grid(sphere)
    sphere.plot(*arrays["u_curve"].T, color=BLUE, linewidth=2.8)
    for number, index in enumerate(selected, start=1):
        point = arrays["u_curve"][index]
        sphere.scatter(*point, color=MARK_COLORS[number - 1], s=60, depthshade=False)
        sphere.text(
            *(point * 1.13 + [0.04, 0.0, 0.025]),
            str(number),
            color=MARK_COLORS[number - 1],
            weight="bold",
            fontsize=12,
        )
    sphere.set_xlim(-1.05, 1.05)
    sphere.set_ylim(-1.05, 1.05)
    sphere.set_zlim(-1.05, 1.05)
    sphere.set_box_aspect((1, 1, 1), zoom=1.42)
    sphere.view_init(elev=24, azim=-57)
    sphere.set_axis_off()
    figure.text(0.09, 0.919, r"Crystal frame: $u=R^{\mathsf{T}}\hat{s}$", fontsize=13, weight="bold")
    _coordinate_triad(figure)

    pose = figure.add_axes([0.51, 0.415, 0.445, 0.53], projection="3d")
    pose.plot(*arrays["relative_rotvec"].T, color=BLUE, linewidth=2.8)
    for number, index in enumerate(selected, start=1):
        point = arrays["relative_rotvec"][index]
        pose.scatter(*point, color=MARK_COLORS[number - 1], s=60, depthshade=False)
        pose.text(*point, f"  {number}", color=MARK_COLORS[number - 1], weight="bold", fontsize=12)
    _decorate_pose_axes(pose)
    pose.set_box_aspect((1, 1, 1), zoom=1.05)
    figure.text(0.585, 0.919, r"Pose fiber: $F_{3\! -\! 5}(R)=d$", fontsize=13, weight="bold")

    _direction_sphere(figure.add_axes([0.025, 0.068, 0.205, 0.29]), data)
    for number in range(1, 5):
        axis = figure.add_axes([0.238 + (number - 1) * 0.187, 0.068, 0.177, 0.29])
        _pose_panel(axis, data, number)
        axis.set_title(str(number), color=MARK_COLORS[number - 1], fontsize=14, pad=12)
    figure.text(0.30, 0.028, "Incident", color="#c96635", fontsize=11)
    figure.text(0.425, 0.028, "Internal", color=INK, fontsize=11)
    figure.text(0.55, 0.028, "Outgoing", color=BLUE, fontsize=11)
    figure.text(
        0.04,
        0.006,
        "One regular closed component; completeness unknown. Apparent sky direction is −d.",
        color=GRAY,
        fontsize=9,
    )
    return figure


def render_figure(data: IllustrationData):
    import matplotlib.pyplot as plt
    from _style import apply_style

    apply_style()
    figure = plt.figure(figsize=(16, 10), dpi=150)
    try:
        return _assemble_figure(data, figure)
    except Exception:
        plt.close(figure)
        raise


def replot_data(data_root: Path | str, output_directory: Path | str) -> Path:
    import matplotlib.pyplot as plt

    data = load_data(data_root)
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    output = output_directory / OUTPUT_NAME
    figure = render_figure(data)
    try:
        figure.savefig(output, dpi=150)
    finally:
        plt.close(figure)
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument("--recompute", action="store_true")
    operation.add_argument("--replot", action="store_true")
    parser.add_argument("--canonical-data", type=Path, help="existing shared canonical-data directory; recompute only")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.recompute:
        if args.output_dir is not None:
            parser.error("--recompute does not accept --output-dir")
        result = recompute_data(args.data_dir, args.canonical_data)
        print(result.directory)
        print(
            "ray residuals: "
            f"direction={result.metadata['validation']['max_direction_error_l2']:.3g}, "
            f"Snell={result.metadata['validation']['max_snell_tangential_error_l2']:.3g}"
        )
        return 0
    if args.canonical_data is not None:
        parser.error("--canonical-data is valid only with --recompute")
    if args.output_dir is None:
        parser.error("--replot requires --output-dir")
    try:
        output = replot_data(args.data_dir, args.output_dir)
    except IllustrationDataError as error:
        parser.error(str(error))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
