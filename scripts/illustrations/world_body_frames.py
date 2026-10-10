"""Recompute or replot the world/body coordinate-frame illustration."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from _world_body_data import (
    OUTPUT_NAME,
    IllustrationData,
    IllustrationDataError,
    load_data,
    recompute_data,
)


DEFAULT_CAMERA = {
    "azimuth_deg": 65.0,
    "elevation_deg": 24.0,
    "distance": 7.0,
    "projection": "perspective",
}
LANDMARK_FACE_NUMBER = 3


def _frame_triad(ax, camera, labels: Sequence[str]) -> None:
    from _style import GRAY
    from _crystal_view import Camera
    import numpy as np

    origin = np.array([-2.10, -1.75])
    key_camera = Camera(
        azimuth_deg=camera.azimuth_deg,
        elevation_deg=camera.elevation_deg,
        projection="orthographic",
    )
    projected = key_camera.project_xy(np.eye(3))
    for name, direction in zip(labels, projected):
        endpoint = origin + 0.44 * direction
        ax.annotate(
            "",
            endpoint,
            origin,
            arrowprops={"arrowstyle": "->", "color": GRAY, "lw": 1.0},
            zorder=12,
        )
        ax.text(
            *(origin + 0.60 * direction),
            name,
            color=GRAY,
            fontsize=12,
            ha="center",
            va="center",
            zorder=13,
        )


def _configuration(
    ax,
    *,
    data: IllustrationData,
    vertices,
    normals,
    centers,
    source_direction,
    number: int,
    frame: str,
    camera,
) -> None:
    from _crystal_view import draw_crystal
    from _style import ORANGE
    import numpy as np

    geometry = data.metadata["geometry"]
    draw_crystal(
        ax,
        vertices=vertices,
        face_numbers=data.arrays["face_numbers"],
        face_vertex_ids=geometry["face_vertex_ids"],
        edges=geometry["edges"],
        face_normals=normals,
        face_centers=centers,
        camera=camera,
        highlight_face=LANDMARK_FACE_NUMBER,
    )
    source = np.asarray(source_direction) * 2.3
    center_xy, source_xy = camera.project_xy(np.array([np.zeros(3), source]))
    ax.annotate(
        "",
        source_xy,
        center_xy,
        arrowprops={
            "arrowstyle": "-|>",
            "color": ORANGE,
            "lw": 2.0,
            "mutation_scale": 14,
        },
        zorder=12,
    )
    ax.scatter(*source_xy, color=ORANGE, s=70, marker="o", zorder=15)
    ax.annotate(
        "Sun" if frame == "world" else "source",
        source_xy,
        xytext=(0, 13),
        textcoords="offset points",
        ha="center",
        color=ORANGE,
        fontsize=12,
        zorder=16,
    )
    anchor = center_xy * 0.35 + source_xy * 0.65
    symbol = r"$\hat{\mathbf{s}}$" if frame == "world" else rf"$\mathbf{{u}}_{number}$"
    ax.annotate(
        symbol,
        anchor,
        xytext=(8, -13),
        textcoords="offset points",
        color=ORANGE,
        fontsize=14,
        zorder=16,
    )
    _frame_triad(ax, camera, ("X", "Y", "Z") if frame == "world" else ("x", "y", "z"))
    ax.set_aspect("equal")
    ax.set_xlim(-2.8, 2.8)
    ax.set_ylim(-2.4, 2.4)
    ax.axis("off")


def _assemble_figure(data: IllustrationData, figure):
    from matplotlib.patches import FancyArrowPatch

    from _crystal_view import Camera
    from _style import GRAY, INK, MARK_COLORS

    camera = Camera(**DEFAULT_CAMERA)
    arrays = data.arrays
    figure.suptitle(
        "The same configuration, seen in two coordinate frames",
        x=0.04,
        y=0.975,
        ha="left",
        fontsize=22,
        weight="bold",
        color=INK,
    )
    figure.text(0.04, 0.897, "World frame", fontsize=16, weight="bold", color=INK)
    figure.text(0.18, 0.897, "fixed Sun · changing crystal pose", fontsize=14, color=GRAY)
    figure.text(0.04, 0.452, "Crystal frame", fontsize=16, weight="bold", color=INK)
    figure.text(0.18, 0.452, "fixed crystal · changing source direction", fontsize=14, color=GRAY)

    for index in range(4):
        left = 0.03 + 0.242 * index
        upper = figure.add_axes([left, 0.58, 0.23, 0.28])
        lower = figure.add_axes([left, 0.14, 0.23, 0.28])
        _configuration(
            upper,
            data=data,
            vertices=arrays["vertices_world"][index],
            normals=arrays["face_normals_world"][index],
            centers=arrays["face_centers_world"][index],
            source_direction=arrays["source_world"],
            number=index + 1,
            frame="world",
            camera=camera,
        )
        _configuration(
            lower,
            data=data,
            vertices=arrays["vertices_body"],
            normals=arrays["face_normals_body"],
            centers=arrays["face_centers_body"],
            source_direction=arrays["source_body"][index],
            number=index + 1,
            frame="body",
            camera=camera,
        )
        center = left + 0.115
        for y in (0.859, 0.414):
            figure.text(
                center,
                y,
                str(index + 1),
                fontsize=15,
                weight="bold",
                ha="center",
                color=MARK_COLORS[index],
            )
        figure.add_artist(
            FancyArrowPatch(
                (center, 0.555),
                (center, 0.480),
                transform=figure.transFigure,
                arrowstyle="->",
                color=GRAY,
                linewidth=1.4,
                mutation_scale=12,
            )
        )
        figure.text(
            center + 0.015,
            0.515,
            rf"$R_{index + 1}^{{\mathsf{{T}}}}$",
            fontsize=15,
            va="center",
            color=INK,
        )

    figure.text(
        0.5,
        0.074,
        r"$\mathbf{u}_i=R_i^{\mathsf{T}}\hat{\mathbf{s}}"
        r"\qquad\longleftrightarrow\qquad"
        r"\hat{\mathbf{s}}=R_i\mathbf{u}_i$",
        fontsize=19,
        ha="center",
        color=INK,
    )
    figure.text(
        0.5,
        0.028,
        "Arrows point toward the light source. Each column describes the same physical configuration.",
        fontsize=12,
        color=GRAY,
        ha="center",
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
    operation.add_argument("--recompute", action="store_true", help="rebuild data only; do not write a PNG")
    operation.add_argument("--replot", action="store_true", help="read existing data and write only the PNG")
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="batch data root containing the world-body-frames leaf directory",
    )
    parser.add_argument("--output-dir", type=Path, help="PNG output directory; required only with --replot")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.recompute:
        if args.output_dir is not None:
            parser.error("--recompute does not accept --output-dir")
        result = recompute_data(args.data_dir)
        validation = result.metadata["validation"]
        print(result.directory)
        print(
            "dense residuals: "
            f"retraction={validation['dense_retraction_residual_max_l2']:.3g}, "
            f"outgoing={validation['dense_outgoing_direction_max_l2']:.3g}"
        )
        return 0
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
