"""Recompute or replot the canonical deviation-level-set illustration."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from _deviation_data import IllustrationData, IllustrationDataError, OUTPUT_NAME, load_data, recompute_data


LEVELS_DEG = (22.0, 26.0, 30.0, 35.0, 40.0, 45.0)
FIELD_RANGE_DEG = (21.8, 50.1)


def _label_curve(ax, points, indices) -> None:
    from matplotlib import patheffects
    from _style import MARK_COLORS

    for number, index in enumerate(indices, start=1):
        point = points[index]
        ax.scatter(
            *point,
            color=MARK_COLORS[number - 1],
            edgecolors="white",
            linewidths=0.7,
            s=60,
            zorder=6,
        )
        annotation = ax.annotate(
            str(number),
            point,
            xytext=(9, 4),
            textcoords="offset points",
            color=MARK_COLORS[number - 1],
            weight="bold",
            fontsize=12,
            zorder=7,
        )
        annotation.set_path_effects(
            [patheffects.withStroke(linewidth=2.5, foreground="white")]
        )


def _assemble_figure(data: IllustrationData, figure):
    import matplotlib.pyplot as plt
    from matplotlib import colors, patheffects
    import numpy as np

    from _style import BLUE, GRAY, INK, MARK_COLORS

    arrays = data.arrays
    norm = colors.Normalize(vmin=FIELD_RANGE_DEG[0], vmax=FIELD_RANGE_DEG[1])
    cmap = plt.get_cmap("viridis")
    figure.suptitle(
        "3–5: a fiber projects to a constant-deviation contour",
        x=0.04,
        y=0.97,
        ha="left",
        fontsize=20,
        weight="bold",
        color=INK,
    )
    figure.text(
        0.07,
        0.884,
        r"Deviation field $D_P(u)$ on the body-frame sphere",
        fontsize=14,
        color=INK,
    )
    figure.text(0.59, 0.884, "The same field, with level contours", fontsize=14, color=INK)

    sphere = figure.add_axes([0.0, 0.205, 0.51, 0.65], projection="3d", computed_zorder=False)
    points = arrays["sphere_points"]
    facecolors = cmap(norm(arrays["sphere_d_deg"]))
    facecolors[~arrays["sphere_valid"]] = colors.to_rgba("#eef1f4")
    sphere.plot_surface(
        points[..., 0],
        points[..., 1],
        points[..., 2],
        facecolors=facecolors,
        rstride=1,
        cstride=1,
        shade=False,
        linewidth=0,
        antialiased=False,
        rasterized=True,
        zorder=1,
    )
    center = arrays["chart_center"]
    azimuth = np.degrees(np.arctan2(center[1], center[0]))
    elevation = 20.0
    view_direction = np.array(
        [
            np.cos(np.deg2rad(elevation)) * np.cos(np.deg2rad(azimuth)),
            np.cos(np.deg2rad(elevation)) * np.sin(np.deg2rad(azimuth)),
            np.sin(np.deg2rad(elevation)),
        ]
    )
    angle = np.linspace(0.0, 2.0 * np.pi, 240)
    mesh_lines = []
    for latitude in np.deg2rad([-60, -30, 0, 30, 60]):
        mesh_lines.append(
            np.stack(
                [
                    np.cos(latitude) * np.cos(angle),
                    np.cos(latitude) * np.sin(angle),
                    np.full_like(angle, np.sin(latitude)),
                ],
                axis=-1,
            )
        )
    for longitude in np.deg2rad(np.arange(0, 180, 30)):
        mesh_lines.append(
            np.stack(
                [np.cos(angle) * np.cos(longitude), np.cos(angle) * np.sin(longitude), np.sin(angle)],
                axis=-1,
            )
        )
    for line in mesh_lines:
        visible = line.copy() * 1.003
        visible[line @ view_direction <= 0.02] = np.nan
        sphere.plot(*visible.T, color="#bbc9d4", alpha=0.35, linewidth=0.6, zorder=2)

    curve = arrays["u_curve"] * 1.015
    sphere.plot(*curve.T, color="white", linewidth=5.2, zorder=3)
    sphere.plot(*curve.T, color=BLUE, linewidth=2.5, zorder=4)
    for number, index in enumerate(arrays["selected_indices"], start=1):
        point = curve[index]
        sphere.scatter(
            *point,
            color=MARK_COLORS[number - 1],
            edgecolors="white",
            linewidths=0.7,
            s=65,
            depthshade=False,
            zorder=5,
        )
        label = sphere.text(
            *(point * 1.08),
            f"  {number}",
            color=MARK_COLORS[number - 1],
            fontsize=12,
            weight="bold",
            zorder=6,
        )
        label.set_path_effects([patheffects.withStroke(linewidth=2.5, foreground="white")])
    sphere.set_xlim(-1.05, 1.05)
    sphere.set_ylim(-1.05, 1.05)
    sphere.set_zlim(-1.05, 1.05)
    sphere.view_init(elev=elevation, azim=azimuth)
    sphere.set_box_aspect((1, 1, 1), zoom=1.25)
    sphere.set_axis_off()

    chart = figure.add_axes([0.57, 0.24, 0.37, 0.60])
    chart.set_facecolor("#eef1f4")
    masked = np.ma.masked_where(~arrays["chart_valid"], arrays["chart_d_deg"])
    chart.pcolormesh(
        arrays["chart_x"],
        arrays["chart_y"],
        masked,
        cmap=cmap,
        norm=norm,
        shading="auto",
        rasterized=True,
    )
    contours = chart.contour(
        arrays["chart_x"],
        arrays["chart_y"],
        masked,
        levels=LEVELS_DEG,
        colors="white",
        linewidths=0.8,
        alpha=0.65,
    )
    chart.clabel(contours, fmt=lambda value: f"{value:g}°", fontsize=10)
    chart.contour(
        arrays["chart_x"],
        arrays["chart_y"],
        arrays["chart_valid"].astype(float),
        levels=[0.5],
        colors=GRAY,
        linewidths=1.0,
    )
    line, = chart.plot(*arrays["curve_xy"].T, color=BLUE, linewidth=2.4)
    line.set_path_effects(
        [patheffects.Stroke(linewidth=4.5, foreground="white"), patheffects.Normal()]
    )
    _label_curve(chart, arrays["curve_xy"], arrays["selected_indices"])
    chart.set_xlim(-0.70, 0.70)
    chart.set_ylim(-0.78, 0.78)
    chart.set_aspect("equal")
    chart.set_xlabel(r"$u\cdot e_1$")
    chart.set_ylabel(r"$u\cdot e_2$")
    chart.tick_params(labelsize=10)

    connector = figure.add_axes([0.475, 0.46, 0.075, 0.11])
    connector.annotate(
        "",
        (0.98, 0.45),
        (0.05, 0.45),
        arrowprops={"arrowstyle": "->", "lw": 1.5, "color": GRAY},
    )
    connector.text(0.5, 0.8, "local view", ha="center", color=GRAY, fontsize=10)
    connector.set_xlim(0, 1)
    connector.set_ylim(0, 1)
    connector.axis("off")

    colorbar_axis = figure.add_axes([0.105, 0.155, 0.30, 0.020])
    figure.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=colorbar_axis,
        orientation="horizontal",
        label=r"Deviation $D_P$ [degrees]",
    )
    figure.text(
        0.60,
        0.151,
        r"Blue curve: fiber projection, $D_P(u)=\delta$",
        color=BLUE,
        fontsize=13,
    )
    figure.text(
        0.60,
        0.107,
        f"δ = {np.degrees(float(arrays['delta_rad'])):.3f}°",
        color=BLUE,
        fontsize=12,
    )
    figure.text(
        0.04,
        0.038,
        "Grey: outside the path's optical domain.  Pose labels 1–4 match the fiber overview.",
        color=GRAY,
        fontsize=10,
    )
    return figure


def render_figure(data: IllustrationData):
    import matplotlib.pyplot as plt
    from _style import apply_style

    apply_style()
    figure = plt.figure(figsize=(14.5, 8), dpi=150)
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
            "projected level residual: "
            f"{result.metadata['validation']['fiber_projected_level_error_rad']:.3g} rad"
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
